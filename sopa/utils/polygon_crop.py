from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.widgets import PolygonSelector
from scipy import ndimage
from shapely.geometry import Polygon
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.segmentation import expand_labels
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import get_transformation

from .._constants import ROI
from .._sdata import get_spatial_image
from .image import resize

log = logging.getLogger(__name__)

HELPER = """Enclose cells within a polygon. Helper:
    - Click on the plot to add a polygon vertex
    - Press the 'esc' key to start a new polygon
    - Try holding the 'ctrl' key to move a single vertex
    - Once the polygon is finished and overlaid in red, you can close the window
"""

VALID_N_CHANNELS = [1, 3]


def _prepare(sdata: SpatialData, channels: list[str], scale_factor: float):
    image_key, spatial_image = get_spatial_image(sdata, return_key=True)
    image = spatial_image.transpose("y", "x", "c")

    if channels is not None and len(channels):
        assert (
            len(channels) in VALID_N_CHANNELS
        ), f"Number of channels provided must be in: {', '.join(VALID_N_CHANNELS)}"
        image = image.sel(c=channels)
    else:
        assert (
            len(image.coords["c"]) in VALID_N_CHANNELS
        ), f"Choose one or three channels among {image.c.values} by using the --channels argument"

    log.info(f"Resizing image by a factor of {scale_factor}")
    return image_key, resize(image, scale_factor).compute()


def _bbox_multiregion(
    img: np.array,
    scale_factor: float,
    sigma: float,
    disk_size: float,
    threshold_size: float,
    expand: int,
) -> Polygon:

    # Take the max along the channel axis
    img = img.max(axis=-1)

    # Apply gaussian smoothing and thresholding
    img = gaussian(img, sigma=sigma)
    thr = threshold_otsu(img)
    img = (img > thr).astype(int)

    # Fill holes in binary image and apply median filter to remove outliers
    img = ndimage.binary_fill_holes(img).astype("uint8")
    img = median(img, disk(disk_size))

    # Label connected components (regions) and expland the regions
    img = measure.label(img)
    img = expand_labels(img, distance=expand)

    # Extract region properties
    regions = measure.regionprops(img)

    # Filter regions by size if specified
    if threshold_size is not None:
        regions = [region for region in regions if region.area > threshold_size]

    # Calculate the combined bounding box
    if regions:
        if len(regions) > 1:
            log.info(f"Found {len(regions)} regions. Combining their bounding boxes.")
        else:
            log.info("Found 1 region.")
        x_min = min(region.bbox[1] for region in regions) * scale_factor
        y_min = min(region.bbox[0] for region in regions) * scale_factor
        x_max = max(region.bbox[3] for region in regions) * scale_factor
        y_max = max(region.bbox[2] for region in regions) * scale_factor

    else:
        log.warning("No region found. Using the whole image as the bounding box.")
        x_min, y_min = 0, 0
        x_max = img.shape[1] * scale_factor
        y_max = img.shape[0] * scale_factor

    return Polygon(
        [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_max),
            (x_max, y_min),
        ]
    )


class _Selector:
    def __init__(self, ax):
        self.poly = PolygonSelector(ax, self.onselect, draw_bounding_box=True)
        log.info(HELPER)
        plt.show()

    def onselect(self, vertices):
        self.vertices = np.array(vertices)
        log.info(f"Selected polygon with {len(self.vertices)} vertices.")

    def disconnect(self):
        self.poly.disconnect_events()


def _draw_polygon(image: np.ndarray, scale_factor: float, margin_ratio: float):
    _, ax = plt.subplots()
    ax.imshow(image)

    dy, dx, *_ = image.shape
    plt.xlim(-margin_ratio * dx, dx + margin_ratio * dx)
    plt.ylim(dy + margin_ratio * dy, -margin_ratio * dy)

    selector = _Selector(ax)

    return Polygon(selector.vertices * scale_factor)


def intermediate_selection(intermediate_image: str, intermediate_polygon: str, margin_ratio: float = 0.1):
    log.info(f"Reading intermediate image {intermediate_image}")

    z = zarr.open(intermediate_image, mode="r")
    image = z[ROI.IMAGE_ARRAY_KEY][:]

    polygon = _draw_polygon(image, z.attrs[ROI.SCALE_FACTOR], margin_ratio)

    with zarr.ZipStore(intermediate_polygon, mode="w") as store:
        g = zarr.group(store=store)
        g.attrs.put({ROI.IMAGE_KEY: z.attrs[ROI.IMAGE_KEY], ROI.ELEMENT_TYPE: "images"})

        coords = np.array(polygon.exterior.coords)
        g.array(ROI.POLYGON_ARRAY_KEY, coords, dtype=coords.dtype, chunks=coords.shape)


def polygon_selection(
    sdata: SpatialData,
    intermediate_image: str | None = None,
    intermediate_polygon: str | None = None,
    channels: list[str] | None = None,
    scale_factor: float = 10,
    margin_ratio: float = 0.1,
):
    """Crop an image based on a user-defined polygon (interactive mode).

    Args:
        sdata: A `SpatialData` object
        intermediate_image: Path to the intermediate image, with a `.zip` extension. Use this only if the interactive mode is not available
        intermediate_polygon: Path to the intermediate polygon, with a `.zip` extension. Use this locally, after downloading the `intermediate_image`
        channels: List of channel names to be displayed. Optional if there are already only 1 or 3 channels.
        scale_factor: Resize the image by this value (high value for a lower memory usage)
        margin_ratio: Ratio of the image margin on the display (compared to the image size)
    """
    if intermediate_polygon is None:
        image_key, image = _prepare(sdata, channels, scale_factor)

        if intermediate_image is not None:
            log.info(f"Resized image will be saved to {intermediate_image}")
            with zarr.ZipStore(intermediate_image, mode="w") as store:
                g = zarr.group(store=store)
                g.attrs.put({ROI.SCALE_FACTOR: scale_factor, ROI.IMAGE_KEY: image_key})
                g.array(ROI.IMAGE_ARRAY_KEY, image, dtype=image.dtype, chunks=image.shape)
            return

        polygon = _draw_polygon(image, scale_factor, margin_ratio)
    else:
        log.info(f"Reading polygon at path {intermediate_polygon}")
        z = zarr.open(intermediate_polygon, mode="r")
        polygon = Polygon(z[ROI.POLYGON_ARRAY_KEY][:])
        image_key = z.attrs[ROI.IMAGE_KEY]

        image = get_spatial_image(sdata, image_key)

    geo_df = gpd.GeoDataFrame(geometry=[polygon])

    geo_df = ShapesModel.parse(geo_df, transformations=get_transformation(sdata[image_key], get_all=True).copy())
    sdata.shapes[ROI.KEY] = geo_df
    if sdata.is_backed():
        sdata.write_element(ROI.KEY, overwrite=True)

    log.info(f"Polygon saved in sdata['{ROI.KEY}']")


def automatic_polygon_selection(
    sdata: SpatialData,
    scale_factor: float = 10,
    channels: list[str] | None = None,
    sigma: float = 240,
    expand: int = 480,
    disk_size: int = 240,
    threshold_size: float = 6400,
):
    """Automatically identify a rectangular region of interest.

    Works well for images which have a clear background and foreground, e.g. organoids.

    Args:
        sdata: A `SpatialData` object
        scale_factor: Resize the image by this value (high value for a lower memory usage)
        channels: List of channel names to be used. Optional if there are already only 1 or 3 channels.
        sigma: Standard deviation for gaussian smoothing
        expand: Distance to expand the regions
        disk_size: Radius of the disk used in median filtering
        threshold_size: Minimum size of regions

    """

    image_key, image = _prepare(sdata, channels=channels, scale_factor=scale_factor)

    # compute the bounding box for all regions, make sure to adjust the parameters for the `scale_factor`
    polygon = _bbox_multiregion(
        image,
        scale_factor=scale_factor,
        sigma=sigma / scale_factor,
        disk_size=int(disk_size / scale_factor),
        threshold_size=threshold_size / (scale_factor**2),
        expand=int(expand / scale_factor),
    )

    geo_df = gpd.GeoDataFrame(geometry=[polygon])

    geo_df = ShapesModel.parse(geo_df, transformations=get_transformation(sdata[image_key], get_all=True).copy())
    sdata.shapes[ROI.KEY] = geo_df

    if sdata.is_backed():
        sdata.write_element(ROI.KEY, overwrite=True)

    log.info(f"Polygon saved in sdata['{ROI.KEY}']")
