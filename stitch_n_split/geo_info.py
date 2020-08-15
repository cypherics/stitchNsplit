import numpy as np
import math

import rasterio
from affine import Affine
from rasterio.transform import rowcol

from rasterio.warp import transform_bounds


def get_window(extent: tuple, transform: Affine) -> (tuple, tuple):
    row_start, col_start = rowcol(transform, extent[0], extent[-1], op=int)

    row_stop, col_stop = rowcol(transform, extent[2], extent[1], op=int)

    return (row_start, row_stop), (col_start, col_stop)


def get_mesh_transform(width: int, height: int, transform: Affine) -> Affine:
    bounds = compute_bounds(width, height, transform)
    mesh_transform = get_affine_transform(
        bounds[0], bounds[-1], *get_pixel_resolution(transform)
    )
    return mesh_transform


def get_affine_transform(
    min_x: float, max_y: float, pixel_width: float, pixel_height: float
) -> Affine:
    """

    :param min_x:
    :param max_y:
    :param pixel_width: width of pixels in the units of its coordinate reference system
    :param pixel_height: height of pixels in the units of its coordinate reference system
    :return:
    """
    return Affine.translation(min_x, max_y) * Affine.scale(pixel_width, -pixel_height)


def compute_bounds(width, height, transform):
    """
    Computes the bounds of w x h given the transform
    :param width:
    :param height:
    :param transform:
    :return: bounds for w x h , format bounds returned in (w, s, e, n)
    """
    bounds = rasterio.transform.array_bounds(height, width, transform)
    return bounds


def geo_transform_to_26190(width, height, bounds, crs) -> Affine:
    west, south, east, north = transform_bounds(crs, {"init": "epsg:26910"}, *bounds)
    return rasterio.transform.from_bounds(west, south, east, north, width, height)


def re_project_crs_to_26190(bounds, from_crs) -> (float, float, float, float):
    west, south, east, north = transform_bounds(
        from_crs, {"init": "epsg:26910"}, *bounds
    )
    return west, south, east, north


def re_project_from_26190(bounds, to_crs) -> (float, float, float, float):
    west, south, east, north = transform_bounds({"init": "epsg:26910"}, to_crs, *bounds)
    return west, south, east, north


def get_pixel_resolution(transform: Affine) -> (float, float):
    """
    Pixel Resolution
    :param transform:
    :return: width and height of pixels in the units of its coordinate reference system extracted from
    transformation of image
    """
    return transform[0], -transform[4]


def compute_num_of_col_and_ros(grid_size: tuple, mesh_size: tuple):
    """
    num_col grids will fit in x direction
    num_row grids will fit in Y direction

    Computes How many Number of grids to draw
    :return: number of grid in x direction, number of grid in y direction
    """
    num_col = int(np.ceil(mesh_size[0] / grid_size[0]))
    num_row = int(np.ceil(mesh_size[1] / grid_size[1]))

    return num_col, num_row


def compute_dimension(bounds, pixel_resolution: tuple):
    """

    :param bounds:
    :param pixel_resolution: width and height of pixels in the units of its coordinate reference system extracted from
    transformation of image
    :return:
    """
    output_width = int(math.ceil((bounds[2] - bounds[0]) / pixel_resolution[0]))
    output_height = int(math.ceil((bounds[3] - bounds[1]) / pixel_resolution[1]))
    return output_width, output_height
