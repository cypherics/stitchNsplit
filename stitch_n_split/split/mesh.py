import math
import rasterio
from dataclasses import dataclass

import numpy as np

from affine import Affine

from rasterio.warp import transform_bounds


class Mesh:
    def _compute_step(self):
        """
        Compute Step in X and Y direction
        :return:
        """

        raise NotImplementedError

    @staticmethod
    def _step_in_x(bound, normalizer=1):
        """
        Step Size to take in X
        :param bound:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((bound[2] - bound[0]) / normalizer))

    @staticmethod
    def _step_in_y(bound, normalizer=1):
        """
        Step Size to take in Y
        :param bound:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((bound[-1] - bound[1]) / normalizer))

    def extent(self):
        """
        Compute Mesh

        :return:
        """

        raise NotImplementedError


@dataclass
class ImageNonOverLapMesh(Mesh):
    """
    The Class will compute Grid bounded within complete_size to provide non overlapping grid,
    The class will adjust the grid to evenly fit the number of tiles

    Working of this class depends on the geo reference information of the image which acts as the starting point

    The geo reference information to be present in the image is source_min_x, source_max_y and pixel resolution

    Based on the geo reference information present in the image, compute grid of size
    complete_size // int(np.ceil(dst_img_size / src_img_size) over complete_size

    Given an starting image size, final size and its transform this will find all the grid of size
    complete_size // int(np.ceil(dst_img_size / src_img_size) between the given complete size

    The start position of grid and the step size of grid is computed from the transform info provided, usually
    present in geo referenced image

    NOTE - The COORDINATES MUST BE IN `EPSG:26910`
    """

    grid_size: tuple
    mesh_size: tuple
    sections: tuple
    transform: Affine
    mesh_bound: tuple

    def _compute_step(self):
        """
        Compute Step in X and Y direction
        :return:
        """

        step_in_x = self._step_in_x(self.mesh_bound, self.sections[0])
        step_in_y = self._step_in_y(self.mesh_bound, self.sections[1])

        return step_in_x, step_in_y

    def extent(self):
        """
        Compute non overlapping grid bounded within complete_size

        :return:
        """

        (step_in_x, step_in_y) = self._compute_step()

        for y in range(self.sections[1]):

            for x in range(self.sections[0]):
                tx_start = x * step_in_x + self.mesh_bound[0]

                ty_start = y * step_in_y + self.mesh_bound[1]
                tx_end = tx_start + step_in_x - 1
                ty_end = ty_start + step_in_y - 1

                yield tx_start, ty_start, tx_end, ty_end


@dataclass
class ImageOverLapMesh(Mesh):
    """
    The Class will compute Grid bounded within complete_size and if the provided grid size overlaps, the the class will
    tune accordingly to provide overlapping grid, The class wont hamper the grid size in any manner, it will find all
    the possible grid of size provided that could fit in complete_size

    Working of this class depends on the geo reference information of the image which acts as the starting point

    The geo reference information to be present in the image is source_min_x, source_max_y and pixel resolution

    Based on the geo reference information present in the image, compute grid of size grid_size over complete_size

    Given an starting image size, final size and its transform this will find all the grid of size image size
    between the given complete size


    The start position of grid and the step size of grid is computed from the transform info provided, usually
    present in geo referenced image

    NOTE - The COORDINATES MUST BE IN `EPSG:26910`

    """

    grid_size: tuple
    mesh_size: tuple
    sections: tuple
    transform: Affine
    mesh_bound: tuple
    overlap_mesh_bound: tuple
    buffer_mesh_bound: tuple

    def _is_overlap_in_col_direction(self):
        """
        Check if there is any overlap in X direction
        :return:
        """
        return True if self.mesh_size[0] % self.grid_size[0] else False

    def _is_overlap_in_row_direction(self):
        """
        Check if there is any overlap in Y direction
        :return:
        """
        return True if self.mesh_size[1] % self.grid_size[1] else False

    def _compute_buffer_step(self):
        """
        To Compute overlapping steps it is essential to compute the the max_x and min_y not based on the complete
        size but to extrapolate the grid size by number of tiles.

        i.e grid_size = grid_size * tiles

        :return:
        """

        buffered_step_in_x = self._step_in_x(self.buffer_mesh_bound, self.sections[0])
        buffered_step_in_y = self._step_in_y(self.buffer_mesh_bound, self.sections[1])

        return buffered_step_in_x, buffered_step_in_y

    def _compute_overlap_step(self):
        """
        The overlapping step is nothing but keeping the coordinates in the bound provided in the form of
        complete_img_size, the overlapping step is difference between complete_size  and grid_size

        :return:
        """
        overlap_step_in_x = None
        overlap_step_in_y = None

        if self._is_overlap_in_col_direction():
            overlap_step_in_x = self._step_in_x(self.overlap_mesh_bound)
        if self._is_overlap_in_row_direction():
            overlap_step_in_y = self._step_in_y(self.overlap_mesh_bound)
        return overlap_step_in_x, overlap_step_in_y

    def _compute_step(self):
        return self._compute_buffer_step(), self._compute_overlap_step()

    def extent(self):
        """
        Compute Overlapping Grid
        :return:
        """

        (
            (buffered_step_in_x, buffered_step_in_y),
            (overlap_step_in_x, overlap_step_in_y),
        ) = self._compute_step()

        for y in range(self.sections[1]):
            for x in range(self.sections[0]):
                if (x == self.sections[0] - 1) and self._is_overlap_in_col_direction():

                    tx_start = overlap_step_in_x + self.mesh_bound[0]
                else:
                    tx_start = x * buffered_step_in_x + self.mesh_bound[0]
                if y == (self.sections[1] - 1) and self._is_overlap_in_row_direction():
                    ty_start = overlap_step_in_y + self.mesh_bound[1]
                else:
                    ty_start = y * buffered_step_in_y + self.mesh_bound[1]
                tx_end = tx_start + buffered_step_in_x - 1
                ty_end = ty_start + buffered_step_in_y - 1

                yield tx_start, ty_start, tx_end, ty_end


class GeoInfo:
    @staticmethod
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

    @staticmethod
    def geo_transform_to_26190(width, height, bounds, crs) -> Affine:
        west, south, east, north = transform_bounds(
            crs, {"init": "epsg:26910"}, *bounds
        )
        return rasterio.transform.from_bounds(west, south, east, north, width, height)

    @staticmethod
    def re_project_crs_to_26190(bounds, from_crs) -> (float, float, float, float):
        west, south, east, north = transform_bounds(
            from_crs, {"init": "epsg:26910"}, *bounds
        )
        return west, south, east, north

    @staticmethod
    def re_project_from_26190(bounds, to_crs) -> (float, float, float, float):
        west, south, east, north = transform_bounds(
            {"init": "epsg:26910"}, to_crs, *bounds
        )
        return west, south, east, north

    @staticmethod
    def get_pixel_resolution(transform: Affine) -> (float, float):
        """
        Pixel Resolution
        :param transform:
        :return:
        """
        return transform[0], transform[4]

    @staticmethod
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

    @staticmethod
    def compute_dimension(bounds, res: tuple):
        output_width = int(math.ceil((bounds[2] - bounds[0]) / res[0]))
        output_height = int(math.ceil((bounds[3] - bounds[1]) / (-res[1])))
        return output_width, output_height


def mesh_from_geo_transform(
    grid_size=None,
    mesh_size=None,
    grid_geo_transform=None,
    grid_bounds=None,
    overlap=True,
):
    """

    :param overlap: it set to true will find overlapping grid if any
    :param grid_bounds: initial starting point for grid computation, (w, s, e, n), if grid size
    is provided, then this argument is optional, this argument is used for computing grid_size
    :param grid_size: size of grid in pixel dimension (w x h)
    :param mesh_size: mesh grid in (w x h)
    :param grid_geo_transform: transform of the initial grid
    :return:
    """

    if grid_geo_transform is None:
        raise ValueError("grid_transform can't be None")
    if mesh_size is None:
        raise ValueError("complete_size can't be None")
    res = GeoInfo.get_pixel_resolution(grid_geo_transform)

    if grid_size is None:
        if grid_bounds is None:
            raise ValueError("Bounds can't be None")
        grid_size = GeoInfo.compute_dimension(grid_bounds, res)
    if grid_size[0] > mesh_size[0] or grid_size[1] > mesh_size[1]:
        raise ValueError(
            "Size to Split Can't Be Greater than Image, Given {},"
            " Expected less than equal to {}".format(grid_size, mesh_size)
        )
    sections = GeoInfo.compute_num_of_col_and_ros(grid_size, mesh_size)

    if overlap:
        buffer_mesh_bound = GeoInfo.compute_bounds(
            grid_size[0] * sections[0],
            grid_size[1] * sections[1],
            transform=grid_geo_transform,
        )

        overlap_mesh_bound = GeoInfo.compute_bounds(
            mesh_size[0] - grid_size[0],
            mesh_size[1] - grid_size[1],
            transform=grid_geo_transform,
        )

        mesh_bound = GeoInfo.compute_bounds(
            mesh_size[0], mesh_size[1], transform=grid_geo_transform
        )

        grid_data = ImageOverLapMesh(
            grid_size,
            mesh_size,
            sections,
            grid_geo_transform,
            mesh_bound,
            overlap_mesh_bound,
            buffer_mesh_bound,
        )
    else:
        mesh_bound = GeoInfo.compute_bounds(
            mesh_size[0], mesh_size[1], transform=grid_geo_transform
        )

        grid_data = ImageNonOverLapMesh(
            grid_size, mesh_size, sections, grid_geo_transform, mesh_bound
        )
    return grid_data
