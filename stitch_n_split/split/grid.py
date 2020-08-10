import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from affine import Affine


class GeoGrid:
    def _compute_max_x(self, scale: float):
        """
        Compute the the max_x bounded within complete_size
        :param scale: how far max_x should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        raise NotImplementedError

    def _compute_min_y(self, scale: float):
        """
        Compute the the min_y bounded within complete_size
        :param scale: how far min_y should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        raise NotImplementedError

    def _compute_step(self):
        """
        Compute Step in X and Y direction
        :return:
        """

        raise NotImplementedError

    def _step_in_x(self, max_x, normalizer=1):
        """
        Step Size to take in X
        :param max_x:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        raise NotImplementedError

    def _step_in_y(self, min_y, normalizer=1):
        """
        Step Size to take in Y
        :param min_y:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        raise NotImplementedError

    def extent(self):
        """
        Compute Mesh

        :return:
        """

        raise NotImplementedError


@dataclass
class ImageNonOverLapGeoGrid(GeoGrid):
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

    src_min_x: Any
    src_max_y: Any
    pixel_res: tuple
    sections: tuple
    mesh_size: tuple

    def _compute_step(self):
        """
        Compute Step in X and Y direction
        :return:
        """
        max_x = self._compute_max_x(self.mesh_size[0])
        min_y = self._compute_min_y(self.mesh_size[1])

        step_in_x = self._step_in_x(max_x, self.sections[0])
        step_in_y = self._step_in_y(min_y, self.sections[1])

        return step_in_x, step_in_y

    def _compute_max_x(self, scale: float):
        """
        Compute the the max_x bounded within complete_size
        :param scale: how far max_x should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.src_min_x + self.pixel_res[0] * scale

    def _compute_min_y(self, scale: float):
        """
        Compute the the min_y bounded within complete_size
        :param scale: how far min_y should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.src_max_y + self.pixel_res[1] * scale

    def _step_in_x(self, max_x, normalizer=1):
        """
        Step Size to take in X
        :param max_x:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((max_x - self.src_min_x) / normalizer))

    def _step_in_y(self, min_y, normalizer=1):
        """
        Step Size to take in Y
        :param min_y:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((self.src_max_y - min_y) / normalizer))

    def extent(self):
        """
        Compute non overlapping grid bounded within complete_size

        :return:
        """

        (step_in_x, step_in_y) = self._compute_step()

        for y in range(self.sections[1]):
            for x in range(self.sections[0]):
                tx_start = x * step_in_x + self.src_min_x

                ty_start = (
                    y * step_in_y
                    + self.src_max_y
                    + self.pixel_res[1] * self.mesh_size[1]
                )
                tx_end = tx_start + step_in_x - 1
                ty_end = ty_start + step_in_y - 1

                yield tx_start, ty_start, tx_end, ty_end


@dataclass
class ImageOverLapGeoGrid(GeoGrid):
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

    src_min_x: Any
    src_max_y: Any
    pixel_res: tuple
    sections: tuple
    grid_size: tuple
    mesh_size: tuple

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
        buffer_max_x = self._compute_max_x(self.grid_size[0] * self.sections[0])
        buffer_min_y = self._compute_min_y(self.grid_size[1] * self.sections[1])

        buffered_step_in_x = self._step_in_x(buffer_max_x, self.sections[0])
        buffered_step_in_y = self._step_in_y(buffer_min_y, self.sections[1])

        return buffered_step_in_x, buffered_step_in_y

    def _compute_max_x(self, scale: float):
        """
        Compute the the max_x bounded within complete_size
        :param scale: how far max_x should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.src_min_x + self.pixel_res[0] * scale

    def _compute_min_y(self, scale: float):
        """
        Compute the the min_y bounded within complete_size
        :param scale: how far min_y should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.src_max_y + self.pixel_res[1] * scale

    def _step_in_x(self, max_x, normalizer=1):
        """
        Step Size to take in X
        :param max_x:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """

        return int(((max_x - self.src_min_x) / normalizer))

    def _step_in_y(self, min_y, normalizer=1):
        """
        Step Size to take in Y
        :param min_y:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """

        return int(((self.src_max_y - min_y) / normalizer))

    def _compute_overlap_step(self):
        """
        The overlapping step is nothing but keeping the coordinates in the bound provided in the form of
        complete_img_size, the overlapping step is difference between complete_size  and grid_size

        :return:
        """
        overlap_step_in_x = None
        overlap_step_in_y = None

        if self._is_overlap_in_col_direction():

            over_lap_max_x = self._compute_max_x(self.mesh_size[0] - self.grid_size[0])
            overlap_step_in_x = self._step_in_x(over_lap_max_x)
        if self._is_overlap_in_row_direction():
            overlap_min_y = self._compute_min_y(self.mesh_size[1] - self.grid_size[1])
            overlap_step_in_y = self._step_in_y(overlap_min_y)
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

                    tx_start = overlap_step_in_x + self.src_min_x
                else:
                    tx_start = x * buffered_step_in_x + self.src_min_x
                if y == (self.sections[1] - 1) and self._is_overlap_in_row_direction():
                    ty_start = (
                        overlap_step_in_y
                        + self.src_max_y
                        + self.pixel_res[1] * self.mesh_size[1]
                    )
                else:
                    ty_start = (
                        y * buffered_step_in_y
                        + self.src_max_y
                        + self.pixel_res[1] * self.mesh_size[1]
                    )
                tx_end = tx_start + buffered_step_in_x - 1
                ty_end = ty_start + buffered_step_in_y - 1

                yield tx_start, ty_start, tx_end, ty_end


class GeoInfo:
    @staticmethod
    def get_min_x_and_max_y(transform: Affine) -> (float, float):
        return transform[2], transform[5]

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
    grid_size=None, mesh_size=None, grid_geo_transform=None, bounds=None, overlap=True,
):
    """

    :param overlap:
    :param bounds:
    :param grid_size: typical image size
    :param mesh_size: size to which size of the image is to be extented, typically greater than gris size
    :param grid_geo_transform: transform of the image which is used for grid size
    :return:
    """

    if grid_geo_transform is None:
        raise ValueError("grid_transform can't be None")
    if mesh_size is None:
        raise ValueError("complete_size can't be None")
    source_min_x, source_max_y = GeoInfo.get_min_x_and_max_y(grid_geo_transform)
    res = GeoInfo.get_pixel_resolution(grid_geo_transform)

    if grid_size is None:
        if bounds is None:
            raise ValueError("Bounds can't be None")
        grid_size = GeoInfo.compute_dimension(bounds, res)
    if grid_size[0] > mesh_size[0] or grid_size[1] > mesh_size[1]:
        raise ValueError(
            "Size to Split Can't Be Greater than Image, Given {},"
            " Expected less than equal to {}".format(grid_size, mesh_size)
        )
    sections = GeoInfo.compute_num_of_col_and_ros(grid_size, mesh_size)

    if overlap:
        grid_data = ImageOverLapGeoGrid(
            source_min_x, source_max_y, res, sections, grid_size, mesh_size
        )
    else:
        grid_data = ImageNonOverLapGeoGrid(
            source_min_x, source_max_y, res, sections, mesh_size
        )
    return grid_data


def mesh_from_pixel_resolution(
    pixel_resolution=None,
    mesh_size=None,
    grid_geo_transform=None,
    bounds=None,
    overlap=True,
):
    """
    https://blogs.bing.com/maps/2006/02/25/map-control-zoom-levels-gt-resolution
    estimate pixel resolution based on zoom level

    :param overlap:
    :param bounds:
    :param pixel_resolution: pixel resolution
    :param mesh_size: size to which size of the image is to be extented, typically greater than gris size
    :param grid_geo_transform: transform of the image which is used for grid size
    :return:
    """

    if grid_geo_transform is None:
        raise ValueError("grid_transform can't be None")
    if mesh_size is None:
        raise ValueError("complete_size can't be None")
    if pixel_resolution is None:
        raise ValueError("Pixel Resolution Can't be None")
    if type(pixel_resolution) != tuple:
        raise TypeError("Pixel Resolution Must Be tuple")
    if len(pixel_resolution) != 2:
        raise ValueError("Length of Pixel Resolution Must be 2")
    if bounds is None:
        raise ValueError("Bounds can't be None")
    source_min_x, source_max_y = GeoInfo.get_min_x_and_max_y(grid_geo_transform)
    res = GeoInfo.get_pixel_resolution(grid_geo_transform)

    grid_size = GeoInfo.compute_dimension(bounds, pixel_resolution)
    if grid_size[0] > mesh_size[0] or grid_size[1] > mesh_size[1]:
        raise ValueError(
            "Size to Split Can't Be Greater than Image, Given {},"
            " Expected less than equal to {}".format(grid_size, mesh_size)
        )
    sections = GeoInfo.compute_num_of_col_and_ros(grid_size, mesh_size)

    if overlap:
        grid_data = ImageOverLapGeoGrid(
            source_min_x, source_max_y, res, sections, grid_size, mesh_size
        )
    else:
        grid_data = ImageNonOverLapGeoGrid(
            source_min_x, source_max_y, res, sections, mesh_size
        )
    return grid_data
