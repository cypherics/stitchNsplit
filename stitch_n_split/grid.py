from dataclasses import dataclass
from typing import Any

import numpy as np
from collections import OrderedDict

from affine import Affine

from stitch_n_split.windows import Window


@dataclass
class ImageNonOverLapGeoGrid:
    """
    The Class will compute Grid bounded within complete_size to provide non overlapping grid,
    The class will adjust the grid to evenly fit the number of tiles
    """

    min_x: Any
    max_y: Any
    pixel_res: tuple
    tiles: tuple
    complete_size: tuple

    def __len__(self):
        return len(self.grid_data)

    def __getitem__(self, index):
        grid_data = list(self.grid_data.items())
        return grid_data[index]

    def __post_init__(self):
        self.grid_data = self.compute_non_overlapping_grid()

    def _compute_step(self):
        """
        Compute Step in X and Y direction
        :return:
        """
        max_x = self._compute_max_x(self.complete_size[0])
        min_y = self._compute_min_y(self.complete_size[1])

        step_in_x = self._step_in_x(max_x, self.tiles[0])
        step_in_y = self._step_in_y(min_y, self.tiles[1])

        return step_in_x, step_in_y

    def _compute_max_x(self, scale: float):
        """
        Compute the the max_x bounded within complete_size
        :param scale: how far max_x should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.min_x + self.pixel_res[0] * scale

    def _compute_min_y(self, scale: float):
        """
        Compute the the min_y bounded within complete_size
        :param scale: how far min_y should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.max_y + self.pixel_res[1] * scale

    def _step_in_x(self, max_x, normalizer=1):
        """
        Step Size to take in X
        :param max_x:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((max_x - self.min_x) / normalizer))

    def _step_in_y(self, min_y, normalizer=1):
        """
        Step Size to take in Y
        :param min_y:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """
        return int(((self.max_y - min_y) / normalizer))

    def compute_non_overlapping_grid(self):
        """
        Compute non overlapping grid bounded within complete_size

        :return:
        """

        grid_data = OrderedDict()

        iterator = 0
        windows = Window.image_geo_windows(
            (
                self.complete_size[0] // self.tiles[0],
                self.complete_size[1] // self.tiles[1],
            ),
            self.complete_size,
        ).windows

        (step_in_x, step_in_y) = self._compute_step()

        for y in range(self.tiles[1]):
            for x in range(self.tiles[0]):

                tx_start = x * step_in_x + self.min_x

                ty_start = (
                    y * step_in_y
                    + self.max_y
                    + self.pixel_res[1] * self.complete_size[1]
                )
                tx_end = tx_start + step_in_x - 1
                ty_end = ty_start + step_in_y - 1

                grid_data[(tx_start, ty_start, tx_end, ty_end)] = {
                    "window": windows[iterator]
                }
                iterator += 1
        return grid_data


@dataclass
class ImageOverLapGeoGrid:
    """
    The Class will compute Grid bounded within complete_size and if the provided grid size overlaps the the class will
    tune accordingly to provide overlapping grid, The class wont hamper the grid size in any manner, it will find all
    the possible grid of size that could fit in complete_size
    """

    min_x: Any
    max_y: Any
    pixel_res: tuple
    tiles: tuple
    grid_size: tuple
    complete_size: tuple

    def __len__(self):
        return len(self.grid_data)

    def __getitem__(self, index):
        grid_data = list(self.grid_data.items())
        return grid_data[index]

    def __post_init__(self):
        self.grid_data = self.compute_overlapping_grid()

    def _is_overlap_in_col_direction(self):
        """
        Check if there is any overlap in X direction
        :return:
        """
        return True if self.complete_size[0] % self.grid_size[0] else False

    def _is_overlap_in_row_direction(self):
        """
        Check if there is any overlap in Y direction
        :return:
        """
        return True if self.complete_size[1] % self.grid_size[1] else False

    def _compute_buffer_step(self):
        """
        To Compute overlapping steps it is essential to compute the the max_x and min_y not based on the complete
        size but to extrapolate the grid size by number of tiles.

        i.e grid_size = grid_size * tiles

        :return:
        """
        buffer_max_x = self._compute_max_x(self.grid_size[0] * self.tiles[0])
        buffer_min_y = self._compute_min_y(self.grid_size[1] * self.tiles[1])

        buffered_step_in_x = self._step_in_x(buffer_max_x, self.tiles[0])
        buffered_step_in_y = self._step_in_y(buffer_min_y, self.tiles[1])

        return buffered_step_in_x, buffered_step_in_y

    def _compute_max_x(self, scale: float):
        """
        Compute the the max_x bounded within complete_size
        :param scale: how far max_x should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.min_x + self.pixel_res[0] * scale

    def _compute_min_y(self, scale: float):
        """
        Compute the the min_y bounded within complete_size
        :param scale: how far min_y should grow, when scale is 1, it will grow in amount propotional to complete_size
        :return:
        """
        return self.max_y + self.pixel_res[1] * scale

    def _step_in_x(self, max_x, normalizer=1):
        """
        Step Size to take in X
        :param max_x:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """

        return int(((max_x - self.min_x) / normalizer))

    def _step_in_y(self, min_y, normalizer=1):
        """
        Step Size to take in Y
        :param min_y:
        :param normalizer: How small the step to take, Larger the value smaller and smaller step it will take
        :return:
        """

        return int(((self.max_y - min_y) / normalizer))

    def _compute_overlap_step(self):
        """
        The overlapping step is nothing but keeping the coordinates in the bound provided in the form of
        complete_img_size, the overlapping step is difference between complete_size  and grid_size

        :return:
        """
        overlap_step_in_x = None
        overlap_step_in_y = None

        if self._is_overlap_in_col_direction():

            over_lap_max_x = self._compute_max_x(
                self.complete_size[0] - self.grid_size[0]
            )
            overlap_step_in_x = self._step_in_x(over_lap_max_x)
        if self._is_overlap_in_row_direction():
            overlap_min_y = self._compute_min_y(
                self.complete_size[1] - self.grid_size[1]
            )
            overlap_step_in_y = self._step_in_y(overlap_min_y)
        return overlap_step_in_x, overlap_step_in_y

    def compute_overlapping_grid(self):
        """
        Compute Overlapping Grid
        :return:
        """

        grid_data = OrderedDict()

        iterator = 0
        windows = Window.image_geo_windows(self.grid_size, self.complete_size).windows

        (buffered_step_in_x, buffered_step_in_y) = self._compute_buffer_step()
        (overlap_step_in_x, overlap_step_in_y) = self._compute_overlap_step()

        for y in range(self.tiles[1]):
            for x in range(self.tiles[0]):

                if (x == self.tiles[0] - 1) and self._is_overlap_in_col_direction():

                    tx_start = x * overlap_step_in_x + self.min_x
                else:
                    tx_start = x * buffered_step_in_x + self.min_x
                if y == (self.tiles[1] - 1) and self._is_overlap_in_row_direction():
                    ty_start = (
                        y * overlap_step_in_y
                        + self.max_y
                        + self.pixel_res[1] * self.complete_size[1]
                    )
                else:
                    ty_start = (
                        y * buffered_step_in_y
                        + self.max_y
                        + self.pixel_res[1] * self.complete_size[1]
                    )
                tx_end = tx_start + buffered_step_in_x - 1
                ty_end = ty_start + buffered_step_in_y - 1

                grid_data[(tx_start, ty_start, tx_end, ty_end)] = {
                    "window": windows[iterator]
                }
                iterator += 1
        return grid_data


class GeoGrid:
    def __init__(self, grid_data):
        self.grid_data = grid_data

    @staticmethod
    def _get_min_x_and_max_y(transform: Affine) -> (float, float):
        return transform[2], transform[5]

    @staticmethod
    def _get_pixel_resolution(transform: Affine) -> (float, float):
        """
        Pixel Resolution
        :param transform:
        :return:
        """
        return transform[0], transform[4]

    @staticmethod
    def _compute_num_of_tiles(src_img_size: tuple, dst_img_size: tuple):
        """
        Computes How many Number of grids to draw
        :return: number of grid in x direction, number of grid in y direction
        """
        col_tiles = int(np.ceil(dst_img_size[0] / src_img_size[0]))
        row_tiles = int(np.ceil(dst_img_size[1] / src_img_size[1]))

        return col_tiles, row_tiles

    @classmethod
    def overlapping_grid_from_transform(
        cls, grid_size: tuple, complete_size: tuple, grid_transform: Affine,
    ):
        """
        Given an starting image size, final size and its transform this will find all the grid of size image size
        between the given complete size

        :param grid_size: typical image size
        :param complete_size: size to which size of the image is to be extented, typically greater than gris size
        :param grid_transform: transform of the image which is used for grid size
        :return:
        """

        if grid_size[0] > complete_size[0] or grid_size[1] > complete_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected less than equal to {}".format(grid_size, complete_size)
            )
        source_min_x, source_max_y = cls._get_min_x_and_max_y(grid_transform)
        res = cls._get_pixel_resolution(grid_transform)
        tiles = cls._compute_num_of_tiles(grid_size, complete_size)
        grid_data = ImageOverLapGeoGrid(
            source_min_x, source_max_y, res, tiles, grid_size, complete_size
        )
        return cls(grid_data)

    @classmethod
    def non_overlapping_grid_from_transform(
        cls, grid_size: tuple, complete_size: tuple, grid_transform: Affine,
    ):
        """
        Given an starting image size, final size and its transform this will find all the grid of size image size
        between the given complete size

        :param grid_size: typical image size
        :param complete_size: size to which size of the image is to be extented, typically greater than gris size
        :param grid_transform: transform of the image which is used for grid size
        :return:
        """
        if grid_size[0] > complete_size[0] or grid_size[1] > complete_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected less than equal to {}".format(grid_size, complete_size)
            )
        source_min_x, source_max_y = cls._get_min_x_and_max_y(grid_transform)
        res = cls._get_pixel_resolution(grid_transform)
        tiles = cls._compute_num_of_tiles(grid_size, complete_size)
        grid_data = ImageNonOverLapGeoGrid(
            source_min_x, source_max_y, res, tiles, complete_size
        )
        return cls(grid_data)
