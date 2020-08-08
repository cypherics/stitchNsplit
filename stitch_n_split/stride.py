import numpy as np


class StrideOver:
    def __init__(self, stride_size: tuple, img_size: tuple):
        """
        W = Columns
        H = Rows

        Input = W x H

        :param stride_size: tuple(W x H), Size to split the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split operation is to be performed
        """
        if stride_size[0] > img_size[0] or stride_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(stride_size, (img_size[0], img_size[1]))
            )
        assert len(img_size) == 3, "Pass Image size in w x h x b"
        assert len(stride_size) == 2, "Pass Split size in w x h"

        self.img_size = img_size
        self.stride_size = stride_size

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, win_number):
        raise NotImplementedError

    def _calculate_windows(self) -> list:
        """
        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows

        :return:
        """

        raise NotImplementedError


class StrideImage(StrideOver):
    def __init__(self, stride_size: tuple, img_size: tuple):
        """
        W = Columns
        H = Rows

        Input = W x H

        :param stride_size: tuple(W x H), Size to split the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split operation is to be performed
        """
        super().__init__(stride_size, img_size)
        self.windows = self._calculate_windows()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, win_number):
        window = self.windows[win_number]
        return win_number, window

    def _calculate_windows(self) -> list:
        """
        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows

        :return:
        """
        cropped_windows = list()
        split_col, split_row = self.stride_size

        img_col = self.img_size[0]
        img_row = self.img_size[1]

        iter_col = 1
        iter_row = 1

        for col in range(0, img_col, split_col):
            if iter_col == np.ceil(img_col / split_col):
                col = img_col - split_col
            else:
                iter_col += 1
            for row in range(0, img_row, split_row):
                if iter_row == np.ceil(img_row / split_row):
                    row = img_row - split_row
                else:
                    iter_row += 1
                if row + split_row <= img_row and col + split_col <= img_col:
                    cropped_windows.append(
                        ((row, row + split_row), (col, col + split_col))
                    )
            iter_row = 1
        return cropped_windows


class StrideGeoTransform(StrideOver):
    def __init__(
        self,
        stride_size: tuple,
        img_size: tuple,
        min_x=None,
        max_y=None,
        transform=None,
    ):
        """
        W = Columns
        H = Rows

        Input = W x H

        :param stride_size: tuple(W x H), Size to split the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split operation is to be performed
        """

        super().__init__(stride_size, img_size)
        self._min_x = min_x
        self._max_y = max_y
        self._transform = transform

        if (self._min_x, self._max_y, self._transform) is not None:
            self.windows = self._calculate_windows()

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, win_number):
        window = self.windows[win_number]
        return win_number, window

    def _compute_step_size(self):
        """

        :return:
        """
        d_col_tiles = int(np.ceil(self.img_size[0] / self.stride_size[0]))
        d_row_tiles = int(np.ceil(self.img_size[1] / self.stride_size[1]))

        return d_col_tiles, d_row_tiles

    def _compute_overlap(self):
        """

        :return:
        """
        over_lap_in_x = None
        overlap_step_in_x = None

        overlap_in_y = None
        overlap_step_in_y = None

        if self._if_overlap_in_col_direction():
            over_lap_in_x = self._min_x + self._transform[0] * (
                self.img_size[0] - self.stride_size[0]
            )
            overlap_step_in_x = int((over_lap_in_x - self._min_x))
        if self._if_overlap_in_row_direction():
            overlap_in_y = self._max_y + self._transform[4] * (
                self.img_size[1] - self.stride_size[1]
            )
            overlap_step_in_y = int((self._max_y - overlap_in_y))
        return over_lap_in_x, overlap_in_y, overlap_step_in_x, overlap_step_in_y

    def _compute_offset(self, col, row, d_col, d_row):
        """

        :param col:
        :param row:
        :param d_col:
        :param d_row:
        :return:
        """
        offset_max_x = self._min_x + self._transform[0] * (col * d_col)
        offset_min_y = self._max_y + self._transform[4] * (row * d_row)

        offset_step_in_x = int(((offset_max_x - self._min_x) / d_col))
        offset_step_in_y = int(((self._max_y - offset_min_y) / d_row))

        return offset_max_x, offset_min_y, offset_step_in_x, offset_step_in_y

    def _if_overlap_in_col_direction(self):
        """

        :return:
        """
        return True if self.img_size[0] % self.stride_size[0] else False

    def _if_overlap_in_row_direction(self):
        """

        :return:
        """
        return True if self.img_size[1] % self.stride_size[1] else False

    def _calculate_windows(self) -> list:
        """
        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows

        :return:
        """

        extents = list()

        split_col, split_row = self.stride_size[0], self.stride_size[1]

        d_col, d_row = self._compute_step_size()

        (
            offset_max_x,
            offset_min_y,
            offset_step_in_x,
            offset_step_in_y,
        ) = self._compute_offset(split_col, split_row, d_col, d_row)

        (
            over_lap_in_x,
            overlap_in_y,
            overlap_step_in_x,
            overlap_step_in_y,
        ) = self._compute_overlap()

        for y in range(d_row):
            for x in range(d_col):

                if (x == d_col - 1) and self._if_overlap_in_col_direction():

                    tx_start = x * overlap_step_in_x + self._min_x
                else:
                    tx_start = x * offset_step_in_x + self._min_x
                if y == (d_row - 1) and self._if_overlap_in_row_direction():
                    ty_start = (
                        y * overlap_step_in_y
                        + self._max_y
                        + self._transform[4] * self.img_size[1]
                    )
                else:
                    ty_start = (
                        y * offset_step_in_y
                        + self._max_y
                        + self._transform[4] * self.img_size[1]
                    )
                tx_end = tx_start + offset_step_in_x - 1
                ty_end = ty_start + offset_step_in_y - 1

                extents.append((tx_start, ty_start, tx_end, ty_end))
        return extents
