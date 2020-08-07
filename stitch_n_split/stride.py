import numpy as np


class StrideOver:
    def __init__(self, split_size: tuple, img_size: tuple):
        """
        W = Columns
        H = Rows

        Input = W x H

        :param split_size: tuple(W x H), Size to split the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split operation is to be performed
        """
        if split_size[0] > img_size[0] or split_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(split_size, (img_size[0], img_size[1]))
            )

        assert len(img_size) == 3, "Pass Image size in w x h x b"
        assert len(split_size) == 2, "Pass Split size in w x h"

        self._data = None
        self.img_size = img_size
        self.windows = self.get_windows(split_size, img_size)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, win_number):
        window = self.windows[win_number]
        return win_number, window

    @staticmethod
    def get_windows(split_size: tuple, img_size: tuple) -> list:
        """
        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows


        :param split_size:
        :param img_size:
        :return:
        """
        cropped_windows = list()
        split_col, split_row = split_size

        img_row = img_size[0]
        img_col = img_size[1]

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
