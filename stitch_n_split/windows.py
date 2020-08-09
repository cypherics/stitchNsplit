import numpy as np


class Window:
    def __init__(self, windows: list):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, win_number):
        window = self.windows[win_number]
        return win_number, window

    @classmethod
    def image_windows(cls, stride_size: tuple, img_size: tuple):
        """
        W = Columns
        H = Rows

        Input = W x H

        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows

        :return:
        """
        if stride_size[0] > img_size[0] or stride_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(stride_size, (img_size[0], img_size[1]))
            )
        cropped_windows = list()
        split_col, split_row = (stride_size[0], stride_size[1])

        img_col = img_size[0]
        img_row = img_size[1]

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
        return cls(cropped_windows)

    @classmethod
    def image_geo_windows(cls, stride_size: tuple, img_size: tuple):
        """
        Provides Window information while computing grid extent over geo referenced image
        :param stride_size:
        :param img_size:
        :return:
        """
        if stride_size[0] > img_size[0] or stride_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(stride_size, (img_size[0], img_size[1]))
            )
        cropped_windows = list()
        split_col, split_row = (stride_size[0], stride_size[1])

        img_col = img_size[0]
        img_row = img_size[1]

        iter_col = 1
        iter_row = 1

        for row in range(0, img_row, split_row):
            if iter_row == np.ceil(img_row / split_row):
                row = img_row - split_row
            else:
                iter_row += 1
            for col in range(img_col, 0, -split_col):
                if iter_col == np.ceil(img_col / split_col):
                    col = img_col - col
                else:
                    iter_col += 1
                cropped_windows.append(((row, row + split_row), (col - split_col, col)))
            iter_col = 1
        return cls(cropped_windows)
