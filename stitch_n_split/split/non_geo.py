import os
import time

import numpy as np

from stitch_n_split.stride import StrideOver
from stitch_n_split.utility import make_save_dir
from stitch_n_split.utility import open_image, save_image


class SplitNonGeoReference:
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(W x H), Size to split the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split operation is to be performed
        """
        if split_size[0] > img_size[0] or split_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(split_size, (img_size[0], img_size[1]))
            )
        self.split_size = split_size
        self.img_size = img_size

        self.stride = StrideOver(self.split_size, self.img_size)

    def perform_split(self, dir_path: str):
        """

        :param dir_path: str
        :return:
        """
        files = [file for file in os.listdir(dir_path)]
        save_path = make_save_dir(
            os.getcwd(), os.path.join("stitchNsplit_store", str(int(time.time())))
        )
        for iterator, file in enumerate(files):
            file_path = os.path.join(dir_path, file)

            image = open_image(file_path)
            w, h, b = image.shape
            if b > 3:
                raise ValueError(
                    "For Non Geo Reference Imagery More than 3 band is not supported"
                )
            image_save_path = os.path.join(save_path, file)
            self._split_image(image, image_save_path)

    def _split_image(self, image: np.ndarray, image_save_path: str):
        """

        :param image:
        :return:
        """
        for index, tiff_window in zip(
                range(0, len(self.stride.windows)), self.stride.windows
        ):
            split_image = self._extract_data(image, tiff_window)
            split_path = image_save_path.split(".")
            save_path = "{}_{}.{}".format(split_path[0], index, split_path[-1])
            save_image(save_path, split_image)

    @staticmethod
    def _extract_data(image: np.ndarray, window: tuple) -> np.ndarray:
        """

        :param image:
        :param window:
        :return:
        """

        return image[window[0][0]: window[0][1], window[1][0]: window[1][1]]

    def win_number_split(self, image: np.ndarray, win_number: int) -> np.ndarray:
        if type(win_number) != int:
            raise TypeError("Given {}, Expected {}".format(type(win_number), "Int"))
        if win_number < 0 or win_number >= len(self.stride.windows):
            raise ValueError(
                "Given {}, Expected In between {} to {}".format(
                    win_number, 0, len(self.stride.windows)
                )
            )
        window = self.stride.windows[win_number]
        return self._extract_data(image, window)

    def window_split(self, image: np.ndarray, window: tuple) -> np.ndarray:
        return self._extract_data(image, window)
