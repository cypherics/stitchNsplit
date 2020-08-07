import os
import numpy as np
import time

import rasterio

from stitch_n_split.stride import StrideOver
from stitch_n_split.utility import make_save_dir, open_image, save_image


class Split:
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(W x H), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split1 operation is to be performed
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

        :param dir_path: dir path over which split1 is to be performed
        :return:
        """
        raise NotImplementedError

    def _split_image(self, image, image_save_path: str):
        """

        :param image:
        :return:
        """
        raise NotImplementedError

    def _extract_data(self, image, window):
        """

        :param image:
        :param window:
        :return:
        """
        raise NotImplementedError

    def win_number_split(self, image, win_number: int):
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

    def window_split(self, image, window):
        return self._extract_data(image, window)


class SplitNonGeo(Split):
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(W x H), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split1 operation is to be performed
        """
        super().__init__(split_size, img_size)

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

    def _extract_data(self, image: np.ndarray, window: tuple) -> np.ndarray:
        """

        :param image:
        :param window:
        :return:
        """

        return image[window[0][0] : window[0][1], window[1][0] : window[1][1]]


class SplitGeo(Split):
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(W x H), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(W x H X 3), Size on which split1 operation is to be performed
        """
        super().__init__(split_size, img_size)

    def perform_split(self, dir_path: str):
        """
        The images in the directory must have .tif extention

        :param dir_path: dir path over which split1 is to be performed
        :return:
        """
        files = [file for file in os.listdir(dir_path)]
        save_path = make_save_dir(
            os.getcwd(), os.path.join("stitchNsplit_store", str(int(time.time())))
        )
        for iterator, file in enumerate(files):
            file_path = os.path.join(dir_path, file)

            image = open_image(file_path, is_geo_reference=True)
            image_save_path = os.path.join(save_path, file)
            self._split_image(image, image_save_path)

    def _split_image(self, image: rasterio.io.DatasetReader, image_save_path: str):
        """

        :param image: an image open via rasterio
        :return:
        """
        for index, tiff_window in zip(
            range(0, len(self.stride.windows)), self.stride.windows
        ):
            split_image, kwargs_split_image = self._extract_data(image, tiff_window)
            split_path = image_save_path.split(".")
            save_path = "{}_{}.{}".format(split_path[0], index, split_path[-1])
            save_image(save_path, split_image, True, **kwargs_split_image)

    def _extract_data(
        self, image: rasterio.io.DatasetReader, window
    ) -> (np.ndarray, dict):
        """
        The operation of spiting the images and copying its geo reference is carried out using a sliding window
        approach, where window specifies which part of the original image is to be processed

        :param image:
        :param window: the split1 window size
        :return:
        """
        split_image = image.read(window=window)

        kwargs_split_image = image.meta.copy()
        kwargs_split_image.update(
            {
                "height": self.split_size[1],
                "width": self.split_size[0],
                "transform": image.window_transform(window),
            }
        )

        return split_image, kwargs_split_image