import os
import time
import numpy as np
import rasterio

from stitch_n_split.stride_over import Stride
from stitch_n_split.utility import make_save_dir
from stitch_n_split.utility import open_image, save_image


class SplitGeoReference:
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size:
        :param img_size:
        """
        self.split_size = split_size
        self.img_size = img_size

        self.stride = Stride(self.split_size, self.img_size)

    def split_geo_reference_images_in_dir(self, dir_path: str):
        """

        :param dir_path:
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

        :param image:
        :return:
        """
        for index, tiff_window in zip(
            range(0, len(self.stride.windows)), self.stride.windows
        ):
            split_image, kwargs_split_image = self._extract_data(image, tiff_window)
            split_path = image_save_path.split(".")
            save_path = "{}_{}.{}".format(split_path[0], index, split_path[-1])
            save_image(save_path, split_image, True, **kwargs_split_image)

    def _extract_data(self, image: rasterio.io.DatasetReader, window) -> (np.ndarray, dict):
        """

        :param image:
        :param window:
        :return:
        """
        split_image = image.read(window=window)

        kwargs_split_image = image.meta.copy()
        kwargs_split_image.update(
            {
                "crs": "EPSG:4326",
                "height": self.split_size[1],
                "width": self.split_size[0],
                "transform": image.window_transform(window),
            }
        )

        return split_image, kwargs_split_image

    def win_number_split(self, image: rasterio.io.DatasetReader, win_number: int) -> (np.ndarray, dict):
        window = self.stride.windows[win_number]
        return self._extract_data(image, window)

    def window_split(self, image: rasterio.io.DatasetReader, window: tuple) -> (np.ndarray, dict):
        return self._extract_data(image, window)
