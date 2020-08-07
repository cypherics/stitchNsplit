import os
import time

import numpy as np
import rasterio
from affine import Affine

from stitch_n_split.stride import StrideOver
from stitch_n_split.utility import make_save_dir, open_image, save_image


class Stitch:
    def __init__(self, img_size: tuple, stitch_size: tuple):
        """

        :param img_size:
        :param stitch_size:
        """
        self.img_size = img_size
        self.stitch_size = stitch_size

        self.stride = StrideOver(self.img_size, self.stitch_size)

        self._stitched_image = None
        self._stitched_iterator = 0
        self._kwargs = None

    def _refresh_stitch_meta(self):
        self._stitched_image = np.zeros(self.stitch_size)
        self._stitched_iterator = 0
        self._kwargs = None

    def _stitch_image(self, image: np.ndarray):

        part_1_x = self.stride.windows[self._stitched_iterator][0][0]
        part_1_y = self.stride.windows[self._stitched_iterator][0][1]
        part_2_x = self.stride.windows[self._stitched_iterator][1][0]
        part_2_y = self.stride.windows[self._stitched_iterator][1][1]

        cropped_image = self._stitched_image[part_1_x:part_1_y, part_2_x:part_2_y]

        image = cropped_image + image

        if np.any(cropped_image):
            intersecting_prediction_elements = np.zeros(cropped_image.shape)
            intersecting_prediction_elements[cropped_image > 0] = 1

            non_intersecting_prediction_elements = 1 - intersecting_prediction_elements

            intersected_prediction = image * intersecting_prediction_elements
            aggregate_prediction = intersected_prediction / 2

            non_intersected_prediction = np.multiply(
                non_intersecting_prediction_elements, image
            )
            image = aggregate_prediction + non_intersected_prediction
        self._stitched_image[part_1_x:part_1_y, part_2_x:part_2_y] = image

    def perform_stitch(self, dir_path: str):
        """

        :param dir_path:
        :return:
        """
        raise NotImplementedError

    def _stitch_reference(self, image: rasterio.io.DatasetReader):
        raise NotImplementedError

    def _generate_reference(self):
        raise NotImplementedError


class StitchGeo(Stitch):
    def __init__(self, img_size: tuple, stitch_size: tuple):
        """

        :param img_size:
        :param stitch_size:
        """
        super().__init__(img_size, stitch_size)

        self._xs = list()
        self._ys = list()
        self._res = None

    def perform_stitch(self, dir_path: str):
        """

        :param dir_path:
        :return:
        """
        files = [file for file in os.listdir(dir_path)]
        save_path = make_save_dir(
            os.getcwd(), os.path.join("stitchNsplit_store", str(int(time.time())))
        )

        self._refresh_stitch_meta()

        for iterator, file in enumerate(files):
            file_path = os.path.join(dir_path, file)

            image = open_image(file_path, is_geo_reference=True)
            if self._kwargs is None:
                self._kwargs = image.meta.copy()
            self._stitch_reference(image)

            self._stitch_image(self._adjust_shape(image.read()))

            if self._stitched_iterator == len(self.stride.windows) - 1:

                self._stitched_image = self._stitched_image.swapaxes(-1, 0)
                self._stitched_image = self._stitched_image.swapaxes(-1, 1)

                self._kwargs.update(
                    {
                        "transform": self._generate_reference(),
                        "width": self.stitch_size[0],
                        "height": self.stitch_size[1],
                    }
                )
                save_image(
                    os.path.join(save_path, file),
                    np.array(self._stitched_image, dtype=np.uint8),
                    is_geo_reference=True,
                    **self._kwargs
                )

                self._refresh_stitch_meta()
            else:
                self._stitched_iterator += 1

    def _stitch_reference(self, image: rasterio.io.DatasetReader):

        if self._res is None:
            self._res = image.res
        left, bottom, right, top = image.bounds
        self._xs.extend([left, right])
        self._ys.extend([bottom, top])

    def _generate_reference(self):
        if len(self._res) == 1:
            res = (self._res[0], self._res[0])
        else:
            res = self._res
        west, south, east, north = (
            min(self._xs),
            min(self._ys),
            max(self._xs),
            max(self._ys),
        )
        output_transform = Affine.translation(west, north)
        output_transform *= Affine.scale(res[0], -res[1])
        return output_transform

    @staticmethod
    def _adjust_shape(np_image: np.ndarray) -> np.ndarray:
        np_image = np_image.swapaxes(-1, 0)
        np_image = np_image.swapaxes(0, 1)
        return np_image
