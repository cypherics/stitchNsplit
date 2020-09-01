import os
import numpy as np
import time

import rasterio

from fragment.fragment import ImageFragment, Fragment
from stitch_n_split.utility import make_save_dir, open_image, save_image, Printer


class Split:
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(H x W X B), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(H x W X B), Size on which split1 operation is to be performed
        """
        if split_size[0] > img_size[0] or split_size[1] > img_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(split_size, (img_size[0], img_size[1]))
            )
        self.split_size = split_size
        self.img_size = img_size

        self.image_fragment = ImageFragment.image_fragment_3d(fragment_size=self.split_size, org_size=self.img_size)

    def __len__(self):
        return len(self.image_fragment.collection)

    def __getitem__(self, index):
        return index, self.image_fragment.collection[index]

    def perform_directory_split(self, dir_path: str):
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

    def _extract_data(self, image, fragment: Fragment):
        """

        :param image:
        :param fragment:
        :return:
        """
        raise NotImplementedError

    def window_split(self, image: np.ndarray, fragment: Fragment):
        return self._extract_data(image, fragment)


class SplitNonGeo(Split):
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(H x W), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(H x W X 3), Size on which split1 operation is to be performed
        """
        super().__init__(split_size, img_size)

    def perform_directory_split(self, dir_path: str):
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
            Printer.print("Split In Progress {} / {}".format(iterator + 1, len(files)))
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
            range(0, len(self.image_fragment.collection)), self.image_fragment.collection
        ):
            split_image = self._extract_data(image, tiff_window)
            split_path = image_save_path.split(".")
            save_path = "{}_{}.{}".format(split_path[0], index, split_path[-1])
            save_image(save_path, split_image)

    def _extract_data(self, image: np.ndarray, fragment: Fragment) -> np.ndarray:
        """

        :param image:
        :param fragment:
        :return:
        """

        return fragment.get_fragment_data(image)


class SplitGeo(Split):
    def __init__(self, split_size: tuple, img_size: tuple):
        """

        :param split_size: tuple(H x W), Size to split1 the Image in, typically smaller than img_size
        :param img_size: tuple(H x W X 3), Size on which split1 operation is to be performed
        """
        super().__init__(split_size, img_size)

    def perform_directory_split(self, dir_path: str):
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
            Printer.print("Split In Progress {} / {}".format(iterator + 1, len(files)))

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
            range(0, len(self.image_fragment.collection)), self.image_fragment.collection
        ):
            split_image, kwargs_split_image = self._extract_data(image, tiff_window)
            split_path = image_save_path.split(".")
            save_path = "{}_{}.{}".format(split_path[0], index, split_path[-1])
            save_image(save_path, split_image, True, **kwargs_split_image)

    def _extract_data(
        self, image: rasterio.io.DatasetReader, fragment: Fragment
    ) -> (np.ndarray, dict):
        """
        The operation of spiting the images and copying its geo reference is carried out using a sliding window
        approach, where fragment specifies which part of the original image is to be processed

        :param image:
        :param fragment: the split1 size
        :return:
        """
        split_image = image.read(window=fragment.position)

        kwargs_split_image = image.meta.copy()
        kwargs_split_image.update(
            {
                "height": self.split_size[0],
                "width": self.split_size[1],
                "transform": image.window_transform(fragment.position),
            }
        )

        return split_image, kwargs_split_image
