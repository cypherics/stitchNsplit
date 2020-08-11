import time

import numpy as np
import os

from stitch_n_split.utility import make_save_dir, open_image, save_image, Printer
from stitch_n_split.windows import get_window


class Stitch:
    def __init__(self, src_size: tuple, dst_size: tuple):
        """

        :param src_size: tuple(W x H X B), Size to stitch the Image in, typically smaller than img_size
        :param dst_size: tuple(W x H X B), Size on which image is going to be stitched operation is to be performed
        """
        if src_size[0] > dst_size[0] or src_size[1] > dst_size[1]:
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(src_size, dst_size)
            )
        self.src_size = src_size
        self.dst_size = dst_size

        self.window = get_window(
            stride_size=self.src_size, img_size=self.dst_size
        )

    def __len__(self):
        return len(self.window.window_collection)

    def __getitem__(self, index):
        return index, self.window.window_collection[index]

    @staticmethod
    def stitch_image(image: np.ndarray, stitched_image: np.ndarray, window):
        """

        :param image:
        :param stitched_image:
        :param window:
        :return:
        """

        part_1_x = window[0][0]
        part_1_y = window[0][1]
        part_2_x = window[1][0]
        part_2_y = window[1][1]

        cropped_image = stitched_image[part_1_x:part_1_y, part_2_x:part_2_y]

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
        stitched_image[part_1_x:part_1_y, part_2_x:part_2_y] = image
        return stitched_image

    def stitch_generator(self, files):
        """
        A generator to pass chunks of files equal to len of windows

        :param files:
        :return:
        """
        for i in range(0, len(files), len(self.window.window_collection)):
            yield files[i : i + len(self.window.window_collection)]

    def perform_stitch(self, dir_path: str):

        """
        The methid makes an assumption that all the incoming images are in sequence to which a stitch is performed

        Stitch Images from the given directory based on the dst_size and src_size
        :param dir_path:
        :return:
        """
        files = [file for file in os.listdir(dir_path)]
        save_path = make_save_dir(
            os.getcwd(), os.path.join("stitchNsplit_store", str(int(time.time())))
        )
        stitch_gen = self.stitch_generator(files)

        for i, collection in enumerate(stitch_gen):
            Printer.print(
                "Stitching In Progress for {} out of {}".format(
                    (i + 1) * len(collection), len(files)
                )
            )

            stitched_image = np.zeros(self.dst_size)

            for iterator, file in enumerate(collection):
                file_path = os.path.join(dir_path, file)
                image = open_image(file_path)
                stitched_image = self.stitch_image(
                    image, stitched_image, self.window.window_collection[iterator]
                )
            save_image(
                os.path.join(save_path, "stitched_{}.png".format(i)),
                np.array(stitched_image, dtype=np.uint8),
            )
