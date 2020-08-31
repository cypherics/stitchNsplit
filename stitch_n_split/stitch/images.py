import time

import numpy as np
import os

from stitch_n_split.utility import make_save_dir, open_image, save_image, Printer
from fragment.fragment import Fragment, ImageFragment


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

        self.image_fragment = ImageFragment.get_image_fragment(fragment_size=self.src_size, org_size=self.dst_size)

    def __len__(self):
        return len(self.image_fragment.collection)

    def __getitem__(self, index):
        return index, self.image_fragment.collection[index]

    @staticmethod
    def stitch_image(image: np.ndarray, stitched_image: np.ndarray, fragment: Fragment):
        """

        :param fragment:
        :param image:
        :param stitched_image:
        :return:
        """

        return fragment.transfer_fragment(transfer_from=image, transfer_to=stitched_image)

    def stitch_generator(self, files):
        """
        A generator to pass chunks of files equal to len of fragments

        :param files:
        :return:
        """
        for i in range(0, len(files), len(self.image_fragment.collection)):
            yield files[i : i + len(self.image_fragment.collection)]

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
                    image, stitched_image, self.image_fragment.collection[iterator]
                )
            save_image(
                os.path.join(save_path, "stitched_{}.png".format(i)),
                np.array(stitched_image, dtype=np.uint8),
            )
