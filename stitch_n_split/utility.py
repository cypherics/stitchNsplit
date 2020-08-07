import os
import cv2
import rasterio


def open_image(path, is_geo_reference=False):
    if is_geo_reference:
        return rasterio.open(path, driver="GTiff")
    else:
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(path, image, is_geo_reference=False, **kwargs_image):
    if is_geo_reference:
        with rasterio.open(path, "w", **kwargs_image) as dst:
            dst.write(image)
    else:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def make_save_dir(current_dir, folder_name):
    new_dir = os.path.join(current_dir, folder_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    return new_dir
