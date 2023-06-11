import os.path

import cv2
import numpy as np
import shutil

width = 28
height = 28
channels = 3  # RGB image


def generate_image(shade, pos_x, pos_y):
    image = np.zeros((height, width, channels), dtype=np.uint8)
    image[pos_x][pos_y] = shade
    cv2.imwrite(path_to_folder + "/img_" + str(pos_x) + "_" + str(pos_y) + ".png", image)


if __name__ == '__main__':
    base_folder = "../../data/generated/pixels_placed_different_locations/"
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.mkdir(base_folder)

    for k in range(0, 255, 25):
        print("Generating: " + str(k))

        path_to_folder = "../../data/generated/pixels_placed_different_locations/" + str(k)
        if not os.path.isdir(path_to_folder):
            os.mkdir(path_to_folder)
        for i in range(28):
            for j in range(28):
                generate_image(k, i, j)

