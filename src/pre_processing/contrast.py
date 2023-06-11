import numpy as np
import math
import cv2
import shutil
import os
import random


def contrast_stretch(input_image, total_entries):
    stretched_mat = input_image.copy()
    frame_width, frame_height = stretched_mat.shape
    for x in range(frame_height):
        for y in range(frame_width):
            if total_entries > 0:
                stretched_mat[y][x] = math.ceil(stretched_mat[y][x] / total_entries * 255)
    return stretched_mat


if __name__ == '__main__':
    print("Contrast methods!")