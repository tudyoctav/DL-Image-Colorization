import numpy as np
import cv2
from itertools import product, permutations, combinations
import copy


# Define the width and height of the image
width = 28
height = 28


def generate_gradient_image(start_color, end_color, width, height):
    # Create an empty image with the desired dimensions
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the color step for each channel
    step = (end_color - start_color) / (width - 1)

    # Generate the gradient image
    for x in range(width):
        color = start_color + (step * x)
        image[:, x] = color

    return image

current_combo = [0, 0, 0]


def generate_permutations(i, color):
    current_combo[i] = color
    if i >= 2:
        if 255 in current_combo and 0 in current_combo:
            ans.append(copy.deepcopy(current_combo))
        return

    generate_permutations(i + 1, 0)
    # print("temp1:", ans)
    # ans.append(temp1)

    generate_permutations(i + 1, 255)
    # print("temp2:", ans)

    # ans.append(temp2)

    # return ans


if __name__ == '__main__':

    ans = []
    generate_permutations(0, 0)
    generate_permutations(0, 255)
    print(ans)

    for i in range(len(ans)):
        for j in range(len(ans)):
            if i != j:
                # Define the start and end colors (in BGR format)
                # start_color = np.array([255, 0, 0])  # Red
                # end_color = np.array([0, 0, 255])    # Blue

                start_color = np.array(ans[i])
                end_color = np.array(ans[j])

                # Generate the gradient image
                gradient_image = generate_gradient_image(start_color, end_color, width, height)

                # Save the gradient image
                path_to_generated_gradient = "../../data/generated/gradients_28x28/gradient_image_"
                cv2.imwrite(path_to_generated_gradient + str(i) + '_' + str(j) + '.jpg', gradient_image)