import matplotlib.pyplot as plt
import cv2
import numpy as np


def show_histogram(bnw_image, plot_name='plot', amount_of_bins=10):
    """
    Save histogram to file
    :param bnw_image: 1D image (black&white), can also be just 1 color channel if you'd like
    :param plot_name: name of the output file
    :param amount_of_bins: how many bins to show, larger it is, the more detail the histogram
    :return: nothing!
    """
    hist, bins = np.histogram(bnw_image, bins=np.arange(0, 255))
    bin_size = 255 / amount_of_bins

    # Showing the plot
    plt.bar(bins[:-1] / bin_size, hist/1000, width=1)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title(plot_name)
    plt.savefig('plots/' + plot_name + '.png')
    plt.clf()   # Clear figure


# Histogram equalization
def histogram_equalization(bnw_image, plot_name='original_and_equalized_image'):
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(bnw_image)

    # Display original and equalized images
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(bnw_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(equalized_image, cmap='gray')
    axes[1].set_title('Equalized Image')
    plt.savefig('plots/' + plot_name + '.png')
    plt.show()
    return equalized_image


if __name__ == '__main__':
    print("Histogram methods!")
    # for i in range(1, 9):
    image_ind = 9
    color_image = cv2.imread("data/initial_color/polaroids/p" + str(image_ind) + ".jpg", cv2.IMREAD_COLOR)
    gray_image = cv2.imread("data/initial_color/polaroids/p" + str(image_ind) + ".jpg", cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("color_image", color_image)
    # cv2.waitKey(0)
    blue_channel = color_image[:, :, 0]
    green_channel = color_image[:, :, 1]
    red_channel = color_image[:, :, 2]

    # Black & White image histogram
    show_histogram(gray_image, 'b&w', 255)

    # All color channels
    show_histogram(blue_channel, '_blue')
    show_histogram(green_channel, '_green')
    show_histogram(red_channel, '_red')

    equalized = histogram_equalization(gray_image)
    cv2.imwrite("data/equalized/p"+str(image_ind)+".png", equalized)
    show_histogram(equalized, 'equalized', 255)
