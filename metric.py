import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
# define the function to compute MSE between two images
def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err/(float(h*w))
    return mse, diff

def grayscale_squared_error_rate(image1, image2, display=False):
    # Convert images to grayscale
    assert image1.shape[0] == image2.shape[0] and image1.shape[1] == image2.shape[1] , "Images must have the same dimensions"
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    error, diff = mse(image1_gray, image2_gray)
    if display == True:
        cv2.imshow("Grayscale_difference", diff)
    return error

def chroma_error_ratio(image1, image2, display=False):
    # As described https://ieeexplore.ieee.org/document/9899824
    # Convert images to the YUV color space

    h, w, c = image1.shape
    image1_yuv = cv2.cvtColor(image1, cv2.COLOR_BGR2YUV)
    image2_yuv = cv2.cvtColor(image2, cv2.COLOR_BGR2YUV)
    # Extract U and V channels from the YUV images
    y_i = image1_yuv[:, :, 0].flatten() # we don't need this
    u_i = image1_yuv[:, :, 1].flatten().astype(np.float64)
    v_i = image1_yuv[:, :, 2].flatten().astype(np.float64)

    y_k = image2_yuv[:, :, 0].flatten() # we don't need this
    u_k = image2_yuv[:, :, 1].flatten().astype(np.float64)
    v_k = image2_yuv[:, :, 2].flatten().astype(np.float64)

    # Return U and V channels as one-dimensional arrays

    # I think the float(h*w) cancels out, but included for consistency with the paper
    numerator = np.sum(np.square(u_k) + np.square(v_k)) / (float(h*w))
    u_diff = u_i - u_k
    v_diff = v_i - v_k

    denominator = np.sum(np.square(u_diff) + np.square(v_diff)) / (float(h*w))
    result = 10 * np.log(numerator / denominator)

    if display == True:
        diff1 = np.empty_like(image1)
        diff1[:, :, 0] = np.reshape(y_k, (h,w))
        diff1[:, :, 1] = np.reshape(u_diff, (h,w))
        diff1[:, :, 2] = 0
        diff1 = cv2.cvtColor(diff1, cv2.COLOR_YUV2BGR)
        cv2.imshow("U difference in BGR", diff1)
        cv2.imshow("U difference", u_diff.reshape((h,w)))
        # Create a new figure and plot the data
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(u_diff.reshape((h,w)))
        # Add a colorbar
        cbar1 = fig1.colorbar(im1)
        # Set a title for the figure
        ax1.set_title("U difference")

        diff2 = np.empty_like(image1)
        diff2[:, :, 0] = np.reshape(y_k, (h,w))
        diff2[:, :, 1] = 0
        diff2[:, :, 2] = np.reshape(v_diff, (h,w))
        diff2 = cv2.cvtColor(diff2, cv2.COLOR_YUV2BGR)
        cv2.imshow("v difference in BGR", diff2)
        pixel_plot_v = plt.imshow(v_diff.reshape((h,w)))
        # Create a new figure and plot the data
        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(v_diff.reshape((h,w)))
        # Add a colorbar
        cbar2 = fig2.colorbar(im2)
        # Set a title for the figure
        ax2.set_title("V difference")


        diff = np.empty((h,w,3))
        diff[:, :, 0] = np.reshape(y_k, (h,w))
        diff[:, :, 1] = np.reshape(u_diff, (h,w))
        diff[:, :, 2] = np.reshape(v_diff, (h,w)) 
        diff = cv2.cvtColor(diff.astype('uint8'), cv2.COLOR_YUV2BGR)
        cv2.imshow("Colour difference in BGR", diff)
        
        plt.show(block=False)
    
    return result


def calculate_difference(image1, image2, display=False):
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1]:
        print("Resizing the images to match")
        image1 = cv2.resize(image1,(image2.shape[1], image2.shape[0]))
    if display:
        cv2.imshow("Input image", image1)
        cv2.imshow("Output image", image2)
    print("Grayscale squared error rate:", grayscale_squared_error_rate(image1, image2, display=display))
    print("Chroma error ratio (CER):", chroma_error_ratio(image1, image2, display=display))

def main():
    parser = argparse.ArgumentParser(description='Extract U and V channels from two images.')
    parser.add_argument('image1', help='Path to the first input image')
    parser.add_argument('image2', help='Path to the second input image')
    args = parser.parse_args()

    # Read the input images
    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)

    if image1 is None:
        print('Error: Failed to read image1')
        return
    if image2 is None:
        print('Error: Failed to read image2')
        return

    calculate_difference(image1, image2, display=True)
    print("Press any key inside the figures...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
