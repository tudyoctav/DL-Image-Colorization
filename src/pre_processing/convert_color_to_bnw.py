import cv2

if __name__ == '__main__':
    # Convert to Grayscale
    for i in range(1, 9):
        img_gray = cv2.imread("data/color/polaroids/p" + str(i) + ".jpg", 0)
        cv2.imwrite("data/bnw/polaroids/p" + str(i) + ".jpg", img_gray)
        cv2.waitKey(0)
