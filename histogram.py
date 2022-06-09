import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""

Visualise distribution of pixel intensities

"""

# Read image:
base_img = cv.imread('photos/bonsai.jpg')


def rescaleFrame(frame, scale=0.40):
    # Rescale size by 50%
    # Images, Videos and Live Video(e.g. webcams)

    width = int(frame.shape[1] * scale // 0.8)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Original image
img = rescaleFrame(base_img)
cv.imshow('Original Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')  # Dimensions of mask must be same size as original img.

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale', gray)

mask = cv.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Mask', masked)

# Computing histograms for greyscale images
# Change 'None' to <mask name> to get histogram of mask
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])  # Computes histogram for selected img

# Plot grayscale histogram
# plt.figure()
# plt.title('Grayscale Histogram (Masked)')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_hist)
# plt.xlim([0, 256])
# plt.show()

# Colour Histograms
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Colour Histogram (Masked)')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
for i, col in enumerate(colors):
    # plot histogram
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv.waitKey(0)
