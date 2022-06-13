import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
Gradients / Edge Detection
Video timestamp: 2:26:27

"""

# Read image:
base_img = cv.imread('photos/bonsai.jpg')


# def rescaleFrame(frame, scale=0.40):
#     # Rescale size by 50%
#     # Images, Videos and Live Video(e.g. webcams)
#
#     width = int(frame.shape[1] * scale // 0.8)
#     height = int(frame.shape[1] * scale)
#     dimensions = (width, height)
#
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Original image
img = base_img
cv.imshow('Original Image', img)

# Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
# Computes gradients of image
lap = cv.Laplacian(gray, cv.CV_64F)

# Calculate absolute values for image
# Then convert to image specific datatype
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)

# Combine X + Y
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('Combined Sobel', combined_sobel)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)

# Canny edge detector
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)
