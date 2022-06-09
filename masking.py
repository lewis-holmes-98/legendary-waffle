import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""

Allows for focus on specific parts of image. I.e focusing on people faces only. 

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
cv.imshow('Blank Image', blank)

circle = cv.circle(blank.copy(), (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

rectangle = cv.rectangle(blank.copy(), (30, 30), (500, 500), 255, -1)

weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Circle + Rectangle', weird_shape)

# Create masked image
masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked Image', masked)

cv.waitKey(0)

