import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""

Split image into three colour channels

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

# Create blank image for demonstration of individual colour channels
blank = np.zeros(img.shape[:2], dtype='uint8')

b, g, r = cv.split(img)

# 'blank' sets colour channel to black, only showing one colour
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Original Image', img)
cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

# Visualise shape of image
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

# Merge colour channels
merged = cv.merge([b, g, r])
cv.imshow('Merged Image', merged)

cv.waitKey(0)

