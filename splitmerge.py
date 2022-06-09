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
cv.imshow('Original', img)

b, g, r = cv.split(img)
