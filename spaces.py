import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
    Switch between colour Spaces
    
    BGR - Blue, Green, Red
    RBG - Red, Green, Blue
    
    openCV uses BGR
    matplotlib used RBG
"""

# Read image:
base_img = cv.imread('photos/bonsai.jpg')


def rescaleFrame(frame, scale=0.40):
    # Rescale size by 50%
    # Images, Videos and Live Video(eg. webcams)

    width = int(frame.shape[1] * scale // 0.8)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Original image
img = rescaleFrame(base_img)
cv.imshow('Original', img)

# BGR to Greyscale
# Note: Greyscale cannot be converted directly to LAB
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Greyscale', grey)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*A*B
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
# Colours will be inverse as openCV does not use RBG colour space
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RBG (openCV revered)', rgb)

# Matplotlib uses RBG colour space, img will appear correctly
# plt.imshow(rgb)
# plt.show()

# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV -> BGR', hsv_bgr)

# LAB to BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB -> BGR', lab_bgr)

# LAB -> BGR -> Greyscale
lab_bgr_grey = cv.cvtColor(lab_bgr, cv.COLOR_BGR2GRAY)
cv.imshow('LAB -> BGR -> Greyscale', lab_bgr_grey)


cv.waitKey(0)
