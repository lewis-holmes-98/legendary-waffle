import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""

Four basic bitwise operators:
AND
OR
XOR
NOT

Very common in image processing, especially when working with masks
Bitwise operators operate in a binary manor, pixels are turned off at a value of '0'
and turned of at a value of '1'.

XOR - OR = AND
AND - OR = XOR

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
# cv.imshow('Original Image', img)

# Create blank var as base to draw onto
blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# Bitwise AND --> intersecting regions
bitwise_and = cv.bitwise_and(rectangle, circle)
cv.imshow('Bitwise AND', bitwise_and)

# Bitwise OR --> intersecting and non-intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle)
cv.imshow('Bitwise OR', bitwise_or)

# Bitwise XOR --> non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)
cv.imshow('Bitwise XOR', bitwise_xor)

# Bitwise NOT --> inverts binary value
bitwise_not_r = cv.bitwise_not(rectangle)
bitwise_not_c = cv.bitwise_not(circle)
cv.imshow('Bitwise NOT (Rectangle)', bitwise_not_r)
cv.imshow('Bitwise NOT (Circle)', bitwise_not_c)

cv.waitKey(0)
