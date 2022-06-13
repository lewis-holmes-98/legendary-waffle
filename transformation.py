import cv2.cv2 as cv
import numpy as np

# Reading photos:

img = cv.imread('photos/bonsai.jpg')


def rescaleFrame(frame, scale=0.40):
    # Rescale size by 50%
    # Images, Videos and Live Video(eg. webcams)

    width = int(frame.shape[1] * scale // 0.8)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


rescaled_img = rescaleFrame(img)
cv.imshow('Original', rescaled_img)


# Translation - Moving img along x or y axis
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)


# -y --> Up
# y --> Down
# -x --> Left
# x --> Right

translated = translate(rescaled_img, 100, 100)  # Shift 100px down, 100px right
cv.imshow('Translated', translated)


# Rotation
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


rotated = rotate(rescaled_img, -90)  # Negative values for clockwise
cv.imshow('Rotated', rotated)

# Rotate an already rotated image - can cause additional black space.
rotated_rotated = rotate(rotated, -90)
cv.imshow('Rotated x 2', rotated_rotated)

# Resizing
resized = cv.resize(rescaled_img, (300, 300), interpolation=cv.INTER_AREA)
cv.imshow('Resized', resized)

# Flipping
flip = cv.flip(rescaled_img, 0)  # 0 - flip vertically, 1 - flip horizontally, -1 - flip both
cv.imshow('Flipped', flip)

cv.waitKey(0)
