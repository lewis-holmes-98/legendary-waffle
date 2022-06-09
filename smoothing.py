import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""

Different blurring techniques

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
cv.imshow('Original Image No Blur', img)

# Averaging
# Define kernel window over portion of image
# Window will compute the pixel intensity of middle pixel (true center) from average surround pixel intensity
average = cv.blur(img, (3, 3))  # Higher specified kernel size = more blur
cv.imshow('Average Blur', average)

# Gaussian Blur
# Similar method to average, instead of averaging pixel intensity, pixels are given a weight
# Average of weights used for true center value
# Method can result in less blurring, but more natural
gauss = cv.GaussianBlur(img, (3, 3), 0)  # 'sigmaX' = Standard deviation in x direction
cv.imshow('Gaussian Blur', gauss)

# Median Blur
# Almost same as averaging, instead of finding average or surrounding pixels, finds median
# Median blurring can be more effective in reducing noise compared to averaging and Gaussian
# Choice for advanced CV projects that cant have noise
# Not intended for high kernel blurring sizes ( 5 + )
median = cv.medianBlur(img, 3)  # Kernel size not tuple, just an Int - openCV assumes a 3 x 3 value
cv.imshow('Median Blur', median)

# Bilateral Blurring
# Most effective blurring technique
# Applies blurring while retaining edges in the image
bilateral = cv.bilateralFilter(img, 10, 35, 25)  # 'Diameter' not 'Kernel size' for int value.
# Sigma colour = how many colours to be considered when the blur is applied.
# Space sigma = Distance of pixels that influence blurring calculation
cv.imshow('Bilateral Blur', bilateral)

cv.waitKey(0)
