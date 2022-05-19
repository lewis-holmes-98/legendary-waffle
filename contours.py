import cv2 as cv
import numpy as np

"""
    Contour Detection - Useful for image recognition
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

# Create blank image with same dimensions for drawing
blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

# Convert img to grey scale
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Greyed', grey)

# Blur
blur = cv.GaussianBlur(grey, (5, 5), cv.BORDER_DEFAULT)
cv.imshow('Grey-Blurred', blur)

# Detect edges using canny edge detector
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# Find contours

"""
    Looks at the structuring element (edges). Returns two values in a python list - location of contours on img.
    hierarchies - hierarchical representation within contours. Eg, square in rectangle in circle.
    
    Return methods:
    RETR_LIST returns all contours in img.
    RETR_EXTERNAL returns all external (outside) contours.
    RETR_TREE returns all contours that are in a hierarchical system.
    
    Contour approximation method:
    CHAIN_APPROX_NONE does nothing; Returns all contours
    CHAIN_APPROX_SIMPLE compresses and simplifies contours. 
"""

# Different function for edge detection - Thresholding
# Converts img to binary - Below 125 = 0 (Black). Above 255 = 1 (White)

ret, thresh = cv.threshold(grey, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found')

# Draw contours on blank image
# Attempt to use canny edge detection first, then thresholding.
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours-Drawn', blank)


cv.waitKey(0)
