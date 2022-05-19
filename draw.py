import cv2 as cv
import numpy as np


# img = cv.imread('photos/bonsai.jpg')
# cv.imshow('bonsai', img)

# Create new blank image using numpy
blank = np.zeros((500, 500, 3),  # Width, height, No. of colour channels
                 dtype='uint8')
cv.imshow('blank', blank)

# 1. Paint the image a certain colour
# Range of pixels
blank[200:300, 300:400] = 0, 255, 0  # Green
cv.imshow('green', blank)

# 2. Draw a rectangle
cv.rectangle(blank, (0, 0), (blank.shape[1] // 2, blank.shape[0] // 2), (0, 255, 0),
             thickness=-1)  # Scaled rectangle based off original image.
cv.imshow('rectangle', blank)

# 3. Draw a circle
cv.circle(blank, (blank.shape[1] // 2, blank.shape[0] // 2), 40, (0, 0, 255),
          thickness=-1)  # center of 250, 250 and radius of 40px
cv.imshow('circle', blank)

# 4. Draw a line
cv.line(blank, (100, 250), (300, 400), (255, 255, 255),
        thickness=3)
cv.imshow('line', blank)

# 5. Write text on an image
cv.putText(blank, 'Hello', (225, 425), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv.imshow('text', blank)


cv.waitKey(0)
