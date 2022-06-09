import cv2.cv2 as cv

# Reading images:

img = cv.imread('photos/bonsai.jpg')


def rescaleFrame(frame, scale=0.40):
    # Rescale size by 50%
    # Images, Videos and Live Video(eg. webcams)

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


rescaled_img = rescaleFrame(img)
cv.imshow('Original', rescaled_img)

# Converting to greyscale
grey = cv.cvtColor(rescaled_img, cv.COLOR_BGR2GRAY)
cv.imshow('Grey', grey)

# Blur
blur = cv.GaussianBlur(rescaled_img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blurred', blur)

# Edge Cascade - find edges that are present in image
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges - Blurred', canny)

# Dilate image with specific structuring element
dilated = cv.dilate(canny, (3, 3), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3, 3), iterations=3)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (600, 450), interpolation=cv.INTER_AREA)  # For scaling UP use, INTER_LINEAR or INTER_CUBIC
cv.imshow('resized', resized)

# Cropping using array slicing
cropped = img[50: 200, 200: 400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)
