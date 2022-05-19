import cv2 as cv

# Reading images:

# img = cv.imread('photos/bonsai.jpg')
#
# cv.imshow('Bonsai', img)


# Reading videos:
capture = cv.VideoCapture('videos/bonsai.mp4')

while True:
    isTrue, frame = capture.read()  # Reads video frame-by-frame
    cv.imshow('video', frame)  # Display each frame

    if cv.waitKey(20) & 0xFF == ord('d'):  # Break out of while loop if letter d is pressed.
        break

capture.release()
cv.destroyAllWindows()


