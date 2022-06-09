import cv2.cv2 as cv

# Rescale image:
img = cv.imread('photos/bonsai.jpg')


def rescaleFrame(frame, scale=0.50):
    # Rescale size by 50%
    # Images, Videos and Live Video(eg. webcams)

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[1] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


resized_img = rescaleFrame(img)
cv.imshow('rescaled_img', resized_img)


def changeRes(width, height):
    # Live Video (External camera, webcam, etc. Recording live.)

    capture.set(3, width)  # Numbers are Class references for width(3) and height(4)
    capture.set(4, height)


# Rescale video:
capture = cv.VideoCapture('videos/bonsai.mp4')

while True:
    isTrue, frame = capture.read()  # Reads video frame-by-frame

    frame_resized = rescaleFrame(frame)

    cv.imshow('video', frame)  # Display each frame
    cv.imshow('rescale_video', frame_resized)  # Resized video

    if cv.waitKey(20) & 0xFF == ord('d'):  # Break out of while loop if letter d is pressed.
        break

capture.release()
cv.destroyAllWindows()
