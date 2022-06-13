import cv2 as cv

img = cv.imread('photos/large_group1.jpg')
cv.imshow('Large group of people', img)

# Convert to grayscale
# Haarcascades only look at shapes in a face, so colour does not matter.
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

# Read xml code and store to variable
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detect the face
# Returns rectangular coordinates to detected face location
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
print(f"Number of faces found: {len(faces_rect)}")

# Get detetcted face coordinates and draw rectangle over image
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)
