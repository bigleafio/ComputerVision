import cv2
import os
import numpy as np
from imutils.object_detection import non_max_suppression

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_alt.xml".format(
    base_path=os.path.abspath(os.path.dirname(__file__)))

imagePath = 'images/kids.png'

# load the image and convert
image = cv2.imread(imagePath)


# convert the image to grayscale, load the face cascade detector,
# and detect faces in the image
cimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
rects = detector.detectMultiScale(cimage, scaleFactor=1.1, minNeighbors=5,
                                  minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

# draw the final bounding boxes
for (xA, yA, xB, yB) in rects:
    cv2.rectangle(cimage, (xA, yA), (xB, yB), (0, 255, 0), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(cimage, (xA, yA), (xB, yB), (0, 255, 0), 2)

cv2.imshow("After NMS", cimage)
cv2.waitKey(0)

