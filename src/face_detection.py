import cv2
import numpy as np

cfg_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
names_path = 'coco.names'

net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

videoPath = "../Video-Photo Data/Classroom.mov"

cap = cv2.VideoCapture(videoPath)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(videoPath)

if not videoCapture.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = videoCapture.read()

    if not ret:
        print("Reached the end of the video or cannot read the frame.")
        break

    grayVideo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(grayVideo, scaleFactor=1.05, minNeighbors=4, minSize=(25, 25))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()