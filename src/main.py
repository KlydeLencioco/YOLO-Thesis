import os
from pathlib import Path

import cv2
import numpy as np

from common import constants
from yolo import YOLO

current_path = Path(__file__).parent.absolute()
yolo = YOLO(current_path)

classes = yolo.get_classes()

vid_path = os.path.join(current_path, "data", "1.mp4")
cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    ret, img = cap.read()

    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    outs = yolo.detect_objects(img)

    boxes = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == constants.person:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    person_count = 0
    for index in indexes:
        x, y, w, h = boxes[index[0]]
        cv2.rectangle(img, (x, y), (x + w, y + h), constants.white, 1)
        person_count = person_count + 1

    print(person_count)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
