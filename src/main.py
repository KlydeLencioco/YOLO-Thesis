import os
from pathlib import Path

import cv2
import numpy as np

import constants
import utils
from tracker import CentroidTracker
from yolo import YOLO

yolo_kwargs = utils.get_config_kwargs(
    "yolo", (("model", str), ("config", str), ("name", str))
)
tracker_kwargs = utils.get_config_kwargs(
    "centroid_tracker", (("max_life", int), ("max_distance", int))
)

current_path = Path(__file__).parent.absolute()
yolo = YOLO(current_path, **yolo_kwargs)
tracker = CentroidTracker(**tracker_kwargs)

classes = yolo.get_classes()

vid_path = os.path.join(current_path, "data", "1.mp4")
cap = cv2.VideoCapture(vid_path)

people_flag = {}
people_count = 0

main_config = utils.config.get("main", "main", fallback={})
detection_threshold = utils.get_config("main", "detection_threshold", float, 0.5)
nms_threshold = utils.get_config("main", "nms_threshold", float, 0.2)
notification_log = utils.get_config(
    "main", "notification_log", str, "/tmp/pc_notification.log"
)
max_capacity = utils.get_config("main", "max_capacity", int, 10)

while cap.isOpened():
    ret, img = cap.read()

    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    marker = int(height / 2)

    outs = yolo.detect_objects(img)

    boxes = []
    centroids = []
    confidences = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] == constants.person:
                if confidence > detection_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append((x, y, w, h))
                    centroids.append((center_x, center_y))
                    confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, detection_threshold, nms_threshold)
    centroids_filtered = [centroids[i[0]] for i in indexes]

    tracks = tracker.track(centroids_filtered)
    for deleted in tracker.deleted:
        people_flag.pop(deleted, None)

    is_people_count_changed = False

    for object_id, centroid in tracks.items():
        latest = centroid[-1]
        flag = 0 if latest[1] < marker else 1

        if object_id not in people_flag:
            people_flag[object_id] = flag
        else:
            if people_flag[object_id] != flag:
                if flag:
                    people_count += 1
                else:
                    people_count -= 1

                is_people_count_changed = True
                people_flag[object_id] = flag
                people_flag.pop(object_id)

        cv2.circle(img, (latest[0], latest[1]), 4, (255, 255, 255), -1)
        cv2.putText(
            img,
            str(object_id),
            (latest[0] - 10, latest[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    cv2.line(img, (0, marker), (int(width), marker), (255, 255, 255), 3)
    cv2.putText(
        img,
        f"People: {people_count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )

    if is_people_count_changed:
        notif_log = "1" if people_count >= max_capacity else "0"
        with open(notification_log, "w") as f:
            f.write(notif_log)

    cv2.imshow("Frame", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
