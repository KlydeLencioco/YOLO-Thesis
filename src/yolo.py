import os

import cv2


class YOLO:
    def __init__(
        self,
        path: str,
        model: str = "yolov3-tiny.weights",
        config: str = "yolov3-tiny.cfg",
        names: str = "coco.names",
    ):
        yolo_dir = "yolo"

        self.model_path = os.path.join(path, yolo_dir, model)
        self.config_path = os.path.join(path, yolo_dir, config)
        self.class_path = os.path.join(path, yolo_dir, names)

        self._net = None
        self._layers = None
        self.create_net()

    @property
    def net(self):
        return self._net

    def create_net(self):
        self._net = cv2.dnn.readNet(self.model_path, self.config_path)

        layers = self.net.getLayerNames()
        self._layers = [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def get_classes(self):
        with open(self.class_path) as f:
            return [line.strip() for line in f.readlines()]

    def detect_objects(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (320, 320), (0, 0, 0), True, crop=False
        )
        self._net.setInput(blob)
        return self._net.forward(self._layers)
