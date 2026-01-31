from ultralytics import YOLO
 
class Detector:
    def __init__(self, path_weights):
        self.model = YOLO(path_weights)

    def detect_objects(self, img, conf, iou):
        results = self.model(img, conf=conf, iou=iou)
        return results
