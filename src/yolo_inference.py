# src/yolo_inference.py
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOInference:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def predict(self, image_path):
        """
        Returns YOLO results, including bboxes, confidences, etc.
        Also returns an annotated image array.
        """
        results = self.model.predict(image_path)
        annotated_frame = results[0].plot()
        return results, annotated_frame
