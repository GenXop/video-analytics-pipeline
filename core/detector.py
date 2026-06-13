import numpy as np
from ultralytics import YOLO

class VideoDetector:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        
    def detect_frames(self, frame: np.ndarray, confidence: float = 0.25, classes: list = None):
        results = self.model.predict(source=frame, conf=confidence, classes=classes, verbose=False)
        if not results:
            return []
        detections = results[0].boxes.data.cpu().numpy()
        return detections