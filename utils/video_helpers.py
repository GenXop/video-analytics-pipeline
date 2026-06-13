import cv2
import numpy as np

def draw_inference_overlays(frame: np.ndarray, tracked_objects: list, count: int, class_names: dict):
    cv2.line(frame, (0, 240), (640, 240), (0, 255, 255), 3)
    cv2.putText(frame, f"Cross Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id, cls_id = obj
        label = f"{class_names.get(cls_id, 'Object')} #{track_id}"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return frame