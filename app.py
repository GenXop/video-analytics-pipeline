import streamlit as tf
import cv2
import tempfile
import os
from core.detector import VideoDetector
from core.tracker import IdentityTracker
from core.analytics import SpatialAnalyticsEngine
from utils.video_helpers import draw_inference_overlays

tf.set_page_config(page_title="Enterprise Video Analytics Pipeline", layout="wide")
tf.title(" Scalable Video Analytics & Multi-Object Tracking Pipeline")

model_path = r"runs\detect\runs\detect\enterprise_analytics_model\weights\best.onnx"
detector = VideoDetector(model_path=model_path)
tracker = IdentityTracker()
analytics = SpatialAnalyticsEngine(line_coords=[(0, 240), (640, 240)])

VISDRONE_CLASSES = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van', 
    5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor'
}

uploaded_file = tf.file_uploader("Upload Target Video Stream (MP4/AVI)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    
    video_capture = cv2.VideoCapture(tfile.name)
    st_frame = tf.empty()
    
    tf.sidebar.header("Pipeline Configurations")
    conf_threshold = tf.sidebar.slider("Detector Confidence Score", 0.1, 1.0, 0.35)
    
    selected_classes = tf.sidebar.multiselect(
        "Target Detection Filters", 
        list(VISDRONE_CLASSES.values()), 
        default=['car', 'bus', 'pedestrian', 'van', 'truck']
    )
    
    class_indices = [k for k, v in VISDRONE_CLASSES.items() if v in selected_classes]

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 480))
        
        detections = detector.detect_frames(frame, confidence=conf_threshold, classes=class_indices)
        tracked_outputs = tracker.update(detections)
        
        current_count = 0
        for obj in tracked_outputs:
            x1, y1, x2, y2, t_id, cls_id = obj
            current_count = analytics.check_line_crossing(t_id, [x1, y1, x2, y2])
            
        annotated_frame = draw_inference_overlays(frame, tracked_outputs, current_count, VISDRONE_CLASSES)
        
        st_frame.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        
    video_capture.release()
    tfile.close()
    os.unlink(tfile.name)
    tf.success("Processing complete.")