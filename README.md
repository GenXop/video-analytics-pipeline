# Video Analytics Pipeline

A real-time multi-object detection, tracking, and spatial analytics pipeline built with YOLOv8 and Streamlit. Detects and tracks vehicles and pedestrians in video streams, and counts objects crossing a user-defined spatial boundary line.

## Features

- Real-time object detection using YOLOv8 (nano variant, CPU-optimized via ONNX)
- Centroid-based multi-object tracking with persistent track IDs across frames
- Geometric line-crossing analytics using Shapely — counts unique objects that cross a boundary
- Streamlit web UI with live annotated video playback
- Configurable detection confidence threshold and per-class filtering
- Trained on the VisDrone2019 benchmark dataset (drone-view vehicle/pedestrian detection)

## Tech Stack

| Layer | Library |
|---|---|
| Web UI | Streamlit |
| Object Detection | Ultralytics YOLOv8 |
| Inference Runtime | ONNX Runtime |
| Computer Vision | OpenCV, NumPy |
| Geometry | Shapely |
| Dataset Management | Roboflow |

## Project Structure

```
video-analytics-pipeline/
├── app.py                  # Streamlit application entry point
├── train.py                # YOLOv8 training script (VisDrone dataset)
├── convert.py              # Export PyTorch weights to ONNX
├── requirements.txt
├── core/
│   ├── detector.py         # VideoDetector — loads model, runs inference
│   ├── tracker.py          # IdentityTracker — centroid-based tracking
│   └── analytics.py        # SpatialAnalyticsEngine — line-crossing counts
├── utils/
│   └── video_helpers.py    # Bounding box / overlay drawing helpers
└── archive/
    ├── VisDrone.yaml        # Dataset config and class definitions
    └── VisDrone2019-*/      # Training, validation, and test image sets
```

## Detection Classes

The model detects 10 classes from the VisDrone dataset:

`pedestrian` · `people` · `bicycle` · `car` · `van` · `truck` · `tricycle` · `awning-tricycle` · `bus` · `motor`

## Setup

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/your-username/video-analytics-pipeline.git
cd video-analytics-pipeline
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

Upload an MP4, AVI, or MOV video file through the browser UI. Adjust the confidence threshold slider and select which classes to detect. The annotated stream shows bounding boxes, track IDs, and a running line-crossing count.

## Training

To retrain on VisDrone2019:

1. Place the dataset under `archive/` following the existing directory structure.
2. Run training:

```bash
python train.py
```

This trains YOLOv8n for 10 epochs at 640×640 and exports weights to ONNX under `runs/`.

To convert an existing `.pt` checkpoint to ONNX:

```bash
python convert.py
```

## Model

The pipeline ships with `yolov8n.pt` (YOLOv8 nano, ~6.5 MB). ONNX export is used at inference time for faster CPU throughput via `onnxruntime`.

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) — a large-scale benchmark collected by drones over various urban scenes, providing 6,471 training images, 548 validation images, and 1,610 test images.
