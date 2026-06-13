from ultralytics import YOLO

best_weights_path = r"runs\detect\runs\detect\enterprise_analytics_model\weights\best.pt"
trained_model = YOLO(best_weights_path)
trained_model.export(format="onnx")