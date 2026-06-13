import os
import logging
from ultralytics import YOLO

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "training.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Enterprise Video Analytics Training Pipeline...")
    
    try:
        logger.info("Loading base model (yolov8n.pt)...")
        model = YOLO("yolov8n.pt") 
        logger.info("Base model loaded successfully.")

        dataset_path = os.path.join("archive", "VisDrone.yaml")
        epochs = 10
        img_size = 640
        project_dir = os.path.join("runs", "detect")
        run_name = "enterprise_analytics_model"

        results = model.train(
            data=dataset_path, 
            epochs=epochs, 
            imgsz=img_size,
            device=0,
            workers = 0,
            project=project_dir,
            name=run_name,
            exist_ok=True 
        )

        best_weights_path = os.path.join(project_dir, run_name, "weights", "best.pt")
        logger.info(f"Training completed. Best weights saved to: {best_weights_path}")

        logger.info("Exporting model to ONNX format for faster CPU inference...")
        trained_model = YOLO(best_weights_path)
        onnx_path = trained_model.export(format="onnx")
        logger.info(f"ONNX export complete. Saved to: {onnx_path}")

    except Exception as e:
        logger.error("An error occurred during training.", exc_info=True)

if __name__ == "__main__":
    main()