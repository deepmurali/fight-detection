# src/detector.py

from ultralytics import YOLO
from src.config import MODEL_REGISTRY, DEFAULTS

def load_yolo_model(model_key: str = None):
    """
    Load the YOLOv8 model specified in config. Defaults to lightweight yolov8n.
    """
    if model_key is None:
        model_key = DEFAULTS["yolo_light"]
    
    model_config = MODEL_REGISTRY[model_key]
    model_path = model_config["weight_path"]
    model = YOLO(str(model_path))
    return model


def run_yolo_inference(model, frame, person_class_id=0):
    """
    Run inference and return only person detections.
    """
    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        class_id = int(box.cls.item())
        if class_id != person_class_id:
            continue  # Only keep "person" class (COCO ID 0)
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf.item())

        detections.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "class_id": class_id
        })

    return detections
