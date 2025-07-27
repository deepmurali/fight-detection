from pathlib import Path
import torch

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

# YOLO model paths
YOLO_DIR = MODEL_DIR / "yolo"
MMDET_DIR = MODEL_DIR / "mm"
MMDET_CONFIGS = MMDET_DIR / "configs"
MMDET_WEIGHTS = MMDET_DIR / "pth"

# Runtime device selection
DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Frame extraction settings
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
CLIP_LEN = 16        # Frames per clip
FPS = 15             # Preprocess video to this FPS
STRIDE = 4           # How often to sample clips from video

# Model registry
MODEL_REGISTRY = {
    "mobilenet_v3_small": {
        "type": "image",
        "source": "torchvision",
        "backbone": "mobilenet_v3_small",
        "input_shape": (3, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": MODEL_DIR / "mobilenet_v3_small.pth",
    },
    "mobilenet_v3_large": {
        "type": "image",
        "source": "torchvision",
        "backbone": "mobilenet_v3_large",
        "input_shape": (3, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": MODEL_DIR / "mobilenet_v3_large.pth",
    },
    "mvitv2_base": {
        "type": "video",
        "source": "pytorchvideo",
        "backbone": "mvitv2_b",
        "input_shape": (3, CLIP_LEN, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": MODEL_DIR / "mvitv2_base.pth",
    },
    "movinet_a0": {
        "type": "video",
        "source": "pytorchvideo",
        "backbone": "movinet_a0",
        "input_shape": (3, CLIP_LEN, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": MODEL_DIR / "movinet_a0.pth",
    },
    "timesformer": {
        "type": "video",
        "source": "huggingface",
        "backbone": "facebook/timesformer-base-finetuned-k400",
        "input_shape": (3, CLIP_LEN, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": None,  # loaded from HF
    },

    # YOLOv8 detectors
    "yolov8n": {
        "type": "detector",
        "source": "ultralytics",
        "backbone": "yolov8n",
        "weight_path": YOLO_DIR / "yolov8n.pt",
    },
    "yolov8m": {
        "type": "detector",
        "source": "ultralytics",
        "backbone": "yolov8m",
        "weight_path": YOLO_DIR / "yolov8m.pt",
    },
    "yolov8l": {
        "type": "detector",
        "source": "ultralytics",
        "backbone": "yolov8l",
        "weight_path": YOLO_DIR / "yolov8l.pt",
    },

    # MMDetection Pose models
    "litehrnet_18": {
        "type": "pose",
        "source": "mmdetection",
        "config_path": MMDET_CONFIGS / "td-hm_litehrnet-18_8xb64-210e_coco-256x192.py",
        "weight_path": MMDET_WEIGHTS / "litehrnet18_coco_256x192-6bace359_20211230.pth",
    },
    "hrnet_w32": {
        "type": "pose",
        "source": "mmdetection",
        "config_path": MMDET_CONFIGS / "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        "weight_path": MMDET_WEIGHTS / "td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth",
    },
    "rtmo_s": {
        "type": "pose",
        "source": "mmdetection",
        "config_path": MMDET_CONFIGS / "rtmo-s_8xb32-600e_body7-640x640.py",
        "weight_path": MMDET_WEIGHTS / "rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth",
    },
}

# ReID model (StrongSORT)
REID_MODEL_REGISTRY = {
    "osnet_x0_25": MODEL_DIR / "reid" / "osnet_x0_25_market1501.pth"
}

# Lightweight and heavy model defaults (change these in one place)
DEFAULTS = {
    "yolo_light": "yolov8n",
    "yolo_heavy": "yolov8l",
    "pose_light": "litehrnet_18",
    "pose_heavy": "hrnet_w32",
    "video": "mvitv2_base",
    "image": "mobilenet_v3_large"
}
