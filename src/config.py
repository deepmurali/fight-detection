from pathlib import Path
import torch

# Base paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

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
    "yolov8n": {
        "type": "detector",
        "source": "ultralytics",
        "backbone": "yolov8n",
        "weight_path": MODEL_DIR / "yolov8n.pt",
    },
    "timesformer": {
        "type": "video",
        "source": "huggingface",
        "backbone": "facebook/timesformer-base-finetuned-k400",
        "input_shape": (3, CLIP_LEN, FRAME_HEIGHT, FRAME_WIDTH),
        "weight_path": None,  # loaded from HF
    },
}

# Default model
DEFAULT_MODEL = "mobilenet_v3_large"
