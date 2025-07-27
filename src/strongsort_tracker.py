
from boxmot.trackers.strongsort.strongsort import StrongSort
from src.config import DEVICE, REID_MODEL_REGISTRY, DEFAULTS

# ✅ Load the default ReID model from config
REID_CKPT = REID_MODEL_REGISTRY["osnet_x0_25"]

def load_strongsort():
    tracker = StrongSort(
        reid_weights=REID_CKPT,
        device=DEVICE,
        half=False,               # MPS doesn't support float16
        per_class=False,
        min_conf=0.1,
        max_cos_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.98,
        ema_alpha=0.9
    )
    return tracker