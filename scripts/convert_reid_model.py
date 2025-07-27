import sys
import torch
from pathlib import Path

# Add the deep-person-reid path to Python's search path
sys.path.append(str(Path(__file__).resolve().parent / "external/deep-person-reid"))

from torchreid.models.osnet import osnet_x0_25

# Paths
ckpt_path = "models/reid/osnet_x0_25_market1501.pth"
output_path = "models/reid/osnet_x0_25_market1501.pt"

# Load raw state_dict
state_dict = torch.load(ckpt_path, map_location='cpu')
print(f"[INFO] Loaded checkpoint keys: {list(state_dict.keys())[:5]} ...")

# Initialize model and load weights
model = osnet_x0_25(num_classes=751)
model.load_state_dict(state_dict)
model.eval()

# Convert to TorchScript
example = torch.randn(1, 3, 256, 128)
traced = torch.jit.trace(model, example)
traced.save(output_path)

print(f"✅ TorchScript model saved to: {output_path}")
