import torch
import openvino as ov
from pathlib import Path

base_dir = Path(__file__).parent
torch_model = base_dir / "model.pt"
openvino_model = base_dir / "model_ov"