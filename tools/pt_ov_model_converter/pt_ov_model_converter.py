import torch
import openvino as ov
from pathlib import Path
import argparse

def handle_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--model", type=Path, help="path to pytorch model")
    parser.add_argument("--arch", type=str, help="model architecture, currently supported: YOLO", choices=["YOLO"], default="YOLO")
    return parser.parse_args()

if __name__ == "__main__":
    args = handle_args()
    base_dir = Path(__file__).parent

    if args.model == None:
        torch_model = base_dir / "model.pt"
        if not torch_model.exists():
            print("ERROR: No model provided")
            exit(1)
    else:
        torch_model = args.model
        
    architecture = args.arch
    
    if architecture == "YOLO":
        from ultralytics import YOLO
        model = YOLO(torch_model)
        model.export(format="openvino")