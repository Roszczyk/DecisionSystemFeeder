# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from model_api.models import Model
from model_api.visualizer import Visualizer

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.xml"
IMAGE_PATH = HERE / "image.jpg"
OUTPUT_PATH = HERE / "result.jpg"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Sample image not found:{IMAGE_PATH}")


def load_model() -> Model:
    print(f"Loading model from {MODEL_PATH}...")
    return Model.create_model(str(MODEL_PATH))


def load_image() -> cv2.Mat:
    print(f"Loading image from {IMAGE_PATH}...")
    # IMREAD_UNCHANGED preserves the original bit depth (e.g. 16-bit PNG/TIFF images).
    image_raw = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_UNCHANGED)
    if image_raw is None:
        raise RuntimeError(f"Failed to decode image: {IMAGE_PATH}")

    # Add explicit channel dimension for 2D grayscale: (H, W) -> (H, W, 1)
    if image_raw.ndim == 2:
        image_raw = image_raw[..., np.newaxis]

    # Convert BGR to RGB for standard 3-channel images
    if image_raw.ndim == 3 and image_raw.shape[2] == 3:
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)

    return image_raw


def visualise_result(image, result) -> None:
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    Visualizer().show(image, result)
    
    display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = Visualizer().render(display_image, result)
    cv2.imwrite(str(OUTPUT_PATH), output)
    print(f"Saved annotated result to {OUTPUT_PATH}")

