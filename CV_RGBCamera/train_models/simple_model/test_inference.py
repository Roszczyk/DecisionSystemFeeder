from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "best.pt"
IMAGE_PATH = Path(__file__).parent / 'test_image_squirrel_gorky.png'

model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)

annotated_frame = results[0].plot()
annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(annotated_frame_rgb)
plt.axis('off')
plt.show()