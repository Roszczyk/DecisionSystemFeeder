import cv2
import numpy as np
from pathlib import Path
import json

with open(Path(__file__).parent / "config.json") as f:
    config = json.load(f)

cap = cv2.VideoCapture(config["webcam_no"])

i = 0
if not cap.isOpened():
    print(f"Camera (number {config['webcam_no']}) unavailable")
    exit()

while True:
    ret, frame = cap.read()  

    cv2.imshow('RGB View', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()