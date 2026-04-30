import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import os
from pathlib import Path
import json
from time import sleep
import copy

from processing import process_ir_frame

def get_camera_config(config_file):
    with open(config_file) as f:
        return json.load(f)

def take_frame(cam_no, is_ir=False, rotate=False):
    cap = cv2.VideoCapture(cam_no)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    ret, frame = cap.read()
    if not ret:
        return
    cap.release()
    if is_ir:
        frame = process_ir_frame(frame)
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame

SAVE_DIR = Path(__file__).parent / "birds"
COOLDOWN = 60
CONF_THRESHOLD = 0.5

config = get_camera_config(Path(__file__).parent / "config.json")
CAMERA_RGB = config["RGBCAM"]
CAMERA_IR = config["IRCAM"]
CAMERA_RGB_2 = config["RGBCAM2"]

os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

last_photo_time = 0

print("Bird detection started...")

while True:
    frame = take_frame(CAMERA_RGB, rotate=True)
    frame_copy = copy.deepcopy(frame)

    results = model(frame, verbose=False)[0]

    bird_boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        if (name == "bird" or name == "person") and conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bird_boxes.append((name, conf, x1, y1, x2, y2))

            cv2.rectangle(frame_copy, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame_copy, f"{name} {conf:.2f}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    now = time.time()

    if len(bird_boxes) > 0 and (now - last_photo_time) > COOLDOWN:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        img_path = f"{SAVE_DIR}/{name}_{timestamp}.jpg"
        txt_path = f"{SAVE_DIR}/{name}_{timestamp}.txt"
        bb_img_path = f"{SAVE_DIR}/{name}_{timestamp}_bb.jpg"
        img_rgb2_path = f"{SAVE_DIR}/{name}_{timestamp}_RGB2.txt"

        ir_photo = take_frame(CAMERA_IR, is_ir = True)
        ir_path = f"{SAVE_DIR}/{name}_{timestamp}_ir.jpg"

        cv2.imwrite(img_path, frame)
        if ir_photo is not None:
            cv2.imwrite(ir_path, ir_photo)
        cv2.imwrite(bb_img_path, frame_copy)

        # second RGB camera save:
        if CAMERA_RGB_2 != -1:
            rgb2_photo = take_frame(CAMERA_RGB_2)
            if rgb2_photo is not None:
                cv2.imwrite(img_rgb2_path, rgb_photo)

        with open(txt_path, "w") as f:
            for name, conf, x1, y1, x2, y2 in bird_boxes:
                f.write(f"{name} {conf:.3f} {x1} {y1} {x2} {y2}\n")

        print(f"📸 Saved: {img_path}")
        print(f"📦 Boxes: {txt_path}")
        print(f"📸 Saved IR: {ir_path}")
        print(f"📸 Saved with boxes: {bb_img_path}")

        last_photo_time = now

    if cv2.waitKey(1) == 27:
        break

    sleep(15)

cv2.destroyAllWindows()
