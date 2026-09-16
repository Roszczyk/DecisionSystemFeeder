import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import os
from pathlib import Path
import json
from time import sleep
import copy

from CV_InfraredCamera.infrared_utils import process_ir_frame
from thirdparty.geti.utils import load_model, visualise_result

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
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

IR_MODEL_PATH = Path(__file__).parent / "CV_InfraredCamera/Infrared_Bird_Detection/ir_birds_detection_ver1.xml"
CONF_MATRIX_FILE = Path(__file__).parent / f"ir_confusion_matrix_{start_time}.txt"

config = get_camera_config(Path(__file__).parent / "config.json")
CAMERA_RGB = config["RGBCAM"]
CAMERA_IR = config["IRCAM"]
CAMERA_RGB_2 = config["RGBCAM2"]

os.makedirs(SAVE_DIR, exist_ok=True)

model_rgb = YOLO("yolov8n.pt")
model_ir = load_model(IR_MODEL_PATH)

last_photo_time = 0

print("Bird detection started...")
print("Timestamp: ", start_time)

ir_confusion_matrix = {
    "rgb 1 ir 0" : 0,
    "rgb 0 ir 1" : 0,
    "rgb 1 ir 1" : 0
}

rgb_detected = False
ir_detected = False

frame = take_frame(CAMERA_RGB, rotate=True)
frame_copy = copy.deepcopy(frame)

frame_ir = take_frame(CAMERA_IR, is_ir=True)

results = model_rgb(frame, verbose=False)[0]
ir_results = model_ir(frame_ir)

bird_boxes = []

for box in results.boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    name = model_rgb.names[cls]

    if (name == "bird") and conf > CONF_THRESHOLD:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bird_boxes.append((name, conf, x1, y1, x2, y2))

        cv2.rectangle(frame_copy, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame_copy, f"{name} {conf:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

if len(bird_boxes) > 0:
    rgb_detected = True

now = time.time()

if 'Bird' in ir_results.label_names:
    ir_detected = True

if ir_detected and rgb_detected:
    ir_confusion_matrix["rgb 1 ir 1"] += 1
if not ir_detected and rgb_detected:
    ir_confusion_matrix["rgb 1 ir 0"] += 1
if ir_detected and not rgb_detected:
    ir_confusion_matrix["rgb 0 ir 1"] += 1

if (ir_detected or rgb_detected) and (now - last_photo_time) > COOLDOWN:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    img_path = f"{SAVE_DIR}/bird_{timestamp}.jpg"
    txt_path = f"{SAVE_DIR}/birds_{timestamp}.txt"
    bb_img_path = f"{SAVE_DIR}/bird_{timestamp}_bb.jpg"
    img_rgb2_path = f"{SAVE_DIR}/bird_{timestamp}_RGB2.jpg"
    ir_path = f"{SAVE_DIR}/bird_{timestamp}_ir.jpg"
    bb_ir_path = f"{SAVE_DIR}/bird_{timestamp}_irbb.jpg"

    cv2.imwrite(img_path, frame)
    cv2.imwrite(ir_path, frame_ir)  
    if ir_detected:
        bb_ir_frame = visualise_result(frame_ir)
        cv2.imwrite(bb_ir_path, bb_ir_frame)
    if rgb_detected:
        cv2.imwrite(bb_img_path, frame_copy)

    # second RGB camera save:
    if CAMERA_RGB_2 != -1:
        rgb2_photo = take_frame(CAMERA_RGB_2)
        if rgb2_photo is not None:
            cv2.imwrite(img_rgb2_path, rgb2_photo)

    with open(txt_path, "w") as f:
        for name, conf, x1, y1, x2, y2 in bird_boxes:
            f.write(f"{name} {conf:.3f} {x1} {y1} {x2} {y2}\n")

    print(f"📸 Saved: {img_path}")
    print(f"📦 Boxes: {txt_path}")
    print(f"📸 Saved IR: {ir_path}")
    print(f"📸 Saved with boxes: {bb_img_path}")

    confusion_matrix_text = f"\t RGB 1 \t RGB 0 \n IR 1 \t {ir_confusion_matrix["rgb 1 ir 1"]} \t {ir_confusion_matrix["rgb 0 ir 1"]} \n IR 0 \t {ir_confusion_matrix["rgb 1 ir 0"]} \t N/A"

    print(confusion_matrix_text)
    with open(CONF_MATRIX_FILE, "w") as f:
        f.write(confusion_matrix_text)
        f.write(f"Start time: {start_time}")
        f.write(f"Timestamp: {timestamp}")

    last_photo_time = now

cv2.destroyAllWindows()
