import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import os
from pathlib import Path
from time import sleep
import copy

from layer1_utils import take_frame, save_frame, get_camera_config, record_and_save_video
from thirdparty.geti.utils import load_model, visualise_result

SAVE_DIR = Path(__file__).parent / "birds"
COOLDOWN = 90
SLEEP_TIME = 30
CONF_THRESHOLD = 0.5
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

IR_MODEL_PATH = Path(__file__).parent / "CV_InfraredCamera/Infrared_Bird_Detection/ir_birds_detection_ver1.xml"

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

mode = None
while True:
    if mode != "night" and (datetime.now().hour >= 19 or datetime.now().hour <= 6):
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mode = "night"
        CONF_MATRIX_FILE = SAVE_DIR / f"ir_confusion_matrix_{start_time}_night.txt"
        ir_confusion_matrix = {
            "rgb 1 ir 0" : 0,
            "rgb 0 ir 1" : 0,
            "rgb 1 ir 1" : 0
        }
    elif mode != "day" and (datetime.now().hour < 19 and datetime.now().hour > 6):
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        mode = "day"
        CONF_MATRIX_FILE = SAVE_DIR / f"ir_confusion_matrix_{start_time}_day.txt"
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

    for i in range(len(ir_results.label_names)):
        if ir_results.label_names[i] == "Bird" and ir_results.scores[i] > CONF_THRESHOLD:
            ir_detected = True
            break

    now = time.time()

    if (ir_detected or rgb_detected) and (now - last_photo_time) > COOLDOWN:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if ir_detected and rgb_detected:
            ir_confusion_matrix["rgb 1 ir 1"] += 1
        if not ir_detected and rgb_detected:
            ir_confusion_matrix["rgb 1 ir 0"] += 1
        if ir_detected and not rgb_detected:
            ir_confusion_matrix["rgb 0 ir 1"] += 1

        img_path = f"{SAVE_DIR}/bird_{timestamp}.jpg"
        bb_img_path = f"{SAVE_DIR}/bird_{timestamp}_bb.jpg"
        img_rgb2_path = f"{SAVE_DIR}/bird_{timestamp}_RGB2.jpg"
        ir_path = f"{SAVE_DIR}/bird_{timestamp}_ir.jpg"
        bb_ir_path = f"{SAVE_DIR}/bird_{timestamp}_irbb.jpg"

        # =============================================================
        # RECORDING: to be deleted in future, just wanted some mp4 data
        rgb1_recording_path = f"{SAVE_DIR}/recording_{timestamp}_rgb1.mp4"
        rgb2_recording_path = f"{SAVE_DIR}/recording_{timestamp}_rgb2.mp4"
        ir_recording_path = f"{SAVE_DIR}/recording_{timestamp}_rgb2.mp4"
        if rgb_detected:
            record_and_save_video(CAMERA_RGB, rgb1_recording_path, duration=3, rotate=True, verbose=True)
            record_and_save_video(CAMERA_RGB_2, rgb2_recording_path, duration=3, rotate=False, verbose=True)
            record_and_save_video(CAMERA_IR, ir_recording_path, duration=3, ir_ir=True, verbose=True)
        # =============================================================

        save_frame(img_path, frame, verbose=True)
        save_frame(ir_path, frame_ir, verbose=True)  
        if ir_detected:
            bb_ir_frame = visualise_result(frame_ir, ir_results)
            save_frame(bb_ir_path, bb_ir_frame, verbose=True)

        save_frame(bb_img_path, frame_copy, verbose=True)

        # second RGB camera save:
        if CAMERA_RGB_2 != -1:
            rgb2_photo = take_frame(CAMERA_RGB_2)
            if rgb2_photo is not None:
                save_frame(img_rgb2_path, rgb2_photo, verbose=True)

        confusion_matrix_text = f"\t RGB 1 \t RGB 0 \n IR 1 \t {ir_confusion_matrix["rgb 1 ir 1"]} \t {ir_confusion_matrix["rgb 0 ir 1"]} \n IR 0 \t {ir_confusion_matrix["rgb 1 ir 0"]} \t N/A\n"

        print(confusion_matrix_text)
        with open(CONF_MATRIX_FILE, "w") as f:
            f.write(confusion_matrix_text)
            f.write(f"\nStart time: {start_time}")
            f.write(f"\nTimestamp: {timestamp}")

        last_photo_time = now
    sleep(SLEEP_TIME)
