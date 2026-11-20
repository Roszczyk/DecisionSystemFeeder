import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import os
from pathlib import Path
from time import sleep
import copy

from layer1_utils import take_frame, get_camera_config, CV_Det
from thirdparty.geti.utils import load_model

COOLDOWN = 90
SLEEP_TIME = 30

config = get_camera_config(Path(__file__).parent / "config.json")
CAMERA_RGB = config["RGBCAM"]
CAMERA_IR = config["IRCAM"]
CAMERA_RGB_2 = config["RGBCAM2"]

CONF_THRESHOLD_RGB = 0.5
CONF_THRESHOLD_IR = 0.7
CONF_THRESHOLD_RGB2 = 0.5

IR_MODEL_PATH = Path(__file__).parent / "CV_InfraredCamera/Infrared_Bird_Detection/ir_birds_detection_ver1.xml"

model_rgb = YOLO("yolov8n.pt")
model_ir = load_model(IR_MODEL_PATH)

last_photo_time = 0

def detect(camera_number, model, conf_threshold, rotate = False, is_ir = False):
    frame = take_frame(camera_number, rotate=rotate, is_ir=is_ir)
    results = model(frame)  
    count_birds = 0
    conf_list = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model_rgb.names[cls]

        if (name == "bird") and conf > conf_threshold:
            conf_list.append(conf)
            count_birds += 1

    assert len(conf_list) == count_birds, \
        f"Number of confidences ({len(conf_list)}) is not the same as bird count ({count_birds})"

    if count_birds == 0:
        output = CV_Det(False, 0, [], conf_threshold)
    else:
        output = CV_Det(True, count_birds, conf_list, conf_threshold)

    return output

def model_helper(model, frame, type = "RGB"):
    assert type in ["RGB", "IR"]
    if type == "RGB":
        results = model(frame, verbose=False)[0]
    elif type == "IR":
        results = model(frame)
    return results