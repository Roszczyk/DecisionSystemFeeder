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
from mqtt_publisher import MQTT_Publisher

COOLDOWN = 120
SLEEP_TIME_BETWEEN_CAMERAS = 5
SLEEP_TIME_BETWEEN_ITERATIONS = 10

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
    if is_ir:
        for i in range(len(results.label_names)):
            if results.label_names[i] == "Bird" and results.scores[i] > conf_threshold:
                count_birds += 1
                conf_list.append(results.scores[i])
    else:
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

def check_if_publish(previous_state, current_state : CV_Det, cooldown = COOLDOWN):
    if previous_state[0] == None or previous_state[0] == True:
        return True
    if previous_state[0] != current_state.is_detected:
        return True
    if (datetime.now() - previous_state[1]).total_seconds() > cooldown:
        return True
    return False

if __name__ == "__main__":
    mqtt = MQTT_Publisher("0.0.0.0", 1883, "", "")
    previous_state_rgb_1 = [None, datetime.now()]
    previous_state_rgb_2 = [None, datetime.now()]
    previous_state_ir = [None, datetime.now()]
    while True:
        try:
            results_rgb_1 = detect(CAMERA_RGB, lambda x: model_rgb(x, verbose=False)[0], 
                                CONF_THRESHOLD_RGB, True, False)
            if check_if_publish(previous_state_rgb_1, results_rgb_1):
                mqtt.publish("Feeder/CV/RGB1", results_rgb_1.to_mqtt_json())
            previous_state_rgb_1 = results_rgb_1.save_as_previous()
            time.sleep(SLEEP_TIME_BETWEEN_CAMERAS)
            results_rgb_2 = detect(CAMERA_RGB_2, lambda x: model_rgb(x, verbose=False)[0], 
                                CONF_THRESHOLD_RGB2, False, False)
            if check_if_publish(previous_state_rgb_2, results_rgb_2):
                mqtt.publish("Feeder/CV/RGB2", results_rgb_2.to_mqtt_json())
            previous_state_rgb_2 = results_rgb_2.save_as_previous()
            time.sleep(SLEEP_TIME_BETWEEN_CAMERAS)
            results_ir = detect(CAMERA_IR, model_ir, CONF_THRESHOLD_IR, False, True)
            if check_if_publish(previous_state_ir, results_ir):
                mqtt.publish("Feeder/CV/IR", results_ir.to_mqtt_json())
            previous_state_ir = results_ir.save_as_previous()
            time.sleep(SLEEP_TIME_BETWEEN_ITERATIONS)
        except Exception as e:
            print(f"======= Exception happened =======")
            print(f"time: {datetime.now().strftime("%Y-%m-%d %H:%M")}")
            print(e)
            print(f"==================================")
            time.sleep(60)
        