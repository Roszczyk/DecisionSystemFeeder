import cv2
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import numpy as np

CAM = sys.argv[1]
RECORD_DIR = Path(__file__).parent / "recordings" / CAM
CONFIG_FILE = Path(__file__).parent / "config.json"
RECORD_SECONDS = 60 * 60  # 1 hour
DELETE_OLDER_THAN_HOURS = 76
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 15.0
IS_MODE_COMBINED = False

os.makedirs(RECORD_DIR, exist_ok=True)

if CAM == "COMBINED":
    IS_MODE_COMBINED = True

def get_camera_config(config_file):
    with open(config_file) as f:
        return json.load(f)

def delete_old_files():
    now = datetime.now()
    for filename in os.listdir(RECORD_DIR):
        filepath = RECORD_DIR / filename
        if not os.path.isfile(filepath):
            continue
        try:
            file_time = datetime.strptime(filename, "%Y-%m-%d_%H.mp4")
            if now - file_time > timedelta(hours=DELETE_OLDER_THAN_HOURS):
                os.remove(filepath)
                print(f"Deleted {filename}")
        except ValueError:
            continue

def process_ir_frame(frame):
    if frame.shape != (384, 256, 2):
        print(f"Unexpected IR frame shape: {frame.shape}")
        return None
    frame = np.split(frame, 2, axis=0)
    raw = frame[1].astype(np.intc).squeeze()
    raw = (raw[:, :, 1] << 8) + raw[:, :, 0]
    temp = raw / 64 - 273.2

    brightness = 0.01
    contrast = 0.95
    temp = (temp - temp.min()) / (temp.max() - temp.min()) * contrast + brightness

    norm = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return colored

def record_ir_camera(cam_no):
    cap = cv2.VideoCapture(cam_no)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("IR Camera unavailable")
        return

    end=False

    while True:
        start_time = time.time()
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H.mp4")
        filepath = RECORD_DIR / filename

        out = cv2.VideoWriter(str(filepath), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (256, 192))
        print(f"{now} IR Recording started: {filename}")

        try: 
            while time.time() - start_time < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab IR frame")
                    break
                processed = process_ir_frame(frame)

                if processed is not None:
                    out.write(processed)
        except KeyboardInterrupt:
            end=True

        out.release()
        delete_old_files()
        
        if IS_MODE_COMBINED:
            end = True

        if end:
            break

    cap.release()
    cv2.destroyAllWindows()

    if IS_MODE_COMBINED:
        return

def record_regular_camera(cam_no):
    cap = cv2.VideoCapture(cam_no)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Unable to run regular camera")
        return

    while True:
        start_time = time.time()
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H.mp4")
        filepath = RECORD_DIR / filename
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        print(f"{now} Recording started: {filename}")

        while time.time() - start_time < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Camera failed")
                break
            out.write(frame)

        out.release()
        delete_old_files()

        if IS_MODE_COMBINED:
            break

    cap.release()
    cv2.destroyAllWindows()

    if IS_MODE_COMBINED:
        return

def main():
    config = get_camera_config(CONFIG_FILE)
    FIRST=True

    while IS_MODE_COMBINED or FIRST:
        FIRST = False

        now_time = datetime.now().hour
        
        if IS_MODE_COMBINED and (now_time >= 20 or now_time < 5):
            CAM = "IRCAM"
        elif IS_MODE_COMBINED and (now_time < 20 and now_time >= 5):
            CAM = "RGBCAM"
        
        cam_no = config[CAM]

        if CAM == "IRCAM":
            record_ir_camera(cam_no)
        else:
            record_regular_camera(cam_no)

if __name__ == "__main__":
    main()
