import cv2
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

CAM = sys.argv[1]
RECORD_DIR = Path(__file__).parent / "recordings" / CAM
CONFIG_FILE = Path(__file__).parent / "config.json"
RECORD_SECONDS = 60 * 60  #1 hour
DELETE_OLDER_THAN_HOURS = 24
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 2.0

os.makedirs(RECORD_DIR, exist_ok=True)

def get_camera_config(config_file):
    with open(Path(__file__).parent / "config.json") as f:
        config = json.load(f)
    return config


def delete_old_files():
    now = datetime.now()
    for filename in os.listdir(RECORD_DIR):
        filepath = RECORD_DIR / filename
        if not os.path.isfile(filepath):
            continue
        try:
            file_time = datetime.strptime(filename, "%Y-%m-%d_%H.avi")
            if now - file_time > timedelta(hours=DELETE_OLDER_THAN_HOURS):
                os.remove(filepath)
                print(f"Deleted {filename}")
        except ValueError:
            continue


def main():
    cam_no = get_camera_config(CONFIG_FILE)[CAM]
    cap = cv2.VideoCapture(cam_no)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Unable to run camera")
        return

    while True:
        start_time = time.time()
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H.avi")
        filepath = RECORD_DIR / filename

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
