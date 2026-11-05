from pathlib import Path
import cv2
import time
import json

from CV_InfraredCamera.infrared_utils import process_ir_frame

def record_and_save_video(cam_no, save_path, duration=5, is_ir=False, rotate=False, verbose=False):
    cap = cv2.VideoCapture(cam_no)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 120:
        fps = 20.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(save_path),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print(f"Cannot create video: {save_path}")
        cap.release()
        return

    start = time.time()

    while time.time() - start < duration:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame")
            break

        if is_ir:
            frame = process_ir_frame(frame)
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        out.write(frame)
    cap.release()
    out.release()

    if verbose:
        print(f"🎥 Saved video: {save_path}")

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

def save_frame(save_path : Path, frame, verbose=False):
    cv2.imwrite(save_path, frame)
    if verbose:
        print(f"📸 Saved: {save_path}")