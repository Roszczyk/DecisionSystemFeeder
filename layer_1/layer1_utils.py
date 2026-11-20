from pathlib import Path
import cv2
import time
import json

from CV_InfraredCamera.infrared_utils import process_ir_frame

class CV_Det:
    def __init__(self, is_detected : bool, how_many : int, certainties : list[float], 
                 detection_threshold : float):
        assert (is_detected and how_many > 0) or (not is_detected and how_many == 0), \
            "Number of objects detected does not match the detection bool"
        self.is_detected = is_detected
        self.how_many = how_many
        assert len(certainties) == how_many, \
            f"Number of provided certainties ({len(certainties)}) " \
            f"does not match the number of objects ({how_many})"
        for c in certainties + [detection_threshold]:
            assert c >= 0.0 and c <= 1.0, f"certainties need to be [0.0, 1.0], 0.0<{c}<1.0 does not hold"
        self.certainties = certainties
        self.detection_threshold = detection_threshold

    def to_mqtt_json(self, common_certainty_mode = "highest"):
        assert common_certainty_mode in ["highest", "mean"] 
        # TODO: some more fair mode - if we have two birds with prob p1, then the common 
        #       prob should be >p1, as the level of certainty rises
        if self.is_detected:
            common_certainty = max(self.certainties) if common_certainty_mode=="highest" \
                else sum(self.certainties)/len(self.certainties)
        else:
            common_certainty = self.detection_threshold
            # TODO: Maybe here it's possible to work out some better method...
        prep_dict = dict({
            "is_detected" : 1 if self.is_detected else 0,
            "how_many" : self.how_many,
            "certainty" : common_certainty
        })
        return json.dumps(prep_dict)

def record_and_save_video(cam_no, save_path, duration=5, is_ir=False, rotate=False, verbose=False):
    cap = cv2.VideoCapture(cam_no)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 20.0

    ret, frame = cap.read()

    if not ret:
        print("Cannot read first frame")
        cap.release()
        return
    if is_ir:
        frame = process_ir_frame(frame)
    if frame is None:
        print("IR processing failed")
        cap.release()
        return
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        str(save_path),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("Cannot create video:", save_path)
        cap.release()
        return

    start = time.time()
    out.write(frame)

    while time.time() - start < duration:
        ret, frame = cap.read()

        if not ret:
            break
        if is_ir:
            frame = process_ir_frame(frame)
        if frame is None:
            print("Invalid frame")
            continue
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if frame.shape[:2] != (height, width):
            print("Frame dimensions mismatch:", frame.shape)
            continue

        out.write(frame)

    cap.release()
    out.release()

    if verbose:
        print(f"Saved video: {save_path}")

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