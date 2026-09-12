import cv2
import numpy as np

def process_ir_frame(frame, type = "static"):
    assert(type == "static" or type == "adjusted")
    if frame.shape != (384, 256, 3):
        print(f"Unexpected IR frame shape: {frame.shape}")
        return None
    frame = np.split(frame, 2, axis=0)
    raw = frame[1].astype(np.intc).squeeze()
    raw = (raw[:, :, 1] << 8) + raw[:, :, 0]
    temp = raw / 64 - 273.2

    brightness = 0.01
    contrast = 0.95
    if type == "static":
        minimum = -10
        maximum = 30
    elif type == "adjusted":
        minimum = temp.min()
        maximum = temp.max()
    temp = (temp - minimum) / (maximum - minimum) * contrast + brightness
    gamma = 2.5
    temp = np.power(temp, gamma)

    norm = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored = cv2.rotate(colored, cv2.ROTATE_90_CLOCKWISE)
    return colored