import cv2
import numpy as np
from pathlib import Path
import json

with open(Path(__file__).parent / "config.json") as f:
    config = json.load(f)

cap = cv2.VideoCapture(config["webcam_no"])
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) 

i = 0
if not cap.isOpened():
    print(f"Camera (number {config['webcam_no']}) unavailable")
    exit()

while True:
    ret, frame = cap.read()  

    frame = np.reshape(frame[0],(2,192,256,2))
    raw = frame[1,:,:,:].astype(np.intc)
    raw = (raw[:,:,1] << 8) + raw[:,:,0]
        
    temp = raw/64 - 273.2
    
    i += 1
    if i%20 == 0 :
        print(f"min = {temp.min():1.4}, max = {temp.max():1.4}, avg = {temp.mean():1.4},")
    
    brightness = 0.01
    contrast = 0.95

    temp = (temp-temp.min())/(temp.max()-temp.min()) * contrast + brightness
    
    norm = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    cv2.imshow('Thermal View', colored)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()