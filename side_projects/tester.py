import cv2
from Photo_Taker.processing import process_ir_frame

for i in range(30):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

    if cap.isOpened():
        print("Camera found at index", i)

        ret, frame = cap.read()

        if ret:
            frame_copy = process_ir_frame(frame)
            if frame_copy is not None:
                print("Processed")
                frame = frame_copy
            filename = f"camera_{i}.jpg"
            cv2.imwrite(filename, frame)
            print("Saved photo:", filename, " shape: ", frame.shape)
        else:
            print("Could not read frame from camera", i)

        cap.release()

# To print all the cameras the following linux tool can be used:
# sudo apt install v4l-utils
# v4l2-ctl --list-devices
