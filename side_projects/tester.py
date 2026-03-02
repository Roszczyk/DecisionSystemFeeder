import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)

    if cap.isOpened():
        print("Camera found at index", i)

        ret, frame = cap.read()

        if ret:
            filename = f"camera_{i}.jpg"
            cv2.imwrite(filename, frame)
            print("Saved photo:", filename)
        else:
            print("Could not read frame from camera", i)

        cap.release()
