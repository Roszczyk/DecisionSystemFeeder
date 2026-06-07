from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(4)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Kamera</title>
</head>
<body>
    <h1>Podgląd kamery</h1>
    <img src="/video_feed" width="800">
</body>
</html>
"""

def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame, [
            int(cv2.IMWRITE_JPEG_QUALITY), 80
        ])

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
