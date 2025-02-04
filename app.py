from flask import Flask, render_template, Response, jsonify
import cv2
from utils.liveness import LivenessDetector

app = Flask(__name__)

detector = LivenessDetector()
is_running = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global is_running
    is_running = True
    return jsonify({'message': 'Liveness detection started!'})

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    is_running = False
    return jsonify({'message': 'Liveness detection stopped.'})

def generate_frames():
    camera = cv2.VideoCapture(0)
    while is_running:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, result = detector.detect(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

if __name__ == '__main__':
    app.run(debug=True)