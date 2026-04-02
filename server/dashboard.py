# Updated dashboard.py with face recognition integration

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import eventlet
import requests
import base64
import cv2
import numpy as np
import json

eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Store latest detections and annotated video frame
latest_frame = None
latest_detections = []

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

# WebRTC or simulated video feed
@app.route('/video_feed')
def video_feed():
    """Generate video feed with face labels"""
    def generate():
        global latest_frame
        cap = cv2.VideoCapture(0)  # Use webcam (0) or IP camera
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Send frame to server for face recognition
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode()
            
            # Call server's face recognition endpoint
            try:
                response = requests.post(
                    "http://localhost:8000/recognize_faces",
                    json={"image": f"data:image/jpeg;base64,{img_base64}"},
                    timeout=0.5
                )
                if response.status_code == 200:
                    data = response.json()
                    latest_detections = data['detections']
                    
                    # Get annotated frame from server
                    annotated_data = data['annotated_image'].split(',')[1]
                    annotated_bytes = base64.b64decode(annotated_data)
                    annotated_nparr = np.frombuffer(annotated_bytes, np.uint8)
                    frame = cv2.imdecode(annotated_nparr, cv2.IMREAD_COLOR)
                    
                    # Emit detections via WebSocket
                    socketio.emit('detections', latest_detections)
            except:
                pass
            
            # Encode and yield frame
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Dashboard connected')
    emit('alert', {'type': 'system', 'message': 'Face recognition ready'})

@app.route('/webhook/alert', methods=['POST'])
def receive_alert():
    data = request.json
    socketio.emit('alert', data)
    return 'OK'

if __name__ == '__main__':
    print("📊 Dashboard running at http://localhost:5000")
    print("🎥 Video feed at http://localhost:5000/video_feed")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)