import asyncio
import base64
import json
import threading
import queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from google import genai
from google.genai import types

from functions import TOOL_DEFINITIONS, FunctionHandler
from cv_detector import MockDetector  # Use MockDetector for testing
from face_recognizer import OfficerFaceRecognizer, SimpleMockRecognizer

# ---------- Setup ----------
app = FastAPI()
detector = MockDetector()
dashboard_alerts = queue.Queue()  # Simple queue to send alerts to dashboard

def send_to_dashboard(alert):
    dashboard_alerts.put(alert)

func_handler = FunctionHandler(send_to_dashboard)

# Gemini client (replace with your API key)
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")
model = "gemini-3.1-flash-live-preview"

try:
    face_recognizer = OfficerFaceRecognizer()
    print("✅ Face recognizer initialized")
except:
    face_recognizer = SimpleMockRecognizer()
    print("⚠️ Using mock face recognizer")
    
# ---------- WebSocket for ESP32/Hardware (Simulated) ----------
@app.websocket("/ws/bot")
async def bot_websocket(websocket: WebSocket):
    await websocket.accept()
    print("✅ Bot connected (simulated)")
    
    # Start Gemini Live session
    config = {
        "response_modalities": ["AUDIO", "TEXT"],
        "tools": TOOL_DEFINITIONS,
        "system_instruction": """You are Sentinel, a helpful city patrol bot. 
        Listen to citizens. If someone asks for an escort, immediately call request_police_escort().
        If you see a parking violation, call report_parking_violation()."""
    }
    
    async with client.aio.live.connect(model=model, config=config) as session:
        # Send initial greeting
        await session.send_realtime_input(text="Hello, I am Sentinel. How can I help you?")
        
        # Receive loop
        async for response in session.receive():
            # Handle text transcription
            if response.server_content and response.server_content.input_transcription:
                user_text = response.server_content.input_transcription.text
                print(f"👤 User: {user_text}")
                await websocket.send_text(json.dumps({"type": "transcript", "text": user_text}))
            
            # Handle function calls
            if response.tool_call:
                for fc in response.tool_call.function_calls:
                    if fc.name == "request_police_escort":
                        result = await func_handler.request_police_escort(**fc.args)
                    elif fc.name == "report_parking_violation":
                        result = await func_handler.report_parking_violation(**fc.args)
                    else:
                        result = {"error": "unknown function"}
                    
                    await session.send_tool_response(function_responses=[types.FunctionResponse(
                        name=fc.name, id=fc.id, response=result
                    )])
            
            # Handle audio output (for hardware speaker)
            if response.server_content and response.server_content.model_turn:
                for part in response.server_content.model_turn.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("audio"):
                        audio_bytes = part.inline_data.data
                        # In real hardware, send to ESP32 speaker
                        await websocket.send_bytes(audio_bytes)

# ---------- WebSocket for Dashboard ----------
@app.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket):
    await websocket.accept()
    print("📊 Dashboard connected")
    while True:
        try:
            # Wait for alerts from Gemini functions
            alert = dashboard_alerts.get()
            await websocket.send_text(json.dumps(alert))
        except WebSocketDisconnect:
            break

# ---------- Video Stream Endpoint (Simulates bot's camera) ----------
@app.post("/process_frame")
async def process_frame(frame_data: dict):
    # In real hardware, this receives JPEG from ESP32
    # For demo, we'll simulate detection
    import numpy as np
    fake_frame = np.zeros((480,640,3), dtype=np.uint8)
    detections = detector.detect(fake_frame)
    return {"detections": detections}


@app.post("/recognize_faces")
async def recognize_faces(frame_data: dict):
    """Receive image and return face detections with names"""
    import base64
    import numpy as np
    
    # Decode base64 image
    image_data = base64.b64decode(frame_data['image'].split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect and recognize faces
    detections = face_recognizer.detect_faces(frame)
    
    # Draw on frame (optional, for display)
    annotated_frame = face_recognizer.draw_detections(frame.copy(), detections)
    
    # Encode back to base64 for dashboard
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_base64 = base64.b64encode(buffer).decode()
    
    return {
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{annotated_base64}"
    }

# ---------- Run Server ----------
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    print("🚀 Starting Sentinel Server on http://localhost:8000")
    run_server()