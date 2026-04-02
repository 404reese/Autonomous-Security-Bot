import cv2
import numpy as np
import os
import pickle
from typing import List, Dict, Tuple, Optional

# Try multiple backends for compatibility
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("✅ Using face_recognition library")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("⚠️ face_recognition not available, falling back to OpenCV + LBPH")
    # Fallback to OpenCV's LBPH face recognizer
    import cv2.face

class OfficerFaceRecognizer:
    def __init__(self, known_faces_path: str = "known_faces.pkl"):
        self.known_faces_path = known_faces_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load pre-registered officer faces"""
        if os.path.exists(self.known_faces_path):
            with open(self.known_faces_path, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_names = data['names']
            print(f"✅ Loaded {len(self.known_face_names)} known officers")
        else:
            print("⚠️ No known faces found. Please register officers first.")
            
    def register_officer(self, image_path: str, name: str):
        """Register a new officer from an image file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
            
        if USE_FACE_RECOGNITION:
            # Convert BGR to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            
            if len(encodings) == 0:
                raise ValueError(f"No face found in {image_path}")
            
            encoding = encodings[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
        else:
            # Fallback: store face ROI for simple matching
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                raise ValueError(f"No face found in {image_path}")
            
            # Store face ROI and name for template matching
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            self.known_face_encodings.append(face_roi)  # Store ROI instead of encoding
            self.known_face_names.append(name)
            
        self.save_known_faces()
        print(f"✅ Registered officer: {name}")
        
    def save_known_faces(self):
        """Save registered faces to disk"""
        with open(self.known_faces_path, 'wb') as f:
            pickle.dump({
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }, f)
            
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces and recognize known officers.
        Returns: [{"name": "Officer Smith", "bbox": [x,y,w,h], "confidence": 0.95}]
        """
        # Convert to RGB if needed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face locations
        if USE_FACE_RECOGNITION:
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        else:
            # Fallback: use Haar cascade
            face_locations = self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            face_encodings = [None] * len(face_locations)
            # Convert locations to same format as face_recognition (top, right, bottom, left)
            face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in face_locations]
        
        results = []
        
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            if USE_FACE_RECOGNITION:
                # Compare with known officers
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown Person"
                confidence = 0.0
                
                if True in matches:
                    match_index = matches.index(True)
                    name = self.known_face_names[match_index]
                    # Calculate confidence based on face distance
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    confidence = 1 - face_distances[match_index] if face_distances[match_index] < 0.6 else 0.5
                else:
                    confidence = 0.3
            else:
                # Fallback: simple template matching with stored ROIs
                top, right, bottom, left = face_location
                face_roi = gray_frame[top:bottom, left:right]
                
                best_match = "Unknown Person"
                best_score = 0
                
                for idx, known_roi in enumerate(self.known_face_encodings):
                    if known_roi.shape == face_roi.shape:
                        result = cv2.matchTemplate(face_roi, known_roi, cv2.TM_CCOEFF_NORMED)
                        score = result[0][0]
                        if score > best_score and score > 0.5:
                            best_score = score
                            best_match = self.known_face_names[idx]
                
                name = best_match
                confidence = best_score
            
            # Convert location to [x, y, w, h] format
            top, right, bottom, left = face_location
            results.append({
                "name": name,
                "bbox": [left, top, right - left, bottom - top],
                "confidence": float(confidence),
                "is_officer": name != "Unknown Person"
            })
            
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and names on frame"""
        for det in detections:
            x, y, w, h = det["bbox"]
            name = det["name"]
            confidence = det["confidence"]
            is_officer = det["is_officer"]
            
            # Choose color: blue for officer, red for unknown
            color = (255, 0, 0) if is_officer else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw background for text
            label = f"{name} ({confidence:.0%})" if confidence > 0 else name
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return frame

# For testing without face_recognition library
class SimpleMockRecognizer:
    def detect_faces(self, frame):
        return [{"name": "Officer Johnson", "bbox": [100,100,200,200], "confidence": 0.92, "is_officer": True}]
    
    def draw_detections(self, frame, detections):
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, det["name"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame