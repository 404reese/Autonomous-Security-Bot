import cv2
import numpy as np
from typing import List, Dict, Any

class PersonPoliceDetector:
    def __init__(self):
        # Load pre-trained models
        self.net = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",  # You'll download this
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        # Simple uniform color detection for police (blue/black)
        self.police_colors = [(0,0,255), (0,0,139), (25,25,112)]  # BGR
    
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns: [{"type": "police" or "unknown", "bbox": [x,y,w,h], "confidence": 0.xx}]
        """
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,177,123))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        results = []
        for i in range(detections.shape[2]):
            conf = detections[0,0,i,2]
            if conf > 0.5:
                box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                (x,y,x2,y2) = box.astype(int)
                
                # Check if person is wearing police-like uniform
                person_roi = frame[y:y2, x:x2]
                is_police = self._check_uniform(person_roi)
                
                results.append({
                    "type": "police" if is_police else "unknown",
                    "bbox": [x,y,x2-x,y2-y],
                    "confidence": float(conf)
                })
        return results
    
    def _check_uniform(self, roi: np.ndarray) -> bool:
        if roi.size == 0:
            return False
        # Average color detection (simplified)
        avg_color = cv2.mean(roi)[:3]
        for police_color in self.police_colors:
            if all(abs(avg_color[i] - police_color[i]) < 50 for i in range(3)):
                return True
        return False

# For quick testing without model files
class MockDetector:
    def detect(self, frame):
        return [{"type": "unknown", "bbox": [100,100,200,400], "confidence": 0.85}]