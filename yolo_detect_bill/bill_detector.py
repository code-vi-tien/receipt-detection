#!/usr/bin/env python3
"""
YOLO-based bill detector module
"""

from typing import List, Dict
from pathlib import Path

# Try to import YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    print("Warning: ultralytics not found. YOLO detection disabled.")
    HAS_YOLO = False

from config import BillOCRConfig


class BillDetector:
    """YOLO-based bill detector"""
    
    def __init__(self, config: BillOCRConfig = None, model_path: str = None):
        if config is None:
            from config import default_config
            config = default_config
            
        self.config = config
        self.model = None
        
        # Use provided model path or default
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = self.config.yolo_model_path
        
    def load_model(self):
        """Load YOLO model"""
        if not HAS_YOLO:
            print("YOLO not available")
            return False
            
        if not self.model_path.exists():
            print(f"YOLO model not found: {self.model_path}")
            return False
            
        try:
            self.model = YOLO(str(self.model_path))
            print(f"✅ YOLO model loaded from: {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def detect_bills(self, image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect bills in image"""
        if not self.model:
            if not self.load_model():
                return []
            
        try:
            results = self.model(image_path)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confidences):
                        class_name = self.model.names[int(cls)]
                        if class_name.lower() == "receipt" and conf > confidence_threshold:
                            detections.append({
                                'bbox': box.tolist(),
                                'confidence': float(conf),
                                'class': class_name,
                                'x1': float(box[0]),
                                'y1': float(box[1]),
                                'x2': float(box[2]),
                                'y2': float(box[3])
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection failed: {e}")
            return []
    
    def detect_bills_from_frame(self, frame, confidence_threshold: float = 0.5) -> List[Dict]:
        """Detect bills from video frame (numpy array)"""
        if not self.model:
            if not self.load_model():
                return []
            
        try:
            results = self.model(frame)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for box, cls, conf in zip(boxes, classes, confidences):
                        class_name = self.model.names[int(cls)]
                        if class_name.lower() == "receipt" and conf > confidence_threshold:
                            detections.append({
                                'bbox': box.tolist(),
                                'confidence': float(conf),
                                'class': class_name,
                                'x1': float(box[0]),
                                'y1': float(box[1]),
                                'x2': float(box[2]),
                                'y2': float(box[3])
                            })
            
            return detections
            
        except Exception as e:
            print(f"Detection failed: {e}")
            return []


# Convenience function
def create_bill_detector(model_path: str = None) -> BillDetector:
    """Create a bill detector instance"""
    return BillDetector(model_path=model_path)
