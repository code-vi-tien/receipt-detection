#!/usr/bin/env python3
"""
Complete OCR Pipeline Demo
Demo script for testing YOLO + SVTR v6 + PaddleOCR integration
"""

from pathlib import Path
import json
import sys

# Add paths
sys.path.append(str(Path(__file__).parent / "yolo_detect_bill"))
sys.path.append(str(Path(__file__).parent / "svtr_v6_ocr"))

from yolo_detect_bill.bill_detector import BillDetector
from svtr_v6_ocr.svtr_v6_ocr import SVTRv6TrueInference
import cv2
import numpy as np

class PaddleOCREngine:
    """Simple PaddleOCR wrapper"""
    def __init__(self):
        from paddleocr import PaddleOCR
        det_model_path = str(Path(__file__).parent / "paddle_ocr" / "ch_db_res18")
        self.ocr = PaddleOCR(
            det_model_dir=det_model_path,
            rec=True,
            use_angle_cls=False,
            use_gpu=False,
            lang='ch'
        )
    
    def predict(self, image_np):
        """Predict from numpy array"""
        temp_path = "temp_demo.jpg"
        cv2.imwrite(temp_path, image_np)
        result = self.ocr.ocr(temp_path, cls=False)
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        output_data = []
        if result and len(result[0]) > 0:
            for line in result[0]:
                coords = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]
                
                output_data.append({
                    "text": text,
                    "confidence": confidence,
                    "coordinates": coords
                })
        
        return output_data

def demo_complete_pipeline():
    """Demo complete pipeline"""
    print("🚀 Complete OCR Pipeline Demo")
    print("=" * 60)
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # YOLO detector
    yolo_model_path = Path(__file__).parent / "yolo_detect_bill" / "bill_models.pt"
    yolo_detector = BillDetector(model_path=str(yolo_model_path))
    yolo_detector.load_model()
    
    # SVTR v6
    svtr_engine = SVTRv6TrueInference()
    
    # PaddleOCR
    paddle_engine = PaddleOCREngine()
    
    print("✅ All components initialized!")
    
    # Test image
    test_image = Path("image_test/X51007103668.jpg")
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"\n📸 Processing: {test_image.name}")
    
    # Step 1: YOLO Detection
    print("\n1️⃣ YOLO Bill Detection...")
    image = cv2.imread(str(test_image))
    detections = yolo_detector.detect_bills_from_frame(image, confidence_threshold=0.1)
    
    if not detections:
        print("❌ No bills detected!")
        return
    
    best_detection = max(detections, key=lambda x: x['confidence'])
    print(f"✅ Best bill detected (confidence: {best_detection['confidence']:.3f})")
    
    # Crop bill region
    x1, y1, x2, y2 = (
        int(best_detection['x1']), int(best_detection['y1']),
        int(best_detection['x2']), int(best_detection['y2'])
    )
    
    # Add padding
    padding = 10
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    bill_crop = image[y1:y2, x1:x2]
    print(f"✂️ Bill crop size: {bill_crop.shape}")
    
    # Step 2: SVTR v6 Processing
    print("\n2️⃣ SVTR v6 Text Recognition...")
    temp_path = "temp_demo_svtr.jpg"
    cv2.imwrite(temp_path, bill_crop)
    
    svtr_result = svtr_engine.predict_text(temp_path)
    svtr_texts = svtr_result.get('texts', [])
    
    import os
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"🤖 SVTR v6: {len(svtr_texts)} texts detected")
    
    # Step 3: PaddleOCR Processing
    print("\n3️⃣ PaddleOCR Text Recognition...")
    paddle_texts = paddle_engine.predict(bill_crop)
    print(f"🧠 PaddleOCR: {len(paddle_texts)} texts detected")
    
    # Step 4: Comparison
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 60)
    print(f"📝 Text Detection:")
    print(f"   - SVTR v6: {len(svtr_texts)} texts")
    print(f"   - PaddleOCR: {len(paddle_texts)} texts")
    print(f"   - Difference: {len(paddle_texts) - len(svtr_texts)} texts")
    
    # Confidence comparison
    if svtr_texts:
        svtr_avg_conf = sum(t['confidence'] for t in svtr_texts) / len(svtr_texts)
        svtr_high_conf = sum(1 for t in svtr_texts if t['confidence'] > 0.9)
        print(f"\n🤖 SVTR v6 Stats:")
        print(f"   - Average confidence: {svtr_avg_conf:.3f}")
        print(f"   - High confidence (>0.9): {svtr_high_conf}/{len(svtr_texts)} ({svtr_high_conf/len(svtr_texts)*100:.1f}%)")
    
    if paddle_texts:
        paddle_avg_conf = sum(t['confidence'] for t in paddle_texts) / len(paddle_texts)
        paddle_high_conf = sum(1 for t in paddle_texts if t['confidence'] > 0.9)
        print(f"\n🧠 PaddleOCR Stats:")
        print(f"   - Average confidence: {paddle_avg_conf:.3f}")
        print(f"   - High confidence (>0.9): {paddle_high_conf}/{len(paddle_texts)} ({paddle_high_conf/len(paddle_texts)*100:.1f}%)")
    
    # Sample texts
    print(f"\n📝 Sample Texts (First 5):")
    print("=" * 60)
    
    print("🤖 SVTR v6:")
    for i, text_info in enumerate(svtr_texts[:5], 1):
        print(f"   {i}. '{text_info['text']}' (conf: {text_info['confidence']:.3f})")
    
    print("\n🧠 PaddleOCR:")
    for i, text_info in enumerate(paddle_texts[:5], 1):
        print(f"   {i}. '{text_info['text']}' (conf: {text_info['confidence']:.3f})")
    
    # Save results
    results = {
        "input_image": str(test_image),
        "yolo_detection": {
            "total_detections": len(detections),
            "best_detection": best_detection,
            "crop_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        },
        "svtr_results": {
            "model": "SVTR v6",
            "total_texts": len(svtr_texts),
            "texts": svtr_texts
        },
        "paddle_results": {
            "model": "PaddleOCR",
            "total_texts": len(paddle_texts),
            "texts": paddle_texts
        }
    }
    
    output_file = "demo_pipeline_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    demo_complete_pipeline()
