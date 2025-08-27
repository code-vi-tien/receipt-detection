#!/usr/bin/env python3
"""
Quick test for SVTR v6 integration in pipeline
"""
"""Author: Tôn Thất Thanh Tuấn"""
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "svtr_v6_ocr"))

from svtr_v6_ocr.svtr_v6_ocr import SVTRv6TrueInference
import cv2
import numpy as np

def test_svtr_integration():
    """Test SVTR v6 integration"""
    print("🔍 Testing SVTR v6 integration...")
    
    # Initialize SVTR
    svtr_engine = SVTRv6TrueInference()
    
    # Test image
    test_image = Path("image_test/X51007103668.jpg")
    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    # Test predict_text method
    print("\n📋 Testing predict_text method...")
    result = svtr_engine.predict_text(test_image)
    
    print(f"📊 SVTR Result structure:")
    print(f"   Keys: {list(result.keys())}")
    print(f"   Status: {result.get('status', 'N/A')}")
    print(f"   Text count: {result.get('text_count', 0)}")
    print(f"   Texts field exists: {'texts' in result}")
    
    if 'texts' in result and result['texts']:
        print(f"\n📝 First 3 texts:")
        for i, text_info in enumerate(result['texts'][:3], 1):
            print(f"   {i}. {text_info}")
    
    # Test with numpy array (like GUI would use)
    print("\n🖼️ Testing with numpy array (bill crop simulation)...")
    image = cv2.imread(str(test_image))
    if image is not None:
        # Simulate a bill crop
        h, w = image.shape[:2]
        crop = image[50:h-50, 50:w-50]  # Simple crop
        
        # Save temp and test
        temp_path = "temp_test_crop.jpg"
        cv2.imwrite(temp_path, crop)
        
        crop_result = svtr_engine.predict_text(temp_path)
        print(f"   Crop result keys: {list(crop_result.keys())}")
        print(f"   Crop text count: {crop_result.get('text_count', 0)}")
        
        # Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    print("\n✅ SVTR v6 integration test completed!")
    return result

if __name__ == "__main__":
    test_svtr_integration()
