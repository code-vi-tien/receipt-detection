#!/usr/bin/env python3
"""
Quick Test Script for OCR Pipeline
Demo the complete pipeline with a sample image
"""
"""Author: Tôn Thất Thanh Tuấn"""
import os
import sys
from pathlib import Path

def main():
    print("🚀 OCR Pipeline Quick Test")
    print("=" * 50)
    
    # Check if sample image exists
    sample_image = Path(__file__).parent / "image_test" / "X51005200931.jpg"
    
    if not sample_image.exists():
        print("❌ Sample image not found")
        print(f"Looking for: {sample_image}")
        return
    
    print(f"📁 Using sample image: {sample_image.name}")
    print("\n🎯 Pipeline will:")
    print("1. 🔍 YOLO: Detect bill regions")
    print("2. ✂️ Crop: Extract best bill region") 
    print("3. 🤖 SVTR v6: OCR processing")
    print("4. 🧠 PaddleOCR: OCR processing")
    print("5. 📊 Compare: Side-by-side results")
    print("6. 💾 Save: JSON results")
    
    print("\n🚀 Starting GUI...")
    print("💡 In GUI:")
    print("   - Click 'Select Image' and choose the sample")
    print("   - Click 'Process OCR' to run pipeline")
    print("   - View results in comparison tabs")
    print("   - Save results as JSON")
    
    # Launch GUI
    try:
        from launch_ocr_pipeline import main as launcher_main
        launcher_main()
    except KeyboardInterrupt:
        print("\n👋 Pipeline stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
