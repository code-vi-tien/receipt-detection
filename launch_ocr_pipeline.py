#!/usr/bin/env python3
"""
OCR Pipeline Launcher
Quick start script for the complete OCR pipeline
"""
"""Author: Tôn Thất Thanh Tuấn"""

import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    package_imports = {
        'PySide6': 'PySide6',
        'paddlepaddle': 'paddle', 
        'paddleocr': 'paddleocr',
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} found")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} missing")
    
    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("pip install " + " ".join(missing_packages))
        print("\nOr install all requirements:")
        print("pip install -r requirements_gui.txt")
        return False
    
    return True

def check_models():
    """Check if model files exist"""
    base_dir = Path(__file__).parent
    
    required_files = [
        base_dir / "yolo_detect_bill" / "bill_models.pt",
        base_dir / "svtr_v6_ocr" / "best_accuracy.pdparams",
        base_dir / "paddle_ocr" / "ch_db_res18"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("❌ Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Make sure all model files are in place")
        return False
    
    return True

def main():
    """Main launcher"""
    print("🚀 OCR Pipeline Launcher")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All required packages found")
    
    # Check models
    print("🔍 Checking model files...")
    if not check_models():
        sys.exit(1)
    print("✅ All model files found")
    
    # Launch GUI
    print("🎯 Launching OCR Pipeline GUI...")
    try:
        from ocr_pipeline_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print("\n💡 Try running directly:")
        print("python ocr_pipeline_gui.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
