#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test setup for OCR Pipeline - Kiểm tra cài đặt hệ thống
Author: Tôn Thất Thanh Tuấn
Date: 2025-08-25
"""

import sys
import os
from pathlib import Path

def print_header():
    """Print header"""
    print("=" * 80)
    print("🏦 HỆ THỐNG OCR HÓA ĐƠN - KIỂM TRA CÀI ĐẶT")
    print("   Bill Detection System - Setup Check")
    print("=" * 80)

def check_python_version():
    """Check Python version"""
    print("\n📋 KIỂM TRA PYTHON:")
    version = sys.version_info
    print(f"   • Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("   ✅ Python version hỗ trợ")
        return True
    else:
        print("   ❌ Python version không hỗ trợ (cần >= 3.8)")
        return False

def check_dependencies():
    """Check all dependencies"""
    print("\n📦 KIỂM TRA DEPENDENCIES:")
    
    dependencies = [
        ("PySide6", "PySide6"),
        ("PyTorch", "torch"),
        ("TorchVision", "torchvision"),
        ("Ultralytics (YOLO)", "ultralytics"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Pillow", "PIL"),
        ("PaddlePaddle", "paddle"),
        ("PaddleOCR", "paddleocr")
    ]
    
    all_ok = True
    
    for name, module in dependencies:
        try:
            __import__(module)
            # Get version if available
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'Unknown')
                print(f"   ✅ {name}: {version}")
            except:
                print(f"   ✅ {name}: OK")
        except ImportError:
            print(f"   ❌ {name}: Chưa cài đặt")
            all_ok = False
    
    return all_ok

def check_files():
    """Check required files"""
    print("\n📁 KIỂM TRA TỆP TIN:")
    
    required_files = [
        "ocr_pipeline_gui.py",
        "YOLO_Coor.py",
        "bill_models.pt"
    ]
    
    optional_files = [
        "requirements.txt",
        "HUONG_DAN_CAI_DAT.md",
        "run_windows.bat",
        "run_linux.sh"
    ]
    
    all_required = True
    
    print("   📋 Tệp bắt buộc:")
    for file in required_files:
        if Path(file).exists():
            print(f"      ✅ {file}")
        else:
            print(f"      ❌ {file} - Không tìm thấy!")
            all_required = False
    
    print("   📋 Tệp tùy chọn:")
    for file in optional_files:
        if Path(file).exists():
            print(f"      ✅ {file}")
        else:
            print(f"      ⚠️  {file} - Không có")
    
    return all_required

def check_directories():
    """Check required directories"""
    print("\n📂 KIỂM TRA THƯ MỤC:")
    
    required_dirs = [
        "dataset",
        "image_test",
        "svtr_v6_ocr",
        "yolo_detect_bill"
    ]
    
    optional_dirs = [
        "model_kien_ocr_2",
        "model_son_ocr_1",
        "training_data"
    ]
    
    all_required = True
    
    print("   📋 Thư mục bắt buộc:")
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"      ✅ {dir_name}/")
        else:
            print(f"      ❌ {dir_name}/ - Không tìm thấy!")
            all_required = False
    
    print("   📋 Thư mục tùy chọn:")
    for dir_name in optional_dirs:
        if Path(dir_name).exists():
            print(f"      ✅ {dir_name}/")
        else:
            print(f"      ⚠️  {dir_name}/ - Không có")
    
    return all_required

def test_imports():
    """Test critical imports"""
    print("\n🧪 KIỂM TRA IMPORT:")
    
    tests = [
        ("YOLO Detector", "from YOLO_Coor import BillDetector"),
        ("SVTR v6", "from svtr_v6_ocr.svtr_v6_ocr import SVTRv6TrueInference"),
        ("PySide6 GUI", "from PySide6.QtWidgets import QApplication"),
        ("Computer Vision", "import cv2; import numpy as np")
    ]
    
    all_ok = True
    
    for name, import_code in tests:
        try:
            exec(import_code)
            print(f"   ✅ {name}: Import thành công")
        except Exception as e:
            print(f"   ❌ {name}: Lỗi import - {e}")
            all_ok = False
    
    return all_ok

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️  KIỂM TRA GPU:")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ CUDA khả dụng: {gpu_count} GPU(s)")
            print(f"   🎮 GPU chính: {gpu_name}")
            return True
        else:
            print("   ⚠️  CUDA không khả dụng - Sẽ sử dụng CPU")
            return False
    except:
        print("   ⚠️  Không thể kiểm tra GPU")
        return False

def provide_recommendations(python_ok, deps_ok, files_ok, dirs_ok, imports_ok):
    """Provide setup recommendations"""
    print("\n💡 KHUYẾN NGHỊ:")
    
    if not python_ok:
        print("   🔧 Cài đặt Python 3.8+ từ: https://www.python.org/downloads/")
    
    if not deps_ok:
        print("   🔧 Cài đặt dependencies:")
        print("      pip install -r requirements.txt")
        print("      hoặc:")
        print("      pip install torch torchvision ultralytics PySide6 opencv-python paddlepaddle paddleocr numpy Pillow")
    
    if not files_ok:
        print("   🔧 Đảm bảo có đầy đủ tệp tin cần thiết")
        print("   📥 Tải model bill_models.pt từ nguồn được cung cấp")
    
    if not dirs_ok:
        print("   🔧 Tạo thư mục cần thiết:")
        print("      mkdir -p dataset/img dataset/box dataset/entities")
        print("      mkdir -p image_test svtr_v6_ocr yolo_detect_bill")
    
    if not imports_ok:
        print("   🔧 Kiểm tra cấu trúc project và đường dẫn module")
    
    if all([python_ok, deps_ok, files_ok, dirs_ok, imports_ok]):
        print("   🎉 Hệ thống sẵn sàng! Có thể chạy ứng dụng:")
        print("      python ocr_pipeline_gui.py")
        print("      hoặc:")
        print("      ./run_linux.sh       (Linux)")
        print("      run_windows.bat      (Windows)")

def main():
    """Main test function"""
    print_header()
    
    # Run all checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    files_ok = check_files()
    dirs_ok = check_directories()
    imports_ok = test_imports()
    check_gpu()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TÓM TẮT KIỂM TRA:")
    
    checks = [
        ("Python Version", python_ok),
        ("Dependencies", deps_ok),
        ("Required Files", files_ok),
        ("Required Directories", dirs_ok),
        ("Critical Imports", imports_ok)
    ]
    
    for name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {name}")
    
    # Overall status
    all_ok = all([python_ok, deps_ok, files_ok, dirs_ok, imports_ok])
    
    if all_ok:
        print("\n🎉 TẤT CẢ KIỂM TRA ĐỀU THÀNH CÔNG!")
        print("🚀 Hệ thống sẵn sàng để chạy!")
    else:
        print("\n⚠️  MỘT SỐ KIỂM TRA THẤT BẠI!")
        print("🔧 Vui lòng xem khuyến nghị bên dưới:")
    
    provide_recommendations(python_ok, deps_ok, files_ok, dirs_ok, imports_ok)
    
    print("\n" + "=" * 80)
    print("=" * 80)

if __name__ == "__main__":
    main()
