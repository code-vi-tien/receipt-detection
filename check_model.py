#!/usr/bin/env python3
"""
Model Check Script for OCR Pipeline

This script checks that model directories for PaddleOCR and SVTR v6 contain files.
Customize expected_files nếu cần kiểm tra tên file bắt buộc.
Author: Tôn Thất Thanh Tuấn
Date: 2025-08-25
"""

import os
from pathlib import Path

def check_model_directory(directory: Path, expected_files=None) -> bool:
    """
    Check if the given directory exists and contains model files.
    """
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return False

    # Liệt kê các tập tin trong thư mục
    files = [f for f in directory.iterdir() if f.is_file()]
    if not files:
        print(f"❌ No model files found in directory: {directory}")
        return False
    
    if expected_files:
        missing = [f for f in expected_files if not (directory / f).exists()]
        if missing:
            print(f"❌ Missing expected model files in {directory}: {missing}")
            return False

    print(f"✅ Model directory '{directory.name}' has {len(files)} file(s).")
    return True

def main():
    current_dir = Path(__file__).parent
    paddle_dir = current_dir / "paddle_ocr"
    svtr_dir = current_dir / "svtr_v6_ocr"

    expected_paddle_files = []  
    expected_svtr_files = []    

    print("🔍 Checking model files in PaddleOCR and SVTR v6 directories...\n")
    paddle_ok = check_model_directory(paddle_dir, expected_paddle_files)
    svtr_ok = check_model_directory(svtr_dir, expected_svtr_files)

    if paddle_ok and svtr_ok:
        print("\n✅ All model directories have the required files.")
    else:
        print("\n❌ Some model directories are missing required files.")

if __name__ == "__main__":
    main()