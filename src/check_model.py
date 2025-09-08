#!/usr/bin/env python3
"""
Model Check Script for OCR Pipeline

This script checks that model directories for the OCR Model contain files.
Customize expected_files nếu cần kiểm tra tên file bắt buộc.
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
    current_dir = Path(__file__).parent.parent
    dbnet_dir = current_dir / "dbnet"
    svtr_dir = current_dir / "svtr"

    expected_paddle_files = ["inference.yml", "inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel"]  
    expected_svtr_files = ["inference.yml", "inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel"]    

    print("🔍 Checking model files in DBNet and SVTR directories...\n")
    paddle_ok = check_model_directory(dbnet_dir, expected_paddle_files)
    svtr_ok = check_model_directory(svtr_dir, expected_svtr_files)

    if paddle_ok and svtr_ok:
        print("\n✅ All model directories have the required files.")
    else:
        print("\n❌ Some model directories are missing required files.")

if __name__ == "__main__":
    main()