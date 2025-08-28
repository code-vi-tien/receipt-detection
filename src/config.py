#!/usr/bin/env python3
"""
Configuration for OCR Pipeline
"""

from pathlib import Path
from dataclasses import dataclass

@dataclass
class BillOCRConfig:
    """Configuration for Bill OCR Pipeline"""
    
    # Base paths
    base_dir: Path = Path(__file__).parent
    
    # YOLO Detection
    yolo_model_path: Path = base_dir / "yolo_detect_bill" / "bill_models.pt"
    yolo_confidence_threshold: float = 0.1
    
    # SVTR v6 OCR
    svtr_model_dir: Path = base_dir / "svtr_v6_ocr"
    svtr_model_file: str = "best_accuracy.pdparams"
    svtr_config_file: str = "inference.yml"
    
    # PaddleOCR
    paddle_det_model_dir: Path = base_dir / "paddle_ocr" / "ch_db_res18"
    paddle_lang: str = "ch"
    paddle_use_gpu: bool = False
    
    # Processing
    image_test_dir: Path = base_dir / "image_test"
    
    # Output settings
    save_images: bool = True
    save_crops: bool = True
    save_comparison: bool = True
    

# Default configuration instance
default_config = BillOCRConfig()
