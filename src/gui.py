#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete OCR Pipeline with PySide6 GUI
Integrated YOLO Detection + OCR Model + PaddleOCR comparison
"""

import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import time
from typing import Dict, List, Any, Optional
import traceback

# PySide6 imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
        QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
        QProgressBar, QTextEdit, QSplitter, QGroupBox, QGridLayout,
        QFileDialog, QTabWidget, QScrollArea, QFrame, QMessageBox
    )
    from PySide6.QtCore import Qt, QThread, Signal, QTimer
    from PySide6.QtGui import QPixmap, QFont, QColor, QPalette, QImage
    HAS_PYSIDE6 = True
    print("✅ PySide6 imported successfully")
except ImportError as e:
    print(f"❌ PySide6 import failed: {e}")
    print("Please install: pip install PySide6")
    HAS_PYSIDE6 = False

# Import our modules - direct imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from yolo_detect_bill.bill_detector import BillDetector
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    print("Make sure all model modules are in place")
    sys.exit(1)


class CustomOCREngine:
    """Custom OCR Engine using DBNet + SVTR pipeline (matching benchmark_ocr.py)"""
    
    def __init__(self):
        self.ocr_engine = None
        self.baseline_engine = None
        self.initialized = False
    
    def initialize(self):
        """Initialize DBNet detection and SVTR recognition engines"""
        try:
            from paddleocr import PaddleOCR
            
            # Use the custom detection model path
            det_model_path = str("..\dbnet\model")
            rec_model_path = str("..\svtr\model")
            
            self.ocr_engine = PaddleOCR(
                det_model_dir=det_model_path,
                rec_model_dir=rec_model_path,
                use_angle_cls=False,
                use_gpu=False,
                lang='en'
            )

            self.baseline_engine = PaddleOCR(
                use_angle_cls=False,
                use_gpu=False,
                lang='en',
                show_log=False
            )

            self.initialized = True
            return True
        except Exception as e:
            print(f"❌ Failed to initialize PaddleOCR: {e}")
            return False
    
    def predict_ocr(self, image_np: np.ndarray) -> List[Dict]:
        """Predict text from image array"""
        if not self.initialized and not self.initialize():
            return {
                'model_name': 'OCR Model',
                'texts': [], 
                'total_texts': 0, 
                'avg_confidence': 0, 
                'processing_time': 0,
                'error': 'OCR engine not initialized'
            }
        
        try:
            # Convert numpy array to image path temporarily
            temp_path = f"temp_ocr_input_{int(time.time())}.jpg"  # Added timestamp to avoid conflicts
            cv2.imwrite(temp_path, image_np)
            
            start_time = time.time()
            result = self.ocr_engine.ocr(temp_path, cls=False)
            processing_time = time.time() - start_time

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            texts = self._parse_result(result, "OCR Model")
            
            return {
                'model_name': 'OCR Model',
                'texts': texts,
                'total_texts': len(texts),
                'processing_time': processing_time,
                'avg_confidence': np.mean([t['confidence'] for t in texts]) if texts else 0,
                'raw_result': texts
            }
            
        except Exception as e:
            print(f"❌ OCR Model prediction failed: {e}")
            return {
                'model_name': 'OCR Model',
                'texts': [],
                'total_texts': 0,
                'processing_time': 0,
                'avg_confidence': 0,
                'error': str(e)
            }

    def predict_baseline(self, image_np: np.ndarray) -> List[Dict]:
        """Predict text using baseline PaddleOCR"""
        if not self.initialized and not self.initialize():
            return {
                'model_name': 'PaddleOCR',
                'texts': [], 
                'total_texts': 0, 
                'avg_confidence': 0, 
                'processing_time': 0,
                'error': 'Baseline engine not initialized'
            }
        
        try:
            # Save image temporarily
            temp_path = f"temp_baseline_input_{int(time.time())}.jpg"  # Added timestamp
            cv2.imwrite(temp_path, image_np)
            
            start_time = time.time()
            result = self.baseline_engine.ocr(temp_path, cls=False)  # Fixed variable name
            processing_time = time.time() - start_time
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            texts = self._parse_result(result, "PaddleOCR")
            
            return {
                'model_name': 'PaddleOCR',
                'texts': texts,
                'total_texts': len(texts),
                'processing_time': processing_time,
                'avg_confidence': np.mean([t['confidence'] for t in texts]) if texts else 0,
                'raw_result': texts
            }
            
        except Exception as e:
            print(f"❌ Baseline OCR prediction failed: {e}")
            return {
                'model_name': 'PaddleOCR',
                'texts': [],
                'total_texts': 0,
                'processing_time': 0,
                'avg_confidence': 0,
                'error': str(e)
            }

    def _parse_result(self, result, model_name: str) -> List[Dict]:
        """Parse PaddleOCR result into standardized format"""
        output_data = []
        
        if result and len(result[0]) > 0:
            for line in result[0]:
                coords = line[0]  # Bounding box coordinates
                text_info = line[1]  # (text, confidence)
                text = text_info[0]
                confidence = text_info[1]
                
                # Create bounding box format for consistency
                bbox = [
                    min([p[0] for p in coords]),  # x1
                    min([p[1] for p in coords]),  # y1
                    max([p[0] for p in coords]),  # x2
                    max([p[1] for p in coords])   # y2
                ]
                
                output_data.append({
                    "text": text,
                    "confidence": confidence,
                    "coordinates": coords,
                    "bbox": bbox,  # Added for UI compatibility
                    "model": model_name
                })
        
        return output_data
    
class OCRProcessingThread(QThread):
    """Background thread for OCR processing (fixed to match benchmark_ocr.py)"""
    progress_updated = Signal(str)
    yolo_completed = Signal(dict)
    ocr_completed = Signal(dict) 
    baseline_completed = Signal(dict)
    processing_completed = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self, image_path: str, yolo_detector, ocr_engine):
        super().__init__()
        self.image_path = image_path
        self.yolo_detector = yolo_detector
        self.ocr_engine = ocr_engine
        
    def run(self):
        """Main processing pipeline (matching benchmark_ocr.py logic)"""
        try:
            results = {
                'input_image': self.image_path,
                'timestamp': datetime.now().isoformat(),
                'yolo_detection': None,
                'ocr_results': None,
                'baseline_results': None,
                'comparison': None
            }
            
            # Step 1: YOLO Detection
            self.progress_updated.emit("🔍 YOLO: Detecting bill regions...")
            yolo_results = self._yolo_detection()
            results['yolo_detection'] = yolo_results
            self.yolo_completed.emit(yolo_results)
            
            if not yolo_results['detections']:
                self.error_occurred.emit("❌ No bills detected in image")
                return
            
            # Get best detection
            best_detection = max(yolo_results['detections'], key=lambda x: x['confidence'])
            bill_crop = yolo_results['bill_crop']
            
            self.progress_updated.emit(f"✅ Best bill found (confidence: {best_detection['confidence']:.3f})")
            
            # Step 2: OCR Model Processing
            self.progress_updated.emit("🤖 OCR Model: Processing bill text...")
            ocr_results = self._ocr_processing(bill_crop)
            results['ocr_results'] = ocr_results
            self.ocr_completed.emit(ocr_results)
            
            # Step 3: PaddleOCR Processing
            self.progress_updated.emit("🧠 PaddleOCR: Processing bill text...")
            baseline_results = self._baseline_processing(bill_crop)
            results['baseline_results'] = baseline_results
            self.baseline_completed.emit(baseline_results)
            
            # Step 4: Comparison
            self.progress_updated.emit("📊 Generating comparison...")
            comparison = self._generate_comparison(ocr_results, baseline_results)
            results['comparison'] = comparison
            
            # Step 5: Save results
            self.progress_updated.emit("💾 Saving results...")
            self._save_results(results)
            
            self.progress_updated.emit("✅ Processing completed!")
            self.processing_completed.emit(results)
            
        except Exception as e:
            error_msg = f"❌ Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def _yolo_detection(self) -> Dict:
        """YOLO bill detection (same as before)"""
        try:
            # Load image
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Run YOLO detection
            detections = self.yolo_detector.detect_bills_from_frame(image, confidence_threshold=0.1)
            
            if not detections:
                return {
                    'detections': [],
                    'bill_crop': None,
                    'annotated_image': image
                }
            
            # Get best detection (highest confidence)
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # Crop bill region
            x1, y1, x2, y2 = (
                int(best_detection['x1']), 
                int(best_detection['y1']),
                int(best_detection['x2']), 
                int(best_detection['y2'])
            )
            
            # Add padding
            padding = 10
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            bill_crop = image[y1:y2, x1:x2]
            
            # Create annotated image
            annotated_image = image.copy()
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(annotated_image, f"Bill: {best_detection['confidence']:.3f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return {
                'detections': detections,
                'best_detection': best_detection,
                'bill_crop': bill_crop,
                'annotated_image': annotated_image,
                'crop_coordinates': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            }
            
        except Exception as e:
            raise Exception(f"YOLO detection failed: {e}")
    
    def _ocr_processing(self, bill_crop: np.ndarray) -> Dict:
        """Custom OCR Model text recognition"""
        try:
            # Run OCR Model prediction
            ocr_result = self.ocr_engine.predict_ocr(bill_crop)
            return ocr_result
        
        except Exception as e:
            return {
                'model_name': 'OCR Model',
                'texts': [],
                'total_texts': 0,
                'avg_confidence': 0,
                'error': str(e)
            }
    
    def _baseline_processing(self, bill_crop: np.ndarray) -> Dict:
        """PaddleOCR text recognition"""
        try:
            baseline_results = self.ocr_engine.predict_baseline(bill_crop)
            return baseline_results
            
        except Exception as e:
            return {
                'model_name': 'Baseline PaddleOCR', 
                'texts': [],
                'total_texts': 0,
                'processing_time': 0,
                'avg_confidence': 0,
                'error': str(e)
            }
    
    def _generate_comparison(self, ocr_results: Dict, baseline_results: Dict) -> Dict:
        """Generate comparison between models"""
        ocr_texts = ocr_results.get('texts', [])
        baseline_texts = baseline_results.get('texts', [])
        
        # Calculate statistics
        ocr_confidences = [t.get('confidence', 0) for t in ocr_texts]
        baseline_confidences = [t.get('confidence', 0) for t in baseline_texts]
        
        return {
            'ocr_stats': {
                'total_texts': len(ocr_texts),
                'avg_confidence': np.mean(ocr_confidences) if ocr_confidences else 0,
                'max_confidence': max(ocr_confidences) if ocr_confidences else 0,
                'min_confidence': min(ocr_confidences) if ocr_confidences else 0,
                'high_confidence_count': sum(1 for c in ocr_confidences if c > 0.9),
                'very_high_confidence_count': sum(1 for c in ocr_confidences if c > 0.95),
                'good_confidence_count': sum(1 for c in ocr_confidences if c > 0.8), 
                'low_confidence_count': sum(1 for c in ocr_confidences if c < 0.5),
                'confidence_std': np.std(ocr_confidences),
                'confidence_median': np.median(ocr_confidences)
                
            },
            'baseline_stats': {
                'total_texts': len(baseline_texts),
                'avg_confidence': np.mean(baseline_confidences) if baseline_confidences else 0,
                'max_confidence': max(baseline_confidences) if baseline_confidences else 0,
                'min_confidence': min(baseline_confidences) if baseline_confidences else 0,
                'high_confidence_count': sum(1 for c in baseline_confidences if c > 0.9),
                'very_high_confidence_count': sum(1 for c in baseline_confidences if c > 0.95),
                'good_confidence_count': sum(1 for c in baseline_confidences if c > 0.8), 
                'low_confidence_count': sum(1 for c in baseline_confidences if c < 0.5),
                'confidence_std': np.std(baseline_confidences),
                'confidence_median': np.median(baseline_confidences)
            }
        }
    
    def _save_results(self, results: Dict):
        """Save results to JSON in folder gui_result (same as before)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(self.image_path).stem
        filename = f"ocr_pipeline_results_{base_name}_{timestamp}.json"
        
        # Create folder 'gui_result' if it does not exist
        results_folder = Path(__file__).parent.parent / "gui_result"
        results_folder.mkdir(exist_ok=True) 
            
        file_path = results_folder / filename
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

class OCRPipelineGUI(QMainWindow):
    """Main GUI application (fixed to match benchmark_ocr.py architecture)"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Receipt OCR System – DBNet + SVTR vs PaddleOCR")
        self.setGeometry(50, 50, 1700, 1000)
        
        # Initialize engines (matching benchmark_ocr.py)
        self.yolo_detector = None
        self.ocr_engine = None
        
        # Processing thread
        self.processing_thread = None
        
        # Current results
        self.current_results = None
        
        # Set modern styling
        self.setStyleSheet(self.get_modern_stylesheet())
        
        self.init_ui()
        self.init_engines()
    
    def get_modern_stylesheet(self) -> str:
        """Get modern dark theme stylesheet"""
        return """
        QMainWindow {
            background-color: #2b2b2b;
            color: #ffffff;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
            background-color: #3a3a3a;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            color: #4CAF50;
            font-size: 14px;
            font-weight: bold;
        }
        
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            font-size: 14px;
            font-weight: bold;
            margin: 4px 2px;
            border-radius: 6px;
            min-height: 25px;
        }
        
        QPushButton:hover {
            background-color: #45a049;
            transform: translateY(-1px);
        }
        
        QPushButton:pressed {
            background-color: #3d8b40;
        }
        
        QPushButton:disabled {
            background-color: #666666;
            color: #999999;
        }
        
        QLabel {
            color: #ffffff;
            font-size: 13px;
        }
        
        QTabWidget::pane {
            border: 1px solid #555555;
            border-radius: 6px;
            background-color: #3a3a3a;
        }
        
        QTabBar::tab {
            background-color: #4a4a4a;
            color: #ffffff;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-weight: bold;
        }
        
        QTabBar::tab:selected {
            background-color: #4CAF50;
        }
        
        QTabBar::tab:hover {
            background-color: #5a5a5a;
        }
        
        QTableWidget {
            gridline-color: #555555;
            background-color: #3a3a3a;
            alternate-background-color: #404040;
            selection-background-color: #4CAF50;
            border: 1px solid #555555;
            border-radius: 6px;
        }
        
        QTableWidget::item {
            padding: 8px;
            border-bottom: 1px solid #555555;
        }
        
        QHeaderView::section {
            background-color: #4a4a4a;
            color: #ffffff;
            padding: 10px;
            border: 1px solid #555555;
            font-weight: bold;
        }
        
        QTextEdit {
            background-color: #1e1e1e;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 6px;
            padding: 8px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
        }
        
        QProgressBar {
            border: 2px solid #555555;
            border-radius: 6px;
            text-align: center;
            background-color: #3a3a3a;
        }
        
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 4px;
        }
        
        QScrollArea {
            border: 1px solid #555555;
            border-radius: 6px;
            background-color: #3a3a3a;
        }
        
        QFrame[frameShape="1"] {
            border: 1px solid #555555;
        }
        """
    
    def init_ui(self):
        """Initialize UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
    def init_ui(self):
        """Initialize modern UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with margins
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Header section (compact)
        header_layout = self.create_header_section()
        main_layout.addLayout(header_layout)
        
        # Main content area (80% of space)
        main_content = self.create_main_content_area()
        main_layout.addWidget(main_content, stretch=8)
        
        # Footer section (compact)
        footer_layout = self.create_footer_section()
        main_layout.addLayout(footer_layout)
    
    def create_header_section(self) -> QHBoxLayout:
        """Create compact header with title and controls"""
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)
        
        # App title (compact)
        title_label = QLabel("OCR-based Receipt Recognition System")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #4CAF50; margin: 5px;")
        header_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Detected YOLO • OCR Model • PaddleOCR")
        subtitle_label.setStyleSheet("color: #888888; font-size: 12px; margin: 5px;")
        header_layout.addWidget(subtitle_label)
        
        header_layout.addStretch()
        
        # Control buttons (inline) - Increased width
        self.select_btn = QPushButton("📁 Select Image")
        self.select_btn.setFixedSize(150, 35)
        self.select_btn.clicked.connect(self.select_image)
        header_layout.addWidget(self.select_btn)
        
        self.process_btn = QPushButton("🚀 Processing")
        self.process_btn.setFixedSize(150, 35)
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_image)
        header_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("💾 Export Results")
        self.save_btn.setFixedSize(150, 35)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        header_layout.addWidget(self.save_btn)
        
        return header_layout
    
    def create_main_content_area(self) -> QWidget:
        """Create main content area with focus on detection and results"""
        # Three-panel layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(8)
        
        # Left panel: Image processing (35%)
        image_panel = self.create_enhanced_image_panel()
        main_splitter.addWidget(image_panel)
        
        # Center panel: YOLO Detection (35%)
        detection_panel = self.create_detection_panel()
        main_splitter.addWidget(detection_panel)
        
        # Right panel: OCR Results (30%)
        results_panel = self.create_enhanced_results_panel()
        main_splitter.addWidget(results_panel)
        
        # Set proportions
        main_splitter.setSizes([550, 550, 500])
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555555;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
            }
        """)
        
        return main_splitter
    
    def create_enhanced_image_panel(self) -> QWidget:
        """Create enhanced image panel"""
        panel = QGroupBox("📷 Input Image")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Current image status
        self.current_image_label = QLabel("No image selected")
        self.current_image_label.setStyleSheet("""
            background-color: #4a4a4a; 
            padding: 8px; 
            border-radius: 4px;
            color: #cccccc;
            font-weight: bold;
        """)
        layout.addWidget(self.current_image_label)
        
        # Image display with tabs
        self.image_tabs = QTabWidget()
        self.image_tabs.setTabPosition(QTabWidget.South)
        
        # Original image
        self.original_image_label = QLabel("Drag and drop an image or click Select Imag")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumHeight(350)
        self.original_image_label.setStyleSheet("""
            border: 3px dashed #666666; 
            background-color: #333333;
            border-radius: 8px;
            color: #888888;
            font-size: 14px;
        """)
        
        original_scroll = QScrollArea()
        original_scroll.setWidget(self.original_image_label)
        original_scroll.setWidgetResizable(True)
        self.image_tabs.addTab(original_scroll, "Original image")
        
        # Bill crop
        self.crop_image_label = QLabel("The receipt area will be displayed here")
        self.crop_image_label.setAlignment(Qt.AlignCenter)
        self.crop_image_label.setStyleSheet("""
            border: 2px solid #555555; 
            background-color: #333333;
            border-radius: 8px;
            color: #888888;
        """)
        
        crop_scroll = QScrollArea()
        crop_scroll.setWidget(self.crop_image_label)
        crop_scroll.setWidgetResizable(True)
        self.image_tabs.addTab(crop_scroll, "Receipt area")
        
        layout.addWidget(self.image_tabs)
        return panel
    
    def create_detection_panel(self) -> QWidget:
        """Create YOLO detection visualization panel"""
        panel = QGroupBox("🎯 YOLO Detection Analysis")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Detection status
        self.detection_status_label = QLabel("Detection Ready")
        self.detection_status_label.setStyleSheet("""
            background-color: #4a4a4a; 
            padding: 8px; 
            border-radius: 4px;
            color: #cccccc;
            font-weight: bold;
        """)
        layout.addWidget(self.detection_status_label)
        
        # YOLO detection display with scroll
        self.yolo_image_label = QLabel("YOLO detection results will be displayed here")
        self.yolo_image_label.setAlignment(Qt.AlignCenter)
        self.yolo_image_label.setMinimumHeight(450)
        self.yolo_image_label.setStyleSheet("""
            border: 2px solid #555555; 
            background-color: #333333;
            border-radius: 8px;
            color: #888888;
            font-size: 14px;
        """)
        
        yolo_scroll = QScrollArea()
        yolo_scroll.setWidget(self.yolo_image_label)
        yolo_scroll.setWidgetResizable(True)
        yolo_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        yolo_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        layout.addWidget(yolo_scroll)
        
        # Enhanced detection statistics
        self.detection_stats_label = QLabel("Detection statistics will be displayed here")
        self.detection_stats_label.setWordWrap(True)
        self.detection_stats_label.setMaximumHeight(200)
        self.detection_stats_label.setStyleSheet("""
            background-color: #1e1e1e; 
            padding: 10px; 
            border-radius: 6px;
            color: #cccccc;
            font-family: 'Consolas', monospace;
            font-size: 12px;
        """)
        
        # Add scroll for stats
        stats_scroll = QScrollArea()
        stats_scroll.setWidget(self.detection_stats_label)
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMaximumHeight(200)
        layout.addWidget(stats_scroll)
        
        return panel
    
    def create_enhanced_results_panel(self) -> QWidget:
        """Create enhanced OCR results panel (updated naming)"""
        panel = QGroupBox("📊 OCR Analysis Results")
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # Processing status
        self.processing_status_label = QLabel("OCR processing ready")
        self.processing_status_label.setStyleSheet("""
            background-color: #4a4a4a; 
            padding: 8px; 
            border-radius: 4px;
            color: #cccccc;
            font-weight: bold;
        """)
        layout.addWidget(self.processing_status_label)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Model comparison tab
        comparison_widget = self.create_enhanced_comparison_tab()
        self.results_tabs.addTab(comparison_widget, "📊 Model Comparison")
        
        # ocr details tab
        ocr_widget = self.create_enhanced_detail_tab("OCR Model")
        self.results_tabs.addTab(ocr_widget, "🤖 OCR Model Details")
        
        # PaddleOCR details tab
        baseline_widget = self.create_enhanced_detail_tab("PaddleOCR")
        self.results_tabs.addTab(baseline_widget, "🧠 PaddleOCR Details")
        
        # Performance analysis tab
        performance_widget = self.create_performance_analysis_tab()
        self.results_tabs.addTab(performance_widget, "⚡ Performance Analysis")
        
        layout.addWidget(self.results_tabs)
        return panel
    
    def create_enhanced_comparison_tab(self) -> QWidget:
        """Create enhanced comparison tab (updated naming)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Summary metrics display
        self.summary_label = QLabel("Processing results will be displayed here")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("""
            background-color: #1e1e1e; 
            padding: 15px; 
            border-radius: 8px;
            color: #ffffff;
            font-size: 13px;
            line-height: 1.4;
        """)
        layout.addWidget(self.summary_label)
        
        # Enhanced comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["Metrics", "OCR Model", "PaddleOCR"])
        self.comparison_table.setAlternatingRowColors(True)
        self.comparison_table.verticalHeader().setVisible(False)
        self.comparison_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.comparison_table)
        
        return widget
    
    def create_performance_analysis_tab(self) -> QWidget:
        """Create detailed performance analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Performance overview
        self.performance_overview_label = QLabel("Performance overview will be displayed here")
        self.performance_overview_label.setWordWrap(True)
        self.performance_overview_label.setStyleSheet("""
            background-color: #1e1e1e; 
            padding: 15px; 
            border-radius: 8px;
            color: #ffffff;
            font-size: 13px;
            line-height: 1.4;
        """)
        layout.addWidget(self.performance_overview_label)
        
        # Detailed metrics table
        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(4)
        self.performance_table.setHorizontalHeaderLabels([
            "Metrics", "OCR Model", "PaddleOCR", "Evaluation"
        ])
        self.performance_table.setAlternatingRowColors(True)
        self.performance_table.verticalHeader().setVisible(False)
        self.performance_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.performance_table)
        
        # Model architecture comparison
        self.architecture_label = QLabel("Model architecture comparison will be displayed here")
        self.architecture_label.setWordWrap(True)
        self.architecture_label.setStyleSheet("""
            background-color: #2a2a2a; 
            padding: 12px; 
            border-radius: 6px;
            color: #cccccc;
            font-size: 12px;
            line-height: 1.3;
        """)
        layout.addWidget(self.architecture_label)
        
        return widget
    
    def create_enhanced_detail_tab(self, model_name: str) -> QWidget:
        """Create enhanced detailed results tab (updated naming)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Model info header
        model_info_label = QLabel(f"Recognition Results {model_name}")
        model_info_label.setStyleSheet("""
            background-color: #4CAF50; 
            color: white; 
            padding: 8px; 
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        """)
        layout.addWidget(model_info_label)
        
        # Enhanced results table
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["ID", "Text", "Conf", "Evaluate"])
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSortingEnabled(True)
        
        # Store reference
        if model_name == "OCR Model":
            self.ocr_table = table
        else:
            self.baseline_table = table
        
        layout.addWidget(table)
        return widget
    
    def create_footer_section(self) -> QHBoxLayout:
        """Create compact footer section"""
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        footer_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to process image")
        self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        footer_layout.addWidget(self.status_label)
        
        footer_layout.addStretch()
        
        # Log toggle button
        self.log_toggle_btn = QPushButton("📋 Logs")
        self.log_toggle_btn.setFixedSize(120, 25)
        self.log_toggle_btn.clicked.connect(self.toggle_logs)
        footer_layout.addWidget(self.log_toggle_btn)
        
        return footer_layout
    
    def toggle_logs(self):
        """Toggle log display"""
        if not hasattr(self, 'log_widget'):
            self.create_log_widget()
        
        if self.log_widget.isVisible():
            self.log_widget.hide()
            self.log_toggle_btn.setText("📋 Show Logs")
        else:
            self.log_widget.show()
            self.log_toggle_btn.setText("📋 Hide Logs")
    
    def create_log_widget(self):
        """Create log widget as popup"""
        self.log_widget = QWidget()
        self.log_widget.setWindowTitle("Processing log")
        self.log_widget.setGeometry(200, 200, 800, 300)
        self.log_widget.setStyleSheet(self.get_modern_stylesheet())
        
        layout = QVBoxLayout(self.log_widget)
        
        self.log_text = QTextEdit()
        self.log_text.setPlaceholderText("Processing log will be displayed here")
        layout.addWidget(self.log_text)
    
    def create_control_panel(self) -> QWidget:
        """Create control panel"""
        panel = QGroupBox("📁 Input Controls")
        layout = QHBoxLayout(panel)
        
        # Select image button
        self.select_btn = QPushButton("📁 Select Image")
        self.select_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_btn)
        
        # Current image label
        self.current_image_label = QLabel("No image selected")
        layout.addWidget(self.current_image_label)
        
        layout.addStretch()
        
        # Process button
        self.process_btn = QPushButton("🚀 Process OCR")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self.process_image)
        layout.addWidget(self.process_btn)
        
        # Save results button
        self.save_btn = QPushButton("💾 Save Results")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_results)
        layout.addWidget(self.save_btn)
        
        return panel
    
    def create_image_panel(self) -> QWidget:
        """Create image display panel"""
        panel = QGroupBox("🖼️ Images")
        layout = QVBoxLayout(panel)
        
        # Tab widget for different images
        self.image_tabs = QTabWidget()
        
        # Original image tab
        self.original_image_label = QLabel("Select an image to begin")
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setMinimumHeight(200)
        self.original_image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        
        original_scroll = QScrollArea()
        original_scroll.setWidget(self.original_image_label)
        original_scroll.setWidgetResizable(True)
        self.image_tabs.addTab(original_scroll, "📷 Original")
        
        # YOLO detection tab
        self.yolo_image_label = QLabel("YOLO detection will appear here")
        self.yolo_image_label.setAlignment(Qt.AlignCenter)
        self.yolo_image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        
        yolo_scroll = QScrollArea()
        yolo_scroll.setWidget(self.yolo_image_label)
        yolo_scroll.setWidgetResizable(True)
        self.image_tabs.addTab(yolo_scroll, "🎯 YOLO Detection")
        
        # Bill crop tab
        self.crop_image_label = QLabel("Bill crop will appear here")
        self.crop_image_label.setAlignment(Qt.AlignCenter)
        self.crop_image_label.setStyleSheet("border: 2px dashed #ccc; background-color: #f9f9f9;")
        
        crop_scroll = QScrollArea()
        crop_scroll.setWidget(self.crop_image_label)
        crop_scroll.setWidgetResizable(True)
        self.image_tabs.addTab(crop_scroll, "✂️ Bill Crop")
        
        layout.addWidget(self.image_tabs)
        return panel
    
    def create_results_panel(self) -> QWidget:
        """Create results display panel"""
        panel = QGroupBox("📊 OCR Results")
        layout = QVBoxLayout(panel)
        
        # Results tabs
        self.results_tabs = QTabWidget()
        
        # Comparison tab
        comparison_widget = self.create_comparison_tab()
        self.results_tabs.addTab(comparison_widget, "📊 Comparison")
        
        # ocr details tab
        ocr_widget = self.create_detail_tab("OCR Model")
        self.results_tabs.addTab(ocr_widget, "🤖 OCR Model Details")
        
        # PaddleOCR details tab
        baseline_widget = self.create_detail_tab("PaddleOCR")
        self.results_tabs.addTab(baseline_widget, "🧠 PaddleOCR Details")
        
        layout.addWidget(self.results_tabs)
        return panel
    
    def create_comparison_tab(self) -> QWidget:
        """Create comparison table tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary stats
        self.summary_label = QLabel("Processing results will be displayed here")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 15px;
                font-size: 12px;
                color: #ffffff;
            }
        """)
        layout.addWidget(self.summary_label)
        
        # Comparison table
        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(3)
        self.comparison_table.setHorizontalHeaderLabels(["📊 Metrics", "🤖 OCR Model", "🧠 PaddleOCR"])
        layout.addWidget(self.comparison_table)
        
        return widget
    
    def create_detail_tab(self, model_name: str) -> QWidget:
        """Create detailed results tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model info header
        model_info_label = QLabel(f"Recognition Results {model_name}")
        model_info_label.setStyleSheet("""
            background-color: #4CAF50; 
            color: white; 
            padding: 8px; 
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        """)
        layout.addWidget(model_info_label)

        # Create table for this model
        table = QTableWidget()
        table.setColumnCount(5)  # Fixed: ensure 5 columns
        table.setHorizontalHeaderLabels(["ID", "Text", "Conf", "Coords", "Evaluate"])
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSortingEnabled(True)

        # Store reference
        if model_name == "OCR Model":
            self.ocr_table = table
        else:
            self.baseline_table = table
        
        layout.addWidget(table)
        return widget
    
    def init_engines(self):
        """Initialize OCR engines (matching benchmark_ocr.py)"""
        self.log("🚀 Initializing OCR engines...")
        
        try:
            # Initialize YOLO detector
            yolo_model_path = Path(__file__).parent.parent / "yolo_detect_bill" / "bill_models.pt"
            self.yolo_detector = BillDetector(model_path=str(yolo_model_path))
            if self.yolo_detector.load_model():
                self.log("✅ YOLO detector loaded")
            else:
                self.log("❌ Failed to load YOLO detector")
            
            # Initialize PaddleOCR Engine (handles both custom and baseline)
            self.ocr_engine = PaddleOCREngine()
            if self.ocr_engine.initialize():
                self.log("✅ PaddleOCR engine loaded")
            else:
                self.log("❌ Failed to load PaddleOCR engine")

            self.log("🎉 All engines initialized successfully!")
            
        except Exception as e:
            self.log(f"❌ Engine initialization failed: {e}")
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Create log widget if needed
        if not hasattr(self, 'log_text'):
            self.create_log_widget()
        
        self.log_text.append(log_message)
        
        # Also update status
        self.status_label.setText(message)
        
        # Update processing status if relevant
        if "YOLO" in message:
            self.detection_status_label.setText(message)
        elif "ocr" in message or "PaddleOCR" in message or "Xử lý" in message:
            self.processing_status_label.setText(message)
        
        # Auto scroll to bottom
        if hasattr(self, 'log_text'):
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.log_text.setTextCursor(cursor)
        
        # Process events to update UI
        QApplication.processEvents()
    
    def select_image(self):
        """Select image file"""
        image_test_dir = Path(__file__).parent.parent / "image_test"
        print("DEBUG >> Default image_test folder:", image_test_dir)

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select imag to process",
            str(image_test_dir),  # trỏ đến image_test cùng cấp src
            "Tệp Ảnh (*.jpg *.jpeg *.png *.bmp);;All files (*)"
        )
        
        if file_path:
            self.current_image_path = file_path
            filename = Path(file_path).name
            self.current_image_label.setText(f"📁 {filename}")
            self.process_btn.setEnabled(True)
            
            # Display original image
            self.display_image(file_path, self.original_image_label, max_size=600)
            self.log(f"📁 Image uploaded: {filename}")
            
            # Switch to original tab
            self.image_tabs.setCurrentIndex(0)
    
    def display_image(self, image_path: str, label: QLabel, max_size: int = 600):
        """Display image in label with enhanced styling"""
        try:
            if isinstance(image_path, str):
                pixmap = QPixmap(image_path)
            else:
                # numpy array
                height, width, channel = image_path.shape
                bytes_per_line = 3 * width
                q_image = QImage(image_path.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap
            scaled_pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)
            
            # Update label style for successful image display
            label.setStyleSheet("""
                border: 2px solid #4CAF50; 
                background-color: #333333;
                border-radius: 8px;
                padding: 5px;
            """)
            
        except Exception as e:
            label.setText(f"❌ Image display erro: {e}")
            label.setStyleSheet("""
                border: 2px solid #f44336; 
                background-color: #333333;
                border-radius: 8px;
                color: #f44336;
            """)
    
    def process_image(self):
        """Start image processing (updated for custom vs baseline)"""
        if not hasattr(self, 'current_image_path'):
            QMessageBox.warning(self, "Warning", "Please select an image first")
            return
        
        # Disable UI during processing
        self.process_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        self.log("🚀 Starting OCR pipeline...")
        
        # Create and start processing thread
        self.processing_thread = OCRProcessingThread(
            self.current_image_path,
            self.yolo_detector,
            self.ocr_engine
        )
        
        # Connect signals (updated signal names)
        self.processing_thread.progress_updated.connect(self.log)
        self.processing_thread.yolo_completed.connect(self.on_yolo_completed)
        self.processing_thread.ocr_completed.connect(self.on_ocr_completed)
        self.processing_thread.baseline_completed.connect(self.on_baseline_completed)
        self.processing_thread.processing_completed.connect(self.on_processing_completed)
        self.processing_thread.error_occurred.connect(self.on_error_occurred)
        
        self.processing_thread.start()
    
    def on_yolo_completed(self, yolo_results: Dict):
        """Handle YOLO completion with enhanced visualization"""
        try:
            # Display YOLO detection image
            if 'annotated_image' in yolo_results:
                self.display_image_array(yolo_results['annotated_image'], self.yolo_image_label, max_size=800)
            
            # Display bill crop
            if 'bill_crop' in yolo_results and yolo_results['bill_crop'] is not None:
                self.display_image_array(yolo_results['bill_crop'], self.crop_image_label, max_size=400)
                # Switch to bill crop tab
                self.image_tabs.setCurrentIndex(1)
            
            # Update enhanced detection statistics
            detections = yolo_results.get('detections', [])
            best_detection = yolo_results.get('best_detection', {})
            crop_coords = yolo_results.get('crop_coordinates', {})
            
            # Calculate additional metrics
            total_detections = len(detections)
            avg_confidence = sum(d.get('confidence', 0) for d in detections) / total_detections if total_detections > 0 else 0
            high_conf_count = sum(1 for d in detections if d.get('confidence', 0) > 0.8)
            
            # Get image dimensions
            if 'bill_crop' in yolo_results and yolo_results['bill_crop'] is not None:
                crop_shape = yolo_results['bill_crop'].shape
                crop_area = crop_shape[0] * crop_shape[1]
            else:
                crop_shape = (0, 0, 0)
                crop_area = 0
            
            stats_text = f"""
🎯 <b>YOLO DETECTION RESULTS</b><br>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>
<br>
📊 <b>DETECTION STATISTICS:</b><br>
   • Total detected receipts: <span style="color: #4CAF50;"><b>{total_detections}</b></span><br>
   • Average confidence: <span style="color: #2196F3;"><b>{avg_confidence:.3f}</b></span><br>
   • High-confidence detection (>0.8): <span style="color: #FF9800;"><b>{high_conf_count}/{total_detections}</b></span><br>
<br>
🏆 <b>BEST DETECTION:</b><br>
   • Confidence: <span style="color: #4CAF50;"><b>{best_detection.get('confidence', 0):.3f}</b></span><br>
   • Top-left coordinates: <span style="color: #9C27B0;">({int(best_detection.get('x1', 0))}, {int(best_detection.get('y1', 0))})</span><br>
   • Bottom-right coordinates: <span style="color: #9C27B0;">({int(best_detection.get('x2', 0))}, {int(best_detection.get('y2', 0))})</span><br>
   • Region size”: <span style="color: #FF5722;">{int(best_detection.get('x2', 0)) - int(best_detection.get('x1', 0))} × {int(best_detection.get('y2', 0)) - int(best_detection.get('y1', 0))} pixels</span><br>
<br>
✂️ <b>CROPPED REGION INFORMATION:</b><br>
   • Cropped size: <span style="color: #607D8B;"><b>{crop_shape[1]} × {crop_shape[0]} pixels</b></span><br>
   • Color channels: <span style="color: #795548;">{crop_shape[2] if len(crop_shape) > 2 else 'N/A'}</span><br>
   • Cropped area: <span style="color: #3F51B5;"><b>{crop_area:,} pixels²</b></span><br>
   • Aspect ratio: <span style="color: #009688;"><b>{crop_shape[1]/crop_shape[0]:.2f}</b></span><br>
<br>
⚙️ <b>Processing Parameters:</b><br>
   • Applied padding: <span style="color: #FFC107;">10 pixels</span><br>
   • Minimum confidence threshold: <span style="color: #E91E63;">0.1</span><br>
   • Strategy: <span style="color: #00BCD4;">YOLOv8 Detection</span><br>
            """
            
            self.detection_stats_label.setText(stats_text.strip())
            
        except Exception as e:
            self.log(f"❌ YOLO results display error: {e}")
    
    def display_image_array(self, image_array: np.ndarray, label: QLabel, max_size: int = 600):
        """Display numpy image array in label with enhanced styling"""
        try:
            from PySide6.QtGui import QImage
            
            height, width, channel = image_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap
            scaled_pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(scaled_pixmap)
            label.setAlignment(Qt.AlignCenter)
            
            # Update label style for successful image display
            label.setStyleSheet("""
                border: 2px solid #4CAF50; 
                background-color: #333333;
                border-radius: 8px;
                padding: 5px;
            """)
            
        except Exception as e:
            label.setText(f"❌ Error displaying image: {e}")
            label.setStyleSheet("""
                border: 2px solid #f44336; 
                background-color: #333333;
                border-radius: 8px;
                color: #f44336;
            """)
    
    def on_ocr_completed(self, ocr_results: Dict):
        """Handle OCR Model completion"""
        self.update_detail_table(self.ocr_table, ocr_results)
    
    def on_baseline_completed(self, baseline_results: Dict):
        """Handle PaddleOCR completion"""
        self.update_detail_table(self.baseline_table, baseline_results)
    
    def update_detail_table(self, table: QTableWidget, results: Dict):
        """Update detail table with enhanced Vietnamese formatting"""
        texts = results.get('texts', [])
        table.setRowCount(len(texts))

        if table.columnCount() != 5:
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["ID", "Text", "Conf", "Coords", "Evaluate"])
        
        for i, text_info in enumerate(texts):
            # STT (Row number)
            row_item = QTableWidgetItem(str(i + 1))
            row_item.setTextAlignment(Qt.AlignCenter)
            row_item.setFont(QFont("Arial", 10))
            table.setItem(i, 0, row_item)
            
            # Văn bản (Text)
            text = text_info.get('text', '')
            text_item = QTableWidgetItem(text)
            text_item.setFont(QFont("Arial", 10, QFont.Bold))
            text_item.setToolTip(f"Full text: {text}")
            table.setItem(i, 1, text_item)
            
            # Độ tin cậy (Confidence) with enhanced color coding
            confidence = text_info.get('confidence', 0)
            conf_item = QTableWidgetItem(f"{confidence:.3f}")
            conf_item.setTextAlignment(Qt.AlignCenter)
            conf_item.setFont(QFont("Arial", 10))
            
            # Enhanced color coding
            if confidence > 0.95:
                conf_item.setBackground(QColor(76, 175, 80))  # Dark green
                conf_item.setForeground(QColor(255, 255, 255))
            elif confidence > 0.85:
                conf_item.setBackground(QColor(139, 195, 74))  # Light green
                conf_item.setForeground(QColor(0, 0, 0))
            elif confidence > 0.7:
                conf_item.setBackground(QColor(255, 235, 59))  # Yellow
                conf_item.setForeground(QColor(0, 0, 0))
            elif confidence > 0.5:
                conf_item.setBackground(QColor(255, 152, 0))  # Orange
                conf_item.setForeground(QColor(255, 255, 255))
            else:
                conf_item.setBackground(QColor(244, 67, 54))  # Red
                conf_item.setForeground(QColor(255, 255, 255))
            
            table.setItem(i, 2, conf_item)
            
            # Tọa độ (Coordinates) with formatting
            coords = text_info.get('coordinates', [])
            bbox = text_info.get('bbox', [])
            
            if bbox and len(bbox) >= 4:
                coord_str = f"({int(bbox[0])},{int(bbox[1])}) → ({int(bbox[2])},{int(bbox[3])})"
            elif coords and len(coords) > 0:
                if isinstance(coords[0], list) and len(coords[0]) >= 2:
                    # Handle coordinate points format
                    coord_str = f"[{len(coords)} points]"
                else:
                    coord_str = f"({coords[0]:.0f},{coords[1]:.0f})" if len(coords) >= 2 else "Invalid"
            else:
                coord_str = "None"
            
            coord_item = QTableWidgetItem(coord_str)
            coord_item.setFont(QFont("Consolas", 9))
            coord_item.setTextAlignment(Qt.AlignCenter)
            coord_item.setForeground(QColor(150, 150, 150))
            coord_item.setToolTip("Top-left → Bottom-right coordinates")
            table.setItem(i, 3, coord_item)
            
            # Đánh giá (Assessment) based on confidence
            if confidence > 0.95:
                assessment = "🌟 Excellent"
                assessment_color = QColor(76, 175, 80, 100)
            elif confidence > 0.9:
                assessment = "✅ Very Good"
                assessment_color = QColor(139, 195, 74, 100)
            elif confidence > 0.8:
                assessment = "👍 Good"
                assessment_color = QColor(255, 235, 59, 100)
            elif confidence > 0.7:
                assessment = "⚠️ Great"
                assessment_color = QColor(255, 152, 0, 100)
            elif confidence > 0.5:
                assessment = "📝 Average"
                assessment_color = QColor(255, 152, 0, 150)
            else:
                assessment = "❌ Needs Improvement"
                assessment_color = QColor(244, 67, 54, 100)
            
            assessment_item = QTableWidgetItem(assessment)
            assessment_item.setFont(QFont("Arial", 10))
            assessment_item.setBackground(assessment_color)
            assessment_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(i, 4, assessment_item)
        
        # Enhanced column sizing for Vietnamese text
        table.resizeColumnsToContents()
        table.setColumnWidth(0, 60)   # STT column
        table.setColumnWidth(1, max(350, table.columnWidth(1)))  # Văn bản column (wider for Vietnamese)
        table.setColumnWidth(2, 100)  # Độ tin cậy column
        table.setColumnWidth(3, 180)  # Tọa độ column
        table.setColumnWidth(4, 140)  # Đánh giá column
    
    def on_processing_completed(self, results: Dict):
        """Handle processing completion"""
        self.current_results = results
        
        # Update comparison table
        self.update_comparison_table(results.get('comparison', {}))
        
        # Update summary
        self.update_summary(results)
        
        # Switch to comparison tab
        self.results_tabs.setCurrentIndex(0)
        
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.log("🎉 OCR pipeline completed successfully!")
    
    def update_comparison_table(self, comparison: Dict):
        """Update comparison table with enhanced formatting"""
        ocr_stats = comparison.get('ocr_stats', {})
        baseline_stats = comparison.get('baseline_stats', {})
        
        metrics = [
            ("📊 Total Texts", "total_texts"),
            ("📈 Average Confidence", "avg_confidence"),
            ("🏆 Highest Confidence", "max_confidence"),
            ("📉 Lowest Confidence", "min_confidence"),
            ("⭐ High Confidence (>0.9)", "high_confidence_count"),
            ("🌟 Very High Confidence (>0.95)", "very_high_confidence_count"),
            ("✅ Good Confidence (>0.8)", "good_confidence_count"),
            ("⚠️ Low Confidence (<0.5)", "low_confidence_count"),
            ("📊 Confidence Standard Deviation", "confidence_std"),
            ("🔢 Median Confidence", "confidence_median"),
            ("⏱️ Processing Time (seconds)", "processing_time")
        ]
        self.comparison_table.setRowCount(len(metrics))
        
        for i, (metric_name, metric_key) in enumerate(metrics):
            # Metric name with icon
            metric_item = QTableWidgetItem(metric_name)
            metric_item.setFont(QFont("Arial", 11, QFont.Bold))
            self.comparison_table.setItem(i, 0, metric_item)
            
            # OCR Model value
            ocr_value = ocr_stats.get(metric_key, 0)
            if metric_key in ['avg_confidence', 'max_confidence', 'min_confidence', 'confidence_std', 'confidence_median']:
                ocr_str = f"{ocr_value:.3f}"
            else:
                ocr_str = str(int(ocr_value))
            
            ocr_item = QTableWidgetItem(ocr_str)
            ocr_item.setFont(QFont("Arial", 11))
            ocr_item.setBackground(QColor(33, 150, 243, 50))  # Light blue for OCR Model
            self.comparison_table.setItem(i, 1, ocr_item)
            
            # PaddleOCR value
            baseline_value = baseline_stats.get(metric_key, 0)
            if metric_key in ['avg_confidence', 'max_confidence', 'min_confidence', 'confidence_std', 'confidence_median']:
                baseline_str = f"{baseline_value:.3f}"
            else:
                baseline_str = str(int(baseline_value))
            
            baseline_item = QTableWidgetItem(baseline_str)
            baseline_item.setFont(QFont("Arial", 11))
            baseline_item.setBackground(QColor(255, 87, 34, 50))  # Light orange for PaddleOCR
            self.comparison_table.setItem(i, 2, baseline_item)
            
            # Highlight better performance
            if metric_key not in ["min_confidence", "low_confidence_count", "confidence_std"]:  # Higher is better
                if ocr_value > baseline_value:
                    ocr_item.setBackground(QColor(76, 175, 80, 100))  # Green highlight
                elif baseline_value > ocr_value:
                    baseline_item.setBackground(QColor(76, 175, 80, 100))  # Green highlight
            else:  # Lower is better for these metrics
                if ocr_value < baseline_value:
                    ocr_item.setBackground(QColor(76, 175, 80, 100))  # Green highlight
                elif baseline_value < ocr_value:
                    baseline_item.setBackground(QColor(76, 175, 80, 100))  # Green highlight
        
        # Enhanced column sizing
        self.comparison_table.resizeColumnsToContents()
        self.comparison_table.setColumnWidth(0, 280)  # Metric column
        self.comparison_table.setColumnWidth(1, 120)  # ocr column
        self.comparison_table.setColumnWidth(2, 120)  # PaddleOCR column
    
    def update_summary(self, results: Dict):
        """Update summary with enhanced formatting (updated naming)"""
        yolo = results.get('yolo_detection', {})
        ocr = results.get('ocr_results', {})
        baseline = results.get('baseline_results', {})
        best_detection = yolo.get('best_detection', {})
        
        # Calculate processing stats
        ocr_texts = ocr.get('total_texts', 0)
        baseline_texts = baseline.get('total_texts', 0)
        
        # Determine better performer
        ocr_avg_conf = ocr.get('avg_confidence', 0) if ocr_texts > 0 else 0
        baseline_avg_conf = baseline.get('avg_confidence', 0) if baseline_texts > 0 else 0
        
        better_model = "ocr v6" if ocr_avg_conf > baseline_avg_conf else "PaddleOCR"
        better_color = "#4CAF50" if ocr_avg_conf > baseline_avg_conf else "#FF9800"
        
        # Calculate additional metrics
        ocr_high_conf = sum(1 for t in ocr.get('texts', []) if t.get('confidence', 0) > 0.9)
        baseline_high_conf = sum(1 for t in baseline.get('texts', []) if t.get('confidence', 0) > 0.9)
        
        summary_text = f"""
<div style="font-family: Arial; line-height: 1.6;">
<h3 style="color: #4CAF50; margin-bottom: 15px;">🎯 Processing Results Summary</h3>
<hr style="border: 1px solid #555; margin: 10px 0;">

<p><strong>📁 Input Image:</strong> <span style="color: #81C784;">{Path(results['input_image']).name}</span></p>

<h4 style="color: #2196F3; margin: 15px 0 10px 0;">🎯 YOLO Detection Results</h4>
<p>• Number of detected receipts: <strong>{len(yolo.get('detections', []))}</strong></p>
<p>• Best confidence: <strong>{yolo.get('best_detection', {}).get('confidence', 0):.3f}</strong></p>
<p>• Region size: <span style="color: #FF5722;">{int(best_detection.get('x2', 0)) - int(best_detection.get('x1', 0))} × {int(best_detection.get('y2', 0)) - int(best_detection.get('y1', 0))} pixels</span></p>

<h4 style="color: #FF9800; margin: 15px 0 10px 0;">📊 Text Recognition Results</h4>
<table style="width: 100%; border-collapse: collapse;">
<tr style="background-color: #444;">
    <th style="padding: 8px; text-align: left; border: 1px solid #666;">Model</th>
    <th style="padding: 8px; text-align: center; border: 1px solid #666;">Number of Texts</th>
    <th style="padding: 8px; text-align: center; border: 1px solid #666;">Average Confidence</th>
    <th style="padding: 8px; text-align: center; border: 1px solid #666;">High (>0.9)</th>
</tr>
<tr>
    <td style="padding: 8px; border: 1px solid #666;">🤖 OCR Model</td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{ocr_texts}</strong></td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{ocr_avg_conf:.3f}</strong></td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{ocr_high_conf}</strong></td>
</tr>
<tr>
    <td style="padding: 8px; border: 1px solid #666;">🧠 PaddleOCR</td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{baseline_texts}</strong></td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{baseline_avg_conf:.3f}</strong></td>
    <td style="padding: 8px; text-align: center; border: 1px solid #666;"><strong>{baseline_high_conf}</strong></td>
</tr>
</table>

<p style="margin-top: 15px;"><strong>🏆 Better Model:</strong> 
<span style="color: {better_color}; font-weight: bold;">{better_model}</span></p>

<p style="margin-top: 10px; color: #888; font-size: 12px;">⏱️ Processing time: {results.get('timestamp', 'Not available')[:19]}</p>

<h4 style="color: #9C27B0; margin: 15px 0 10px 0;">📈 Detailed Statistics</h4>
<p>• Total Processing Time: <strong>{results.get('total_processing_time', 'Not available')}</strong></p>
<p>• Number of very high confidence texts (>0.95): <strong>OCR Model: {sum(1 for t in ocr.get('texts', []) if t.get('confidence', 0) > 0.95)} | PaddleOCR: {sum(1 for t in baseline.get('texts', []) if t.get('confidence', 0) > 0.95)}</strong></p>
<p>• High-quality text rate: <strong>OCR Model: {(ocr_high_conf/ocr_texts*100 if ocr_texts > 0 else 0):.1f}% | PaddleOCR: {(baseline_high_conf/baseline_texts*100 if baseline_texts > 0 else 0):.1f}%</strong></p>
</div>
        """
        
        self.summary_label.setText(summary_text.strip())
    
    def on_error_occurred(self, error_message: str):
        """Handle processing error"""
        self.log(error_message)
        
        # Re-enable UI
        self.process_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show error dialog
        QMessageBox.critical(self, "Processing Error", error_message)
    
    def save_results(self):
        """Save current results with Vietnamese interface"""
        if not self.current_results:
            QMessageBox.warning(self, "Warning", "No results to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(self.current_results['input_image']).stem
        filename = f"OCR_Result_{base_name}_{timestamp}.json"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save OCR Results",
            filename,
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_results, f, indent=2, ensure_ascii=False, default=str)
                
                self.log(f"💾 Results have been saved to: {Path(file_path).name}")
                QMessageBox.information(self, "Success", f"Results have been saved to:\n{file_path}")
                
            except Exception as e:
                error_msg = f"Error saving results: {e}"
                self.log(f"❌ {error_msg}")
                QMessageBox.critical(self, "Save Error", error_msg)


def main():
    """Main application entry point"""
    if not HAS_PYSIDE6:
        print("❌ PySide6 is required but not available")
        print("Install with: pip install PySide6")
        return 1
    
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("OCR Pipeline")
        app.setApplicationDisplayName("AI OCR Pipeline - Professional Bill Recognition")
        
        # Set modern Fusion style
        app.setStyle('Fusion')
        
        # Set app icon if available
        app_icon = QPixmap(32, 32)
        app_icon.fill(QColor(76, 175, 80))  # Green color
        app.setWindowIcon(app_icon)
        
        # Create and show main window
        window = OCRPipelineGUI()
        window.show()
        
        print("🎉 Modern OCR Pipeline GUI launched successfully!")
        print("🚀 Ready for professional bill text recognition")
        
        # Run application
        return app.exec()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        print(f"Error details: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    main()