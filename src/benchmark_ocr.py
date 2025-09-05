#!/usr/bin/env python3
"""
OCR Performance Benchmark
Benchmarks OCR Model vs PaddleOCR on multiple images
"""

from pathlib import Path
import json
import time
import sys
import os
from statistics import mean, stdev

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from yolo_detect_bill.bill_detector import BillDetector
from baseline_model.baseline_ocr import Baseline_Model
import cv2

class OCRBenchmark:
    """OCR performance benchmark"""
    
    def __init__(self):
        # Initialize YOLO
        yolo_model_path = Path(__file__).parent.parent / "yolo_detect_bill" / "bill_models.pt"
        self.yolo_detector = BillDetector(model_path=str(yolo_model_path))
        self.yolo_detector.load_model()
        
        # Initialize PaddleOCR
        from paddleocr import PaddleOCR
        src_folder = Path(__file__).parent
        project_root = src_folder.parent
        det_model_path = str(project_root / "dbnet" / "model")
        
        self.ocr_engine = PaddleOCR(
            det_model_dir=str(det_model_path),
            rec=True,
            use_angle_cls=False,
            use_gpu=False,
            lang='ch'
        )

        self.baseline_engine = Baseline_Model()
    
    def crop_bill_region(self, image, confidence_threshold=0.1):
        """Crop best bill region from image"""
        detections = self.yolo_detector.detect_bills_from_frame(image, confidence_threshold)
        
        if not detections:
            return None
        
        best_detection = max(detections, key=lambda x: x['confidence'])
        
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
        
        return image[y1:y2, x1:x2]
    
    def benchmark_baseline(self, bill_crop):
        """Benchmark PaddleOCR"""
        temp_path = "temp_benchmark_baseline.jpg"
        cv2.imwrite(temp_path, bill_crop)

        start_time = time.time()
        result = self.baseline_engine.predict_text(temp_path)
        processing_time = time.time() - start_time
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        texts = result.get('texts', [])
        
        return {
            'model': 'PaddleOCR',
            'processing_time': processing_time,
            'text_count': len(texts),
            'texts': texts,
            'avg_confidence': mean([t['confidence'] for t in texts]) if texts else 0,
            'high_confidence_count': sum(1 for t in texts if t['confidence'] > 0.9)
        }
    
    def benchmark_ocr(self, bill_crop):
        """Benchmark OCR Model"""
        temp_path = "temp_benchmark_ocr.jpg"
        cv2.imwrite(temp_path, bill_crop)
        
        start_time = time.time()
        result = self.ocr_engine.ocr(temp_path, cls=False)
        processing_time = time.time() - start_time
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        texts = []
        if result and len(result[0]) > 0:
            for line in result[0]:
                coords = line[0]
                text_info = line[1]
                text = text_info[0]
                confidence = text_info[1]
                
                texts.append({
                    "text": text,
                    "confidence": confidence,
                    "coordinates": coords
                })
        
        return {
            'model': 'OCR Model',
            'processing_time': processing_time,
            'text_count': len(texts),
            'texts': texts,
            'avg_confidence': mean([t['confidence'] for t in texts]) if texts else 0,
            'high_confidence_count': sum(1 for t in texts if t['confidence'] > 0.9)
        }
    
    def run_benchmark(self, num_images=10):
        """Run benchmark on multiple images"""
        print("🏁 OCR Performance Benchmark")
        print("=" * 60)
        
        # Get test images
        image_folder = Path(__file__).parent.parent / "image_test"
        test_images = list(image_folder.glob("*.jpg"))[:num_images]
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        print(f"📸 Testing {len(test_images)} images...")
        
        baseline_results = []
        ocr_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n🔄 Processing {i}/{len(test_images)}: {image_path.name}")
            
            # Load and crop image
            image = cv2.imread(str(image_path))
            bill_crop = self.crop_bill_region(image)
            
            if bill_crop is None:
                print(f"⚠️ No bill detected in {image_path.name}")
                continue
            
            # Benchmark PaddleOCR
            print("   🤖 Testing PaddleOCR...")
            baseline_result = self.benchmark_baseline(bill_crop)
            baseline_result['image'] = image_path.name
            baseline_results.append(baseline_result)
            
            # Benchmark OCR Model
            print("   🧠 Testing OCR Model...")
            ocr_result = self.benchmark_ocr(bill_crop)
            ocr_result['image'] = image_path.name
            ocr_results.append(ocr_result)
            
            print(f"   ✅ Paddle: {baseline_result['text_count']} texts ({baseline_result['processing_time']:.2f}s)")
            print(f"   ✅ OCR Mdodel: {ocr_result['text_count']} texts ({ocr_result['processing_time']:.2f}s)")
        
        # Generate report
        self.generate_report(baseline_results, ocr_results)
    
    def generate_report(self, baseline_results, ocr_results):
        """Generate benchmark report"""
        print(f"\n📊 BENCHMARK REPORT")
        print("=" * 60)
        
        if not baseline_results or not ocr_results:
            print("❌ No results to analyze!")
            return
        
        # Processing time analysis
        baseline_times = [r['processing_time'] for r in baseline_results]
        ocr_times = [r['processing_time'] for r in ocr_results]
        
        print(f"⏱️ Processing Time (seconds):")
        print(f"   PaddleOCR:   avg={mean(baseline_times):.3f}s, std={stdev(baseline_times):.3f}s")
        print(f"   OCR Model: avg={mean(ocr_times):.3f}s, std={stdev(ocr_times):.3f}s")
        print(f"   Speed ratio: {mean(ocr_times)/mean(baseline_times):.2f}x (OCR Model faster)")
        
        # Text detection analysis
        baseline_counts = [r['text_count'] for r in baseline_results]
        ocr_counts = [r['text_count'] for r in ocr_results]
        
        print(f"\n📝 Text Detection Count:")
        print(f"   PaddleOCR:   avg={mean(baseline_counts):.1f}, std={stdev(baseline_counts):.1f}")
        print(f"   OCR Model: avg={mean(ocr_counts):.1f}, std={stdev(ocr_counts):.1f}")
        
        # Confidence analysis
        baseline_confs = [r['avg_confidence'] for r in baseline_results if r['avg_confidence'] > 0]
        ocr_confs = [r['avg_confidence'] for r in ocr_results if r['avg_confidence'] > 0]
        
        if baseline_confs and ocr_confs:
            print(f"\n🎯 Average Confidence:")
            print(f"   PaddleOCR:   {mean(baseline_confs):.3f}")
            print(f"   OCR Model: {mean(ocr_confs):.3f}")
        
        # High confidence analysis
        baseline_high = sum(r['high_confidence_count'] for r in baseline_results)
        baseline_total = sum(r['text_count'] for r in baseline_results)
        ocr_high = sum(r['high_confidence_count'] for r in ocr_results)
        ocr_total = sum(r['text_count'] for r in ocr_results)
        
        print(f"\n🏆 High Confidence (>0.9) Ratio:")
        if baseline_total > 0:
            print(f"   PaddleOCR:   {baseline_high}/{baseline_total} ({baseline_high/baseline_total*100:.1f}%)")
        if ocr_total > 0:
            print(f"   OCR Model: {ocr_high}/{ocr_total} ({ocr_high/ocr_total*100:.1f}%)")
        
        # Individual image comparison
        print(f"\n🔍 Individual Image Analysis:")
        print("-" * 60)
        print(f"{'Image':<20} {'OCR Model(t/s)':<12} {'PaddleOCR(t/s)':<12} {'Winner'}")
        print("-" * 60)
        
        baseline_wins = 0
        ocr_wins = 0
        
        for i in range(min(len(baseline_results), len(ocr_results))):
            baseline = baseline_results[i]
            ocr = ocr_results[i]
            
            # Determine winner (more texts + higher confidence + faster)
            baseline_score = baseline['text_count'] * baseline['avg_confidence'] / baseline['processing_time']
            ocr_score = ocr['text_count'] * ocr['avg_confidence'] / ocr['processing_time']
            
            if baseline_score > ocr_score:
                winner = "PaddleOCR ⭐"
                baseline_wins += 1
            else:
                winner = "OCR Model ⭐"
                ocr_wins += 1
            
            baseline_info = f"{baseline['text_count']}/{baseline['processing_time']:.2f}s"
            ocr_info = f"{ocr['text_count']}/{ocr['processing_time']:.2f}s"
            print(f"{baseline['image'][:18]:<20} {baseline_info:<12} {ocr_info:<12} {winner}")
        
        print("-" * 60)
        print(f"🏆 Overall Winner: {'PaddleOCR' if baseline_wins > ocr_wins else 'OCR Model'} ({max(baseline_wins, ocr_wins)}/{baseline_wins + ocr_wins} images)")
        
        # Save detailed results into folder benchmark_ocr_result
        benchmark_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'baseline_avg_time': mean(baseline_times),
                'ocr_avg_time': mean(ocr_times),
                'baseline_avg_texts': mean(baseline_counts),
                'ocr_avg_texts': mean(ocr_counts),
                'baseline_wins': baseline_wins,
                'ocr_wins': ocr_wins
            },
            'baseline_results': baseline_results,
            'ocr_results': ocr_results
        }
        
        results_folder = os.path.join(os.path.dirname(__file__), "benchmark_ocr_result")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        results_file = os.path.join(results_folder, "benchmark_results.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        print("🎉 Benchmark completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OCR Performance Benchmark')
    parser.add_argument('--images', type=int, default=10, help='Number of images to test (default: 10)')
    args = parser.parse_args()
    
    benchmark = OCRBenchmark()
    benchmark.run_benchmark(num_images=args.images)