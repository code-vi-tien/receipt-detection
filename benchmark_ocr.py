#!/usr/bin/env python3
"""
OCR Performance Benchmark
Benchmarks SVTR v6 vs PaddleOCR on multiple images
"""

from pathlib import Path
import json
import time
import sys
from statistics import mean, stdev

# Add paths
sys.path.append(str(Path(__file__).parent / "yolo_detect_bill"))
sys.path.append(str(Path(__file__).parent / "svtr_v6_ocr"))

from yolo_detect_bill.bill_detector import BillDetector
from svtr_v6_ocr.svtr_v6_ocr import SVTRv6TrueInference
import cv2

class OCRBenchmark:
    """OCR performance benchmark"""
    
    def __init__(self):
        # Initialize YOLO
        yolo_model_path = Path(__file__).parent / "yolo_detect_bill" / "bill_models.pt"
        self.yolo_detector = BillDetector(model_path=str(yolo_model_path))
        self.yolo_detector.load_model()
        
        # Initialize SVTR v6
        self.svtr_engine = SVTRv6TrueInference()
        
        # Initialize PaddleOCR
        from paddleocr import PaddleOCR
        det_model_path = str(Path(__file__).parent / "paddle_ocr" / "ch_db_res18")
        self.paddle_engine = PaddleOCR(
            det_model_dir=det_model_path,
            rec=True,
            use_angle_cls=False,
            use_gpu=False,
            lang='ch'
        )
    
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
    
    def benchmark_svtr_v6(self, bill_crop):
        """Benchmark SVTR v6"""
        temp_path = "temp_benchmark_svtr.jpg"
        cv2.imwrite(temp_path, bill_crop)
        
        start_time = time.time()
        result = self.svtr_engine.predict_text(temp_path)
        processing_time = time.time() - start_time
        
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        texts = result.get('texts', [])
        
        return {
            'model': 'SVTR v6',
            'processing_time': processing_time,
            'text_count': len(texts),
            'texts': texts,
            'avg_confidence': mean([t['confidence'] for t in texts]) if texts else 0,
            'high_confidence_count': sum(1 for t in texts if t['confidence'] > 0.9)
        }
    
    def benchmark_paddle_ocr(self, bill_crop):
        """Benchmark PaddleOCR"""
        temp_path = "temp_benchmark_paddle.jpg"
        cv2.imwrite(temp_path, bill_crop)
        
        start_time = time.time()
        result = self.paddle_engine.ocr(temp_path, cls=False)
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
            'model': 'PaddleOCR',
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
        image_folder = Path("image_test")
        test_images = list(image_folder.glob("*.jpg"))[:num_images]
        
        if not test_images:
            print("❌ No test images found!")
            return
        
        print(f"📸 Testing {len(test_images)} images...")
        
        svtr_results = []
        paddle_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n🔄 Processing {i}/{len(test_images)}: {image_path.name}")
            
            # Load and crop image
            image = cv2.imread(str(image_path))
            bill_crop = self.crop_bill_region(image)
            
            if bill_crop is None:
                print(f"⚠️ No bill detected in {image_path.name}")
                continue
            
            # Benchmark SVTR v6
            print("   🤖 Testing SVTR v6...")
            svtr_result = self.benchmark_svtr_v6(bill_crop)
            svtr_result['image'] = image_path.name
            svtr_results.append(svtr_result)
            
            # Benchmark PaddleOCR
            print("   🧠 Testing PaddleOCR...")
            paddle_result = self.benchmark_paddle_ocr(bill_crop)
            paddle_result['image'] = image_path.name
            paddle_results.append(paddle_result)
            
            print(f"   ✅ SVTR: {svtr_result['text_count']} texts ({svtr_result['processing_time']:.2f}s)")
            print(f"   ✅ Paddle: {paddle_result['text_count']} texts ({paddle_result['processing_time']:.2f}s)")
        
        # Generate report
        self.generate_report(svtr_results, paddle_results)
    
    def generate_report(self, svtr_results, paddle_results):
        """Generate benchmark report"""
        print(f"\n📊 BENCHMARK REPORT")
        print("=" * 60)
        
        if not svtr_results or not paddle_results:
            print("❌ No results to analyze!")
            return
        
        # Processing time analysis
        svtr_times = [r['processing_time'] for r in svtr_results]
        paddle_times = [r['processing_time'] for r in paddle_results]
        
        print(f"⏱️ Processing Time (seconds):")
        print(f"   SVTR v6:   avg={mean(svtr_times):.3f}s, std={stdev(svtr_times):.3f}s")
        print(f"   PaddleOCR: avg={mean(paddle_times):.3f}s, std={stdev(paddle_times):.3f}s")
        print(f"   Speed ratio: {mean(paddle_times)/mean(svtr_times):.2f}x (SVTR faster)")
        
        # Text detection analysis
        svtr_counts = [r['text_count'] for r in svtr_results]
        paddle_counts = [r['text_count'] for r in paddle_results]
        
        print(f"\n📝 Text Detection Count:")
        print(f"   SVTR v6:   avg={mean(svtr_counts):.1f}, std={stdev(svtr_counts):.1f}")
        print(f"   PaddleOCR: avg={mean(paddle_counts):.1f}, std={stdev(paddle_counts):.1f}")
        
        # Confidence analysis
        svtr_confs = [r['avg_confidence'] for r in svtr_results if r['avg_confidence'] > 0]
        paddle_confs = [r['avg_confidence'] for r in paddle_results if r['avg_confidence'] > 0]
        
        if svtr_confs and paddle_confs:
            print(f"\n🎯 Average Confidence:")
            print(f"   SVTR v6:   {mean(svtr_confs):.3f}")
            print(f"   PaddleOCR: {mean(paddle_confs):.3f}")
        
        # High confidence analysis
        svtr_high = sum(r['high_confidence_count'] for r in svtr_results)
        svtr_total = sum(r['text_count'] for r in svtr_results)
        paddle_high = sum(r['high_confidence_count'] for r in paddle_results)
        paddle_total = sum(r['text_count'] for r in paddle_results)
        
        print(f"\n🏆 High Confidence (>0.9) Ratio:")
        if svtr_total > 0:
            print(f"   SVTR v6:   {svtr_high}/{svtr_total} ({svtr_high/svtr_total*100:.1f}%)")
        if paddle_total > 0:
            print(f"   PaddleOCR: {paddle_high}/{paddle_total} ({paddle_high/paddle_total*100:.1f}%)")
        
        # Individual image comparison
        print(f"\n🔍 Individual Image Analysis:")
        print("-" * 60)
        print(f"{'Image':<20} {'SVTR(t/s)':<12} {'Paddle(t/s)':<12} {'Winner'}")
        print("-" * 60)
        
        svtr_wins = 0
        paddle_wins = 0
        
        for i in range(min(len(svtr_results), len(paddle_results))):
            svtr = svtr_results[i]
            paddle = paddle_results[i]
            
            # Determine winner (more texts + higher confidence + faster)
            svtr_score = svtr['text_count'] * svtr['avg_confidence'] / svtr['processing_time']
            paddle_score = paddle['text_count'] * paddle['avg_confidence'] / paddle['processing_time']
            
            if svtr_score > paddle_score:
                winner = "SVTR ⭐"
                svtr_wins += 1
            else:
                winner = "Paddle ⭐"
                paddle_wins += 1
            
            svtr_info = f"{svtr['text_count']}/{svtr['processing_time']:.2f}s"
            paddle_info = f"{paddle['text_count']}/{paddle['processing_time']:.2f}s"
            print(f"{svtr['image'][:18]:<20} {svtr_info:<12} {paddle_info:<12} {winner}")
        
        print("-" * 60)
        print(f"🏆 Overall Winner: {'SVTR v6' if svtr_wins > paddle_wins else 'PaddleOCR'} ({max(svtr_wins, paddle_wins)}/{svtr_wins + paddle_wins} images)")
        
        # Save detailed results
        benchmark_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'svtr_avg_time': mean(svtr_times),
                'paddle_avg_time': mean(paddle_times),
                'svtr_avg_texts': mean(svtr_counts),
                'paddle_avg_texts': mean(paddle_counts),
                'svtr_wins': svtr_wins,
                'paddle_wins': paddle_wins
            },
            'svtr_results': svtr_results,
            'paddle_results': paddle_results
        }
        
        with open('benchmark_results.json', 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Detailed results saved to: benchmark_results.json")
        print("🎉 Benchmark completed!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OCR Performance Benchmark')
    parser.add_argument('--images', type=int, default=10, help='Number of images to test (default: 10)')
    args = parser.parse_args()
    
    benchmark = OCRBenchmark()
    benchmark.run_benchmark(num_images=args.images)
