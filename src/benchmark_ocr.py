#!/usr/bin/env python3
"""
OCR Performance Benchmark
Benchmarks SVTR vs PaddleOCR on multiple images
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
        
        det_model_path = project_root / "dbnet" / "model"
        rec_model_path = project_root / "svtr" / "model"
        
        self.ocr_engine = PaddleOCR(
            det_model_dir=det_model_path,
            rec_model_dir=rec_model_path,
            use_angle_cls=False,
            lang='en'
        )

        self.baseline_engine = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            show_log=False
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
    
    def process_image(self, image, engine_name="custom"):
        """Process image with OCR Model and PaddleOCR and return parsed results"""
        try:
            if engine_name == "custom":
                engine = self.ocr_engine
            elif engine_name == "baseline":
                engine = self.baseline_engine
            else:
                raise ValueError(f"Invalid engine_name: {engine_name}. Use 'custom' or 'baseline'")

            # Run OCR
            result = engine.ocr(image, cls=False)
            
            # Parse results
            parsed_results = []
            confidences = []

            if result and result[0]:
                for line in result[0]:
                    coords = line[0]  # Bounding coordinates
                    text_info = line[1]  # (text, confidence)
                    text = text_info[0]
                    confidence = text_info[1]
                    
                    parsed_results.append({
                        'coords': coords,
                        'text': text,
                        'confidence': confidence
                    })
                    confidences.append(confidence)
            
            # Calculate metrics
            avg_confidence = mean(confidences) if confidences else 0
            high_confidence_count = sum(1 for c in confidences if c > 0.9)
            
            return {
                'results': parsed_results,
                'text_count': len(parsed_results),
                'avg_confidence': avg_confidence,
                'high_confidence_count': high_confidence_count,
                'confidences': confidences
            }
            
        except Exception as e:
            print(f"❌ Error processing with PaddleOCR ({engine_name}): {e}")
            return {
                'results': [],
                'text_count': 0,
                'avg_confidence': 0,
                'high_confidence_count': 0,
                'confidences': []
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
        
        custom_results = []
        baseline_results = []
        
        for i, image_path in enumerate(test_images, 1):
            print(f"\n🔄 Processing {i}/{len(test_images)}: {image_path.name}")
            
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    print(f"❌ Could not load image: {image_path.name}")
                    continue

                bill_crop = self.crop_bill_region(image)
                
                if bill_crop is None:
                    print(f"⚠️ No bill detected in {image_path.name}")
                    continue

                print("   🔧 Processing with Custom DBNet+SVTR...")
                start_time = time.time()
                custom_result = self.process_image(bill_crop, "custom")
                custom_time = time.time() - start_time

                print("   🔧 Processing with Baseline PaddleOCR...")
                start_time = time.time()
                baseline_result = self.process_image(bill_crop, "baseline")
                baseline_time = time.time() - start_time
                
                
                # Store results
                custom_results.append({
                    'image': image_path.name,
                    'processing_time': custom_time,
                    'text_count': custom_result['text_count'],
                    'avg_confidence': custom_result['avg_confidence'],
                    'high_confidence_count': custom_result['high_confidence_count'],
                    'results': custom_result['results']
                })
                
                baseline_results.append({
                    'image': image_path.name,
                    'processing_time': baseline_time,
                    'text_count': baseline_result['text_count'],
                    'avg_confidence': baseline_result['avg_confidence'],
                    'high_confidence_count': baseline_result['high_confidence_count'],
                    'results': baseline_result['results']
                })

                print(f"   ✅ Custom:   {custom_result['text_count']} texts in {custom_time:.2f}s (avg conf: {custom_result['avg_confidence']:.3f})")
                print(f"   ✅ Baseline: {baseline_result['text_count']} texts in {baseline_time:.2f}s (avg conf: {baseline_result['avg_confidence']:.3f})")
                
                # Show winner for this image
                custom_score = (custom_result['text_count'] * custom_result['avg_confidence']) / custom_time if custom_time > 0 else 0
                baseline_score = (baseline_result['text_count'] * baseline_result['avg_confidence']) / baseline_time if baseline_time > 0 else 0
                winner = "Custom" if custom_score > baseline_score else "Baseline"
                print(f"   🏆 Winner for this image: {winner}")

            except Exception as e:
                print(f"❌ Error processing {image_path.name}: {e}")
                continue
        if custom_results and baseline_results:
            print(f"\n📊 Generating comprehensive comparison report...")
            self.generate_report(custom_results, baseline_results)
        else:
            print("❌ No images were successfully processed!")
    
    def generate_report(self, custom_results, baseline_results):
        """Generate benchmark report"""
        print(f"\n📊 BENCHMARK REPORT")
        print("=" * 60)

        if not custom_results or not baseline_results:
            print("❌ No results to analyze!")
            return

        # Processing time analysis
        custom_times = [r['processing_time'] for r in custom_results]
        baseline_times = [r['processing_time'] for r in baseline_results]

        print(f"⏱️  PROCESSING TIME ANALYSIS:")
        print(f"   Custom DBNet+SVTR: avg={mean(custom_times):.3f}s, std={stdev(custom_times) if len(custom_times) > 1 else 0:.3f}s")
        print(f"   Baseline PaddleOCR: avg={mean(baseline_times):.3f}s, std={stdev(baseline_times) if len(baseline_times) > 1 else 0:.3f}s")

        speed_ratio = mean(baseline_times) / mean(custom_times) if mean(custom_times) > 0 else 0
        faster_model = "Custom" if speed_ratio > 1 else "Baseline"
        print(f"   Speed ratio: {abs(speed_ratio):.2f}x ({faster_model} faster)")

        # Text detection analysis
        custom_counts = [r['text_count'] for r in custom_results]
        baseline_counts = [r['text_count'] for r in baseline_results]

        print(f"\n📝 TEXT DETECTION ANALYSIS:")
        print(f"   Custom DBNet+SVTR: avg={mean(custom_counts):.1f}, std={stdev(custom_counts) if len(custom_counts) > 1 else 0:.1f}")
        print(f"   Baseline PaddleOCR: avg={mean(baseline_counts):.1f}, std={stdev(baseline_counts) if len(baseline_counts) > 1 else 0:.1f}")

        # Confidence analysis
        custom_confs = [r['avg_confidence'] for r in custom_results if r['avg_confidence'] > 0]
        baseline_confs = [r['avg_confidence'] for r in baseline_results if r['avg_confidence'] > 0]

        if custom_confs and baseline_confs:
            print(f"\n🎯 CONFIDENCE ANALYSIS:")
            print(f"   Custom DBNet+SVTR: {mean(custom_confs):.3f}")
            print(f"   Baseline PaddleOCR: {mean(baseline_confs):.3f}")

        # High confidence analysis
        custom_high = sum(r['high_confidence_count'] for r in custom_results)
        custom_total = sum(r['text_count'] for r in custom_results)
        baseline_high = sum(r['high_confidence_count'] for r in baseline_results)
        baseline_total = sum(r['text_count'] for r in baseline_results)

        print(f"\n🏆 HIGH CONFIDENCE (>0.9) ANALYSIS:")
        if custom_total > 0:
            print(f"   Custom DBNet+SVTR: {custom_high}/{custom_total} ({custom_high/custom_total*100:.1f}%)")
        if baseline_total > 0:
            print(f"   Baseline PaddleOCR: {baseline_high}/{baseline_total} ({baseline_high/baseline_total*100:.1f}%)")

        # Individual image comparison
        print(f"\n🔍 INDIVIDUAL IMAGE ANALYSIS:")
        print("-" * 80)
        print(f"{'Image':<20} {'Custom(t/s/c)':<15} {'Baseline(t/s/c)':<15} {'Winner'}")
        print("-" * 80)

        custom_wins = 0
        baseline_wins = 0

        for i in range(min(len(custom_results), len(baseline_results))):
            custom = custom_results[i]
            baseline = baseline_results[i]

            # Determine winner using composite score
            custom_score = (custom['text_count'] * custom['avg_confidence']) / custom['processing_time'] if custom['processing_time'] > 0 else 0
            baseline_score = (baseline['text_count'] * baseline['avg_confidence']) / baseline['processing_time'] if baseline['processing_time'] > 0 else 0

            if custom_score > baseline_score:
                winner = "Custom ⭐"
                custom_wins += 1
            else:
                winner = "Baseline ⭐"
                baseline_wins += 1

            custom_info = f"{custom['text_count']}/{custom['processing_time']:.2f}s/{custom['avg_confidence']:.2f}"
            baseline_info = f"{baseline['text_count']}/{baseline['processing_time']:.2f}s/{baseline['avg_confidence']:.2f}"
            print(f"{custom['image'][:18]:<20} {custom_info:<15} {baseline_info:<15} {winner}")

        print("-" * 80)
        overall_winner = 'Custom DBNet+SVTR' if custom_wins > baseline_wins else 'Baseline PaddleOCR'
        print(f"🏆 OVERALL WINNER: {overall_winner} ({max(custom_wins, baseline_wins)}/{custom_wins + baseline_wins} images)")

        # Save detailed results
        benchmark_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'custom_avg_time': mean(custom_times),
                'baseline_avg_time': mean(baseline_times),
                'custom_avg_texts': mean(custom_counts),
                'baseline_avg_texts': mean(baseline_counts),
                'custom_avg_confidence': mean(custom_confs) if custom_confs else 0,
                'baseline_avg_confidence': mean(baseline_confs) if baseline_confs else 0,
                'custom_wins': custom_wins,
                'baseline_wins': baseline_wins,
                'overall_winner': overall_winner
            },
            'custom_results': custom_results,
            'baseline_results': baseline_results
        }

        results_folder = os.path.join(os.path.dirname(__file__), "benchmark_ocr_result")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        results_file = os.path.join(results_folder, f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json")

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")

        # Print sample detected texts
        print(f"\n📝 SAMPLE DETECTED TEXTS:")
        print("-" * 60)
        print("🔧 Custom DBNet+SVTR Results:")
        for i, result in enumerate(custom_results[:3]):
            print(f"  📄 {result['image']}:")
            for j, text_result in enumerate(result['results'][:3]):
                print(f"    {j+1}. '{text_result['text']}' (conf: {text_result['confidence']:.3f})")

        print("\n🔧 Baseline PaddleOCR Results:")
        for i, result in enumerate(baseline_results[:3]):
            print(f"  📄 {result['image']}:")
            for j, text_result in enumerate(result['results'][:3]):
                print(f"    {j+1}. '{text_result['text']}' (conf: {text_result['confidence']:.3f})")

        print("\n🎉 Benchmark completed successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='OCR Performance Benchmark')
    parser.add_argument('--images', type=int, default=10, help='Number of images to test (default: 10)')
    args = parser.parse_args()
    
    benchmark = OCRBenchmark()
    benchmark.run_benchmark(num_images=args.images)
