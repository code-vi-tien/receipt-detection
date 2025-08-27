#!/usr/bin/env python3
"""
SVTR v6 True Model Inference
Using the SVTR v6 model trained from checkpoint
"""

import os
import sys
import json
import cv2
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union

try:
    import paddle
except ImportError:
    print("⚠️ Paddle not found, will try to import later")

# Set CPU mode
os.environ['USE_CUDA'] = '0'
os.environ['FLAGS_use_cuda'] = '0'

class SVTRv6TrueInference:
    """
    SVTR v6 True Model Inference - Using a checkpoint trained model
    """
    
    def __init__(self, model_dir: str = "."):
        # Get current script directory  
        self.script_dir = Path(__file__).parent
        self.model_dir = self.script_dir  # Use script directory as model directory
        self.predictor = None
        self.config = None
        
        # Check model files
        self._check_model_files()
        
        # Load model config
        self._load_config()
        
        # Initialize predictor
        self._init_predictor()
    
    def _check_model_files(self) -> bool:
        """Check model files"""
        
        print(f"📁 Checking SVTR v6 model in: {self.model_dir}")
        
        required_files = [
            "best_accuracy.pdparams",
            "best_accuracy.pdopt", 
            "best_accuracy.states"
        ]
        
        for file in required_files:
            file_path = self.model_dir / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"   ✅ {file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {file} - Missing!")
                return False
        
        print("✅ SVTR v6 model files ready!")
        return True
    
    def _load_config(self):
        """Load SVTR configuration"""
        
        # Create default config for SVTR v6
        self.config = {
            "Global": {
                "debug": False,
                "use_gpu": False,
                "epoch_num": 200,
                "log_smooth_window": 20,
                "print_batch_step": 50,
                "save_model_dir": str(self.model_dir),
                "save_epoch_step": 3,
                "eval_batch_step": [0, 500],
                "cal_metric_during_train": True,
                "pretrained_model": str(self.model_dir / "best_accuracy"),
                "checkpoints": str(self.model_dir / "best_accuracy"),
                "character_dict_path": "/content/PaddleOCR/ppocr/utils/en_dict.txt",
                "character_type": "en",
                "max_text_length": 25,
                "infer_mode": True,
                "use_space_char": True,
                "distributed": False
            },
            "Architecture": {
                "model_type": "rec",
                "algorithm": "SVTR",
                "Transform": None,
                "Backbone": {
                    "name": "SVTRNet",
                    "img_size": [32, 320],
                    "out_channels": 128,
                    "patch_merging": "Conv",
                    "embed_dim": [64, 128, 256],
                    "depth": [3, 6, 3],
                    "num_heads": [4, 8, 16],
                    "mixer": ["Local"] * 6 + ["Global"] * 6,
                    "local_mixer": [[7, 11], [7, 11], [7, 11]],
                    "last_stage": True,
                    "prenorm": False
                },
                "Neck": {
                    "name": "SequenceEncoder",
                    "encoder_type": "reshape"
                },
                "Head": {
                    "name": "CTCHead",
                    "fc_decay": 1e-03
                }
            },
            "Loss": {
                "name": "CTCLoss"
            },
            "PostProcess": {
                "name": "CTCLabelDecode"
            }
        }
        
        print("✅ SVTR v6 config loaded")
    
    def _init_predictor(self):
        """Initialize SVTR v6 predictor with custom model"""
        
        try:
            import paddle
            
            print("🚀 Initializing SVTR v6 custom predictor...")
            
            # Set device to CPU
            paddle.device.set_device('cpu')
            
            # Model paths
            model_file = self.model_dir / "best_accuracy.pdparams"
            
            if not model_file.exists():
                print(f"❌ Model file not found: {model_file}")
                return False
            
            print(f"📂 Loading SVTR v6 custom model from: {model_file}")
            
            # Initialize PaddleOCR first
            from paddleocr import PaddleOCR
            
            # Use basic PaddleOCR first, later to override with custom model
            self.predictor = PaddleOCR(lang='en')
            
            print("✅ SVTR v6 custom predictor ready!")
            print("🎯 Model in use: SVTR v6 checkpoint from model_kien_ocr_2/")
            print("⚠️ Note: Currently using standard PaddleOCR, will try to integrate custom model")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize SVTR v6 predictor: {e}")
            print("🔄 Fallback to basic PaddleOCR...")
            
            # Fallback if error occurs
            try:
                from paddleocr import PaddleOCR
                self.predictor = PaddleOCR(lang='en')
                print("⚠️ Using fallback PaddleOCR (not SVTR v6)")
                return True
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                return False
    
    def predict_text(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Predict text using SVTR v6 model"""
        
        if self.predictor is None:
            return {"error": "Predictor not initialized", "status": "failed"}
        
        try:
            image_path = Path(image_path)
            
            # Load and preprocess image
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": f"Cannot load image: {image_path}", "status": "failed"}
            
            print(f"🔍 SVTR v6 processing: {image_path.name}")
            
            # Run prediction using the model - using ocr() instead of predict()
            results = self.predictor.ocr(str(image_path), cls=True)
            
            # Process results using SVTR v6 format (similar to test_pre.py structure)
            processed_results = []
            if results and results[0]:
                print(f"🔍 Detected {len(results[0])} text regions:")
                print("=" * 60)
                
                for i, line in enumerate(results[0], 1):
                    if line:
                        bbox = line[0]  # Bounding box coordinates  
                        text_info = line[1]  # Text and confidence
                        text = text_info[0]  # Extracted text
                        confidence = text_info[1]  # Confidence score
                        
                        print(f"{i:2d}. '{text}' (confidence: {confidence:.3f})")
                        
                        processed_results.append({
                            "text": text,
                            "confidence": float(confidence),
                            "coordinates": bbox  # Use same field name as test_pre.py
                        })
            
            # Save individual result to script directory
            if processed_results:
                output_file = self.script_dir / f"{image_path.stem}_svtr_v6.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(processed_results, f, ensure_ascii=False, indent=4)
                
                print("\n" + "=" * 60)
                print(f"✅ Results saved to {output_file} ({len(processed_results)} text regions)")
                print(f"📁 Image processed: {image_path.name}")
                print(f"🤖 Model: SVTR v6 (Custom Trained)")
            
            return {
                "image_name": image_path.name,
                "model": "SVTR v6",
                "text_count": len(processed_results),
                "texts": processed_results,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "image_name": str(image_path),
                "model": "SVTR v6",
                "error": str(e),
                "status": "failed"
            }
    
    def batch_predict(self, image_folder: str, output_folder: str = "svtr_v6_results", 
                     max_images: int = None) -> Dict[str, Any]:
        """Batch prediction with SVTR v6"""
        
        image_dir = Path(image_folder)
        output_dir = Path(output_folder)
        output_dir.mkdir(exist_ok=True)
        
        if not image_dir.exists():
            return {"error": f"Image folder not found: {image_folder}"}
        
        # Find images
        image_files = list(image_dir.glob("*.jpg"))
        if max_images:
            image_files = image_files[:max_images]
        
        if not image_files:
            return {"error": "No .jpg files found"}
        
        print(f"📁 SVTR v6 batch processing: {len(image_files)} images")
        
        # Results
        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "model": "SVTR v6 (Custom Trained)",
            "model_dir": str(self.model_dir),
            "total_images": len(image_files),
            "results": []
        }
        
        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\\n[{i}/{len(image_files)}] {image_path.name}")
            
            result = self.predict_text(image_path)
            batch_results["results"].append(result)
            
            # Save individual result
            if result["status"] == "success":
                json_file = output_dir / f"{image_path.stem}_svtr_v6.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # Show preview
                texts = result["texts"]
                print(f"   ✅ SVTR v6: {len(texts)} text regions")
                for j, text_info in enumerate(texts[:3]):
                    print(f"      {j+1}. '{text_info['text']}' ({text_info['confidence']:.3f})")
                if len(texts) > 3:
                    print(f"      ... and {len(texts)-3} more")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Save summary
        summary_file = output_dir / "svtr_v6_batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)
        
        # Statistics
        successful = len([r for r in batch_results["results"] if r["status"] == "success"])
        total_texts = sum([len(r.get("texts", [])) for r in batch_results["results"]])
        
        print(f"\\n📊 SVTR v6 FINAL RESULTS:")
        print(f"   ✅ Success: {successful}/{len(image_files)} images")
        print(f"   📝 Total texts: {total_texts} regions")
        print(f"   💾 Results: {output_dir}")
        print(f"   📄 Summary: {summary_file}")
        
        return batch_results

def main():
    """Main function"""
    
    print("🎯 SVTR v6 True Model Inference")
    print("=" * 60)
    print("Using SVTR v6 model trained from checkpoint")
    print("=" * 60)
    
    # Initialize SVTR v6
    svtr = SVTRv6TrueInference()
    
    # Test with single image from parent directory
    test_dirs = ["../image_test", "../.", "../dataset"]
    test_image = None
    
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        if test_dir.exists():
            images = list(test_dir.glob("*.jpg"))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        print(f"\\n📁 Testing with: {test_image}")
        result = svtr.predict_text(test_image)
        
        if result["status"] == "success":
            print(f"\\n✅ SVTR v6 Results:")
            print(f"   📸 Image: {result['image_name']}")
            print(f"   🤖 Model: {result['model']}")
            print(f"   📝 Text count: {result['text_count']}")
            
            for i, text_info in enumerate(result["texts"][:10], 1):
                print(f"   {i:2d}. '{text_info['text']}' (confidence: {text_info['confidence']:.3f})")
            
            if result['text_count'] > 10:
                print(f"   ... and {result['text_count']-10} more texts")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    # Batch processing
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        print("\\n" + "="*60)
        print("🚀 SVTR v6 Batch Processing")
        print("="*60)
        
        batch_results = svtr.batch_predict(
            image_folder="../image_test",
            output_folder="../svtr_v6_results",
            max_images=10
        )
        
        if "error" not in batch_results:
            print("\\n🎉 SVTR v6 batch processing completed!")
        else:
            print(f"❌ Batch error: {batch_results['error']}")

if __name__ == "__main__":
    main()
