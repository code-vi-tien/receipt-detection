import json
import os
from paddleocr import PaddleOCR

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

ocr = PaddleOCR(
    det_model_dir=r"model_son_ocr_1/ch_db_res18",
    rec=True,  # Enable text recognition
    use_angle_cls=False,
    use_gpu=False,  # CPU mode for compatibility
    lang='ch'  # Chinese language
)

img_path = "image_test/X51005719905.jpg"

result = ocr.ocr(img_path, cls=False)

output_data = []
if result and len(result[0]) > 0:
    print(f"🔍 Detected {len(result[0])} text regions:")
    print("=" * 60)
    
    for i, line in enumerate(result[0], 1):
        coords = line[0]  # Bounding box coordinates
        text_info = line[1]  # Text and confidence
        text = text_info[0]  # Extracted text
        confidence = text_info[1]  # Confidence score
        
        print(f"{i:2d}. '{text}' (confidence: {confidence:.3f})")
        
        output_data.append({
            "text": text,
            "confidence": confidence,
            "coordinates": coords
        })

# Save to script directory
output_file = os.path.join(script_dir, "output.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)

print("\n" + "=" * 60)
print(f"✅ Kết quả đã lưu vào {output_file} ({len(output_data)} text regions)")
print(f"📁 Image processed: {img_path}")
print(f"🤖 Model: Custom detection model + PaddleOCR recognition")
