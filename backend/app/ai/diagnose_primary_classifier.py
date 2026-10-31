import os
import sys
import hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

backend_dir = Path(__file__).resolve().parents[2]
model_path = backend_dir / "models" / "xray" / "bodypart_full" / "model.onnx"
labels_path = backend_dir / "models" / "xray" / "bodypart_full" / "classes.txt"
uploads_dir = backend_dir / "uploads" / "xrays"

labels = [line.strip().upper() for line in labels_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

test_images = [
    {"name": "01_chest.png", "stored": "65de04ed2358482392b4f93d2a1ac6e6.png", "truth": "CHEST"},
    {"name": "02_skull.png", "stored": "7ec9ae95e9f34c27919ae9764c6e3ac9.png", "truth": "SKULL"},
    {"name": "03_cervical_spine.png", "stored": "b420b368e93a422e893adfea5d1fbb30.png", "truth": "CERVICAL_SPINE"},
    {"name": "04_shoulder.png", "stored": "931f6bda95934d6480f6995a0e08b18b.png", "truth": "SHOULDER"},
    {"name": "05_elbow.png", "stored": "c31df909386b4956a1a0c80f7ffb825f.png", "truth": "ELBOW"},
    {"name": "06_wrist.png", "stored": "9fa2c9d58fbf4a0f8ce91c34878df301.png", "truth": "WRIST"},
    {"name": "07_hand.png", "stored": "acf006c78480402a9e1f7295995b45dc.png", "truth": "HAND"},
    {"name": "08_knee.png", "stored": "10653bb005a8444a8c7f30b894af6bea.png", "truth": "KNEE"},
    {"name": "09_ankle.png", "stored": "54f31ae08f364399bd4d462285049246.png", "truth": "ANKLE"},
    {"name": "10_abdomen_kub.png", "stored": "867c1e5a0fc54221887a2906e48b9a66.png", "truth": "ABDOMEN"},
]

def preprocess(img, mode="center_crop", color_mode="RGB", norm="imagenet"):
    img = ImageOps.exif_transpose(img)
    if color_mode == "RGB":
        img = img.convert("RGB")
    elif color_mode == "BGR":
        img = img.convert("RGB")
        # swap channels later
    elif color_mode == "L":
        img = img.convert("L").convert("RGB")
    
    w, h = img.size
    if mode == "center_crop":
        scale = 224 / min(w, h)
        new_w = max(224, int(round(w * scale)))
        new_h = max(224, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
    elif mode == "direct_resize":
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if color_mode == "BGR":
        arr = arr[:, :, ::-1] # RGB to BGR
    
    if norm == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif norm == "0_1":
        pass
    elif norm == "minus1_1":
        arr = (arr - 0.5) / 0.5
        
    tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    return tensor

def run_diagnostics():
    print("=" * 80)
    print("PRIMARY 33-CLASS CLASSIFIER DETAILED FORENSIC AUDIT")
    print("=" * 80)
    
    # Check model hash
    model_bytes = model_path.read_bytes()
    print(f"Model path: {model_path}")
    print(f"Model size: {len(model_bytes)} bytes")
    print(f"Model SHA256: {hashlib.sha256(model_bytes).hexdigest()}")
    print(f"Classes count: {len(labels)}")
    print(f"Classes list:\n{labels}\n")

    print("=" * 80)
    print("RUNNING INFERENCE ON 10 SEPARATE TEST X-RAYS (PRODUCTION PREPROCESSING)")
    print("=" * 80)

    for item in test_images:
        path = uploads_dir / item["stored"]
        print(f"\n--- {item['name']} (Expected Truth: {item['truth']}) ---")
        if not path.is_file():
            print(f"File not found: {path}")
            continue
        
        with Image.open(path) as pil_img:
            tensor = preprocess(pil_img, mode="center_crop", color_mode="RGB", norm="imagenet")
            raw_out = session.run([output_name], {input_name: tensor})[0][0]
            
            # Check sum of probabilities
            prob_sum = float(np.sum(raw_out))
            top5_indices = np.argsort(raw_out)[::-1][:5]
            
            print(f"Raw output shape: {raw_out.shape}, sum: {prob_sum:.4f}")
            print("Top 5 Predictions:")
            for rank, idx in enumerate(top5_indices, 1):
                lbl = labels[idx]
                score = float(raw_out[idx])
                print(f"  {rank}. [{idx:02d}] {lbl:<18}: {score * 100:6.2f}%")
                
            top1_label = labels[top5_indices[0]]
            top1_score = float(raw_out[top5_indices[0]])
            top2_score = float(raw_out[top5_indices[1]])
            margin = top1_score - top2_score
            print(f"Margin (top1 - top2): {margin * 100:.2f}%")
            print(f"Matches Ground Truth: {top1_label == item['truth']} (Truth rank: {[labels[i] for i in np.argsort(raw_out)[::-1]].index(item['truth']) + 1 if item['truth'] in labels else 'N/A'})")

    print("\n" + "=" * 80)
    print("PREPROCESSING VARIATION EXPERIMENT ON FAILING CASES")
    print("=" * 80)
    
    variations = [
        ("center_crop + ImageNet", "center_crop", "RGB", "imagenet"),
        ("direct_resize + ImageNet", "direct_resize", "RGB", "imagenet"),
        ("center_crop + BGR", "center_crop", "BGR", "imagenet"),
        ("center_crop + [0, 1] norm", "center_crop", "RGB", "0_1"),
        ("center_crop + [-1, 1] norm", "center_crop", "RGB", "minus1_1"),
    ]
    
    failing_cases = [
        {"name": "05_elbow.png", "stored": "c31df909386b4956a1a0c80f7ffb825f.png", "truth": "ELBOW"},
        {"name": "09_ankle.png", "stored": "54f31ae08f364399bd4d462285049246.png", "truth": "ANKLE"},
        {"name": "03_cervical_spine.png", "stored": "b420b368e93a422e893adfea5d1fbb30.png", "truth": "CERVICAL_SPINE"},
    ]
    
    for item in failing_cases:
        path = uploads_dir / item["stored"]
        print(f"\nVariations for {item['name']} (Truth: {item['truth']}):")
        with Image.open(path) as pil_img:
            for vname, mode, cmode, norm in variations:
                t = preprocess(pil_img, mode=mode, color_mode=cmode, norm=norm)
                out = session.run([output_name], {input_name: t})[0][0]
                top1_idx = int(np.argmax(out))
                top1_lbl = labels[top1_idx]
                top1_val = float(out[top1_idx])
                truth_idx = labels.index(item["truth"]) if item["truth"] in labels else -1
                truth_val = float(out[truth_idx]) if truth_idx >= 0 else 0.0
                print(f"  {vname:<28} -> Top: {top1_lbl:<16} ({top1_val*100:5.1f}%) | Truth ({item['truth']}): {truth_val*100:5.1f}%")

if __name__ == "__main__":
    run_diagnostics()
