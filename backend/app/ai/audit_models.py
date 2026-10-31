import hashlib
import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

import onnx
import onnxruntime as ort
import torch

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier, _default_model_path, _default_labels_path
from app.ai.xray_router import _get_body_part_classifier

print("=" * 80)
print("MODEL PROVENANCE & ARCHITECTURE AUDIT")
print("=" * 80)

# 1. Check default model loaded by classifier
clf = _get_body_part_classifier()
loaded_path = clf.model_path
labels_path = clf.labels_path

print(f"Loaded model path by running backend: {loaded_path}")
print(f"Loaded labels path by running backend: {labels_path}")

# Checksums
def get_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()

print(f"model.onnx size: {loaded_path.stat().st_size:,} bytes")
print(f"model.onnx sha256: {get_sha256(loaded_path)}")
print(f"classes.txt size: {labels_path.stat().st_size} bytes")
print(f"classes.txt sha256: {get_sha256(labels_path)}")

# Read classes
with open(labels_path, "r", encoding="utf-8-sig") as f:
    classes = [line.strip() for line in f if line.strip()]
print(f"Total classes in classes.txt: {len(classes)}")

# Inspect ONNX model interface
onnx_model = onnx.load(str(loaded_path))
print(f"ONNX IR version: {onnx_model.ir_version}")
print(f"ONNX Producer: {onnx_model.producer_name} {onnx_model.producer_version}")

sess = ort.InferenceSession(str(loaded_path), providers=["CPUExecutionProvider"])
inputs = sess.get_inputs()
outputs = sess.get_outputs()

print("\n--- ONNX Interface ---")
for i, inp in enumerate(inputs):
    print(f"Input {i}: name='{inp.name}', shape={inp.shape}, type={inp.type}")

for i, out in enumerate(outputs):
    print(f"Output {i}: name='{out.name}', shape={out.shape}, type={out.type}")

# Check output layer in ONNX graph: does it have Softmax or is it Logits?
node_types = [n.op_type for n in onnx_model.graph.node]
last_nodes = [(n.op_type, n.name, n.input, n.output) for n in onnx_model.graph.node[-5:]]
print("\nLast 5 graph nodes:")
for item in last_nodes:
    print(f"  Op: {item[0]}, Name: {item[1]}, In: {item[2]}, Out: {item[3]}")
print(f"Contains Softmax in graph: {'Softmax' in node_types}")

# Inspect best_model.pth if present
pth_path = backend_dir / "models" / "xray" / "bodypart" / "best_model.pth"
if pth_path.is_file():
    print("\n--- PyTorch Checkpoint (bodypart/best_model.pth) ---")
    print(f"pth size: {pth_path.stat().st_size:,} bytes")
    print(f"pth sha256: {get_sha256(pth_path)}")
    try:
        checkpoint = torch.load(str(pth_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            if "state_dict" in checkpoint:
                sd = checkpoint["state_dict"]
                print(f"state_dict num keys: {len(sd)}")
                layer_names = list(sd.keys())
                print(f"First 3 layers: {layer_names[:3]}")
                print(f"Last 3 layers: {layer_names[-3:]}")
                # check final layer shape
                for k in reversed(layer_names):
                    if "weight" in k:
                        print(f"Final weight layer '{k}' shape: {sd[k].shape}")
                        break
            elif "model" in checkpoint:
                print(f"Model key in checkpoint: {type(checkpoint['model'])}")
            else:
                # Direct state dict
                layer_names = list(checkpoint.keys())
                print(f"Direct state dict num keys: {len(layer_names)}")
                print(f"First 3 layers: {layer_names[:3]}")
                print(f"Last 3 layers: {layer_names[-3:]}")
                for k in reversed(layer_names):
                    if "weight" in k:
                        print(f"Final weight layer '{k}' shape: {checkpoint[k].shape}")
                        break
        else:
            print(f"Checkpoint object type: {type(checkpoint)}")
    except Exception as e:
        print(f"Error reading pth: {e}")
