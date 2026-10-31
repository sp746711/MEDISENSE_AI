import sys
from pathlib import Path
sys.path.insert(0, str(Path("backend").resolve()))
import onnxruntime as ort
import numpy as np
from PIL import Image

model_path = Path("backend/models/xray/bodypart_full/model.onnx")
classes_path = Path("backend/models/xray/bodypart_full/classes.txt")

print(f"Model exists: {model_path.exists()}, size: {model_path.stat().st_size if model_path.exists() else 0} bytes")
print(f"Classes file exists: {classes_path.exists()}")

classes = [line.strip() for line in classes_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
print(f"Number of classes in classes.txt: {len(classes)}")
print("Classes:", classes)

session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
inputs = session.get_inputs()
outputs = session.get_outputs()

print("\n--- ONNX Model Signature ---")
for inp in inputs:
    print(f"Input: name={inp.name}, shape={inp.shape}, type={inp.type}")
for out in outputs:
    print(f"Output: name={out.name}, shape={out.shape}, type={out.type}")

# Test with a dummy input tensor of zeros/ones
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
out = session.run([outputs[0].name], {inputs[0].name: dummy})[0]
print(f"\nSample output shape: {out.shape}")
print(f"Sample raw values min={out.min():.4f}, max={out.max():.4f}, sum={out.sum():.4f}")
# If sum is close to 1.0, it's already softmaxed. If not, it's raw logits.
is_softmax = np.isclose(out.sum(), 1.0, atol=1e-2)
print(f"Is output already softmaxed?: {is_softmax}")
