from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "xray" / "bodypart_full" / "model.onnx"
LABELS = ROOT / "models" / "xray" / "bodypart_full" / "classes.txt"
CSV = ROOT / "xray_goal1_unique_nonplaceholder_audit.csv"
IMAGE_DIR = ROOT / "uploads" / "xrays"
SIZE = 224

classes = [
    line.strip().upper()
    for line in LABELS.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]
if len(classes) != 33:
    raise RuntimeError(f"Expected 33 labels; found {len(classes)}")

session = ort.InferenceSession(
    str(MODEL), providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(image, mode):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if mode == "center_crop":
        width, height = image.size
        scale = SIZE / min(width, height)
        new_size = (
            max(SIZE, int(round(width * scale))),
            max(SIZE, int(round(height * scale))),
        )
        image = image.resize(new_size, Image.Resampling.BILINEAR)
        left = (new_size[0] - SIZE) // 2
        top = (new_size[1] - SIZE) // 2
        image = image.crop((left, top, left + SIZE, top + SIZE))
    elif mode == "direct_resize":
        image = image.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    else:
        raise ValueError(mode)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(arr, (2, 0, 1))[None].astype(np.float32)

def predict(image, mode):
    raw = np.asarray(
        session.run([output_name], {
            input_name: preprocess(image, mode)
        })[0],
        dtype=np.float32,
    )[0]

    # Match the production classifier's current score interpretation.
    if (
        np.isfinite(raw).all()
        and (raw >= 0).all()
        and (raw <= 1).all()
        and abs(float(raw.sum()) - 1.0) <= 0.03
    ):
        probs = raw
    else:
        shifted = raw - raw.max()
        exps = np.exp(shifted)
        probs = exps / exps.sum()

    order = np.argsort(probs)[::-1]
    return (
        classes[int(order[0])],
        float(probs[order[0]]),
        classes[int(order[1])],
        float(probs[order[1]]),
    )

import csv

if not CSV.is_file():
    raise FileNotFoundError(CSV)

rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig", newline="")))
seen = set()

print("DIAGNOSTIC ONLY — NOT A MODEL ACCURACY TEST")
print("Original classifier and model files are not modified.\n")

for row in rows:
    name = row.get("filename", "")
    digest = row.get("sha256", "")
    if not name or digest in seen:
        continue
    seen.add(digest)

    path = IMAGE_DIR / name
    if not path.is_file():
        print(f"{name}: FILE NOT FOUND")
        continue

    try:
        with Image.open(path) as image:
            crop = predict(image, "center_crop")
            resize = predict(image, "direct_resize")

        changed = crop[0] != resize[0]
        print(
            f"{name}\n"
            f"  center crop:   {crop[0]} {crop[1]:.2%}; "
            f"runner-up {crop[2]} {crop[3]:.2%}\n"
            f"  direct resize: {resize[0]} {resize[1]:.2%}; "
            f"runner-up {resize[2]} {resize[3]:.2%}\n"
            f"  top class changed: {changed}\n"
        )
    except Exception as exc:
        print(f"{name}: ERROR: {type(exc).__name__}: {exc}")

print(f"Unique image hashes processed: {len(seen)}")
