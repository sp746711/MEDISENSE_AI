
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier

root = Path(__file__).resolve().parent
image_path = (
    root / "uploads" / "xrays"
    / "84c1796fb060444ea44fcadcbbe867e3.jpg"
)

classifier = XRayAnatomyClassifier()
classifier.load()

session = classifier.session

print("Model:", classifier.model_path)
print("Model size:", classifier.model_path.stat().st_size)
print("Providers:", session.get_providers())
print("Inputs:", [(x.name, x.shape, x.type) for x in session.get_inputs()])
print("Outputs:", [(x.name, x.shape, x.type) for x in session.get_outputs()])
print("Metadata:", session.get_modelmeta().custom_metadata_map)
print("First and last labels:", classifier.classes[:5], classifier.classes[-5:])

if not image_path.is_file():
    raise FileNotFoundError(
        f"Image not found: {image_path}. "
        "Update image_path to the audited file's actual location."
    )

with Image.open(image_path) as image:
    tensor = classifier._preprocess(image)

raw = np.asarray(
    session.run(
        [classifier.output_name],
        {classifier.input_name: tensor},
    )[0]
)[0]

print("Raw output shape:", raw.shape)
print("Raw output min/max:", float(raw.min()), float(raw.max()))
print("Raw output sum:", float(raw.sum()))
print("All finite:", bool(np.isfinite(raw).all()))
print("First 10 raw scores:", raw[:10])

print("\nTop raw output indices:")
for index in np.argsort(raw)[::-1][:10]:
    print(
        int(index),
        classifier.classes[int(index)],
        float(raw[index]),
    )

print("\nCurrent classifier result:")
print(classifier.predict_file(image_path))
