
"""Train/export a dedicated 7-class anatomy classifier from the MURA directory.

Expected dataset layout (standard MURA-v1.1 layout):
  DATA_ROOT/train/XR_ELBOW/patient.../study.../image*.png
  DATA_ROOT/train/XR_FINGER/...
  DATA_ROOT/train/XR_FOREARM/...
  DATA_ROOT/train/XR_HAND/...
  DATA_ROOT/train/XR_HUMERUS/...
  DATA_ROOT/train/XR_SHOULDER/...
  DATA_ROOT/train/XR_WRIST/...

This trains BODY-PART classification, not normal/abnormal detection.
Patient-grouped validation is exploratory, not clinical validation.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASSES = (
    "ELBOW",
    "FINGER",
    "FOREARM",
    "HAND",
    "HUMERUS",
    "SHOULDER",
    "WRIST",
)
MURA_DIRS = {f"XR_{name}": name for name in CLASSES}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def collect_images(dataset_root: Path) -> list[dict[str, Any]]:
    train_root = dataset_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(
            f"Expected MURA training directory: {train_root}"
        )

    rows: list[dict[str, Any]] = []

    for class_folder, label in MURA_DIRS.items():
        root = train_root / class_folder

        if not root.is_dir():
            raise FileNotFoundError(
                f"MURA class folder not found: {root}"
            )

        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in IMAGE_EXTENSIONS
            ):
                continue

            patient = next(
                (
                    part
                    for part in path.parts
                    if re.fullmatch(
                        r"patient\d+", part, flags=re.I
                    )
                ),
                None,
            )

            if patient is None:
                patient = (
                    f"{class_folder}:{path.parent.as_posix()}"
                )

            rows.append(
                {
                    "path": path,
                    "label": CLASSES.index(label),
                    "class": label,
                    "patient": patient,
                }
            )

    if not rows:
        raise ValueError(
            f"No image files found under {train_root}"
        )

    counts = Counter(row["class"] for row in rows)
    missing = [
        name for name in CLASSES if counts[name] == 0
    ]

    if missing:
        raise ValueError(
            f"Missing training images for classes: {missing}"
        )

    return rows


def grouped_split(
    rows: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
):
    """Split by patient groups to reduce image-level leakage."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        groups[row["patient"]].append(row)

    group_names = list(groups)
    random.Random(seed).shuffle(group_names)

    target_count = max(
        1, round(len(rows) * validation_fraction)
    )
    validation_groups: set[str] = set()
    val_count = Counter()

    for group in group_names:
        if sum(val_count.values()) >= target_count:
            break

        validation_groups.add(group)

        for row in groups[group]:
            val_count[row["class"]] += 1

    train_rows = [
        row for row in rows
        if row["patient"] not in validation_groups
    ]
    val_rows = [
        row for row in rows
        if row["patient"] in validation_groups
    ]

    train_counts = Counter(
        row["class"] for row in train_rows
    )
    val_counts = Counter(
        row["class"] for row in val_rows
    )

    if any(
        train_counts[c] == 0 or val_counts[c] == 0
        for c in CLASSES
    ):
        raise ValueError(
            "Patient-grouped split does not include every class "
            "in both train and validation. "
            f"Train={dict(train_counts)} "
            f"Validation={dict(val_counts)}. "
            "Increase data or change --seed."
        )

    return train_rows, val_rows


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        type=Path,
        help="MURA dataset root containing train/XR_*/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "models"
            / "xray"
            / "upper_extremity"
        ),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4
    )
    parser.add_argument(
        "--validation-fraction", type=float, default=0.15
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error(
            "--epochs and --batch-size must be positive"
        )

    if not 0.05 <= args.validation_fraction <= 0.4:
        parser.error(
            "--validation-fraction must be between 0.05 and 0.4"
        )

    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image, ImageOps
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
        )
    except Exception as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install torch, "
            "torchvision, Pillow, numpy and scikit-learn."
        ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = collect_images(args.data_root)
    train_rows, val_rows = grouped_split(
        rows, args.validation_fraction, args.seed
    )

    print(
        f"Images: train={len(train_rows)} "
        f"validation={len(val_rows)}; classes={CLASSES}"
    )
    print(
        "Train class counts:",
        dict(Counter(row["class"] for row in train_rows)),
    )
    print(
        "Validation class counts:",
        dict(Counter(row["class"] for row in val_rows)),
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.RandomRotation(degrees=7),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((SIZE, SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    class MuraBodyPartDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            row = self.items[index]

            with Image.open(row["path"]) as opened:
                img = ImageOps.exif_transpose(
                    opened
                ).convert("RGB")
                x = self.transform(img)

            return x, int(row["label"])

    train_loader = DataLoader(
        MuraBodyPartDataset(train_rows, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    val_loader = DataLoader(
        MuraBodyPartDataset(val_rows, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    weights = (
        models.DenseNet121_Weights.DEFAULT
        if args.pretrained
        else None
    )
    model = models.densenet121(weights=weights)
    model.classifier = nn.Linear(
        model.classifier.in_features, len(CLASSES)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5,
    )
    criterion = nn.CrossEntropyLoss()

    best_f1 = -1.0
    best_state = None
    best_report = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += (
                float(loss.item()) * labels.size(0)
            )
            seen += labels.size(0)

        model.eval()
        actual, predicted = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                logits = model(images.to(device))
                predicted.extend(
                    logits.argmax(dim=1).cpu().tolist()
                )
                actual.extend(labels.tolist())

        val_accuracy = float(
            accuracy_score(actual, predicted)
        )
        val_macro_f1 = float(
            f1_score(
                actual,
                predicted,
                average="macro",
                labels=list(range(len(CLASSES))),
                zero_division=0,
            )
        )

        scheduler.step(val_macro_f1)

        print(
            f"epoch={epoch}/{args.epochs} "
            f"train_loss={running_loss / max(1, seen):.4f} "
            f"val_accuracy={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_f1:
            best_f1 = val_macro_f1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            best_report = {
                "accuracy": val_accuracy,
                "macro_f1": val_macro_f1,
                "classification_report": classification_report(
                    actual,
                    predicted,
                    labels=list(range(len(CLASSES))),
                    target_names=CLASSES,
                    output_dict=True,
                    zero_division=0,
                ),
                "confusion_matrix": confusion_matrix(
                    actual,
                    predicted,
                    labels=list(range(len(CLASSES))),
                ).tolist(),
                "train_counts": dict(
                    Counter(
                        row["class"] for row in train_rows
                    )
                ),
                "validation_counts": dict(
                    Counter(
                        row["class"] for row in val_rows
                    )
                ),
                "split": (
                    "patient_grouped_random_validation_from_"
                    "MURA_train_split"
                ),
                "seed": args.seed,
            }

    if best_state is None or best_report is None:
        raise RuntimeError(
            "Training completed without a usable checkpoint"
        )

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    model.load_state_dict(best_state)
    model.cpu().eval()

    checkpoint = (
        args.output_dir
        / "bodypart7_densenet121_state_dict.pt"
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_order": list(CLASSES),
            "architecture": "torchvision.densenet121",
            "preprocessing": {
                "resize": [SIZE, SIZE],
                "color_mode": "RGB",
                "mean": MEAN,
                "std": STD,
            },
        },
        checkpoint,
    )

    class LogitWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    onnx_path = (
        args.output_dir
        / "bodypart7_densenet121.onnx"
    )

    dummy = torch.zeros(
        1, 3, SIZE, SIZE, dtype=torch.float32
    )

    torch.onnx.export(
        LogitWrapper(model),
        dummy,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    (args.output_dir / "classes.txt").write_text(
        "\n".join(CLASSES) + "\n",
        encoding="utf-8",
    )

    report = {
        "model_name": (
            "densenet121_mura_upper_extremity_7class"
        ),
        "class_order": list(CLASSES),
        "preprocessing": {
            "resize": [SIZE, SIZE],
            "color_mode": "RGB",
            "mean": MEAN,
            "std": STD,
            "interpolation": "PIL bilinear",
            "exif_orientation_correction": True,
        },
        "checkpoint": checkpoint.name,
        "onnx": onnx_path.name,
        "validation": best_report,
        "warning": (
            "Exploratory validation only. Not clinically validated. "
            "Use a separate held-out test set before accuracy claims "
            "or production use."
        ),
    }

    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved ONNX model: {onnx_path}")
    print(f"Saved class labels: {args.output_dir / 'classes.txt'}")
    print(
        "Saved validation report: "
        f"{args.output_dir / 'validation_report.json'}"
    )
    print(
        "Important: this validation split is not an independent "
        "test set and does not establish clinical performance."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
