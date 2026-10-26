"""Offline unit tests for the 33-class ONNX X-ray anatomy classifier."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.ai.xray_anatomy_classifier import XRayAnatomyClassifier

LABELS = [
    "ABDOMEN", "ANKLE", "ARM", "CERVICAL_SPINE", "CHEST", "DORSAL_SPINE",
    "ELBOW", "EXTREMITY", "FEMUR", "FINGER", "FOOT", "FOREARM", "HAND",
    "HEAD", "HEEL", "HIP", "KNEE", "KUB", "LEG", "LOWER_EXTREMITY",
    "LUMBAR_SPINE", "NASOPHARYNX", "NECK", "PELVIS", "PNS", "SHOULDER",
    "SI_JOINT", "SKULL", "SPINE", "TEMPORAL_BONE", "TIBIA", "UPPER_EXTREMITY",
    "WRIST",
]


class FakeSession:
    def __init__(self, model_path, providers=None, probabilities=None, output_shape=None):
        self.model_path = model_path
        self.providers = providers
        self.probabilities = probabilities
        self.output_shape = output_shape or ["batch", 33]
        self.last_tensor = None

    def get_inputs(self):
        return [SimpleNamespace(name="images", shape=["batch", 3, 224, 224], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="probs", shape=self.output_shape, type="tensor(float)")]

    def run(self, output_names, input_feed):
        self.last_tensor = input_feed["images"]
        return [self.probabilities]


def _install_fake_runtime(monkeypatch, session):
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers=None: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def _make_classifier(tmp_path, probabilities=None, threshold=0.6):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX for mocked test")
    labels_path = tmp_path / "classes.txt"
    labels_path.write_text("\n".join(LABELS) + "\n", encoding="utf-8")
    probs = probabilities
    if probs is None:
        probs = np.full((1, 33), 0.08 / 32, dtype=np.float32)
        probs[0, LABELS.index("KNEE")] = 0.92
    session = FakeSession(str(model_path), probabilities=probs)
    return XRayAnatomyClassifier(
        checkpoint_path=model_path,
        labels_path=labels_path,
        min_confidence=threshold,
    ), session


def test_33_class_model_loads_and_predicts_knee(tmp_path, monkeypatch):
    classifier, session = _make_classifier(tmp_path)
    _install_fake_runtime(monkeypatch, session)

    classifier.load()
    result = classifier.predict_image(Image.new("RGB", (300, 200), color="gray"))

    assert result["status"] == "completed"
    assert result["body_part_label"] == "KNEE"
    assert result["candidate_body_part"] == "KNEE"
    assert result["body_part_confidence"] == pytest.approx(0.92, abs=1e-4)
    assert len(result["classifier_labels"]) == 33
    assert result["top_predictions"][0]["label"] == "KNEE"
    assert session.last_tensor.shape == (1, 3, 224, 224)
    assert session.last_tensor.dtype == np.float32


def test_low_confidence_result_is_unverified_and_has_no_routable_label(tmp_path, monkeypatch):
    probabilities = np.full((1, 33), 1 / 33, dtype=np.float32)
    classifier, session = _make_classifier(tmp_path, probabilities=probabilities, threshold=0.6)
    _install_fake_runtime(monkeypatch, session)

    result = classifier.predict_image(Image.new("RGB", (224, 224), color="gray"))

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["candidate_body_part"] in LABELS


def test_wrong_output_shape_fails_closed(tmp_path, monkeypatch):
    classifier, session = _make_classifier(tmp_path)
    session.output_shape = ["batch", 7]
    _install_fake_runtime(monkeypatch, session)

    with pytest.raises(ValueError, match="output has 7 classes"):
        classifier.load()


def test_missing_or_wrong_model_extension_is_rejected(tmp_path):
    path = tmp_path / "old_mura_checkpoint.pth"
    path.write_bytes(b"not a full-body ONNX model")
    labels = tmp_path / "classes.txt"
    labels.write_text("\n".join(LABELS), encoding="utf-8")
    classifier = XRayAnatomyClassifier(checkpoint_path=path, labels_path=labels)

    with pytest.raises(ValueError, match="requires model.onnx"):
        classifier.load()


def test_invalid_label_count_fails_before_runtime_load(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX")
    labels = tmp_path / "classes.txt"
    labels.write_text("KNEE\nCHEST\n", encoding="utf-8")
    classifier = XRayAnatomyClassifier(checkpoint_path=model_path, labels_path=labels)
    session = FakeSession(str(model_path), probabilities=np.zeros((1, 33), dtype=np.float32))
    _install_fake_runtime(monkeypatch, session)

    with pytest.raises(ValueError, match="Expected 33 anatomy labels"):
        classifier.load()
