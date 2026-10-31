
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
    "SI_JOINT", "SKULL", "SPINE", "TEMPORAL_BONE", "TIBIA",
    "UPPER_EXTREMITY", "WRIST",
]


class FakeSession:
    """Mock ONNX Runtime session for offline unit tests."""

    def __init__(
        self,
        model_path,
        providers=None,
        probabilities=None,
        output_shape=None,
    ):
        self.model_path = model_path
        self.providers = providers
        self.probabilities = probabilities
        self.output_shape = output_shape or ["batch", 33]
        self.last_tensor = None

    def get_inputs(self):
        return [
            SimpleNamespace(
                name="images",
                shape=["batch", 3, 224, 224],
                type="tensor(float)",
            )
        ]

    def get_outputs(self):
        return [
            SimpleNamespace(
                name="probs",
                shape=self.output_shape,
                type="tensor(float)",
            )
        ]

    def run(self, output_names, input_feed):
        self.last_tensor = input_feed["images"]
        return [self.probabilities]


def _install_fake_runtime(monkeypatch, session):
    """Install a mocked ONNX Runtime session."""
    fake_ort = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        InferenceSession=lambda path, providers=None: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)


def _make_classifier(tmp_path, probabilities=None, threshold=0.6):
    """Create a classifier using temporary model and label files."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX for mocked test")

    labels_path = tmp_path / "classes.txt"
    labels_path.write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    probs = probabilities
    if probs is None:
        probs = np.full((1, 33), 0.08 / 32, dtype=np.float32)
        probs[0, LABELS.index("KNEE")] = 0.92

    session = FakeSession(
        str(model_path),
        probabilities=probs,
    )

    classifier = XRayAnatomyClassifier(
        checkpoint_path=model_path,
        labels_path=labels_path,
        min_confidence=threshold,
    )

    return classifier, session


def test_33_class_model_loads_and_predicts_knee(tmp_path, monkeypatch):
    """A high-confidence prediction returns the expected anatomy label."""
    classifier, session = _make_classifier(tmp_path)
    _install_fake_runtime(monkeypatch, session)

    classifier.load()
    result = classifier.predict_image(
        Image.new("RGB", (300, 200), color="gray")
    )

    assert result["status"] == "completed"
    assert result["body_part_label"] == "KNEE"
    assert result["candidate_body_part"] == "KNEE"
    assert result["body_part_confidence"] == pytest.approx(
        0.92, abs=1e-4
    )
    assert len(result["classifier_labels"]) == 33
    assert result["top_predictions"][0]["label"] == "KNEE"
    assert session.last_tensor.shape == (1, 3, 224, 224)
    assert session.last_tensor.dtype == np.float32


def test_low_confidence_result_is_unverified_and_has_no_routable_label(
    tmp_path, monkeypatch
):
    """Low-confidence predictions must not return a routable label."""
    probabilities = np.full((1, 33), 1 / 33, dtype=np.float32)

    classifier, session = _make_classifier(
        tmp_path,
        probabilities=probabilities,
        threshold=0.6,
    )
    _install_fake_runtime(monkeypatch, session)

    result = classifier.predict_image(
        Image.new("RGB", (224, 224), color="gray")
    )

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["candidate_body_part"] in LABELS



def test_default_preprocessing_matches_documented_model_transform(
    tmp_path, monkeypatch
):
    """Default is aspect-ratio-preserving letterbox, preserving peripheral anatomy."""
    classifier, session = _make_classifier(tmp_path)
    _install_fake_runtime(monkeypatch, session)
    classifier.load()

    gradient = np.tile(
        np.linspace(0, 255, 448, dtype=np.uint8), (224, 1)
    )
    image = Image.fromarray(gradient, mode="L").convert("RGB")
    tensor = classifier._preprocess(image)

    assert classifier.preprocessing_mode == "letterbox"
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor[0, 0, 112, 112] > -2.0
    assert tensor[0, 0, 112, -1] < 2.5


def test_direct_resize_is_available_only_for_comparison(tmp_path, monkeypatch):
    """Full-frame resize remains selectable for controlled A/B testing."""
    classifier, session = _make_classifier(tmp_path)
    classifier.preprocessing_mode = "direct_resize"
    _install_fake_runtime(monkeypatch, session)
    classifier.load()

    gradient = np.tile(
        np.linspace(0, 255, 448, dtype=np.uint8), (224, 1)
    )
    image = Image.fromarray(gradient, mode="L").convert("RGB")
    tensor = classifier._preprocess(image)

    assert classifier.preprocessing_mode == "direct_resize"
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor[0, 0, 100, 0] < -2.0
    assert tensor[0, 0, 100, -1] > 2.0


def test_invalid_preprocessing_mode_is_rejected(tmp_path):
    """Do not silently run an unknown model transform."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX for mocked test")
    labels_path = tmp_path / "classes.txt"
    labels_path.write_text("\n".join(LABELS) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported XRAY_ANATOMY_PREPROCESSING"):
        XRayAnatomyClassifier(
            checkpoint_path=model_path,
            labels_path=labels_path,
            preprocessing_mode="something_unknown",
        )

def test_wrong_output_shape_fails_closed(tmp_path, monkeypatch):
    """A model with the wrong number of output classes must be rejected."""
    classifier, session = _make_classifier(tmp_path)
    session.output_shape = ["batch", 7]
    _install_fake_runtime(monkeypatch, session)

    with pytest.raises(ValueError, match="output has 7 classes"):
        classifier.load()


def test_missing_or_wrong_model_extension_is_rejected(tmp_path):
    """An incompatible checkpoint extension must be rejected."""
    path = tmp_path / "old_mura_checkpoint.pth"
    path.write_bytes(b"not a full-body ONNX model")

    labels = tmp_path / "classes.txt"
    labels.write_text("\n".join(LABELS), encoding="utf-8")

    classifier = XRayAnatomyClassifier(
        checkpoint_path=path,
        labels_path=labels,
    )

    with pytest.raises(ValueError, match="requires model.onnx"):
        classifier.load()


def test_invalid_label_count_fails_before_runtime_load(
    tmp_path, monkeypatch
):
    """Invalid class labels must fail before loading the model."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX")

    labels = tmp_path / "classes.txt"
    labels.write_text("KNEE\nCHEST\n", encoding="utf-8")

    classifier = XRayAnatomyClassifier(
        checkpoint_path=model_path,
        labels_path=labels,
    )

    session = FakeSession(
        str(model_path),
        probabilities=np.zeros((1, 33), dtype=np.float32),
    )
    _install_fake_runtime(monkeypatch, session)

    with pytest.raises(ValueError, match="Expected 33 anatomy labels"):
        classifier.load()


def test_known_white_placeholder_is_rejected(tmp_path, monkeypatch):
    """Known white placeholder cards must not reach anatomy inference."""
    classifier, session = _make_classifier(tmp_path)
    _install_fake_runtime(monkeypatch, session)

    placeholder = Image.new("RGB", (600, 200), color="white")

    with pytest.raises(ValueError, match="blank/placeholder"):
        classifier.predict_image(placeholder)

    # Rejection should happen before the mocked ONNX model is called.
    assert session.last_tensor is None


def test_low_confidence_diagram_like_input_remains_unverified(
    tmp_path, monkeypatch
):
    """
    Verify low-confidence handling for a diagram-like test input.

    This test checks confidence handling only. It does not prove that
    the classifier can detect whether an image is a genuine radiograph.
    """
    probabilities = np.full((1, 33), 0.01, dtype=np.float32)

    probabilities[0, LABELS.index("UPPER_EXTREMITY")] = 0.4861
    probabilities[0, LABELS.index("PELVIS")] = 0.1363
    probabilities[0, LABELS.index("ABDOMEN")] = 0.0474
    probabilities[0, LABELS.index("LOWER_EXTREMITY")] = 0.0423
    probabilities[0, LABELS.index("FOREARM")] = 0.0264

    # Normalize the mock scores to a valid probability distribution.
    probabilities /= probabilities.sum()

    classifier, session = _make_classifier(
        tmp_path,
        probabilities=probabilities,
        threshold=0.6,
    )
    _install_fake_runtime(monkeypatch, session)

    # Synthetic input: no external dataset files are required.
    diagram_like_image = Image.new(
        "RGB",
        (900, 900),
        color="black",
    )

    result = classifier.predict_image(diagram_like_image)

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["candidate_body_part"] in LABELS
    assert result["confidence_threshold"] == pytest.approx(0.6)


def test_colored_overlay_cleanup_is_disabled_by_default(tmp_path):
    """Unvalidated annotation inpainting must be opt-in, not the production default."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake ONNX")
    labels_path = tmp_path / "classes.txt"
    labels_path.write_text("\\n".join(LABELS) + "\\n", encoding="utf-8")

    classifier = XRayAnatomyClassifier(
        checkpoint_path=model_path,
        labels_path=labels_path,
    )

    assert classifier.clean_colored_overlays is False


def test_close_top_two_scores_remain_unverified(tmp_path, monkeypatch):
    """A close top-two prediction must not be automatically marked completed."""
    probabilities = np.full((1, 33), 0.001, dtype=np.float32)
    probabilities[0, LABELS.index("WRIST")] = 0.36
    probabilities[0, LABELS.index("HAND")] = 0.34
    probabilities[0, LABELS.index("KNEE")] = 0.30
    probabilities /= probabilities.sum()

    classifier, session = _make_classifier(
        tmp_path,
        probabilities=probabilities,
        threshold=0.30,
    )
    _install_fake_runtime(monkeypatch, session)

    result = classifier.predict_image(Image.new("RGB", (224, 224), color="gray"))

    assert result["status"] == "unverified"
    assert result["body_part_label"] is None
    assert result["top_two_margin"] < result["top_two_margin_threshold"]
