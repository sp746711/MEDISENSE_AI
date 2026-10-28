"""Focused regression tests for Goal 1 broad anatomy routing.

These tests stub project configuration/logging so they can validate routing
logic without loading real X-ray weights or making clinical predictions.

A successful classifier result is not the same as independent anatomy
verification. Broad or unsupported regions must remain unverified and must
not select a region-specific image interpretation model.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "app" / "ai" / "xray_router.py"


@pytest.fixture
def router(monkeypatch):
    app = types.ModuleType("app")
    app.__path__ = []
    core = types.ModuleType("app.core")
    core.__path__ = []
    config = types.ModuleType("app.core.config")
    config.get_settings = lambda: SimpleNamespace(
        xray_default_model="chest_resnet18",
        xray_chest_checkpoint="",
        xray_model_path=str(SOURCE.parent),
    )
    utils = types.ModuleType("app.utils")
    utils.__path__ = []
    logging_mod = types.ModuleType("app.utils.logging")
    logging_mod.get_logger = lambda name: logging.getLogger(name)

    for name, mod in {
        "app": app,
        "app.core": core,
        "app.core.config": config,
        "app.utils": utils,
        "app.utils.logging": logging_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    module_name = "xray_router_goal1_under_test"
    spec = importlib.util.spec_from_file_location(module_name, SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


class FakeClassifier:
    def __init__(self, result):
        self.result = result

    def predict_file(self, image_path):
        return dict(self.result)


def _result(label: str, confidence: float = 0.8):
    return {
        "status": "completed",
        "body_part_label": label,
        "candidate_body_part": label,
        "body_part_confidence": confidence,
        "top_predictions": [{"label": label, "probability": confidence}],
        "message": "Test classifier result",
    }


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("UPPER_EXTREMITY", "upper extremity"),
        ("LOWER_EXTREMITY", "lower extremity"),
        ("EXTREMITY", "extremity"),
        ("ARM", "upper extremity"),
        ("LEG", "lower extremity"),
        ("HEAD", "head region"),
    ],
)
def test_broad_anatomy_labels_never_route_to_narrow_model(router, label, expected):
    """Broad labels are recognized candidates, not verified study regions."""
    router._BODY_PART_CLASSIFIER = FakeClassifier(_result(label))

    result = router.route_xray("unused-test-image.png")

    assert result["region"] == expected
    assert result["reason_code"] == "broad_anatomy_label"
    assert result["is_recognized"] is True
    assert result["is_available"] is False
    assert result["classification_status"] == "completed"
    assert result["anatomy_verification_status"] == "confirmation_required"
    assert result["anatomy_independently_verified"] is False
    assert result["registry_entry"] is None
    assert "No regional disease model was run" in result["message"]


def test_specific_but_unconfigured_knee_is_recognized_not_bad_label(router):
    """Recognizing knee anatomy must not imply verification or model availability."""
    router._BODY_PART_CLASSIFIER = FakeClassifier(_result("KNEE", 0.91))

    result = router.route_xray("unused-test-image.png")

    assert result["region"] == "knee"
    assert result["is_recognized"] is True
    assert result["is_available"] is False
    assert result["reason_code"] == "unsupported_region"
    assert result["classification_status"] == "completed"
    assert result["anatomy_verification_status"] == "confirmation_required"
    assert result["anatomy_independently_verified"] is False
    assert result["registry_entry"] is None
    assert result["body_part_confidence"] == 0.91
    assert "no compatible region-specific" in result["message"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("PNS", "sinuses"),
        ("NASOPHARYNX", "nasopharynx"),
        ("SI_JOINT", "sacroiliac joint"),
        ("TEMPORAL_BONE", "temporal bone"),
    ],
)
def test_classifier_catalog_labels_are_recognized(router, label, expected):
    canonical, recognized, is_broad = router._normalize_anatomy_classifier_label(label)
    assert canonical == expected
    assert recognized is True
    assert is_broad is False


def test_low_confidence_remains_unverified_and_does_not_route(router):
    router._BODY_PART_CLASSIFIER = FakeClassifier(
        {
            "status": "unverified",
            "body_part_label": None,
            "candidate_body_part": "HAND",
            "body_part_confidence": 0.4437,
            "message": "Below provisional review threshold",
        }
    )

    result = router.route_xray("unused-test-image.png")

    assert result["reason_code"] == "anatomy_unverified"
    assert result["anatomy_verification_status"] == "unverified"
    assert result["is_available"] is False
    assert result["registry_entry"] is None
