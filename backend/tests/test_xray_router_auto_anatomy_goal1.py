"""Focused Goal 1 tests for automatic body-part routing."""

from app.ai import xray_router


class _Classifier:
    """Mock automatic anatomy classifier for isolated router tests."""

    def __init__(self, result=None, error=None):
        self.result = result or {}
        self.error = error

    def predict_file(self, image_path):
        if self.error:
            raise self.error
        return dict(self.result)


def test_auto_classifier_routes_using_detected_body_part(monkeypatch):
    """A confident image-classifier result should select the matching region."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "completed",
                "body_part_label": "HAND",
                "candidate_body_part": "HAND",
                "body_part_confidence": 0.91,
                "confidence_threshold_validated": False,
                "message": "Detected HAND",
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["region"] == "hand"
    assert result["detected_region"] == "hand"
    assert result["region_source"] == "image_classifier"
    assert result["anatomy_verification_status"] == "completed"
    assert result["body_part_confidence"] == 0.91
    assert result["anatomy_detection"]["body_part_label"] == "HAND"


def test_low_confidence_classifier_does_not_route_to_regional_model(
    monkeypatch,
):
    """An unverified classification must not select a regional model."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "unverified",
                "body_part_label": None,
                "candidate_body_part": "WRIST",
                "body_part_confidence": 0.42,
                "message": "Low-confidence result",
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["region"] == "unknown"
    assert result["anatomy_verification_status"] == "unverified"
    assert result["reason_code"] == "anatomy_unverified"
    assert result["candidate_region"] == "wrist"
    assert result["registry_entry"] is None


def test_classifier_load_error_returns_safe_unavailable_state(monkeypatch):
    """If the classifier cannot load, routing must fail safely."""
    def fail_load():
        raise FileNotFoundError("test checkpoint missing")

    monkeypatch.setattr(
        xray_router,
        "_get_body_part_classifier",
        fail_load,
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["region"] == "unknown"
    assert result["anatomy_verification_status"] == "unavailable"
    assert result["reason_code"] == "anatomy_classifier_unavailable"
    assert result["supported"] is False


def test_explicit_legacy_region_path_does_not_invoke_auto_classifier(
    monkeypatch,
):
    """Explicit-region compatibility must not call the automatic classifier."""
    def should_not_load():
        raise AssertionError(
            "Classifier should not be called for explicit legacy routing"
        )

    monkeypatch.setattr(
        xray_router,
        "_get_body_part_classifier",
        should_not_load,
    )

    result = xray_router.route_xray(
        "unused-image.png",
        declared_region="knee",
    )

    assert result["region"] == "knee"
    assert result["supported"] is False
    assert result["reason_code"] == "unsupported_region"