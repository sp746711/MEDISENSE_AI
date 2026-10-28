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


def test_auto_classifier_candidate_requires_independent_anatomy_verification(
    monkeypatch,
):
    """High classifier confidence identifies a candidate, not verified anatomy."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "completed",
                "body_part_label": "HAND",
                "candidate_body_part": "HAND",
                "body_part_confidence": 0.91,
                "confidence_threshold": 0.60,
                "min_confidence_margin": 0.10,
                "top_predictions": [
                    {"label": "HAND", "probability": 0.91},
                    {"label": "WRIST", "probability": 0.04},
                ],
                "confidence_threshold_validated": False,
                "message": "Detected HAND",
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    # The classifier completed and produced a specific candidate.
    assert result["classification_status"] == "completed"
    assert result["region"] == "hand"
    assert result["detected_region"] == "hand"
    assert result["region_source"] == "image_classifier"
    assert result["body_part_confidence"] == 0.91
    assert result["anatomy_detection"]["body_part_label"] == "HAND"

    # Confidence alone must not mark anatomy independently verified or select a model.
    assert result["anatomy_verification_status"] == "confirmation_required"
    assert result["anatomy_independently_verified"] is False
    assert result["supported"] is False
    assert result["registry_entry"] is None
    assert result["reason_code"] == "anatomy_confirmation_required"


def test_low_confidence_classifier_does_not_route_to_regional_model(monkeypatch):
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


def test_completed_result_below_threshold_is_rejected(monkeypatch):
    """A completed upstream status cannot bypass the router confidence gate."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "completed",
                "body_part_label": "HAND",
                "candidate_body_part": "HAND",
                "body_part_confidence": 0.59,
                "confidence_threshold": 0.60,
                "top_predictions": [
                    {"label": "HAND", "probability": 0.59},
                    {"label": "WRIST", "probability": 0.20},
                ],
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["region"] == "unknown"
    assert result["anatomy_verification_status"] == "unverified"
    assert result["reason_code"] == "anatomy_confidence_below_threshold"
    assert result["registry_entry"] is None


def test_ambiguous_top_two_predictions_are_rejected(monkeypatch):
    """Close leading scores must not automatically select a regional model."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "completed",
                "body_part_label": "HAND",
                "candidate_body_part": "HAND",
                "body_part_confidence": 0.72,
                "confidence_threshold": 0.60,
                "min_confidence_margin": 0.10,
                "top_predictions": [
                    {"label": "HAND", "probability": 0.72},
                    {"label": "WRIST", "probability": 0.68},
                    {"label": "ELBOW", "probability": 0.04},
                ],
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["region"] == "unknown"
    assert result["anatomy_verification_status"] == "unverified"
    assert result["reason_code"] == "ambiguous_anatomy_prediction"
    assert result["registry_entry"] is None


def test_broad_anatomy_label_does_not_select_narrow_model(monkeypatch):
    """A broad region such as UPPER_EXTREMITY is not a specific study."""
    monkeypatch.setattr(
        xray_router,
        "_BODY_PART_CLASSIFIER",
        _Classifier(
            {
                "status": "completed",
                "body_part_label": "UPPER_EXTREMITY",
                "candidate_body_part": "UPPER_EXTREMITY",
                "body_part_confidence": 0.91,
                "confidence_threshold": 0.60,
                "top_predictions": [
                    {"label": "UPPER_EXTREMITY", "probability": 0.91},
                    {"label": "CHEST", "probability": 0.03},
                ],
            }
        ),
    )

    result = xray_router.route_xray("unused-image.png")

    assert result["supported"] is False
    assert result["registry_entry"] is None
    assert result["reason_code"] == "broad_anatomy_label"
    assert result["anatomy_verification_status"] == "confirmation_required"
    assert result["anatomy_independently_verified"] is False


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


def test_explicit_legacy_region_path_does_not_invoke_auto_classifier(monkeypatch):
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
