from app.ai.xray_model import XRayModelService


def test_xray_unavailable_without_checkpoint():
    service = XRayModelService()
    result = service.analyze("nonexistent.jpg")
    assert result["status"] == "unavailable"
    assert result["prediction"] is None
    assert "unavailable" in result["message"].lower() or "not" in result["message"].lower()
