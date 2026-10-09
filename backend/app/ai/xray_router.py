"""X-ray body-region router (Stage 7)."""

from typing import Any, Optional


SUPPORTED_REGIONS = {"chest"}  # Expand only when trained models exist


def route_xray(image_path: str, declared_region: Optional[str] = None) -> dict[str, Any]:
    raw = (declared_region or "").strip().lower()
    region = raw if raw else "unknown"
    if region in SUPPORTED_REGIONS:
        return {
            "region": region,
            "supported": True,
            "message": f"Routed to {region} model adapter.",
        }
    return {
        "region": region,
        "supported": False,
        "message": (
            "X-ray received. Automated interpretation for the selected "
            "X-ray type is currently unavailable."
        ),
    }
