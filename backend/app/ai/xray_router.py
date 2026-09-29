"""X-ray body-region router (Stage 7)."""

from typing import Any, Optional


SUPPORTED_REGIONS = {"chest"}  # Expand only when trained models exist


def route_xray(image_path: str, declared_region: Optional[str] = None) -> dict[str, Any]:
    region = (declared_region or "unknown").lower()
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
            "X-ray received successfully. Automated interpretation for this X-ray "
            "type is currently unavailable. If you have the associated radiology "
            "report, upload it for supported text-based analysis."
        ),
    }
