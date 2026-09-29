"""Controlled specialty mapping (Stage 11)."""

from typing import Optional

# Controlled mappings — LLM must not invent specialties freely.
SPECIALTY_MAP = {
    "respiratory": ["Pulmonology", "Internal Medicine"],
    "cardiovascular": ["Cardiology"],
    "skin": ["Dermatology"],
    "bones": ["Orthopedics"],
    "joints": ["Orthopedics"],
    "neurological": ["Neurology"],
    "gi": ["Gastroenterology"],
    "endocrine": ["Endocrinology"],
    "thyroid": ["Endocrinology"],
    "eye": ["Ophthalmology"],
    "ent": ["ENT"],
    "dental": ["Dentistry"],
    "general": ["General Physician", "Internal Medicine"],
}


def map_specialty(domains: list[str] | None) -> dict:
    if not domains:
        return {
            "suggested_specialty": None,
            "candidates": [],
            "message": "Insufficient evidence for a suggested specialty.",
            "wording": "Suggested specialty",
        }

    candidates: list[str] = []
    for domain in domains:
        key = domain.strip().lower()
        for specialty in SPECIALTY_MAP.get(key, []):
            if specialty not in candidates:
                candidates.append(specialty)

    if not candidates:
        candidates = list(SPECIALTY_MAP["general"])

    return {
        "suggested_specialty": candidates[0],
        "candidates": candidates,
        "message": f"Suggested specialty: {candidates[0]}",
        "wording": "Suggested specialty",
    }


def primary_suggestion(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    result = map_specialty([domain])
    return result.get("suggested_specialty")


def map_evidence_to_specialty(evidence: dict | None) -> dict:
    """Map structured multimodal evidence to suggested specialty."""
    if not evidence or evidence.get("evidence_state") == "INSUFFICIENT_EVIDENCE":
        return {
            "suggested_specialty": None,
            "candidates": [],
            "message": "Insufficient evidence for a suggested specialty.",
            "wording": "Suggested specialty",
        }

    domains: list[str] = []

    # From symptoms
    symptoms = (evidence.get("symptoms") or {}).get("present", [])
    for s in symptoms:
        d = s.get("domain")
        if d and d not in domains:
            domains.append(d)

    # From reports
    reports = (evidence.get("report") or {}).get("abnormal_findings", [])
    for r in reports:
        d = r.get("domain")
        if d and d not in domains:
            domains.append(d)

    # From X-ray
    xray = evidence.get("xray") or {}
    if xray.get("prediction"):
        reg = (xray.get("region") or "").lower()
        if "chest" in reg and "respiratory" not in domains:
            domains.append("respiratory")
        elif ("bone" in reg or "joint" in reg) and "bones" not in domains:
            domains.append("bones")

    return map_specialty(domains if domains else ["general"])
