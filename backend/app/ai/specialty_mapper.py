"""Controlled specialty mapping for MediSense AI.

Maps validated multimodal evidence to appropriate clinical specialties:
- Headache-dominant evidence -> Neurology
- Respiratory-dominant evidence -> Pulmonology / Internal Medicine
- GI / Abdominal pain / Liver findings -> Gastroenterology / Hepatology
- Endocrine / Glucose -> Endocrinology
- Kidney / Renal -> Nephrology
- Multi-systemic / General -> General Physician / Internal Medicine

Never invents diagnoses. Specialty recommendation represents a clinical navigation step.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

SPECIALTY_MAP: dict[str, list[str]] = {
    "neurological": ["Neurology", "General Physician"],
    "respiratory": ["Pulmonology", "Internal Medicine"],
    "cardiovascular": ["Cardiology", "Internal Medicine"],
    "gi": ["Gastroenterology", "Internal Medicine"],
    "liver": ["Gastroenterology", "Hepatology"],
    "kidney": ["Nephrology", "Internal Medicine"],
    "endocrine": ["Endocrinology", "Internal Medicine"],
    "thyroid": ["Endocrinology", "Internal Medicine"],
    "skin": ["Dermatology"],
    "bones": ["Orthopedics"],
    "joints": ["Orthopedics"],
    "hematology": ["Hematology", "General Physician"],
    "inflammation": ["Internal Medicine", "General Physician"],
    "eye": ["Ophthalmology"],
    "ent": ["ENT"],
    "dental": ["Dentistry"],
    "systemic": ["General Physician", "Internal Medicine"],
    "general": ["General Physician", "Internal Medicine"],
}


def map_specialty(domains: list[str] | None) -> dict[str, Any]:
    """Map domain list to prioritized candidates."""
    if not domains:
        return {
            "suggested_specialty": "General Physician / Internal Medicine",
            "candidates": ["General Physician", "Internal Medicine"],
            "message": "Suggested specialty based on general evaluation.",
            "wording": "Suggested specialty",
        }

    # Rank domains by frequency, giving specific organ systems priority over generic 'systemic'
    counts = Counter([d.strip().lower() for d in domains if d])
    
    # Sort domains: specific organ domains with >= 1 count take priority over purely systemic
    def domain_priority(item: tuple[str, int]) -> tuple[int, int]:
        dom, cnt = item
        # Specific domains get a priority boost over generic 'systemic'/'general'
        is_specific = 1 if dom not in {"systemic", "general"} else 0
        return (is_specific, cnt)

    ranked_domains = sorted(counts.items(), key=domain_priority, reverse=True)

    candidates: list[str] = []
    for dom, _ in ranked_domains:
        for spec in SPECIALTY_MAP.get(dom, []):
            if spec not in candidates:
                candidates.append(spec)

    if not candidates:
        candidates = ["General Physician", "Internal Medicine"]

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


def map_evidence_to_specialty(evidence: dict[str, Any] | None) -> dict[str, Any]:
    """Map structured multimodal evidence dynamically to suggested specialty."""
    if not evidence or evidence.get("evidence_state") == "INSUFFICIENT_EVIDENCE":
        return {
            "suggested_specialty": "General Physician / Internal Medicine",
            "candidates": ["General Physician", "Internal Medicine"],
            "message": "General evaluation recommended.",
            "wording": "Suggested specialty",
        }

    domain_instances: list[str] = []

    # 1. From active present symptoms
    symptoms = (evidence.get("symptoms") or {}).get("present", [])
    for s in symptoms:
        d = s.get("domain")
        if d:
            # If domain has slash like respiratory/cardiovascular, split
            for part in d.split("/"):
                part_clean = part.strip().lower()
                if part_clean:
                    domain_instances.append(part_clean)

    # 2. From abnormal report findings
    reports = (evidence.get("report") or {}).get("abnormal_findings", [])
    for r in reports:
        d = r.get("domain")
        if d:
            domain_instances.append(d.strip().lower())

    # 3. From X-ray model
    xray = evidence.get("xray") or {}
    if xray.get("prediction"):
        reg = (xray.get("region") or "").lower()
        if "chest" in reg:
            domain_instances.append("respiratory")
        elif "bone" in reg or "joint" in reg:
            domain_instances.append("bones")

    # If only normal report findings and no symptoms
    if not domain_instances:
        normal_reports = (evidence.get("report") or {}).get("normal_findings", [])
        if normal_reports:
            return {
                "suggested_specialty": "General Physician / Internal Medicine",
                "candidates": ["General Physician", "Internal Medicine"],
                "message": "Routine preventive check / General Physician consultation.",
                "wording": "Suggested specialty",
            }
        return map_specialty(["general"])

    return map_specialty(domain_instances)
