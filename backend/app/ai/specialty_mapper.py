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
    "musculoskeletal": ["Orthopedics"],
    "injury/trauma": ["Orthopedics"],
    "injury": ["Orthopedics"],
    "trauma": ["Orthopedics"],
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
    """Map structured multimodal evidence dynamically to suggested specialty.

    Primary active symptom presentation has strict priority over incidental/contextual
    laboratory abnormalities. Abnormal labs cannot hijack primary specialty.
    """
    if not evidence or evidence.get("evidence_state") == "INSUFFICIENT_EVIDENCE":
        return {
            "suggested_specialty": "General Physician / Internal Medicine",
            "candidates": ["General Physician", "Internal Medicine"],
            "message": "General evaluation recommended.",
            "wording": "Suggested specialty",
        }

    # 1. From active present symptoms
    symptom_domains: list[str] = []
    symptoms = (evidence.get("symptoms") or {}).get("present", [])
    for s in symptoms:
        s_name = s.get("symptom", "").lower()
        d = s.get("domain")
        if s_name in {"cough", "sore throat", "shortness of breath", "wheezing"}:
            symptom_domains.append("respiratory")
        elif s_name in {"headache", "migraine", "photophobia", "phonophobia"}:
            symptom_domains.append("neurological")
        elif s_name in {"wrist pain", "wrist injury", "fall/trauma", "wrist swelling"}:
            symptom_domains.append("musculoskeletal")
        elif d:
            for part in d.split("/"):
                part_clean = part.strip().lower()
                if part_clean:
                    symptom_domains.append(part_clean)

    # 2. From X-ray model
    xray_domains: list[str] = []
    xray = evidence.get("xray") or {}
    if xray.get("prediction"):
        reg = (xray.get("region") or "").lower()
        if "chest" in reg:
            xray_domains.append("respiratory")
        elif "bone" in reg or "joint" in reg:
            xray_domains.append("bones")

    # 3. From abnormal report findings
    report_domains: list[str] = []
    reports = (evidence.get("report") or {}).get("abnormal_findings", [])
    for r in reports:
        d = r.get("domain")
        if d:
            report_domains.append(d.strip().lower())

    # Specific organ domains from primary symptoms take absolute priority
    specific_symptom_domains = [d for d in symptom_domains if d not in {"systemic", "general"}]

    if specific_symptom_domains:
        primary_mapping = map_specialty(specific_symptom_domains)
        primary_spec = primary_mapping["suggested_specialty"]

        # Append secondary report candidates without changing the primary suggested specialty
        candidates = list(primary_mapping["candidates"])
        if report_domains:
            report_mapping = map_specialty(report_domains)
            for c in report_mapping["candidates"]:
                if c not in candidates:
                    candidates.append(c)

        return {
            "suggested_specialty": primary_spec,
            "candidates": candidates,
            "message": f"Suggested specialty: {primary_spec}",
            "wording": "Suggested specialty",
        }

    if xray_domains:
        return map_specialty(xray_domains)

    if symptom_domains:
        if report_domains:
            return map_specialty(report_domains)
        return map_specialty(symptom_domains)

    if report_domains:
        return map_specialty(report_domains)

    normal_reports = (evidence.get("report") or {}).get("normal_findings", [])
    if normal_reports:
        return {
            "suggested_specialty": "General Physician / Internal Medicine",
            "candidates": ["General Physician", "Internal Medicine"],
            "message": "Routine preventive check / General Physician consultation.",
            "wording": "Suggested specialty",
        }

    return map_specialty(["general"])
