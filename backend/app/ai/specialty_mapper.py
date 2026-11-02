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
            "specialty_selection_mode": "rule_based_evidence_mapping",
            "is_ml_specialty_model": False,
        }

    # 1. From active present symptoms
    symptom_domains: list[str] = []
    symptoms = (evidence.get("symptoms") or {}).get("present", [])
    for s in symptoms:
        s_name = s.get("symptom", "").lower()
        d = s.get("domain")
        if s_name in {"cough", "sore throat", "shortness of breath", "wheezing", "chest tightness"}:
            symptom_domains.append("respiratory")
        elif s_name in {"headache", "migraine", "photophobia", "phonophobia", "facial weakness", "speech difficulty"}:
            symptom_domains.append("neurological")
        elif (
            s_name in {"wrist pain", "wrist injury", "fall/trauma", "wrist swelling", "knee pain", "medial knee pain", "joint pain", "morning stiffness", "joint clicking/catching", "reduced movement", "arthralgia"}
            or any(part in s_name for part in ["knee", "joint", "wrist", "ankle", "shoulder", "elbow", "hip", "spine", "clicking", "stiffness", "movement"])
        ):
            symptom_domains.append("musculoskeletal")
        elif d:
            for part in d.split("/"):
                part_clean = part.strip().lower()
                if part_clean:
                    symptom_domains.append(part_clean)

    # 2. From X-ray image model (only if actionable prediction present)
    xray_domains: list[str] = []
    xray = evidence.get("xray") or {}
    if xray.get("prediction"):
        reg = (xray.get("region") or "").lower()
        if "chest" in reg:
            xray_domains.append("respiratory")
        elif any(part in reg for part in ["knee", "wrist", "bone", "joint", "elbow", "shoulder", "hand", "finger", "forearm", "humerus", "hip", "ankle"]):
            xray_domains.append("bones")

    # 3. From radiology report qualitative findings
    rad_domains: list[str] = []
    qual_findings = (
        (evidence.get("report") or {}).get("qualitative_findings", [])
        + (evidence.get("xray") or {}).get("qualitative_findings", [])
    )
    for q in qual_findings:
        if q.get("state") in {"PRESENT", "POSITIVE"}:
            q_dom = q.get("domain")
            if q_dom:
                rad_domains.append(q_dom.strip().lower())
    for r in (evidence.get("xray") or {}).get("reports", []):
        r_reg = (r.get("region") or "").lower()
        if any(part in r_reg for part in ["knee", "wrist", "bone", "joint", "elbow", "shoulder", "hip", "ankle"]):
            rad_domains.append("musculoskeletal")
        elif "chest" in r_reg:
            rad_domains.append("respiratory")

    # 4. From abnormal laboratory report findings
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

        # Append secondary radiology and lab candidates without changing the primary suggested specialty
        candidates = list(primary_mapping["candidates"])
        for sec_domains in [rad_domains, report_domains]:
            if sec_domains:
                sec_mapping = map_specialty(sec_domains)
                for c in sec_mapping["candidates"]:
                    if c not in candidates:
                        candidates.append(c)

        return {
            "suggested_specialty": primary_spec,
            "candidates": candidates,
            "message": f"Suggested specialty: {primary_spec}",
            "wording": "Suggested specialty",
            "specialty_selection_mode": "rule_based_evidence_mapping",
            "is_ml_specialty_model": False,
        }

    if rad_domains:
        rad_mapping = map_specialty(rad_domains)
        res = dict(rad_mapping)
        res["specialty_selection_mode"] = "rule_based_evidence_mapping"
        res["is_ml_specialty_model"] = False
        return res

    if xray_domains:
        x_mapping = map_specialty(xray_domains)
        res = dict(x_mapping)
        res["specialty_selection_mode"] = "rule_based_evidence_mapping"
        res["is_ml_specialty_model"] = False
        return res

    if symptom_domains:
        if report_domains:
            res = map_specialty(report_domains)
        else:
            res = map_specialty(symptom_domains)
        res["specialty_selection_mode"] = "rule_based_evidence_mapping"
        res["is_ml_specialty_model"] = False
        return res

    if report_domains:
        res = map_specialty(report_domains)
        res["specialty_selection_mode"] = "rule_based_evidence_mapping"
        res["is_ml_specialty_model"] = False
        return res

    normal_reports = (evidence.get("report") or {}).get("normal_findings", [])
    if normal_reports:
        return {
            "suggested_specialty": "General Physician / Internal Medicine",
            "candidates": ["General Physician", "Internal Medicine"],
            "message": "Routine preventive check / General Physician consultation.",
            "wording": "Suggested specialty",
            "specialty_selection_mode": "rule_based_evidence_mapping",
            "is_ml_specialty_model": False,
        }

    res = map_specialty(["general"])
    res["specialty_selection_mode"] = "rule_based_evidence_mapping"
    res["is_ml_specialty_model"] = False
    return res
