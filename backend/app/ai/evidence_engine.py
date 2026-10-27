"""Evidence combination engine for MediSense AI.

Combines multimodal evidence (symptoms, medical report findings, X-ray analysis)
into a unified clinical evidence representation:
- Preserves PRESENT / ABSENT / UNKNOWN states without losing qualifiers
- Preserves laterality, quality, duration, severity, triggers, and onset
- Identifies supporting findings (e.g. cough + report respiratory findings, normal lab confirmations)
- Identifies contradictory findings (e.g. fever denied vs high objective temp)
- Identifies missing critical information accurately without re-asking answered questions
- Truthful X-ray representation (received vs interpreted vs unavailable vs not provided)
- Establishes evidence_state: EVIDENCE_PRESENT, INSUFFICIENT_EVIDENCE, CONFLICTING_EVIDENCE
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

CRITICAL_NEGATIVES = ["chest pain", "shortness of breath", "high fever", "severe bleeding"]


# Canonical aliases are intentionally conservative. A generic mention of a body
# part in a report is not enough to prove which anatomy was imaged; extraction
# below requires examination/study-style context or an explicit metadata field.
_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    # Mirrors the router's canonical region vocabulary; dynamic router aliases
    # are also included during extraction so new configured regions stay aligned.
    "chest": ("chest", "thorax", "cxr", "lungs", "pulmonary"),
    "skull": ("skull", "cranium", "calvarium", "head"),
    "facial bones": ("facial bones", "facial bone", "face", "facial", "orbits", "orbit", "zygoma", "cheekbone"),
    "mandible": ("mandible", "jaw", "chin", "tmj"),
    "sinuses": ("sinuses", "sinus", "paranasal sinuses"),
    "neck": ("neck", "throat", "adenoids"),
    "shoulder": ("shoulder", "shoulder joint"),
    "clavicle": ("clavicle", "collarbone"),
    "humerus": ("humerus", "arm", "upper arm"),
    "elbow": ("elbow",),
    "forearm": ("forearm", "radius", "ulna", "radius ulna"),
    "wrist": ("wrist", "carpus", "carpal", "carpal bones"),
    "hand": ("hand", "metacarpal"),
    "finger": ("finger", "fingers", "digit", "digits", "thumb"),
    "spine": ("spine", "vertebra", "vertebrae", "back", "lumbosacral", "lower back", "low back"),
    "cervical spine": ("cervical spine", "c spine", "c-spine", "cervical", "neck spine"),
    "thoracic spine": ("thoracic spine", "t spine", "t-spine", "dorsal spine"),
    "lumbar spine": ("lumbar spine", "l spine", "l-spine", "lumbar"),
    "sacrum": ("sacrum", "coccyx", "tailbone", "sacrococcygeal"),
    "ribs": ("rib", "ribs", "ribcage", "bony thorax"),
    "sternum": ("sternum", "breastbone"),
    "abdomen": ("abdomen", "abdominal", "belly", "stomach", "tummy", "kub", "bowel"),
    "pelvis": ("pelvis", "pelvic"),
    "hip": ("hip", "hip joint"),
    "femur": ("femur", "thigh"),
    "knee": ("knee", "knee joint", "patella"),
    "lower leg": ("lower leg", "tibia", "fibula", "shin"),
    "ankle": ("ankle", "ankle joint"),
    "foot": ("foot", "feet", "metatarsal"),
    "toes": ("toe", "toes"),
    "calcaneus": ("calcaneus", "heel", "heel bone"),
}


def _canonical_region(value: Any) -> str | None:
    """Normalize a declared region using the same aliases as the X-ray router."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from app.ai.xray_router import normalize_region_name
        canonical, recognized = normalize_region_name(value)
        if recognized and isinstance(canonical, str) and canonical.strip():
            return canonical.strip().casefold()
    except Exception:
        # Evidence compilation remains usable in isolated tests when the router
        # cannot import; the local conservative aliases below are the fallback.
        pass
    import re

    text = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    if not text:
        return None
    # Prefer exact alias/region matches so e.g. "thoracic spine" is not reduced
    # to generic spine and "finger" is not reduced to hand.
    for canonical, aliases in _REGION_ALIASES.items():
        options = (canonical.replace("_", " "),) + aliases
        for alias in options:
            alias_text = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
            if text == alias_text:
                return canonical
    # Region selector labels may append view/type words, e.g. "Wrist X-Ray".
    for canonical, aliases in sorted(_REGION_ALIASES.items(), key=lambda item: max(map(len, item[1])), reverse=True):
        options = (canonical.replace("_", " "),) + aliases
        for alias in options:
            alias_text = re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip()
            if alias_text and re.search(rf"(?<![a-z0-9]){re.escape(alias_text)}(?![a-z0-9])", text):
                return canonical
    return None


def _extract_report_region(report: dict[str, Any]) -> tuple[str | None, str]:
    """Return (canonical region, source) only when report text gives usable context.

    Metadata from an upstream extractor takes precedence. Text matching is limited
    to examination/study-style phrases to reduce false mismatches from clinical
    history mentioning an unrelated body part. Filename is only a last resort.
    """
    import re

    # Do not trust a generic ``region`` field here: some callers populate it
    # from the user's dropdown selection, which would make the comparison
    # circular. Only use fields intended to represent report-extracted metadata.
    metadata_candidates: list[Any] = [
        report.get("region_hint"), report.get("body_region"), report.get("anatomical_region"),
        report.get("exam_region"), report.get("study_region"), report.get("examination_region"),
    ]
    ocr_meta = report.get("ocr_metadata")
    if isinstance(ocr_meta, dict):
        metadata_candidates.extend([
            ocr_meta.get("region_hint"), ocr_meta.get("body_region"),
            ocr_meta.get("anatomical_region"), ocr_meta.get("exam_region"),
        ])
    for candidate in metadata_candidates:
        canonical = _canonical_region(candidate)
        if canonical:
            return canonical, "metadata"

    raw_text = str(report.get("extracted_text") or report.get("text") or "")
    # Use only the report header/early text for exam identification. Later text
    # may discuss unrelated anatomy in history or comparison notes.
    header = "\n".join(raw_text.splitlines()[:15])[:1800].casefold()
    filename = str(report.get("file_name") or report.get("source_file") or "").casefold()

    modalities = r"(?:x[\s-]?ray|xray|radiograph(?:y|ic)?|imaging|examination|exam|study|views?)"
    context_prefixes = (
        r"(?:exam(?:ination)?|study|procedure|body\s*part|anatom(?:y|ical\s+region)|region|site)\s*[:\-]\s*",
        rf"\b{modalities}\s+(?:of\s+)?(?:the\s+)?",
    )
    alias_pairs: set[tuple[str, str]] = set()
    for canonical, aliases in _REGION_ALIASES.items():
        alias_pairs.add((canonical.replace("_", " "), canonical))
        for alias in aliases:
            alias_pairs.add((alias, canonical))
    try:
        from app.ai.xray_router import REGION_ALIASES as ROUTER_REGION_ALIASES
        for alias, canonical in ROUTER_REGION_ALIASES.items():
            alias_pairs.add((str(alias), str(canonical)))
            alias_pairs.add((str(canonical), str(canonical)))
    except Exception:
        pass
    aliases_by_specificity = sorted(alias_pairs, key=lambda item: len(item[0]), reverse=True)

    def _term_pattern(alias: str) -> str:
        return re.escape(alias).replace(r"\ ", r"[\s_-]+")

    found: set[str] = set()
    for alias, canonical in aliases_by_specificity:
        term = _term_pattern(alias)
        # Region preceding modality, e.g. "CHEST RADIOGRAPH" or "right knee X-ray".
        if re.search(rf"(?<![a-z0-9])(?:left\s+|right\s+|bilateral\s+)?{term}\s+{modalities}(?![a-z0-9])", header):
            found.add(canonical)
        # Modality/exam label preceding region, e.g. "X-ray of the wrist".
        if re.search(rf"\b{modalities}\s+(?:of\s+)?(?:the\s+)?(?:left\s+|right\s+|bilateral\s+)?{term}(?![a-z0-9])", header):
            found.add(canonical)
        # Explicit exam field, e.g. "Exam: Wrist".
        for prefix in context_prefixes:
            if re.search(rf"{prefix}(?:left\s+|right\s+|bilateral\s+)?{term}(?![a-z0-9])", header):
                found.add(canonical)
    if len(found) == 1:
        return next(iter(found)), "report_text"
    if len(found) > 1:
        # Multiple region-specific exam phrases are ambiguous unless one region
        # is declared through structured metadata above.
        return None, "ambiguous_report_text"

    # Filename fallback is deliberately weaker and is never called image-anatomy
    # verification. It helps legacy records whose report text lacks a title.
    filename_regions: set[str] = set()
    for alias, canonical in aliases_by_specificity:
        term = _term_pattern(alias)
        if re.search(rf"(?<![a-z0-9]){term}(?![a-z0-9])", filename):
            filename_regions.add(canonical)
    if len(filename_regions) == 1:
        return next(iter(filename_regions)), "filename_fallback"
    return None, "ambiguous_filename" if len(filename_regions) > 1 else "not_identified"


def _is_generic_xray_class_label(value: Any) -> bool:
    """True for broad screening classes that must not be presented as diseases."""
    if not isinstance(value, str):
        return False
    import re
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    return normalized in {
        "normal", "abnormal", "positive", "negative", "normal study", "abnormal study",
        "no abnormality", "no acute abnormality", "no finding", "no findings", "unremarkable",
    }


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Return only dictionary records from an optional list-like input."""
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _report_has_extracted_content(report: dict[str, Any]) -> bool:
    """Whether a report has readable text or at least one structured finding.

    Accept both the nested ``structured_findings`` schema and the legacy/top-level
    qualitative/narrative lists used by some extraction callers.
    """
    text = report.get("extracted_text") or report.get("text")
    if isinstance(text, str) and text.strip():
        return True

    structured = report.get("structured_findings")
    structured = structured if isinstance(structured, dict) else {}

    for source in (structured, report):
        for key in ("lab_parameters", "qualitative_findings", "narrative_findings"):
            value = source.get(key)
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, dict) and bool(item) for item in value
            ):
                return True

    total = structured.get("total_extracted", report.get("total_extracted"))
    try:
        return int(total) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def _normalize_xray_findings(value: Any) -> list[dict[str, Any]]:
    """Normalize multi-label model outputs without treating scores as diagnoses."""
    findings: list[dict[str, Any]] = []
    if not isinstance(value, (list, tuple)):
        return findings
    for item in value:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("finding") or item.get("name") or "").strip()
        if not label:
            continue
        normalized = dict(item)
        normalized["label"] = label
        # Keep the original model score field and value; it is not a calibrated
        # probability and must not be presented as diagnostic certainty.
        findings.append(normalized)
    return findings


def _xray_finding_label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("finding") or item.get("name") or "").strip()


def _xray_finding_is_actionable(item: dict[str, Any]) -> bool:
    """Only explicit, model/service-designated positives can support clinical evidence.

    DenseNet multi-label scores are raw model outputs unless an upstream validated
    thresholding/interpretation layer explicitly marks them actionable.
    """
    if not isinstance(item, dict):
        return False
    if item.get("is_actionable_finding") is True:
        return True
    if item.get("detected") is True or item.get("is_positive") is True:
        return True
    state = str(item.get("state") or item.get("status") or "").strip().upper()
    return state in {"PRESENT", "POSITIVE", "DETECTED"}


def _state(record: dict[str, Any]) -> str:
    """Normalize symptom state without converting unanswered findings to negatives."""
    raw = str(record.get("state") or record.get("status") or "").strip().upper()
    aliases = {
        "YES": "PRESENT", "TRUE": "PRESENT", "POSITIVE": "PRESENT",
        "NO": "ABSENT", "FALSE": "ABSENT", "NEGATIVE": "ABSENT",
        "NOT_SURE": "UNKNOWN", "NOT SURE": "UNKNOWN", "UNSURE": "UNKNOWN",
        "UNANSWERED": "UNKNOWN", "UNASSESSED": "UNKNOWN", "": "UNKNOWN",
    }
    return aliases.get(raw, raw if raw in {"PRESENT", "ABSENT", "UNKNOWN"} else "UNKNOWN")


def _status(record: dict[str, Any]) -> str:
    """Normalize lab status, prioritizing explicit interpretation/flag over workflow status."""
    # Some records use `status=RECORDED` for workflow state while the actual
    # clinical flag is in `interpretation` or `explicit_flag`.
    for key in ("explicit_flag", "interpretation", "calculated_status", "status"):
        value = str(record.get(key) or "").strip().upper()
        if value in {"HIGH", "LOW", "ABNORMAL", "NORMAL", "UNKNOWN"}:
            return value
    return "RECORDED" if any(record.get(k) for k in ("value", "test_name")) else "UNKNOWN"


def _lookup_symptom_domain(symptom_name: str | None) -> str:
    if not symptom_name:
        return "general"
    try:
        from app.ai.symptom_nlp import SYMPTOM_LEXICON
    except ImportError:
        # Keep evidence compilation usable in isolated tests/tools; the full
        # application supplies the lexicon and should be used in production.
        return "general"

    key = str(symptom_name).strip().lower()
    if key in SYMPTOM_LEXICON:
        return SYMPTOM_LEXICON[key].get("domain", "general")
    for k, v in SYMPTOM_LEXICON.items():
        if k in key or key in k:
            return v.get("domain", "general")
    return "general"


def combine_evidence(
    symptoms: list[dict[str, Any]] | None = None,
    report_findings: list[dict[str, Any]] | None = None,
    xray_result: dict[str, Any] | None = None,
    report_provided: bool | None = None,
    qualitative_report_findings: list[dict[str, Any]] | None = None,
    narrative_report_findings: list[dict[str, Any]] | None = None,
    xray_results: list[dict[str, Any]] | None = None,
    xray_reports: list[dict[str, Any]] | None = None,
    medical_reports: list[dict[str, Any]] | None = None,
    medical_report_provided: bool | None = None,
    radiology_report_provided: bool | None = None,
    selected_input_types: list[str] | None = None,
) -> dict[str, Any]:
    """Combine multimodal inputs into structured clinical evidence."""
    # Normalize inputs at the boundary so malformed records do not crash the
    # entire assessment. Copy symptom dictionaries before adding inferred onset.
    symptoms = [dict(item) for item in _as_dict_list(symptoms)]
    report_findings = _as_dict_list(report_findings)
    qualitative_report_findings = _as_dict_list(qualitative_report_findings)
    narrative_report_findings = _as_dict_list(narrative_report_findings)
    xray_images = _as_dict_list(xray_results) if xray_results is not None else ([dict(xray_result)] if isinstance(xray_result, dict) else [])
    xray_reports = _as_dict_list(xray_reports)
    medical_reports = _as_dict_list(medical_reports)

    # If qualitative findings are passed inside report_findings, normalize them
    for f in list(report_findings):
        if "finding" in f and "test_name" not in f and f not in qualitative_report_findings:
            qualitative_report_findings.append(dict(f))

    # Keep report categories separate. The legacy report_provided argument
    # remains supported as an overall report-availability signal.
    # A false availability flag must not erase evidence that is present in the
    # actual source records. True may be used to represent an uploaded-but-empty
    # report, so OR it with extracted records rather than replacing them.
    is_medical_report_provided = bool(medical_report_provided) or bool(
        report_findings or qualitative_report_findings or narrative_report_findings or medical_reports
    )
    is_radiology_report_provided = bool(radiology_report_provided) or bool(xray_reports)
    has_usable_medical_report = bool(
        report_findings or qualitative_report_findings or narrative_report_findings
    ) or any(_report_has_extracted_content(report) for report in medical_reports)
    has_usable_radiology_report = any(
        _report_has_extracted_content(report) for report in xray_reports
    )
    # Overall report availability must not be false when the caller passes
    # report_provided=False for lab-only state but a radiology report is present.
    # Explicit true is retained for legacy callers that track upload separately
    # from extracted content; source records always take precedence over false.
    is_report_provided = bool(report_provided) or is_medical_report_provided or is_radiology_report_provided

    selected_types = {
        str(item).strip().lower().replace("-", "_").replace(" ", "_")
        for item in (selected_input_types or []) if str(item).strip()
    }
    expects_medical_report = any(
        item in selected_types
        for item in {"medical_report", "report", "lab_report", "laboratory_report"}
    )
    # Keep image and radiology-report requirements independent. The combined
    # X-Ray module (`xray`/`x_ray`) expects both sources, but a report-only
    # selection must not incorrectly produce a missing-image warning.
    expects_xray_image = any(
        item in selected_types
        for item in {"xray", "x_ray", "xray_image", "xray_images", "radiology_image", "radiographic_image"}
    )
    expects_radiology_report = any(
        item in selected_types
        for item in {"xray", "x_ray", "radiology", "xray_report", "radiology_report"}
    )

    current_present = [
        s for s in symptoms
        if _state(s) == "PRESENT"
        and str(s.get("type") or "").strip().upper() != "HISTORICAL"
        and str(s.get("symptom") or "").strip().lower() not in {"previous similar headache", "gradual onset", "sudden onset"}
    ]
    historical = [
        s for s in symptoms
        if _state(s) == "PRESENT"
        and (str(s.get("type") or "").strip().upper() == "HISTORICAL" or str(s.get("symptom") or "").strip().lower() == "previous similar headache")
    ]
    # Propagate onset attribute from gradual onset / sudden onset pseudo-symptoms if not already set
    has_gradual = any(s.get("symptom") == "gradual onset" for s in symptoms)
    has_sudden = any(s.get("symptom") == "sudden onset" for s in symptoms)
    if has_gradual or has_sudden:
        target_onset = "GRADUAL" if has_gradual else "SUDDEN"
        for s in current_present:
            if not s.get("onset"):
                if s.get("symptom") in {"headache", "migraine", "cough", "chest pain"}:
                    s["onset"] = target_onset
                    break

    symptoms_present = current_present
    symptoms_absent = [s for s in symptoms if _state(s) == "ABSENT"]
    symptoms_unknown = [s for s in symptoms if _state(s) == "UNKNOWN"]

    report_abnormal = [f for f in report_findings if _status(f) in {"HIGH", "LOW", "ABNORMAL"}]
    report_normal = [f for f in report_findings if _status(f) == "NORMAL"]
    report_unknown_range = [f for f in report_findings if _status(f) in {"UNKNOWN", "RECORDED", ""}]

    supporting: list[str] = []
    reassuring: list[str] = []
    separate: list[str] = []
    contradictory: list[str] = []
    missing: list[str] = []
    unassessed: list[str] = []
    evidence_relationships: list[dict[str, Any]] = []

    present_names = {str(s.get("symptom") or "").strip().lower() for s in current_present} - {""}
    absent_names = {str(s.get("symptom") or "").strip().lower() for s in symptoms_absent} - {""}
    all_known_names = present_names.union(absent_names).union({str(s.get("symptom") or "").strip().lower() for s in historical} - {""})

    # 1. Contradiction Detection
    has_fever_absent = "fever" in absent_names
    for s in current_present:
        if s.get("symptom") == "temperature":
            val_str = str(s.get("context") or "")
            import re
            m = re.search(r"(\d{2,3}(?:\.\d+)?)", val_str)
            if m:
                try:
                    t_val = float(m.group(1))
                    if t_val >= 100.4 and has_fever_absent:
                        contradictory.append(
                            f"Contradiction: Patient reported absence of fever, but objective temperature is {t_val}°F (febrile range)."
                        )
                except ValueError:
                    pass

    # Laboratory explicit flag vs reference range conflict detection
    for item in report_findings:
        if item.get("conflict"):
            fname = item.get("test_name", "Laboratory parameter")
            ex_flag = item.get("explicit_flag")
            calc_st = item.get("calculated_status")
            val = item.get("value")
            unit = item.get("unit") or ""
            ref = item.get("reference_range") or "unspecified"
            contradictory.append(
                f"Contradictory laboratory evidence for {fname}: Explicit report flag is '{ex_flag}', but measured value {val} {unit} evaluates to '{calc_st}' against reference range ({ref})."
            )
            evidence_relationships.append({
                "source_a": "report_flag",
                "source_b": "report_reference_range",
                "finding": fname,
                "relationship": "CONFLICTING",
                "reason": f"Explicit flag '{ex_flag}' conflicts with calculated status '{calc_st}' for {fname}.",
            })

    # 2. Supporting Clinical Correlations & Reassuring Evidence
    has_respiratory_symptom = bool(
        present_names.intersection({"cough", "dry cough", "productive cough", "shortness of breath", "difficulty breathing", "breathing difficulty", "dyspnea", "wheezing", "chest discomfort", "chest pain"})
    )
    has_fever = "fever" in present_names or "temperature" in present_names

    normal_lab_tests = {
        str(f.get("test_name") or "").lower() for f in report_normal
        if str(f.get("test_name") or "").strip()
    }
    has_normal_wbc = any(k in t for t in normal_lab_tests for k in ["wbc", "white blood cell"])
    has_normal_crp = any(k in t for t in normal_lab_tests for k in ["crp", "c-reactive"])
    has_abnormal_wbc = any(
        _status(f) in {"HIGH", "ABNORMAL"}
        and any(k in str(f.get("test_name") or "").lower() for k in ["wbc", "white blood cell"])
        for f in report_abnormal
    )
    has_abnormal_crp = any(
        _status(f) in {"HIGH", "ABNORMAL"}
        and any(k in str(f.get("test_name") or "").lower() for k in ["crp", "c-reactive"])
        for f in report_abnormal
    )

    if has_respiratory_symptom:
        reassuring_lab_parts = []
        if has_normal_wbc:
            reassuring_lab_parts.append("WBC")
        if has_normal_crp:
            reassuring_lab_parts.append("CRP")
        if reassuring_lab_parts:
            reassuring.append(f"Available {' and '.join(reassuring_lab_parts)} value(s) are within the supplied reference ranges.")
        elif has_fever and not (has_normal_wbc or has_normal_crp):
            supporting.append("Respiratory symptoms correlate with systemic febrile presentation.")

    if "headache" in present_names:
        headache_sym = next((s for s in current_present if s.get("symptom") == "headache"), None)
        features = []
        if headache_sym and headache_sym.get("laterality"):
            features.append(f"{headache_sym.get('laterality').lower()}-sided")
        if headache_sym and headache_sym.get("quality"):
            features.append(f"{headache_sym.get('quality').lower()}")
        if "photophobia" in present_names:
            features.append("photophobia")
        if "phonophobia" in present_names:
            features.append("phonophobia")
        if "nausea" in present_names:
            features.append("nausea")
        if features:
            supporting.append(f"Headache presentation characterized by {', '.join(features)}.")

    # Musculoskeletal correlations
    if any(n in present_names for n in {"wrist pain", "wrist injury", "fall/trauma", "swelling", "tenderness"}):
        msk_features = []
        for s in current_present:
            if s.get("body_area") == "wrist" or s.get("symptom") in {"wrist pain", "wrist injury", "fall/trauma"}:
                lat = f"{s.get('laterality').lower()} " if s.get("laterality") else ""
                msk_features.append(f"{lat}{s.get('symptom')}")
        if msk_features:
            supporting.append(f"Musculoskeletal trauma indicators: {', '.join(list(dict.fromkeys(msk_features)))}.")

    # Abnormal laboratory parameters: contextual laboratory evidence (never causal proof)
    for item in report_abnormal:
        fname = item.get("test_name", "")
        val = item.get("value")
        unit = item.get("unit") or ""
        interp = item.get("interpretation") or "ABNORMAL"
        ref = item.get("reference_range")
        ref_text = f" (reference: {ref})" if ref else ""

        separate.append(f"Additional abnormal laboratory finding: {fname} recorded outside reference range ({val} {unit}{ref_text}, {interp}); requires clinical interpretation and does not by itself establish the cause of the presenting complaint.")
        evidence_relationships.append({
            "source_a": "symptoms",
            "source_b": "medical_report",
            "finding": fname,
            "relationship": "SEPARATE",
            "reason": f"{fname} is outside standard reference limits but does not directly establish the cause of the presenting complaints."
        })

    if report_abnormal and "headache" in present_names:
        separate.append("Additional abnormal laboratory findings were identified and require clinical interpretation; they do not by themselves establish the cause of the headache.")
    elif report_abnormal and has_respiratory_symptom:
        separate.append("Additional abnormal laboratory findings were identified and require clinical interpretation; they do not by themselves establish the cause of the respiratory symptoms.")

    # Normal laboratory parameters (reassuring evidence)
    if report_normal:
        normal_names = [f.get("test_name") for f in report_normal if f.get("test_name")]
        reassuring.append(f"Available laboratory values ({', '.join(normal_names)}) are within standard supplied reference ranges.")

    # Cross-modal relevance: Aligned vs Separate/Contextual qualitative findings
    for qf in qualitative_report_findings:
        f_name = qf.get("finding", "")
        f_state = str(qf.get("state") or "PRESENT").strip().upper()
        f_lower = str(f_name or "").lower()

        is_relevant = False
        if has_respiratory_symptom:
            if any(k in f_lower for k in ["lung", "breath", "cough", "throat", "chest", "wheez", "rales", "rhonchi", "bronch", "pharynx"]):
                is_relevant = True
        elif "headache" in present_names or any(s.get("domain") == "neurological" for s in current_present):
            if any(k in f_lower for k in ["head", "neuro", "pupil", "cranial", "mening", "vision", "speech", "reflex"]):
                is_relevant = True
        elif any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present):
            if any(k in f_lower for k in ["wrist", "hand", "joint", "bone", "swelling", "fracture", "tendon", "ligament", "sprain", "strain"]):
                is_relevant = True

        if not is_relevant:
            separate.append(f"Qualitative finding of {f_name} does not directly account for the primary presenting clinical presentation.")
            evidence_relationships.append({
                "source_a": "symptoms",
                "source_b": "medical_report",
                "finding": f_name,
                "relationship": "SEPARATE",
                "reason": f"{f_name.capitalize()} does not directly explain the primary symptom presentation."
            })
        else:
            if f_state == "PRESENT":
                supporting.append(f"Medical report qualitative finding confirmed: {f_name}.")
            elif f_state == "ABSENT":
                reassuring.append(f"Medical report explicitly confirms absence of: {f_name}.")

    import re
    synthetic_instruction_re = re.compile(
        r"\b(?:synthetic|demonstration|demo\s+only|sample\s+report|test\s+case|no\s+actual\s+clinical\s+findings|instructions?|placeholder|dummy\s+data|simulated|example\s+report)\b",
        re.IGNORECASE,
    )
    for nf in narrative_report_findings:
        sec = str(nf.get("section") or "").strip()
        cnt = str(nf.get("content") or "").strip()
        # Narrative text may be history, a disclaimer, or imperfect OCR. Do not
        # automatically treat every narrative sentence as clinical support.
        # Preserve it as contextual information; structured findings retain states.
        if cnt and not synthetic_instruction_re.search(cnt) and not synthetic_instruction_re.search(sec):
            short_cnt = cnt[:240] + "..." if len(cnt) > 240 else cnt
            separate.append(f"Report narrative for review ({sec or 'section unspecified'}): {short_cnt}")

    # 3. Truthful X-ray Status and Interpretation (Multi-Image & Multi-Report Aware)
    processed_images: list[dict[str, Any]] = []
    # Filled by the report-region validation pass later in this function.
    # It is initialized here because image handling also consults this set.
    mismatched_image_files: set[str] = set()
    xray_finding_str = None
    xray_received = bool(xray_images)
    xray_status_display = "NOT_PROVIDED"
    xray_interpretation = "NOT_PROVIDED"
    xray_evidence_status = "NOT_ASSESSED"
    xray_region = "not declared"

    if xray_images:
        completed_preds = []
        any_unavailable = False
        all_regions = set()

        for idx, img in enumerate(xray_images):
            img_status = str(img.get("status") or "unavailable").strip().lower()
            reg_raw = img.get("region")
            if isinstance(reg_raw, (list, tuple)):
                reg_raw = reg_raw[0] if reg_raw else ""
            img_reg = str(reg_raw or "").strip().lower() or "not declared"
            all_regions.add(img_reg)
            img_fname = img.get("file_name") or img.get("source_file") or f"xray_image_{idx+1}"
            img_label = f"{img_reg.capitalize()} X-ray ({img_fname})" if img_reg not in {"not declared", "unknown"} else f"X-ray ({img_fname})"
            pred = img.get("prediction")
            findings = _normalize_xray_findings(img.get("findings"))
            has_model_output = bool(pred) or bool(findings)

            p_entry = {
                "id": img.get("id") or img.get("xray_result_id"),
                "file_name": img_fname,
                "region": img_reg,
                "status": img_status.upper(),
                "prediction": pred,
                "findings": findings,
                "model_version": img.get("model_version") or "N/A",
                "confidence": img.get("confidence_if_valid"),
                "uncertainty": img.get("uncertainty"),
                "message": img.get("message"),
                "source": "xray_image",
                "source_type": "XRAY_IMAGE",
                "source_file": img_fname,
            }
            processed_images.append(p_entry)

            # Check anatomical alignment with primary complaints.
            is_anatomically_aligned = True
            if has_respiratory_symptom and img_reg in {"wrist", "hand", "ankle", "knee", "extremity"}:
                is_anatomically_aligned = False
            elif any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present) and not has_respiratory_symptom and img_reg == "chest":
                is_anatomically_aligned = False

            if img_status == "completed" and has_model_output:
                # A legacy single prediction is retained for compatibility.
                # Multi-label score records are NOT clinical findings unless the
                # model/service explicitly marks them actionable.
                actionable_findings = [f for f in findings if _xray_finding_is_actionable(f)]
                prediction_label = str(pred).strip() if isinstance(pred, str) and pred.strip() else None
                generic_class_output = bool(prediction_label and _is_generic_xray_class_label(prediction_label))
                labels = [] if generic_class_output else ([prediction_label] if prediction_label else [
                    _xray_finding_label(f) for f in actionable_findings
                ])
                labels = [label for label in labels if label]
                raw_score_labels = [
                    _xray_finding_label(f)
                    for f in findings
                    if not _xray_finding_is_actionable(f) and _xray_finding_label(f)
                ]

                # Preserve generic screening-class outputs for transparent display,
                # but never treat "normal" or "abnormal" as a specific disease or
                # as a positive clinical finding.
                if generic_class_output and prediction_label:
                    completed_preds.append(prediction_label)
                    unassessed.append(
                        f"{img_label} model output is '{prediction_label}', a broad normal/abnormal class rather than a specific disease finding. "
                        "It is recorded for transparency and is not treated as diagnostic confirmation or a triage signal."
                    )

                if labels:
                    completed_preds.extend(labels)
                    if is_anatomically_aligned and str(img.get("file_name") or "xray_image") not in mismatched_image_files:
                        supporting.append(
                            f"{img_label} model/service returned explicitly actionable candidate finding(s): "
                            f"{', '.join(labels[:18])}."
                            + (f" And {len(labels) - 18} more." if len(labels) > 18 else "")
                            + " This is not a confirmed diagnosis and requires clinical interpretation."
                        )
                    else:
                        separate.append(
                            f"{img_label} model output ({', '.join(labels[:18])}) is not used as supporting evidence because it is anatomically separate from the primary complaint or its report pairing is mismatched."
                        )
                        evidence_relationships.append({
                            "source_a": "symptoms",
                            "source_b": "xray_image",
                            "finding": f"{img_label}: {', '.join(labels[:18])}",
                            "relationship": "SEPARATE",
                            "reason": f"Image region ({img_reg}) is separate from the complaint or is involved in a region-mismatched image/report pair.",
                        })

                if raw_score_labels:
                    # Preserve raw scores in processed_images for transparent display,
                    # but never promote them to clinical support, diagnosis, or triage.
                    score_preview = ", ".join(raw_score_labels[:18])
                    unassessed.append(
                        f"{img_label} inference completed and returned raw, unthresholded model scores "
                        f"({score_preview}{'; and more' if len(raw_score_labels) > 18 else ''}). "
                        "These scores are not calibrated diagnostic probabilities and are not counted as actionable clinical evidence."
                    )

                if not labels and not raw_score_labels and not generic_class_output:
                    any_unavailable = True
                    unassessed.append(
                        f"{img_label} was marked completed but contained no usable prediction or findings; interpretation cannot be verified."
                    )
            elif img_status == "unavailable":
                any_unavailable = True
                unassessed.append(f"{img_label} received; automated interpretation unavailable for this region. No disease finding fabricated.")
            else:
                any_unavailable = True
                unassessed.append(f"{img_label} status: {img_status.upper()}. Automated interpretation unavailable.")

        xray_region = ", ".join(sorted(all_regions)) if all_regions else "not declared"
        if completed_preds:
            xray_finding_str = "; ".join(completed_preds)
            xray_interpretation = "AVAILABLE"
            aligned_prediction_added = any(
                "explicitly actionable candidate finding" in item
                and "X-ray (" in item
                for item in supporting
            )
            xray_evidence_status = "SUPPORTING" if aligned_prediction_added else "RECORDED"
            xray_status_display = "PARTIAL" if any_unavailable else "COMPLETED"
            if any_unavailable:
                unassessed.append(
                    "Some uploaded X-ray images were interpreted, while one or more images remain unavailable or unprocessed."
                )
        elif any(
            str(img.get("status") or "").strip().lower() == "completed"
            and (img.get("prediction") or _normalize_xray_findings(img.get("findings")))
            for img in xray_images
        ):
            # Inference completed, but only raw non-actionable scores were returned.
            # This is recorded as model output, not as an interpreted clinical finding.
            xray_interpretation = "RECORDED_UNINTERPRETED"
            xray_evidence_status = "RECORDED"
            xray_status_display = "PARTIAL" if any_unavailable else "COMPLETED"
        elif any_unavailable:
            xray_interpretation = "UNAVAILABLE"
            xray_evidence_status = "UNKNOWN"
            xray_status_display = "UNAVAILABLE"
        else:
            xray_interpretation = "UNAVAILABLE"
            xray_evidence_status = "UNKNOWN"
            xray_status_display = "RECEIVED"

    # Process X-ray / Radiology Reports
    processed_xray_reports: list[dict[str, Any]] = []
    for idx, xr in enumerate(xray_reports):
        xr_fname = xr.get("file_name") or f"xray_report_{idx+1}.pdf"
        struct = xr.get("structured_findings")
        struct = dict(struct) if isinstance(struct, dict) else {}
        # Support both nested and top-level extractor payloads. Nested values
        # take precedence when present; top-level values are a compatibility
        # fallback, not a second copy of the same findings.
        xr_qual = _as_dict_list(
            struct.get("qualitative_findings")
            if struct.get("qualitative_findings") is not None
            else xr.get("qualitative_findings", [])
        )
        xr_narr = _as_dict_list(
            struct.get("narrative_findings")
            if struct.get("narrative_findings") is not None
            else xr.get("narrative_findings", [])
        )
        if "qualitative_findings" not in struct and xr_qual:
            struct["qualitative_findings"] = xr_qual
        if "narrative_findings" not in struct and xr_narr:
            struct["narrative_findings"] = xr_narr
        xr_entry = {
            "id": xr.get("id") or xr.get("report_id"),
            "file_name": xr_fname,
            "source_type": "XRAY_REPORT",
            "source_file": xr_fname,
            "status": "EXTRACTED" if _report_has_extracted_content(xr) else "UNREADABLE",
            "structured_findings": struct,
            "qualitative_findings": xr_qual,
            "narrative_findings": xr_narr,
            "created_at": xr.get("created_at"),
            "is_historical": xr.get("is_historical", False),
        }
        processed_xray_reports.append(xr_entry)

        # Keep source statements in processed_xray_reports. Do not automatically
        # promote each statement to supporting/reassuring evidence here.
        # Classify radiology findings across all reports below, so conflicts
        # between separate reports remain visible.

        # Region matching is performed after all image/report records are built.
        # This keeps filename/body-part mentions from being mistaken for proof
        # and lets a report be explicitly linked to the correct image.

    # Verify the region relationship between image records and radiology reports.
    # This is a report-to-declared-region check, NOT independent image anatomy
    # detection. The upstream image verifier must establish actual PNG anatomy.
    has_region_mismatch = False
    region_verification_records: list[dict[str, Any]] = []
    normalized_image_regions = {
        str(img.get("file_name") or "xray_image"): _canonical_region(img.get("region"))
        for img in processed_images
    }
    all_declared_image_regions = {r for r in normalized_image_regions.values() if r}

    def _report_linked_image_names(report: dict[str, Any]) -> list[str]:
        """Return image names only for an explicitly confirmed persisted association.

        Region compatibility, filename similarity, assessment membership, or a
        one-image/one-report batch are not user confirmation. Keep those sources
        separate until the pairing workflow persists an explicit confirmation.
        """
        import re

        status_candidates = (
            report.get("association_status"),
            report.get("pairing_status"),
            report.get("report_image_association_status"),
            report.get("link_status"),
            report.get("association_decision"),
        )
        status_values = {str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
                         for value in status_candidates}
        explicit_confirmed = any(value in {
            "CONFIRMED", "USER_CONFIRMED", "EXPLICITLY_CONFIRMED",
            "CONFIRMED_MATCH", "ACCEPTED", "APPROVED",
        } for value in status_values) or any(
            report.get(key) is True
            for key in (
                "association_confirmed", "pairing_confirmed",
                "user_confirmed_pair", "explicitly_confirmed",
                "report_image_confirmed",
            )
        )
        # A rejected association must never be revived by another field.
        if any(value in {"REJECTED", "DECLINED", "DISMISSED", "CANCELLED"}
               for value in status_values) or any(
            report.get(key) is True
            for key in ("association_rejected", "pairing_rejected", "user_rejected_pair")
        ):
            return []
        if not explicit_confirmed:
            return []

        explicit_ids = {
            str(report.get(key)).strip()
            for key in (
                "xray_result_id", "xray_image_id", "image_id",
                "linked_xray_result_id", "source_xray_result_id",
                "confirmed_xray_result_id", "confirmed_image_id",
            )
            if report.get(key) not in (None, "")
        }
        explicit_names = {
            str(report.get(key)).strip()
            for key in ("confirmed_image_file", "linked_image_file", "xray_image_file")
            if isinstance(report.get(key), str) and report.get(key).strip()
        }
        matches = []
        for img in processed_images:
            image_ids = {
                str(img.get(key)).strip()
                for key in ("id", "xray_result_id", "image_id")
                if img.get(key) not in (None, "")
            }
            image_name = str(img.get("file_name") or "xray_image")
            if (explicit_ids and explicit_ids.intersection(image_ids)) or image_name in explicit_names:
                matches.append(image_name)
        # Do not infer links from filenames, anatomy, batch membership, or counts.
        return list(dict.fromkeys(matches))

    for report in processed_xray_reports:
        # Locate original record for full OCR metadata and explicit region hints.
        source_report = next(
            (item for item in xray_reports if str(item.get("file_name") or item.get("source_file") or "") == str(report.get("file_name") or "")),
            {},
        )
        report_region, region_source = _extract_report_region(source_report)
        linked_images = _report_linked_image_names(source_report)
        linked_region_values = {
            normalized_image_regions.get(name)
            for name in linked_images
            if normalized_image_regions.get(name)
        }
        linked_region_values.discard(None)

        if not processed_images:
            verification_status = "REPORT_ONLY"
            mismatch_regions: list[dict[str, str]] = []
        elif not linked_images:
            verification_status = "UNPAIRED" if report_region else "UNVERIFIED"
            mismatch_regions = []
        elif not report_region or region_source in {"filename_fallback", "ambiguous_report_text", "ambiguous_filename"}:
            # A filename can suggest a region, but it is not reliable enough to
            # establish agreement or a mismatch between a report and an image.
            verification_status = "UNVERIFIED"
            mismatch_regions = []
        else:
            mismatch_regions = [
                {"image_file": name, "image_region": normalized_image_regions.get(name) or "unknown", "report_region": report_region}
                for name in linked_images
                if normalized_image_regions.get(name) and normalized_image_regions.get(name) != report_region
            ]
            if mismatch_regions:
                verification_status = "MISMATCH"
                has_region_mismatch = True
                for item in mismatch_regions:
                    mismatched_image_files.add(item["image_file"])
                    contradictory.append(
                        f"Conflicting evidence / body region mismatch: image '{item['image_file']}' is declared as '{item['image_region']}', "
                        f"but report '{report.get('file_name')}' identifies '{report_region}'. Keep these sources separate and correct the region/report pairing."
                    )
                    evidence_relationships.append({
                        "source_a": f"xray_image ({item['image_file']})",
                        "source_b": f"xray_report ({report.get('file_name')})",
                        "finding": "body_region_mismatch",
                        "relationship": "CONFLICTING",
                        "reason": f"Image declared region '{item['image_region']}' differs from report-identified region '{report_region}' (source: {region_source}).",
                    })
            elif linked_region_values and all(region == report_region for region in linked_region_values):
                # The region check is informative only. The report was not paired
                # unless the upstream workflow persisted an explicit confirmation.
                verification_status = "MATCH"
            else:
                verification_status = "UNVERIFIED"

        report["region_verification"] = {
            "status": verification_status,
            "report_region": report_region,
            "region_source": region_source,
            "matched_image_files": linked_images,
            "mismatches": mismatch_regions,
            "verified_match": verification_status == "MATCH",
            "association_confirmed": bool(linked_images),
            "association_status": (
                str(
                    source_report.get("association_status")
                    or source_report.get("pairing_status")
                    or source_report.get("report_image_association_status")
                    or source_report.get("link_status")
                    or source_report.get("association_decision")
                    or ("CONFIRMED" if linked_images else "UNCONFIRMED")
                ).strip().upper()
            ),
            "message": (
                "Image/report association was explicitly confirmed and the regions match."
                if verification_status == "MATCH" and linked_images
                else "The report and image regions appear compatible, but no explicit report-to-image confirmation is recorded."
                if verification_status == "MATCH"
                else "Image region and report region disagree; keep the sources unlinked and correct the pairing."
                if verification_status == "MISMATCH"
                else "The report region could not be reliably paired/verified against an image. This is not proof that the regions match."
                if verification_status in {"UNVERIFIED", "UNPAIRED"}
                else "Radiology report processed without an accompanying image."
            ),
        }
        region_verification_records.append({
            "report_file": report.get("file_name"),
            **report["region_verification"],
        })
        if verification_status in {"UNVERIFIED", "UNPAIRED"} and processed_images:
            unassessed.append(
                f"Radiology report '{report.get('file_name')}' region could not be confidently matched to its X-ray image; image/report agreement is unverified."
            )

    # If an image/report pair is known to conflict, remove that image's candidate
    # output from the aggregate X-ray summary. The raw per-image record remains
    # visible for auditing, but it must not flow onward as if the pair were valid.
    if has_region_mismatch:
        for image_file in mismatched_image_files:
            source_img = next(
                (img for img in processed_images if str(img.get("file_name") or "xray_image") == image_file),
                {},
            )
            img_region = str(source_img.get("region") or "").strip()
            img_label = (
                f"{img_region.capitalize()} X-ray ({image_file})"
                if img_region and img_region.lower() not in {"not declared", "unknown"}
                else f"X-ray ({image_file})"
            )
            supporting[:] = [
                item for item in supporting
                if not item.startswith(f"{img_label} model/service returned explicitly actionable candidate finding(s):")
            ]
        # Mark conflicted image records explicitly so downstream triage cannot
        # consume their prediction as valid evidence. Preserve the raw prediction
        # separately for traceability/audit.
        for img in processed_images:
            image_file = str(img.get("file_name") or "xray_image")
            if image_file in mismatched_image_files:
                img["inference_status"] = img.get("status")
                img["region_verification_status"] = "MISMATCH"
                img["raw_model_prediction"] = img.get("prediction")
                img["prediction"] = None
                img["status"] = "REGION_MISMATCH"
                img["message"] = (
                    "Image inference output was not accepted into the assessment because its associated "
                    "radiology report states a different body region. Correct the image/report pairing and retry."
                )
        # Rebuild the aggregate result using only images that did not participate
        # in a known mismatched pair.
        valid_image_outputs: list[str] = []
        for img in processed_images:
            image_file = str(img.get("file_name") or "xray_image")
            if image_file in mismatched_image_files:
                continue
            if str(img.get("status") or "").strip().lower() != "completed":
                continue
            pred = img.get("prediction")
            if isinstance(pred, str) and pred.strip():
                valid_image_outputs.append(pred.strip())
                continue
            for finding in _normalize_xray_findings(img.get("findings")):
                if _xray_finding_is_actionable(finding):
                    label = _xray_finding_label(finding)
                    if label:
                        valid_image_outputs.append(label)
        xray_finding_str = "; ".join(valid_image_outputs) if valid_image_outputs else None
        if xray_finding_str:
            xray_interpretation = "AVAILABLE"
            xray_status_display = "PARTIAL"
            xray_evidence_status = "SUPPORTING" if any(
                "model/service returned explicitly actionable candidate finding(s)" in item and "X-ray (" in item
                for item in supporting
            ) else "RECORDED"
        else:
            xray_interpretation = "REGION_MISMATCH"
            xray_status_display = "REGION_MISMATCH"
            xray_evidence_status = "UNKNOWN"

    # Problem 1: classify radiology report findings across all reports before
    # allowing them to support or reassure. Keep source statements intact.
    # Reports paired to a conflicting/unverified image remain visible as source
    # material but are not promoted to combined image/report evidence.
    radiology_groups: dict[str, list[dict[str, Any]]] = {}
    radiology_conflicts: list[dict[str, Any]] = []
    radiology_uncertain: list[dict[str, Any]] = []

    def _radiology_key(value: Any) -> str:
        import re
        text = str(value or "").lower().strip()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    for rep in processed_xray_reports:
        region_status = str((rep.get("region_verification") or {}).get("status") or "").upper()
        if processed_images and region_status in {"MISMATCH", "UNVERIFIED", "UNPAIRED"}:
            # Preserve report text in xray.reports for display/audit, but do not
            # merge unverified sources into image evidence.
            continue
        if processed_images and not bool((rep.get("region_verification") or {}).get("association_confirmed")):
            # Matching anatomy is not explicit user confirmation of the pairing.
            continue
        for qf in rep.get("qualitative_findings", []):
            if not isinstance(qf, dict):
                continue
            finding_name = str(qf.get("finding") or "").strip()
            if not finding_name:
                continue
            state = str(qf.get("state") or qf.get("status") or "UNKNOWN").strip().upper()
            if state not in {"PRESENT", "ABSENT", "UNCERTAIN", "CONFLICTING", "UNKNOWN", "HISTORICAL"}:
                state = "UNKNOWN"
            statement = {
                **qf,
                "finding": finding_name,
                "state": state,
                "source": "radiology_report",
                "source_file": rep.get("file_name"),
                "source_report_id": rep.get("id"),
            }
            key = _radiology_key(finding_name)
            if key:
                radiology_groups.setdefault(key, []).append(statement)

    for key, statements in radiology_groups.items():
        states = {item["state"] for item in statements}
        display_name = statements[0]["finding"]
        is_conflicting = "CONFLICTING" in states or ("PRESENT" in states and "ABSENT" in states)
        if is_conflicting:
            radiology_conflicts.append({
                "finding": display_name,
                "state": "CONFLICTING",
                "source": "radiology_report",
                "source_statements": statements,
                "message": "Conflicting radiology statements; finding remains unresolved and is not confirmed.",
            })
            contradictory.append(
                f"Conflicting radiology evidence for '{display_name}': "
                + " | ".join(f"{item['state']}: {item['finding']} (source: {item.get('source_file') or 'radiology report'})" for item in statements)
                + ". Finding remains unresolved and is excluded from confirmed supporting evidence."
            )
            evidence_relationships.append({
                "source_a": statements[0].get("source_file") or "radiology_report",
                "source_b": statements[-1].get("source_file") or "radiology_report",
                "finding": display_name,
                "relationship": "CONFLICTING",
                "reason": "Radiology sources contain both PRESENT and ABSENT statements, or an explicit CONFLICTING state.",
                "source_statements": statements,
            })
            continue

        if "UNCERTAIN" in states:
            radiology_uncertain.append({
                "finding": display_name,
                "state": "UNCERTAIN",
                "source": "radiology_report",
                "source_statements": statements,
                "message": "Uncertain radiology finding; not treated as confirmed supporting evidence.",
            })
            separate.append(
                f"Radiology report finding remains UNCERTAIN: {display_name}; not counted as confirmed supporting evidence."
            )
            evidence_relationships.append({
                "source_a": statements[0].get("source_file") or "radiology_report",
                "source_b": "assessment_evidence",
                "finding": display_name,
                "relationship": "UNCERTAIN",
                "reason": "Radiology wording/state is uncertain; excluded from confirmed supporting evidence.",
                "source_statements": statements,
            })
            continue

        report_region_statuses = {
            str(
                next(
                    (rep.get("region_verification", {}).get("status") for rep in processed_xray_reports
                     if str(rep.get("file_name") or "") == str(statement.get("source_file") or "")),
                    "REPORT_ONLY",
                ) or "REPORT_ONLY"
            ).upper()
            for statement in statements
        }
        has_unverified_image_pair = bool(processed_images) and bool(
            report_region_statuses.intersection({"UNVERIFIED", "UNPAIRED"})
        )
        if has_unverified_image_pair:
            # Keep the radiologist's extracted statement visible, but do not
            # claim it corroborates the uploaded image until pairing is verified.
            separate.append(
                f"Radiology report statement for review: {display_name} ({', '.join(sorted(states))}); "
                "the image/report region pairing is unverified, so this statement is kept separate from image evidence."
            )
            evidence_relationships.append({
                "source_a": statements[0].get("source_file") or "radiology_report",
                "source_b": "xray_image",
                "finding": display_name,
                "relationship": "UNVERIFIED_PAIR",
                "reason": "The report statement is preserved, but image/report region pairing has not been established.",
                "source_statements": statements,
            })
            continue

        if states == {"PRESENT"}:
            supporting.append(
                f"Radiology report finding: {display_name} (PRESENT; source: {statements[0].get('source_file') or 'radiology report'})."
            )
        elif states == {"ABSENT"}:
            reassuring.append(
                f"Radiology report explicitly records absence of: {display_name} (source: {statements[0].get('source_file') or 'radiology report'})."
            )
        else:
            separate.append(
                f"Radiology report finding '{display_name}' has state {', '.join(sorted(states)) or 'UNKNOWN'}; not used as confirmed evidence."
            )

    # Cross-modal concordance must not promote uncertain or conflicting report
    # text into confirmed image/report support.
    cross_modal_concordance: list[dict[str, Any]] = []
    if processed_images and processed_xray_reports:
        for img in processed_images:
            img_pred = img.get("prediction")
            img_findings = _normalize_xray_findings(img.get("findings"))
            img_labels = [str(img_pred).strip()] if img_pred else [
                _xray_finding_label(f)
                for f in img_findings
                if _xray_finding_label(f) and _xray_finding_is_actionable(f)
            ]
            img_status = (img.get("status") or "").lower()
            img_file = img.get("file_name") or "xray_image"
            for rep in processed_xray_reports:
                rep_file = rep.get("file_name") or "radiology_report.pdf"
                region_check = rep.get("region_verification") or {}
                region_check_status = str(region_check.get("status") or "").upper()
                allowed_image_files = region_check.get("matched_image_files") or []
                if region_check_status in {"MISMATCH", "UNVERIFIED", "UNPAIRED"}:
                    if img_file in allowed_image_files:
                        cross_modal_concordance.append({
                            "image_file": img_file,
                            "report_file": rep_file,
                            "image_prediction": None,
                            "image_findings": img_findings,
                            "concordance_status": "REGION_MISMATCH" if region_check_status == "MISMATCH" else "UNVERIFIED_REGION",
                            "details": region_check.get("message") or "Image/report regions were not verified; no cross-modal agreement is asserted.",
                        })
                    continue
                if processed_images and (img_file not in allowed_image_files):
                    continue
                rep_records = [q for q in rep.get("qualitative_findings", []) if isinstance(q, dict)]
                rep_findings = [
                    q.get("finding", "").lower() for q in rep_records
                    if str(q.get("state") or q.get("status") or "UNKNOWN").upper() == "PRESENT"
                ]
                rep_uncertain_or_conflicting = any(
                    str(q.get("state") or q.get("status") or "UNKNOWN").upper() in {"UNCERTAIN", "CONFLICTING"}
                    for q in rep_records
                )
                rep_narrative = " ".join([n.get("content", "").lower() for n in rep.get("narrative_findings", []) if isinstance(n, dict)])

                if img_status == "completed" and img_labels:
                    labels_lower = [label.lower() for label in img_labels]
                    matches = [
                        label for label in labels_lower
                        if any(label in rf for rf in rep_findings) or label in rep_narrative
                    ]
                    absent_matches = [
                        label for label in labels_lower
                        if any(f"no {label}" in rf or f"without {label}" in rf for rf in rep_findings)
                        or f"no {label}" in rep_narrative
                    ]
                    if rep_uncertain_or_conflicting and not matches:
                        status_str = "UNRESOLVED"
                        reason = f"Radiology report '{rep_file}' contains uncertain or conflicting findings; no confirmed report corroboration is asserted."
                    elif matches:
                        status_str = "PARTIALLY_CONCORDANT" if len(matches) < len(labels_lower) else "CONCORDANT"
                        reason = (
                            f"Model candidate pattern(s) {', '.join(matches)} overlap with radiology report '{rep_file}'. "
                            "This is text-level overlap, not independent diagnostic confirmation."
                        )
                        supporting.append(f"Cross-modal text-level overlap: model candidate pattern(s) {', '.join(matches)} also appear in radiology report '{rep_file}'; this does not establish a diagnosis.")
                    elif absent_matches:
                        status_str = "DISCORDANT"
                        reason = f"Model candidate pattern(s) {', '.join(absent_matches)} contrast with negative/absent wording in radiology report '{rep_file}'. Manual clinical review is needed."
                        contradictory.append(f"Cross-modal discordance: X-ray model candidate(s) {', '.join(absent_matches)} contrast with absence wording in radiology report '{rep_file}'; neither source is treated as definitive.")
                    else:
                        status_str = "INDEPENDENT"
                        reason = f"Model candidate pattern(s) {', '.join(img_labels[:18])} are not explicitly confirmed or ruled out by radiology report '{rep_file}'."
                    cross_modal_concordance.append({
                        "image_file": img_file,
                        "report_file": rep_file,
                        "image_prediction": img_pred,
                        "image_findings": img_findings,
                        "concordance_status": status_str,
                        "details": reason,
                    })
                else:
                    cross_modal_concordance.append({
                        "image_file": img_file,
                        "report_file": rep_file,
                        "image_prediction": None,
                        "image_findings": img_findings,
                        "concordance_status": "NOT_APPLICABLE",
                        "details": f"Automated image model interpretation is unavailable for '{img_file}'. Relying on extracted radiology report findings from '{rep_file}'.",
                    })

    # 4. Missing critical information (context-sensitive, never generic across unrelated domains)
    relevant_critical_checks: list[str] = []
    # Use independent domain checks rather than an elif chain so a patient with
    # symptoms from more than one domain does not lose relevant safety checks.
    if has_respiratory_symptom or present_names.intersection({"chest pain", "chest discomfort", "difficulty breathing", "shortness of breath", "dyspnea"}):
        relevant_critical_checks.extend(["difficulty breathing", "chest pain", "high fever"])
    if "headache" in present_names or any(s.get("domain") == "neurological" for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "weakness", "speech difficulty", "fever"])
    if any(s.get("domain") in {"musculoskeletal", "injury/trauma"} for s in current_present):
        relevant_critical_checks.extend(["severe swelling/deformity", "numbness", "severe bleeding"])
    if any(s.get("domain") in {"gi", "gastrointestinal", "abdominal"} for s in current_present) or any("abdominal" in str(s.get("symptom") or "").lower() for s in current_present):
        relevant_critical_checks.extend(["sudden onset", "vomiting", "blood in stool", "fever"])
    relevant_critical_checks = list(dict.fromkeys(relevant_critical_checks))

    for crit in relevant_critical_checks:
        addressed = False
        if crit in all_known_names:
            addressed = True
        elif crit == "chest pain" and any("chest pain" in n for n in all_known_names):
            addressed = True
        elif crit in {"difficulty breathing", "shortness of breath"} and any("breath" in n or "dyspnea" in n for n in all_known_names):
            addressed = True
        elif crit in {"high fever", "fever"} and ("fever" in all_known_names or "temperature" in all_known_names):
            addressed = True
        elif crit == "sudden onset" and (any(s.get("onset") for s in current_present) or any("onset" in n for n in all_known_names)):
            addressed = True
        elif crit == "weakness" and any("weakness" in n or "numb" in n for n in all_known_names):
            addressed = True
        elif crit == "speech difficulty" and any("speech" in n for n in all_known_names):
            addressed = True
        elif crit == "vomiting" and any("vomit" in n for n in all_known_names):
            addressed = True
        elif crit == "blood in stool" and any("blood in stool" in n for n in all_known_names):
            addressed = True
        elif crit == "severe bleeding" and any("bleed" in n or "hemorrhage" in n for n in all_known_names):
            addressed = True
        elif crit == "severe swelling/deformity" and any("deform" in n or "swelling" in n for n in all_known_names):
            addressed = True

        if not addressed:
            missing.append(f"Status of '{crit}' (neither confirmed nor denied)")

    durations = [s.get("duration") for s in current_present if s.get("duration")]
    if current_present and not durations:
        missing.append("Symptom duration (onset timeframe unspecified)")

    # Do not list an unselected modality as missing. When the assessment service
    # explicitly supplies selected_input_types, a selected-but-missing source is
    # surfaced; otherwise standalone/module-level analyses stay focused on inputs
    # actually submitted.
    if expects_medical_report and not has_usable_medical_report:
        if is_medical_report_provided:
            missing.append("Medical/laboratory report was uploaded, but no readable text or structured findings were extracted")
        else:
            missing.append("Medical/laboratory report was selected but no usable report was received")
    if expects_xray_image and not xray_received:
        missing.append("X-Ray image was selected but no image result was received")
    if expects_radiology_report and not has_usable_radiology_report:
        if is_radiology_report_provided:
            missing.append("Radiology report was uploaded, but no readable text or structured findings were extracted")
        else:
            missing.append("Radiology report was selected/required but was not received")

    # 5. Determine overall evidence state
    definite_qualitative_signals = sum(
        1 for item in qualitative_report_findings
        if isinstance(item, dict)
        and str(item.get("state") or item.get("status") or "").strip().upper() in {"PRESENT", "ABSENT"}
    )
    # Radiology PDFs are a distinct input stream; their extracted findings must
    # also count as evidence, without treating UNKNOWN/UNCERTAIN as confirmed.
    definite_qualitative_signals += sum(
        1
        for report in processed_xray_reports
        if not processed_images or (
            str((report.get("region_verification") or {}).get("status") or "").upper() not in {"MISMATCH", "UNVERIFIED", "UNPAIRED"}
            and bool((report.get("region_verification") or {}).get("association_confirmed"))
        )
        for item in report.get("qualitative_findings", [])
        if isinstance(item, dict)
        and str(item.get("state") or item.get("status") or "UNKNOWN").strip().upper() in {"PRESENT", "ABSENT"}
    )
    # Broad binary outputs ("normal"/"abnormal") and raw unthresholded
    # scores are useful to display, but are not specific clinical findings and
    # must not alone make the evidence state look diagnostically sufficient.
    total_active_signals = (
        len(current_present)
        + len(report_abnormal)
        + definite_qualitative_signals
        + (1 if xray_finding_str and xray_evidence_status == "SUPPORTING" else 0)
    )

    if contradictory:
        evidence_state = "CONFLICTING_EVIDENCE"
    else:
        # Upload presence, absent-only symptoms, historical-only symptoms, and
        # unavailable inference are not themselves positive clinical evidence.
        # A valid extracted value can still be evidence even when its reference
        # range/interpretation is unknown; an empty/unreadable upload cannot.
        has_recorded_lab_value = any(
            isinstance(item, dict)
            and str(item.get("test_name") or "").strip()
            and item.get("value") not in (None, "")
            for item in report_findings
        )
        if total_active_signals > 0 or bool(report_normal) or has_recorded_lab_value:
            evidence_state = "EVIDENCE_PRESENT"
        else:
            evidence_state = "INSUFFICIENT_EVIDENCE"

    has_definite_medical_report_evidence = bool(
        report_abnormal
        or report_normal
        or any(
            isinstance(item, dict)
            and str(item.get("state") or item.get("status") or "").strip().upper() in {"PRESENT", "ABSENT"}
            for item in qualitative_report_findings
        )
    )
    has_definite_radiology_report_evidence = any(
        isinstance(item, dict)
        and str(item.get("state") or item.get("status") or "UNKNOWN").strip().upper() in {"PRESENT", "ABSENT"}
        for report in processed_xray_reports
        if not processed_images or (
            str((report.get("region_verification") or {}).get("status") or "").upper() not in {"MISMATCH", "UNVERIFIED", "UNPAIRED"}
            and bool((report.get("region_verification") or {}).get("association_confirmed"))
        )
        for item in report.get("qualitative_findings", [])
    )
    has_definite_report_evidence = has_definite_medical_report_evidence or has_definite_radiology_report_evidence
    report_evidence_status = (
        "SUPPORTING" if has_definite_report_evidence
        else ("NOT_ASSESSED" if not is_report_provided else "RECORDED")
    )

    current_present_list = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "PRESENT",
            "state": "PRESENT",
            "duration": s.get("duration"),
            "severity": (s.get("severity") or "UNKNOWN").upper(),
            "body_area": s.get("body_area"),
            "laterality": s.get("laterality"),
            "quality": s.get("quality"),
            "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
            "context": s.get("context"),
            "trigger": s.get("trigger"),
            "type": s.get("type") or "CURRENT",
            "onset": s.get("onset"),
            "source": "user_input",
            "provenance": "symptom_nlp",
        }
        for s in current_present
    ]

    historical_list = [
        {
            "finding": s.get("symptom"),
            "symptom": s.get("symptom"),
            "status": "PRESENT",
            "state": "PRESENT",
            "duration": None,
            "severity": (s.get("severity") or "UNKNOWN").upper(),
            "body_area": s.get("body_area"),
            "laterality": s.get("laterality"),
            "quality": s.get("quality"),
            "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
            "context": s.get("context"),
            "trigger": s.get("trigger"),
            "type": "HISTORICAL",
            "history": s.get("history") or s.get("context") or "historical occurrence",
            "source": "user_input",
            "provenance": "symptom_nlp",
        }
        for s in historical
    ]

    return {
        "evidence_state": evidence_state,
        "input_availability": {
            "symptoms": {
                "provided": bool(symptoms),
                "present_count": len(current_present),
                "absent_count": len(symptoms_absent),
                "unknown_count": len(symptoms_unknown),
            },
            "medical_report": {
                "provided": is_medical_report_provided,
                "usable": has_usable_medical_report,
                "selected": expects_medical_report,
                "has_structured_findings": bool(report_findings or qualitative_report_findings or narrative_report_findings),
            },
            "xray_image": {
                "provided": xray_received,
                "selected": expects_xray_image,
                "status": xray_status_display,
                "region_report_mismatch": has_region_mismatch,
            },
            "radiology_report": {
                "provided": is_radiology_report_provided,
                "usable": has_usable_radiology_report,
                "selected": expects_radiology_report,
                "count": len(processed_xray_reports),
            },
        },
        "symptoms": {
            "current_present": current_present_list,
            "historical": historical_list,
            "present": current_present_list,
            "all_present": current_present_list + historical_list,
            "absent": [
                {
                    "finding": s.get("symptom"),
                    "symptom": s.get("symptom"),
                    "status": "ABSENT",
                    "state": "ABSENT",
                    "type": "NEGATIVE",
                    "body_area": s.get("body_area"),
                    "context": s.get("context"),
                    "domain": s.get("domain") or _lookup_symptom_domain(s.get("symptom")),
                    "source": "user_input",
                    "provenance": "symptom_nlp",
                }
                for s in symptoms_absent
            ],
            "unknown": [
                {
                    "finding": s.get("symptom") if isinstance(s, dict) else str(s),
                    "symptom": s.get("symptom") if isinstance(s, dict) else str(s),
                    "status": "UNKNOWN",
                    "state": "UNKNOWN",
                    "type": "UNCERTAIN",
                    "body_area": s.get("body_area") if isinstance(s, dict) else None,
                    "laterality": s.get("laterality") if isinstance(s, dict) else None,
                    "quality": s.get("quality") if isinstance(s, dict) else None,
                    "duration": s.get("duration") if isinstance(s, dict) else None,
                    "severity": (s.get("severity") or "UNKNOWN").upper() if isinstance(s, dict) else "UNKNOWN",
                    "onset": s.get("onset") if isinstance(s, dict) else None,
                    "trigger": s.get("trigger") if isinstance(s, dict) else None,
                    "context": s.get("context") if isinstance(s, dict) else None,
                    "domain": (s.get("domain") or _lookup_symptom_domain(s.get("symptom") if isinstance(s, dict) else str(s))) if isinstance(s, dict) else _lookup_symptom_domain(str(s)),
                    "source": "user_input",
                    "provenance": "symptom_nlp",
                }
                for s in symptoms_unknown
            ],
        },
        "report": {
            # `provided` now means a medical/laboratory report was provided.
            "provided": is_medical_report_provided,
            "medical_report_provided": is_medical_report_provided,
            "radiology_report_provided": is_radiology_report_provided,
            "any_report_provided": is_report_provided,
            "evidence_status": report_evidence_status,
            "medical_reports": medical_reports,
            "xray_reports": processed_xray_reports,
            "abnormal_findings": [
                {
                    **f,
                    "status": f.get("status") or f.get("interpretation") or "ABNORMAL",
                    "interpretation": f.get("interpretation") or "ABNORMAL",
                    "source": "uploaded_report",
                    "provenance": "report_ocr_nlp",
                }
                for f in report_abnormal
            ],
            "normal_findings": [
                {
                    **f,
                    "status": f.get("status") or "NORMAL",
                    "interpretation": "NORMAL",
                    "source": "uploaded_report",
                    "provenance": "report_ocr_nlp",
                }
                for f in report_normal
            ],
            "qualitative_findings": qualitative_report_findings,
            "narrative_findings": narrative_report_findings,
            "radiology_conflicts": radiology_conflicts,
            "radiology_uncertain_findings": radiology_uncertain,
            "unknown_range_findings": report_unknown_range,
            "total_extracted": (
                len(report_findings)
                + len(qualitative_report_findings)
                + len(narrative_report_findings)
                + sum(
                    len(rep.get("qualitative_findings", [])) + len(rep.get("narrative_findings", []))
                    for rep in processed_xray_reports
                )
            ),
        },
        "xray": {
            "received": xray_received,
            "interpretation": xray_interpretation,
            "evidence_status": xray_evidence_status,
            "status": xray_status_display,
            "region": xray_region,
            "prediction": xray_finding_str,
            "images": processed_images,
            "reports": processed_xray_reports,
            "cross_modal_concordance": cross_modal_concordance,
            "region_verification": region_verification_records,
            "has_region_mismatch": has_region_mismatch,
            "model_version": (xray_images[0] if xray_images else (xray_result or {})).get("model_version") or "N/A",
            "uncertainty": (xray_images[0] if xray_images else (xray_result or {})).get("uncertainty"),
            "explainability_artifact": (xray_images[0] if xray_images else (xray_result or {})).get("explainability_artifact"),
            "message": (xray_images[0] if xray_images else (xray_result or {})).get("message"),
            "provenance": "xray_deep_learning" if xray_received else "none",
        },
        "supporting": supporting,
        "reassuring": reassuring,
        "separate": separate,
        "separate_or_contextual": separate,
        "unassessed": unassessed,
        "contradictory": contradictory,
        "missing": missing,
        # These are unanswered safety checks, not confirmed negative symptoms.
        # Downstream triage/follow-up layers can consume them without parsing prose.
        "critical_unknowns": [
            item for item in missing
            if any(term in item.lower() for term in (
                "difficulty breathing", "chest pain", "high fever", "severe bleeding",
                "sudden onset", "speech difficulty", "weakness", "numbness",
                "severe swelling/deformity", "vomiting", "blood in stool", "fever",
            ))
        ],
        "follow_up_required": any(
            any(term in item.lower() for term in (
                "difficulty breathing", "chest pain", "high fever", "severe bleeding",
                "sudden onset", "speech difficulty", "weakness", "numbness",
                "severe swelling/deformity", "vomiting", "blood in stool", "fever",
            ))
            for item in missing
        ),
        "evidence_relationships": evidence_relationships,
        "message": (
            "Multimodal evidence compiled successfully."
            if evidence_state == "EVIDENCE_PRESENT"
            else "Available information is insufficient for a specific clinical conclusion."
        ),
    }
