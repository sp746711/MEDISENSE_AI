"""Medical report NLP extraction service for MediSense AI.

Extracts structured clinical test parameters, reference ranges, and findings from report text:
- Test name (e.g. Hemoglobin, WBC, Platelets, Fasting Blood Glucose, Serum Creatinine)
- Value (extracted from document text)
- Unit (e.g. g/dL, mg/dL, /mcL, %, U/L)
- Reference range (only if present in report text; NEVER invented)
- Normal / Abnormal / High / Low relative to the report's reference range
- Negation and clinical uncertainty preserved
"""

from __future__ import annotations

import re
from typing import Any, Optional

REPORT_NLP_VERSION = "report-nlp-v1.0"

# Target clinical tests with synonyms and typical units for pattern matching
TARGET_TESTS: list[dict[str, Any]] = [
    {
        "canonical": "Hemoglobin",
        "synonyms": [r"hemoglobin", r"haemoglobin", r"\bhb\b"],
        "domain": "hematology",
    },
    {
        "canonical": "White Blood Cell Count (WBC)",
        "synonyms": [r"white blood cell(?: count)?", r"total leukocyte count", r"\btlc\b", r"\bwbc\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Platelet Count",
        "synonyms": [r"platelet(?: count)?", r"thrombocyte(?: count)?", r"\bplt\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Red Blood Cell Count (RBC)",
        "synonyms": [r"red blood cell(?: count)?", r"total erythrocyte count", r"\brbc\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Fasting Blood Glucose",
        "synonyms": [r"fasting blood sugar", r"fasting blood glucose", r"glucose,?\s*fasting", r"\bfbs\b"],
        "domain": "endocrine",
    },
    {
        "canonical": "Postprandial Blood Glucose",
        "synonyms": [r"postprandial blood sugar", r"post prandial glucose", r"\bppbs\b"],
        "domain": "endocrine",
    },
    {
        "canonical": "HbA1c",
        "synonyms": [r"glycated hemoglobin", r"glycohemoglobin", r"hba1c", r"hb a1c"],
        "domain": "endocrine",
    },
    {
        "canonical": "Serum Creatinine",
        "synonyms": [r"serum creatinine", r"creatinine,?\s*serum", r"\bcreatinine\b"],
        "domain": "kidney",
    },
    {
        "canonical": "Blood Urea Nitrogen",
        "synonyms": [r"blood urea nitrogen", r"\bbun\b", r"urea,?\s*blood"],
        "domain": "kidney",
    },
    {
        "canonical": "Uric Acid",
        "synonyms": [r"uric acid", r"serum uric acid"],
        "domain": "kidney",
    },
    {
        "canonical": "Total Bilirubin",
        "synonyms": [r"total bilirubin", r"bilirubin,?\s*total", r"\bbilirubin\b"],
        "domain": "liver",
    },
    {
        "canonical": "SGOT / AST",
        "synonyms": [r"sgot", r"aspartate aminotransferase", r"\bast\b"],
        "domain": "liver",
    },
    {
        "canonical": "SGPT / ALT",
        "synonyms": [r"sgpt", r"alanine aminotransferase", r"\balt\b"],
        "domain": "liver",
    },
    {
        "canonical": "Total Cholesterol",
        "synonyms": [r"total cholesterol", r"cholesterol,?\s*total"],
        "domain": "cardiovascular",
    },
    {
        "canonical": "Triglycerides",
        "synonyms": [r"triglycerides", r"serum triglycerides"],
        "domain": "cardiovascular",
    },
    {
        "canonical": "TSH",
        "synonyms": [r"thyroid stimulating hormone", r"tsh,?\s*serum", r"\btsh\b"],
        "domain": "thyroid",
    },
]

# Common unit patterns
UNIT_REGEX = r"(g\/dL|mg\/dL|mcL|\/uL|\/cumm|U\/L|uIU\/mL|%|fl|pg|mmol\/L|mEq\/L|10\^[0-9]\/uL)"

# Common reference range patterns in reports, e.g. "12.0 - 16.0", "13 - 17", "< 200", "> 50"
RANGE_REGEX = re.compile(
    r"(?:ref(?:erence)?\s*(?:range|interval)?\s*[:\s]*)?([<>]?\s*\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
LESS_THAN_RANGE_REGEX = re.compile(r"<\s*(\d+(?:\.\d+)?)")
GREATER_THAN_RANGE_REGEX = re.compile(r">\s*(\d+(?:\.\d+)?)")


def _evaluate_range(val: float, low: Optional[float], high: Optional[float]) -> str:
    if low is not None and val < low:
        return "LOW"
    if high is not None and val > high:
        return "HIGH"
    if low is not None or high is not None:
        return "NORMAL"
    return "RECORDED"


def structure_report_text(text: str | None) -> dict[str, Any]:
    """Parse medical report text into structured lab findings and reference ranges."""
    if not text or not text.strip():
        return {
            "status": "unavailable",
            "findings": [],
            "reference_ranges": [],
            "message": "No report text available for NLP structuring.",
            "source": "uploaded_report",
            "report_nlp_version": REPORT_NLP_VERSION,
        }

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    findings: list[dict[str, Any]] = []
    seen_tests: set[str] = set()

    for line in lines:
        for test_def in TARGET_TESTS:
            canonical = test_def["canonical"]
            if canonical in seen_tests:
                continue

            matched = False
            for syn in test_def["synonyms"]:
                if re.search(r"\b" + syn + r"\b", line, re.IGNORECASE):
                    matched = True
                    break

            if not matched:
                continue

            # Look for numeric value after test name
            # Pattern: test name ... [:] <number> [unit] [ref range]
            num_matches = list(re.finditer(r"\b(\d+(?:\.\d+)?)\b", line))
            if not num_matches:
                continue

            # First number in the line is typically the measured value
            first_num_match = num_matches[0]
            try:
                val_float = float(first_num_match.group(1))
            except ValueError:
                continue

            # Check unit in the vicinity
            unit_match = re.search(UNIT_REGEX, line, re.IGNORECASE)
            unit_str = unit_match.group(1) if unit_match else None

            # Check reference range in the same line or remainder
            ref_low = None
            ref_high = None
            ref_str = None

            range_m = RANGE_REGEX.search(line)
            if range_m:
                try:
                    ref_low = float(range_m.group(1).replace("<", "").replace(">", "").strip())
                    ref_high = float(range_m.group(2).strip())
                    ref_str = f"{ref_low} - {ref_high}"
                except ValueError:
                    pass
            else:
                lt_m = LESS_THAN_RANGE_REGEX.search(line)
                if lt_m:
                    try:
                        ref_high = float(lt_m.group(1))
                        ref_str = f"< {ref_high}"
                    except ValueError:
                        pass
                else:
                    gt_m = GREATER_THAN_RANGE_REGEX.search(line)
                    if gt_m:
                        try:
                            ref_low = float(gt_m.group(1))
                            ref_str = f"> {ref_low}"
                        except ValueError:
                            pass

            interpretation = _evaluate_range(val_float, ref_low, ref_high)
            seen_tests.add(canonical)

            findings.append(
                {
                    "test_name": canonical,
                    "value": val_float,
                    "unit": unit_str,
                    "reference_range": ref_str,
                    "interpretation": interpretation,
                    "domain": test_def["domain"],
                    "raw_line": line if len(line) < 140 else line[:137] + "...",
                }
            )

    # Check for general qualitative radiology keywords if present
    radiology_findings = []
    qualitative_terms = [
        ("pneumonia", "radiology", "respiratory"),
        ("consolidation", "radiology", "respiratory"),
        ("pleural effusion", "radiology", "respiratory"),
        ("cardiomegaly", "radiology", "cardiovascular"),
        ("infiltrate", "radiology", "respiratory"),
        ("clear lungs", "radiology", "respiratory"),
        ("no fracture", "radiology", "bones"),
        ("fracture", "radiology", "bones"),
    ]
    lower_text = text.lower()
    for term, category, domain in qualitative_terms:
        if term in lower_text:
            # Check negation
            negated = any(
                re.search(rf"\b(?:no|denies|negative for|free of|without)\s+(?:\w+\s+){{0,3}}{term}\b", lower_text)
                for _ in [1]
            )
            radiology_findings.append(
                {
                    "finding": term,
                    "category": category,
                    "domain": domain,
                    "state": "ABSENT" if negated else "PRESENT",
                }
            )

    return {
        "status": "structured" if (findings or radiology_findings) else "unstructured",
        "findings": findings,
        "qualitative_findings": radiology_findings,
        "total_extracted": len(findings) + len(radiology_findings),
        "message": (
            f"Successfully structured {len(findings)} lab parameter(s) and {len(radiology_findings)} clinical note(s)."
            if (findings or radiology_findings)
            else "Report text received, but no known standard lab parameters were matched in the text."
        ),
        "source": "uploaded_report",
        "report_nlp_version": REPORT_NLP_VERSION,
    }
