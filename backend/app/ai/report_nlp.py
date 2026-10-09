"""Medical report NLP extraction service for MediSense AI.

Extracts structured clinical test parameters, reference ranges, and findings from report text and tables:
- Test name (e.g. Hemoglobin, WBC, Platelets, Fasting Blood Glucose, Serum Creatinine, etc.)
- Value (extracted directly from document text/tables; NEVER invented)
- Unit (e.g. g/dL, mg/dL, /µL, U/L, %)
- Reference range (only if present in report; NEVER invented)
- Normal / Abnormal / High / Low relative to the report's reference range or explicit flag
- Qualitative radiology keywords and clinical findings preserved
"""

from __future__ import annotations

import re
from typing import Any, Optional

REPORT_NLP_VERSION = "report-nlp-v2.0"

# Target clinical tests with synonyms and typical units for pattern matching
TARGET_TESTS: list[dict[str, Any]] = [
    {
        "canonical": "Hemoglobin",
        "synonyms": [r"hemoglobin", r"haemoglobin", r"\bhb\b"],
        "domain": "hematology",
    },
    {
        "canonical": "White Blood Cell Count (WBC)",
        "synonyms": [r"white blood cell(?: count)?", r"total leukocyte count", r"\btlc\b", r"\bwbc(?: count)?\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Platelet Count",
        "synonyms": [r"platelet(?:s)?(?: count)?", r"thrombocyte(?: count)?", r"\bplt\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Red Blood Cell Count (RBC)",
        "synonyms": [r"red blood cell(?: count)?", r"total erythrocyte count", r"\brbc\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Fasting Blood Glucose",
        "synonyms": [
            r"fasting blood sugar",
            r"fasting blood glucose",
            r"fasting glucose",
            r"glucose,?\s*fasting",
            r"glucose\s*\(fasting\)",
            r"\bfbs\b",
        ],
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
        "synonyms": [r"ast\s*\(sgot\)", r"ast\/sgot", r"sgot", r"aspartate aminotransferase", r"\bast\b"],
        "domain": "liver",
    },
    {
        "canonical": "SGPT / ALT",
        "synonyms": [r"alt\s*\(sgpt\)", r"alt\/sgpt", r"sgpt", r"alanine aminotransferase", r"\balt\b"],
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
    {
        "canonical": "C-Reactive Protein (CRP)",
        "synonyms": [r"c-reactive protein", r"\bcrp\b"],
        "domain": "inflammation",
    },
    {
        "canonical": "Neutrophils",
        "synonyms": [r"neutrophil(?:s)?\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Lymphocytes",
        "synonyms": [r"lymphocyte(?:s)?\b"],
        "domain": "hematology",
    },
]

# Common unit patterns
UNIT_REGEX = r"(g\/dL|mg\/dL|mcL|\/uL|\/cumm|U\/L|uIU\/mL|%|fl|pg|mmol\/L|mEq\/L|mg\/L|lakh\/[u\u00b5\u03bc]L|\/[u\u00b5\u03bc]L)"

# Reference range patterns supporting hyphens, en-dashes, em-dashes, and unicode dashes
RANGE_REGEX = re.compile(
    r"(?:ref(?:erence)?\s*(?:range|interval)?\s*[:\s]*)?([<>]?\s*[\d,]+(?:\.\d+)?)\s*(?:-|to|[–—−\x96\u2013\u2014])\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
LESS_THAN_RANGE_REGEX = re.compile(r"<\s*([\d,]+(?:\.\d+)?)")
GREATER_THAN_RANGE_REGEX = re.compile(r">\s*([\d,]+(?:\.\d+)?)")


def _evaluate_range(val: float, low: Optional[float], high: Optional[float]) -> str:
    if low is not None and val < low:
        return "LOW"
    if high is not None and val > high:
        return "HIGH"
    if low is not None or high is not None:
        return "NORMAL"
    return "UNKNOWN"


def structure_report_text(
    text: str | None,
    tables: list[list[list[str]]] | None = None,
) -> dict[str, Any]:
    """Parse medical report text and tabular data into structured lab findings and reference ranges."""
    if (not text or not text.strip()) and not tables:
        return {
            "status": "unavailable",
            "findings": [],
            "reference_ranges": [],
            "qualitative_findings": [],
            "message": "No report text or tables available for NLP structuring.",
            "source": "uploaded_report",
            "report_nlp_version": REPORT_NLP_VERSION,
        }

    findings: list[dict[str, Any]] = []
    seen_tests: set[str] = set()

    # 1. First, process structured tables if provided by OCR/PDF parser
    if tables:
        for table in tables:
            if not table or len(table) < 2:
                continue

            # Detect column headers
            header = [str(cell or "").strip().lower() for cell in table[0]]
            col_test = 0
            col_val = 1
            col_ref = 2 if len(header) > 2 else None
            col_flag = 3 if len(header) > 3 else None

            for i, h in enumerate(header):
                if any(k in h for k in ["test", "parameter", "investigation"]):
                    col_test = i
                elif any(k in h for k in ["result", "value", "observed"]):
                    col_val = i
                elif any(k in h for k in ["reference", "range", "interval", "normal"]):
                    col_ref = i
                elif any(k in h for k in ["flag", "status", "remark", "interpretation"]):
                    col_flag = i

            for row in table[1:]:
                if len(row) <= max(col_test, col_val):
                    continue

                test_raw = str(row[col_test] or "").strip()
                val_raw = str(row[col_val] or "").strip()
                ref_raw = str(row[col_ref] or "").strip() if (col_ref is not None and len(row) > col_ref) else ""
                flag_raw = str(row[col_flag] or "").strip() if (col_flag is not None and len(row) > col_flag) else ""

                for test_def in TARGET_TESTS:
                    canonical = test_def["canonical"]
                    if canonical in seen_tests:
                        continue

                    matched = False
                    for syn in test_def["synonyms"]:
                        if re.search(r"\b" + syn + r"\b", test_raw, re.IGNORECASE):
                            matched = True
                            break

                    if not matched:
                        continue

                    # Extract numeric value
                    num_match = re.search(r"([\d,]+(?:\.\d+)?)", val_raw)
                    if not num_match:
                        continue
                    try:
                        val_float = float(num_match.group(1).replace(",", ""))
                    except ValueError:
                        continue

                    # Extract unit
                    unit_m = re.search(UNIT_REGEX, val_raw, re.IGNORECASE)
                    unit_str = unit_m.group(1) if unit_m else None

                    # Extract reference range
                    ref_low = None
                    ref_high = None
                    ref_str = ref_raw if ref_raw else None

                    if ref_raw:
                        rm = RANGE_REGEX.search(ref_raw)
                        if rm:
                            try:
                                ref_low = float(rm.group(1).replace(",", "").replace("<", "").replace(">", "").strip())
                                ref_high = float(rm.group(2).replace(",", "").strip())
                                ref_str = f"{ref_low} - {ref_high}"
                            except ValueError:
                                pass
                        else:
                            lt_m = LESS_THAN_RANGE_REGEX.search(ref_raw)
                            if lt_m:
                                try:
                                    ref_high = float(lt_m.group(1).replace(",", ""))
                                    ref_str = f"< {ref_high}"
                                except ValueError:
                                    pass
                            else:
                                gt_m = GREATER_THAN_RANGE_REGEX.search(ref_raw)
                                if gt_m:
                                    try:
                                        ref_low = float(gt_m.group(1).replace(",", ""))
                                        ref_str = f"> {ref_low}"
                                    except ValueError:
                                        pass

                    # Determine interpretation
                    interp = None
                    flag_lower = flag_raw.lower()
                    if "high" in flag_lower:
                        interp = "HIGH"
                    elif "low" in flag_lower:
                        interp = "LOW"
                    elif "normal" in flag_lower:
                        interp = "NORMAL"
                    else:
                        interp = _evaluate_range(val_float, ref_low, ref_high)

                    seen_tests.add(canonical)
                    findings.append({
                        "test_name": canonical,
                        "value": val_float,
                        "unit": unit_str,
                        "reference_range": ref_str,
                        "status": interp,
                        "interpretation": interp,
                        "domain": test_def["domain"],
                        "source_report": "uploaded_report",
                        "raw_line": f"{test_raw} | {val_raw} | {ref_raw} | {flag_raw}".strip(),
                    })
                    break

    # 2. Text line-by-line fallback with multi-line sliding window for any unextracted tests
    if text and text.strip():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
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

                # Build candidate text block (current line + up to 4 following lines, stopping at next test)
                block = [line]
                for forward in range(idx + 1, min(idx + 5, len(lines))):
                    fline = lines[forward]
                    if any(
                        re.search(r"\b" + s + r"\b", fline, re.IGNORECASE)
                        for td in TARGET_TESTS
                        for s in td["synonyms"]
                    ):
                        break
                    block.append(fline)
                combined = " ".join(block)

                # Look for numeric value
                val_float = None
                # Check if a line in block is value + unit
                for b_line in block:
                    m = re.search(r"^([\d,]+(?:\.\d+)?)\s*" + UNIT_REGEX, b_line, re.IGNORECASE)
                    if m:
                        try:
                            val_float = float(m.group(1).replace(",", ""))
                            break
                        except ValueError:
                            pass
                if val_float is None:
                    # Look for first number in combined text that is not in the test name
                    num_matches = list(re.finditer(r"\b([\d,]+(?:\.\d+)?)\b", combined))
                    for nm in num_matches:
                        num_str = nm.group(1).replace(",", "")
                        try:
                            vf = float(num_str)
                            # Avoid matching year like 2026
                            if vf < 1900 or vf > 2100:
                                val_float = vf
                                break
                        except ValueError:
                            continue

                if val_float is None:
                    continue

                # Unit
                unit_match = re.search(UNIT_REGEX, combined, re.IGNORECASE)
                unit_str = unit_match.group(1) if unit_match else None

                # Reference range
                ref_low = None
                ref_high = None
                ref_str = None

                range_m = RANGE_REGEX.search(combined)
                if range_m:
                    try:
                        ref_low = float(range_m.group(1).replace(",", "").replace("<", "").replace(">", "").strip())
                        ref_high = float(range_m.group(2).replace(",", "").strip())
                        ref_str = f"{ref_low} - {ref_high}"
                    except ValueError:
                        pass
                else:
                    lt_m = LESS_THAN_RANGE_REGEX.search(combined)
                    if lt_m:
                        try:
                            ref_high = float(lt_m.group(1).replace(",", ""))
                            ref_str = f"< {ref_high}"
                        except ValueError:
                            pass
                    else:
                        gt_m = GREATER_THAN_RANGE_REGEX.search(combined)
                        if gt_m:
                            try:
                                ref_low = float(gt_m.group(1).replace(",", ""))
                                ref_str = f"> {ref_low}"
                            except ValueError:
                                pass

                # Interpretation / flag
                interp = None
                comb_lower = combined.lower()
                if re.search(r"\bhigh\*?\b", comb_lower):
                    interp = "HIGH"
                elif re.search(r"\blow\*?\b", comb_lower):
                    interp = "LOW"
                elif re.search(r"\bnormal\b", comb_lower):
                    interp = "NORMAL"
                else:
                    interp = _evaluate_range(val_float, ref_low, ref_high)

                seen_tests.add(canonical)
                findings.append({
                    "test_name": canonical,
                    "value": val_float,
                    "unit": unit_str,
                    "reference_range": ref_str,
                    "status": interp,
                    "interpretation": interp,
                    "domain": test_def["domain"],
                    "source_report": "uploaded_report",
                    "raw_line": line if len(line) < 140 else line[:137] + "...",
                })
                break

    # 3. Qualitative and narrative clinical findings extraction
    qualitative_findings: list[dict[str, Any]] = []
    narrative_findings: list[dict[str, Any]] = []

    if text and text.strip():
        lower_text = text.lower()

        # Section extraction (Clinical History, Examination, Impression, Findings)
        section_pattern = re.compile(
            r"(?:^|\n)\s*(Clinical History|History|Clinical Indication|Indication|Chief Complaint|Physical Examination|Clinical Examination|Examination|Impression|Conclusion|Diagnosis|Radiology Findings|Findings)\s*:\s*(.*?)(?=(?:\n\s*(?:Clinical History|History|Clinical Indication|Indication|Chief Complaint|Physical Examination|Clinical Examination|Examination|Impression|Conclusion|Diagnosis|Radiology Findings|Findings)\s*:)|$)",
            re.IGNORECASE | re.DOTALL,
        )
        for smatch in section_pattern.finditer(text):
            sec_name = smatch.group(1).strip()
            sec_content = smatch.group(2).strip()
            if sec_content:
                narrative_findings.append({
                    "section": sec_name,
                    "content": sec_content,
                    "source": "medical_report_narrative",
                })

        # Explicit fracture assessment:
        # "No fracture" -> ABSENT
        # "Fracture identified" / "Fracture noted" -> PRESENT
        # neither -> not added (remains UNKNOWN, not inferred from wrist injury)
        has_no_fracture = bool(
            re.search(r"\b(?:no|denies|negative for|free of|without)\s+(?:\w+\s+){0,3}fracture\b", lower_text)
            or "no fracture" in lower_text
        )
        has_pos_fracture = bool(
            re.search(r"\b(?:fracture identified|fracture seen|fracture noted|acute fracture|evidence of fracture)\b", lower_text)
            or (re.search(r"\bfracture\b", lower_text) and not has_no_fracture)
        )

        if has_no_fracture:
            qualitative_findings.append({
                "finding": "fracture",
                "category": "radiology/bone",
                "domain": "musculoskeletal",
                "state": "ABSENT",
            })
        elif has_pos_fracture:
            qualitative_findings.append({
                "finding": "fracture",
                "category": "radiology/bone",
                "domain": "musculoskeletal",
                "state": "PRESENT",
            })

        # Examination findings: tenderness, swelling, reduced range of motion
        if re.search(r"\b(?:tenderness|tender)\b", lower_text):
            neg_tender = bool(re.search(r"\b(?:no|without)\s+(?:\w+\s+){0,2}(?:tenderness|tender)\b", lower_text))
            qualitative_findings.append({
                "finding": "localized tenderness",
                "category": "examination",
                "domain": "musculoskeletal",
                "state": "ABSENT" if neg_tender else "PRESENT",
            })

        if re.search(r"\b(?:swelling|swollen|edema)\b", lower_text):
            neg_swell = bool(re.search(r"\b(?:no|without)\s+(?:\w+\s+){0,2}(?:swelling|swollen|edema)\b", lower_text))
            qualitative_findings.append({
                "finding": "swelling",
                "category": "examination",
                "domain": "musculoskeletal",
                "state": "ABSENT" if neg_swell else "PRESENT",
            })

        if re.search(r"\b(?:reduced|decreased|limited|restricted)\s+(?:range of motion|movement|mobility|rom)\b", lower_text):
            qualitative_findings.append({
                "finding": "reduced range of motion",
                "category": "examination",
                "domain": "musculoskeletal",
                "state": "PRESENT",
            })

        # Respiratory qualitative findings
        resp_terms = [
            ("pneumonia", "radiology", "respiratory"),
            ("consolidation", "radiology", "respiratory"),
            ("pleural effusion", "radiology", "respiratory"),
            ("cardiomegaly", "radiology", "cardiovascular"),
            ("infiltrate", "radiology", "respiratory"),
            ("clear lungs", "radiology", "respiratory"),
        ]
        for term, cat, dom in resp_terms:
            if term in lower_text:
                neg = bool(re.search(rf"\b(?:no|denies|negative for|free of|without)\s+(?:\w+\s+){{0,3}}{term}\b", lower_text))
                qualitative_findings.append({
                    "finding": term,
                    "category": cat,
                    "domain": dom,
                    "state": "ABSENT" if neg else "PRESENT",
                })

    has_content = bool(findings or qualitative_findings or narrative_findings)
    status_str = "structured" if has_content else "unstructured"

    if findings:
        msg = f"Successfully extracted {len(findings)} structured lab parameter(s)."
    elif qualitative_findings or narrative_findings:
        msg = f"Extracted {len(qualitative_findings)} clinical finding(s) and {len(narrative_findings)} narrative section(s) from document."
    else:
        msg = "No structured clinical values or narrative sections detected in the uploaded document."

    return {
        "status": status_str,
        "findings": findings,
        "qualitative_findings": qualitative_findings,
        "narrative_findings": narrative_findings,
        "reference_ranges": [
            {"test_name": f["test_name"], "range": f["reference_range"]}
            for f in findings
            if f.get("reference_range")
        ],
        "message": msg,
        "source": "uploaded_report",
        "report_nlp_version": REPORT_NLP_VERSION,
    }
