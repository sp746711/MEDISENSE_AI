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
        "synonyms": [r"white blood cell(?:s)?(?: count)?", r"total leukocyte(?:s)?(?: count)?", r"\btlc\b", r"\bwbc(?: count)?\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Platelet Count",
        "synonyms": [r"platelet(?:s)?(?: count)?", r"thrombocyte(?: count)?", r"\bplt\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Red Blood Cell Count (RBC)",
        "synonyms": [r"red blood cell(?:s)?(?: count)?", r"total erythrocyte(?:s)?(?: count)?", r"\brbc\b"],
        "domain": "hematology",
    },
    {
        "canonical": "Hematocrit",
        "synonyms": [
            r"hematocrit",
            r"haematocrit",
            r"\bhct\b",
            r"packed cell volume",
            r"\bpcv\b",
        ],
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

# Comprehensive clinical unit patterns ordered from most specific/compound to least specific
UNIT_PATTERNS = [
    # RBC and cell count compound units with multiplier phrases and volume
    r"(?:million\s+cells?|mil(?:lion)?|cells?|lakh)\s*\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"(?:x\s*10\^?6|10\^?6|x\s*10\*6|10\*6)\s*\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"(?:x\s*10\^?12|10\^?12|x\s*10\*12|10\*12)\s*\/\s*L",
    r"mil\s*\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"M\s*\/\s*(?:[u\u00b5\u03bc]L|mcL)",
    r"(?:million\s+cells?|mil(?:lion)?)\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    # Standard mass concentration units
    r"g\/dL",
    r"mg\/dL",
    r"g\/L",
    r"mg\/L",
    r"mcg\/dL",
    r"ug\/dL",
    r"[\u00b5\u03bc]g\/dL",
    r"mmol\/L",
    r"mEq\/L",
    r"[\u00b5\u03bc]mol\/L",
    r"umol\/L",
    r"uIU\/mL",
    r"[\u00b5\u03bc]IU\/mL",
    r"mIU\/L",
    r"U\/L",
    r"IU\/L",
    r"ng\/mL",
    r"ng\/dL",
    # Percentage (Hematocrit, differential counts)
    r"%",
    # Cell volume and hemoglobin weight indices
    r"fl|fL",
    r"pg",
    # Base cell counts per volume
    r"cells\s*\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"lakh\s*\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"\/\s*(?:[u\u00b5\u03bc]L|mcL|cumm|mm\^?3)",
    r"\/uL",
    r"\/cumm",
    r"mcL",
]
UNIT_REGEX = r"(" + "|".join(UNIT_PATTERNS) + r")"

# Reference range patterns supporting hyphens, en-dashes, em-dashes, and unicode dashes
RANGE_REGEX = re.compile(
    r"(?:ref(?:erence)?\s*(?:range|interval)?\s*[:\s]*)?([<>]?\s*[\d,]+(?:\.\d+)?)\s*(?:-|to|[–—−\x96\u2013\u2014])\s*([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
LESS_THAN_RANGE_REGEX = re.compile(r"<\s*([\d,]+(?:\.\d+)?)")
GREATER_THAN_RANGE_REGEX = re.compile(r">\s*([\d,]+(?:\.\d+)?)")

SYNTHETIC_INSTRUCTION_RE = re.compile(
    r"\b(?:synthetic|demonstration|demo\s+only|sample\s+report|test\s+case|no\s+actual\s+clinical\s+findings|instructions?|placeholder|dummy\s+data|simulated|example\s+report)\b",
    re.IGNORECASE,
)


def is_synthetic_text(text: str | None) -> bool:
    if not text:
        return False
    return bool(SYNTHETIC_INSTRUCTION_RE.search(text))


def _evaluate_range(val: float, low: Optional[float], high: Optional[float]) -> str:
    if low is not None and val < low:
        return "LOW"
    if high is not None and val > high:
        return "HIGH"
    if low is not None or high is not None:
        return "NORMAL"
    return "UNKNOWN"


def _is_new_test_or_section(line_text: str) -> bool:
    """Determine if a line begins a new test record or section header to prevent cross-row contamination."""
    clean = line_text.strip()
    if not clean:
        return False

    # 1. Any synonym of target tests
    for td in TARGET_TESTS:
        for s in td["synonyms"]:
            if re.search(r"\b" + s + r"\b", clean, re.IGNORECASE):
                return True

    # 2. Recognized section header
    if re.match(
        r"^(?:clinical history|history|impression|conclusion|diagnosis|findings|examination|radiology|comments?|notes?|method|department|sample|specimen|patient|doctor|dr\.|reported\s+on|collected\s+on)\s*[:\-]",
        clean,
        re.IGNORECASE,
    ):
        return True

    # 3. Another clinical parameter line with separator (e.g., "ESR: 25", "MCV = 85", "Sodium: 140")
    if re.match(
        r"^(?!(?:ref(?:erence)?|normal|flag|status|remark|interpretation|result|observed|units?)\b)[A-Za-z][A-Za-z0-9\s\(\)\/\-\,\.]{1,35}\s*[:=\t|]\s*[\d<>]",
        clean,
        re.IGNORECASE,
    ):
        return True

    # 4. Another clinical parameter line with direct value and unit
    if re.match(
        r"^(?!(?:ref(?:erence)?|normal|flag|status|remark|interpretation|result|observed|units?)\b)[A-Za-z][A-Za-z0-9\s\(\)\/\-\,\.]{2,30}\s+[\d\.]+\s*(?:%|g\/|mg\/|cells|\/|fl|pg|mmol|mil|million)",
        clean,
        re.IGNORECASE,
    ):
        return True

    return False


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
            col_unit = None
            col_ref = None
            col_flag = None

            for i, h in enumerate(header):
                if any(k in h for k in ["test", "parameter", "investigation"]):
                    col_test = i
                elif any(k in h for k in ["result", "value", "observed"]):
                    col_val = i
                elif any(k in h for k in ["unit", "units"]):
                    col_unit = i
                elif any(k in h for k in ["reference", "range", "interval", "normal"]):
                    col_ref = i
                elif any(k in h for k in ["flag", "status", "remark", "interpretation"]):
                    col_flag = i

            for row in table[1:]:
                if len(row) <= max(col_test, col_val):
                    continue

                test_raw = str(row[col_test] or "").strip()
                val_raw = str(row[col_val] or "").strip()
                unit_raw = str(row[col_unit] or "").strip() if (col_unit is not None and len(row) > col_unit) else ""
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

                    # Extract unit (from unit column if present, else from val_raw)
                    unit_str = None
                    if unit_raw:
                        um = re.search(UNIT_REGEX, unit_raw, re.IGNORECASE)
                        unit_str = um.group(1).strip() if um else unit_raw
                    if not unit_str:
                        um = re.search(UNIT_REGEX, val_raw, re.IGNORECASE)
                        unit_str = um.group(1).strip() if um else None

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

                    # Determine interpretation & detect explicit flag conflict
                    calculated_status = _evaluate_range(val_float, ref_low, ref_high)
                    explicit_flag = None
                    flag_lower = flag_raw.lower().strip()
                    if "high" in flag_lower:
                        explicit_flag = "HIGH"
                    elif "low" in flag_lower:
                        explicit_flag = "LOW"
                    elif "normal" in flag_lower:
                        explicit_flag = "NORMAL"
                    elif "abnormal" in flag_lower:
                        explicit_flag = "ABNORMAL"

                    conflict = False
                    conflict_detail = None
                    if explicit_flag and calculated_status != "UNKNOWN":
                        if explicit_flag == "NORMAL" and calculated_status in {"HIGH", "LOW"}:
                            conflict = True
                        elif explicit_flag in {"HIGH", "LOW"} and calculated_status == "NORMAL":
                            conflict = True
                        elif explicit_flag == "HIGH" and calculated_status == "LOW":
                            conflict = True
                        elif explicit_flag == "LOW" and calculated_status == "HIGH":
                            conflict = True

                    if conflict:
                        conflict_detail = f"Explicit flag '{explicit_flag}' conflicts with calculated status '{calculated_status}'"
                        final_status = explicit_flag
                    elif explicit_flag:
                        final_status = explicit_flag
                    else:
                        final_status = calculated_status

                    seen_tests.add(canonical)
                    raw_parts = [test_raw, val_raw]
                    if unit_raw:
                        raw_parts.append(unit_raw)
                    if ref_raw:
                        raw_parts.append(ref_raw)
                    if flag_raw:
                        raw_parts.append(flag_raw)

                    findings.append({
                        "test_name": canonical,
                        "original_test_name": test_raw,
                        "value": val_float,
                        "unit": unit_str,
                        "reference_range": ref_str,
                        "reference_low": ref_low,
                        "reference_high": ref_high,
                        "explicit_flag": explicit_flag,
                        "calculated_status": calculated_status,
                        "conflict": conflict,
                        "conflict_detail": conflict_detail,
                        "status": final_status,
                        "interpretation": final_status,
                        "domain": test_def["domain"],
                        "source_report": "uploaded_report",
                        "raw_line": " | ".join(raw_parts).strip(),
                    })
                    break

    # 2. Text line-by-line fallback with tightly bounded candidate blocks
    if text and text.strip():
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for idx, line in enumerate(lines):
            for test_def in TARGET_TESTS:
                canonical = test_def["canonical"]
                if canonical in seen_tests:
                    continue

                matched = False
                matched_syn = None
                for syn in test_def["synonyms"]:
                    m_syn = re.search(r"\b" + syn + r"\b", line, re.IGNORECASE)
                    if m_syn:
                        matched = True
                        matched_syn = m_syn.group(0)
                        break

                if not matched:
                    continue

                # Build candidate text block strictly bounded to this test
                block = [line]

                # Check if current line already contains a value
                line_has_val = bool(
                    re.search(r"[:=\t|]\s*[\d<>]", line)
                    or re.search(r"[\d,]+(?:\.\d+)?\s*" + UNIT_REGEX, line, re.IGNORECASE)
                )

                # Look forward up to 4 lines only if the current line does NOT already form a complete test line
                for forward in range(idx + 1, min(idx + 5, len(lines))):
                    fline = lines[forward]
                    # Stop if next line is another test or section header
                    if _is_new_test_or_section(fline):
                        break
                    # If current line already had value, range, and flag, do NOT greedily swallow next line
                    if line_has_val and RANGE_REGEX.search(line) and re.search(r"\b(HIGH|LOW|NORMAL)\b", line, re.IGNORECASE):
                        break
                    # If fline looks like an unrelated line when current line already had value and range
                    if line_has_val and RANGE_REGEX.search(line) and not re.search(r"^(?:flag|status|remark|interpretation|result)\b", fline, re.IGNORECASE) and fline.upper() not in {"HIGH", "LOW", "NORMAL", "ABNORMAL"}:
                        break
                    block.append(fline)
                    # If this line was an explicit flag or completed the record, stop extending
                    if re.search(r"^(?:flag|status|remark|interpretation)\b", fline, re.IGNORECASE) or fline.upper() in {"HIGH", "LOW", "NORMAL", "ABNORMAL"}:
                        break

                combined = " ".join(block)

                # 1. Look for numeric value
                val_float = None
                val_line = None

                # First check: value immediately preceding a unit on the same line
                for b_line in block:
                    m_val_unit = re.search(r"([\d,]+(?:\.\d+)?)\s*" + UNIT_REGEX, b_line, re.IGNORECASE)
                    if m_val_unit:
                        try:
                            vf = float(m_val_unit.group(1).replace(",", ""))
                            if vf < 1900 or vf > 2100:
                                val_float = vf
                                val_line = b_line
                                break
                        except ValueError:
                            pass

                # Second check: number after colon, equal sign, or tab
                if val_float is None:
                    for b_line in block:
                        m_sep = re.search(r"[:=\t|]\s*([\d,]+(?:\.\d+)?)", b_line)
                        if m_sep:
                            try:
                                vf = float(m_sep.group(1).replace(",", ""))
                                if vf < 1900 or vf > 2100:
                                    val_float = vf
                                    val_line = b_line
                                    break
                            except ValueError:
                                pass

                # Third check: first number in combined text avoiding year
                if val_float is None:
                    num_matches = list(re.finditer(r"\b([\d,]+(?:\.\d+)?)\b", combined))
                    for nm in num_matches:
                        num_str = nm.group(1).replace(",", "")
                        try:
                            vf = float(num_str)
                            if vf < 1900 or vf > 2100:
                                val_float = vf
                                break
                        except ValueError:
                            continue

                if val_float is None:
                    continue

                # 2. Extract unit (preferring the line where the value was found)
                unit_str = None
                if val_line:
                    um = re.search(UNIT_REGEX, val_line, re.IGNORECASE)
                    if um:
                        unit_str = um.group(1).strip()
                if not unit_str:
                    um = re.search(UNIT_REGEX, combined, re.IGNORECASE)
                    if um:
                        unit_str = um.group(1).strip()

                # 3. Extract reference range
                ref_low = None
                ref_high = None
                ref_str = None

                range_m = None
                for b_line in block:
                    if re.search(r"\b(?:ref|range|interval|normal)\b", b_line, re.IGNORECASE):
                        rm = RANGE_REGEX.search(b_line)
                        if rm:
                            range_m = rm
                            break
                if not range_m:
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

                # 4. Explicit Flag & Interpretation
                calculated_status = _evaluate_range(val_float, ref_low, ref_high)
                explicit_flag = None

                # Look for explicit flag labeled: "flag: HIGH", "status: LOW", "interpretation: NORMAL"
                flag_hdr_m = re.search(
                    r"\b(?:flag|status|remark|interpretation)\s*[:\s]\s*(\*?\s*(?:HIGH|LOW|NORMAL|ABNORMAL)\*?)\b",
                    combined,
                    re.IGNORECASE,
                )
                if flag_hdr_m:
                    f_val = flag_hdr_m.group(1).replace("*", "").strip().upper()
                    if f_val in {"HIGH", "LOW", "NORMAL", "ABNORMAL"}:
                        explicit_flag = f_val

                if not explicit_flag:
                    # Look for flag immediately following reference range or at end of line: e.g. ", LOW" or ", NORMAL" or ") HIGH"
                    for b_line in block:
                        eol_m = re.search(r"[,|\t\)]\s*(HIGH|LOW|NORMAL|ABNORMAL)\*?\s*$", b_line, re.IGNORECASE)
                        if eol_m:
                            explicit_flag = eol_m.group(1).upper()
                            break
                        if b_line.strip().upper() in {"HIGH", "LOW", "NORMAL", "ABNORMAL"}:
                            explicit_flag = b_line.strip().upper()
                            break

                if not explicit_flag:
                    comb_eol_m = re.search(r"[,|\t\)]\s*(HIGH|LOW|NORMAL|ABNORMAL)\*?\s*$", combined.strip(), re.IGNORECASE)
                    if comb_eol_m:
                        explicit_flag = comb_eol_m.group(1).upper()

                if not explicit_flag:
                    # Look for standalone HIGH or LOW keyword (avoiding HPF, risk, grade, etc.)
                    high_m = re.search(r"\b(HIGH\*?)\b(?!\s*(?:power|risk|grade|density))", combined, re.IGNORECASE)
                    low_m = re.search(r"\b(LOW\*?)\b(?!\s*(?:power|risk|grade|density))", combined, re.IGNORECASE)
                    if high_m and not low_m:
                        explicit_flag = "HIGH"
                    elif low_m and not high_m:
                        explicit_flag = "LOW"

                conflict = False
                conflict_detail = None
                if explicit_flag and calculated_status != "UNKNOWN":
                    if explicit_flag == "NORMAL" and calculated_status in {"HIGH", "LOW"}:
                        conflict = True
                    elif explicit_flag in {"HIGH", "LOW"} and calculated_status == "NORMAL":
                        conflict = True
                    elif explicit_flag == "HIGH" and calculated_status == "LOW":
                        conflict = True
                    elif explicit_flag == "LOW" and calculated_status == "HIGH":
                        conflict = True

                if conflict:
                    conflict_detail = f"Explicit flag '{explicit_flag}' conflicts with calculated status '{calculated_status}'"
                    final_status = explicit_flag
                elif explicit_flag:
                    final_status = explicit_flag
                else:
                    final_status = calculated_status

                seen_tests.add(canonical)
                findings.append({
                    "test_name": canonical,
                    "original_test_name": matched_syn or canonical,
                    "value": val_float,
                    "unit": unit_str,
                    "reference_range": ref_str,
                    "reference_low": ref_low,
                    "reference_high": ref_high,
                    "explicit_flag": explicit_flag,
                    "calculated_status": calculated_status,
                    "conflict": conflict,
                    "conflict_detail": conflict_detail,
                    "status": final_status,
                    "interpretation": final_status,
                    "domain": test_def["domain"],
                    "source_report": "uploaded_report",
                    "raw_line": line if len(line) < 140 else line[:137] + "...",
                })
                break

    # 3. Qualitative and narrative clinical findings extraction
    qualitative_findings: list[dict[str, Any]] = []
    narrative_findings: list[dict[str, Any]] = []

    is_synthetic_doc = is_synthetic_text(text)

    if text and text.strip():
        # Remove lines that contain synthetic demo markers, test instructions, or placeholders
        valid_lines = [line for line in text.splitlines() if not is_synthetic_text(line)]
        cleaned_text = "\n".join(valid_lines)
        lower_text = cleaned_text.lower()

        # Section extraction (Clinical History, Examination, Impression, Findings)
        section_pattern = re.compile(
            r"(?:^|\n)\s*(Clinical History|History|Clinical Indication|Indication|Chief Complaint|Physical Examination|Clinical Examination|Examination|Impression|Conclusion|Diagnosis|Radiology Findings|Findings)\s*:\s*(.*?)(?=(?:\n\s*(?:Clinical History|History|Clinical Indication|Indication|Chief Complaint|Physical Examination|Clinical Examination|Examination|Impression|Conclusion|Diagnosis|Radiology Findings|Findings)\s*:)|$)",
            re.IGNORECASE | re.DOTALL,
        )
        for smatch in section_pattern.finditer(cleaned_text):
            sec_name = smatch.group(1).strip()
            sec_content = smatch.group(2).strip()
            if sec_content and not is_synthetic_text(sec_content):
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
        "is_synthetic": is_synthetic_doc,
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
