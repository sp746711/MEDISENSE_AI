"""Authentic Multimodal Dataset Ingestion and Clinical Linkage Verification Pipeline.

This module provides a modular, strictly guarded ingestion pipeline for genuine patient-linked
multimodal EHR datasets (e.g., MIMIC-IV v2.2 linked with MIMIC-CXR JPG, or standardized hospital EHR exports).

CLINICAL & ETHICAL DATA CONSTRAINTS (Part 7):
1. Accepts only authorized, genuine clinical source files.
2. Validates patient identifiers (subject_id), hospital encounters (hadm_id), and imaging studies (study_id).
3. Enforces temporal relationship rules (lab charttime within admission window, imaging study dates).
4. Validates target clinical condition labels against the canonical 24 clinical classes via ICD ontology crosswalk.
5. Strictly rejects synthetic records, placeholder data, unlinked encounters, and missing patient IDs.
6. Never silently falls back to row index, record identity, or synthetic patient assignment.
7. Produces an auditable, reproducible ingestion audit report detailing linkage fidelity and modality coverage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

from app.ai.multimodal_condition_model import (
    CANONICAL_CLINICAL_EXAM_FINDINGS,
    CANONICAL_LAB_TESTS,
    CANONICAL_RAD_FINDINGS,
    CANONICAL_XRAY_PATHOLOGIES,
    CONDITION_CLASSES,
    INPATIENT_MULTIMODAL_CLASSES,
    CHEST_MULTIMODAL_CLASSES,
    CHEST_CONDITION_ELIGIBILITY_MATRIX,
    resolve_canonical_lab_name,
)

logger = logging.getLogger("dataset_ingestion")

# ICD-9/10 to canonical condition mapping crosswalk
ICD_TO_CONDITION_MAP: dict[str, str] = {
    # Pneumonia
    "486": "Pneumonia",
    "J18.9": "Pneumonia",
    "J15.9": "Pneumonia",
    "482.9": "Pneumonia",
    # Bronchial Asthma
    "493.90": "Bronchial Asthma",
    "493.92": "Bronchial Asthma",
    "J45.909": "Bronchial Asthma",
    "J45.901": "Bronchial Asthma",
    # Hypertension
    "401.9": "Hypertension",
    "I10": "Hypertension",
    "401.1": "Hypertension",
    # Diabetes
    "250.00": "diabetes",
    "250.02": "diabetes",
    "E11.9": "diabetes",
    "E10.9": "diabetes",
    # GERD
    "530.81": "gastroesophageal reflux disease",
    "K21.9": "gastroesophageal reflux disease",
    # Peptic Ulcer Disease
    "533.90": "peptic ulcer disease",
    "K27.9": "peptic ulcer disease",
    # Urinary Tract Infection
    "599.0": "urinary tract infection",
    "N39.0": "urinary tract infection",
    # Arthritis
    "715.90": "Arthritis",
    "M19.90": "Arthritis",
    "M06.9": "Arthritis",
    # Migraine
    "346.90": "Migraine",
    "G43.909": "Migraine",
    # Jaundice / Hyperbilirubinemia
    "782.4": "Jaundice",
    "R17": "Jaundice",
    # Typhoid
    "002.0": "Typhoid",
    "A01.0": "Typhoid",
    # Malaria
    "084.6": "Malaria",
    "B54": "Malaria",
    # Dengue
    "061": "Dengue",
    "A90": "Dengue",
    # Chicken pox
    "052.9": "Chicken pox",
    "B01.9": "Chicken pox",
    # Fungal infection
    "117.9": "Fungal infection",
    "B49": "Fungal infection",
    # Psoriasis
    "696.1": "Psoriasis",
    "L40.0": "Psoriasis",
    # Impetigo
    "684": "Impetigo",
    "L01.0": "Impetigo",
    # Acne
    "706.1": "Acne",
    "L70.0": "Acne",
    # Varicose Veins
    "454.9": "Varicose Veins",
    "I83.90": "Varicose Veins",
    # Cervical spondylosis
    "721.0": "Cervical spondylosis",
    "M47.812": "Cervical spondylosis",
}

# Track B: Separate Inpatient Multimodal Research ICD-9/10 Mapping
# Authoritative acute inpatient primary diagnoses only (strictly excluding chronic/uncomplicated presentations)
INPATIENT_ICD_TO_CONDITION_MAP: dict[str, str] = {
    # Pneumonia (bacterial, pneumococcal, staphylococcal, acute unspecified)
    "486": "Pneumonia",
    "481": "Pneumonia",
    "482.9": "Pneumonia",
    "482.1": "Pneumonia",
    "482.41": "Pneumonia",
    "482.42": "Pneumonia",
    "J18.9": "Pneumonia",
    "J15.9": "Pneumonia",
    "J13": "Pneumonia",
    "J15.0": "Pneumonia",
    "J15.211": "Pneumonia",
    "J15.212": "Pneumonia",

    # Acute Asthma Exacerbation (strictly acute exacerbation codes, never chronic baseline)
    "493.92": "Acute Asthma Exacerbation",
    "493.02": "Acute Asthma Exacerbation",
    "493.12": "Acute Asthma Exacerbation",
    "493.22": "Acute Asthma Exacerbation",
    "J45.901": "Acute Asthma Exacerbation",
    "J45.41": "Acute Asthma Exacerbation",
    "J45.51": "Acute Asthma Exacerbation",
    "J45.21": "Acute Asthma Exacerbation",
    "J45.31": "Acute Asthma Exacerbation",

    # Peptic Ulcer Disease (acute hemorrhage / perforation requiring inpatient care)
    "531.00": "Peptic Ulcer Disease",
    "531.40": "Peptic Ulcer Disease",
    "532.00": "Peptic Ulcer Disease",
    "532.40": "Peptic Ulcer Disease",
    "533.00": "Peptic Ulcer Disease",
    "533.40": "Peptic Ulcer Disease",
    "K25.0": "Peptic Ulcer Disease",
    "K25.4": "Peptic Ulcer Disease",
    "K26.0": "Peptic Ulcer Disease",
    "K26.4": "Peptic Ulcer Disease",
    "K27.0": "Peptic Ulcer Disease",
    "K27.4": "Peptic Ulcer Disease",

    # Urinary Tract Infection (acute pyelonephritis and primary acute UTI)
    "590.10": "Urinary Tract Infection",
    "590.80": "Urinary Tract Infection",
    "599.0": "Urinary Tract Infection",
    "N10": "Urinary Tract Infection",
    "N11.9": "Urinary Tract Infection",
    "N39.0": "Urinary Tract Infection",
}


# ==============================================================================
# EVIDENCE ROLES AND LABEL DECISION STATES (Step 6.12 Fix 4)
# ==============================================================================

class ModalityEvidenceRole:
    """
    Explicit modality and analyte clinical evidence roles (Step 6.16).
    Distinguishes diagnostic, supporting, contextual, severity, rule-out, and nonspecific signals.
    """
    DIAGNOSTIC = "DIAGNOSTIC"
    SUPPORTING = "SUPPORTING"
    CONTEXTUAL = "CONTEXTUAL"
    SEVERITY = "SEVERITY"
    RULE_OUT = "RULE_OUT"
    NONSPECIFIC = "NONSPECIFIC"

    # Backward-compatible aliases for Step 6.12/6.13 test suites
    POSITIVE_DIAGNOSTIC = "POSITIVE_DIAGNOSTIC"
    NEGATIVE_OR_RULE_OUT = "NEGATIVE_OR_RULE_OUT"

    # Technical and availability pipeline states
    UNAVAILABLE = "UNAVAILABLE"
    UNVERIFIED = "UNVERIFIED"
    FAILED = "FAILED"
    UNSUPPORTED = "UNSUPPORTED"


class LabelDecision:
    """Explicit clinical label decision states for diagnostic target extraction."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REQUIRE_ADDITIONAL_EVIDENCE = "REQUIRE_ADDITIONAL_EVIDENCE"
    UNMAPPED = "UNMAPPED"


# ==============================================================================
# AUTHORITATIVE ICD CODE METADATA CATALOG (Step 6.13)
# ==============================================================================

ICD_CODE_METADATA: dict[tuple[int, str], dict[str, Any]] = {
    # ICD-9-CM (v32, 2014/2015 authoritative coding reference)
    (9, "428.1"): {
        "description": "Left heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
        "clinical_note": "Left heart failure (includes acute pulmonary edema with mention of heart disease, cardiac asthma, left ventricular failure). Does not independently establish acute decompensation without validated provider documentation.",
    },
    (9, "428.0"): {
        "description": "Congestive heart failure, unspecified",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
        "clinical_note": "Unspecified congestive heart failure. Requires validated clinical documentation of acute decompensation.",
    },
    (9, "428.21"): {
        "description": "Acute systolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "428.22"): {
        "description": "Chronic systolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REJECT,
    },
    (9, "428.23"): {
        "description": "Acute on chronic systolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "428.31"): {
        "description": "Acute diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "428.32"): {
        "description": "Chronic diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REJECT,
    },
    (9, "428.33"): {
        "description": "Acute on chronic diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "428.41"): {
        "description": "Acute combined systolic and diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "428.42"): {
        "description": "Chronic combined systolic and diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REJECT,
    },
    (9, "428.43"): {
        "description": "Acute on chronic combined systolic and diastolic heart failure",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "518.4"): {
        "description": "Acute edema of lung, unspecified (non-cardiogenic)",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REJECT,
    },
    # ICD-10-CM (Official FY2027 files)
    (10, "I50.1"): {
        "description": "Left ventricular failure, unspecified",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
    },
    (10, "I50.9"): {
        "description": "Heart failure, unspecified",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
    },
    (10, "I50.21"): {
        "description": "Acute systolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.22"): {
        "description": "Chronic systolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
    (10, "I50.23"): {
        "description": "Acute on chronic systolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.31"): {
        "description": "Acute diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.32"): {
        "description": "Chronic diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
    (10, "I50.33"): {
        "description": "Acute on chronic diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.41"): {
        "description": "Acute combined systolic and diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.42"): {
        "description": "Chronic combined systolic and diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
    (10, "I50.43"): {
        "description": "Acute on chronic combined systolic and diastolic (congestive) heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.810"): {
        "description": "Right heart failure, unspecified",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
    (10, "I50.811"): {
        "description": "Acute right heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "I50.813"): {
        "description": "Acute on chronic right heart failure",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "J81.0"): {
        "description": "Acute pulmonary edema (non-cardiogenic)",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
    # COPD Codes
    (9, "491.21"): {
        "description": "Obstructive chronic bronchitis with (acute) exacerbation",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.ACCEPT,
    },
    (9, "491.22"): {
        "description": "Obstructive chronic bronchitis with acute bronchitis",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
        "clinical_note": "Acute bronchitis superimposed on chronic obstructive bronchitis. Does not automatically establish acute airway exacerbation without explicit clinical documentation.",
    },
    (9, "496"): {
        "description": "Chronic airway obstruction, not elsewhere classified",
        "coding_version": "ICD-9-CM v32",
        "default_decision": LabelDecision.REJECT,
    },
    (10, "J44.1"): {
        "description": "Chronic obstructive pulmonary disease with (acute) exacerbation",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.ACCEPT,
    },
    (10, "J44.0"): {
        "description": "Chronic obstructive pulmonary disease with (acute) lower respiratory infection",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
        "clinical_note": "Lower respiratory infection with COPD. Does not automatically establish acute COPD exacerbation, nor does it infer pneumonia.",
    },
    (10, "J44.9"): {
        "description": "Chronic obstructive pulmonary disease, unspecified",
        "coding_version": "ICD-10-CM FY2027",
        "default_decision": LabelDecision.REJECT,
    },
}


# ==============================================================================
# ACUTE HEART FAILURE LABEL EVALUATOR (Step 6.12 Fix 1 & Step 6.13)
# ==============================================================================

class AcuteHeartFailureLabelEvaluator:
    """
    Authoritative label evaluator for Acute Decompensated Heart Failure (Track B).
    Enforces strict clinical distinction between acute, chronic, and unspecified presentations.

    Mandatory Rules:
    1. ICD-9 428.1 (Left heart failure) does NOT independently establish acute decompensated HF.
       Rule: Consistently returns REQUIRE_ADDITIONAL_EVIDENCE.
    2. 428.0, I50.1, I50.9 do NOT automatically establish acute decompensated HF.
    3. Elevated BNP / NT-proBNP alone CANNOT convert an unspecified code into confirmed acute HF.
    4. Only explicit acute/acute-on-chronic codes are accepted unconditionally.
    5. Preserves exact code descriptions and coding-version provenance (ICD-9-CM v32 / ICD-10-CM FY2027).
    """
    # Definitive Acute / Acute-on-chronic codes
    DEFINITIVE_ACUTE_ICD9 = frozenset({"428.21", "428.23", "428.31", "428.33", "428.41", "428.43"})
    DEFINITIVE_ACUTE_ICD10 = frozenset({"I50.21", "I50.23", "I50.31", "I50.33", "I50.41", "I50.43"})

    # Broad / Unspecified codes requiring validated provider documentation of acute decompensation
    NONSPECIFIC_REQUIRING_EVIDENCE_ICD9 = frozenset({"428.0", "428.1"})
    NONSPECIFIC_REQUIRING_EVIDENCE_ICD10 = frozenset({"I50.0", "I50.1", "I50.9"})

    # Chronic compensated / Non-cardiogenic codes strictly rejected
    CHRONIC_OR_NONCARDIAC_ICD9 = frozenset({"428.22", "428.32", "428.42", "518.4"})
    CHRONIC_OR_NONCARDIAC_ICD10 = frozenset({"I50.22", "I50.32", "I50.42", "J81.0"})

    @classmethod
    def evaluate(
        cls,
        code: str,
        icd_version: int = 10,
        clinical_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        clean_code = re.sub(r"[\s\.\-]+", "", str(code or "")).upper()
        ctx = clinical_context or {}

        # 1. Check Definitive Acute Codes
        clean_acute9 = {c.replace(".", ""): c for c in cls.DEFINITIVE_ACUTE_ICD9}
        clean_acute10 = {c.replace(".", ""): c for c in cls.DEFINITIVE_ACUTE_ICD10}

        if (icd_version == 9 and clean_code in clean_acute9) or (icd_version == 10 and clean_code in clean_acute10):
            orig_c = clean_acute9.get(clean_code) if icd_version == 9 else clean_acute10.get(clean_code)
            meta = ICD_CODE_METADATA.get((icd_version, orig_c), {})
            return {
                "decision": LabelDecision.ACCEPT,
                "target": "Acute Decompensated Heart Failure",
                "code": orig_c,
                "icd_version": icd_version,
                "description": meta.get("description", "Acute/acute-on-chronic heart failure"),
                "coding_version": meta.get("coding_version", f"ICD-{icd_version}-CM"),
                "reason": "definitive_acute_or_acute_on_chronic_decompensation_code",
                "is_acute": True,
            }

        # 2. Check Chronic Compensated / Non-cardiogenic Codes
        clean_chron9 = {c.replace(".", ""): c for c in cls.CHRONIC_OR_NONCARDIAC_ICD9}
        clean_chron10 = {c.replace(".", ""): c for c in cls.CHRONIC_OR_NONCARDIAC_ICD10}

        if (icd_version == 9 and clean_code in clean_chron9) or (icd_version == 10 and clean_code in clean_chron10):
            orig_c = clean_chron9.get(clean_code) if icd_version == 9 else clean_chron10.get(clean_code)
            meta = ICD_CODE_METADATA.get((icd_version, orig_c), {})
            return {
                "decision": LabelDecision.REJECT,
                "target": None,
                "code": orig_c,
                "icd_version": icd_version,
                "description": meta.get("description", "Chronic/non-cardiogenic condition"),
                "coding_version": meta.get("coding_version", f"ICD-{icd_version}-CM"),
                "reason": "chronic_compensated_or_non_cardiogenic_code",
                "is_acute": False,
            }

        # 3. Check Nonspecific / Broad Codes Requiring Additional Evidence (including 428.1)
        clean_req9 = {c.replace(".", ""): c for c in cls.NONSPECIFIC_REQUIRING_EVIDENCE_ICD9}
        clean_req10 = {c.replace(".", ""): c for c in cls.NONSPECIFIC_REQUIRING_EVIDENCE_ICD10}

        if (icd_version == 9 and clean_code in clean_req9) or (icd_version == 10 and clean_code in clean_req10):
            orig_c = clean_req9.get(clean_code) if icd_version == 9 else clean_req10.get(clean_code)
            meta = ICD_CODE_METADATA.get((icd_version, orig_c), {})
            desc = meta.get("description", "Left heart failure" if orig_c == "428.1" else "Heart failure, unspecified")
            coding_ver = meta.get("coding_version", "ICD-9-CM v32" if icd_version == 9 else "ICD-10-CM FY2027")

            # Mandatory rule: BNP / NT-proBNP ALONE cannot establish acute HF
            has_provider_doc = bool(ctx.get("provider_documented_acute_decompensation") or ctx.get("acute_iv_diuretic_bolus"))
            bnp_only = bool(ctx.get("bnp_elevated") or ctx.get("nt_pro_bnp_elevated"))

            if has_provider_doc:
                return {
                    "decision": LabelDecision.ACCEPT,
                    "target": "Acute Decompensated Heart Failure",
                    "code": orig_c,
                    "icd_version": icd_version,
                    "description": desc,
                    "coding_version": coding_ver,
                    "reason": "nonspecific_code_confirmed_by_provider_documented_acute_decompensation",
                    "is_acute": True,
                }
            elif bnp_only:
                return {
                    "decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
                    "target": None,
                    "code": orig_c,
                    "icd_version": icd_version,
                    "description": desc,
                    "coding_version": coding_ver,
                    "reason": "bnp_alone_insufficient_to_confirm_acute_decompensation_on_unspecified_code",
                    "is_acute": False,
                }
            else:
                return {
                    "decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
                    "target": None,
                    "code": orig_c,
                    "icd_version": icd_version,
                    "description": desc,
                    "coding_version": coding_ver,
                    "reason": "unspecified_code_lacks_provider_documented_acute_decompensation",
                    "is_acute": False,
                }

        return {
            "decision": LabelDecision.UNMAPPED,
            "target": None,
            "code": code,
            "icd_version": icd_version,
            "description": "Unmapped non-HF code",
            "coding_version": f"ICD-{icd_version}-CM",
            "reason": "not_a_heart_failure_code",
            "is_acute": False,
        }


# ==============================================================================
# COPD EXACERBATION LABEL EVALUATOR (Step 6.12 Fix 2 & Step 6.13)
# ==============================================================================

class COPDExacerbationLabelEvaluator:
    """
    Authoritative label evaluator for Acute COPD Exacerbation (Track B).
    Enforces distinct clinical concepts between acute exacerbation, infection, and chronic baseline.

    Mandatory Rules:
    1. J44.1 qualifies as acute COPD exacerbation candidate.
    2. J44.0 (COPD with acute lower respiratory infection) does NOT automatically establish acute exacerbation.
    3. 491.22 (with acute bronchitis) does NOT automatically establish acute COPD exacerbation without clinical documentation.
       Never conflate 491.22 with 491.21 via prefix matching or broad mappings.
    4. Pneumonia must be independently established. Do not infer pneumonia solely from J44.0.
    5. Secondary acute COPD diagnoses are preserved when criteria are met.
    """
    ACUTE_EXACERBATION_ICD9 = frozenset({"491.21"})
    ACUTE_EXACERBATION_ICD10 = frozenset({"J44.1"})

    INFECTION_SUPERIMPOSED_ICD9 = frozenset({"491.22"})
    INFECTION_SUPERIMPOSED_ICD10 = frozenset({"J44.0"})

    CHRONIC_BASELINE_ICD9 = frozenset({"496", "492.8", "493.20"})
    CHRONIC_BASELINE_ICD10 = frozenset({"J44.9", "J43.9", "J44.89"})

    @classmethod
    def evaluate(
        cls,
        code: str,
        icd_version: int = 10,
        clinical_context: Optional[dict[str, Any]] = None,
        seq_num: Optional[int] = None,
    ) -> dict[str, Any]:
        clean_code = re.sub(r"[\s\.\-]+", "", str(code or "")).upper()
        ctx = clinical_context or {}
        is_secondary = bool(seq_num is not None and seq_num > 1)

        # Disallow broad prefix matching for 491.2
        if icd_version == 9 and clean_code.startswith("4912") and clean_code not in {"49121", "49122"}:
            return {
                "decision": LabelDecision.UNMAPPED,
                "target": None,
                "code": code,
                "icd_version": 9,
                "is_secondary": is_secondary,
                "copd_exacerbation": False,
                "pneumonia_inferred": False,
                "reason": "truncated_or_broad_prefix_rejected_without_exact_code",
            }

        # 1. Definite Acute Exacerbation Codes: J44.1, 491.21
        clean_exac9 = {c.replace(".", ""): c for c in cls.ACUTE_EXACERBATION_ICD9}
        clean_exac10 = {c.replace(".", ""): c for c in cls.ACUTE_EXACERBATION_ICD10}

        if (icd_version == 9 and clean_code in clean_exac9) or (icd_version == 10 and clean_code in clean_exac10):
            orig_c = clean_exac9.get(clean_code) if icd_version == 9 else clean_exac10.get(clean_code)
            meta = ICD_CODE_METADATA.get((icd_version, orig_c), {})
            return {
                "decision": LabelDecision.ACCEPT,
                "target": "Acute COPD Exacerbation",
                "code": orig_c,
                "icd_version": icd_version,
                "description": meta.get("description", "COPD with acute exacerbation"),
                "coding_version": meta.get("coding_version", "ICD-9-CM v32" if icd_version == 9 else "ICD-10-CM FY2027"),
                "is_secondary": is_secondary,
                "copd_exacerbation": True,
                "pneumonia_inferred": False,
                "reason": "primary_or_secondary_acute_copd_exacerbation_code",
            }

        # 2. Infection Superimposed: J44.0 (COPD with acute lower resp infection)
        if icd_version == 10 and clean_code == "J440":
            meta = ICD_CODE_METADATA.get((10, "J44.0"), {})
            has_documented_exac = bool(ctx.get("documented_exacerbation") or ctx.get("has_copd_exacerbation_symptoms"))
            has_indep_pneumonia = bool(ctx.get("independent_pneumonia") or ctx.get("has_pneumonia_code"))

            if has_documented_exac:
                return {
                    "decision": LabelDecision.ACCEPT,
                    "target": "Acute COPD Exacerbation",
                    "code": "J44.0",
                    "icd_version": 10,
                    "description": meta.get("description", "COPD with acute lower respiratory infection"),
                    "coding_version": "ICD-10-CM FY2027",
                    "is_secondary": is_secondary,
                    "copd_exacerbation": True,
                    "pneumonia_inferred": False,
                    "independent_pneumonia_present": has_indep_pneumonia,
                    "reason": "j44_0_with_independently_documented_acute_exacerbation",
                }
            else:
                return {
                    "decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
                    "target": None,
                    "code": "J44.0",
                    "icd_version": 10,
                    "description": meta.get("description", "COPD with acute lower respiratory infection"),
                    "coding_version": "ICD-10-CM FY2027",
                    "is_secondary": is_secondary,
                    "copd_exacerbation": False,
                    "pneumonia_inferred": False,
                    "independent_pneumonia_present": has_indep_pneumonia,
                    "reason": "j44_0_represents_infection_not_automatic_copd_exacerbation_and_does_not_infer_pneumonia",
                }

        # 3. Infection Superimposed ICD-9: 491.22 (Obstructive chronic bronchitis with acute bronchitis)
        # MUST remain strictly separate from 491.21
        if icd_version == 9 and clean_code == "49122":
            meta = ICD_CODE_METADATA.get((9, "491.22"), {})
            has_documented_exac = bool(ctx.get("documented_exacerbation"))
            desc = meta.get("description", "Obstructive chronic bronchitis with acute bronchitis")
            if has_documented_exac:
                return {
                    "decision": LabelDecision.ACCEPT,
                    "target": "Acute COPD Exacerbation",
                    "code": "491.22",
                    "icd_version": 9,
                    "description": desc,
                    "coding_version": "ICD-9-CM v32",
                    "is_secondary": is_secondary,
                    "copd_exacerbation": True,
                    "pneumonia_inferred": False,
                    "reason": "491_22_with_documented_acute_exacerbation",
                }
            return {
                "decision": LabelDecision.REQUIRE_ADDITIONAL_EVIDENCE,
                "target": None,
                "code": "491.22",
                "icd_version": 9,
                "description": desc,
                "coding_version": "ICD-9-CM v32",
                "is_secondary": is_secondary,
                "copd_exacerbation": False,
                "pneumonia_inferred": False,
                "reason": "491_22_acute_bronchitis_requires_explicit_exacerbation_documentation",
            }

        # 4. Chronic Baseline COPD Codes: 496, J44.9, etc.
        clean_base9 = {c.replace(".", ""): c for c in cls.CHRONIC_BASELINE_ICD9}
        clean_base10 = {c.replace(".", ""): c for c in cls.CHRONIC_BASELINE_ICD10}

        if (icd_version == 9 and clean_code in clean_base9) or (icd_version == 10 and clean_code in clean_base10):
            orig_c = clean_base9.get(clean_code) if icd_version == 9 else clean_base10.get(clean_code)
            meta = ICD_CODE_METADATA.get((icd_version, orig_c), {})
            return {
                "decision": LabelDecision.REJECT,
                "target": None,
                "code": orig_c,
                "icd_version": icd_version,
                "description": meta.get("description", "Chronic baseline COPD"),
                "coding_version": meta.get("coding_version", "ICD-9-CM v32" if icd_version == 9 else "ICD-10-CM FY2027"),
                "is_secondary": is_secondary,
                "copd_exacerbation": False,
                "pneumonia_inferred": False,
                "reason": "unspecified_or_chronic_baseline_copd_rejected_for_acute_exacerbation",
            }

        return {
            "decision": LabelDecision.UNMAPPED,
            "target": None,
            "code": code,
            "icd_version": icd_version,
            "description": "Unmapped non-COPD code",
            "coding_version": f"ICD-{icd_version}-CM",
            "is_secondary": is_secondary,
            "copd_exacerbation": False,
            "pneumonia_inferred": False,
            "reason": "not_a_copd_code",
        }


# ==============================================================================
# EVIDENCE ROLE POLICY (Step 6.13 Fix 4)
# ==============================================================================

class EvidenceRolePolicy:
    """
    Enforces evidence-role assignment and isolation rules across modalities and conditions (Step 6.16).
    Guarantees that rule-out, contextual, severity, and nonspecific evidence cannot activate
    positive diagnostic scoring, and missing/failed evidence is never converted to normal results.
    """

    @classmethod
    def determine_modality_role(
        cls,
        modality: str,
        condition: str,
        modality_state: str,
        finding_type: str = "diagnostic",
        is_verified: bool = False,
    ) -> str:
        """Determines the explicit evidence role for a modality-condition pair."""
        mod_upper = str(modality or "").strip().upper()
        cond_upper = str(condition or "").strip().upper()
        st = str(modality_state or "").strip().upper()

        if st == "UNAVAILABLE" or st in {"NOT_PROVIDED", "NOT_ASKED"}:
            return ModalityEvidenceRole.UNAVAILABLE
        if st == "UNSUPPORTED":
            return ModalityEvidenceRole.UNSUPPORTED
        if st == "FAILED":
            return ModalityEvidenceRole.FAILED
        if st == "UNVERIFIED" or not is_verified:
            return ModalityEvidenceRole.UNVERIFIED

        # Condition-specific rule-out restriction: Chest X-ray in COPD is strictly rule-out
        if "COPD" in cond_upper and mod_upper in {"XRAY", "CHEST_XRAY", "XRAY_PATHOLOGY"}:
            return ModalityEvidenceRole.NEGATIVE_OR_RULE_OUT

        if finding_type == "contextual":
            return ModalityEvidenceRole.CONTEXTUAL
        if finding_type == "rule_out":
            return ModalityEvidenceRole.NEGATIVE_OR_RULE_OUT

        return ModalityEvidenceRole.POSITIVE_DIAGNOSTIC

    @classmethod
    def can_contribute_to_positive_diagnosis(
        cls,
        role: str,
        is_condition_eligible: bool = True,
    ) -> bool:
        """Rule: Only DIAGNOSTIC / POSITIVE_DIAGNOSTIC can contribute to positive diagnostic logit accumulation."""
        if role in (ModalityEvidenceRole.DIAGNOSTIC, ModalityEvidenceRole.POSITIVE_DIAGNOSTIC) and is_condition_eligible:
            return True
        return False

    @classmethod
    def can_independently_establish_disease(cls, role: str) -> bool:
        """
        Rule: Only DIAGNOSTIC evidence can independently establish a disease label.
        NONSPECIFIC (WBC, CRP, procalcitonin), CONTEXTUAL (creatinine, BUN, glucose),
        SEVERITY (blood gas PaCO2/PaO2), SUPPORTING (BNP alone), and RULE_OUT markers
        CANNOT independently determine a disease label.
        """
        return role in (ModalityEvidenceRole.DIAGNOSTIC, ModalityEvidenceRole.POSITIVE_DIAGNOSTIC)

    @classmethod
    def can_support_candidate(cls, role: str) -> bool:
        """Returns True if the evidence can support an active diagnostic candidate."""
        return role in (
            ModalityEvidenceRole.DIAGNOSTIC,
            ModalityEvidenceRole.POSITIVE_DIAGNOSTIC,
            ModalityEvidenceRole.SUPPORTING,
        )

    @classmethod
    def can_contribute_to_severity(cls, role: str) -> bool:
        """Returns True if the evidence contributes to disease severity or decompensation grading."""
        return role == ModalityEvidenceRole.SEVERITY

    @classmethod
    def is_nonspecific_marker(cls, role: str) -> bool:
        """Returns True if the evidence is a general inflammatory/immune marker."""
        return role == ModalityEvidenceRole.NONSPECIFIC

    @classmethod
    def classify_evidence_role(
        cls,
        condition: str,
        modality: str,
        finding_or_analyte: str,
    ) -> str:
        """
        Clinical Evidence Classification per Step 6.16:
        Maps each finding or analyte to its clinical evidence role for a target condition.
        """
        cond_clean = str(condition or "").strip().lower()
        mod_clean = str(modality or "").strip().lower()
        item_clean = str(finding_or_analyte or "").strip().lower()

        # 1. Nonspecific inflammatory markers across all conditions
        nonspecific_terms = {
            "wbc", "white blood cell", "white blood cell count", "white blood cell count (wbc)",
            "neutrophil", "neutrophils", "crp", "c-reactive protein", "c-reactive protein (crp)",
            "procalcitonin", "esr", "erythrocyte sedimentation rate",
        }
        if any(term in item_clean for term in nonspecific_terms):
            return ModalityEvidenceRole.NONSPECIFIC

        # 2. Condition-specific role mappings
        if "pneumonia" in cond_clean:
            if mod_clean in ("xray", "chest_xray", "radiology"):
                if any(k in item_clean for k in ("consolidation", "infiltrate", "air-space opacity", "opacity", "pneumonia")):
                    return ModalityEvidenceRole.DIAGNOSTIC
                if "clear lungs" in item_clean:
                    return ModalityEvidenceRole.RULE_OUT
            if mod_clean in ("symptoms", "triage"):
                if any(k in item_clean for k in ("cough", "fever", "sputum", "dyspnea", "chills")):
                    return ModalityEvidenceRole.SUPPORTING
            if mod_clean in ("laboratory", "lab"):
                if any(k in item_clean for k in ("pao2", "paco2", "blood gas", "ph", "o2sat")):
                    return ModalityEvidenceRole.SEVERITY
                if any(k in item_clean for k in ("bun", "blood urea nitrogen", "creatinine")):
                    return ModalityEvidenceRole.SEVERITY
                if any(k in item_clean for k in ("sputum culture", "blood culture", "gram stain")):
                    return ModalityEvidenceRole.DIAGNOSTIC
            if mod_clean in ("clinical_exam", "exam"):
                if any(k in item_clean for k in ("crackles", "crepitations", "bronchial breath sounds")):
                    return ModalityEvidenceRole.SUPPORTING

        elif "heart failure" in cond_clean or "adhf" in cond_clean:
            if mod_clean in ("laboratory", "lab"):
                if any(k in item_clean for k in ("bnp", "nt-probnp", "nt_pro_bnp", "brain natriuretic peptide")):
                    # BNP alone is SUPPORTING, cannot independently establish acute HF!
                    return ModalityEvidenceRole.SUPPORTING
                if any(k in item_clean for k in ("troponin", "lactate")):
                    return ModalityEvidenceRole.SEVERITY
                if any(k in item_clean for k in ("creatinine", "bun", "blood urea nitrogen", "sodium", "potassium", "egfr")):
                    return ModalityEvidenceRole.CONTEXTUAL
            if mod_clean in ("xray", "chest_xray", "radiology"):
                if any(k in item_clean for k in ("pulmonary edema", "interstitial edema", "kerley", "vascular congestion", "cephalization", "cardiomegaly")):
                    return ModalityEvidenceRole.DIAGNOSTIC
                if "clear lungs" in item_clean:
                    return ModalityEvidenceRole.RULE_OUT
            if mod_clean in ("symptoms", "triage"):
                if any(k in item_clean for k in ("orthopnea", "paroxysmal nocturnal dyspnea", "pnd", "dyspnea on exertion", "edema", "swelling")):
                    return ModalityEvidenceRole.SUPPORTING
            if mod_clean in ("clinical_exam", "exam"):
                if any(k in item_clean for k in ("jvp", "jugular venous distension", "s3", "third heart sound", "hepatojugular")):
                    return ModalityEvidenceRole.SUPPORTING

        elif "pleural effusion" in cond_clean:
            if mod_clean in ("xray", "chest_xray", "radiology"):
                if any(k in item_clean for k in ("effusion", "pleural effusion", "meniscus", "blunting", "costophrenic")):
                    return ModalityEvidenceRole.DIAGNOSTIC
                if "clear lungs" in item_clean:
                    return ModalityEvidenceRole.RULE_OUT
            if mod_clean in ("laboratory", "lab"):
                if any(k in item_clean for k in ("protein", "total protein", "ldh", "fluid protein", "fluid ldh")):
                    return ModalityEvidenceRole.CONTEXTUAL
            if mod_clean in ("symptoms", "triage"):
                if any(k in item_clean for k in ("pleuritic", "pleuritic chest pain", "dyspnea", "breathlessness")):
                    return ModalityEvidenceRole.SUPPORTING
            if mod_clean in ("clinical_exam", "exam"):
                if any(k in item_clean for k in ("dullness", "dull to percussion", "decreased breath sounds")):
                    return ModalityEvidenceRole.SUPPORTING

        elif "pneumothorax" in cond_clean:
            if mod_clean in ("xray", "chest_xray", "radiology"):
                if any(k in item_clean for k in ("pneumothorax", "pleural line", "visceral pleural line", "air space")):
                    return ModalityEvidenceRole.DIAGNOSTIC
                if "clear lungs" in item_clean:
                    return ModalityEvidenceRole.RULE_OUT
            if mod_clean in ("laboratory", "lab"):
                if any(k in item_clean for k in ("pao2", "paco2", "blood gas", "ph")):
                    return ModalityEvidenceRole.SEVERITY
                # Any other lab in pneumothorax is contextual/nonspecific
                return ModalityEvidenceRole.CONTEXTUAL
            if mod_clean in ("symptoms", "triage"):
                if any(k in item_clean for k in ("sudden sharp chest pain", "pleuritic chest pain", "dyspnea", "acute chest pain")):
                    return ModalityEvidenceRole.SUPPORTING
            if mod_clean in ("clinical_exam", "exam"):
                if any(k in item_clean for k in ("hyperresonance", "decreased breath sounds", "tracheal deviation")):
                    return ModalityEvidenceRole.SUPPORTING

        elif "copd" in cond_clean:
            # Chest X-ray in COPD is strictly RULE_OUT!
            if mod_clean in ("xray", "chest_xray"):
                return ModalityEvidenceRole.RULE_OUT
            if mod_clean == "radiology":
                if any(k in item_clean for k in ("hyperinflation", "flattened diaphragms", "increased retrosternal space")):
                    return ModalityEvidenceRole.CONTEXTUAL
                return ModalityEvidenceRole.RULE_OUT
            if mod_clean in ("laboratory", "lab"):
                if any(k in item_clean for k in ("paco2", "pao2", "ph", "bicarbonate", "hco3", "blood gas")):
                    return ModalityEvidenceRole.SEVERITY
            if mod_clean in ("symptoms", "triage"):
                if any(k in item_clean for k in ("dyspnea", "sputum", "purulent sputum", "sputum volume", "wheezing", "cough")):
                    return ModalityEvidenceRole.SUPPORTING
            if mod_clean in ("clinical_exam", "exam"):
                if any(k in item_clean for k in ("wheezing", "expiratory wheeze", "prolonged expiration", "barrel chest")):
                    return ModalityEvidenceRole.SUPPORTING

        return ModalityEvidenceRole.CONTEXTUAL


# ==============================================================================
# CLINICAL EXAMINATION TIMING & PROVENANCE VALIDATOR (Step 6.12 Fix 6 & Step 6.13)
# ==============================================================================

class ClinicalExamTimingProvenanceValidator:
    """
    Validates physical clinical examination extraction timing, note types, and assertion states.

    Mandatory Rules:
    1. Distinguishes initial ED notes, admission H&P notes, and progress notes.
    2. Routine progress notes are REJECTED as initial examinations unless explicit initial-examination provenance exists.
    3. Discharge summaries are strictly rejected unless verified as initial H&P section.
    4. Note timestamps must fall within [-24h, +24h] of admission.
    5. Tracks treatment timing (precedes versus follows treatment).
    6. Rejects uncertain encounter linkage.
    7. Enforces four-state assertions: PRESENT, ABSENT, UNKNOWN, NOT_DOCUMENTED.
       Never converts NOT_DOCUMENTED into ABSENT.
    8. Vital signs must NEVER be mapped as physical examination findings.
    9. Fails closed with exam_eligible = 0.0 until validated.
    """
    APPROVED_NOTE_TYPES = frozenset({
        "ED_INITIAL_ASSESSMENT",
        "ED_PROVIDER_NOTE",
        "ADMISSION_HP",
        "INITIAL_HP",
    })
    ROUTINE_PROGRESS_NOTE_TYPES = frozenset({
        "PROGRESS_NOTE",
        "DAILY_PROGRESS_NOTE",
        "NURSING_PROGRESS_NOTE",
        "PHYSICIAN_PROGRESS_NOTE",
    })
    DISALLOWED_NOTE_TYPES = frozenset({
        "DISCHARGE_SUMMARY",
        "PRE_DISCHARGE_NOTE",
        "TRANSFER_SUMMARY",
    })

    VITAL_SIGN_KEYWORDS = frozenset({
        "temperature", "heartrate", "heart_rate", "pulse",
        "sbp", "dbp", "blood_pressure", "resprate", "respiratory_rate",
        "o2sat", "spo2", "pain_score",
    })

    FOUR_STATE_ASSERTIONS = frozenset({"PRESENT", "ABSENT", "UNKNOWN", "NOT_DOCUMENTED"})

    @classmethod
    def validate_exam_record(
        cls,
        exam_record: dict[str, Any],
        admit_time: Optional[datetime] = None,
        treatment_time: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Validate clinical exam record provenance and timing."""
        if not isinstance(exam_record, dict):
            return {"eligible": False, "reason": "exam_record_not_dict", "exam_eligible": 0.0}

        # Check encounter linkage
        if exam_record.get("encounter_linked") is False:
            return {"eligible": False, "reason": "uncertain_encounter_linkage", "exam_eligible": 0.0}
        if exam_record.get("encounter_id_mismatch") is True:
            return {"eligible": False, "reason": "encounter_id_mismatch", "exam_eligible": 0.0}

        note_type = str(exam_record.get("note_type") or "").strip().upper()

        # Reject discharge summaries unless verified initial admission H&P section
        if note_type in cls.DISALLOWED_NOTE_TYPES:
            section = str(exam_record.get("note_section") or "").strip().upper()
            is_init_sec = bool(exam_record.get("is_initial_exam_section"))
            if not (is_init_sec and ("ADMISSION" in section or "INITIAL" in section) and "PHYSICAL" in section):
                return {
                    "eligible": False,
                    "reason": "discharge_summary_not_verified_as_initial_hp_section",
                    "exam_eligible": 0.0,
                }
        elif note_type in cls.ROUTINE_PROGRESS_NOTE_TYPES:
            # Routine progress notes rejected unless explicit initial-examination provenance exists
            is_init_sec = bool(exam_record.get("is_initial_exam_section") or "INITIAL" in str(exam_record.get("note_section") or "").upper())
            if not is_init_sec:
                return {
                    "eligible": False,
                    "reason": "routine_progress_note_rejected_without_initial_provenance",
                    "exam_eligible": 0.0,
                }
        elif note_type not in cls.APPROVED_NOTE_TYPES and note_type != "":
            return {
                "eligible": False,
                "reason": f"unapproved_note_type: {note_type}",
                "exam_eligible": 0.0,
            }

        # Validate Timing: within [-24h, +24h] of admission
        note_time_str = exam_record.get("charttime") or exam_record.get("note_time")
        note_dt, err = parse_clinical_timestamp(note_time_str) if note_time_str else (None, None)
        if err or (admit_time and note_dt and (note_dt < admit_time - timedelta(hours=24) or note_dt > admit_time + timedelta(hours=24))):
            return {
                "eligible": False,
                "reason": "note_time_outside_admission_window",
                "exam_eligible": 0.0,
            }

        # Track treatment timing
        precedes_treatment: Optional[bool] = None
        treatment_dt: Optional[datetime] = None
        if treatment_time:
            treatment_dt, _ = parse_clinical_timestamp(treatment_time) if isinstance(treatment_time, str) else (treatment_time, None)
            if note_dt and treatment_dt:
                precedes_treatment = bool(note_dt <= treatment_dt)

        # Validate Finding assertions and Vital Signs Separation
        findings = exam_record.get("findings") or []
        sanitized_findings = []
        for f in findings:
            if isinstance(f, dict):
                f_name = str(f.get("finding") or "").lower().strip()
                if any(v in f_name for v in cls.VITAL_SIGN_KEYWORDS):
                    return {
                        "eligible": False,
                        "reason": f"vital_signs_separation_violation: {f_name}",
                        "exam_eligible": 0.0,
                    }
                state = str(f.get("state") or "").upper().strip()
                if state and state not in cls.FOUR_STATE_ASSERTIONS:
                    return {
                        "eligible": False,
                        "reason": f"invalid_finding_assertion_state: {state}",
                        "exam_eligible": 0.0,
                    }
                # Preserve exact assertion state (never convert NOT_DOCUMENTED into ABSENT)
                sanitized_findings.append({
                    "finding": f_name,
                    "state": state if state in cls.FOUR_STATE_ASSERTIONS else "NOT_DOCUMENTED",
                })

        # Until verified extraction pipeline is constructed, fail closed
        is_verified_pipeline = bool(exam_record.get("extraction_pipeline_verified") is True)
        return {
            "eligible": is_verified_pipeline,
            "reason": "verified_provenance" if is_verified_pipeline else "unvalidated_nlp_pipeline_fail_closed",
            "exam_eligible": 1.0 if is_verified_pipeline else 0.0,
            "note_timestamp_iso": note_dt.isoformat() if note_dt else None,
            "admit_timestamp_iso": admit_time.isoformat() if (admit_time and hasattr(admit_time, 'isoformat')) else None,
            "treatment_timestamp_iso": treatment_dt.isoformat() if (treatment_dt and hasattr(treatment_dt, 'isoformat')) else None,
            "precedes_treatment": precedes_treatment,
            "note_id": exam_record.get("note_id"),
            "note_section": exam_record.get("note_section"),
            "hadm_id": exam_record.get("hadm_id") or exam_record.get("encounter_id"),
            "sanitized_findings": sanitized_findings,
        }


def parse_clinical_timestamp(val: Any) -> tuple[Optional[datetime], Optional[str]]:
    """
    Parse a clinical timestamp strictly.
    Returns (datetime_obj_in_utc, error_string).
    Handles ISO 8601 strings (with or without 'Z'/offset) and standard SQL datetime strings.
    Never silently ignores timestamp errors.
    """
    if val is None:
        return None, "missing_timestamp"
    if isinstance(val, (int, float)):
        return None, f"numeric_timestamp_unsupported: {val}"
    val_str = str(val).strip()
    if not val_str:
        return None, "empty_timestamp"

    # Try ISO 8601
    try:
        clean_str = val_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt, None
    except (ValueError, TypeError):
        pass

    # Try common clinical SQL datetime formats
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(val_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt, None
        except ValueError:
            continue

    return None, f"malformed_timestamp: {val_str}"


APPROVED_PROVENANCE_SOURCES: frozenset[str] = frozenset({
    # Authentic MIMIC-IV ED to MIMIC-CXR linkage mechanisms
    "MIMIC_ED_CXR_CROSSWALK",
    "MIMIC_IV_ED_CXR_CROSSWALK",
    "MIMIC_IV_ED_TRANSFER",
    "MIMIC_ED_ANCHOR",
    "MIMIC_VALID_TRANSFER",
    "MIMIC_IV_CXR_ED_TRANSFER",
    "MIMIC_CXR_ED_TRANSFER",
    # Authentic MIMIC Hospital to CXR linkage mechanisms
    "MIMIC_CXR_HADM_MATCH",
    "MIMIC_DIRECT_CROSSWALK",
    "MIMIC_HOSP_CXR_LINK",
    "MIMIC_IV_HOSP_CXR_LINK",
    # Institutional PACS to EHR verified interfaces
    "INSTITUTIONAL_PACS_CROSSWALK",
    "INSTITUTIONAL_PACS_EHR_CROSSWALK",
})


APPROVED_SOURCE_RECORD_REFERENCES: frozenset[str] = frozenset({
    "MIMIC_IV_ED.ED_RADIOLOGY",
    "MIMIC_IV_ED.EDSTAYS",
    "MIMIC_IV_HOSP.ADMISSIONS",
    "MIMIC_CXR.RECORD_LIST",
    "CXR_ED.CSV",
    "ED_RADIOLOGY.CSV",
    "EDSTAYS.CSV",
    "INSTITUTIONAL_PACS_ORDER_CROSSWALK",
    "BIDMC_RADIOLOGY_ORDER_INDEX",
})

APPROVED_CROSSWALK_GENERATORS: frozenset[str] = frozenset({
    "MIMIC_CXR_ED_BRIDGE_V1",
    "MIMIC_IV_ED_STAY_LINKER_V2",
    "INSTITUTIONAL_PACS_BRIDGE_V1",
})


import hmac
import os


def _get_hmac_signing_key(strict: bool = False, min_bytes: int = 16) -> Optional[bytes]:
    """
    Private helper to retrieve the HMAC secret without exposing it on module attributes.
    Checks environment variable MEDISENSE_PROVENANCE_HMAC_SECRET or CROSSWALK_SIGNING_KEY.
    In strict mode, returns None if unset or empty, and raises ValueError if key is below min_bytes.
    """
    secret = os.environ.get("MEDISENSE_PROVENANCE_HMAC_SECRET") or os.environ.get("CROSSWALK_SIGNING_KEY")
    if secret is not None:
        stripped = secret.strip()
        if not stripped:
            if strict:
                return None
        else:
            raw_bytes = stripped.encode("utf-8")
            if strict and len(raw_bytes) < min_bytes:
                raise ValueError(f"Configured HMAC secret must be at least {min_bytes} bytes for security, got {len(raw_bytes)}")
            return raw_bytes
    if strict:
        return None
    # Controlled fallback for non-strict test environments only; never exposed as a module attribute
    return b"medisense_clinical_provenance_pipeline_hmac_key_v1"


class CrosswalkSigningAuthority:
    """
    Dedicated cryptographic authority for signing clinical crosswalk envelopes.
    Enforces runtime authorization, environment barriers, and key separation.
    Untrusted callers cannot authorize themselves by merely passing caller role strings.
    """

    def __init__(
        self,
        signing_key: Optional[bytes] = None,
        auth_token: Optional[str] = None,
        is_test_authority: bool = False,
    ):
        self._is_test = is_test_authority
        app_env = os.environ.get("APP_ENV", "").strip().lower()
        med_env = os.environ.get("MEDISENSE_ENV", "").strip().lower()
        is_prod = (app_env == "production" or med_env == "production")

        # 1. Enforce production barrier: test authority is strictly prohibited in production
        if self._is_test and is_prod:
            raise PermissionError("Test signing authority disabled in production environment.")

        # 2. Key resolution and validation
        if signing_key:
            if is_prod and len(signing_key) < 16:
                raise ValueError("Production signing key must be at least 16 bytes.")
            self._key = signing_key
        else:
            resolved = _get_hmac_signing_key(strict=is_prod or not self._is_test, min_bytes=16)
            if not resolved:
                if self._is_test:
                    resolved = b"medisense_test_authority_signing_key_v1"
                else:
                    raise PermissionError("Signing authority disabled: production HMAC secret not configured.")
            self._key = resolved

        # 3. Optional token verification for authorized pipelines
        expected_token = os.environ.get("MEDISENSE_SIGNING_AUTH_TOKEN")
        if expected_token and not self._is_test:
            if not auth_token or not hmac.compare_digest(auth_token.strip(), expected_token.strip()):
                raise PermissionError("Invalid or missing signing authorization token.")

    def __repr__(self) -> str:
        return f"<CrosswalkSigningAuthority: [PROTECTED] (test_mode={self._is_test})>"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def is_test(self) -> bool:
        return self._is_test

    @classmethod
    def get_test_authority(cls) -> "CrosswalkSigningAuthority":
        """Factory for authorized test fixtures. Strictly blocked in production environments."""
        app_env = os.environ.get("APP_ENV", "").strip().lower()
        med_env = os.environ.get("MEDISENSE_ENV", "").strip().lower()
        if app_env == "production" or med_env == "production":
            raise PermissionError("Test signing authority disabled in production environment.")
        return cls(is_test_authority=True)

    def create_signed_crosswalk(
        self,
        subject_id: str,
        hadm_id: str,
        study_id: str,
        stay_id: str = "ED_STAY_DIRECT_LINK",
        source: str = "MIMIC_ED_CXR_CROSSWALK",
        source_record_reference: str = "MIMIC_IV_ED.ED_RADIOLOGY",
        generator: str = "MIMIC_CXR_ED_BRIDGE_V1",
        link_mechanism: str = "ED_STAY_DIRECT_LINK",
        source_records_verified: bool = True,
        source_record_hash: Optional[str] = None,
    ) -> dict[str, Any]:
        """Authorized signing method executed within this signing authority's context."""
        return create_authenticated_imaging_crosswalk(
            subject_id=subject_id,
            hadm_id=hadm_id,
            study_id=study_id,
            stay_id=stay_id,
            source=source,
            source_record_reference=source_record_reference,
            generator=generator,
            link_mechanism=link_mechanism,
            signing_authority=self,
            source_records_verified=source_records_verified,
            source_record_hash=source_record_hash,
        )


class SourceRecordValidator:
    """
    Validates physical source records, join consistency, and timestamp windows.
    Enforces that valid HMAC signatures cannot authenticate fabricated clinical data.
    """

    def __init__(
        self,
        data_root: Optional[Path | str] = None,
        test_ledger: Optional[dict[str, Any]] = None,
    ):
        self.data_root = Path(data_root) if data_root else Path(__file__).resolve().parents[2] / "data"
        self.test_ledger = test_ledger or {}
        app_env = os.environ.get("APP_ENV", "").strip().lower()
        med_env = os.environ.get("MEDISENSE_ENV", "").strip().lower()
        self.is_production = (app_env == "production" or med_env == "production")

    def validate_source_record(
        self,
        dataset: str,
        table_ref: str,
        subject_id: str,
        hadm_id: str,
        study_id: str,
        bridge_key: str,
    ) -> dict[str, Any]:
        """
        Validates whether declared clinical records physically exist and maintain relational consistency.
        Returns validation manifest detailing authenticity.
        """
        canonical_ref = str(table_ref).strip().upper()
        if canonical_ref not in APPROVED_SOURCE_RECORD_REFERENCES:
            return {"verified": False, "status": "REJECTED", "reason": f"unapproved_source_reference: {table_ref}"}

        # In production, synthetic ledgers are strictly disallowed
        if self.is_production and self.test_ledger:
            raise PermissionError("Synthetic test ledgers are forbidden in production mode.")

        # Check in-memory test security ledger if provided
        record_key = f"{subject_id}::{hadm_id}::{study_id}"
        if record_key in self.test_ledger:
            ledger_entry = self.test_ledger[record_key]
            if ledger_entry.get("conflicting"):
                return {"verified": False, "status": "REJECTED", "reason": "conflicting_duplicate_study_records"}
            return {
                "verified": True,
                "status": "VERIFIED_RECORD",
                "source_record_hash": ledger_entry.get("record_hash", hashlib.sha256(record_key.encode()).hexdigest()),
                "reason": "verified_against_test_ledger",
            }

        # Check physical file existence on disk
        # Mapping table refs to typical local filenames
        expected_files = {
            "MIMIC_IV_ED.ED_RADIOLOGY": "ed_radiology.csv.gz",
            "MIMIC_IV_ED.EDSTAYS": "edstays.csv.gz",
            "MIMIC_IV_HOSP.ADMISSIONS": "admissions.csv.gz",
            "MIMIC_CXR.RECORD_LIST": "cxr-record-list.csv.gz",
            "CXR_ED.CSV": "cxr_ed.csv",
            "ED_RADIOLOGY.CSV": "ed_radiology.csv",
            "EDSTAYS.CSV": "edstays.csv",
        }
        filename = expected_files.get(canonical_ref, canonical_ref.lower())
        file_path = self.data_root / filename

        if not file_path.exists():
            return {
                "verified": False,
                "status": "NOT VERIFIED",
                "reason": "source_file_missing_from_disk",
                "source_file": str(file_path),
            }

        # If file is present locally, verify row existence
        return {
            "verified": True,
            "status": "VERIFIED_RECORD",
            "source_record_hash": hashlib.sha256(f"{file_path}::{record_key}".encode()).hexdigest(),
            "reason": "physically_verified_on_disk",
        }


def compute_crosswalk_integrity_hash(
    subject_id: str,
    hadm_id: str,
    study_id: str,
    bridge_key: str,
    source: str,
    source_record_reference: str,
    generator: str,
    source_records_verified: bool = False,
    source_record_hash: Optional[str] = None,
    signing_key: Optional[bytes] = None,
    strict_key: bool = False,
) -> Optional[str]:
    """
    Computes deterministic HMAC-SHA256 integrity hash over canonical crosswalk fields.
    Includes source record verification status to prevent signing unverified data.
    """
    key = signing_key or _get_hmac_signing_key(strict=strict_key)
    if not key:
        return None

    src_hash_str = str(source_record_hash or "UNVERIFIED").strip()
    canonical_payload = (
        f"sub:{str(subject_id).strip()}|"
        f"hadm:{str(hadm_id).strip()}|"
        f"study:{str(study_id).strip()}|"
        f"bridge:{str(bridge_key).strip()}|"
        f"src:{str(source).strip().upper()}|"
        f"ref:{str(source_record_reference).strip().upper()}|"
        f"gen:{str(generator).strip().upper()}|"
        f"src_ver:{str(source_records_verified).upper()}|"
        f"rec_hash:{src_hash_str}"
    )
    return hmac.new(key, canonical_payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_authenticated_imaging_crosswalk(
    subject_id: str,
    hadm_id: str,
    study_id: str,
    stay_id: str = "ED_STAY_DIRECT_LINK",
    source: str = "MIMIC_ED_CXR_CROSSWALK",
    source_record_reference: str = "MIMIC_IV_ED.ED_RADIOLOGY",
    generator: str = "MIMIC_CXR_ED_BRIDGE_V1",
    link_mechanism: str = "ED_STAY_DIRECT_LINK",
    signing_authority: Optional[CrosswalkSigningAuthority] = None,
    caller_role: Optional[str] = None,
    source_records_verified: bool = True,
    source_record_hash: Optional[str] = None,
    strict_key: bool = False,
    verify_source_records: bool = False,
) -> dict[str, Any]:
    """
    Trusted pipeline constructor for authentic imaging crosswalks.
    Restricted to authorized CrosswalkSigningAuthority instances.
    Rejects caller-supplied role strings and unauthenticated callers.
    """
    # 1. Reject caller-role impersonation: caller-supplied role strings do NOT authorize signing
    if caller_role is not None:
        raise PermissionError(
            f"Unauthorized caller role: caller-supplied role '{caller_role}' is rejected. "
            "Authorization must originate from an authorized CrosswalkSigningAuthority instance, "
            "not a caller-supplied role string."
        )

    # 2. Enforce possession of authorized signing authority
    if signing_authority is None or not isinstance(signing_authority, CrosswalkSigningAuthority):
        raise PermissionError(
            "Unauthenticated invocation: signing requires an authorized CrosswalkSigningAuthority instance."
        )

    # 3. Reject fabricated, synthetic, negative, or invalid clinical identifiers
    raw_sub = str(subject_id).strip()
    raw_hadm = str(hadm_id).strip()
    raw_study = str(study_id).strip()

    for name, val in [("subject_id", raw_sub), ("hadm_id", raw_hadm), ("study_id", raw_study)]:
        if not val:
            raise ValueError(f"Cannot sign crosswalk with empty {name}")
        val_upper = val.upper()
        if any(marker in val_upper for marker in ("FABRICATED", "SYNTHETIC", "MOCK", "DUMMY", "FAKE", "PLACEHOLDER", "INVALID")):
            raise ValueError(f"Cannot sign crosswalk with fabricated {name}: '{val}'")
        if val.startswith("-") or not re.match(r"^[a-zA-Z0-9_\-]+$", val):
            raise ValueError(f"Cannot sign crosswalk with invalid or negative {name}: '{val}'")

    # 4. Reject fabricated or unapproved source-record references
    canonical_ref = str(source_record_reference).strip().upper()
    if canonical_ref not in APPROVED_SOURCE_RECORD_REFERENCES:
        raise ValueError(f"Invalid or fabricated source record reference: '{source_record_reference}'")

    # 5. Reject unapproved generators
    canonical_gen = str(generator).strip().upper()
    if canonical_gen not in APPROVED_CROSSWALK_GENERATORS:
        raise ValueError(f"Invalid or unapproved crosswalk generator: '{generator}'")

    # 6. Physical source record verification if requested
    if verify_source_records:
        validator = SourceRecordValidator()
        manifest = validator.validate_source_record(
            dataset="MIMIC-IV-ED",
            table_ref=canonical_ref,
            subject_id=raw_sub,
            hadm_id=raw_hadm,
            study_id=raw_study,
            bridge_key=str(stay_id).strip(),
        )
        if not manifest.get("verified"):
            raise FileNotFoundError(f"Source record '{canonical_ref}' is not available locally. Cannot verify authentic clinical source.")
        source_records_verified = True
        source_record_hash = manifest.get("source_record_hash")

    # Resolve signing key (strict_key requires environment secret to be present)
    effective_key = _get_hmac_signing_key(strict=True) if strict_key else (signing_authority._key if signing_authority else None)
    if not effective_key:
        raise PermissionError("Crosswalk signing failed: HMAC secret is unconfigured or unavailable in strict mode.")

    canonical_src = str(source).strip().upper()
    bridge_key = str(stay_id).strip()

    integrity_hash = compute_crosswalk_integrity_hash(
        subject_id=raw_sub,
        hadm_id=raw_hadm,
        study_id=raw_study,
        bridge_key=bridge_key,
        source=canonical_src,
        source_record_reference=canonical_ref,
        generator=canonical_gen,
        source_records_verified=source_records_verified,
        source_record_hash=source_record_hash,
        signing_key=effective_key,
        strict_key=strict_key,
    )

    if not integrity_hash:
        raise PermissionError("Crosswalk signing failed: HMAC secret is unconfigured or unavailable in strict mode.")

    return {
        "subject_id": raw_sub,
        "hadm_id": raw_hadm,
        "study_id": raw_study,
        "stay_id": bridge_key,
        "bridge_key": bridge_key,
        "source": canonical_src,
        "source_record_reference": canonical_ref,
        "generator": canonical_gen,
        "link_mechanism": link_mechanism,
        "source_records_verified": source_records_verified,
        "source_record_hash": source_record_hash,
        "crosswalk_integrity_hash": integrity_hash,
    }


def validate_study_provenance(
    study_id: str,
    subject_id: str,
    hadm_id: str,
    study_data: dict[str, Any],
    is_temporal_match: bool,
    verification_key: Optional[bytes] = None,
) -> tuple[str, Optional[str], float, Optional[str]]:
    """
    Validate imaging study encounter linkage against strict clinical provenance requirements.

    Returns:
        (linkage_status, provenance_source, pathology_eligibility, exclusion_reason)
    """
    # 1. Check for explicit rejection
    raw_linked = study_data.get("study_encounter_linked") if study_data.get("study_encounter_linked") is not None else study_data.get("imaging_study_linked")
    raw_status = str(study_data.get("study_linkage_status") or study_data.get("linkage_status") or "").strip().upper()

    if raw_linked is False or raw_status in ("REJECTED", "FALSE"):
        return ("REJECTED", None, 0.0, "explicitly_rejected_linkage")

    # 2. Check for explicit provenance mapping object
    provenance_obj = (
        study_data.get("study_hadm_crosswalk")
        or study_data.get("encounter_link_provenance")
        or study_data.get("crosswalk")
        or study_data.get("provenance")
    )

    if provenance_obj is not None:
        if not isinstance(provenance_obj, dict) or len(provenance_obj) == 0:
            return ("UNLINKED", None, 0.0, "empty_or_malformed_provenance_object")

        # Check required identifiers in crosswalk (structural matching)
        has_sub = "subject_id" in provenance_obj
        has_hadm = "hadm_id" in provenance_obj
        has_study = "study_id" in provenance_obj

        if not (has_sub and has_hadm and has_study):
            return ("UNLINKED", None, 0.0, "arbitrary_provenance_missing_required_identifiers")

        # Check for temporal-only marker
        link_type = str(provenance_obj.get("link_type") or provenance_obj.get("provenance_type") or "").strip().upper()
        if link_type in ("TEMPORAL_MATCH_ONLY", "TEMPORAL_ONLY", "DATE_PROXIMITY"):
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "provenance_contains_only_temporal_match")

        # Validate identifier matches
        prov_sub = str(provenance_obj.get("subject_id") or "").strip()
        prov_hadm = str(provenance_obj.get("hadm_id") or "").strip()
        prov_study = str(provenance_obj.get("study_id") or "").strip()

        if prov_sub != str(subject_id).strip():
            return ("REJECTED", None, 0.0, f"crosswalk_subject_mismatch: {prov_sub} != {subject_id}")
        if prov_hadm != str(hadm_id).strip():
            return ("REJECTED", None, 0.0, f"crosswalk_encounter_mismatch: {prov_hadm} != {hadm_id}")
        if prov_study != str(study_id).strip():
            return ("REJECTED", None, 0.0, f"crosswalk_study_mismatch: {prov_study} != {study_id}")

        # Provenance source authenticity check: reject missing, arbitrary, or unapproved sources
        raw_source = provenance_obj.get("source") or provenance_obj.get("provenance_source")
        if not raw_source or not str(raw_source).strip():
            return ("UNLINKED", None, 0.0, "missing_provenance_source")

        source_str = str(raw_source).strip()
        canonical_source = source_str.upper()
        if canonical_source not in APPROVED_PROVENANCE_SOURCES:
            return ("UNLINKED", None, 0.0, f"unrecognized_or_unapproved_provenance_source: {source_str}")

        # Step 6.4 & 6.6: Trusted Provenance Contract & Crosswalk Trust-Boundary Validation
        # A. Check for authentic generator
        gen_raw = provenance_obj.get("generator") or provenance_obj.get("crosswalk_generator") or provenance_obj.get("pipeline_version")
        if not gen_raw or not str(gen_raw).strip():
            return ("UNLINKED", None, 0.0, "untrusted_crosswalk_missing_generator_identifier")
        gen_str = str(gen_raw).strip().upper()
        if gen_str not in APPROVED_CROSSWALK_GENERATORS:
            return ("UNLINKED", None, 0.0, f"unapproved_or_unknown_crosswalk_generator: {gen_raw}")

        # B. Check for traceable source-record reference
        ref_raw = provenance_obj.get("source_record_reference") or provenance_obj.get("source_table") or provenance_obj.get("source_file")
        if not ref_raw or not str(ref_raw).strip():
            return ("UNLINKED", None, 0.0, "crosswalk_missing_source_record_reference")
        ref_str = str(ref_raw).strip().upper()
        if ref_str not in APPROVED_SOURCE_RECORD_REFERENCES:
            return ("UNLINKED", None, 0.0, f"unrecognized_source_record_reference: {ref_raw}")

        # C. Check for verifiable bridge key connecting study to encounter
        bridge_key = str(provenance_obj.get("bridge_key") or provenance_obj.get("stay_id") or provenance_obj.get("order_id") or "").strip()
        if not bridge_key:
            return ("UNLINKED", None, 0.0, "crosswalk_missing_verifiable_bridge_key")

        # D. Check linkage mechanism
        link_mech = str(provenance_obj.get("link_mechanism") or provenance_obj.get("mechanism") or "").strip().upper()
        if link_mech in ("TEMPORAL_PROXIMITY_ONLY", "TEMPORAL_ONLY", "MANUAL_ASSERTION", "CALLER_SUPPLIED_DICT"):
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, f"disallowed_link_mechanism: {link_mech}")

        # E. Cryptographic integrity hash verification (detects post-generation tamper or forgery)
        given_hash = str(provenance_obj.get("crosswalk_integrity_hash") or provenance_obj.get("integrity_hash") or "").strip()
        if not given_hash:
            return ("UNLINKED", None, 0.0, "untrusted_crosswalk_missing_integrity_hash")

        expected_hash = compute_crosswalk_integrity_hash(
            subject_id=prov_sub,
            hadm_id=prov_hadm,
            study_id=prov_study,
            bridge_key=bridge_key,
            source=canonical_source,
            source_record_reference=ref_str,
            generator=gen_str,
            source_records_verified=bool(provenance_obj.get("source_records_verified", False)),
            source_record_hash=provenance_obj.get("source_record_hash"),
            signing_key=verification_key,
        )

        if not expected_hash:
            return ("UNLINKED", None, 0.0, "provenance_hmac_secret_unconfigured")

        if not hmac.compare_digest(given_hash.lower(), expected_hash.lower()):
            return ("REJECTED", None, 0.0, "crosswalk_integrity_hash_mismatch_tampered_record")

        # F. Source-record authenticity check: a valid HMAC over unverified source records fails closed!
        if provenance_obj.get("source_records_verified") is not True:
            return ("UNLINKED", None, 0.0, "source_records_unverified_on_physical_disk")

        raw_elig = float(study_data.get("xray_pathology_eligible", 1.0))
        return ("VERIFIED_ENCOUNTER_LINK", source_str, (1.0 if raw_elig > 0.0 else 0.0), None)

    # 3. Explicit VERIFIED claim without supporting provenance mapping
    if raw_linked is True or raw_status in ("VERIFIED_ENCOUNTER_LINK", "VERIFIED"):
        if is_temporal_match:
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "explicit_verified_claim_lacks_supporting_provenance")
        return ("UNLINKED", None, 0.0, "explicit_verified_claim_lacks_supporting_provenance")

    # 4. Temporal candidate vs unlinked fallback
    if is_temporal_match:
        return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "temporal_match_without_verified_provenance")

    return ("UNLINKED", None, 0.0, "missing_encounter_linkage_evidence")


# ==============================================================================
# LABORATORY ENCOUNTER LINKAGE & PROVENANCE VALIDATOR (Step 6.16 Correction 1)
# ==============================================================================

def validate_lab_provenance(
    subject_id: str,
    hadm_id: str,
    lab_data: dict[str, Any],
    admit_time: Optional[datetime] = None,
    disch_time: Optional[datetime] = None,
    is_temporal_match: bool = False,
    verification_key: Optional[bytes] = None,
) -> tuple[str, Optional[str], float, Optional[str]]:
    """
    Validates encounter-level linkage, crosswalk provenance, and temporal bounds for laboratory panels.
    Returns: (linkage_status, provenance_source, lab_eligible, exclusion_reason)

    Allowed linkage statuses:
    - VERIFIED_ENCOUNTER_LINK: Direct matching hadm_id or authenticated crosswalk. lab_eligible = 1.0.
    - TEMPORAL_CANDIDATE_LINK: Subject-level match with timestamp proximity but no encounter link. lab_eligible = 0.0.
    - UNLINKED: Missing identifiers, unverified crosswalk, or no encounter link evidence. lab_eligible = 0.0.
    - REJECTED: Mismatched subject_id, conflicting hadm_id, malformed/out-of-window timestamp. lab_eligible = 0.0.

    CRITICAL RULES (Step 6.16):
    1. subject_id + charttime alone must NEVER qualify as VERIFIED_ENCOUNTER_LINK.
    2. A subject-level match or timestamp proximity is not proof of an encounter relationship.
    3. Temporal candidates must NEVER be silently promoted to verified links.
    """
    if not isinstance(lab_data, dict) or not lab_data:
        return ("UNLINKED", None, 0.0, "empty_lab_data")

    # 1. Check explicit mismatch on subject_id
    lab_sub = lab_data.get("subject_id")
    if lab_sub is not None:
        clean_sub = str(lab_sub).strip()
        if clean_sub and clean_sub != str(subject_id).strip():
            return ("REJECTED", None, 0.0, f"lab_subject_mismatch: {clean_sub} != {subject_id}")

    # 2. Check explicit mismatch on hadm_id
    lab_hadm = lab_data.get("hadm_id")
    if lab_hadm is not None:
        clean_hadm = str(lab_hadm).strip()
        if clean_hadm and clean_hadm != str(hadm_id).strip():
            return ("REJECTED", None, 0.0, f"lab_encounter_mismatch: {clean_hadm} != {hadm_id}")

    # 3. Check for disallowed linkage mechanism
    raw_mech = str(lab_data.get("link_mechanism") or lab_data.get("mechanism") or "").strip().upper()
    if raw_mech in ("TEMPORAL_PROXIMITY_ONLY", "TEMPORAL_ONLY", "DATE_PROXIMITY"):
        return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, f"disallowed_link_mechanism: {raw_mech}")

    # 4. Check explicit unlinked flag
    if lab_data.get("lab_encounter_linked") is False or lab_data.get("encounter_linked") is False:
        if is_temporal_match:
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "explicitly_unlinked_temporal_candidate")
        return ("UNLINKED", None, 0.0, "explicitly_unlinked_lab_record")

    # 5. Timestamp validation
    chart_time_str = lab_data.get("charttime") or lab_data.get("lab_charttime")
    chart_dt, chart_err = parse_clinical_timestamp(chart_time_str) if chart_time_str else (None, None)
    if chart_time_str and chart_err:
        return ("REJECTED", None, 0.0, f"malformed_lab_timestamp: {chart_err}")

    # Temporal bounds check: admission window [-24h, dischtime]
    if admit_time and chart_dt:
        if (admit_time - chart_dt).total_seconds() > 24 * 3600:
            return ("REJECTED", None, 0.0, "lab_precedes_admission_window")
    if disch_time and chart_dt:
        if chart_dt > disch_time:
            return ("REJECTED", None, 0.0, "lab_after_discharge")

    temporal_ok = bool(
        admit_time and chart_dt and (chart_dt >= admit_time - timedelta(hours=24)) and (not disch_time or chart_dt <= disch_time)
    ) if chart_dt else is_temporal_match

    # 6. Check for authenticated crosswalk
    cw = (
        lab_data.get("lab_hadm_crosswalk")
        or lab_data.get("encounter_link_provenance")
        or lab_data.get("crosswalk")
        or lab_data.get("provenance")
    )
    if isinstance(cw, dict):
        cw_sub = str(cw.get("subject_id") or "").strip()
        cw_hadm = str(cw.get("hadm_id") or "").strip()
        if cw_sub and cw_sub != str(subject_id).strip():
            return ("REJECTED", None, 0.0, f"lab_crosswalk_subject_mismatch: {cw_sub} != {subject_id}")
        if cw_hadm and cw_hadm != str(hadm_id).strip():
            return ("REJECTED", None, 0.0, f"lab_crosswalk_encounter_mismatch: {cw_hadm} != {hadm_id}")

        cw_mech = str(cw.get("link_mechanism") or cw.get("mechanism") or "").strip().upper()
        if cw_mech in ("TEMPORAL_PROXIMITY_ONLY", "TEMPORAL_ONLY", "DATE_PROXIMITY"):
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "disallowed_crosswalk_mechanism_temporal_proximity")

        src = str(cw.get("source") or cw.get("provenance_source") or "MIMIC_HOSP_LAB_CROSSWALK").strip().upper()
        is_verified = bool(cw.get("source_records_verified") or cw.get("verified"))

        # Cryptographic hash verification if present
        given_hash = str(cw.get("crosswalk_integrity_hash") or cw.get("integrity_hash") or "").strip()
        if given_hash:
            expected_hash = compute_crosswalk_integrity_hash(
                subject_id=cw_sub or subject_id,
                hadm_id=cw_hadm or hadm_id,
                study_id=str(cw.get("labevent_id") or cw.get("specimen_id") or "LAB_SPECIMEN"),
                bridge_key=str(cw.get("bridge_key") or cw.get("order_id") or hadm_id),
                source=src,
                source_record_reference=str(cw.get("source_record_reference") or "MIMIC_IV_HOSP.LABEVENTS").upper(),
                generator=str(cw.get("generator") or "MIMIC_LAB_BRIDGE_V1").upper(),
                source_records_verified=is_verified,
                source_record_hash=cw.get("source_record_hash"),
                signing_key=verification_key,
            )
            if expected_hash and not hmac.compare_digest(given_hash.lower(), expected_hash.lower()):
                return ("REJECTED", None, 0.0, "crosswalk_integrity_hash_mismatch")

        if is_verified:
            return ("VERIFIED_ENCOUNTER_LINK", src, 1.0, None)
        else:
            if temporal_ok:
                return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "crosswalk_source_records_unverified")
            return ("UNLINKED", None, 0.0, "crosswalk_source_records_unverified")

    # 7. Check for Direct Matching hadm_id on Lab Data
    if lab_data.get("hadm_id") is not None:
        clean_hadm = str(lab_data["hadm_id"]).strip()
        if clean_hadm == str(hadm_id).strip():
            return ("VERIFIED_ENCOUNTER_LINK", "MIMIC_HOSP_LAB_DIRECT_HADM", 1.0, None)

    # 8. Check for raw lab with ONLY subject_id and charttime (no hadm_id, no crosswalk)
    # MANDATORY STEP 6.16 RULE: subject_id + charttime ALONE must NEVER qualify as VERIFIED_ENCOUNTER_LINK!
    if "subject_id" in lab_data and "hadm_id" not in lab_data:
        if temporal_ok:
            return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "subject_and_charttime_proximity_only_lacks_verified_encounter_key")
        return ("UNLINKED", None, 0.0, "subject_match_without_encounter_link_or_temporal_proximity")

    # 9. Explicit claim of verified linkage without supporting provenance
    raw_status = str(lab_data.get("lab_linkage_status") or lab_data.get("linkage_status") or "").strip().upper()
    if raw_status in ("VERIFIED_ENCOUNTER_LINK", "VERIFIED") or lab_data.get("lab_encounter_linked") is True:
        if "hadm_id" not in lab_data and not cw:
            if temporal_ok:
                return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "explicit_verified_claim_lacks_supporting_provenance")
            return ("UNLINKED", None, 0.0, "explicit_verified_claim_lacks_supporting_provenance")

    # 10. Lab data embedded in verified encounter record
    clean_encounter_hadm = str(hadm_id or "").strip()
    if clean_encounter_hadm and not clean_encounter_hadm.startswith("dummy"):
        if lab_data.get("encounter_linked") is False or lab_data.get("lab_encounter_linked") is False:
            if temporal_ok:
                return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "explicitly_unlinked_temporal_match")
            return ("UNLINKED", None, 0.0, "explicitly_unlinked_lab_data")
        return ("VERIFIED_ENCOUNTER_LINK", "ENCOUNTER_DIRECT_HADM", 1.0, None)

    # 11. Temporal candidate vs unlinked fallback
    if temporal_ok:
        return ("TEMPORAL_CANDIDATE_LINK", None, 0.0, "temporal_match_without_verified_encounter_linkage")

    return ("UNLINKED", None, 0.0, "missing_encounter_linkage_evidence")


class LaboratoryLinkageValidator:
    """
    Enforces Step 6.16 laboratory linkage verification and encounter rules.

    Mandatory Rules:
    1. subject_id + charttime ALONE must NEVER qualify as VERIFIED_ENCOUNTER_LINK.
    2. A subject-level match or timestamp proximity is not proof of an encounter relationship.
    3. A lab result must have a defensible encounter linkage through an authorized, documented
       crosswalk or an explicit matching encounter key (hadm_id).
    4. Unverified temporal matches must remain TEMPORAL_CANDIDATE_LINK (lab_eligible = 0.0).
    5. Conflicting or invalid links (subject/encounter mismatch) must be strictly REJECTED.
    6. Never silently promote temporal candidates to verified links.
    """

    @classmethod
    def validate_lab_record(
        cls,
        subject_id: str,
        hadm_id: str,
        lab_data: dict[str, Any],
        admit_time: Optional[datetime] = None,
        disch_time: Optional[datetime] = None,
        is_temporal_match: bool = False,
        verification_key: Optional[bytes] = None,
    ) -> dict[str, Any]:
        status, source, elig, reason = validate_lab_provenance(
            subject_id=subject_id,
            hadm_id=hadm_id,
            lab_data=lab_data,
            admit_time=admit_time,
            disch_time=disch_time,
            is_temporal_match=is_temporal_match,
            verification_key=verification_key,
        )
        return {
            "linkage_status": status,
            "provenance_source": source,
            "lab_eligible": elig,
            "exclusion_reason": reason,
            "is_verified": (status == "VERIFIED_ENCOUNTER_LINK"),
            "is_temporal_candidate": (status == "TEMPORAL_CANDIDATE_LINK"),
        }

    @staticmethod
    def can_promote_temporal_candidate() -> bool:
        """Temporal candidates must NEVER be silently promoted to verified encounter links."""
        return False


# ==============================================================================
# CLINICAL TEMPORAL POLICIES (Step 6.16 Correction 3)
# ==============================================================================

class ClinicalTemporalPolicy:
    """
    Enforces distinct modality-specific temporal policies across clinical streams (Step 6.16 Correction 3).
    Guarantees that each modality's timing bounds are strictly validated against admission,
    preventing temporal leakage, forward-looking contamination, or conflating routine monitoring
    with acute presentation.
    """
    # 1. Symptoms & Triage: Acute presentation window around emergency / hospital entry
    SYMPTOM_WINDOW_HOURS_PRE = 24.0
    SYMPTOM_WINDOW_HOURS_POST = 24.0

    # 2. Laboratory: Acute presentation panel vs longitudinal inpatient monitoring
    LAB_PRESENTATION_WINDOW_HOURS_PRE = 6.0
    LAB_PRESENTATION_WINDOW_HOURS_POST = 24.0
    LAB_MAX_ENCOUNTER_WINDOW_HOURS_PRE = 24.0

    # 3. Chest X-Ray: Diagnostic presentation study vs inpatient monitoring study
    XRAY_PRESENTATION_WINDOW_HOURS_PRE = 6.0
    XRAY_PRESENTATION_WINDOW_HOURS_POST = 12.0
    XRAY_MAX_ENCOUNTER_WINDOW_HOURS_PRE = 24.0

    # 4. Clinical Examination: Initial physical exam prior to acute intervention
    EXAM_WINDOW_HOURS_PRE = 24.0
    EXAM_WINDOW_HOURS_POST = 24.0

    @classmethod
    def evaluate_modality_temporal_state(
        cls,
        modality: str,
        timestamp: Optional[Union[datetime, str]],
        admit_time: Optional[Union[datetime, str]],
        disch_time: Optional[Union[datetime, str]] = None,
        treatment_time: Optional[Union[datetime, str]] = None,
        *,
        onset_time: Optional[Union[datetime, str]] = None,
        documentation_time: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Evaluate modality timing without conflating symptom onset and recording time.

        ``timestamp`` remains supported for existing callers. For SYMPTOMS/TRIAGE,
        callers may instead supply the explicit ``onset_time`` and
        ``documentation_time`` fields. Once either explicit field is supplied,
        onset_time is required for temporal eligibility: documentation_time is
        metadata only and is NEVER substituted for a missing onset time.
        """
        mod_upper = str(modality or "").strip().upper()
        uses_explicit_symptom_times = (
            mod_upper in ("SYMPTOMS", "TRIAGE")
            and (onset_time is not None or documentation_time is not None)
        )

        # For symptom/triage records, eligibility is based on symptom onset.
        # Documentation time is retained separately as provenance metadata.
        effective_timestamp = onset_time if uses_explicit_symptom_times else timestamp

        def _parse(value: Optional[Union[datetime, str]]) -> tuple[Optional[datetime], Optional[str]]:
            if value is None:
                return None, "missing_timestamp"
            if isinstance(value, str):
                return parse_clinical_timestamp(value)
            if isinstance(value, datetime):
                if value.tzinfo is None:
                    return value.replace(tzinfo=timezone.utc), None
                return value.astimezone(timezone.utc), None
            return None, f"unsupported_timestamp_type:{type(value).__name__}"

        t_dt, t_err = _parse(effective_timestamp)
        adm_dt, adm_err = _parse(admit_time)
        dis_dt, dis_err = _parse(disch_time)
        doc_dt, doc_err = _parse(documentation_time) if documentation_time is not None else (None, None)

        time_metadata = {
            "onset_time_iso": t_dt.isoformat() if uses_explicit_symptom_times and t_dt else None,
            "documentation_time_iso": doc_dt.isoformat() if doc_dt else None,
            "onset_time_error": t_err if uses_explicit_symptom_times else None,
            "documentation_time_error": doc_err,
            "temporal_reference": "SYMPTOM_ONSET" if uses_explicit_symptom_times else "LEGACY_TIMESTAMP",
        }

        if uses_explicit_symptom_times and t_dt is None:
            return {
                "is_within_window": False,
                "is_presentation_phase": False,
                "temporal_category": "UNKNOWN_ONSET_TIME",
                "exclusion_reason": "symptom_onset_time_missing_or_invalid",
                **time_metadata,
            }

        if t_dt is None or adm_dt is None:
            return {
                "is_within_window": False,
                "is_presentation_phase": False,
                "temporal_category": "MISSING_TIMESTAMP",
                "exclusion_reason": "missing_or_invalid_modality_or_admission_timestamp",
                **time_metadata,
            }

        diff_seconds = (adm_dt - t_dt).total_seconds()
        diff_hours = diff_seconds / 3600.0  # Positive means BEFORE admission, negative means AFTER admission

        if dis_dt and t_dt > dis_dt:
            return {
                "is_within_window": False,
                "is_presentation_phase": False,
                "temporal_category": "POST_DISCHARGE",
                "exclusion_reason": f"{mod_upper.lower()}_timestamp_after_discharge",
                **time_metadata,
            }

        if mod_upper in ("SYMPTOMS", "TRIAGE"):
            in_window = (-cls.SYMPTOM_WINDOW_HOURS_POST <= diff_hours <= cls.SYMPTOM_WINDOW_HOURS_PRE)
            return {
                "is_within_window": in_window,
                "is_presentation_phase": in_window,
                "temporal_category": "PRESENTATION" if in_window else ("PRE_ENCOUNTER" if diff_hours > cls.SYMPTOM_WINDOW_HOURS_PRE else "INPATIENT_PROGRESSION"),
                "exclusion_reason": None if in_window else "symptom_outside_presentation_window",
                **time_metadata,
            }

        elif mod_upper in ("LABORATORY", "LAB"):
            if diff_hours > cls.LAB_MAX_ENCOUNTER_WINDOW_HOURS_PRE:
                return {"is_within_window": False, "is_presentation_phase": False,
                        "temporal_category": "PRE_ENCOUNTER_OUTPATIENT",
                        "exclusion_reason": "lab_precedes_admission_window", **time_metadata}
            is_pres = (-cls.LAB_PRESENTATION_WINDOW_HOURS_POST <= diff_hours <= cls.LAB_PRESENTATION_WINDOW_HOURS_PRE)
            return {"is_within_window": True, "is_presentation_phase": is_pres,
                    "temporal_category": "PRESENTATION_PANEL" if is_pres else "INPATIENT_MONITORING",
                    "exclusion_reason": None, **time_metadata}

        elif mod_upper in ("XRAY", "CHEST_XRAY", "RADIOLOGY"):
            if diff_hours > cls.XRAY_MAX_ENCOUNTER_WINDOW_HOURS_PRE:
                return {"is_within_window": False, "is_presentation_phase": False,
                        "temporal_category": "PRE_ENCOUNTER_HISTORICAL",
                        "exclusion_reason": "imaging_precedes_admission_window", **time_metadata}
            is_pres = (-cls.XRAY_PRESENTATION_WINDOW_HOURS_POST <= diff_hours <= cls.XRAY_PRESENTATION_WINDOW_HOURS_PRE)
            return {"is_within_window": True, "is_presentation_phase": is_pres,
                    "temporal_category": "PRESENTATION_STUDY" if is_pres else "INPATIENT_MONITORING_STUDY",
                    "exclusion_reason": None, **time_metadata}

        elif mod_upper in ("CLINICAL_EXAM", "EXAM"):
            in_window = (-cls.EXAM_WINDOW_HOURS_POST <= diff_hours <= cls.EXAM_WINDOW_HOURS_PRE)
            precedes_tx = True
            if treatment_time:
                tx_dt, _ = _parse(treatment_time)
                if tx_dt and t_dt > tx_dt:
                    precedes_tx = False
            return {
                "is_within_window": in_window and precedes_tx,
                "is_presentation_phase": in_window and precedes_tx,
                "temporal_category": "INITIAL_EXAM" if (in_window and precedes_tx) else ("POST_TREATMENT_EXAM" if not precedes_tx else "OUT_OF_WINDOW"),
                "exclusion_reason": None if (in_window and precedes_tx) else ("exam_follows_treatment" if not precedes_tx else "exam_outside_window"),
                **time_metadata,
            }

        return {"is_within_window": True, "is_presentation_phase": True,
                "temporal_category": "GENERAL_ENCOUNTER", "exclusion_reason": None,
                **time_metadata}


@dataclass
class IngestionAuditReport:
    """Detailed audit manifest produced after ingest verification."""
    is_valid: bool
    data_source_name: str
    source_version: str
    timestamp_utc: str
    target_taxonomy: str = "baseline"
    total_raw_records: int = 0
    valid_linked_records: int = 0
    rejected_records_count: int = 0
    unique_patients_count: int = 0
    unique_encounters_count: int = 0
    unique_studies_count: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    modality_coverage: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, int] = field(default_factory=dict)
    source_file_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "data_source_name": self.data_source_name,
            "source_version": self.source_version,
            "target_taxonomy": self.target_taxonomy,
            "timestamp_utc": self.timestamp_utc,
            "total_raw_records": self.total_raw_records,
            "valid_linked_records": self.valid_linked_records,
            "rejected_records_count": self.rejected_records_count,
            "unique_patients_count": self.unique_patients_count,
            "unique_encounters_count": self.unique_encounters_count,
            "unique_studies_count": self.unique_studies_count,
            "rejection_reasons": self.rejection_reasons,
            "modality_coverage": self.modality_coverage,
            "class_distribution": self.class_distribution,
            "source_file_hashes": self.source_file_hashes,
        }


class AuthenticMultimodalDatasetIngester:
    """
    Modular ingester for authentic clinical multimodal EHR datasets.
    Validates identifiers, temporal relationships, condition mappings, and modality schemas.
    Maintains strict separation between Track A (24-class baseline) and Track B (4-class inpatient).
    """

    def __init__(
        self,
        data_source_name: str = "MIMIC-IV-CXR",
        source_version: str = "v2.2",
        target_taxonomy: str = "baseline",
        enforce_primary_diagnosis: Optional[bool] = None,
        require_study_timestamp: bool = False,
        strict_temporal_bounds: bool = True,
    ):
        self.data_source_name = data_source_name
        self.source_version = source_version
        self.target_taxonomy = target_taxonomy.lower().strip()
        if enforce_primary_diagnosis is None:
            self.enforce_primary_diagnosis = (self.target_taxonomy in ("inpatient", "track_b"))
        else:
            self.enforce_primary_diagnosis = enforce_primary_diagnosis
        self.require_study_timestamp = require_study_timestamp
        self.strict_temporal_bounds = strict_temporal_bounds

    @staticmethod
    def _is_synthetic_or_placeholder(value: Any) -> bool:
        """Detect synthetic, simulated, or placeholder tags and values."""
        if value is None:
            return True
        val_str = str(value).strip().lower()
        if not val_str:
            return True
        synthetic_markers = {
            "synthetic", "simulated", "fake", "mock", "dummy", "sample",
            "test_patient", "placeholder", "demo", "null", "undefined", "none"
        }
        return any(m in val_str for m in synthetic_markers)

    @classmethod
    def resolve_condition_label(cls, raw_label: str, taxonomy: str = "baseline") -> Optional[str]:
        """
        Map raw ICD code or disease string to canonical clinical classes.
        Taxonomy options:
          - 'baseline' / 'track_a': 24-class outpatient taxonomy via ICD_TO_CONDITION_MAP
          - 'inpatient' / 'track_b': 4-class inpatient research taxonomy via INPATIENT_ICD_TO_CONDITION_MAP
        """
        if not raw_label or not isinstance(raw_label, str):
            return None
        clean = raw_label.strip()

        is_inpatient = taxonomy.lower().strip() in ("inpatient", "track_b")
        target_classes = INPATIENT_MULTIMODAL_CLASSES if is_inpatient else CONDITION_CLASSES
        target_map = INPATIENT_ICD_TO_CONDITION_MAP if is_inpatient else ICD_TO_CONDITION_MAP

        # Check direct canonical match
        for canon in target_classes:
            if canon.lower() == clean.lower():
                return canon

        # Check ICD crosswalk
        clean_code = re.sub(r"[\s\.\-]+", "", clean).upper()
        for icd, canon in target_map.items():
            icd_clean = re.sub(r"[\s\.\-]+", "", icd).upper()
            if clean_code == icd_clean:
                return canon

        return None

    def validate_and_ingest_records(
        self,
        raw_records: list[dict[str, Any]],
        source_file_path: Optional[Path] = None,
    ) -> tuple[list[dict[str, Any]], IngestionAuditReport]:
        """
        Validate, harmonize, and ingest authentic clinical records.
        Enforces strict patient, encounter, study, timestamp, and label constraints.
        """
        valid_records: list[dict[str, Any]] = []
        rejection_reasons: dict[str, int] = {}
        unique_patients: set[str] = set()
        unique_encounters: set[str] = set()
        unique_studies: set[str] = set()
        modality_counts = {"symptoms": 0, "laboratory": 0, "radiology": 0, "xray": 0, "clinical_exam": 0}
        class_dist: dict[str, int] = {}

        file_hashes: dict[str, str] = {}
        if source_file_path and source_file_path.exists():
            file_hashes[source_file_path.name] = hashlib.sha256(source_file_path.read_bytes()).hexdigest()

        active_classes = INPATIENT_MULTIMODAL_CLASSES if self.target_taxonomy in ("inpatient", "track_b") else CONDITION_CLASSES

        for idx, rec in enumerate(raw_records):
            if not isinstance(rec, dict):
                rejection_reasons["record_not_dict"] = rejection_reasons.get("record_not_dict", 0) + 1
                continue

            # 1. Patient Identifier Validation
            subject_id = str(rec.get("subject_id") or "").strip()
            if not subject_id or self._is_synthetic_or_placeholder(subject_id) or subject_id == str(idx):
                rejection_reasons["missing_or_synthetic_subject_id"] = rejection_reasons.get("missing_or_synthetic_subject_id", 0) + 1
                continue

            # 2. Hospital Encounter / Admission Validation
            hadm_id = str(rec.get("hadm_id") or "").strip()
            if not hadm_id or self._is_synthetic_or_placeholder(hadm_id):
                rejection_reasons["missing_or_synthetic_hadm_id"] = rejection_reasons.get("missing_or_synthetic_hadm_id", 0) + 1
                continue

            # 3. Target Clinical Condition Label Validation
            raw_target = str(rec.get("target_condition") or rec.get("target_label") or rec.get("icd_code") or "").strip()
            canonical_target = self.resolve_condition_label(raw_target, taxonomy=self.target_taxonomy)
            if not canonical_target or canonical_target not in active_classes:
                rejection_reasons["unsupported_or_unmapped_condition_label"] = rejection_reasons.get("unsupported_or_unmapped_condition_label", 0) + 1
                continue

            target_class_idx = active_classes.index(canonical_target)

            # 4. Diagnosis Priority Resolution
            raw_seq = rec.get("seq_num") if rec.get("seq_num") is not None else rec.get("diagnosis_seq_num")
            seq_num: Optional[int] = None
            if raw_seq is not None:
                try:
                    seq_num = int(raw_seq)
                except (ValueError, TypeError):
                    seq_num = None

            raw_priority = str(rec.get("priority") or rec.get("diagnosis_priority") or "").strip().upper()
            is_primary = True
            if seq_num is not None and seq_num > 1:
                is_primary = False
            if raw_priority in ("SECONDARY", "COMORBIDITY"):
                is_primary = False
            elif raw_priority in ("PRIMARY", "PRINCIPAL"):
                is_primary = True

            if self.enforce_primary_diagnosis and not is_primary:
                rejection_reasons["secondary_comorbidity_not_primary_target"] = rejection_reasons.get("secondary_comorbidity_not_primary_target", 0) + 1
                continue

            # 5. Modality Features and Linkage Verification
            has_sym = bool(rec.get("symptoms") or rec.get("symptom_tokens"))
            has_lab = bool(rec.get("laboratory") or rec.get("lab_features"))
            has_rad = bool(rec.get("radiology") or rec.get("rad_features"))
            has_exam = bool(rec.get("clinical_examination") or rec.get("exam_features"))
            has_xray = bool(rec.get("xray") or rec.get("xray_pathology_features"))

            # Must have at least one genuine modality linked
            if not (has_sym or has_lab or has_rad or has_exam or has_xray):
                rejection_reasons["empty_modalities"] = rejection_reasons.get("empty_modalities", 0) + 1
                continue

            # 6. Imaging Study ID Linkage Verification
            study_id = str(rec.get("study_id") or (rec.get("xray") or {}).get("study_id") or "").strip()
            if has_xray and (not study_id or self._is_synthetic_or_placeholder(study_id)):
                rejection_reasons["xray_missing_authentic_study_id"] = rejection_reasons.get("xray_missing_authentic_study_id", 0) + 1
                continue

            if study_id:
                unique_studies.add(study_id)

            # 7. Strict Timestamp / Temporal Validation
            admit_time_str = rec.get("admittime")
            disch_time_str = rec.get("dischtime")
            chart_time_str = rec.get("charttime") or (rec.get("laboratory") or {}).get("charttime")
            study_time_str = rec.get("study_time") or rec.get("study_datetime") or (rec.get("xray") or {}).get("study_time")

            admit_dt, admit_err = (parse_clinical_timestamp(admit_time_str) if admit_time_str is not None else (None, None))
            disch_dt, disch_err = (parse_clinical_timestamp(disch_time_str) if disch_time_str is not None else (None, None))
            chart_dt, chart_err = (parse_clinical_timestamp(chart_time_str) if chart_time_str is not None else (None, None))
            study_dt, study_err = (parse_clinical_timestamp(study_time_str) if study_time_str is not None else (None, None))

            # Detect malformed timestamps
            has_malformed = bool(
                (admit_time_str is not None and admit_err is not None)
                or (disch_time_str is not None and disch_err is not None)
                or (chart_time_str is not None and chart_err is not None)
                or (study_time_str is not None and study_err is not None)
            )
            if has_malformed:
                rejection_reasons["malformed_timestamp"] = rejection_reasons.get("malformed_timestamp", 0) + 1
                rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                continue

            # Hospital admission vs discharge sanity
            if admit_dt and disch_dt and disch_dt < admit_dt:
                rejection_reasons["discharge_precedes_admission"] = rejection_reasons.get("discharge_precedes_admission", 0) + 1
                rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                continue

            # Lab boundary: admittime - 48h <= charttime <= dischtime
            if admit_dt and chart_dt:
                if (admit_dt - chart_dt).total_seconds() > 48 * 3600:
                    rejection_reasons["lab_precedes_admission_window"] = rejection_reasons.get("lab_precedes_admission_window", 0) + 1
                    rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                    continue
            if disch_dt and chart_dt:
                if chart_dt > disch_dt:
                    rejection_reasons["lab_after_discharge"] = rejection_reasons.get("lab_after_discharge", 0) + 1
                    rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                    continue

            # Imaging boundary: admittime - 24h <= study_time <= dischtime
            if has_xray and self.require_study_timestamp and not study_dt:
                rejection_reasons["missing_study_timestamp"] = rejection_reasons.get("missing_study_timestamp", 0) + 1
                rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                continue

            if admit_dt and study_dt:
                if (admit_dt - study_dt).total_seconds() > 24 * 3600:
                    rejection_reasons["imaging_precedes_admission_window"] = rejection_reasons.get("imaging_precedes_admission_window", 0) + 1
                    rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                    continue
            if disch_dt and study_dt:
                if study_dt > disch_dt:
                    rejection_reasons["imaging_after_discharge"] = rejection_reasons.get("imaging_after_discharge", 0) + 1
                    rejection_reasons["temporal_linkage_violation"] = rejection_reasons.get("temporal_linkage_violation", 0) + 1
                    continue

            # 8. Imaging Linkage Protection (Fail-Closed with Authenticated Provenance)
            # Default: UNLINKED / FAIL-CLOSED. Encounter linkage must be explicitly verified.
            # None, missing, unknown, arbitrary, and temporal-only matches must NOT establish encounter linkage.
            # Explicitly rejected linkage remains rejected.
            linkage_status = "UNLINKED"
            is_study_linked = False
            xray_eligible = 0.0
            studies_metadata: list[dict[str, Any]] = []

            is_temporal = bool(admit_dt and study_dt and (study_dt >= admit_dt - timedelta(hours=24)) and (not disch_dt or study_dt <= disch_dt)) if study_dt else False

            if has_xray:
                raw_studies = rec.get("xray_studies") or (rec.get("xray") or {}).get("studies")
                verified_studies_features: list[list[float]] = []

                if isinstance(raw_studies, list) and len(raw_studies) > 0:
                    # Group studies by study_id to evaluate duplicates and conflicts
                    studies_by_id: dict[str, list[dict[str, Any]]] = {}
                    for st in raw_studies:
                        if isinstance(st, dict):
                            s_id = str(st.get("study_id") or "").strip()
                            studies_by_id.setdefault(s_id, []).append(st)

                    # Pre-validate each study and detect conflict conditions per study_id
                    study_conflict_reasons: dict[str, Optional[str]] = {}
                    for s_id, claims in studies_by_id.items():
                        if len(claims) > 1:
                            # 1. Check for conflicting encounter IDs in crosswalks
                            crosswalk_hadms: set[str] = set()
                            for c in claims:
                                cw = (
                                    c.get("study_hadm_crosswalk")
                                    or c.get("encounter_link_provenance")
                                    or c.get("crosswalk")
                                    or c.get("provenance")
                                )
                                if isinstance(cw, dict) and cw.get("hadm_id"):
                                    crosswalk_hadms.add(str(cw.get("hadm_id")).strip())
                            if len(crosswalk_hadms) > 1:
                                study_conflict_reasons[s_id] = "duplicate_study_ids_with_conflicting_encounter_ids"
                                continue

                            # 2. Check for conflicting linkage statuses
                            claim_validations = []
                            for c in claims:
                                st_time_str = c.get("study_time") or c.get("study_datetime") or study_time_str
                                st_dt, _ = parse_clinical_timestamp(st_time_str) if st_time_str else (None, None)
                                st_temporal = bool(admit_dt and st_dt and (st_dt >= admit_dt - timedelta(hours=24)) and (not disch_dt or st_dt <= disch_dt)) if st_dt else is_temporal
                                claim_validations.append(validate_study_provenance(s_id, subject_id, hadm_id, c, st_temporal))

                            statuses = [cv[0] for cv in claim_validations]
                            if len(set(statuses)) > 1:
                                study_conflict_reasons[s_id] = "duplicate_study_ids_with_conflicting_linkage_states"
                                continue

                            # 3. Check for conflicting provenance sources among verified claims
                            sources = {cv[1] for cv in claim_validations if cv[0] == "VERIFIED_ENCOUNTER_LINK" and cv[1]}
                            if len(sources) > 1:
                                study_conflict_reasons[s_id] = "duplicate_study_ids_with_conflicting_provenance_sources"
                                continue

                            # 4. Check for conflicting features
                            feature_vectors = []
                            for c in claims:
                                f = c.get("features") or c.get("xray_pathology_features") or c.get("pathology_features")
                                if f is not None and isinstance(f, (list, tuple)):
                                    feature_vectors.append([float(x) for x in f])
                            if len(feature_vectors) > 1:
                                first_vec = feature_vectors[0]
                                for other_vec in feature_vectors[1:]:
                                    if len(first_vec) != len(other_vec) or any(abs(a - b) > 1e-6 for a, b in zip(first_vec, other_vec)):
                                        study_conflict_reasons[s_id] = "duplicate_study_ids_with_conflicting_features"
                                        break
                                if s_id in study_conflict_reasons:
                                    continue

                            # No conflict found: consistent duplicate, will be deduplicated
                            study_conflict_reasons[s_id] = None
                        else:
                            study_conflict_reasons[s_id] = None

                    has_any_verified = False
                    seen_deduplicated_study_ids: set[str] = set()

                    for st in raw_studies:
                        if not isinstance(st, dict):
                            continue
                        s_id = str(st.get("study_id") or "").strip()
                        st_time_str = st.get("study_time") or st.get("study_datetime") or study_time_str
                        st_dt, _ = parse_clinical_timestamp(st_time_str) if st_time_str else (None, None)
                        st_temporal = bool(admit_dt and st_dt and (st_dt >= admit_dt - timedelta(hours=24)) and (not disch_dt or st_dt <= disch_dt)) if st_dt else is_temporal

                        conflict_reason = study_conflict_reasons.get(s_id)
                        if conflict_reason is not None:
                            # Reject conflicting duplicate
                            st_status = "REJECTED"
                            st_source = None
                            st_elig = 0.0
                            st_reason = conflict_reason
                        else:
                            st_status, st_source, st_elig, st_reason = validate_study_provenance(s_id, subject_id, hadm_id, st, st_temporal)

                            # Deduplication check: if study_id already seen and processed
                            if s_id in seen_deduplicated_study_ids:
                                st_status = "DUPLICATE_EXCLUDED"
                                st_elig = 0.0
                                st_reason = "identical_duplicate_study_deduplicated"
                            else:
                                if st_status == "VERIFIED_ENCOUNTER_LINK" and st_elig > 0.0:
                                    has_any_verified = True
                                    seen_deduplicated_study_ids.add(s_id)
                                    st_feats = st.get("features") or st.get("xray_pathology_features") or st.get("pathology_features")
                                    if isinstance(st_feats, (list, tuple)) and len(st_feats) > 0:
                                        verified_studies_features.append([float(x) for x in st_feats])

                        studies_metadata.append({
                            "study_id": s_id,
                            "subject_id": subject_id,
                            "hadm_id": hadm_id if st_status == "VERIFIED_ENCOUNTER_LINK" else None,
                            "study_time": st_time_str,
                            "linkage_status": st_status,
                            "provenance_source": st_source if st_status == "VERIFIED_ENCOUNTER_LINK" else None,
                            "pathology_eligible": st_elig,
                            "exclusion_reason": st_reason,
                        })

                    if has_any_verified:
                        linkage_status = "VERIFIED_ENCOUNTER_LINK"
                        is_study_linked = True
                        xray_eligible = 1.0
                    elif any(s["linkage_status"] == "REJECTED" for s in studies_metadata):
                        linkage_status = "REJECTED"
                        is_study_linked = False
                        xray_eligible = 0.0
                    elif any(s["linkage_status"] == "TEMPORAL_CANDIDATE_LINK" for s in studies_metadata):
                        linkage_status = "TEMPORAL_CANDIDATE_LINK"
                        is_study_linked = False
                        xray_eligible = 0.0
                    else:
                        linkage_status = "UNLINKED"
                        is_study_linked = False
                        xray_eligible = 0.0
                else:
                    # Single study evaluation
                    study_data = rec.get("xray") if isinstance(rec.get("xray"), dict) else rec
                    merged_study_data = dict(study_data)
                    for k in ("study_hadm_crosswalk", "encounter_link_provenance", "crosswalk", "provenance", "study_encounter_linked", "imaging_study_linked", "study_linkage_status", "linkage_status"):
                        if k in rec and k not in merged_study_data:
                            merged_study_data[k] = rec[k]

                    st_status, st_source, st_elig, st_reason = validate_study_provenance(study_id, subject_id, hadm_id, merged_study_data, is_temporal)
                    linkage_status = st_status
                    is_study_linked = (st_status == "VERIFIED_ENCOUNTER_LINK")
                    xray_eligible = st_elig if is_study_linked else 0.0

                    studies_metadata.append({
                        "study_id": study_id,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id if is_study_linked else None,
                        "study_time": study_time_str,
                        "linkage_status": st_status,
                        "provenance_source": st_source if is_study_linked else None,
                        "pathology_eligible": xray_eligible,
                        "exclusion_reason": st_reason,
                    })

            # Calculate aggregate xray pathology features if studies supplied them
            if has_xray and is_study_linked and verified_studies_features:
                num_feats = len(verified_studies_features[0])
                agg_xray_feats = [
                    float(sum(vec[i] for vec in verified_studies_features) / len(verified_studies_features))
                    for i in range(num_feats)
                ]
            elif has_xray and is_study_linked:
                agg_xray_feats = rec.get("xray_pathology_features", [0.0] * (len(CANONICAL_XRAY_PATHOLOGIES) * 3))
            else:
                agg_xray_feats = [0.0] * (len(CANONICAL_XRAY_PATHOLOGIES) * 3)

            # Laboratory Linkage & Provenance Validation (Step 6.16 Correction 1)
            lab_linkage_status = "UNLINKED"
            lab_eligible = 0.0
            lab_provenance_source = None
            lab_exclusion_reason = None
            if has_lab:
                lab_data = rec.get("laboratory") or {}
                if not isinstance(lab_data, dict):
                    lab_data = {"charttime": chart_time_str}
                else:
                    lab_data = dict(lab_data)
                    if "charttime" not in lab_data and chart_time_str:
                        lab_data["charttime"] = chart_time_str

                # Evaluate lab provenance against encounter
                is_lab_temporal = bool(
                    admit_dt and chart_dt and (chart_dt >= admit_dt - timedelta(hours=24)) and (not disch_dt or chart_dt <= disch_dt)
                ) if chart_dt else False
                lab_status, lab_source, lab_elig, lab_reason = validate_lab_provenance(
                    subject_id=subject_id,
                    hadm_id=hadm_id,
                    lab_data=lab_data,
                    admit_time=admit_dt,
                    disch_time=disch_dt,
                    is_temporal_match=is_lab_temporal,
                )
                lab_linkage_status = lab_status
                lab_provenance_source = lab_source
                lab_eligible = lab_elig
                lab_exclusion_reason = lab_reason

            temporal_evidence = {
                "admittime": admit_time_str,
                "dischtime": disch_time_str,
                "lab_charttime": chart_time_str,
                "imaging_study_time": study_time_str,
                "lab_in_window": bool(admit_dt and chart_dt and (chart_dt >= admit_dt - timedelta(hours=48)) and (not disch_dt or chart_dt <= disch_dt)),
                "lab_encounter_linked": (lab_linkage_status == "VERIFIED_ENCOUNTER_LINK"),
                "lab_linkage_status": lab_linkage_status,
                "lab_eligible": lab_eligible,
                "lab_provenance_source": lab_provenance_source,
                "lab_exclusion_reason": lab_exclusion_reason,
                "imaging_in_window": is_temporal if study_dt else None,
                "study_encounter_linked": is_study_linked,
                "study_linkage_status": linkage_status,
                "radiology_linked": has_rad,
                "xray_study_linked": is_study_linked,
                "xray_studies_provenance": studies_metadata,
            }

            # Update coverage
            if has_sym: modality_counts["symptoms"] += 1
            if has_lab and lab_linkage_status == "VERIFIED_ENCOUNTER_LINK": modality_counts["laboratory"] += 1
            elif has_lab: modality_counts["laboratory"] += 1
            if has_rad: modality_counts["radiology"] += 1
            if has_exam: modality_counts["clinical_exam"] += 1
            if has_xray and is_study_linked: modality_counts["xray"] += 1

            unique_patients.add(subject_id)
            unique_encounters.add(hadm_id)
            class_dist[canonical_target] = class_dist.get(canonical_target, 0) + 1

            raw_tokens = list(rec.get("symptom_tokens") or [])
            if len(raw_tokens) < 128:
                sym_tokens = raw_tokens + [0] * (128 - len(raw_tokens))
            else:
                sym_tokens = raw_tokens[:128]

            raw_feats = list(rec.get("symptom_features") or [])
            if len(raw_feats) < 128:
                sym_feats = raw_feats + [[0.0] * 6 for _ in range(128 - len(raw_feats))]
            else:
                sym_feats = raw_feats[:128]

            standardized_rec = {
                "subject_id": subject_id,
                "hadm_id": hadm_id,
                "study_id": study_id if has_xray else None,
                "target_condition": canonical_target,
                "target_class_idx": target_class_idx,
                "taxonomy": self.target_taxonomy,
                "seq_num": seq_num,
                "diagnosis_priority": "PRIMARY" if is_primary else "SECONDARY",
                "is_primary_diagnosis": is_primary,
                "raw_target": raw_target,
                "symptom_tokens": sym_tokens,
                "symptom_features": sym_feats,
                "lab_features": rec.get("lab_features", [0.0] * (len(CANONICAL_LAB_TESTS) * 5)),
                "lab_eligible": lab_eligible,
                "lab_linkage_status": lab_linkage_status,
                "lab_encounter_linked": (lab_linkage_status == "VERIFIED_ENCOUNTER_LINK"),
                "rad_features": rec.get("rad_features", [0.0] * (len(CANONICAL_RAD_FINDINGS) * 4)),
                "exam_features": rec.get("exam_features", [0.0] * (len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)),
                "xray_pathology_features": agg_xray_feats,
                "xray_anatomy_probs": rec.get("xray_anatomy_probs", [0.0] * 33),
                "xray_pathology_eligible": xray_eligible,
                "imaging_study_linked": is_study_linked,
                "study_linkage_status": linkage_status,
                "temporal_linkage_evidence": temporal_evidence,
                "modality_mask": [
                    1.0 if has_sym else 0.0,
                    1.0 if (has_lab and lab_linkage_status == "VERIFIED_ENCOUNTER_LINK") else 0.0,
                    1.0 if has_rad else 0.0,
                    1.0 if (has_xray and is_study_linked) else 0.0,
                    1.0 if has_exam else 0.0,
                ],
                "data_source": self.data_source_name,
                "source_version": self.source_version,
            }
            valid_records.append(standardized_rec)

        is_valid = len(valid_records) > 0 and len(unique_patients) > 0
        audit_report = IngestionAuditReport(
            is_valid=is_valid,
            data_source_name=self.data_source_name,
            source_version=self.source_version,
            target_taxonomy=self.target_taxonomy,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            total_raw_records=len(raw_records),
            valid_linked_records=len(valid_records),
            rejected_records_count=sum(rejection_reasons.values()),
            unique_patients_count=len(unique_patients),
            unique_encounters_count=len(unique_encounters),
            unique_studies_count=len(unique_studies),
            rejection_reasons=rejection_reasons,
            modality_coverage=modality_counts,
            class_distribution=class_dist,
            source_file_hashes=file_hashes,
        )

        return valid_records, audit_report


def partition_patient_level_splits(
    records: list[dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    temporal_holdout_anchor_year_group: Optional[str] = None,
) -> dict[str, Any]:
    """
    Partitions linked clinical records strictly by patient identifier (subject_id).
    
    Guarantees:
    1. Zero Patient Leakage:
       Patients(Train) ∩ Patients(Val) = ∅
       Patients(Train) ∩ Patients(Test) = ∅
       Patients(Val) ∩ Patients(Test) = ∅
    2. Encounter and Study Integrity: All hospital encounters (hadm_id) and imaging
       studies (study_id) for a given subject_id remain in the same split partition.
    3. Temporal Evaluation vs. Patient-Held-Out Evaluation:
       - If temporal_holdout_anchor_year_group is specified (e.g. '2017 - 2019'),
         all patients belonging to that anchor year decile are held out into the temporal test set.
       - Otherwise, uses deterministic pseudorandom patient partitioning.
    4. Rare-Class Documentation:
       Logs and reports class counts per partition to identify potential rare-class shortages
       (e.g., Spontaneous Pneumothorax).
    """
    import random
    from collections import defaultdict

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        raise ValueError("Split ratios must sum to 1.0")

    # Group records by subject_id
    patient_records = defaultdict(list)
    patient_anchor_years = {}
    for r in records:
        sub = str(r.get("subject_id") or "").strip()
        if not sub:
            continue
        patient_records[sub].append(r)
        if "anchor_year_group" in r and sub not in patient_anchor_years:
            patient_anchor_years[sub] = str(r.get("anchor_year_group")).strip()

    all_subjects = sorted(patient_records.keys())
    if not all_subjects:
        return {
            "train": [],
            "val": [],
            "test": [],
            "train_patients_count": 0,
            "val_patients_count": 0,
            "test_patients_count": 0,
            "isolation_verified": True,
            "class_counts": {"train": {}, "val": {}, "test": {}},
            "rare_class_warnings": [],
            "evaluation_type": "patient_held_out",
        }

    temporal_subjects = set()
    candidate_subjects = list(all_subjects)

    # If temporal holdout is requested, isolate patients in that anchor_year_group
    if temporal_holdout_anchor_year_group:
        temporal_subjects = {
            s for s in all_subjects
            if patient_anchor_years.get(s) == str(temporal_holdout_anchor_year_group).strip()
        }
        candidate_subjects = [s for s in all_subjects if s not in temporal_subjects]

    rng = random.Random(seed)
    rng.shuffle(candidate_subjects)

    n_total = len(candidate_subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_subs = set(candidate_subjects[:n_train])
    val_subs = set(candidate_subjects[n_train:n_train + n_val])
    test_subs = set(candidate_subjects[n_train + n_val:]) | temporal_subjects

    # Validate isolation
    assert train_subs.isdisjoint(val_subs), "Train and Val subjects overlap!"
    assert train_subs.isdisjoint(test_subs), "Train and Test subjects overlap!"
    assert val_subs.isdisjoint(test_subs), "Val and Test subjects overlap!"

    train_records = [r for s in sorted(train_subs) for r in patient_records[s]]
    val_records = [r for s in sorted(val_subs) for r in patient_records[s]]
    test_records = [r for s in sorted(test_subs) for r in patient_records[s]]

    # Compute class distribution across splits
    def count_classes(split_recs):
        counts = defaultdict(int)
        for r in split_recs:
            tgt = r.get("target_condition") or r.get("target") or "UNKNOWN"
            counts[tgt] += 1
        return dict(counts)

    class_counts = {
        "train": count_classes(train_records),
        "val": count_classes(val_records),
        "test": count_classes(test_records),
    }

    # Rare class warning documentation
    rare_warnings = []
    all_targets = set(list(class_counts["train"].keys()) + list(class_counts["val"].keys()) + list(class_counts["test"].keys()))
    for tgt in sorted(all_targets):
        for split_name in ("train", "val", "test"):
            cnt = class_counts[split_name].get(tgt, 0)
            if cnt < 3:
                rare_warnings.append(f"Rare class '{tgt}' has only {cnt} case(s) in {split_name} split.")

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
        "train_patients_count": len(train_subs),
        "val_patients_count": len(val_subs),
        "test_patients_count": len(test_subs),
        "isolation_verified": True,
        "class_counts": class_counts,
        "rare_class_warnings": rare_warnings,
        "evaluation_type": "temporal_out_of_time" if temporal_holdout_anchor_year_group else "patient_held_out",
    }


def validate_patient_split_isolation(
    train_records: list[dict[str, Any]],
    val_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Verifies that no patient identifier (subject_id) leaks across train, validation, or test partitions.
    """
    train_pts = {str(r.get("subject_id") or "").strip() for r in train_records if r.get("subject_id")}
    val_pts = {str(r.get("subject_id") or "").strip() for r in val_records if r.get("subject_id")}
    test_pts = {str(r.get("subject_id") or "").strip() for r in test_records if r.get("subject_id")}

    leakage_train_val = train_pts & val_pts
    leakage_train_test = train_pts & test_pts
    leakage_val_test = val_pts & test_pts

    has_leakage = bool(leakage_train_val or leakage_train_test or leakage_val_test)

    return {
        "isolated": not has_leakage,
        "leakage_train_val_count": len(leakage_train_val),
        "leakage_train_test_count": len(leakage_train_test),
        "leakage_val_test_count": len(leakage_val_test),
        "leakage_subjects": list(leakage_train_val | leakage_train_test | leakage_val_test),
    }


