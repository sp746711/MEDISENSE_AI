"""Multimodal Condition Prediction Architecture for MediSense AI.

A genuine multimodal neural network architecture designed to learn from multiple
patient modalities (symptoms, laboratory panels, radiology reports, eligible X-ray features,
and physical clinical examinations) with principled missing-modality masks and gated representation fusion.

CLINICAL SAFETY CONSTRAINTS:
1. Anatomy classification is NOT pathology classification. Anatomy is used solely for
   routing and anatomical context; only verified specialist pathology model outputs constitute
   disease-related features.
2. Missing modalities are explicitly masked out; missing information is NEVER imputed as normal.
3. Physical clinical examination findings ("localized tenderness", "reduced range of motion")
   are strictly separated from radiology report imaging features.
4. Multi-study imaging: All eligible radiographic studies are processed without arbitrary truncation;
   mismatched or unverified studies are never concealed by valid studies.
5. Laboratory preprocessing uses clinically validated reference parameters, unit conversions,
   and explicit presence indicators, eliminating arbitrary heuristic scaling.
6. Checkpoint validation requires declared acceptance gates, dataset provenance, and independent
   evaluation metrics before activating production multimodal prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("multimodal_model")

ARCHITECTURE_VERSION = "2.2.0"
FEATURE_SCHEMA_VERSION = "2.2.0"
LAB_SCHEMA_VERSION = "2.2.0"
SYMPTOM_VOCAB_VERSION = "2.2.0"

# The 24 authentic clinical condition classes preserved from baseline model
CONDITION_CLASSES: list[str] = [
    "Acne",
    "Arthritis",
    "Bronchial Asthma",
    "Cervical spondylosis",
    "Chicken pox",
    "Common Cold",
    "Dengue",
    "Dimorphic Hemorrhoids",
    "Fungal infection",
    "Hypertension",
    "Impetigo",
    "Jaundice",
    "Malaria",
    "Migraine",
    "Pneumonia",
    "Psoriasis",
    "Typhoid",
    "Varicose Veins",
    "allergy",
    "diabetes",
    "drug reaction",
    "gastroesophageal reflux disease",
    "peptic ulcer disease",
    "urinary tract infection",
]

# Supported canonical laboratory tests (21 tests aligned with report_nlp.py TARGET_TESTS)
CANONICAL_LAB_TESTS: list[str] = [
    "White Blood Cell Count (WBC)",
    "Neutrophils",
    "Lymphocytes",
    "C-Reactive Protein (CRP)",
    "Procalcitonin",
    "Hemoglobin",
    "Platelet Count",
    "Red Blood Cell Count (RBC)",
    "Hematocrit",
    "Fasting Blood Glucose",
    "Postprandial Blood Glucose",
    "HbA1c",
    "Serum Creatinine",
    "Blood Urea Nitrogen",
    "Uric Acid",
    "Total Bilirubin",
    "SGOT / AST",
    "SGPT / ALT",
    "Total Cholesterol",
    "Triglycerides",
    "TSH",
]

# Genuine qualitative radiology findings (14 canonical features per Section 5.3)
CANONICAL_RAD_FINDINGS: list[str] = [
    "consolidation",
    "infiltrate",
    "opacity",
    "air-space opacity",
    "pleural effusion",
    "pneumothorax",
    "cardiomegaly",
    "atelectasis",
    "fracture",
    "joint effusion",
    "edema",
    "nodule",
    "dislocation",
    "clear lungs",
]

# Separate clinical examination / physical observation category (NOT radiology features)
CANONICAL_CLINICAL_EXAM_FINDINGS: list[str] = [
    "localized tenderness",
    "reduced range of motion",
]

# Canonical specialist X-ray pathology findings (from TorchXRayVision and MURA)
CANONICAL_XRAY_PATHOLOGIES: list[str] = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Effusion",
    "Pneumonia",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Musculoskeletal Abnormality",
]

# Stable, curated vocabulary of canonical symptoms to prevent hash collisions (Problem E)
CANONICAL_SYMPTOM_VOCABULARY: list[str] = [
    "abdominal pain", "abnormal menstruation", "acidity", "acute liver failure",
    "altered sensorium", "anxiety", "back pain", "belly pain", "blackheads",
    "bladder discomfort", "blister", "blood in sputum", "bloody stool", "blurred and distorted vision",
    "breathlessness", "brittle nails", "bruising", "burning micturition", "chest pain",
    "chills", "cold hands and feets", "coma", "congestion", "constipation",
    "continuous feel of urine", "continuous sneezing", "cough", "cramps", "dark urine",
    "dehydration", "depression", "diarrhea", "dischromic patches", "distention of abdomen",
    "dizziness", "drying and tingling lips", "enlarged thyroid", "excessive hunger",
    "extra marital contacts", "fast heart rate", "fatigue", "fever", "foul smell of urine",
    "frequent urination", "headache", "high fever", "hip joint pain", "history of alcohol consumption",
    "increased appetite", "indigestion", "inflammatory nails", "internal itching",
    "irregular sugar level", "irritability", "irritation in anus", "itching", "joint pain",
    "knee pain", "lack of concentration", "lethargy", "loss of appetite", "loss of balance",
    "loss of smell", "malaise", "mild fever", "mood swings", "movement stiffness",
    "muscle pain", "muscle wasting", "muscle weakness", "nausea", "neck pain",
    "nodal skin eruptions", "obesity", "pain behind the eyes", "pain during bowel movements",
    "pain in anal region", "painful walking", "palpitations", "passage of gases",
    "patches in throat", "phlegm", "polyuria", "prominent veins on calf", "puffy face and eyes",
    "pus filled pimples", "rash", "receiving blood transfusion", "receiving unsterile injections",
    "red sore around nose", "red spots over body", "redness of eyes", "restlessness",
    "runny nose", "rusty sputum", "scurring", "shivering", "silver like dusting",
    "sinus pressure", "skin peeling", "skin rash", "slurred speech", "small dents in nails",
    "sneezing", "sore throat", "spinning movements", "spotting urination", "stiff neck",
    "stomach bleeding", "stomach pain", "sunken eyes", "sweating", "swelled lymph nodes",
    "swelling joints", "swelling of stomach", "swollen blood vessels", "swollen extremities",
    "swollen legs", "throat irritation", "toxic look (typhos)", "ulcers on tongue",
    "unsteadiness", "visual disturbances", "vomiting", "watering from eyes", "weakness in limbs",
    "weakness of one body side", "weight gain", "weight loss", "wheezing", "yellow crust ooze",
    "yellow urine", "yellowing of eyes", "yellowish skin"
]
_SYMPTOM_TOKEN_MAP = {s: i + 1 for i, s in enumerate(CANONICAL_SYMPTOM_VOCABULARY)}


# ==============================================================================
# LABORATORY PREPROCESSING & CANONICAL MAPPING CONTRACT (Problems B & C)
# ==============================================================================

# Clinical distribution scaling parameters: (canonical_unit, mean, std, ref_low_def, ref_high_def)
# Mean and std are derived from clinical laboratory standard reference populations
LAB_PREPROCESSING_SPECS: dict[str, dict[str, Any]] = {
    "White Blood Cell Count (WBC)": {
        "canonical_unit": "cells/mcL",
        "mean": 7500.0,
        "std": 2500.0,
        "ref_low": 4000.0,
        "ref_high": 11000.0,
        "unit_conversions": {
            "x10^9/l": 1000.0,
            "x10^3/mcl": 1000.0,
            "10^9/l": 1000.0,
            "k/ul": 1000.0,
            "/cumm": 1.0,
            "cells/cumm": 1.0,
            "cells/ul": 1.0,
        },
    },
    "Neutrophils": {
        "canonical_unit": "%",
        "mean": 58.0,
        "std": 10.0,
        "ref_low": 40.0,
        "ref_high": 75.0,
        "unit_conversions": {"%": 1.0},
    },
    "Lymphocytes": {
        "canonical_unit": "%",
        "mean": 30.0,
        "std": 8.0,
        "ref_low": 20.0,
        "ref_high": 45.0,
        "unit_conversions": {"%": 1.0},
    },
    "C-Reactive Protein (CRP)": {
        "canonical_unit": "mg/L",
        "mean": 3.0,
        "std": 6.0,
        "ref_low": 0.0,
        "ref_high": 5.0,
        "unit_conversions": {
            "mg/l": 1.0,
            "mg/dl": 10.0,
            "ug/ml": 1.0,
        },
    },
    "Procalcitonin": {
        "canonical_unit": "ng/mL",
        "mean": 0.15,
        "std": 0.5,
        "ref_low": 0.0,
        "ref_high": 0.5,
        "unit_conversions": {
            "ng/ml": 1.0,
            "ug/l": 1.0,
        },
    },
    "Hemoglobin": {
        "canonical_unit": "g/dL",
        "mean": 14.0,
        "std": 2.5,
        "ref_low": 12.0,
        "ref_high": 17.5,
        "unit_conversions": {
            "g/dl": 1.0,
            "g/l": 0.1,
            "gm/dl": 1.0,
        },
    },
    "Platelet Count": {
        "canonical_unit": "cells/mcL",
        "mean": 250000.0,
        "std": 75000.0,
        "ref_low": 150000.0,
        "ref_high": 450000.0,
        "unit_conversions": {
            "lakh": 100000.0,
            "lakh/cumm": 100000.0,
            "lakhs/cumm": 100000.0,
            "x10^9/l": 1000.0,
            "k/ul": 1000.0,
            "x10^3/mcl": 1000.0,
            "/cumm": 1.0,
            "cells/cumm": 1.0,
            "cells/ul": 1.0,
        },
    },
    "Red Blood Cell Count (RBC)": {
        "canonical_unit": "mil/mcL",
        "mean": 4.8,
        "std": 0.7,
        "ref_low": 4.0,
        "ref_high": 6.0,
        "unit_conversions": {
            "mil/mcl": 1.0,
            "million/cumm": 1.0,
            "x10^12/l": 1.0,
            "x10^6/ul": 1.0,
            "m/ul": 1.0,
        },
    },
    "Hematocrit": {
        "canonical_unit": "%",
        "mean": 42.0,
        "std": 5.0,
        "ref_low": 36.0,
        "ref_high": 50.0,
        "unit_conversions": {"%": 1.0},
    },
    "Fasting Blood Glucose": {
        "canonical_unit": "mg/dL",
        "mean": 95.0,
        "std": 25.0,
        "ref_low": 70.0,
        "ref_high": 100.0,
        "unit_conversions": {
            "mg/dl": 1.0,
            "mmol/l": 18.0182,
        },
    },
    "Postprandial Blood Glucose": {
        "canonical_unit": "mg/dL",
        "mean": 125.0,
        "std": 35.0,
        "ref_low": 90.0,
        "ref_high": 140.0,
        "unit_conversions": {
            "mg/dl": 1.0,
            "mmol/l": 18.0182,
        },
    },
    "HbA1c": {
        "canonical_unit": "%",
        "mean": 5.4,
        "std": 1.2,
        "ref_low": 4.0,
        "ref_high": 5.6,
        "unit_conversions": {
            "%": 1.0,
        },
    },
    "Serum Creatinine": {
        "canonical_unit": "mg/dL",
        "mean": 0.9,
        "std": 0.4,
        "ref_low": 0.6,
        "ref_high": 1.3,
        "unit_conversions": {
            "mg/dl": 1.0,
            "umol/l": 1.0 / 88.4,
            "µmol/l": 1.0 / 88.4,
        },
    },
    "Blood Urea Nitrogen": {
        "canonical_unit": "mg/dL",
        "mean": 14.0,
        "std": 6.0,
        "ref_low": 7.0,
        "ref_high": 20.0,
        "unit_conversions": {
            "mg/dl": 1.0,
            "mmol/l": 2.8,
        },
    },
    "Uric Acid": {
        "canonical_unit": "mg/dL",
        "mean": 5.2,
        "std": 1.5,
        "ref_low": 3.5,
        "ref_high": 7.2,
        "unit_conversions": {
            "mg/dl": 1.0,
            "umol/l": 1.0 / 59.48,
        },
    },
    "Total Bilirubin": {
        "canonical_unit": "mg/dL",
        "mean": 0.7,
        "std": 0.4,
        "ref_low": 0.2,
        "ref_high": 1.2,
        "unit_conversions": {
            "mg/dl": 1.0,
            "umol/l": 1.0 / 17.1,
        },
    },
    "SGOT / AST": {
        "canonical_unit": "U/L",
        "mean": 25.0,
        "std": 15.0,
        "ref_low": 10.0,
        "ref_high": 40.0,
        "unit_conversions": {"u/l": 1.0, "iu/l": 1.0},
    },
    "SGPT / ALT": {
        "canonical_unit": "U/L",
        "mean": 30.0,
        "std": 18.0,
        "ref_low": 7.0,
        "ref_high": 56.0,
        "unit_conversions": {"u/l": 1.0, "iu/l": 1.0},
    },
    "Total Cholesterol": {
        "canonical_unit": "mg/dL",
        "mean": 180.0,
        "std": 40.0,
        "ref_low": 125.0,
        "ref_high": 200.0,
        "unit_conversions": {
            "mg/dl": 1.0,
            "mmol/l": 38.67,
        },
    },
    "Triglycerides": {
        "canonical_unit": "mg/dL",
        "mean": 130.0,
        "std": 50.0,
        "ref_low": 50.0,
        "ref_high": 150.0,
        "unit_conversions": {
            "mg/dl": 1.0,
            "mmol/l": 88.57,
        },
    },
    "TSH": {
        "canonical_unit": "uIU/mL",
        "mean": 2.2,
        "std": 1.4,
        "ref_low": 0.4,
        "ref_high": 4.5,
        "unit_conversions": {
            "uiu/ml": 1.0,
            "miu/l": 1.0,
            "u/l": 1.0,
        },
    },
}

# Exhaustive alias lookup table mapping all report variations to single canonical names (Problem C)
CANONICAL_LAB_ALIASES: dict[str, str] = {
    # WBC
    "wbc": "White Blood Cell Count (WBC)",
    "white blood cell count": "White Blood Cell Count (WBC)",
    "white blood cell count (wbc)": "White Blood Cell Count (WBC)",
    "total leukocyte count": "White Blood Cell Count (WBC)",
    "total leukocyte count (wbc)": "White Blood Cell Count (WBC)",
    "tlc": "White Blood Cell Count (WBC)",
    "total wbc count": "White Blood Cell Count (WBC)",
    "total wbc": "White Blood Cell Count (WBC)",
    "leukocyte count": "White Blood Cell Count (WBC)",
    "white blood cells": "White Blood Cell Count (WBC)",
    # Neutrophils
    "neutrophils": "Neutrophils",
    "neutrophil": "Neutrophils",
    "neutrophils %": "Neutrophils",
    "polymorphs": "Neutrophils",
    # Lymphocytes
    "lymphocytes": "Lymphocytes",
    "lymphocyte": "Lymphocytes",
    "lymphocytes %": "Lymphocytes",
    # CRP
    "c-reactive protein (crp)": "C-Reactive Protein (CRP)",
    "c-reactive protein": "C-Reactive Protein (CRP)",
    "crp": "C-Reactive Protein (CRP)",
    "c reactive protein": "C-Reactive Protein (CRP)",
    # Procalcitonin
    "procalcitonin": "Procalcitonin",
    "procalcitonin (pct)": "Procalcitonin",
    "pct": "Procalcitonin",
    # Hemoglobin
    "hemoglobin": "Hemoglobin",
    "haemoglobin": "Hemoglobin",
    "hb": "Hemoglobin",
    "hgb": "Hemoglobin",
    # Platelets
    "platelet count": "Platelet Count",
    "platelets": "Platelet Count",
    "platelet": "Platelet Count",
    "total platelet count": "Platelet Count",
    "plt": "Platelet Count",
    # RBC
    "red blood cell count (rbc)": "Red Blood Cell Count (RBC)",
    "red blood cell count": "Red Blood Cell Count (RBC)",
    "rbc count": "Red Blood Cell Count (RBC)",
    "rbc": "Red Blood Cell Count (RBC)",
    "red blood cells": "Red Blood Cell Count (RBC)",
    "erythrocyte count": "Red Blood Cell Count (RBC)",
    # Hematocrit
    "hematocrit": "Hematocrit",
    "haematocrit": "Hematocrit",
    "hematocrit (pcv)": "Hematocrit",
    "packed cell volume": "Hematocrit",
    "pcv": "Hematocrit",
    "hct": "Hematocrit",
    # Fasting Glucose
    "fasting blood glucose": "Fasting Blood Glucose",
    "fasting blood sugar (fbs)": "Fasting Blood Glucose",
    "fasting blood sugar": "Fasting Blood Glucose",
    "fasting glucose": "Fasting Blood Glucose",
    "fbs": "Fasting Blood Glucose",
    "blood glucose (fasting)": "Fasting Blood Glucose",
    # Postprandial Glucose
    "postprandial blood glucose": "Postprandial Blood Glucose",
    "postprandial blood sugar (ppbs)": "Postprandial Blood Glucose",
    "postprandial blood sugar": "Postprandial Blood Glucose",
    "post prandial blood glucose": "Postprandial Blood Glucose",
    "postprandial glucose": "Postprandial Blood Glucose",
    "ppbs": "Postprandial Blood Glucose",
    # HbA1c
    "hba1c": "HbA1c",
    "glycated hemoglobin": "HbA1c",
    "hba1c (glycated hemoglobin)": "HbA1c",
    "glycated haemoglobin": "HbA1c",
    "glycosylated hemoglobin": "HbA1c",
    "a1c": "HbA1c",
    # Creatinine
    "serum creatinine": "Serum Creatinine",
    "creatinine": "Serum Creatinine",
    "s. creatinine": "Serum Creatinine",
    # BUN
    "blood urea nitrogen": "Blood Urea Nitrogen",
    "blood urea nitrogen (bun)": "Blood Urea Nitrogen",
    "bun": "Blood Urea Nitrogen",
    "urea nitrogen": "Blood Urea Nitrogen",
    # Uric Acid
    "uric acid": "Uric Acid",
    "serum uric acid": "Uric Acid",
    "s. uric acid": "Uric Acid",
    # Bilirubin
    "total bilirubin": "Total Bilirubin",
    "bilirubin total": "Total Bilirubin",
    "serum bilirubin total": "Total Bilirubin",
    "s. bilirubin": "Total Bilirubin",
    # SGOT
    "sgot / ast": "SGOT / AST",
    "sgot (ast)": "SGOT / AST",
    "sgot": "SGOT / AST",
    "ast": "SGOT / AST",
    "aspartate aminotransferase": "SGOT / AST",
    # SGPT
    "sgpt / alt": "SGPT / ALT",
    "sgpt (alt)": "SGPT / ALT",
    "sgpt": "SGPT / ALT",
    "alt": "SGPT / ALT",
    "alanine aminotransferase": "SGPT / ALT",
    # Cholesterol
    "total cholesterol": "Total Cholesterol",
    "cholesterol total": "Total Cholesterol",
    "serum cholesterol": "Total Cholesterol",
    # Triglycerides
    "triglycerides": "Triglycerides",
    "serum triglycerides": "Triglycerides",
    "tgl": "Triglycerides",
    # TSH
    "tsh": "TSH",
    "thyroid stimulating hormone": "TSH",
    "tsh, serum": "TSH",
    "serum tsh": "TSH",
}


def resolve_canonical_lab_name(raw_name: str) -> Optional[str]:
    """Map any lab test name or alias to its single canonical counterpart."""
    if not raw_name or not isinstance(raw_name, str):
        return None
    # Strip punctuation, normalize whitespace and lower case
    clean = re.sub(r"[\s\t\n]+", " ", raw_name).strip().lower()
    clean_no_punct = re.sub(r"[,\-_/]+", " ", clean)
    clean_no_punct = re.sub(r"\s+", " ", clean_no_punct).strip()

    if clean in CANONICAL_LAB_ALIASES:
        return CANONICAL_LAB_ALIASES[clean]
    if clean_no_punct in CANONICAL_LAB_ALIASES:
        return CANONICAL_LAB_ALIASES[clean_no_punct]

    # Partial substring check against canonical tests
    for canon in CANONICAL_LAB_TESTS:
        if canon.lower() == clean or canon.lower() == clean_no_punct:
            return canon

    return None


def normalize_lab_value(
    canonical_test: str,
    raw_value: float,
    raw_unit: Optional[str] = None,
    ref_low: Optional[float] = None,
    ref_high: Optional[float] = None,
    flag: Optional[str] = "NORMAL",
) -> tuple[float, float, float, float, float]:
    """
    Standardize lab value and reference limits to canonical units and z-score representation.
    Returns:
        (val_norm, ref_low_norm, ref_high_norm, flag_code, is_present)
    """
    spec = LAB_PREPROCESSING_SPECS.get(canonical_test)
    if not spec:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    unit_clean = str(raw_unit or "").strip().lower()
    conversion_factor = 1.0
    for u_pattern, mult in spec["unit_conversions"].items():
        if u_pattern in unit_clean:
            conversion_factor = mult
            break

    converted_val = raw_value * conversion_factor
    mean = spec["mean"]
    std = spec["std"]

    val_norm = (converted_val - mean) / std

    # Normalize reference limits
    r_low_conv = (ref_low * conversion_factor) if ref_low is not None else spec["ref_low"]
    r_high_conv = (ref_high * conversion_factor) if ref_high is not None else spec["ref_high"]
    ref_low_norm = (r_low_conv - mean) / std
    ref_high_norm = (r_high_conv - mean) / std

    # Flag code: HIGH=+1.0, LOW=-1.0, ABNORMAL=+0.5, CONFLICTING=+0.75, NORMAL=0.0
    f_str = str(flag or "NORMAL").upper()
    if f_str == "HIGH":
        flag_code = 1.0
    elif f_str == "LOW":
        flag_code = -1.0
    elif f_str == "CONFLICTING":
        flag_code = 0.75
    elif f_str in {"ABNORMAL", "POSITIVE"}:
        flag_code = 0.5
    else:
        flag_code = 0.0

    return val_norm, ref_low_norm, ref_high_norm, flag_code, 1.0


# ==============================================================================
# DATA CONTRACT & SCHEMA DEFINITIONS
# ==============================================================================

@dataclass
class ClinicalSymptomObservation:
    symptom: str
    canonical: str
    state: str = "PRESENT"  # PRESENT, ABSENT, UNKNOWN, HISTORICAL, NOT_ASKED
    duration_days: Optional[float] = None
    severity: str = "UNKNOWN"  # MILD, MODERATE, SEVERE, UNKNOWN
    qualifiers: list[str] = field(default_factory=list)
    body_area: Optional[str] = None
    follow_up_answer: Optional[str] = None


@dataclass
class StructuredLabParameter:
    test_name: str
    canonical_test_name: str
    value: float
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    flag: str = "NORMAL"  # HIGH, LOW, NORMAL, ABNORMAL, CONFLICTING, UNKNOWN
    source_file: Optional[str] = None
    report_id: Optional[str] = None


@dataclass
class StructuredRadiologyFinding:
    finding: str
    canonical_finding: str
    state: str = "PRESENT"  # PRESENT, ABSENT, POSSIBLE, UNCERTAIN, NOT_ASSESSED, EXTRACTION_FAILED, UNSUPPORTED, CONFLICTING
    anatomical_region: str = "chest"
    examination: Optional[str] = None
    source_file: Optional[str] = None
    report_id: Optional[str] = None
    has_conflict: bool = False


@dataclass
class StructuredClinicalFinding:
    finding: str
    canonical_finding: str
    category: str = "examination"  # examination, narrative
    state: str = "PRESENT"  # PRESENT, ABSENT, UNCERTAIN, CONFLICTING
    source_file: Optional[str] = None
    report_id: Optional[str] = None


@dataclass
class XRaySpecialistOutput:
    file_name: str
    detected_anatomy: str
    anatomy_verified: bool = False
    specialist_model_executed: bool = False
    has_region_mismatch: bool = False
    is_pathology_eligible: bool = False
    pathology_scores: dict[str, float] = field(default_factory=dict)
    anatomy_probs: Optional[list[float]] = None
    confidence: Optional[float] = None
    status: str = "PENDING_VERIFICATION"
    specialist_model_name: Optional[str] = None


@dataclass
class MultimodalPatientRecord:
    """Canonical multimodal EHR patient record representation."""
    subject_id: str
    symptoms: list[ClinicalSymptomObservation] = field(default_factory=list)
    laboratory: list[StructuredLabParameter] = field(default_factory=list)
    radiology: list[StructuredRadiologyFinding] = field(default_factory=list)
    clinical_examination: list[StructuredClinicalFinding] = field(default_factory=list)
    xray: Optional[XRaySpecialistOutput] = None
    xray_studies: list[XRaySpecialistOutput] = field(default_factory=list)
    target_condition: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# MODALITY ENCODERS (Problems A, B, C, D, E, F)
# ==============================================================================

class SymptomModalityEncoder(nn.Module):
    """Encodes clinical symptom observations (presence, absence, duration, severity, qualifiers)."""

    def __init__(self, vocab_size: int = 512, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.symptom_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Observation features: [state_code, duration_norm, severity_code, has_qualifiers, has_followup, is_active] (6 dims)
        self.dense_proj = nn.Linear(embed_dim + 6, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, token_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        embeds = self.symptom_embeddings(token_ids)
        combined = torch.cat([embeds, features], dim=-1)
        h = F.relu(self.dense_proj(combined))
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return self.layer_norm(pooled)


class LaboratoryModalityEncoder(nn.Module):
    """Encodes 21 structured laboratory panels with normalized values, reference limits, and presence flags."""

    def __init__(self, num_tests: int = len(CANONICAL_LAB_TESTS), out_dim: int = 64):
        super().__init__()
        # 5 features per test: [val_norm, ref_low_norm, ref_high_norm, flag_code, is_present]
        in_dim = num_tests * 5
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, lab_tensor: torch.Tensor) -> torch.Tensor:
        if lab_tensor.dim() == 3:
            lab_tensor = lab_tensor.view(lab_tensor.size(0), -1)
        return self.net(lab_tensor)


class RadiologyModalityEncoder(nn.Module):
    """Encodes structured qualitative findings from radiology reports with conflict preservation."""

    def __init__(self, num_findings: int = len(CANONICAL_RAD_FINDINGS), out_dim: int = 64):
        super().__init__()
        # 4 features per finding: [state_code, is_reported, certainty_code, conflict_flag]
        in_dim = num_findings * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, rad_tensor: torch.Tensor) -> torch.Tensor:
        if rad_tensor.dim() == 3:
            rad_tensor = rad_tensor.view(rad_tensor.size(0), -1)
        return self.net(rad_tensor)


class ClinicalExamModalityEncoder(nn.Module):
    """Encodes physical clinical examination and observation findings separately from radiology (Problem A)."""

    def __init__(self, num_findings: int = len(CANONICAL_CLINICAL_EXAM_FINDINGS), out_dim: int = 64):
        super().__init__()
        # 3 features per exam finding: [state_code, is_reported, certainty_code]
        in_dim = num_findings * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, exam_tensor: torch.Tensor) -> torch.Tensor:
        if exam_tensor.dim() == 3:
            exam_tensor = exam_tensor.view(exam_tensor.size(0), -1)
        return self.net(exam_tensor)


class XRayModalityAdapter(nn.Module):
    """
    Adapts eligible verified outputs from the protected X-ray system into latent fusion features.
    
    SAFETY RULE: Separates anatomical context from pathology findings.
    Anatomy classification (bodypart) is NEVER treated as pathology. Only specialist model
    pathology outputs (MURA or TorchXRayVision) are eligible as disease-related features.
    If pathology is ineligible, representation is strictly masked out.
    """

    def __init__(
        self,
        num_anatomy_classes: int = 33,
        num_pathologies: int = len(CANONICAL_XRAY_PATHOLOGIES),
        out_dim: int = 64,
    ):
        super().__init__()
        # Pathology feature vector: [score, is_assessed, is_detected] per canonical pathology
        pathology_in_dim = num_pathologies * 3
        self.pathology_net = nn.Sequential(
            nn.Linear(pathology_in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )
        # Anatomical context projection
        self.anatomy_context_net = nn.Sequential(
            nn.Linear(num_anatomy_classes, out_dim // 2),
            nn.ReLU(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(out_dim + (out_dim // 2), out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        pathology_tensor: torch.Tensor,
        anatomy_probs: torch.Tensor,
        is_pathology_eligible: torch.Tensor,
    ) -> torch.Tensor:
        h_path = self.pathology_net(pathology_tensor)
        h_anat = self.anatomy_context_net(anatomy_probs)
        combined = torch.cat([h_path, h_anat], dim=-1)
        h = self.fusion_proj(combined)
        # Strictly zero out X-ray representation if pathology inference is unverified,
        # region-mismatched, or unavailable. Anatomy alone CANNOT predict condition!
        return h * is_pathology_eligible


# ==============================================================================
# GATED MULTIMODAL FUSION
# ==============================================================================

class GatedMultimodalFusion(nn.Module):
    """Gated Multimodal Unit (GMU) fusing up to 5 modalities using explicit missingness masks."""

    def __init__(self, hidden_dim: int = 64, num_modalities: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.gate_layers = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_modalities)
        ])
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        modality_embeds: list[torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        modality_embeds: List of N tensors, each (B, hidden_dim) [sym, lab, rad, xray, exam]
        modality_mask: (B, N) float tensor (1.0 = available, 0.0 = omitted/unavailable)
        """
        batch_size = modality_embeds[0].size(0)
        n_mods = len(modality_embeds)

        # Harmonize mask width with embeds count
        if modality_mask.size(1) < n_mods:
            pad = torch.zeros(batch_size, n_mods - modality_mask.size(1), device=modality_mask.device)
            modality_mask = torch.cat([modality_mask, pad], dim=1)
        elif modality_mask.size(1) > n_mods:
            modality_mask = modality_mask[:, :n_mods]

        raw_gates = []
        for i in range(n_mods):
            gate = self.gate_layers[i](modality_embeds[i])
            raw_gates.append(gate)
        all_gates = torch.cat(raw_gates, dim=1)  # (B, N)

        # Apply modality mask
        masked_gates = torch.sigmoid(all_gates) * modality_mask  # (B, N)
        sum_gates = masked_gates.sum(dim=1, keepdim=True)
        is_evidence_present = (sum_gates > 1e-5).float()

        normalized_gates = masked_gates / torch.clamp(sum_gates, min=1e-6)
        stacked_embeds = torch.stack(modality_embeds, dim=1)  # (B, N, hidden_dim)
        fused = (stacked_embeds * normalized_gates.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        out = self.fusion_proj(fused)
        return out * is_evidence_present, is_evidence_present


# ==============================================================================
# COMPLETE PREDICTOR NEURAL ARCHITECTURE
# ==============================================================================

class MultimodalConditionPredictor(nn.Module):
    """Complete 5-modality neural model for condition differential candidate prediction."""

    def __init__(
        self,
        num_classes: int = len(CONDITION_CLASSES),
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.symptom_encoder = SymptomModalityEncoder(vocab_size=512, embed_dim=hidden_dim, out_dim=hidden_dim)
        self.lab_encoder = LaboratoryModalityEncoder(num_tests=len(CANONICAL_LAB_TESTS), out_dim=hidden_dim)
        self.rad_encoder = RadiologyModalityEncoder(num_findings=len(CANONICAL_RAD_FINDINGS), out_dim=hidden_dim)
        self.exam_encoder = ClinicalExamModalityEncoder(num_findings=len(CANONICAL_CLINICAL_EXAM_FINDINGS), out_dim=hidden_dim)
        self.xray_adapter = XRayModalityAdapter(
            num_anatomy_classes=33,
            num_pathologies=len(CANONICAL_XRAY_PATHOLOGIES),
            out_dim=hidden_dim,
        )

        self.fusion = GatedMultimodalFusion(hidden_dim=hidden_dim, num_modalities=5)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        symptom_tokens: torch.Tensor,
        symptom_features: torch.Tensor,
        lab_features: torch.Tensor,
        rad_features: torch.Tensor,
        xray_pathology_features: torch.Tensor,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Flexible forward pass supporting 4-modality legacy calls and 5-modality calls:
        Supported signatures:
        1. 5-modality standard: (tokens, sym_feat, lab, rad, xray_path, exam_feat, xray_anat, xray_elig, mask)
        2. 4-modality 8-arg: (tokens, sym_feat, lab, rad, xray_path, xray_anat, xray_elig, mask)
        3. 4-modality 7-arg: (tokens, sym_feat, lab, rad, xray_path, xray_elig, mask)
        """
        batch_size = symptom_tokens.size(0)
        device = symptom_tokens.device

        # Adapt positional arguments
        exam_features: Optional[torch.Tensor] = None
        xray_anatomy_probs: Optional[torch.Tensor] = None
        xray_pathology_eligible: Optional[torch.Tensor] = None
        modality_mask: Optional[torch.Tensor] = None

        if len(args) == 4:
            # 5-modality full call
            exam_features, xray_anatomy_probs, xray_pathology_eligible, modality_mask = args
        elif len(args) == 3:
            # 8-arg legacy call: arg6=xray_anat, arg7=xray_elig, arg8=mask
            xray_anatomy_probs, xray_pathology_eligible, modality_mask = args
        elif len(args) == 2:
            # 7-arg legacy call: arg6=xray_elig, arg7=mask
            xray_pathology_eligible, modality_mask = args
            xray_anatomy_probs = torch.zeros(batch_size, 33, device=device)
        else:
            xray_pathology_eligible = kwargs.get("xray_pathology_eligible", torch.zeros(batch_size, 1, device=device))
            modality_mask = kwargs.get("modality_mask", torch.zeros(batch_size, 5, device=device))
            xray_anatomy_probs = kwargs.get("xray_anatomy_probs", torch.zeros(batch_size, 33, device=device))
            exam_features = kwargs.get("exam_features")

        if exam_features is None:
            exam_features = torch.zeros(batch_size, len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3, device=device)
        if xray_anatomy_probs is None:
            xray_anatomy_probs = torch.zeros(batch_size, 33, device=device)

        # Standardize rad_features dimension (adapt from legacy 42 to current 56 if necessary)
        expected_rad_dim = len(CANONICAL_RAD_FINDINGS) * 4
        if rad_features.size(-1) < expected_rad_dim:
            pad_len = expected_rad_dim - rad_features.size(-1)
            rad_features = torch.cat([rad_features, torch.zeros(batch_size, pad_len, device=device)], dim=-1)

        # Standardize symptom_features dimension (adapt from legacy 3 to current 6 if necessary)
        if symptom_features.size(-1) < 6:
            pad_s = torch.zeros(batch_size, symptom_features.size(1), 6 - symptom_features.size(-1), device=device)
            symptom_features = torch.cat([symptom_features, pad_s], dim=-1)

        # Encoders
        h_sym = self.symptom_encoder(symptom_tokens, symptom_features)
        h_lab = self.lab_encoder(lab_features)
        h_rad = self.rad_encoder(rad_features)
        h_exam = self.exam_encoder(exam_features)
        h_xray = self.xray_adapter(xray_pathology_features, xray_anatomy_probs, xray_pathology_eligible)

        # Modality mask handling (expand 4-vector to 5-vector if exam mask omitted)
        if modality_mask.size(1) == 4:
            m_exam = (exam_features.abs().sum(dim=-1, keepdim=True) > 1e-5).float()
            modality_mask = torch.cat([modality_mask, m_exam], dim=1)

        modality_embeds = [h_sym, h_lab, h_rad, h_xray, h_exam]
        h_fused, is_evidence_present = self.fusion(modality_embeds, modality_mask)
        logits = self.classifier(h_fused)
        return logits * is_evidence_present, is_evidence_present


# ==============================================================================
# FEATURE EXTRACTOR (BRIDGING RAW ASSESSMENT INPUTS TO MODEL TENSORS)
# ==============================================================================

class MultimodalFeatureExtractor:
    """Transforms raw patient observations, lab reports, radiology findings, and X-ray data into tensor representations."""

    @staticmethod
    def _tokenize_symptom(text: str) -> int:
        clean = text.strip().lower()
        if clean in _SYMPTOM_TOKEN_MAP:
            return _SYMPTOM_TOKEN_MAP[clean]
        # Clean fallback for variations (deterministic in range 250..510)
        h = 250
        for c in clean:
            h = (h * 31 + ord(c)) % 260
        return 250 + h

    @classmethod
    def extract_tensors_from_record(
        cls,
        record: MultimodalPatientRecord,
        max_symptoms: int = 32,
    ) -> dict[str, torch.Tensor]:
        """Convert a MultimodalPatientRecord into verified model input tensors."""
        # 1. Symptoms (Problem E: vocabulary-based, up to 32 symptoms, no loss of qualifiers/followups)
        tokens = [0] * max_symptoms
        sym_feat = [[0.0] * 6 for _ in range(max_symptoms)]
        for i, sym in enumerate(record.symptoms[:max_symptoms]):
            tokens[i] = cls._tokenize_symptom(sym.canonical or sym.symptom)
            st = str(sym.state).upper()
            if st == "PRESENT":
                st_val = 1.0
            elif st == "ABSENT":
                st_val = -1.0
            elif st == "HISTORICAL":
                st_val = 0.5
            elif st == "NOT_ASKED":
                st_val = -0.5
            else:
                st_val = 0.0

            dur = float(sym.duration_days or 0.0)
            dur_norm = math.log1p(max(0.0, dur)) / 5.0
            dur_clamped = min(1.0, dur_norm)

            sev_map = {"MILD": 0.33, "MODERATE": 0.66, "SEVERE": 1.0}
            sev_val = sev_map.get(str(sym.severity).upper(), 0.0)

            has_qual = 1.0 if (sym.qualifiers and len(sym.qualifiers) > 0) else 0.0
            has_followup = 1.0 if bool(sym.follow_up_answer) else 0.0
            sym_feat[i] = [st_val, dur_clamped, sev_val, has_qual, has_followup, 1.0]

        symptom_tokens_t = torch.tensor([tokens], dtype=torch.long)
        symptom_features_t = torch.tensor([sym_feat], dtype=torch.float32)

        # 2. Laboratory (Problem B & C: Shared preprocessing, unit conversion, z-score scaling, explicit is_present)
        lab_features = [0.0] * (len(CANONICAL_LAB_TESTS) * 5)
        # Build lookup table resolving aliases
        lab_lookup: dict[str, StructuredLabParameter] = {}
        for l_item in record.laboratory:
            canon = resolve_canonical_lab_name(l_item.canonical_test_name or l_item.test_name)
            if canon:
                lab_lookup[canon] = l_item

        for idx, test_name in enumerate(CANONICAL_LAB_TESTS):
            base = idx * 5
            l_item = lab_lookup.get(test_name)
            if l_item:
                val_n, r_low_n, r_high_n, flag_c, is_pres = normalize_lab_value(
                    canonical_test=test_name,
                    raw_value=float(l_item.value),
                    raw_unit=l_item.unit,
                    ref_low=l_item.reference_low,
                    ref_high=l_item.reference_high,
                    flag=l_item.flag,
                )
                lab_features[base] = val_n
                lab_features[base + 1] = r_low_n
                lab_features[base + 2] = r_high_n
                lab_features[base + 3] = flag_c
                lab_features[base + 4] = is_pres
            else:
                # Explicit missingness: all 0.0, is_present=0.0 (Never normal!)
                lab_features[base + 4] = 0.0

        lab_features_t = torch.tensor([lab_features], dtype=torch.float32)

        # 3. Radiology (Problem F: 14 genuine findings, explicit presence/absence/uncertainty/conflict)
        # 4 features per finding: [state_code, is_reported, certainty_code, conflict_flag]
        rad_features = [0.0] * (len(CANONICAL_RAD_FINDINGS) * 4)
        rad_lookup = {r.canonical_finding.lower(): r for r in record.radiology}

        # Check for global conflict: "clear lungs" co-occurring with positive lung pathology
        clear_lungs_item = rad_lookup.get("clear lungs")
        has_clear_lungs = bool(clear_lungs_item and str(clear_lungs_item.state).upper() in {"PRESENT", "POSITIVE", "CONFIRMED"})

        for idx, finding_name in enumerate(CANONICAL_RAD_FINDINGS):
            base = idx * 4
            r_item = rad_lookup.get(finding_name.lower())
            if r_item:
                st = str(r_item.state).upper()
                conflict_flag = 1.0 if getattr(r_item, "has_conflict", False) else 0.0

                if st in {"PRESENT", "POSITIVE", "CONFIRMED"}:
                    st_code = 1.0
                    is_rep = 1.0
                    cert_code = 1.0
                    if has_clear_lungs and finding_name in {"consolidation", "infiltrate", "opacity", "air-space opacity", "pleural effusion", "pneumothorax"}:
                        conflict_flag = 1.0
                elif st in {"ABSENT", "NEGATIVE", "NONE"}:
                    st_code = -1.0
                    is_rep = 1.0
                    cert_code = 1.0
                elif st in {"POSSIBLE", "UNCERTAIN", "EQUIVOCAL", "INDETERMINATE"}:
                    st_code = 0.0
                    is_rep = 1.0
                    cert_code = 0.5
                elif st in {"CONFLICTING", "CONFLICT"}:
                    st_code = 0.0
                    is_rep = 1.0
                    cert_code = 0.25
                    conflict_flag = 1.0
                elif st in {"NOT_ASSESSED", "NOT ASSESSED", "UNASSESSED", "UNSUPPORTED", "EXTRACTION_FAILED"}:
                    st_code = 0.0
                    is_rep = 0.0
                    cert_code = 0.0
                else:
                    st_code = 0.0
                    is_rep = 1.0
                    cert_code = 0.5

                rad_features[base] = st_code
                rad_features[base + 1] = is_rep
                rad_features[base + 2] = cert_code
                rad_features[base + 3] = conflict_flag

        rad_features_t = torch.tensor([rad_features], dtype=torch.float32)

        # 4. Clinical Examination (Problem A: Physical observations encoded separately)
        # 3 features per finding: [state_code, is_reported, certainty_code]
        exam_features = [0.0] * (len(CANONICAL_CLINICAL_EXAM_FINDINGS) * 3)
        exam_lookup = {e.canonical_finding.lower(): e for e in record.clinical_examination}
        for idx, ex_name in enumerate(CANONICAL_CLINICAL_EXAM_FINDINGS):
            base = idx * 3
            e_item = exam_lookup.get(ex_name.lower())
            if e_item:
                st = str(e_item.state).upper()
                if st in {"PRESENT", "POSITIVE", "CONFIRMED"}:
                    st_code = 1.0
                    cert_code = 1.0
                elif st in {"ABSENT", "NEGATIVE"}:
                    st_code = -1.0
                    cert_code = 1.0
                else:
                    st_code = 0.0
                    cert_code = 0.5
                exam_features[base] = st_code
                exam_features[base + 1] = 1.0
                exam_features[base + 2] = cert_code

        exam_features_t = torch.tensor([exam_features], dtype=torch.float32)

        # 5. X-Ray Pathology & Multi-Study Aggregation (Problems D & G)
        # 3 features per pathology: [score, is_assessed, is_detected]
        xray_pathology = [0.0] * (len(CANONICAL_XRAY_PATHOLOGIES) * 3)
        anatomy_probs = [0.0] * 33
        is_xray_eligible = 0.0

        all_studies = list(record.xray_studies)
        if record.xray and record.xray not in all_studies:
            all_studies.append(record.xray)

        eligible_studies = [
            s for s in all_studies
            if s.is_pathology_eligible and not s.has_region_mismatch and s.status.upper() == "COMPLETED"
        ]

        if eligible_studies:
            is_xray_eligible = 1.0
            # Multi-image max-pooling across eligible studies
            for idx, p_name in enumerate(CANONICAL_XRAY_PATHOLOGIES):
                base = idx * 3
                matching_scores = [
                    s.pathology_scores[p_name]
                    for s in eligible_studies
                    if p_name in (s.pathology_scores or {})
                ]
                if matching_scores:
                    max_score = float(max(matching_scores))
                    is_det = 1.0 if max_score >= 0.5 else 0.0
                    xray_pathology[base] = max_score
                    xray_pathology[base + 1] = 1.0  # is_assessed
                    xray_pathology[base + 2] = is_det
                else:
                    # Not assessed by the specialist model(s) that ran for this anatomical region
                    xray_pathology[base + 1] = 0.0

            # Anatomical context from primary eligible study
            primary_study = eligible_studies[0]
            if primary_study.anatomy_probs and len(primary_study.anatomy_probs) == 33:
                anatomy_probs = [float(p) for p in primary_study.anatomy_probs]

        xray_pathology_t = torch.tensor([xray_pathology], dtype=torch.float32)
        xray_anatomy_t = torch.tensor([anatomy_probs], dtype=torch.float32)
        xray_eligibility_t = torch.tensor([[is_xray_eligible]], dtype=torch.float32)

        # Modality mask: [m_sym, m_lab, m_rad, m_xray, m_exam] (5 dimensions)
        m_sym = 1.0 if len(record.symptoms) > 0 else 0.0
        m_lab = 1.0 if len(record.laboratory) > 0 else 0.0
        m_rad = 1.0 if len(record.radiology) > 0 else 0.0
        m_xray = 1.0 if is_xray_eligible > 0.0 else 0.0
        m_exam = 1.0 if len(record.clinical_examination) > 0 else 0.0
        modality_mask_t = torch.tensor([[m_sym, m_lab, m_rad, m_xray, m_exam]], dtype=torch.float32)

        return {
            "symptom_tokens": symptom_tokens_t,
            "symptom_features": symptom_features_t,
            "lab_features": lab_features_t,
            "rad_features": rad_features_t,
            "exam_features": exam_features_t,
            "xray_pathology_features": xray_pathology_t,
            "xray_anatomy_probs": xray_anatomy_t,
            "xray_pathology_eligible": xray_eligibility_t,
            "modality_mask": modality_mask_t,
        }


# ==============================================================================
# STRICT CHECKPOINT VALIDATION (Problem J)
# ==============================================================================

def validate_multimodal_checkpoint(ckpt: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Strict validation gate for production multimodal model checkpoints.
    Validates architecture, feature schemas, acceptance pass, provenance, and evaluation reports.
    """
    reasons: list[str] = []
    if not isinstance(ckpt, dict):
        return False, ["Checkpoint payload is not a valid dictionary."]

    if "model_state" not in ckpt:
        reasons.append("Missing 'model_state' in checkpoint.")

    # 1. Architecture & Schema version checks
    if ckpt.get("architecture_version") != ARCHITECTURE_VERSION:
        reasons.append(f"Architecture version mismatch: expected {ARCHITECTURE_VERSION}, got {ckpt.get('architecture_version')}.")
    if ckpt.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        reasons.append(f"Feature schema version mismatch: expected {FEATURE_SCHEMA_VERSION}, got {ckpt.get('feature_schema_version')}.")

    # 2. Condition classes parity
    classes = ckpt.get("condition_classes")
    if classes != CONDITION_CLASSES:
        reasons.append("Condition classes mismatch with canonical 24 clinical classes.")

    # 3. Production acceptance status
    if ckpt.get("status") != "PRODUCTION_APPROVED":
        reasons.append(f"Checkpoint status '{ckpt.get('status')}' is not 'PRODUCTION_APPROVED'. Experimental or unapproved weights rejected.")

    if not ckpt.get("acceptance_passed"):
        reasons.append("Checkpoint acceptance_passed flag is not True.")

    # 4. Provenance & Evaluation reports
    provenance = ckpt.get("dataset_provenance", {})
    if not isinstance(provenance, dict) or not provenance.get("is_authentic"):
        reasons.append("Dataset provenance is unverified or marked non-authentic.")

    if not ckpt.get("test_evaluation_report"):
        reasons.append("Missing independent test evaluation report.")

    is_valid = len(reasons) == 0
    return is_valid, reasons


# ==============================================================================
# HIGH-LEVEL INFERENCE & VALIDATION SERVICE
# ==============================================================================

class MultimodalInferenceService:
    """Production service wrapper coordinating multimodal prediction with clinical safety."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path
        self.model: Optional[MultimodalConditionPredictor] = None
        self.is_trained: bool = False
        self.model_version = "multimodal_gmu_v2.2-unvalidated"
        self.checkpoint_metadata: dict[str, Any] = {}

        if checkpoint_path and checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                is_valid, rejection_reasons = validate_multimodal_checkpoint(ckpt)
                if is_valid:
                    model = MultimodalConditionPredictor()
                    model.load_state_dict(ckpt["model_state"])
                    model.eval()
                    self.model = model
                    self.is_trained = True
                    self.model_version = ckpt.get("version", self.model_version)
                    self.checkpoint_metadata = ckpt
                else:
                    logger.warning(
                        "Multimodal checkpoint at %s failed strict acceptance gate: %s. Using safe baseline fallback.",
                        checkpoint_path,
                        "; ".join(rejection_reasons),
                    )
                    self.model = None
                    self.is_trained = False
            except Exception as e:
                logger.error("Failed to load multimodal checkpoint from %s: %s", checkpoint_path, e)
                self.model = None
                self.is_trained = False

    def build_record_from_inputs(
        self,
        subject_id: str,
        symptoms_present: list[Any] | None = None,
        symptoms_absent: list[Any] | None = None,
        symptoms_unknown: list[Any] | None = None,
        medical_reports: list[dict[str, Any]] | None = None,
        xray_reports: list[dict[str, Any]] | None = None,
        xray_images: list[dict[str, Any]] | None = None,
        raw_text: str | None = None,
    ) -> tuple[MultimodalPatientRecord, dict[str, Any]]:
        """Construct a validated MultimodalPatientRecord and provenance bundle from assessment inputs."""
        symptom_obs: list[ClinicalSymptomObservation] = []

        # Process symptoms
        for s in (symptoms_present or []):
            name = str(s.get("name") if isinstance(s, dict) else s).strip()
            if name:
                symptom_obs.append(ClinicalSymptomObservation(
                    symptom=name,
                    canonical=name.lower(),
                    state="PRESENT",
                    duration_days=s.get("duration_days") if isinstance(s, dict) else None,
                    severity=str(s.get("severity") or "UNKNOWN").upper() if isinstance(s, dict) else "UNKNOWN",
                    qualifiers=s.get("qualifiers", []) if isinstance(s, dict) else [],
                    body_area=s.get("body_area") if isinstance(s, dict) else None,
                    follow_up_answer=s.get("follow_up_answer") if isinstance(s, dict) else None,
                ))
        for s in (symptoms_absent or []):
            name = str(s.get("name") if isinstance(s, dict) else s).strip()
            if name:
                symptom_obs.append(ClinicalSymptomObservation(
                    symptom=name,
                    canonical=name.lower(),
                    state="ABSENT",
                ))
        for s in (symptoms_unknown or []):
            name = str(s.get("name") if isinstance(s, dict) else s).strip()
            if name:
                symptom_obs.append(ClinicalSymptomObservation(
                    symptom=name,
                    canonical=name.lower(),
                    state="UNKNOWN",
                ))

        # Process laboratory findings from medical reports (with alias resolution)
        lab_params: list[StructuredLabParameter] = []
        for rep in (medical_reports or []):
            if not isinstance(rep, dict):
                continue
            findings = rep.get("findings", []) or []
            structured = rep.get("structured_findings")
            if isinstance(structured, dict):
                findings = list(findings) + (structured.get("abnormal_findings", []) or []) + (structured.get("normal_findings", []) or []) + (structured.get("findings", []) or [])
            src_file = rep.get("file_name") or rep.get("source_file")
            r_id = str(rep.get("id") or rep.get("report_id") or "")
            for f in findings:
                if isinstance(f, dict):
                    t_name = str(f.get("test_name") or f.get("canonical") or f.get("original_test_name") or "").strip()
                    val = f.get("value")
                    canon_name = resolve_canonical_lab_name(t_name)
                    if canon_name and val is not None:
                        try:
                            v_float = float(val)
                            lab_params.append(StructuredLabParameter(
                                test_name=t_name,
                                canonical_test_name=canon_name,
                                value=v_float,
                                unit=str(f.get("unit") or ""),
                                reference_low=f.get("reference_low"),
                                reference_high=f.get("reference_high"),
                                flag=str(f.get("status") or f.get("interpretation") or "NORMAL").upper(),
                                source_file=src_file,
                                report_id=r_id,
                            ))
                        except (ValueError, TypeError):
                            pass

        # Process radiology report findings and physical clinical examinations (Problems A & F)
        rad_findings: list[StructuredRadiologyFinding] = []
        clinical_findings: list[StructuredClinicalFinding] = []

        # From general medical reports (extract examination findings if present)
        for med_rep in (medical_reports or []):
            if not isinstance(med_rep, dict):
                continue
            src_file = med_rep.get("file_name") or med_rep.get("source_file")
            r_id = str(med_rep.get("id") or med_rep.get("report_id") or "")
            q_list = med_rep.get("qualitative_findings", []) or []
            if isinstance(med_rep.get("structured_findings"), dict):
                q_list = list(q_list) + (med_rep["structured_findings"].get("qualitative_findings", []) or [])
            for q in q_list:
                if isinstance(q, dict):
                    f_name = str(q.get("finding") or "").strip()
                    cat = str(q.get("category") or "").lower()
                    st = str(q.get("state") or q.get("status") or "PRESENT").upper()
                    if f_name.lower() in CANONICAL_CLINICAL_EXAM_FINDINGS or cat == "examination":
                        clinical_findings.append(StructuredClinicalFinding(
                            finding=f_name,
                            canonical_finding=f_name.lower(),
                            category="examination",
                            state=st,
                            source_file=src_file,
                            report_id=r_id,
                        ))
                    elif f_name.lower() in CANONICAL_RAD_FINDINGS or cat.startswith("radiology"):
                        rad_findings.append(StructuredRadiologyFinding(
                            finding=f_name,
                            canonical_finding=f_name.lower(),
                            state=st,
                            anatomical_region="general",
                            source_file=src_file,
                            report_id=r_id,
                        ))

        # From radiology/xray reports
        for r_rep in (xray_reports or []):
            if not isinstance(r_rep, dict):
                continue
            src_file = r_rep.get("file_name") or r_rep.get("source_file")
            r_id = str(r_rep.get("id") or r_rep.get("report_id") or "")
            reg = str(r_rep.get("region") or r_rep.get("body_region") or "chest").strip().lower()
            q_findings = r_rep.get("qualitative_findings", []) or []
            if isinstance(r_rep.get("structured_findings"), dict):
                q_findings = list(q_findings) + (r_rep["structured_findings"].get("qualitative_findings", []) or [])
            for q in q_findings:
                if isinstance(q, dict):
                    f_name = str(q.get("finding") or "").strip()
                    cat = str(q.get("category") or "").lower()
                    st = str(q.get("state") or q.get("status") or "PRESENT").upper()
                    if f_name.lower() in CANONICAL_CLINICAL_EXAM_FINDINGS or cat == "examination":
                        clinical_findings.append(StructuredClinicalFinding(
                            finding=f_name,
                            canonical_finding=f_name.lower(),
                            category="examination",
                            state=st,
                            source_file=src_file,
                            report_id=r_id,
                        ))
                    elif f_name:
                        rad_findings.append(StructuredRadiologyFinding(
                            finding=f_name,
                            canonical_finding=f_name.lower(),
                            state=st,
                            anatomical_region=reg,
                            source_file=src_file,
                            report_id=r_id,
                            has_conflict=bool(q.get("has_conflict")),
                        ))

        # Process X-ray specialist image findings via Adapter across ALL images (Problem D & G)
        studies_list: list[XRaySpecialistOutput] = []
        unverified_studies: list[str] = []
        mismatched_studies: list[str] = []

        for img in (xray_images or []):
            if not isinstance(img, dict):
                continue
            fname = str(img.get("file_name") or "xray_image")
            det_anat = str(img.get("region") or img.get("detected_anatomy") or "unspecified").strip().lower()
            is_verified = bool(
                img.get("anatomy_independently_verified") is True
                or str(img.get("anatomy_verification_status") or "").strip().lower() in {"completed", "verified", "accepted"}
            )
            has_mismatch = bool(img.get("has_region_mismatch")) or str(img.get("status") or "").upper() == "MISMATCH"
            is_eligible = is_verified and not has_mismatch and str(img.get("status") or "").upper() == "COMPLETED"

            if not is_verified:
                unverified_studies.append(fname)
            if has_mismatch:
                mismatched_studies.append(fname)

            p_scores: dict[str, float] = {}
            for find in (img.get("findings") or []):
                if isinstance(find, dict):
                    lbl = str(find.get("label") or find.get("name") or "").strip()
                    score = find.get("model_score", find.get("score", 0.0))
                    # Model-specific mapping: MURA outputs map to Musculoskeletal Abnormality
                    if lbl.lower() in {"abnormal", "musculoskeletal abnormality", "abnormality"}:
                        p_scores["Musculoskeletal Abnormality"] = float(score)
                    elif lbl in CANONICAL_XRAY_PATHOLOGIES and isinstance(score, (int, float)):
                        p_scores[lbl] = float(score)

            study = XRaySpecialistOutput(
                file_name=fname,
                detected_anatomy=det_anat,
                anatomy_verified=is_verified,
                specialist_model_executed=bool(p_scores),
                has_region_mismatch=has_mismatch,
                is_pathology_eligible=is_eligible,
                pathology_scores=p_scores,
                anatomy_probs=img.get("anatomy_probs"),
                confidence=img.get("confidence"),
                status=str(img.get("status") or "PENDING"),
                specialist_model_name=img.get("model_name"),
            )
            studies_list.append(study)

        primary_xray = next((s for s in studies_list if s.is_pathology_eligible), (studies_list[0] if studies_list else None))

        record = MultimodalPatientRecord(
            subject_id=subject_id,
            symptoms=symptom_obs,
            laboratory=lab_params,
            radiology=rad_findings,
            clinical_examination=clinical_findings,
            xray=primary_xray,
            xray_studies=studies_list,
            provenance={
                "has_symptoms": len(symptom_obs) > 0,
                "has_laboratory": len(lab_params) > 0,
                "has_radiology": len(rad_findings) > 0,
                "has_clinical_examination": len(clinical_findings) > 0,
                "has_xray": any(s.is_pathology_eligible for s in studies_list),
                "study_count": len(studies_list),
                "unverified_studies": unverified_studies,
                "mismatched_studies": mismatched_studies,
                "raw_text_provided": bool(raw_text),
            },
        )

        tensor_bundle = MultimodalFeatureExtractor.extract_tensors_from_record(record)
        return record, tensor_bundle

    def predict(self, record: MultimodalPatientRecord) -> dict[str, Any]:
        """
        Evaluate a MultimodalPatientRecord.
        When genuine trained weights exist, returns ranked candidate conditions.
        When authentic training is blocked, cleanly discloses the prerequisite blocker and
        falls back to unimodal baseline without executing untrained random weights.
        """
        has_sym = len(record.symptoms) > 0
        has_lab = len(record.laboratory) > 0
        has_rad = len(record.radiology) > 0
        has_xray = any(s.is_pathology_eligible for s in record.xray_studies) or (record.xray is not None and record.xray.is_pathology_eligible)
        has_exam = len(record.clinical_examination) > 0

        if not (has_sym or has_lab or has_rad or has_xray or has_exam):
            return {
                "status": "insufficient_evidence",
                "conditions": [],
                "message": "No submitted medical modalities available for assessment.",
                "is_experimental": True,
                "model_scope": "multimodal_fusion",
            }

        tensor_bundle = MultimodalFeatureExtractor.extract_tensors_from_record(record)
        modality_mask = tensor_bundle["modality_mask"].squeeze(0).tolist()

        if not self.is_trained or self.model is None:
            return {
                "status": "blocked_untrained",
                "is_multimodal_trained": False,
                "model_scope": "unimodal_symptom_baseline_active",
                "active_baseline_model": "condition_pred_v2.0-calibrated_logistic",
                "multimodal_model_version": self.model_version,
                "message": (
                    "Authentic patient-linked multimodal ML training is currently blocked pending "
                    "credentialed clinical EHR access (MIMIC-IV/MIMIC-CXR). The active production "
                    "assessment relies on the validated unimodal symptom baseline and downstream "
                    "evidence reconciliation."
                ),
                "is_experimental": True,
                "modality_mask": modality_mask,
                "modalities_available": {
                    "symptoms": has_sym,
                    "laboratory": has_lab,
                    "radiology": has_rad,
                    "clinical_examination": has_exam,
                    "xray_pathology": has_xray,
                },
                "conditions": [],
            }

        # If validated trained model is present, execute torch forward pass
        try:
            with torch.no_grad():
                logits, is_ev = self.model(
                    tensor_bundle["symptom_tokens"],
                    tensor_bundle["symptom_features"],
                    tensor_bundle["lab_features"],
                    tensor_bundle["rad_features"],
                    tensor_bundle["xray_pathology_features"],
                    tensor_bundle["exam_features"],
                    tensor_bundle["xray_anatomy_probs"],
                    tensor_bundle["xray_pathology_eligible"],
                    tensor_bundle["modality_mask"],
                )
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            candidates = []
            for idx, prob in enumerate(probs):
                if prob >= 0.04:
                    candidates.append({
                        "condition": CONDITION_CLASSES[idx],
                        "probability_score": float(prob),
                        "model_score": float(prob),
                        "confidence_tier": "High Confidence" if prob >= 0.6 else ("Moderate Confidence" if prob >= 0.3 else "Possible Candidate"),
                    })
            candidates.sort(key=lambda x: x["probability_score"], reverse=True)
            for rank, c in enumerate(candidates, start=1):
                c["rank"] = rank

            return {
                "status": "completed",
                "conditions": candidates,
                "message": "Multimodal prediction executed successfully across available modalities.",
                "model_version": self.model_version,
                "model_scope": "multimodal_gmu",
                "is_experimental": False,
                "modality_mask": modality_mask,
            }
        except Exception as exc:
            return {
                "status": "error",
                "conditions": [],
                "message": f"Multimodal forward pass encountered an error: {exc}",
                "model_version": self.model_version,
                "is_experimental": True,
            }


_GLOBAL_MULTIMODAL_INFERENCE_SERVICE: Optional[MultimodalInferenceService] = None


def get_multimodal_inference_service() -> MultimodalInferenceService:
    """Singleton getter for MultimodalInferenceService."""
    global _GLOBAL_MULTIMODAL_INFERENCE_SERVICE
    if _GLOBAL_MULTIMODAL_INFERENCE_SERVICE is None:
        ckpt_dir = Path(__file__).resolve().parents[2] / "models" / "conditions_multimodal"
        ckpt_path = ckpt_dir / "multimodal_model.pth"
        _GLOBAL_MULTIMODAL_INFERENCE_SERVICE = MultimodalInferenceService(checkpoint_path=ckpt_path)
    return _GLOBAL_MULTIMODAL_INFERENCE_SERVICE
