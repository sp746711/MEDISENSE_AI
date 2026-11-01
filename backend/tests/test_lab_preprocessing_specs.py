"""Regression tests for laboratory preprocessing, unit conversions, and feature extraction (Part 4).

Validates canonical schemas, unit conversions, z-score standardization, and missingness preservation across:
1. Complete Blood Count (CBC): WBC, Platelets, Hemoglobin, RBC, Hematocrit, Neutrophils, Lymphocytes
2. Inflammatory Markers: C-Reactive Protein (CRP), Procalcitonin, ESR
3. Glucose Markers: Fasting Blood Glucose, Postprandial Blood Glucose, HbA1c
4. Liver Function Markers: ALT / SGPT, AST / SGOT, Total Bilirubin
5. Kidney Function Markers: Serum Creatinine, Blood Urea Nitrogen (BUN), Uric Acid
"""

from __future__ import annotations

import pytest
import torch

from app.ai.multimodal_condition_model import (
    CANONICAL_LAB_TESTS,
    LAB_PREPROCESSING_SPECS,
    LaboratoryModalityEncoder,
    MultimodalFeatureExtractor,
    MultimodalPatientRecord,
    StructuredLabParameter,
    normalize_lab_value,
    resolve_canonical_lab_name,
)


def test_lab_canonical_name_resolution():
    # CBC aliases
    assert resolve_canonical_lab_name("WBC") == "White Blood Cell Count (WBC)"
    assert resolve_canonical_lab_name("total wbc count") == "White Blood Cell Count (WBC)"
    assert resolve_canonical_lab_name("Platelet") == "Platelet Count"
    assert resolve_canonical_lab_name("platelets") == "Platelet Count"
    assert resolve_canonical_lab_name("Hb") == "Hemoglobin"
    assert resolve_canonical_lab_name("haemoglobin") == "Hemoglobin"
    assert resolve_canonical_lab_name("RBC Count") == "Red Blood Cell Count (RBC)"

    # Inflammatory
    assert resolve_canonical_lab_name("CRP") == "C-Reactive Protein (CRP)"
    assert resolve_canonical_lab_name("c reactive protein") == "C-Reactive Protein (CRP)"
    assert resolve_canonical_lab_name("Procalcitonin") == "Procalcitonin"
    assert resolve_canonical_lab_name("pct") == "Procalcitonin"

    # Glucose
    assert resolve_canonical_lab_name("FBS") == "Fasting Blood Glucose"
    assert resolve_canonical_lab_name("fasting blood sugar") == "Fasting Blood Glucose"
    assert resolve_canonical_lab_name("PPBS") == "Postprandial Blood Glucose"
    assert resolve_canonical_lab_name("HbA1c") == "HbA1c"

    # Liver
    assert resolve_canonical_lab_name("SGPT") == "SGPT / ALT"
    assert resolve_canonical_lab_name("ALT") == "SGPT / ALT"
    assert resolve_canonical_lab_name("SGOT") == "SGOT / AST"
    assert resolve_canonical_lab_name("AST") == "SGOT / AST"
    assert resolve_canonical_lab_name("Bilirubin Total") == "Total Bilirubin"

    # Kidney
    assert resolve_canonical_lab_name("Creatinine") == "Serum Creatinine"
    assert resolve_canonical_lab_name("Serum Creatinine") == "Serum Creatinine"
    assert resolve_canonical_lab_name("BUN") == "Blood Urea Nitrogen"
    assert resolve_canonical_lab_name("Uric Acid") == "Uric Acid"


def test_cbc_preprocessing_and_unit_conversion():
    # Hemoglobin: 14.0 g/dL is canonical mean
    val_norm, r_low, r_high, flag, present = normalize_lab_value(
        "Hemoglobin", raw_value=14.0, raw_unit="g/dL"
    )
    assert pytest.approx(val_norm, rel=1e-3) == 0.0
    assert present == 1.0
    assert flag == 0.0

    # Hemoglobin in g/L: 140 g/L -> 14.0 g/dL
    val_norm_gl, _, _, _, _ = normalize_lab_value(
        "Hemoglobin", raw_value=140.0, raw_unit="g/L"
    )
    assert pytest.approx(val_norm_gl, rel=1e-3) == val_norm

    # Platelet Count in lakh/cumm: 2.5 lakh -> 250,000 cells/mcL (mean)
    p_norm, _, _, _, _ = normalize_lab_value(
        "Platelet Count", raw_value=2.5, raw_unit="lakh/cumm"
    )
    assert pytest.approx(p_norm, rel=1e-3) == 0.0

    # Platelet Count in cells/mcL: 250000 -> 0.0 z-score
    p_norm_cells, _, _, _, _ = normalize_lab_value(
        "Platelet Count", raw_value=250000.0, raw_unit="cells/mcL"
    )
    assert pytest.approx(p_norm_cells, rel=1e-3) == 0.0


def test_inflammatory_markers_crp_and_procalcitonin():
    # CRP: 3.0 mg/L is mean, std is 6.0
    # raw_value 15.0 mg/L -> (15 - 3) / 6 = 2.0 z-score
    c_norm, _, _, flag, present = normalize_lab_value(
        "C-Reactive Protein (CRP)", raw_value=15.0, raw_unit="mg/L", flag="HIGH"
    )
    assert pytest.approx(c_norm, rel=1e-3) == 2.0
    assert flag == 1.0
    assert present == 1.0

    # CRP in mg/dL: 1.5 mg/dL -> 15.0 mg/L (mult 10.0) -> (15 - 3)/6 = 2.0
    c_norm_mgdl, _, _, _, _ = normalize_lab_value(
        "C-Reactive Protein (CRP)", raw_value=1.5, raw_unit="mg/dL"
    )
    assert pytest.approx(c_norm_mgdl, rel=1e-3) == 2.0

    # Procalcitonin: 0.15 ng/mL is mean, std is 0.5
    # raw_value 1.15 ng/mL -> (1.15 - 0.15) / 0.5 = 2.0 z-score
    pct_norm, _, _, _, present_pct = normalize_lab_value(
        "Procalcitonin", raw_value=1.15, raw_unit="ng/mL"
    )
    assert pytest.approx(pct_norm, rel=1e-3) == 2.0
    assert present_pct == 1.0


def test_glucose_markers_fasting_pp_and_hba1c():
    # Fasting glucose in mmol/L: 5.55 mmol/L * 18.0182 = 100 mg/dL
    # mean is 95.0, std is 25.0 -> (100 - 95)/25 = 0.20
    fbs_norm, _, _, _, _ = normalize_lab_value(
        "Fasting Blood Glucose", raw_value=5.55, raw_unit="mmol/L"
    )
    assert pytest.approx(fbs_norm, rel=1e-2) == 0.20

    # Fasting glucose direct in mg/dL: 195.0 mg/dL -> (195 - 95)/25 = 4.0
    fbs_high, _, _, flag_high, _ = normalize_lab_value(
        "Fasting Blood Glucose", raw_value=195.0, raw_unit="mg/dL", flag="HIGH"
    )
    assert pytest.approx(fbs_high, rel=1e-3) == 4.0
    assert flag_high == 1.0

    # HbA1c: 6.6% -> mean 5.4, std 1.2 -> (6.6 - 5.4)/1.2 = 1.0
    hba1c_norm, _, _, _, _ = normalize_lab_value(
        "HbA1c", raw_value=6.6, raw_unit="%"
    )
    assert pytest.approx(hba1c_norm, rel=1e-3) == 1.0


def test_liver_markers_alt_ast_bilirubin():
    # ALT: mean 30.0, std 18.0
    # 66 U/L -> (66 - 30)/18 = 2.0
    alt_norm, _, _, flag, _ = normalize_lab_value(
        "SGPT / ALT", raw_value=66.0, raw_unit="U/L", flag="HIGH"
    )
    assert pytest.approx(alt_norm, rel=1e-3) == 2.0
    assert flag == 1.0

    # AST: mean 25.0, std 15.0
    # 55 U/L -> (55 - 25)/15 = 2.0
    ast_norm, _, _, _, _ = normalize_lab_value(
        "SGOT / AST", raw_value=55.0, raw_unit="IU/L"
    )
    assert pytest.approx(ast_norm, rel=1e-3) == 2.0

    # Total Bilirubin in umol/L: 34.2 umol/L -> 34.2 / 17.1 = 2.0 mg/dL
    # mean 0.7, std 0.4 -> (2.0 - 0.7)/0.4 = 3.25
    bili_norm, _, _, _, _ = normalize_lab_value(
        "Total Bilirubin", raw_value=34.2, raw_unit="umol/L"
    )
    assert pytest.approx(bili_norm, rel=1e-3) == 3.25


def test_kidney_markers_creatinine_bun_uric_acid():
    # Creatinine in umol/L: 176.8 umol/L -> 176.8 / 88.4 = 2.0 mg/dL
    # mean 0.9, std 0.4 -> (2.0 - 0.9)/0.4 = 2.75
    creat_norm, _, _, flag, _ = normalize_lab_value(
        "Serum Creatinine", raw_value=176.8, raw_unit="umol/L", flag="HIGH"
    )
    assert pytest.approx(creat_norm, rel=1e-3) == 2.75
    assert flag == 1.0

    # BUN in mmol/L: 7.0 mmol/L -> 7.0 * 2.8 = 19.6 mg/dL
    # mean 14.0, std 6.0 -> (19.6 - 14.0)/6.0 = 0.9333
    bun_norm, _, _, _, _ = normalize_lab_value(
        "Blood Urea Nitrogen", raw_value=7.0, raw_unit="mmol/L"
    )
    assert pytest.approx(bun_norm, rel=1e-3) == 0.9333

    # Uric Acid: mean 5.2, std 1.5 -> 8.2 mg/dL -> (8.2 - 5.2)/1.5 = 2.0
    uric_norm, _, _, _, _ = normalize_lab_value(
        "Uric Acid", raw_value=8.2, raw_unit="mg/dL"
    )
    assert pytest.approx(uric_norm, rel=1e-3) == 2.0


def test_laboratory_modality_encoder_tensor_contract():
    encoder = LaboratoryModalityEncoder(num_tests=len(CANONICAL_LAB_TESTS), out_dim=128)

    # Empty labs -> all zeros, missingness indicator is 0.0 for all tests
    raw_empty = torch.zeros(1, len(CANONICAL_LAB_TESTS) * 5)
    out_empty = encoder(raw_empty)
    assert out_empty.shape == (1, 128)

    # Structured lab parameters via MultimodalFeatureExtractor
    record = MultimodalPatientRecord(
        subject_id="TEST_PT_01",
        laboratory=[
            StructuredLabParameter(
                test_name="Hemoglobin",
                canonical_test_name="Hemoglobin",
                value=14.0,
                unit="g/dL",
                flag="NORMAL",
            ),
            StructuredLabParameter(
                test_name="CRP",
                canonical_test_name="C-Reactive Protein (CRP)",
                value=25.0,
                unit="mg/L",
                flag="HIGH",
            ),
            StructuredLabParameter(
                test_name="Creatinine",
                canonical_test_name="Serum Creatinine",
                value=2.5,
                unit="mg/dL",
                flag="HIGH",
            ),
        ],
    )

    tensors = MultimodalFeatureExtractor.extract_tensors_from_record(record)
    raw_feat = tensors["lab_features"]
    assert raw_feat.shape == (1, len(CANONICAL_LAB_TESTS) * 5)

    # Verify present flags
    hb_idx = CANONICAL_LAB_TESTS.index("Hemoglobin")
    crp_idx = CANONICAL_LAB_TESTS.index("C-Reactive Protein (CRP)")
    creat_idx = CANONICAL_LAB_TESTS.index("Serum Creatinine")
    wbc_idx = CANONICAL_LAB_TESTS.index("White Blood Cell Count (WBC)")

    # Present tests have is_present = 1.0 (at 5th slot)
    assert raw_feat[0, hb_idx * 5 + 4] == 1.0
    assert raw_feat[0, crp_idx * 5 + 4] == 1.0
    assert raw_feat[0, creat_idx * 5 + 4] == 1.0
    # Absent test has is_present = 0.0
    assert raw_feat[0, wbc_idx * 5 + 4] == 0.0

    # Ensure encoder processes populated tensor without error
    out_active = encoder(raw_feat)
    assert out_active.shape == (1, 128)
    assert not torch.equal(out_empty, out_active)
