"""Validation of all 15 non-empty combinations of multimodal inputs."""

import itertools
from app.ai.symptom_nlp import extract_symptoms
from app.ai.evidence_engine import combine_evidence
from app.ai.triage_engine import apply_triage
from app.ai.specialty_mapper import map_evidence_to_specialty
from app.services.assessment_service import generate_assessment_feedback

symptoms_data = extract_symptoms("I have had a cough for 9 days with mild fever for 4 days.")["symptoms"]
lab_data = [{"test_name": "WBC", "value": 11500, "unit": "/uL", "reference_range": "4000 - 11000", "status": "HIGH", "interpretation": "Mild leukocytosis"}]
xray_image_data = [{"file_name": "chest_xray.png", "region": "chest", "status": "completed", "prediction": "Infiltration", "anatomy_independently_verified": True, "anatomy_verification_status": "completed"}]
xray_report_data = [{"file_name": "radiology_report.pdf", "region": "chest", "qualitative_findings": [{"finding": "patchy airspace opacity", "state": "PRESENT"}]}]

modalities = ["symptoms", "medical_report", "xray_image", "radiology_report"]
all_combinations = []
for r in range(1, 5):
    for combo in itertools.combinations(modalities, r):
        all_combinations.append(list(combo))

print(f"Total combinations to test: {len(all_combinations)}")

for i, combo in enumerate(all_combinations, 1):
    s = symptoms_data if "symptoms" in combo else []
    l = lab_data if "medical_report" in combo else []
    x = xray_image_data if "xray_image" in combo else []
    r = xray_report_data if "radiology_report" in combo else []

    ev = combine_evidence(
        symptoms=s,
        report_findings=l,
        xray_results=x,
        xray_reports=r,
        selected_input_types=combo,
        medical_report_provided="medical_report" in combo,
        radiology_report_provided="radiology_report" in combo,
    )
    tr = apply_triage(ev)
    sp = map_evidence_to_specialty(ev)
    fb = generate_assessment_feedback(ev, tr, sp)

    # Verification rules
    assert fb.get("assessment_summary"), f"Combo {i} missing assessment_summary"

    # Category appearance: only selected categories are described as present
    if "symptoms" not in combo:
        assert fb.get("symptoms", {}).get("present") == [] or fb.get("symptoms") is None
    if "medical_report" not in combo:
        assert fb.get("medical_report", {}).get("findings") == [] or fb.get("medical_report") is None
    if "xray_image" not in combo:
        assert fb.get("xray", {}).get("images") == [] or fb.get("xray") is None
    if "radiology_report" not in combo:
        assert fb.get("radiology_report", {}).get("findings") == [] or fb.get("radiology_report") is None

    triage_pathway = tr.get("pathway") or tr.get("status")
    print(f"Combo {i:2d}: {' + '.join(combo)} -> Triage: {triage_pathway} | Summary OK")

print("\nALL 15 COMBINATIONS SYSTEMATICALLY VERIFIED AND PASSED!")
