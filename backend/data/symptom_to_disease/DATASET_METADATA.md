# Dataset Metadata: Symptom2Disease (Symptom-to-Condition Prediction)

## 1. Overview and Provenance
- **Dataset Name**: Symptom2Disease / `gretelai/symptom_to_diagnosis`
- **Origin / Upstream Repository**: Curated on Kaggle by user `niyarrn` (Symptom2Disease Benchmark) and expanded on Hugging Face as `gretelai/symptom_to_diagnosis`. Also archived in the Mistral AI Open Cookbook (`https://raw.githubusercontent.com/mistralai/cookbook/main/data/Symptom2Disease.csv`).
- **License**: Apache 2.0 / Open Community Data License.
- **Task**: Multi-class text classification mapping patient-reported symptom descriptions to 24 distinct medical conditions.
- **Collection & Annotation Method**: Natural language clinical symptom descriptions representing authentic patient phrasings and presentation patterns across common outpatient and primary care medical complaints.
- **Dataset Size**: 1,200 samples (50 balanced instances per disease condition).

## 2. Target Disease Labels (24 Classes)
1. `Acne`
2. `Arthritis`
3. `Bronchial Asthma`
4. `Cervical spondylosis`
5. `Chicken pox`
6. `Common Cold`
7. `Dengue`
8. `Dimorphic Hemorrhoids`
9. `Fungal infection`
10. `Hypertension`
11. `Impetigo`
12. `Jaundice`
13. `Malaria`
14. `Migraine`
15. `Pneumonia`
16. `Psoriasis`
17. `Typhoid`
18. `Varicose Veins`
19. `allergy`
20. `diabetes`
21. `drug reaction`
22. `gastroesophageal reflux disease`
23. `peptic ulcer disease`
24. `urinary tract infection`

## 3. Data Schema
- `label` (string): Standardized medical condition name.
- `text` (string): Natural language description of symptoms, onset, and clinical manifestations.

## 4. Known Limitations & Clinical Safety Bounds
- **Intended Purpose**: Academic research, educational clinical decision support, and diagnostic triage navigation benchmarking.
- **Safety Restriction**: This dataset and derived machine learning models do **NOT** constitute medical devices or definitive diagnostic tests. They are not cleared by the FDA or international regulatory bodies as standalone diagnostic tools.
- **Uncertainty & Negative Evidence**: Missing symptoms must never be imputed as negative findings. Uncertain symptoms ("I might have a fever", "not sure if I have fever") must not be treated as confirmed positive findings.
- **Independent Safety Triage**: Emergency and red-flag clinical triage rules (such as for chest pain, sudden thunderclap headache, acute dyspnea) must remain completely independent and can never be overridden by model condition probabilities.
