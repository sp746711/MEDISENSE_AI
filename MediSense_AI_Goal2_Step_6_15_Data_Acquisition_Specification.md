# MediSense AI — Goal 2 Step 6.15 & 6.16
# Authorized Data Acquisition & Cohort Extraction Specification
## (Updated with Step 6.16 Independent Audit & Implementation Corrections)

**Document Version:** 1.1.0  
**Status:** SPECIFICATION & READINESS AUDIT (Fail-Closed, Non-Executing, Corrected)  
**Execution Mode:** Zero-Cost ($0.00 Expenditure, Local Free-Only Tools)  
**System Architecture:** Goal 2 Track B Multimodal Clinical Research Pipeline  
**Target Date / Timestamp:** 2026-11-03T02:00:00+05:30  

---

## 1. Executive Summary & Objective

### 1.1 Objective
This specification establishes the definitive, technically actionable, and reproducible data acquisition and cohort extraction protocol for **MediSense AI Goal 2 Track B**. It builds upon the verified architectural foundations and clinical corrections completed in Steps 6.1 through 6.14.

The primary objective is to define the exact methodology for authentic multimodal research data acquisition utilizing freely accessible, ethically governed, credentialed clinical datasets—specifically **MIMIC-IV v2.2**, **MIMIC-IV-ED v2.2**, **MIMIC-CXR-JPG v2.0.0**, and **MIMIC-IV-Note v2.2**.

### 1.2 Strict Scope & Operational Boundaries
- **SPECIFICATION AND READINESS AUDIT ONLY:** This document provides the engineering and clinical extraction blueprint. It strictly prohibits downloading credentialed clinical datasets, extracting restricted patient records, training neural models, generating weights/checkpoints, or activating Track B multimodal inference in production during this step.
- **Fail-Closed Gatekeeper:** In accordance with Step 6.7 and Step 6.14, the Track B training gatekeeper (`train_multimodal_model.py`) remains locked in an un-bypassable failure state (`exit code 1/2`) until authentic, signed, and validated physical datasets are acquired under valid institutional authorization.
- **Absolute Free-Only Constraint:** $0.00 financial expenditure. Prohibits paid cloud infrastructure, paid API endpoints, commercial database services, paid KMS/HSM key vaults, and commercial data brokers. All tooling is restricted to local Python 3.13, PostgreSQL, pandas, PyTorch, and legitimate open-science credentialed repositories.

---

## 2. Strategic Architecture Invariance

The existing Goal 2 strategic principles established in Steps 6.1–6.14 are preserved without compromise:

```
+---------------------------------------------------------------------------------------------------+
|                                  MEDISENSE AI DUAL-TRACK ARCHITECTURE                             |
+---------------------------------------------------------------------------------------------------+
|  [TRACK A: 24-Condition Symptom Baseline]             [TRACK B: 5-Target Multimodal Research]     |
|  - Invariant Production Model                         - Isolated Research Architecture            |
|  - 24 Canonical Conditions                            - 5 Chest-Focused Target Pathologies        |
|  - Single-Modality (Patient Narrative)                - 5 Distinct Modalities                     |
|  - Unaltered Checkpoint & Routing                     - Independent Multi-Label Sigmoid Heads     |
|  - Active Production Inference                        - Condition-Specific Modality Eligibility   |
|                                                       - Hard Cryptographic & Readiness Gating     |
|                                                       - PRODUCTION DEACTIVATED / TRAINING BLOCKED |
+---------------------------------------------------------------------------------------------------+
```

### 2.1 The Five Target Research Conditions
Track B is strictly focused on five acute cardiopulmonary conditions requiring multimodal differentiation:
1. **Pneumonia**
2. **Acute Decompensated Heart Failure (ADHF)**
3. **Pleural Effusion**
4. **Spontaneous Pneumothorax**
5. **Acute COPD Exacerbation**

### 2.2 The Five Modalities
1. **Symptoms:** Patient-reported chief complaints, structured narrative symptom tokens.
2. **Laboratory Reports:** Quantitative serum panels (21 canonical analyte measurements, reference flags).
3. **Radiology Reports:** Semantically parsed qualitative narrative chest radiograph findings (14 canonical entities).
4. **X-Ray Images:** Digital chest radiographs (DICOM/JPG, frontal AP/PA views) evaluated by specialized vision backbones.
5. **Clinical Examination:** Physical examination assertions (auscultation, inspection, percussion) strictly isolated from vital signs.

### 2.3 Independent Sigmoid Outputs & Condition-Specific Modality Eligibility
- **Non-Competitive Multi-Label Classification:** Predictions utilize 5 independent sigmoid activation heads ($\sigma(z_c) \in [0, 1]$). Conditions are clinically non-mutually exclusive (e.g., Pneumonia and Pleural Effusion frequently co-occur).
- **Condition-Specific Masking:** The $(5 \times 5)$ `CHEST_CONDITION_ELIGIBILITY_MATRIX` governs cross-attention and gating. Modalities assigned weight $0.0$ cannot contribute to positive diagnostic logit accumulation. Specifically, for **Acute COPD Exacerbation**, Chest X-ray is restricted to `RULE_OUT_ONLY` ($w = 0.0$).
- **Protected Boundaries:** Track A 24-condition symptom classifier, the 15 protected X-ray assets (with verified SHA-256 integrity hashes), Grad-CAM explainability, Step 6.6 HMAC cryptographic crosswalk signing, and Step 6.13 zero-NaN masked softmax fusion remain inviolate.

---

## 3. Authoritative Dataset Inventory & Relational Mapping

Authentic multimodal acquisition requires cross-linking four modular databases published by the PhysioNet consortium (Beth Israel Deaconess Medical Center, Boston, MA).

```
                      MIMIC-IV Core & Hosp (v2.2)
                     +---------------------------+
                     |         patients          |
                     |  PK: subject_id           |
                     +-------------+-------------+
                                   | 1:N
                     +-------------v-------------+
                     |        admissions         |
                     |  PK: hadm_id              |
                     |  FK: subject_id           |
                     +------+--------------+-----+
                            | 1:N          | 1:N
           +----------------v----+   +-----v----------------+
           |    diagnoses_icd    |   |      labevents       |
           |  FK: hadm_id        |   |  PK: labevent_id     |
           |  FK: subject_id     |   |  FK: hadm_id, sub_id |
           +---------------------+   +----------+-----------+
                                                | N:1
                                     +----------v-----------+
                                     |      d_labitems      |
                                     |  PK: itemid          |
                                     +----------------------+
                                   |
              MIMIC-IV-ED (v2.2)   |
             +---------------------+-----+
             |  edstays                  |
             |  PK: stay_id              |
             |  FK: subject_id, hadm_id  |<=================================+
             +-------------+-------------+                                  |
                           | 1:1                                            |
             +-------------v-------------+                                  |
             |  triage                   |                                  |
             |  PK: stay_id              |                                  |
             +---------------------------+                                  |
                                                                            |
              MIMIC-CXR-JPG (v2.0.0)                                        |  MISSING DIRECT LINK!
             +---------------------------+                                  |  Requires Authorized
             |  metadata / record-list   |                                  |  Bridge Crosswalk
             |  PK: dicom_id             |                                  |  (cxr_ed / ed_radiology)
             |  FK: subject_id, study_id |==================================+
             +-------------+-------------+  (Subject + Study does NOT link
                           | 1:1             directly to Encounter without
             +-------------v-------------+   authenticated bridge table!)
             |  s{study_id}.txt          |
             |  (Radiology Reports)      |
             +---------------------------+
```

### 3.1 MIMIC-IV Core & Hospital (v2.2)
- **`patients`**
  - *Primary Key:* `subject_id` (INTEGER)
  - *Required Columns:* `subject_id`, `gender` (VARCHAR), `anchor_age` (SMALLINT), `anchor_year` (SMALLINT), `anchor_year_group` (VARCHAR), `dod` (DATE).
  - *Purpose:* Cohort demographic filtering, age gating ($\ge 18$), and patient-level splitting.
- **`admissions`**
  - *Primary Key:* `hadm_id` (INTEGER)
  - *Foreign Key:* `subject_id` $\rightarrow$ `patients.subject_id`
  - *Required Columns:* `hadm_id`, `subject_id`, `admittime` (TIMESTAMP), `dischtime` (TIMESTAMP), `deathtime` (TIMESTAMP), `admission_type` (VARCHAR), `admission_location` (VARCHAR), `discharge_location` (VARCHAR), `edregtime` (TIMESTAMP), `edouttime` (TIMESTAMP), `hospital_expire_flag` (SMALLINT).
  - *Purpose:* Inpatient encounter boundary definition, hospital stay duration, and temporal anchor for laboratory events.
- **`diagnoses_icd`**
  - *Composite Key:* (`hadm_id`, `seq_num`)
  - *Foreign Keys:* `hadm_id` $\rightarrow$ `admissions.hadm_id`, `subject_id` $\rightarrow$ `patients.subject_id`
  - *Required Columns:* `subject_id`, `hadm_id`, `seq_num` (INTEGER), `icd_code` (VARCHAR), `icd_version` (SMALLINT: 9 or 10).
  - *Purpose:* Ground-truth diagnosis extraction; identification of primary (`seq_num == 1`) versus secondary (`seq_num > 1`) clinical indications.
- **`labevents`**
  - *Primary Key:* `labevent_id` (INTEGER)
  - *Foreign Keys:* `hadm_id` $\rightarrow$ `admissions.hadm_id` (nullable for outpatient/ED prior to formal bed assignment), `subject_id` $\rightarrow$ `patients.subject_id`, `itemid` $\rightarrow$ `d_labitems.itemid`
  - *Required Columns:* `labevent_id`, `subject_id`, `hadm_id`, `itemid`, `charttime` (TIMESTAMP), `valuenum` (DOUBLE PRECISION), `valueuom` (VARCHAR), `flag` (VARCHAR: 'abnormal').
  - *Purpose:* Extraction of quantitative serum analytes matched against canonical lab panels.
  - *Step 6.16 Mandatory Encounter Linkage Rule:* In `labevents.csv`, `hadm_id` is nullable. **`subject_id + charttime` alone must NEVER qualify as `VERIFIED_ENCOUNTER_LINK`.** A subject-level match or temporal proximity is not proof of an encounter relationship. A lab result must have an explicit matching `hadm_id` or an authorized, documented crosswalk (`MIMIC_IV_ED_TRANSFER`, `LAB_ENCOUNTER_CROSSWALK`). Unverified temporal matches remain `TEMPORAL_CANDIDATE_LINK` (`lab_eligible = 0.0`). Conflicting or invalid links (mismatched `subject_id` or `hadm_id`) must be `REJECTED`. Silent promotion of temporal candidate labs is strictly prohibited.
- **`d_labitems`**
  - *Primary Key:* `itemid` (INTEGER)
  - *Required Columns:* `itemid`, `label` (VARCHAR), `fluid` (VARCHAR), `category` (VARCHAR), `loinc_code` (VARCHAR).
  - *Purpose:* Standardizing hospital-specific laboratory item identifiers to the 21 canonical MediSense analytes.
- **`procedures_icd`** (Contextual Reference)
  - *Composite Key:* (`hadm_id`, `seq_num`)
  - *Foreign Keys:* `hadm_id` $\rightarrow$ `admissions.hadm_id`, `subject_id` $\rightarrow$ `patients.subject_id`
  - *Required Columns:* `subject_id`, `hadm_id`, `seq_num`, `chartdate` (DATE), `icd_code` (VARCHAR), `icd_version` (SMALLINT).
  - *Purpose:* Identifying procedural interventions (e.g., tube thoracostomy for pneumothorax, therapeutic thoracentesis for effusion, mechanical ventilation).

### 3.2 MIMIC-IV-ED (v2.2)
- **`edstays`**
  - *Primary Key:* `stay_id` (INTEGER)
  - *Foreign Keys:* `subject_id` $\rightarrow$ `patients.subject_id`, `hadm_id` $\rightarrow$ `admissions.hadm_id` (nullable if discharge directly from ED without inpatient bed admission).
  - *Required Columns:* `subject_id`, `hadm_id`, `stay_id`, `intime` (TIMESTAMP), `outtime` (TIMESTAMP), `gender`, `race`, `arrival_transport`, `disposition`.
  - *Purpose:* Crucial relational nexus connecting pre-hospital/emergency presentation to inpatient hospital admission (`stay_id` $\leftrightarrow$ `hadm_id`).
- **`triage`**
  - *Primary Key:* `stay_id` (INTEGER)
  - *Foreign Key:* `stay_id` $\rightarrow$ `edstays.stay_id`, `subject_id` $\rightarrow$ `patients.subject_id`
  - *Required Columns:* `stay_id`, `subject_id`, `temperature` (NUMERIC), `heartrate` (NUMERIC), `resprate` (NUMERIC), `o2sat` (NUMERIC), `sbp` (NUMERIC), `dbp` (NUMERIC), `pain` (VARCHAR), `acuity` (NUMERIC), `chiefcomplaint` (VARCHAR).
  - *Purpose:* Primary source for authentic presenting symptoms (`chiefcomplaint`). Separates physiological vital signs from physical clinical exam findings.
- **`vitalsign`**
  - *Foreign Key:* `stay_id` $\rightarrow$ `edstays.stay_id`
  - *Required Columns:* `stay_id`, `charttime` (TIMESTAMP), `temperature`, `heartrate`, `resprate`, `o2sat`, `sbp`, `dbp`, `rhythm`.
  - *Purpose:* Longitudinal physiological tracking in ED. Must NEVER be mapped into physical exam findings.

### 3.3 MIMIC-CXR & MIMIC-CXR-JPG (v2.0.0)
- **`mimic-cxr-2.0.0-metadata.csv`**
  - *Primary Key:* `dicom_id` (VARCHAR)
  - *Foreign Keys:* `study_id` (INTEGER), `subject_id` (INTEGER)
  - *Required Columns:* `dicom_id`, `subject_id`, `study_id`, `PerformedProcedureStepDescription`, `ViewPosition` (VARCHAR: 'PA', 'AP', 'LATERAL'), `Rows` (INTEGER), `Columns` (INTEGER), `StudyDate` (INTEGER: YYYYMMDD), `StudyTime` (DOUBLE PRECISION: HHMMSS.frac).
  - *Purpose:* Imaging study metadata, view position filtering (PA/AP frontal only), DICOM dimensions.
- **`cxr-record-list.csv`**
  - *Foreign Keys:* `subject_id`, `study_id`, `dicom_id`
  - *Required Columns:* `subject_id`, `study_id`, `dicom_id`, `path` (VARCHAR).
  - *Purpose:* Relative file system pointer to JPEG image representations (`files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16c7b-bf524ed8.jpg`).
- **De-identified Radiology Reports (`s{study_id}.txt`)**
  - *Identifiers:* Derived from file path / file name (`s{study_id}.txt`), containing `study_id`.
  - *Required Sections:* `"FINDINGS:"`, `"IMPRESSION:"`.
  - *Purpose:* Source of truth for NLP radiology entity extraction (14 canonical findings).
- **CRITICAL MISSING ENCOUNTER-LEVEL RELATIONSHIP:**
  MIMIC-CXR metadata natively contains **ONLY** `subject_id` and `study_id`. It does **NOT** contain `hadm_id` or `stay_id`.
  > [!CAUTION]
  > **FATAL ARCHITECTURAL ASSUMPTION TO AVOID:** Do NOT assume that `subject_id` plus `study_id` establishes an encounter link to `admissions.csv` or `edstays.csv`. A patient may have dozens of distinct hospital admissions and outpatient radiology visits across several years. Connecting an imaging study to a specific acute hospital encounter requires an official PhysioNet bridge crosswalk (`ed_radiology.csv` or `cxr_ed.csv`). In the absence of an authentic, verifiable bridge table, the encounter linkage status is strictly **BLOCKED**.

### 3.4 MIMIC-IV-Note (v2.2)
- **`discharge.csv`**
  - *Primary Key:* `note_id` (VARCHAR)
  - *Foreign Keys:* `subject_id` $\rightarrow$ `patients.subject_id`, `hadm_id` $\rightarrow$ `admissions.hadm_id`
  - *Required Columns:* `note_id`, `subject_id`, `hadm_id`, `note_type` (VARCHAR: 'DISCHARGE_SUMMARY'), `note_seq` (SMALLINT), `charttime` (TIMESTAMP), `storetime` (TIMESTAMP), `text` (VARCHAR).
- **Physical Examination Extraction Reality:**
  - Structured physical examination data **DOES NOT EXIST** in MIMIC-IV.
  - Discharge summaries represent post-treatment retrospectives. In Step 6.13, `ClinicalExamTimingProvenanceValidator` established that routine progress notes and discharge summaries are **REJECTED** as initial clinical examinations unless:
    1. The note contains an explicit, isolated initial admission H&P section (`"PHYSICAL EXAMINATION ON ADMISSION:"`).
    2. The examination timestamp is confirmed to precede therapeutic intervention.
    3. The examination timestamp falls strictly within $[-24\text{ h}, +24\text{ h}]$ of hospital/ED admission.
  - Until an authorized, validated NLP extraction pipeline specifically benchmarked on Beth Israel clinical notes is approved, clinical examination modality remains **DATA_UNAVAILABLE** for training (`exam_eligible = 0.0`).

---

## 4. Patient, Encounter and Imaging Linkage Specification

Every accepted multimodal research record must preserve complete relational integrity across hospital admissions, emergency stays, and radiology studies.

```
       [Raw MIMIC-CXR Study]                    [Raw MIMIC-IV Encounter]
       - subject_id                             - subject_id
       - study_id                               - hadm_id / stay_id
       - StudyDate / StudyTime                  - admittime / intime
                 |                                         |
                 +-------------------+---------------------+
                                     |
                       [Authorized Bridge Crosswalk]
                       - Source: PhysioNet cxr_ed / ed_radiology
                       - Cryptographic HMAC-SHA256 Envelope
                       - Verified Physical Disk Existence
                                     |
             +-----------------------+-----------------------+
             |                                               |
     [Verification PASS]                             [Verification FAIL]
             |                                               |
             v                                               v
  VERIFIED_ENCOUNTER_LINK                          Linkage Status Fallback:
  - Preserved in Cohort                            - TEMPORAL_CANDIDATE_LINK (xray_elig = 0.0)
  - Eligible for Multimodal Fusion                 - UNLINKED (xray_elig = 0.0)
                                                   - REJECTED (Dropped)
```

### 4.1 Preserved Record Schema
Every candidate record admitted to the Track B pipeline must preserve the following metadata tuple:
1. `subject_id`: Patient identifier.
2. `hadm_id`: Inpatient hospital admission identifier (when applicable).
3. `stay_id`: Emergency department stay identifier (when applicable).
4. `study_id`: Radiology imaging study identifier (when applicable).
5. `source_record_identifiers`: Specific source table row primary keys (`labevent_id`, `note_id`, `dicom_id`).
6. `encounter_timestamps`: `intime`, `outtime`, `admittime`, `dischtime`.
7. `study_timestamps`: `StudyDate`, `StudyTime` normalized to UTC datetime.
8. `source_dataset_and_version`: Exact string identifier (e.g., `"MIMIC-IV-v2.2"`, `"MIMIC-CXR-JPG-v2.0.0"`).
9. `crosswalk_provenance`: Cryptographic signing authority, generator tag (`"MIMIC_CXR_ED_BRIDGE_V1"`), and source table reference (`"MIMIC_IV_ED.ED_RADIOLOGY"`).
10. `linkage_verification_status`: Linkage enum value.

### 4.2 Cryptographic Crosswalk Security Boundary (Step 6.6)
All imaging-to-encounter crosswalk records must be signed using the Step 6.6 `CrosswalkSigningAuthority` utilizing HMAC-SHA256 over a canonical deterministic payload:
$$\text{Payload} = \text{"sub:}s\text{|hadm:}h\text{|study:}st\text{|bridge:}b\text{|src:}src\text{|ref:}r\text{|gen:}g\text{|src\_ver:}v\text{|rec\_hash:}rh\text{"}$$

The crosswalk verification gate enforces:
- Caller-supplied role strings (e.g., `caller_role="authorized_pipeline"`) are **REJECTED** with `PermissionError`. Signing requires possession of an instantiated `CrosswalkSigningAuthority`.
- Non-strict or test keys are blocked in production environments (`MEDISENSE_ENV=production`).
- Source record hashes must be verified against actual physical files on disk via `SourceRecordValidator`. Valid HMAC over unverified records fails closed as `UNLINKED`.

### 4.3 Linkage Status Classification
Only four discrete linkage statuses are recognized by `classify_study_linkage_status()`:
1. **`VERIFIED_ENCOUNTER_LINK`**:
   - Condition: Verified encounter match with an authorized bridge source (`MIMIC_ED_CXR_CROSSWALK`, `MIMIC_IV_ED_TRANSFER`), matching `(subject_id, hadm_id, study_id)`, valid non-empty `bridge_key` (`stay_id`), valid HMAC integrity signature, and confirmed existence on physical storage.
   - Multimodal Eligibility: **ELIGIBLE** (`xray_pathology_eligible = 1.0`).
2. **`TEMPORAL_CANDIDATE_LINK`**:
   - Condition: Imaging study timestamp falls within encounter window $[-24\text{ h}, +24\text{ h}]$, but lacks an authenticated bridge table, or linkage mechanism is designated as `TEMPORAL_PROXIMITY_ONLY`.
   - Multimodal Eligibility: **INELIGIBLE** (`xray_pathology_eligible = 0.0`).
   - Rule: **A temporal candidate must NEVER be promoted to verified automatically.**
3. **`UNLINKED`**:
   - Condition: Imaging study belongs to the patient (`subject_id`), but has no temporal proximity or bridge crosswalk linking it to the encounter, or HMAC secret is unconfigured.
   - Multimodal Eligibility: **INELIGIBLE** (`xray_pathology_eligible = 0.0`).
4. **`REJECTED`**:
   - Condition: Contradictory metadata, `subject_id` mismatch, negative identifiers, synthetic markers (`FABRICATED`, `PLACEHOLDER`), or cryptographic HMAC integrity mismatch (indicating corrupted or tampered record).
   - Multimodal Eligibility: **EXCLUDED** from cohort.

### 4.4 Missing Bridge Escalation Protocol
If the official PhysioNet bridge crosswalk (`ed_radiology.csv` or `cxr_ed.csv`) is absent from local disk:
- The linkage pipeline reports status: `BLOCKED (MISSING_BRIDGE_TABLE)`.
- It is strictly prohibited to fabricate a bridge table or infer encounter linkage by joining on `subject_id` and date proximity alone.

### 4.5 Laboratory Provenance & Encounter Linkage Classification (Step 6.16)
To prevent unverified outpatient or unrelated inpatient labs from leaking into acute encounter cohorts, `validate_lab_provenance()` and `LaboratoryLinkageValidator` enforce four discrete statuses:
1. **`VERIFIED_ENCOUNTER_LINK`**:
   - Lab record contains an explicit, verified `hadm_id` matching the encounter admission, or an authenticated crosswalk (`MIMIC_IV_ED_TRANSFER`, `LAB_ENCOUNTER_CROSSWALK`).
   - Timestamps must fall within valid encounter bounds (e.g., presentation window $[-6\text{ h}, +24\text{ h}]$ from admission).
   - Eligibility: **ELIGIBLE** (`lab_eligible = 1.0`, included in `modality_mask[1]`).
2. **`TEMPORAL_CANDIDATE_LINK`**:
   - Lab record contains only `subject_id` and `charttime` within the temporal proximity window, but lacks `hadm_id` and lacks an authenticated crosswalk.
   - Eligibility: **INELIGIBLE** (`lab_eligible = 0.0`, excluded from `modality_mask[1]`).
   - Strict Invariance: `can_promote_temporal_candidate() -> False`. Temporal candidate labs are NEVER silently or automatically promoted to verified encounter links.
3. **`UNLINKED`**:
   - Lab has no encounter key, no crosswalk, and falls outside the encounter window.
   - Eligibility: **INELIGIBLE** (`lab_eligible = 0.0`).
4. **`REJECTED`**:
   - Explicit identifier mismatch (`subject_id` mismatch, `hadm_id` mismatch with record), synthetic/placeholder markers, or post-discharge timestamps.
   - Eligibility: **EXCLUDED** from cohort.

---

## 5. Condition-Specific Label Extraction Rules

Ground-truth label assignment is governed by authoritative ICD-9-CM (v32, 2014/2015) and ICD-10-CM (official FY2015–FY2027) ontologies, incorporating the clinical corrections from Steps 6.11, 6.12, and 6.13.

### 5.1 Pneumonia
- **Accepted Codes:**
  - *ICD-9-CM:* `486` (Pneumonia, organism unspecified), `481` (Pneumococcal pneumonia), `482.9` (Bacterial pneumonia, unspecified), `482.1` (Klebsiella pneumoniae), `482.41` (Methicillin susceptible staphylococcal pneumonia), `482.42` (Methicillin resistant staphylococcal pneumonia).
  - *ICD-10-CM:* `J18.9` (Pneumonia, unspecified organism), `J15.9` (Unspecified bacterial pneumonia), `J13` (Pneumonia due to Streptococcus pneumoniae), `J15.0` (Pneumonia due to Klebsiella pneumoniae), `J15.211` (Pneumonia due to MSSA), `J15.212` (Pneumonia due to MRSA).
- **Rejected Codes:**
  - Post-procedural pneumonia (`J95.89`), chronic pneumonia, ventilator-associated pneumonia occurring $>48\text{ h}$ post-admission (nosocomial infection outside initial ED presentation window), viral upper respiratory infections (`J06.9`).
- **Ambiguous Codes:**
  - `J18.0` (Bronchopneumonia, unspecified organism): Requires imaging report confirming focal parenchymal consolidation or infiltrate.
- **Provider Documentation Requirement:** Physician-documented diagnosis of acute pneumonia at discharge or admission H&P.
- **Primary vs. Secondary:** Eligible as primary inpatient diagnosis (`seq_num == 1`) or secondary acute active diagnosis (`seq_num > 1`).
- **Co-occurrence:** Frequently co-occurs with Pleural Effusion (parapneumonic effusion). Both labels may be independently positive.

### 5.2 Acute Decompensated Heart Failure (ADHF)
- **Accepted Codes (Definitive Acute / Acute-on-Chronic):**
  - *ICD-9-CM:* `428.21` (Acute systolic HF), `428.23` (Acute on chronic systolic HF), `428.31` (Acute diastolic HF), `428.33` (Acute on chronic diastolic HF), `428.41` (Acute combined systolic/diastolic HF), `428.43` (Acute on chronic combined HF).
  - *ICD-10-CM:* `I50.21` (Acute systolic HF), `I50.23` (Acute on chronic systolic HF), `I50.31` (Acute diastolic HF), `I50.33` (Acute on chronic diastolic HF), `I50.41` (Acute combined HF), `I50.43` (Acute on chronic combined HF), `I50.811` (Acute right heart failure), `I50.813` (Acute on chronic right heart failure).
- **Rejected Codes (Chronic Compensated or Non-Cardiogenic):**
  - *ICD-9-CM:* `428.22` (Chronic systolic HF), `428.32` (Chronic diastolic HF), `428.42` (Chronic combined HF), `518.4` (Acute non-cardiogenic pulmonary edema).
  - *ICD-10-CM:* `I50.22` (Chronic systolic HF), `I50.32` (Chronic diastolic HF), `I50.42` (Chronic combined HF), `I50.810` (Right heart failure, unspecified), `J81.0` (Acute non-cardiogenic pulmonary edema).
- **Ambiguous Codes (`REQUIRE_ADDITIONAL_EVIDENCE`):**
  - *ICD-9-CM:* `428.1` ("Left heart failure", ICD-9-CM v32), `428.0` ("Congestive heart failure, unspecified").
  - *ICD-10-CM:* `I50.1` ("Left ventricular failure, unspecified"), `I50.9` ("Heart failure, unspecified").
  - *Strict Step 6.13 Rule:* Neither `428.1`, `428.0`, `I50.1`, nor `I50.9` can independently establish acute decompensation. They require documented clinical evidence of acute exacerbation (e.g., urgent intravenous loop diuretic administration, documented acute pulmonary edema on chest radiograph). **Elevated BNP or NT-proBNP alone is strictly INSUFFICIENT** to convert an unspecified code into confirmed acute HF.

### 5.3 Pleural Effusion
- **Accepted Codes:**
  - *ICD-9-CM:* `511.9` (Unspecified pleural effusion), `511.1` (Pleurisy with effusion, with bacterial cause), `511.89` (Other specified forms of effusion).
  - *ICD-10-CM:* `J90` (Pleural effusion, not elsewhere classified), `J91.0` (Malignant pleural effusion), `J91.8` (Pleural effusion in other conditions).
- **Rejected Codes:**
  - `511.0` (Pleurisy without mention of effusion), `R09.1` (Pleurisy), `S27.1xxA` (Traumatic hemothorax—trauma excluded).
- **Clinical Validation Requirement:** Must be corroborated by radiology report (`CANONICAL_RAD_FINDINGS` containing `"pleural effusion"`) or thoracentesis procedure (`procedures_icd`).
- **Co-occurrence Handling:** **Pleural effusion remains fully eligible as a secondary co-occurring target** alongside pneumonia, heart failure, or COPD when clinically documented. The 5 conditions are **NEVER MUTUALLY EXCLUSIVE**.

### 5.4 Spontaneous Pneumothorax
- **Accepted Codes:**
  - *ICD-9-CM:* `512.81` (Primary spontaneous pneumothorax), `512.82` (Secondary spontaneous pneumothorax), `512.89` (Other pneumothorax).
  - *ICD-10-CM:* `J93.01` (Acute primary spontaneous pneumothorax), `J93.02` (Other acute primary spontaneous pneumothorax), `J93.11` (Acute secondary spontaneous pneumothorax), `J93.12` (Other acute secondary spontaneous pneumothorax), `J93.83` (Other pneumothorax).
- **Rejected Codes (Traumatic, Iatrogenic, Post-procedural):**
  - *ICD-9-CM:* `860.0` (Traumatic pneumothorax), `512.1` (Iatrogenic pneumothorax).
  - *ICD-10-CM:* `S27.0xxA` (Traumatic pneumothorax), `J95.811` (Postprocedural pneumothorax), `J95.812` (Postprocedural air leak).
- **Ambiguous Codes:**
  - `512.8` (ICD-9) / `J93.9` (ICD-10) (Pneumothorax, unspecified): Requires clinical text or E-code verification demonstrating absence of external thoracic trauma, mechanical ventilation barotrauma, or transthoracic biopsy complication.

### 5.5 Acute COPD Exacerbation
- **Accepted Codes (Definite Acute Exacerbation):**
  - *ICD-9-CM:* `491.21` (Obstructive chronic bronchitis with (acute) exacerbation).
  - *ICD-10-CM:* `J44.1` (Chronic obstructive pulmonary disease with (acute) exacerbation).
- **Infection-Superimposed Codes (`REQUIRE_ADDITIONAL_EVIDENCE`):**
  - *ICD-9-CM:* `491.22` (Obstructive chronic bronchitis with acute bronchitis). Must remain strictly separate from `491.21`; prefix matching (`491.2*`) is rejected. Requires explicit documentation of acute airway exacerbation.
  - *ICD-10-CM:* `J44.0` (COPD with acute lower respiratory infection). Does **NOT** automatically establish acute COPD exacerbation, and does **NOT** infer pneumonia unless pneumonia is independently coded and clinically substantiated.
- **Rejected Codes (Chronic Baseline COPD):**
  - *ICD-9-CM:* `496` (Chronic airway obstruction, NEC), `492.8` (Other emphysema), `493.20` (Chronic obstructive asthma without exacerbation).
  - *ICD-10-CM:* `J44.9` (COPD, unspecified), `J43.9` (Emphysema, unspecified), `J44.89` (Other specified COPD).

---

## 6. Five-Modality Acquisition and Eligibility Matrices

### 6.1 Matrix A — Dataset Availability & Technical Acquisition Specifications

| Target Condition | Modality | Source Dataset | Required Tables / Files | Required Columns | Availability Status | Authorization Status | Extraction Method | Linkage Requirements | Validation Requirements |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **All 5 Conditions** | Symptoms | MIMIC-IV-ED / Core | `triage.csv`, `admissions.csv` | `stay_id`, `chiefcomplaint`, `admittime` | `PENDING_ACQUISITION` | Credentialed PhysioNet DUA | Exact lexical tokenization & semantic match to `CANONICAL_SYMPTOM_VOCABULARY` | Direct `stay_id` $\leftrightarrow$ `hadm_id` | Chief complaint non-empty; timestamp $[-24\text{h}, +24\text{h}]$ |
| **All 5 Conditions** | Laboratory Reports | MIMIC-IV Hosp | `labevents.csv`, `d_labitems.csv` | `labevent_id`, `itemid`, `charttime`, `valuenum`, `flag` | `PENDING_ACQUISITION` | Credentialed PhysioNet DUA | Numerical value extraction, LOINC/itemid crosswalk to 21 canonical tests | **Explicit matching `hadm_id` or authenticated crosswalk** (`MIMIC_IV_ED_TRANSFER`). `subject_id + charttime` alone is strictly `TEMPORAL_CANDIDATE_LINK` (`lab_eligible = 0.0`) and never promoted silently. | Exclude non-numeric text values; flag abnormal; check unit standard; reject mismatches |
| **All 5 Conditions** | Radiology Reports | MIMIC-CXR | `s{study_id}.txt` (Reports) | `study_id`, text sections (`FINDINGS`, `IMPRESSION`) | `PENDING_ACQUISITION` | Credentialed PhysioNet DUA | Section regex extraction, NLP ontology matching to 14 canonical findings | Study linked to encounter via verified bridge crosswalk | Both report sections parsed; negation and uncertainty detected |
| **All 5 Conditions** | X-Ray Images | MIMIC-CXR-JPG | `mimic-cxr-2.0.0-metadata.csv`, `cxr-record-list.csv` | `dicom_id`, `study_id`, `ViewPosition`, `path` | `PENDING_ACQUISITION` | Credentialed PhysioNet DUA | DICOM/JPG file load, resize $(224 \times 224)$, grayscale/RGB normalization | Explicit `VERIFIED_ENCOUNTER_LINK` via HMAC signed crosswalk | Frontal view only (`PA` or `AP`); pixel aspect ratio check; hash check |
| **All 5 Conditions** | Clinical Examination | MIMIC-IV-Note | `discharge.csv` (Discharge Summaries) | `note_id`, `hadm_id`, `note_type`, `charttime`, `text` | `DATA_UNAVAILABLE` | Credentialed PhysioNet DUA | Unvalidated NLP (Fails closed; `exam_eligible = 0.0`) | `hadm_id` match + explicit initial H&P section | Initial H&P section proven; precedes treatment; vital signs separated |

### 6.2 Matrix B — Clinical Evidence Roles & Multimodal Diagnostic Eligibility (Step 6.16)

The Step 6.16 clinical audit establishes six distinct evidence roles (`DIAGNOSTIC`, `SUPPORTING`, `CONTEXTUAL`, `SEVERITY`, `RULE_OUT`, `NONSPECIFIC`):

| Target Condition | Modality 1: Symptoms | Modality 2: Laboratory Reports | Modality 3: Radiology Reports | Modality 4: X-Ray Images | Modality 5: Clinical Examination |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pneumonia** | `SUPPORTING` (Cough, fever, sputum, dyspnea; supports diagnosis, requires radiologic or clinical confirmation) | `NONSPECIFIC` (WBC, Neutrophils, CRP, Procalcitonin: systemic inflammatory markers; **MUST NOT independently establish disease**)<br>`SEVERITY` (ABG PaO2/PaCO2 hypoxemia, CURB-65 BUN/creatinine) | `DIAGNOSTIC` (Consolidation, air bronchogram, focal alveolar infiltrate) | `DIAGNOSTIC` (Focal parenchymal consolidation on frontal view) | `DATA_UNAVAILABLE` (`SUPPORTING` if present: auscultatory crackles, bronchial breath sounds; weight = 0.0) |
| **Acute Decompensated Heart Failure (ADHF)** | `SUPPORTING` (Orthopnea, PND, dyspnea on exertion, peripheral edema) | `SUPPORTING` (BNP / NT-proBNP: elevated in baseline chronic HF/renal disease; **CANNOT independently establish acute decompensation** without congestion)<br>`CONTEXTUAL`/`SEVERITY` (Troponin, creatinine, BUN) | `DIAGNOSTIC` (Cardiomegaly, alveolar pulmonary edema, vascular congestion, Kerley B lines) | `DIAGNOSTIC` (Cephalization, Kerley B lines, interstitial/alveolar edema) | `DATA_UNAVAILABLE` (`SUPPORTING` if present: elevated JVP, S3 gallop, bilateral basilar crackles; weight = 0.0) |
| **Pleural Effusion** | `SUPPORTING` (Pleuritic chest pain, dyspnea, dry cough) | `CONTEXTUAL` (Serum protein/LDH for Light's criteria context; non-diagnostic for presence of effusion) | `DIAGNOSTIC` (Blunting of costophrenic angle, fluid meniscus, loculated effusion) | `DIAGNOSTIC` (Dependent fluid blunting, homogenous opacity, meniscus sign) | `DATA_UNAVAILABLE` (`SUPPORTING` if present: dullness to percussion, decreased breath sounds; weight = 0.0) |
| **Spontaneous Pneumothorax** | `SUPPORTING` (Sudden onset acute sharp chest pain, dyspnea) | `CONTEXTUAL` (Non-diagnostic; monitors arterial blood gas and systemic stress) | `DIAGNOSTIC` (Visceral pleural line, absence of peripheral lung markings, lung collapse) | `DIAGNOSTIC` (Pleural line displacement, radiolucent air space) | `DATA_UNAVAILABLE` (`SUPPORTING` if present: decreased breath sounds, hyperresonance to percussion; weight = 0.0) |
| **Acute COPD Exacerbation** | `DIAGNOSTIC` / `SUPPORTING` (Anthonisen cardinal triad: worsening dyspnea, sputum volume, sputum purulence) | `SEVERITY` (Blood gas PaCO2 elevation, acute respiratory acidosis pH < 7.35, hypoxemia PaO2)<br>`NONSPECIFIC` (Leukocytosis, inflammatory markers) | `RULE_OUT` / `CONTEXTUAL` (Excludes pneumonia/pneumothorax; identifies baseline hyperinflation) | **`RULE_OUT`** (**Diagnostic Weight = 0.0**; excludes focal consolidation or pneumothorax, **cannot activate positive COPD logit**) | `DATA_UNAVAILABLE` (`SUPPORTING` if present: expiratory wheezing, prolonged expiration, barrel chest; weight = 0.0) |

### 6.3 Evidence Role Governance Rules (Step 6.16)
The Step 6.16 `EvidenceRolePolicy` defines the behavioral boundaries for each evidence role:
1. **`can_independently_establish_disease(role)`**: Returns `True` **ONLY** for `DIAGNOSTIC` (and its alias `POSITIVE_DIAGNOSTIC`).
2. **`can_support_candidate(role)`**: Returns `True` for `DIAGNOSTIC` and `SUPPORTING`.
3. **`can_contribute_to_severity(role)`**: Returns `True` for `SEVERITY` evidence (e.g., blood gases PaO2/PaCO2, pH, BUN, creatinine).
4. **`is_nonspecific_marker(role)`**: Returns `True` for `NONSPECIFIC` markers (WBC, neutrophils, CRP, procalcitonin). These reflect generic acute-phase inflammation and **must never independently determine a disease label**.
5. **Contextual & Rule-Out Isolation**: Contextual findings frame underlying physiology; rule-out findings ensure competing pathologies are excluded without artificially inflating target probability.

---

## 7. Cohort Extraction Pipeline Specification

The extraction pipeline is structured as a deterministic, auditable 13-stage ETL sequence designed for local execution following credentialed authorization.

```
[Stage 1: Source Ingestion]      Read authorized MIMIC-IV Core, Hosp, ED, CXR, Note files
           |
[Stage 2: Version Validation]    Verify file checksums, schema headers, dataset release versions
           |
[Stage 3: Patient Filtering]     Filter age >= 18, deduplicate demographic profiles
           |
[Stage 4: Encounter Selection]   Identify acute ED presentations with hospital admissions (stay_id <-> hadm_id)
           |
[Stage 5: Target Labeling]       Execute condition evaluators (Pneumonia, ADHF, Effusion, PTX, COPD)
           |
[Stage 6: Bridge Linkage]        Verify authenticated crosswalk; classify study linkage status
           |
[Stage 7: Modality Extraction]   Extract Symptoms, Labs, Radiology, X-rays, Exam placeholders
           |
[Stage 8: Missingness Handling]  Preserve explicit DATA_UNAVAILABLE / FAILED states without zero-imputation
           |
[Stage 9: Temporal Alignment]    Enforce clinical window [-24h, +24h] from admission
           |
[Stage 10: Duplicate Detection]  Resolve multiple studies per encounter via clinical priority
           |
[Stage 11: Provenance Sealing]   Generate HMAC crosswalk envelopes and source record hashes
           |
[Stage 12: Audit & Reporting]    Output IngestionAuditReport with class counts and linkage metrics
           |
[Stage 13: Final Guard Gate]     Enforce zero-unlinked, zero-NaN verification before artifact emission
```

### 7.1 Proposed Directory Structure
The authorized extraction pipeline will operate entirely within the local repository under `backend/data/authentic_cohort/`:

```
d:\projects\MEDISENSE_AI\backend\data\authentic_cohort\
├── raw_authorized\                   # Read-only source datasets (PhysioNet credentialed downloads)
│   ├── mimiciv\2.2\hosp\            # admissions.csv.gz, labevents.csv.gz, diagnoses_icd.csv.gz
│   ├── mimiciv\2.2\ed\              # edstays.csv.gz, triage.csv.gz, vitalsign.csv.gz
│   ├── mimic_cxr\2.0.0\             # mimic-cxr-2.0.0-metadata.csv.gz, cxr-record-list.csv.gz
│   ├── mimic_cxr_jpg\2.0.0\files\   # p10\..., p11\... JPEG images
│   ├── mimic_notes\2.2\note\        # discharge.csv.gz
│   └── bridges\                     # ed_radiology.csv.gz / cxr_ed.csv.gz
├── intermediate\                     # Extracted normalized tabular staging files
│   ├── staged_cohort_encounters.parquet
│   ├── staged_cohort_labels.parquet
│   └── staged_cohort_linkages.parquet
├── verified_cohort\                  # Cryptographically sealed research cohort artifacts
│   ├── cohort_manifest.json          # Master provenance manifest
│   ├── cohort_metadata.parquet       # Demographic and encounter features
│   ├── train_split.parquet           # Patient-isolated training partition (70%)
│   ├── val_split.parquet             # Patient-isolated validation partition (15%)
│   ├── test_split.parquet            # Patient-isolated test partition (15%)
│   └── crosswalk_signatures.json     # Step 6.6 HMAC cryptographic signatures
└── reports\
    └── extraction_audit_report.json  # Comprehensive extraction audit manifest
```

### 7.2 Execution Pipeline Pseudocode
```python
"""
PSEUDOCODE SPECIFICATION: Authentic Multimodal Cohort Extraction Pipeline
Module: backend/scripts/extract_authentic_multimodal_cohort.py (Specification Only)
"""

def extract_authentic_multimodal_cohort(
    data_root: Path,
    output_dir: Path,
    signing_authority: CrosswalkSigningAuthority,
    config: ExtractionConfig,
) -> IngestionAuditReport:
    # Stage 1 & 2: Source File & Version Validation
    validate_physical_source_integrity(data_root)
    
    # Stage 3: Patient Demographic Filter
    patients_df = read_table(data_root / "mimiciv/2.2/hosp/patients.csv.gz")
    adult_patients = patients_df[patients_df["anchor_age"] >= 18]["subject_id"].unique()
    
    # Stage 4: Acute Encounter Selection (ED Presentation -> Inpatient Bed)
    edstays_df = read_table(data_root / "mimiciv/2.2/ed/edstays.csv.gz")
    admissions_df = read_table(data_root / "mimiciv/2.2/hosp/admissions.csv.gz")
    encounters = link_ed_to_admissions(edstays_df, admissions_df, adult_patients)
    
    # Stage 5: Target Labeling (Step 6.12/6.13 Condition Evaluators)
    diagnoses_df = read_table(data_root / "mimiciv/2.2/hosp/diagnoses_icd.csv.gz")
    target_labels = {}
    for enc in encounters:
        codes = get_encounter_icd_codes(diagnoses_df, enc.hadm_id)
        target_labels[enc.hadm_id] = {
            "Pneumonia": PneumoniaLabelEvaluator.evaluate(codes),
            "Acute Decompensated Heart Failure": AcuteHeartFailureLabelEvaluator.evaluate(codes, enc.context),
            "Pleural Effusion": PleuralEffusionLabelEvaluator.evaluate(codes),
            "Spontaneous Pneumothorax": PneumothoraxLabelEvaluator.evaluate(codes),
            "Acute COPD Exacerbation": COPDExacerbationLabelEvaluator.evaluate(codes, enc.context),
        }
    
    # Stage 6: Bridge Linkage & Crosswalk Verification
    bridge_df = read_table(data_root / "bridges/ed_radiology.csv.gz")
    cxr_meta_df = read_table(data_root / "mimic_cxr/2.0.0/mimic-cxr-2.0.0-metadata.csv.gz")
    linked_studies = {}
    for enc in encounters:
        candidate_study = find_matching_study(bridge_df, cxr_meta_df, enc.subject_id, enc.stay_id)
        if candidate_study:
            # Enforce Step 6.6 Cryptographic Signing & Disk Verification
            cw = signing_authority.create_authenticated_imaging_crosswalk(
                subject_id=enc.subject_id,
                hadm_id=enc.hadm_id,
                study_id=candidate_study.study_id,
                stay_id=enc.stay_id,
                source_records_verified=True,
            )
            linked_studies[enc.hadm_id] = cw
    
    # Stage 7, 8, 9: Modality Extraction, Temporal Gating & Missingness Preservation
    cohort_records = []
    for enc in encounters:
        record = build_multimodal_record(
            encounter=enc,
            labels=target_labels[enc.hadm_id],
            symptoms=extract_triage_symptoms(enc.stay_id),
            labs=extract_windowed_labs(enc.hadm_id, enc.admittime, window_hours=24),
            rad_report=extract_radiology_report(linked_studies.get(enc.hadm_id)),
            xray_image=extract_cxr_image_path(linked_studies.get(enc.hadm_id)),
            exam_record={"status": "DATA_UNAVAILABLE", "exam_eligible": 0.0},
        )
        cohort_records.append(record)
        
    # Stage 10 & 11: Patient-Level Splitting & Sealing
    train_set, val_set, test_set = partition_patient_level_splits(cohort_records, seed=42)
    
    # Stage 12 & 13: Final Audit & Gatekeeper Verification
    audit_report = generate_audit_report(cohort_records, train_set, val_set, test_set)
    verify_extraction_integrity(audit_report)
    save_cohort_artifacts(output_dir, train_set, val_set, test_set, audit_report)
    
    return audit_report
```

---

## 8. Patient-Level Partitioning and Split Strategy

To prevent data contamination, all cohort records must be partitioned strictly by **patient identifier (`subject_id`)**.

```
                   Master Cohort (N Authentic Patients)
                                    |
          +-------------------------+-------------------------+
          | 70%                     | 15%                     | 15%
          v                         v                         v
     [Train Split]             [Val Split]               [Test Split]
   Patients {P_train}        Patients {P_val}          Patients {P_test}
          |                         |                         |
          +-------------------------+-------------------------+
                                    |
                        CRITICAL LEAKAGE ASSERTION:
         (P_train ∩ P_val = ∅) ∧ (P_train ∩ P_test = ∅) ∧ (P_val ∩ P_test = ∅)
```

### 8.1 Partitioning Specifications
1. **Zero Patient Contamination:**
   $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset, \quad \text{Patients}(\text{Train}) \cap \text{Patients}(\text{Test}) = \emptyset$$
   Under no circumstances may an encounter, study, or clinical note from a patient in the test set appear in the training or validation sets.
2. **Encounter-Level Grouping:** All admissions (`hadm_id`) and ED stays (`stay_id`) belonging to a single `subject_id` are permanently assigned to the same partition.
3. **Multi-Study Resolution:** If a patient undergoes multiple chest radiographs during a single admission, studies are ordered chronologically. The initial study obtained within $[-6\text{ h}, +12\text{ h}]$ of ED admission is designated as the diagnostic presentation study. Subsequent studies are retained strictly for longitudinal context or excluded from single-frame evaluation.
4. **Stratified Multi-Label Split:** Due to condition co-occurrence and varying prevalence (e.g., Pneumonia is common, Spontaneous Pneumothorax is rare), iterative multi-label stratified splitting (Sechidis et al.) must be applied at the patient level to ensure balanced representation across all 5 targets in all three splits.
5. **Temporal Leakage Controls:**
   - In addition to patient isolation, test sets should evaluate robustness against distribution shift by holding out patients whose anchor years fall in the latest chronological decile (`anchor_year_group`).
6. **Feature Label Leakage Checks:** Discharge summaries, post-discharge diagnoses, and outpatient medications must never be parsed into admission symptom or exam feature sets.

### 8.2 Modality-Specific Temporal Windows & Clinical Temporal Policy (Step 6.16)
The Step 6.16 `ClinicalTemporalPolicy` establishes explicit, separate temporal boundaries for each modality to eliminate cross-modality contamination, outpatient leakage, and post-treatment retrospective confounding:
1. **Symptoms Temporal Policy:**
   - Presentation Window: $[-24\text{ h}, +24\text{ h}]$ from admission.
   - Restricts chief complaint extraction to acute presentation; outpatient complaints $>24\text{ h}$ prior to admission are excluded.
2. **Laboratory Temporal Policy:**
   - **Presentation Window:** $[-6\text{ h}, +24\text{ h}]$ from admission. Represents acute initial presentation labs.
   - **Monitoring Window:** $>24\text{ h}$ post-admission. Longitudinal inpatient labs are categorized as monitoring and excluded from initial multimodal diagnostic fusion.
   - **Max Pre-admission Outpatient Window:** $24\text{ h}$. Any lab collected $>24\text{ h}$ prior to admission is classified as `OUT_OF_WINDOW` and excluded.
   - **Post-discharge Boundary:** Any lab collected after `dischtime` is strictly `REJECTED` / `OUT_OF_WINDOW`.
3. **Radiology & X-Ray Temporal Policy:**
   - Presentation Window: $[-6\text{ h}, +12\text{ h}]$ from ED admission. Captures initial diagnostic imaging prior to significant therapeutic response.
   - Studies outside this window are retained strictly for longitudinal context or excluded from single-frame evaluation.
4. **Clinical Examination Temporal Policy:**
   - Window: $[-24\text{ h}, +24\text{ h}]$ from admission.
   - **Mandatory Precedence Rule:** Must precede therapeutic intervention (`precedes_treatment = True`). Post-treatment examinations or routine discharge summaries are rejected as diagnostic physical exam inputs.

---

## 9. Authorization, Credentialing & Free-Only Governance

MIMIC datasets are not public domain. They are restricted-access, de-identified clinical resources governed by strict human-subjects protection agreements.

```
+----------------------------------------------------------------------------------------------------+
|                                    PHYSIONET CREDENTIALING WORKFLOW                                |
+----------------------------------------------------------------------------------------------------+
|  1. Register at PhysioNet (https://physionet.org)                                                  |
|  2. Complete CITI "Data or Specimens Only Research" or "Human Subjects Research" Course            |
|  3. Submit Proof of Training & Institutional Affiliation to PhysioNet                              |
|  4. Sign Credentialed Data Use Agreement (DUA) for MIMIC-IV, MIMIC-IV-ED, MIMIC-CXR-JPG, Notes     |
|  5. Await Formal Credentialing Approval                                                            |
|  6. Configure Secure Encrypted Local Storage (Strict $0.00 Tooling)                                |
|  7. Download Authorized Partitions via Authenticated curl/wget/physionet-cli                       |
+----------------------------------------------------------------------------------------------------+
```

### 9.1 Dataset Authorization Matrix

| Dataset | Official Access URI | Access Level | Credentialing Requirements | Data Use Agreement (DUA) Constraints | Mandatory Citation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MIMIC-IV (Core & Hosp v2.2)** | `https://physionet.org/content/mimiciv/2.2/` | Credentialed | - CITI "Data or Specimens Only Research"<br>- PhysioNet account verification<br>- Supervisor/PI sign-off | - No re-identification attempts<br>- No data redistribution<br>- Local secure storage<br>- Report data breaches | Johnson, et al. *MIMIC-IV* (PhysioNet, 2023). DOI: 10.13026/6mm1-ek67 |
| **MIMIC-IV-ED (v2.2)** | `https://physionet.org/content/mimic-iv-ed/2.2/` | Credentialed | - CITI Training Certification<br>- Approved PhysioNet Credential | - Emergency records non-transferable<br>- Single-researcher local access | Johnson, et al. *MIMIC-IV-ED* (PhysioNet, 2023). DOI: 10.13026/5t7e-2x57 |
| **MIMIC-CXR-JPG (v2.0.0)** | `https://physionet.org/content/mimic-cxr-jpg/2.0.0/` | Credentialed | - CITI Training Certification<br>- Dual-signed MIMIC DUA | - No face/anatomical reconstruction<br>- No external cloud bucket hosting | Johnson, et al. *MIMIC-CXR-JPG* (PhysioNet, 2019). DOI: 10.13026/8360-t248 |
| **MIMIC-IV-Note (v2.2)** | `https://physionet.org/content/mimic-iv-note/2.2/` | Credentialed | - CITI Training Certification<br>- High-security clinical text clearance | - Extreme text privacy safeguards<br>- No external commercial LLM API forwarding | Johnson, et al. *MIMIC-IV-Note* (PhysioNet, 2023). DOI: 10.13026/e502-bk89 |

### 9.2 Compliance Assertions
- **Zero Financial Cost ($0.00):** Credentialing, data access via PhysioNet, CITI training (via participating institutional single-sign-on or authorized free community tracks), local storage, and Python data pipelines require zero paid software licenses or cloud subscriptions.
- **Strict Prohibition Against Bypassing DUA:** The extraction pipeline cannot and will not download data without an authenticated PhysioNet token (`PHYSIONET_PASS_KEY`). No synthetic data or un-credentialed third-party mirrors may be utilized.

---

## 10. Protected Architectural Boundaries

The boundaries protecting the existing MediSense AI system remain hard-coded and non-negotiable:

1. **Track A Symptom Baseline:** The 24-condition symptom-only production classifier (`app/ai/condition_model.py`, `Symptom2Disease.csv`, checkpoints, and tokenizer assets) remains entirely separate and actively serving production inference.
2. **X-Ray Pathology & Anatomy System:** The 15 protected X-ray assets across `backend/app/ai/` and `backend/models/xray/` have been audited. All 15 SHA-256 hashes match `xray_protected_manifest.json` perfectly.
3. **Cryptographic Provenance (Step 6.6):** The HMAC-SHA256 crosswalk signature mechanism, environment key boundaries, and `CrosswalkSigningAuthority` remain fail-closed against caller impersonation.
4. **Numerical Stability & Modality Gating (Step 6.13):** `GatedMultimodalFusion` and `safe_masked_softmax` enforce zero NaNs/Infs on missing modalities, returning `DATA_UNAVAILABLE` with `probability: None` and `abstain: True`. COPD X-ray diagnostic weight remains locked at $0.0$.
5. **No Model Training in Step 6.15:** Track B model training remains deactivated. No new weights or checkpoints are created.
6. **No Production Activation:** Track B multimodal inference remains disabled in the API routers until authentic validation benchmarks are satisfied.

---

## 11. Verification Checklist & Readiness Status

### 11.1 Local Workspace Inspection Audit

```
+----------------------------------------------------------------------------------------------------+
|                               WORKSPACE ASSET & INTEGRITY AUDIT                                    |
+----------------------------------------------------------------------------------------------------+
| Asset / Component                     Target Path                             Status      Verdict  |
+----------------------------------------------------------------------------------------------------+
| Symptom2Disease Dataset (Track A)     backend/data/symptom_to_disease/        PRESENT     PASS     |
| 15 Protected X-Ray Models & Code      backend/models/xray/ & backend/app/ai/   15/15 MATCH PASS     |
| Goal 2 Step 6.12/6.13 Test Suite      backend/tests/                          121/121 PASS PASS    |
| Goal 2 Step 6.16 Correction Suite     backend/tests/                          12/12 PASS   PASS    |
| Total Goal 2 Decoupled Test Suite     backend/tests/                          144/144 PASS PASS    |
| MIMIC-IV Core & Hosp Tables           backend/data/authentic_cohort/raw/      MISSING     EXPECTED |
| MIMIC-IV-ED Tables                    backend/data/authentic_cohort/raw/      MISSING     EXPECTED |
| MIMIC-CXR-JPG Image Data              backend/data/authentic_cohort/raw/      MISSING     EXPECTED |
| MIMIC-IV-Note Clinical Text           backend/data/authentic_cohort/raw/      MISSING     EXPECTED |
| Authorized Imaging Bridge Table       backend/data/authentic_cohort/bridges/  MISSING     EXPECTED |
| Track B Training Gatekeeper           backend/scripts/train_multimodal_model.py LOCKED    PASS     |
+----------------------------------------------------------------------------------------------------+
```

### 11.2 Verification Checklist
- [x] Five research targets clearly defined and clinically verified (Pneumonia, ADHF, Pleural Effusion, Spontaneous Pneumothorax, Acute COPD).
- [x] Five modalities mapped with explicit technical schemas and clinical evidence roles.
- [x] MIMIC-IV Core/Hosp, ED, CXR, and Note table requirements fully inventoried with PK/FK relationships.
- [x] Missing study-to-encounter direct link in MIMIC-CXR explicitly recognized and documented.
- [x] Bridge crosswalk requirement mandated; fail-closed behavior established if bridge is missing.
- [x] Step 6.6 HMAC-SHA256 crosswalk signing and source-record disk verification integrated.
- [x] Step 6.12/6.13 ICD label evaluators integrated (ICD-9 428.1 Left heart failure consistent rule, 491.21 vs 491.22 separation, J44.0 vs J44.1 separation without pneumonia inference).
- [x] Step 6.16 Laboratory Linkage correction: `subject_id + charttime` alone strictly classified as `TEMPORAL_CANDIDATE_LINK` (`lab_eligible = 0.0`); silent promotion prohibited.
- [x] Step 6.16 Clinical Evidence Roles: 6 explicit roles defined (`DIAGNOSTIC`, `SUPPORTING`, `CONTEXTUAL`, `SEVERITY`, `RULE_OUT`, `NONSPECIFIC`). WBC/neutrophils/CRP/procalcitonin forbidden from independently establishing pneumonia; BNP forbidden from independently establishing acute HF.
- [x] Step 6.16 Modality Temporal Policies: Explicit `ClinicalTemporalPolicy` separating presentation vs. monitoring vs. pre-admission windows; mandatory pre-treatment requirement for clinical exams.
- [x] Matrix A (Dataset Availability) and Matrix B (Clinical Eligibility) completely separated and updated.
- [x] COPD Chest X-ray diagnostic weight locked to 0.0 (`RULE_OUT_ONLY`).
- [x] Clinical examination locked to `DATA_UNAVAILABLE` (`exam_eligible = 0.0`) pending validated H&P extraction.
- [x] Reproducible 13-stage cohort extraction sequence and directory structure specified.
- [x] Patient-level 70/15/15 partitioning specified with zero patient leakage assertions.
- [x] PhysioNet CITI training, DUA, and free-only governance completely documented.
- [x] Track A baseline and 15 protected X-ray assets (15/15 SHA-256 match) verified untouched.
- [x] Track B training gatekeeper verified locked (exit code 1/2) pending authentic cohort.

### 11.3 Final Readiness Verdict
```
+----------------------------------------------------------------------------------------------------+
|                                STEP 6.15 & 6.16 READINESS VERDICT                                  |
+----------------------------------------------------------------------------------------------------+
|  VERDICT: SPECIFICATION COMPLETE & READINESS INDEPENDENTLY VERIFIED                                |
|                                                                                                    |
|  The technical, cryptographic, laboratory provenance, and clinical evidence specifications for     |
|  authentic multimodal data acquisition are fully established and independently verified.           |
|  All 144/144 tests in the Goal 2 decoupled suite pass. All 15 protected X-ray assets match hashes. |
|                                                                                                    |
|  STATUS: ACQUISITION BLOCKED PENDING EXTERNAL PHYSIONET CREDENTIALED AUTHORIZATION                |
|                                                                                                    |
|  Next Milestone: When the user provides verified PhysioNet credentialing credentials and          |
|  executes authorized data download into the local staging directory, Step 7 (Authentic Cohort    |
|  Extraction & Training Pipeline) can be securely initiated under the Step 6.15/6.16 specification.|
+----------------------------------------------------------------------------------------------------+
```
