"""Multimodal Condition Prediction Architecture for MediSense AI.

A genuine multimodal neural network architecture designed to learn from multiple
patient modalities (symptoms, laboratory values, radiology reports, and eligible X-ray features)
with principled missing-modality masks and gated representation fusion.

CLINICAL SAFETY CONSTRAINTS:
1. Anatomy classification is NOT pathology classification. Anatomy is used solely for
   routing and anatomical context; only verified specialist pathology model outputs constitute
   disease-related features.
2. Missing modalities are explicitly masked out; missing information is NEVER imputed as normal.
3. When all modalities are absent, the model returns an explicit INSUFFICIENT_EVIDENCE result,
   preventing untrained biases from generating phantom diagnoses.
4. Multimodal training requires authentic, patient-linked EHR datasets. When unlinked data is
   detected, training remains strictly BLOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# Supported canonical laboratory tests (aligned with report_nlp.py TARGET_TESTS)
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


# ==============================================================================
# DATA CONTRACT & SCHEMA DEFINITIONS
# ==============================================================================

@dataclass
class ClinicalSymptomObservation:
    symptom: str
    canonical: str
    state: str = "PRESENT"  # PRESENT, ABSENT, UNKNOWN, HISTORICAL
    duration_days: Optional[float] = None
    severity: str = "UNKNOWN"  # MILD, MODERATE, SEVERE, UNKNOWN
    qualifiers: list[str] = field(default_factory=list)
    body_area: Optional[str] = None


@dataclass
class StructuredLabParameter:
    test_name: str
    canonical_test_name: str
    value: float
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    flag: str = "NORMAL"  # HIGH, LOW, NORMAL, UNKNOWN
    source_file: Optional[str] = None
    report_id: Optional[str] = None


@dataclass
class StructuredRadiologyFinding:
    finding: str
    canonical_finding: str
    state: str = "PRESENT"  # PRESENT, ABSENT, POSSIBLE, UNCERTAIN, NOT_ASSESSED, EXTRACTION_FAILED, UNSUPPORTED
    anatomical_region: str = "chest"
    examination: Optional[str] = None
    source_file: Optional[str] = None
    report_id: Optional[str] = None


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


@dataclass
class MultimodalPatientRecord:
    """Canonical multimodal EHR patient record representation."""
    subject_id: str
    symptoms: list[ClinicalSymptomObservation] = field(default_factory=list)
    laboratory: list[StructuredLabParameter] = field(default_factory=list)
    radiology: list[StructuredRadiologyFinding] = field(default_factory=list)
    clinical_examination: list[StructuredClinicalFinding] = field(default_factory=list)
    xray: Optional[XRaySpecialistOutput] = None
    target_condition: Optional[str] = None
    provenance: dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# MODALITY ENCODERS
# ==============================================================================

class SymptomModalityEncoder(nn.Module):
    """Encodes clinical symptom observations (presence, absence, duration, severity)."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.symptom_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Observation features: [state (+1 present, -1 absent, 0 unk), normalized_duration, severity_code]
        self.dense_proj = nn.Linear(embed_dim + 3, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, token_ids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        embeds = self.symptom_embeddings(token_ids)
        combined = torch.cat([embeds, features], dim=-1)
        h = F.relu(self.dense_proj(combined))
        mask = (token_ids != 0).unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return self.layer_norm(pooled)


class LaboratoryModalityEncoder(nn.Module):
    """Encodes structured laboratory panels with normalized values, units, reference limits, and flags."""

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
    """Encodes structured qualitative findings and examination metadata from radiology reports."""

    def __init__(self, num_findings: int = len(CANONICAL_RAD_FINDINGS), out_dim: int = 64):
        super().__init__()
        # 3 features per finding: [state (+1 pres, -1 abs, 0 none), is_reported, region_code]
        in_dim = num_findings * 3
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
        # Pathology feature vector: [score, is_detected, is_eligible] per canonical pathology
        pathology_in_dim = num_pathologies * 3
        self.pathology_net = nn.Sequential(
            nn.Linear(pathology_in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )
        # Optional anatomical context projection
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
        """
        pathology_tensor: (B, num_pathologies * 3)
        anatomy_probs: (B, 33)
        is_pathology_eligible: (B, 1) float tensor (1.0 if verified eligible, 0.0 otherwise)
        """
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
    """Gated Multimodal Unit (GMU) fusing representations using explicit missingness masks."""

    def __init__(self, hidden_dim: int = 64, num_modalities: int = 4):
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

    def forward(self, modality_embeds: list[torch.Tensor], modality_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        modality_embeds: List of 4 tensors, each (B, hidden_dim) [sym, lab, rad, xray]
        modality_mask: (B, 4) float tensor (1.0 = available, 0.0 = omitted/unavailable)
        Returns:
            fused_representation: (B, hidden_dim)
            is_evidence_present: (B, 1) float tensor indicating if at least 1 modality was submitted
        """
        # Compute raw gating logits
        raw_gates = []
        for i, embed in enumerate(modality_embeds):
            gate = self.gate_layers[i](embed)  # (B, 1)
            raw_gates.append(gate)
        all_gates = torch.cat(raw_gates, dim=1)  # (B, 4)

        # Apply modality mask
        masked_gates = torch.sigmoid(all_gates) * modality_mask  # (B, 4)
        sum_gates = masked_gates.sum(dim=1, keepdim=True)
        is_evidence_present = (sum_gates > 1e-5).float()

        # Normalize gates (safely avoid div-by-zero when all modalities absent)
        normalized_gates = masked_gates / torch.clamp(sum_gates, min=1e-6)

        stacked_embeds = torch.stack(modality_embeds, dim=1)  # (B, 4, hidden_dim)
        fused = (stacked_embeds * normalized_gates.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)
        out = self.fusion_proj(fused)
        # If all modalities absent, strictly zero out output
        return out * is_evidence_present, is_evidence_present


# ==============================================================================
# COMPLETE PREDICTOR NEURAL ARCHITECTURE
# ==============================================================================

class MultimodalConditionPredictor(nn.Module):
    """Complete multimodal neural model for condition differential candidate prediction."""

    def __init__(
        self,
        num_classes: int = len(CONDITION_CLASSES),
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        self.symptom_encoder = SymptomModalityEncoder(embed_dim=hidden_dim, out_dim=hidden_dim)
        self.lab_encoder = LaboratoryModalityEncoder(num_tests=len(CANONICAL_LAB_TESTS), out_dim=hidden_dim)
        self.rad_encoder = RadiologyModalityEncoder(num_findings=len(CANONICAL_RAD_FINDINGS), out_dim=hidden_dim)
        self.xray_adapter = XRayModalityAdapter(
            num_anatomy_classes=33,
            num_pathologies=len(CANONICAL_XRAY_PATHOLOGIES),
            out_dim=hidden_dim,
        )

        self.fusion = GatedMultimodalFusion(hidden_dim=hidden_dim, num_modalities=4)

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
        arg6: torch.Tensor,
        arg7: torch.Tensor,
        arg8: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit missing-modality masks.
        Flexible signature supporting:
        1. 8 arguments: (..., xray_pathology_features, xray_anatomy_probs, xray_pathology_eligible, modality_mask)
        2. 7 arguments: (..., xray_pathology_features, xray_pathology_eligible, modality_mask)
        Returns:
            logits: (B, num_classes)
            is_evidence_present: (B, 1)
        """
        batch_size = symptom_tokens.size(0)
        device = symptom_tokens.device

        if arg8 is not None:
            # 8 arguments passed
            xray_anatomy_probs = arg6
            xray_pathology_eligible = arg7
            modality_mask = arg8
        else:
            # 7 arguments passed: arg6 is xray_pathology_eligible, arg7 is modality_mask
            xray_pathology_eligible = arg6
            modality_mask = arg7
            xray_anatomy_probs = torch.zeros(batch_size, 33, device=device)

        h_sym = self.symptom_encoder(symptom_tokens, symptom_features)
        h_lab = self.lab_encoder(lab_features)
        h_rad = self.rad_encoder(rad_features)
        h_xray = self.xray_adapter(xray_pathology_features, xray_anatomy_probs, xray_pathology_eligible)

        modality_embeds = [h_sym, h_lab, h_rad, h_xray]
        h_fused, is_evidence_present = self.fusion(modality_embeds, modality_mask)
        logits = self.classifier(h_fused)
        # When no evidence is present, logits are strictly masked
        return logits * is_evidence_present, is_evidence_present


# ==============================================================================
# FEATURE EXTRACTOR (BRIDGING RAW ASSESSMENT INPUTS TO MODEL TENSORS)
# ==============================================================================

class MultimodalFeatureExtractor:
    """Transforms raw patient observations, lab reports, radiology findings, and X-ray data into tensor representations."""

    @staticmethod
    def _hash_token(text: str, vocab_size: int = 256) -> int:
        """Deterministic string hashing to a vocabulary index in 1..vocab_size-1 (0 is padding)."""
        clean = text.strip().lower()
        if not clean:
            return 0
        h = 0
        for char in clean:
            h = (h * 31 + ord(char)) % (vocab_size - 1)
        return h + 1

    @classmethod
    def extract_tensors_from_record(
        cls,
        record: MultimodalPatientRecord,
        max_symptoms: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Convert a MultimodalPatientRecord into model input tensors."""
        # 1. Symptoms
        tokens = [0] * max_symptoms
        sym_feat = [[0.0, 0.0, 0.0] for _ in range(max_symptoms)]
        for i, sym in enumerate(record.symptoms[:max_symptoms]):
            tokens[i] = cls._hash_token(sym.canonical or sym.symptom)
            # state: PRESENT = +1.0, ABSENT = -1.0, UNKNOWN = 0.0
            st_val = 1.0 if sym.state == "PRESENT" else (-1.0 if sym.state == "ABSENT" else 0.0)
            dur = float(sym.duration_days or 0.0) / 30.0  # normalized to month
            dur_clamped = max(0.0, min(dur, 2.0))
            sev_map = {"MILD": 1.0, "MODERATE": 2.0, "SEVERE": 3.0}
            sev_val = sev_map.get(str(sym.severity).upper(), 0.0) / 3.0
            sym_feat[i] = [st_val, dur_clamped, sev_val]

        symptom_tokens_t = torch.tensor([tokens], dtype=torch.long)
        symptom_features_t = torch.tensor([sym_feat], dtype=torch.float32)

        # 2. Laboratory
        # 5 features per test: [val_norm, ref_low_norm, ref_high_norm, flag_code, is_present]
        lab_features = [0.0] * (len(CANONICAL_LAB_TESTS) * 5)
        lab_lookup = {l.canonical_test_name.lower(): l for l in record.laboratory}

        for idx, test_name in enumerate(CANONICAL_LAB_TESTS):
            base = idx * 5
            l_item = lab_lookup.get(test_name.lower())
            if l_item:
                val = float(l_item.value)
                # Scale value reasonably for neural inputs
                val_norm = val / 100.0 if val > 10 else val
                ref_l = (float(l_item.reference_low) / 100.0) if l_item.reference_low is not None else 0.0
                ref_h = (float(l_item.reference_high) / 100.0) if l_item.reference_high is not None else 0.0
                flag_str = str(l_item.flag).upper()
                flag_code = 1.0 if flag_str == "HIGH" else (-1.0 if flag_str == "LOW" else (0.5 if flag_str == "ABNORMAL" else 0.0))
                lab_features[base] = val_norm
                lab_features[base + 1] = ref_l
                lab_features[base + 2] = ref_h
                lab_features[base + 3] = flag_code
                lab_features[base + 4] = 1.0  # is_present

        lab_features_t = torch.tensor([lab_features], dtype=torch.float32)

        # 3. Radiology
        # 3 features per finding: [state_code, is_reported, certainty_code]
        rad_features = [0.0] * (len(CANONICAL_RAD_FINDINGS) * 3)
        rad_lookup = {r.canonical_finding.lower(): r for r in record.radiology}

        for idx, finding_name in enumerate(CANONICAL_RAD_FINDINGS):
            base = idx * 3
            r_item = rad_lookup.get(finding_name.lower())
            if r_item:
                st = str(r_item.state).upper()
                if st in {"PRESENT", "POSITIVE", "CONFIRMED"}:
                    st_code = 1.0
                    is_rep = 1.0
                    cert_code = 1.0
                elif st in {"ABSENT", "NEGATIVE", "NONE"}:
                    st_code = -1.0
                    is_rep = 1.0
                    cert_code = 1.0
                elif st in {"POSSIBLE", "UNCERTAIN", "EQUIVOCAL", "INDETERMINATE"}:
                    st_code = 0.0
                    is_rep = 1.0
                    cert_code = 0.5
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

        rad_features_t = torch.tensor([rad_features], dtype=torch.float32)

        # 4. X-ray
        # 3 features per pathology: [score, is_detected, is_eligible]
        xray_pathology = [0.0] * (len(CANONICAL_XRAY_PATHOLOGIES) * 3)
        anatomy_probs = [0.0] * 33
        is_xray_eligible = 0.0

        if record.xray and record.xray.is_pathology_eligible and not record.xray.has_region_mismatch:
            is_xray_eligible = 1.0
            p_scores = record.xray.pathology_scores or {}
            for idx, p_name in enumerate(CANONICAL_XRAY_PATHOLOGIES):
                base = idx * 3
                score = float(p_scores.get(p_name, 0.0))
                is_det = 1.0 if score >= 0.5 else 0.0
                xray_pathology[base] = score
                xray_pathology[base + 1] = is_det
                xray_pathology[base + 2] = 1.0
            anat_p = getattr(record.xray, "anatomy_probs", None)
            if anat_p and len(anat_p) == 33:
                anatomy_probs = [float(p) for p in anat_p]

        xray_pathology_t = torch.tensor([xray_pathology], dtype=torch.float32)
        xray_anatomy_t = torch.tensor([anatomy_probs], dtype=torch.float32)
        xray_eligibility_t = torch.tensor([[is_xray_eligible]], dtype=torch.float32)

        # Modality mask: [m_sym, m_lab, m_rad, m_xray]
        m_sym = 1.0 if len(record.symptoms) > 0 else 0.0
        m_lab = 1.0 if len(record.laboratory) > 0 else 0.0
        m_rad = 1.0 if len(record.radiology) > 0 else 0.0
        m_xray = 1.0 if is_xray_eligible > 0.0 else 0.0
        modality_mask_t = torch.tensor([[m_sym, m_lab, m_rad, m_xray]], dtype=torch.float32)

        return {
            "symptom_tokens": symptom_tokens_t,
            "symptom_features": symptom_features_t,
            "lab_features": lab_features_t,
            "rad_features": rad_features_t,
            "xray_pathology_features": xray_pathology_t,
            "xray_anatomy_probs": xray_anatomy_t,
            "xray_pathology_eligible": xray_eligibility_t,
            "modality_mask": modality_mask_t,
        }


# ==============================================================================
# HIGH-LEVEL INFERENCE & VALIDATION SERVICE
# ==============================================================================

class MultimodalInferenceService:
    """Production service wrapper coordinating multimodal prediction with clinical safety."""

    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path
        self.model: Optional[MultimodalConditionPredictor] = None
        self.is_trained: bool = False
        self.model_version = "multimodal_gmu_v2.0-unvalidated"

        if checkpoint_path and checkpoint_path.exists():
            try:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model = MultimodalConditionPredictor()
                model.load_state_dict(ckpt["model_state"])
                model.eval()
                self.model = model
                self.is_trained = True
                self.model_version = ckpt.get("version", self.model_version)
            except Exception:
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

        # Process laboratory findings from medical reports
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
                    t_name = str(f.get("test_name") or f.get("canonical") or "").strip()
                    val = f.get("value")
                    if t_name and val is not None:
                        try:
                            v_float = float(val)
                            lab_params.append(StructuredLabParameter(
                                test_name=t_name,
                                canonical_test_name=t_name,
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

        # Process radiology report findings and clinical examination observations
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
                        ))

        # Process X-ray specialist image findings via Adapter
        xray_output: Optional[XRaySpecialistOutput] = None
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

            p_scores: dict[str, float] = {}
            # Extract actionable specialist findings
            for find in (img.get("findings") or []):
                if isinstance(find, dict):
                    lbl = str(find.get("label") or find.get("name") or "").strip()
                    score = find.get("model_score", find.get("score", 0.0))
                    if lbl and isinstance(score, (int, float)):
                        p_scores[lbl] = float(score)

            xray_output = XRaySpecialistOutput(
                file_name=fname,
                detected_anatomy=det_anat,
                anatomy_verified=is_verified,
                specialist_model_executed=bool(p_scores),
                has_region_mismatch=has_mismatch,
                is_pathology_eligible=is_eligible,
                pathology_scores=p_scores,
                confidence=img.get("confidence"),
                status=str(img.get("status") or "PENDING"),
            )
            break  # Process primary active radiographic study

        record = MultimodalPatientRecord(
            subject_id=subject_id,
            symptoms=symptom_obs,
            laboratory=lab_params,
            radiology=rad_findings,
            clinical_examination=clinical_findings,
            xray=xray_output,
            provenance={
                "has_symptoms": len(symptom_obs) > 0,
                "has_laboratory": len(lab_params) > 0,
                "has_radiology": len(rad_findings) > 0,
                "has_clinical_examination": len(clinical_findings) > 0,
                "has_xray": xray_output is not None and xray_output.is_pathology_eligible,
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
        has_xray = record.xray is not None and record.xray.is_pathology_eligible

        if not (has_sym or has_lab or has_rad or has_xray):
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
                    "xray_pathology": has_xray,
                },
                "conditions": [],
            }

        # If trained model is present, execute torch forward pass
        try:
            with torch.no_grad():
                logits, is_ev = self.model(
                    tensor_bundle["symptom_tokens"],
                    tensor_bundle["symptom_features"],
                    tensor_bundle["lab_features"],
                    tensor_bundle["rad_features"],
                    tensor_bundle["xray_pathology_features"],
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
