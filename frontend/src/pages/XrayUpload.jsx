import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import FilePreviewModal from '../components/FilePreviewModal';
import CardSearchableAnatomySelector from '../components/CardSearchableAnatomySelector';
import { uploadXray, deleteXray, confirmXrayAnatomy, getXrayResults } from '../services/xrayService';
import { uploadReport, deleteReport } from '../services/reportService';
import { useAssessment } from '../hooks/useAssessment';
import './NewAssessment.css';

/* ============================================================
   SVG ICONS
   ============================================================ */

function SparkleIcon({ size = 14, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3L12 3z" />
    </svg>
  );
}

function CheckIcon({ size = 12, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function ArrowRightIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <line x1="5" y1="12" x2="19" y2="12" />
      <polyline points="12 5 19 12 12 19" />
    </svg>
  );
}

function ShieldLockIcon({ size = 14, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <circle cx="12" cy="11" r="1.5" />
      <path d="M12 12.5v3" />
    </svg>
  );
}


/* ============================================================
   CANVAS NEURAL NETWORK BACKGROUND (MINT/CYAN TINT)
   ============================================================ */
function NeuralNetworkCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId;
    let width = (canvas.width = canvas.parentElement.offsetWidth || window.innerWidth);
    let height = (canvas.height = canvas.parentElement.offsetHeight || 800);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.offsetWidth;
      height = canvas.height = canvas.parentElement.offsetHeight;
    };
    window.addEventListener('resize', handleResize);

    const nodeCount = 30;
    const nodes = [];
    for (let i = 0; i < nodeCount; i++) {
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.32,
        vy: (Math.random() - 0.5) * 0.32,
        radius: Math.random() * 2 + 1.2,
        alpha: Math.random() * 0.45 + 0.25,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      for (let i = 0; i < nodeCount; i++) {
        const n = nodes[i];
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(16, 185, 129, ${n.alpha * 0.8})`;
        ctx.shadowColor = '#10b981';
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      const maxDist = 135;
      for (let i = 0; i < nodeCount; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxDist) {
            const lineAlpha = (1 - dist / maxDist) * 0.2;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(16, 185, 129, ${lineAlpha})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="na-neural-canvas" />;
}

/* ============================================================
   3D THORACIC RIBCAGE & LUNG SILHOUETTE (LEFT SIDE)
   ============================================================ */
function ThoracicRibcageVisual() {
  return (
    <div className="na-left-backdrop" aria-hidden="true">
      <svg width="260" height="640" viewBox="0 0 260 640" fill="none">
        <defs>
          <linearGradient id="ribcageGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#34d399" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.45" />
          </linearGradient>
          <filter id="ribcageGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#10b981" floodOpacity="0.35" />
          </filter>
        </defs>

        {/* Central Thoracic Spine */}
        <line x1="130" y1="140" x2="130" y2="460" stroke="url(#ribcageGrad)" strokeWidth="3" strokeDasharray="6 4" filter="url(#ribcageGlow)" />

        {/* Bilateral Ribs 1 to 5 */}
        <path d="M 130 180 C 80 170, 50 190, 60 220" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />
        <path d="M 130 180 C 180 170, 210 190, 200 220" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />

        <path d="M 130 225 C 70 215, 35 240, 50 280" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />
        <path d="M 130 225 C 190 215, 225 240, 210 280" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />

        <path d="M 130 275 C 65 270, 30 300, 45 345" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />
        <path d="M 130 275 C 195 270, 230 300, 215 345" stroke="url(#ribcageGrad)" strokeWidth="2.5" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />

        <path d="M 130 325 C 70 325, 40 360, 55 400" stroke="url(#ribcageGrad)" strokeWidth="2.2" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />
        <path d="M 130 325 C 190 325, 220 360, 205 400" stroke="url(#ribcageGrad)" strokeWidth="2.2" strokeLinecap="round" fill="none" filter="url(#ribcageGlow)" />

        {/* Clavicle bones */}
        <path d="M 80 150 Q 130 165 180 150" stroke="url(#ribcageGrad)" strokeWidth="3" strokeLinecap="round" fill="none" />
      </svg>
    </div>
  );
}

/* ============================================================
   RADIOLOGY FILM SCREEN VISUAL (RIGHT SIDE)
   ============================================================ */
function RadiologyFilmScreenVisual() {
  return (
    <div className="na-right-backdrop" aria-hidden="true">
      <svg width="300" height="640" viewBox="0 0 300 640" fill="none">
        <defs>
          <linearGradient id="filmGrad" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#06b6d4" stopOpacity="0.65" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.4" />
          </linearGradient>
          <filter id="filmGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#10b981" floodOpacity="0.32" />
          </filter>
        </defs>

        {/* Illuminated Radiology Screen Box */}
        <rect x="110" y="170" width="160" height="280" rx="14" stroke="url(#filmGrad)" strokeWidth="2.5" fill="none" filter="url(#filmGlow)" />

        {/* Scan Reticle crosshairs */}
        <line x1="190" y1="180" x2="190" y2="440" stroke="rgba(16, 185, 129, 0.35)" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="120" y1="310" x2="260" y2="310" stroke="rgba(16, 185, 129, 0.35)" strokeWidth="1" strokeDasharray="4 4" />

        {/* Target Reticle in Center */}
        <circle cx="190" cy="310" r="30" stroke="rgba(34, 211, 238, 0.65)" strokeWidth="1.8" fill="none" />
        <circle cx="190" cy="310" r="4" fill="#ffffff" stroke="#10b981" strokeWidth="2" filter="url(#filmGlow)" />

        {/* Scan horizontal laser beam */}
        <line x1="112" y1="260" x2="268" y2="260" stroke="#34d399" strokeWidth="2" strokeLinecap="round" filter="url(#filmGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   FLOATING RADIOLOGY MEDALLIONS
   ============================================================ */
function FloatingXrayObjects() {
  return (
    <div className="na-floating-objects-layer" aria-hidden="true">
      <div className="na-floating-medallion medallion-pos-1" title="Chest Radiograph">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="4" />
          <path d="M7 9a3 3 0 0 1 3 3v3a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1" />
          <path d="M17 9a3 3 0 0 0-3 3v3a1 1 0 0 0 2 0v-3a1 1 0 0 1 1-1" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-2" title="Radiological Scan View">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <line x1="22" y1="12" x2="18" y2="12" />
          <line x1="6" y1="12" x2="2" y2="12" />
          <line x1="12" y1="6" x2="12" y2="2" />
          <line x1="12" y1="22" x2="12" y2="18" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-3" title="Musculoskeletal Density">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 2v20" />
          <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-4" title="Medical Cross">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </div>
    </div>
  );
}

/* Goal 1: All 33 required anatomical classes available for user verification/confirmation.
   When an unsupported or broad region is confirmed, the system safely records the verified
   anatomy and discloses that automated abnormality interpretation is unsupported. */
const XRAY_REGION_OPTIONS = [
  { value: 'ABDOMEN', label: 'Abdomen' },
  { value: 'ANKLE', label: 'Ankle' },
  { value: 'ARM', label: 'Arm (Upper limb)' },
  { value: 'CERVICAL_SPINE', label: 'Cervical spine' },
  { value: 'CHEST', label: 'Chest' },
  { value: 'DORSAL_SPINE', label: 'Thoracic / dorsal spine' },
  { value: 'ELBOW', label: 'Elbow' },
  { value: 'EXTREMITY', label: 'Extremity (Limb)' },
  { value: 'FEMUR', label: 'Femur' },
  { value: 'FINGER', label: 'Finger' },
  { value: 'FOOT', label: 'Foot' },
  { value: 'FOREARM', label: 'Forearm' },
  { value: 'HAND', label: 'Hand' },
  { value: 'HEAD', label: 'Head' },
  { value: 'HEEL', label: 'Heel' },
  { value: 'HIP', label: 'Hip' },
  { value: 'KNEE', label: 'Knee' },
  { value: 'KUB', label: 'KUB / abdomen' },
  { value: 'LEG', label: 'Leg (Lower limb)' },
  { value: 'LOWER_EXTREMITY', label: 'Lower extremity' },
  { value: 'LUMBAR_SPINE', label: 'Lumbar spine' },
  { value: 'NASOPHARYNX', label: 'Nasopharynx' },
  { value: 'NECK', label: 'Neck' },
  { value: 'PELVIS', label: 'Pelvis' },
  { value: 'PNS', label: 'Paranasal sinuses' },
  { value: 'SHOULDER', label: 'Shoulder' },
  { value: 'SI_JOINT', label: 'Sacroiliac joint' },
  { value: 'SKULL', label: 'Skull' },
  { value: 'SPINE', label: 'Spine (Vertebral column)' },
  { value: 'TEMPORAL_BONE', label: 'Temporal bone' },
  { value: 'TIBIA', label: 'Tibia' },
  { value: 'UPPER_EXTREMITY', label: 'Upper extremity' },
  { value: 'WRIST', label: 'Wrist' },
];

const BROAD_REGION_SET = new Set([
  'EXTREMITY', 'UPPER_EXTREMITY', 'LOWER_EXTREMITY', 'ARM', 'LEG', 'HEAD', 'SPINE'
]);

function getSpecificRegionOption(value, allowBroad = true) {
  const token = String(value || '')
    .trim()
    .toUpperCase()
    .replace(/[ -]+/g, '_');
  const aliases = {
    'SACROILIAC_JOINT': 'SI_JOINT',
    'SINUSES': 'PNS',
    'PARANASAL_SINUSES': 'PNS',
    'CALCANEUS': 'HEEL',
    'THORACIC_SPINE': 'DORSAL_SPINE',
    'LOWER_LEG': 'TIBIA',
    'HUMERUS': 'ARM',
  };
  const normalized = aliases[token] || token;
  if (!allowBroad && BROAD_REGION_SET.has(normalized)) {
    return '';
  }
  return XRAY_REGION_OPTIONS.some((option) => option.value === normalized) ? normalized : '';
}

/* ============================================================
   RESPONSE / STATUS HELPERS
   ============================================================ */
function getImageStatusPresentation(item) {
  switch (item.status) {
    case 'READY':
      return { className: 'status-ready', label: 'Waiting to analyze' };
    case 'UPLOADING':
      return { className: 'status-uploading', label: 'Analyzing image anatomy...' };
    case 'DELETING':
      return { className: 'status-uploading', label: 'Removing saved image...' };
    case 'COMPLETED':
      return { className: 'status-completed', label: 'Region confirmed · Abnormality analysis complete' };
    case 'ANATOMY_VERIFIED':
      return {
        className: 'status-completed',
        label: item.analysisStatus === 'completed'
          ? 'Region confirmed · Abnormality analysis complete'
          : 'Region confirmed · Abnormality model unsupported',
      };
    case 'BROAD_REGION_ACCEPTED':
      return {
        className: 'status-warning',
        label: 'Broad region identified · Specific site selection required',
      };
    case 'REGION_CONFIRMATION_REQUIRED': {
      const gatePassed = Boolean(
        item.classifierGatePassed ||
        item.anatomyClassifierAccepted ||
        (typeof item.bodyPartConfidence === 'number' && item.bodyPartConfidence >= 0.60)
      );
      return {
        className: 'status-warning',
        label: gatePassed
          ? 'AI candidate predicted (gate passed) · Awaiting user confirmation'
          : 'Low-confidence prediction · Awaiting user selection',
      };
    }
    case 'ANALYSIS_UNAVAILABLE':
      return {
        className: item.anatomyUserConfirmed ? 'status-warning' : 'status-failed',
        label: item.anatomyUserConfirmed
          ? 'Region confirmed · No compatible abnormality model'
          : (item.error || 'Image uploaded · Automated interpretation unavailable'),
      };
    case 'FAILED':
    default:
      return { className: 'status-failed', label: item.error || 'Image upload or analysis failed' };
  }
}

function getReportStatusPresentation(item) {
  switch (item.status) {
    case 'READY':
      return { className: 'status-ready', label: 'Waiting for extraction' };
    case 'UPLOADING':
      return { className: 'status-uploading', label: 'Uploading and extracting report...' };
    case 'DELETING':
      return { className: 'status-uploading', label: 'Removing saved report...' };
    case 'COMPLETED':
      return { className: 'status-completed', label: 'Report text extracted' };
    case 'EXTRACTION_INCOMPLETE':
      return {
        className: 'status-failed',
        label: item.error || 'Report uploaded; structured extraction incomplete',
      };
    case 'UNREADABLE':
      return {
        className: 'status-failed',
        label: item.error || 'Report uploaded; its text could not be read',
      };
    case 'FAILED':
    default:
      return { className: 'status-failed', label: item.error || 'Report upload failed' };
  }
}

function normalizeStatus(value) {
  return typeof value === 'string' ? value.trim().toLowerCase() : '';
}

function getResponseFindings(response) {
  const candidates = [
    response?.findings,
    response?.extracted_findings,
    response?.structured_findings,
    response?.report_findings,
    response?.ocr_metadata?.findings,
    response?.ocr_metadata?.structured_findings,
  ];
  return candidates.find(Array.isArray) || [];
}

function getResponseText(response) {
  const candidates = [
    response?.extracted_text,
    response?.ocr_text,
    response?.report_text,
    response?.text,
    response?.ocr_metadata?.extracted_text,
    response?.ocr_metadata?.ocr_text,
  ];
  return candidates.find((value) => typeof value === 'string' && value.trim())?.trim() || '';
}

function prettyFinding(finding, index) {
  if (typeof finding === 'string') return finding;
  if (!finding || typeof finding !== 'object') return `Finding ${index + 1}: ${String(finding)}`;

  const label = finding.label || finding.name || finding.finding || finding.finding_name || finding.class || `Finding ${index + 1}`;
  const score = finding.score ?? finding.probability ?? finding.confidence;
  if (typeof score === 'number' && Number.isFinite(score)) {
    const formatted = score >= 0 && score <= 1 ? `${(score * 100).toFixed(1)}%` : score.toFixed(3);
    return `${label} (${formatted})`;
  }
  const interpretation = finding.interpretation || finding.status || finding.value;
  return interpretation !== undefined && interpretation !== null ? `${label}: ${interpretation}` : String(label);
}

function formatFileSize(bytes) {
  if (!bytes) return '';
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

/* ============================================================
   MAIN X-RAY & RADIOLOGY UPLOAD COMPONENT (MULTI-FILE)

   Goal 1 workflow:
   1. Select image -> upload and analyze automatically.
   2. Display body-part classification and available image findings.
   3. Make report upload available after image processing (or immediately
      for report-only mode); extract report text when selected.
   4. Continue only after available inputs have finished processing.
   ============================================================ */
export default function XrayUpload() {
  const navigate = useNavigate();
  const { assessmentId } = useAssessment();
  const [xrayImages, setXrayImages] = useState([]);
  const [xrayReports, setXrayReports] = useState([]);
  const [info, setInfo] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [deletingImageIds, setDeletingImageIds] = useState(() => new Set());
  const [deletingReportIds, setDeletingReportIds] = useState(() => new Set());
  const [dragActiveImages, setDragActiveImages] = useState(false);
  const [dragActiveReports, setDragActiveReports] = useState(false);
  const [previewState, setPreviewState] = useState(null);

  const imageInputRef = useRef(null);
  const reportInputRef = useRef(null);
  const previewUrlRef = useRef(null);
  const processingImageIdsRef = useRef(new Set());
  const processingReportIdsRef = useRef(new Set());

  const ALLOWED_IMAGE_EXTS = ['.png', '.jpg', '.jpeg'];
  const ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

  useEffect(() => () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
  }, []);

  // Reload persisted X-ray results for this assessment on mount or navigation
  useEffect(() => {
    let isMounted = true;
    if (!assessmentId) return;

    async function loadSavedXrayResults() {
      try {
        const response = await getXrayResults(assessmentId);
        const results = Array.isArray(response?.results) ? response.results : [];
        if (!isMounted || results.length === 0) return;

        setXrayImages((prev) => {
          if (prev.length > 0) return prev;
          return results.map((res) => {
            const hasUsableModelOutput = Boolean(
              res.prediction || (Array.isArray(res.findings) && res.findings.length > 0)
            );
            const isCompleted = res.status === 'completed' && hasUsableModelOutput;
            const isBroad = Boolean(res.is_broad_region);
            const broadAccepted = isBroad && Boolean(res.broad_region_classifier_accepted);
            const anatomyVerified = Boolean(
              res.anatomy_user_confirmed || res.anatomy_verification_status === 'confirmed'
            );

            const status = isCompleted
              ? 'COMPLETED'
              : anatomyVerified
                ? 'ANATOMY_VERIFIED'
                : broadAccepted
                  ? 'BROAD_REGION_ACCEPTED'
                  : res.anatomy_confirmation_required
                    ? 'REGION_CONFIRMATION_REQUIRED'
                    : 'ANALYSIS_UNAVAILABLE';

            const rawRegion = res.detected_region || res.region || res.candidate_region || null;

            return {
              id: `saved-${res.xray_result_id}`,
              name: res.file_name || 'Saved X-ray Image',
              size: 0,
              type: 'image/jpeg',
              status,
              uploadStatus: 'uploaded',
              analysisStatus: res.status || 'unknown',
              analysisMessage: res.message || '',
              detectedRegion: res.detected_region ? String(res.detected_region).toUpperCase() : null,
              candidateRegion: res.candidate_region ? String(res.candidate_region).toUpperCase() : null,
              verificationRegion: getSpecificRegionOption(rawRegion, !isBroad),
              topCandidates: Array.isArray(res.anatomy_metadata?.top_candidates) ? res.anatomy_metadata.top_candidates : [],
              bodyPartConfidence: typeof res.body_part_confidence === 'number' ? res.body_part_confidence : null,
              anatomyVerificationStatus: res.anatomy_verification_status || (res.anatomy_user_confirmed ? 'confirmed' : 'confirmation_required'),
              anatomyUserConfirmed: Boolean(res.anatomy_user_confirmed),
              anatomyClassifierAccepted: Boolean(res.anatomy_classifier_accepted || res.classifier_gate_passed),
              classifierGatePassed: Boolean(res.classifier_gate_passed || res.anatomy_classifier_accepted),
              regionalModelConfigured: Boolean(res.regional_model_configured),
              regionalModelAvailable: Boolean(res.regional_model_available),
              regionalModelName: res.regional_model_name || null,
              regionalModelStatus: res.regional_model_status || null,
              primaryPrediction: res.primary_prediction || res.image_type_detection?.primary_classifier_result || res.anatomy_metadata?.classifier_output?.primary_classifier_result || null,
              specialistPrediction: res.specialist_prediction || res.image_type_detection?.upper_extremity_refinement || res.anatomy_metadata?.classifier_output?.upper_extremity_refinement || null,
              twoModelComparison: res.two_model_comparison || res.anatomy_metadata?.two_model_comparison || null,
              finalCandidate: res.final_candidate || res.anatomy_metadata?.final_candidate || res.detected_region || res.candidate_region || null,
              selectionSource: res.selection_source || res.anatomy_metadata?.selection_source || 'PRIMARY MODEL',
              confidenceSource: res.confidence_source || res.anatomy_metadata?.confidence_source || 'primary_model',
              verifyingAnatomy: false,
              prediction: res.prediction || null,
              findings: Array.isArray(res.findings) ? res.findings : [],
              modelVersion: res.model_version || null,
              resultId: res.xray_result_id,
              error: null,
            };
          });
        });
      } catch {
        // Silently ignore if assessment has no X-rays yet
      }
    }

    loadSavedXrayResults();
    return () => {
      isMounted = false;
    };
  }, [assessmentId]);

  const isProcessingFiles =
    deletingImageIds.size > 0 ||
    deletingReportIds.size > 0 ||
    xrayImages.some((item) => item.status === 'READY' || item.status === 'UPLOADING' || item.status === 'DELETING' || item.verifyingAnatomy) ||
    xrayReports.some((item) => item.status === 'READY' || item.status === 'UPLOADING' || item.status === 'DELETING');

  const hasUploadedImage = xrayImages.some((item) => item.uploadStatus === 'uploaded');
  const hasExtractedReport = xrayReports.some((item) => item.status === 'COMPLETED' && item.uploadStatus === 'uploaded');
  const hasAnySelectedInput = xrayImages.length > 0 || xrayReports.length > 0;
  const hasImageStageSettled = xrayImages.length === 0 || xrayImages.every(
    (item) => item.status !== 'READY' && item.status !== 'UPLOADING'
  );
  const reportStageVisible = xrayImages.length === 0 || hasImageStageSettled;

  const updateImage = (id, patch) => {
    setXrayImages((previous) => previous.map((item) => item.id === id ? { ...item, ...patch } : item));
  };

  const updateReport = (id, patch) => {
    setXrayReports((previous) => previous.map((item) => item.id === id ? { ...item, ...patch } : item));
  };

  const handleImageDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActiveImages(true);
    else if (e.type === 'dragleave') setDragActiveImages(false);
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveImages(false);
    if (e.dataTransfer.files) addImages(e.dataTransfer.files);
  };

  const handleReportDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') setDragActiveReports(true);
    else if (e.type === 'dragleave') setDragActiveReports(false);
  };

  const handleReportDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveReports(false);
    if (e.dataTransfer.files) addReports(e.dataTransfer.files);
  };

  const analyzeImage = async (item) => {
    if (!assessmentId) {
      updateImage(item.id, {
        status: 'FAILED',
        uploadStatus: 'failed',
        error: 'No assessment is active. Start a new assessment before uploading an X-ray.',
      });
      return;
    }
    if (processingImageIdsRef.current.has(item.id)) return;
    processingImageIdsRef.current.add(item.id);
    updateImage(item.id, {
      status: 'UPLOADING',
      uploadStatus: 'uploading',
      error: null,
      analysisMessage: 'Uploading image and running automatic image-type recognition...',
    });

    try {
      const response = await uploadXray(assessmentId, item.file);
      const responseStatus = normalizeStatus(response?.status);
      const detection = response?.image_type_detection || response?.anatomy_detection || {};
      const anatomyVerification = response?.image_anatomy_verification || {};
      const detectionStatus = normalizeStatus(
        response?.anatomy_verification_status || detection?.status || anatomyVerification?.status
      );

      // A specific region may pass the classifier gate. Broad labels can also pass
      // that gate at the broad level, but still require a specific-site selection
      // before regional interpretation. Neither status is clinical confirmation.
      const rawRegion =
        response?.detected_region ||
        response?.canonical_region ||
        anatomyVerification?.detected_region ||
        response?.candidate_region ||
        detection?.body_part_label ||
        detection?.candidate_body_part ||
        null;
      const regionKey = String(rawRegion || response?.region || '')
        .trim()
        .toLowerCase()
        .replace(/[ _-]+/g, ' ');
      const broadRegions = new Set([
        'extremity', 'upper extremity', 'lower extremity', 'head region',
        'arm', 'leg', 'head', 'unknown',
      ]);
      const reasonCode = String(response?.reason_code || anatomyVerification?.reason_code || '').toLowerCase();
      const confidenceRaw = response?.body_part_confidence ?? detection?.body_part_confidence ?? anatomyVerification?.body_part_confidence;
      const bodyPartConfidence = typeof confidenceRaw === 'number'
        ? confidenceRaw
        : Number.isFinite(Number(confidenceRaw)) && confidenceRaw !== null && confidenceRaw !== ''
          ? Number(confidenceRaw)
          : null;
      const thresholdRaw = response?.confidence_threshold ?? detection?.confidence_threshold ?? anatomyVerification?.confidence_threshold ?? 0.60;
      const confidenceThreshold = Number.isFinite(Number(thresholdRaw)) ? Number(thresholdRaw) : 0.60;

      // Preserve the classifier's ranked label outputs so broad predictions such
      // as UPPER_EXTREMITY can show which specific labels the model also scored.
      // These remain candidates only; never promote one to a detected region here.
      const rawTopCandidates =
        detection?.top_predictions ||
        detection?.top_candidates ||
        response?.anatomy_metadata?.top_candidates ||
        response?.anatomy_metadata?.classifier_output?.top_predictions ||
        response?.image_anatomy_verification?.top_candidates ||
        [];
      const normalizeCandidate = (entry, source = 'original_upload_copy') => {
        if (typeof entry === 'string') {
          const label = entry.trim().toUpperCase();
          return label ? { label, score: null, source } : null;
        }
        if (!entry || typeof entry !== 'object') return null;
        const label = String(
          entry.label ?? entry.body_part ?? entry.body_part_label ?? entry.region ?? entry.class_name ?? ''
        ).trim().toUpperCase();
        const rawScore = entry.probability ?? entry.confidence ?? entry.score ?? entry.percent;
        let score = rawScore === undefined || rawScore === null || rawScore === ''
          ? null
          : Number(rawScore);
        if (score !== null && Number.isFinite(score) && score > 1 && score <= 100) {
          score /= 100;
        }
        if (!label) return null;
        return {
          label,
          score: score !== null && Number.isFinite(score) && score >= 0 && score <= 1
            ? score
            : null,
          source,
        };
      };
      const primaryCandidates = Array.isArray(rawTopCandidates)
        ? rawTopCandidates.map((entry) => normalizeCandidate(entry)).filter(Boolean)
        : [];
      const cleanupCandidate =
        detection?.colored_cleanup_candidate ||
        response?.anatomy_metadata?.classifier_output?.colored_cleanup_candidate ||
        anatomyVerification?.classifier_output?.colored_cleanup_candidate ||
        null;
      const cleanupCandidates = Array.isArray(cleanupCandidate?.top_predictions)
        ? cleanupCandidate.top_predictions
            .map((entry) => normalizeCandidate(entry, 'annotation_cleaned_copy'))
            .filter(Boolean)
        : [];
      const upperExtremityRefinement =
        detection?.upper_extremity_refinement ||
        response?.anatomy_metadata?.classifier_output?.upper_extremity_refinement ||
        anatomyVerification?.classifier_output?.upper_extremity_refinement ||
        null;
      const upperExtremityCandidates = Array.isArray(upperExtremityRefinement?.top_predictions)
        ? upperExtremityRefinement.top_predictions
            .map((entry) => normalizeCandidate(entry, 'upper_extremity_specialist'))
            .filter(Boolean)
        : [];
      const candidateMap = new Map();
      [...primaryCandidates, ...cleanupCandidates, ...upperExtremityCandidates].forEach((candidate) => {
        const existing = candidateMap.get(candidate.label);
        if (!existing) {
          candidateMap.set(candidate.label, candidate);
        } else if (
          candidate.source === 'annotation_cleaned_copy' &&
          existing.source !== 'annotation_cleaned_copy'
        ) {
          candidateMap.set(`${candidate.label}__cleaned`, candidate);
        } else if (
          candidate.source === 'upper_extremity_specialist' &&
          existing.source !== 'upper_extremity_specialist'
        ) {
          candidateMap.set(`${candidate.label}__upper_specialist`, candidate);
        }
      });
      const topCandidates = [...candidateMap.values()].slice(0, 10);
      const statusAllowsVerification = ['completed', 'verified', 'confirmed'].includes(detectionStatus);
      const classifierGateAccepted = response?.anatomy_classifier_accepted === true ||
        anatomyVerification?.anatomy_classifier_accepted === true;
      const isBroadRegion = broadRegions.has(regionKey) || response?.is_broad_region === true || anatomyVerification?.is_broad_region === true;
      const broadRegionAccepted = isBroadRegion && (
        response?.broad_region_classifier_accepted === true ||
        anatomyVerification?.broad_region_classifier_accepted === true ||
        (
          classifierGateAccepted &&
          (reasonCode === 'broad_anatomy_label' || reasonCode === 'anatomy_specificity_unverified') &&
          bodyPartConfidence !== null && bodyPartConfidence >= confidenceThreshold
        )
      );
      const isUserConfirmed = Boolean(
        response?.anatomy_user_confirmed ||
        anatomyVerification?.anatomy_user_confirmed ||
        response?.region_source === 'user_confirmed'
      );
      const anatomyVerified = isUserConfirmed &&
        !broadRegions.has(regionKey) &&
        reasonCode !== 'broad_anatomy_label';
      const detectedRegion = isUserConfirmed ? (rawRegion || response?.detected_region) : null;
      const candidateRegion = isUserConfirmed ? null : (
        detection?.candidate_body_part ||
        response?.candidate_region ||
        rawRegion ||
        null
      );
      const findings = getResponseFindings(response);
      const prediction = typeof response?.prediction === 'string' ? response.prediction.trim() : '';
      const hasUsableModelOutput = Boolean(prediction || findings.length > 0);
      const isCompleted = responseStatus === 'completed' && hasUsableModelOutput;
      const isSaved = Boolean(response?.xray_result_id);
      const requiresRegionConfirmation = Boolean(
        isSaved &&
        !isUserConfirmed &&
        !hasUsableModelOutput &&
        (
          response?.anatomy_confirmation_required === true ||
          response?.anatomy_confirmation_required == null ||
          response?.specific_region_confirmation_required === true
        )
      );
      const message = response?.message || response?.diagnostic_message || detection?.message || anatomyVerification?.message || '';
      const broadRegionMessage = broadRegionAccepted
        ? `The classifier accepted the broad region ${String(candidateRegion || rawRegion || 'UNKNOWN').replace(/_/g, ' ').toUpperCase()} at ${(bodyPartConfidence * 100).toFixed(1)}%. This does not identify a specific site such as Wrist, Hand, or Elbow. No region-specific abnormality model was run.`
        : message;

      updateImage(item.id, {
        status: isCompleted
          ? 'COMPLETED'
          : broadRegionAccepted
            ? 'BROAD_REGION_ACCEPTED'
            : requiresRegionConfirmation
              ? 'REGION_CONFIRMATION_REQUIRED'
              : anatomyVerified
                ? 'ANATOMY_VERIFIED'
                : 'ANALYSIS_UNAVAILABLE',
        uploadStatus: isSaved ? 'uploaded' : 'failed',
        analysisStatus: responseStatus || 'unknown',
        analysisMessage: broadRegionAccepted
          ? broadRegionMessage
          : anatomyVerified && !isCompleted ? '' : (message || (isCompleted ? 'Image analysis completed.' : requiresRegionConfirmation ? 'The image is saved. Confirm or correct its body region to continue.' : 'The backend did not return a usable image-analysis result.')),
        nextAction: response?.next_action || (requiresRegionConfirmation || broadRegionAccepted ? 'confirm_or_select_body_region' : null),
        error: isCompleted || anatomyVerified || broadRegionAccepted || requiresRegionConfirmation ? null : message || 'The image was received, but automated interpretation is unavailable for this image.',
        detectedRegion: detectedRegion ? String(detectedRegion).toUpperCase() : null,
        candidateRegion: candidateRegion ? String(candidateRegion).toUpperCase() : null,
        verificationRegion: getSpecificRegionOption(candidateRegion || rawRegion, !isBroadRegion),
        topCandidates,
        hasCleanupCandidates: cleanupCandidates.length > 0,
        hasUpperExtremityCandidates: upperExtremityCandidates.length > 0,
        upperExtremityRefinementStatus: upperExtremityRefinement?.status || null,
        upperExtremityRefinementMessage: upperExtremityRefinement?.message || null,
        bodyPartConfidence,
        anatomyVerificationStatus: detectionStatus || 'unverified',
        anatomyUserConfirmed: false,
        classifierGatePassed: Boolean(response?.classifier_gate_passed ?? classifierGateAccepted),
        anatomyClassifierAccepted: Boolean(response?.anatomy_classifier_accepted ?? classifierGateAccepted),
        regionalModelConfigured: Boolean(response?.regional_model_configured),
        regionalModelAvailable: Boolean(response?.regional_model_available),
        regionalModelName: response?.regional_model_name || null,
        regionalModelStatus: response?.regional_model_status || null,
        primaryPrediction: response?.primary_prediction || detection?.primary_classifier_result || response?.anatomy_metadata?.classifier_output?.primary_classifier_result || null,
        specialistPrediction: response?.specialist_prediction || upperExtremityRefinement || response?.anatomy_metadata?.classifier_output?.upper_extremity_refinement || null,
        twoModelComparison: response?.two_model_comparison || response?.anatomy_metadata?.two_model_comparison || null,
        finalCandidate: response?.final_candidate || response?.anatomy_metadata?.final_candidate || response?.detected_region || response?.candidate_region || null,
        selectionSource: response?.selection_source || response?.anatomy_metadata?.selection_source || 'PRIMARY MODEL',
        confidenceSource: response?.confidence_source || response?.anatomy_metadata?.confidence_source || 'primary_model',
        verifyingAnatomy: false,
        prediction: prediction || null,
        findings,
        modelVersion: response?.model_version || null,
        resultId: response?.xray_result_id || null,
        reportRegionVerification: response?.report_region_verification || null,
        reasonCode: response?.reason_code || null,
        uncertainty: response?.uncertainty || null,
      });
    } catch (err) {
      updateImage(item.id, {
        status: 'FAILED',
        uploadStatus: 'failed',
        error: err?.message || 'X-ray upload or analysis failed. Retry this image.',
        analysisMessage: null,
      });
    } finally {
      processingImageIdsRef.current.delete(item.id);
    }
  };

  const handleConfirmAnatomy = async (item, directRegion = null) => {
    if (!assessmentId || !item.resultId) {
      setError('This image has no saved result ID. Remove it and upload the image again before confirming its region.');
      return;
    }
    const targetRegion = directRegion || item.verificationRegion || item.finalCandidate || item.detectedRegion || item.candidateRegion;
    const selectedRegion = getSpecificRegionOption(targetRegion);
    if (!selectedRegion) {
      setError('Choose a specific body region before confirming the X-ray.');
      return;
    }
    if (item.verifyingAnatomy || processingImageIdsRef.current.has(item.id)) return;

    updateImage(item.id, {
      verifyingAnatomy: true,
      error: null,
      analysisMessage: `Confirming ${selectedRegion.replace(/_/g, ' ').toLowerCase()} as the image region and checking model availability...`,
    });
    try {
      const response = await confirmXrayAnatomy(assessmentId, item.resultId, selectedRegion);
      const responseStatus = normalizeStatus(response?.status);
      const confirmedRegion = response?.detected_region || response?.region || selectedRegion;
      const findings = getResponseFindings(response);
      const prediction = typeof response?.prediction === 'string' ? response.prediction.trim() : '';
      const hasUsableModelOutput = Boolean(prediction || findings.length > 0);
      const isCompleted = responseStatus === 'completed' && hasUsableModelOutput;
      const message = response?.message || 'The selected region was saved.';
      const confidenceRaw = response?.body_part_confidence;
      const bodyPartConfidence = typeof confidenceRaw === 'number' && Number.isFinite(confidenceRaw)
        ? confidenceRaw
        : null;

      updateImage(item.id, {
        status: isCompleted ? 'COMPLETED' : 'ANALYSIS_UNAVAILABLE',
        uploadStatus: 'uploaded',
        analysisStatus: responseStatus || 'unavailable',
        analysisMessage: message,
        nextAction: response?.next_action || (isCompleted ? null : 'review_interpretation_status'),
        error: isCompleted ? null : message || 'The region was confirmed, but automated interpretation is unavailable.',
        detectedRegion: confirmedRegion ? String(confirmedRegion).toUpperCase() : selectedRegion,
        candidateRegion: null,
        verificationRegion: selectedRegion,
        bodyPartConfidence,
        anatomyVerificationStatus: 'confirmed',
        anatomyUserConfirmed: true,
        decisionState: response?.decision_state || (response?.anatomy_user_confirmed ? 'USER_CONFIRMED' : null),
        primaryPrediction: response?.primary_prediction || item.primaryPrediction,
        specialistPrediction: response?.specialist_prediction || item.specialistPrediction,
        twoModelComparison: response?.two_model_comparison || item.twoModelComparison,
        finalCandidate: response?.final_candidate || selectedRegion,
        selectionSource: response?.selection_source || item.selectionSource,
        confidenceSource: response?.confidence_source || item.confidenceSource,
        classifierGatePassed: Boolean(response?.classifier_gate_passed ?? response?.anatomy_classifier_accepted),
        anatomyClassifierAccepted: Boolean(response?.anatomy_classifier_accepted),
        regionalModelConfigured: Boolean(response?.regional_model_configured),
        regionalModelAvailable: Boolean(response?.regional_model_available),
        regionalModelName: response?.regional_model_name || null,
        regionalModelStatus: response?.regional_model_status || null,
        broadRegionAccepted: false,
        isBroadRegion: false,
        verifyingAnatomy: false,
        isEditingRegion: false,
        prediction: prediction || null,
        findings,
        modelVersion: response?.model_version || null,
        reasonCode: response?.reason_code || null,
        uncertainty: response?.uncertainty || null,
      });
      setPreviewState((previous) => previous && previous.id === item.id
        ? {
          ...previous,
          detectedRegion: confirmedRegion ? String(confirmedRegion).toUpperCase() : selectedRegion,
          candidateRegion: null,
          bodyPartConfidence,
          anatomyVerificationStatus: 'confirmed',
          broadRegionAccepted: false,
          anatomyUserConfirmed: true,
        }
        : previous);
      setInfo(isCompleted
        ? `Region confirmed as ${String(confirmedRegion).replace(/_/g, ' ')}. The configured model returned output.`
        : `Region confirmed as ${String(confirmedRegion).replace(/_/g, ' ')}. Automated interpretation remains unavailable for this region/model.`);
    } catch (err) {
      updateImage(item.id, {
        verifyingAnatomy: false,
        error: err?.message || 'Unable to save the confirmed X-ray region. Please retry.',
        analysisMessage: err?.message || 'Unable to save the confirmed X-ray region. Please retry.',
      });
    }
  };

  const extractReport = async (item) => {
    if (!assessmentId) {
      updateReport(item.id, {
        status: 'FAILED',
        uploadStatus: 'failed',
        error: 'No assessment is active. Start a new assessment before uploading a report.',
      });
      return;
    }
    if (processingReportIdsRef.current.has(item.id)) return;
    processingReportIdsRef.current.add(item.id);
    updateReport(item.id, {
      status: 'UPLOADING',
      uploadStatus: 'uploading',
      error: null,
    });

    try {
      const response = await uploadReport(assessmentId, item.file, 'xray_report', null);
      const rawStatus = normalizeStatus(response?.extraction_status || response?.status);
      const extractedText = getResponseText(response);
      const findings = getResponseFindings(response);
      const verification = response?.region_verification || response?.ocr_metadata?.region_verification || null;
      const reportRegion = verification?.report_region || response?.report_region || null;
      const verificationStatus = String(verification?.status || '').toUpperCase();
      const uploaded = response?.upload_status !== 'rejected' && response?.status !== 'failed';
      const successfulStatus = ['successful', 'success', 'complete', 'completed', 'extracted'].includes(rawStatus) || response?.status === 'extracted';
      const unreadable = ['unreadable', 'failed_to_read', 'no_text'].includes(rawStatus) || response?.status === 'unreadable';
      const completed = successfulStatus || (uploaded && (Boolean(extractedText) || findings.length > 0) && !unreadable);
      const status = completed ? 'COMPLETED' : unreadable ? 'UNREADABLE' : 'EXTRACTION_INCOMPLETE';
      const mismatchMessage = verificationStatus === 'MISMATCH'
        ? verification?.message || `The report identifies ${reportRegion || 'a region'} that conflicts with an X-ray image classification. Keep their findings separate until reviewed.`
        : null;

      updateReport(item.id, {
        status,
        uploadStatus: uploaded ? 'uploaded' : 'failed',
        reportId: response?.report_id || response?.id || null,
        extractionStatus: rawStatus || 'unknown',
        extractedText,
        extractedFindings: findings,
        reportRegion,
        regionVerificationStatus: verificationStatus || 'UNVERIFIED',
        regionWarning: mismatchMessage,
        error: completed ? null : response?.message || (unreadable ? 'The PDF text could not be read.' : 'The report uploaded, but extraction could not be confirmed.'),
        extractionMessage: response?.message || (completed ? 'Radiology report extraction completed.' : ''),
      });
    } catch (err) {
      updateReport(item.id, {
        status: 'FAILED',
        uploadStatus: 'failed',
        error: err?.message || 'Radiology report upload or extraction failed. Retry this report.',
      });
    } finally {
      processingReportIdsRef.current.delete(item.id);
    }
  };

  const analyzeImagesSequentially = async (items) => {
    // Keep inference requests sequential: the body-part classifier and regional
    // checkpoints are large, so concurrent uploads can multiply memory use.
    for (const item of items) {
      await analyzeImage(item);
    }
  };

  const extractReportsSequentially = async (items) => {
    for (const item of items) {
      await extractReport(item);
    }
  };

  const addImages = (fileList) => {
    if (!fileList || fileList.length === 0) return;
    const accepted = [];
    const rejected = [];

    for (let i = 0; i < fileList.length; i += 1) {
      const file = fileList[i];
      const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`;
      const validExtension = ALLOWED_IMAGE_EXTS.includes(ext);
      const validMime = ALLOWED_IMAGE_TYPES.includes((file.type || '').toLowerCase());
      if (!validExtension || (file.type && !validMime)) {
        rejected.push(file.name);
        continue;
      }
      accepted.push({
        id: `img-${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 9)}`,
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'READY',
        uploadStatus: 'pending',
        analysisStatus: null,
        analysisMessage: null,
        detectedRegion: null,
        candidateRegion: null,
        topCandidates: [],
        bodyPartConfidence: null,
        anatomyVerificationStatus: 'unverified',
        verificationRegion: '',
        verifyingAnatomy: false,
        anatomyUserConfirmed: false,
        prediction: null,
        findings: [],
        error: null,
      });
    }

    const existingKeys = new Set(xrayImages.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`));
    const additions = accepted.filter((item) => {
      const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
      if (existingKeys.has(key)) return false;
      existingKeys.add(key);
      return true;
    });

    if (additions.length > 0) {
      const queuedAdditions = additions.map((item) => ({
        ...item,
        status: 'UPLOADING',
        uploadStatus: 'uploading',
        analysisMessage: 'Queued for automatic image recognition and analysis...',
      }));
      setXrayImages((previous) => {
        const latestKeys = new Set(previous.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`));
        const safeAdditions = queuedAdditions.filter((item) => {
          const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
          if (latestKeys.has(key)) return false;
          latestKeys.add(key);
          return true;
        });
        return [...previous, ...safeAdditions];
      });
      setError('');
      setInfo('Image selected. Automatic image-type recognition and X-ray analysis are starting now.');
      // Run analysis immediately and sequentially; do not wait for Continue.
      void analyzeImagesSequentially(queuedAdditions);
    }

    if (rejected.length > 0) {
      setError(`Skipped unsupported image file(s): ${rejected.join(', ')}. Please use PNG, JPG, or JPEG images.`);
    }
  };

  const addReports = (fileList) => {
    if (!fileList || fileList.length === 0) return;
    const accepted = [];
    const rejected = [];

    for (let i = 0; i < fileList.length; i += 1) {
      const file = fileList[i];
      const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`;
      const validExtension = ext === '.pdf';
      const validMime = !file.type || file.type.toLowerCase() === 'application/pdf';
      if (!validExtension || !validMime) {
        rejected.push(`${file.name} (not a PDF)`);
        continue;
      }
      accepted.push({
        id: `rep-${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).slice(2, 9)}`,
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'READY',
        uploadStatus: 'pending',
        extractionStatus: null,
        extractedText: '',
        extractedFindings: [],
        reportRegion: null,
        regionVerificationStatus: 'UNVERIFIED',
        regionWarning: null,
        error: null,
      });
    }

    const existingKeys = new Set(xrayReports.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`));
    const additions = accepted.filter((item) => {
      const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
      if (existingKeys.has(key)) return false;
      existingKeys.add(key);
      return true;
    });

    if (additions.length > 0) {
      const queuedAdditions = additions.map((item) => ({
        ...item,
        status: 'UPLOADING',
        uploadStatus: 'uploading',
      }));
      setXrayReports((previous) => {
        const latestKeys = new Set(previous.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`));
        const safeAdditions = queuedAdditions.filter((item) => {
          const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
          if (latestKeys.has(key)) return false;
          latestKeys.add(key);
          return true;
        });
        return [...previous, ...safeAdditions];
      });
      setError('');
      setInfo('Radiology report selected. Upload and text extraction are starting now.');
      // Extract reports sequentially so status and resource use remain predictable.
      void extractReportsSequentially(queuedAdditions);
    }

    if (rejected.length > 0) {
      setError(`Skipped unsupported report file(s): ${rejected.join(', ')}. Please use PDF radiology reports.`);
    }
  };

  const handleOpenPreview = (item) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    try {
      const url = URL.createObjectURL(item.file);
      previewUrlRef.current = url;
      const isPdf = item.name.toLowerCase().endsWith('.pdf') || item.type === 'application/pdf';
      setPreviewState({
        id: item.id,
        file: item.file,
        name: item.name,
        size: item.size,
        previewUrl: url,
        previewType: isPdf ? 'pdf' : 'image',
        // Carry the existing backend classification into the preview modal.
        detectedRegion: item.detectedRegion || null,
        candidateRegion: item.candidateRegion || null,
        bodyPartConfidence:
          typeof item.bodyPartConfidence === 'number'
            ? item.bodyPartConfidence
            : null,
        anatomyVerificationStatus: item.anatomyVerificationStatus || 'unverified',
        broadRegionAccepted: item.broadRegionAccepted === true || item.status === 'BROAD_REGION_ACCEPTED',
        anatomyUserConfirmed: item.anatomyUserConfirmed === true,
      });
    } catch {
      setError(`Unable to generate preview for ${item.name}.`);
    }
  };

  const handleClosePreview = () => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setPreviewState(null);
  };

  const handleRemoveImage = async (idToRemove) => {
    const existing = xrayImages.find((item) => item.id === idToRemove);
    if (!existing) return;
    if (existing.status === 'UPLOADING' || existing.status === 'DELETING' || deletingImageIds.has(idToRemove)) {
      return;
    }

    // Files not yet persisted can be removed locally without a backend request.
    if (existing.uploadStatus !== 'uploaded') {
      if (previewState?.id === idToRemove) handleClosePreview();
      setXrayImages((previous) => previous.filter((item) => item.id !== idToRemove));
      setError('');
      return;
    }

    // Uploaded images must be removed from the database and storage, not just hidden.
    if (!assessmentId || !existing.resultId) {
      setError(
        'This image is saved on the server, but its result ID is missing. It cannot be safely removed from this screen. Refresh the assessment or contact support; do not hide the result locally.'
      );
      return;
    }

    setError('');
    setDeletingImageIds((previous) => new Set(previous).add(idToRemove));
    setXrayImages((previous) => previous.map((item) => (
      item.id === idToRemove ? { ...item, status: 'DELETING', error: null } : item
    )));

    try {
      const response = await deleteXray(assessmentId, existing.resultId);
      if (normalizeStatus(response?.status) !== 'deleted') {
        throw new Error(response?.message || 'The backend did not confirm deletion of this X-ray result.');
      }

      if (previewState?.id === idToRemove) handleClosePreview();
      setXrayImages((previous) => previous.filter((item) => item.id !== idToRemove));
      setInfo(response?.message || 'X-ray image and saved result removed from this assessment.');
    } catch (err) {
      setXrayImages((previous) => previous.map((item) => (
        item.id === idToRemove
          ? {
              ...item,
              status: existing.status === 'DELETING' ? 'ANALYSIS_UNAVAILABLE' : existing.status,
              uploadStatus: 'uploaded',
              error: err?.message || 'The saved X-ray could not be removed. Please retry.',
            }
          : item
      )));
      setError(err?.message || 'The saved X-ray could not be removed. The image remains in this assessment.');
    } finally {
      setDeletingImageIds((previous) => {
        const next = new Set(previous);
        next.delete(idToRemove);
        return next;
      });
    }
  };

  const handleRemoveReport = async (idToRemove) => {
    const existing = xrayReports.find((item) => item.id === idToRemove);
    if (!existing) return;
    if (
      existing.status === 'UPLOADING' ||
      existing.status === 'DELETING' ||
      deletingReportIds.has(idToRemove)
    ) return;

    // A report that has not been saved can be removed from the local queue.
    if (existing.uploadStatus !== 'uploaded') {
      if (previewState?.id === idToRemove) handleClosePreview();
      setXrayReports((previous) => previous.filter((item) => item.id !== idToRemove));
      setError('');
      return;
    }

    // Do not hide a persisted report if its backend identifier is missing.
    if (!assessmentId || !existing.reportId) {
      setError(
        'This radiology report is saved, but its report ID is missing. It cannot be safely removed. Refresh the assessment and try again.'
      );
      return;
    }

    setError('');
    setDeletingReportIds((previous) => new Set(previous).add(idToRemove));
    updateReport(idToRemove, { status: 'DELETING', error: null });

    try {
      const response = await deleteReport(assessmentId, existing.reportId);
      if (normalizeStatus(response?.status) !== 'deleted') {
        throw new Error(response?.message || 'The backend did not confirm deletion of this radiology report.');
      }

      if (previewState?.id === idToRemove) handleClosePreview();
      setXrayReports((previous) => previous.filter((item) => item.id !== idToRemove));
      setInfo(response?.message || 'Radiology report and its saved findings were removed from this assessment.');
    } catch (err) {
      setXrayReports((previous) => previous.map((item) => (
        item.id === idToRemove
          ? {
              ...item,
              status: existing.status,
              uploadStatus: 'uploaded',
              error: err?.message || 'The saved radiology report could not be removed. Please retry.',
            }
          : item
      )));
      setError(err?.message || 'The saved radiology report could not be removed. It remains in this assessment.');
    } finally {
      setDeletingReportIds((previous) => {
        const next = new Set(previous);
        next.delete(idToRemove);
        return next;
      });
    }
  };

  const retryImage = (item) => {
    if (item.uploadStatus === 'uploaded') return;
    void analyzeImage(item);
  };

  const retryReport = (item) => {
    if (item.uploadStatus === 'uploaded') return;
    void extractReport(item);
  };

  const onContinue = async () => {
    setError('');
    setInfo('');

    if (!assessmentId) {
      setError('No assessment is in progress. Start a new assessment first.');
      return;
    }
    if (!hasAnySelectedInput) {
      setError('Upload an X-ray image, a radiology report PDF, or both to continue.');
      return;
    }
    if (isProcessingFiles || submitting) {
      setError('Please wait until image analysis and report extraction finish.');
      return;
    }

    const usableImage = xrayImages.some((item) => item.uploadStatus === 'uploaded');
    const usableReport = xrayReports.some((item) => item.uploadStatus === 'uploaded' && item.status === 'COMPLETED');

    if (!usableImage && !usableReport) {
      const failedImage = xrayImages.some((item) => item.status === 'FAILED');
      const failedReport = xrayReports.some((item) => item.status === 'FAILED' || item.status === 'UNREADABLE' || item.status === 'EXTRACTION_INCOMPLETE');
      setError(
        failedImage || failedReport
          ? 'No usable X-ray image or extracted radiology report is available. Retry a failed input or select a readable file.'
          : 'Wait for at least one image analysis or report extraction to complete before continuing.'
      );
      return;
    }

    setSubmitting(true);
    const warnings = [];
    if (xrayImages.some((item) => item.uploadStatus === 'uploaded' && item.status === 'REGION_CONFIRMATION_REQUIRED')) {
      warnings.push('one or more saved X-ray images still need a body-region confirmation and were not sent to a region-specific interpretation model');
    }
    if (xrayImages.some((item) => item.uploadStatus === 'uploaded' && item.status === 'BROAD_REGION_ACCEPTED')) {
      warnings.push('one or more images passed the classifier threshold only at a broad-region level; select a specific region to run a compatible interpretation model');
    }
    if (xrayImages.some((item) => item.uploadStatus === 'uploaded' && item.status === 'ANALYSIS_UNAVAILABLE')) {
      warnings.push('one or more images were uploaded but did not produce a usable model result');
    }
    if (xrayImages.some((item) => item.status === 'FAILED')) warnings.push('some image uploads failed and were excluded');
    if (xrayReports.some((item) => item.status === 'EXTRACTION_INCOMPLETE')) warnings.push('some reports had incomplete structured extraction');
    if (xrayReports.some((item) => item.status === 'UNREADABLE')) warnings.push('some reports could not be read');
    if (xrayReports.some((item) => item.status === 'FAILED')) warnings.push('some reports failed to upload and were excluded');
    if (xrayReports.some((item) => item.regionVerificationStatus === 'MISMATCH' || item.regionWarning)) {
      warnings.push('a report/image type conflict was identified; conflicting findings must remain separate during assessment');
    }

    if (warnings.length > 0) setInfo(`${warnings.join('; ')}. These limitations should remain visible in the final assessment.`);
    else setInfo('Available X-ray image analysis and radiology report extraction completed. Continuing to combined assessment.');

    // The backend has already saved image results and uploaded/extracted reports.
    // Assessment processing is responsible for combining the persisted evidence.
    navigate('/assessment/processing');
  };

  return (
    <div className="assessment-workflow-page">
      <NeuralNetworkCanvas />
      <ThoracicRibcageVisual />
      <RadiologyFilmScreenVisual />
      <FloatingXrayObjects />

      <section className="assessment-glass-panel panel-narrow" aria-label="X-Ray and radiology evidence panel">
        <div className="na-top-badge">
          <span className="na-top-badge-pulse" />
          <SparkleIcon size={14} color="#0284c7" />
          <span>Automatic X-Ray &amp; Radiology Analysis · Goal 1</span>
        </div>

        <div className="na-header-section">
          <h1 className="na-title">
            X-Ray &amp; Radiology <span className="na-title-highlight">Evidence</span>
          </h1>
          <p className="na-description">
            Upload an X-ray image to automatically identify its supported image type and show available model findings. After image analysis finishes, you can optionally add a radiology report PDF. A report can also be processed on its own.
          </p>
        </div>

        <div className="na-progress-bar" aria-label="Assessment progress">
          <div className="na-progress-step completed">
            <span className="na-progress-number"><CheckIcon size={13} color="#ffffff" /></span>
            <span>Input</span>
          </div>
          <div className="na-progress-line completed" />
          <div className="na-progress-step active">
            <span className="na-progress-number">2</span>
            <span>X-Ray</span>
          </div>
          <div className="na-progress-line" />
          <div className="na-progress-step">
            <span className="na-progress-number">3</span>
            <span>Combine</span>
          </div>
          <div className="na-progress-line" />
          <div className="na-progress-step">
            <span className="na-progress-number">4</span>
            <span>Results</span>
          </div>
        </div>

        {error ? (
          <div className="na-error-box">
            <ErrorMessage message={error} onDismiss={() => setError('')} />
          </div>
        ) : null}

        {info ? (
          <div className="info-banner" style={{ width: '100%', marginBottom: '1rem' }} role="status">
            {info}
          </div>
        ) : null}

        {/* Stage 1: image upload and immediate automatic analysis */}
        <div className="section-divider-title">
          <h3>1. X-Ray Images</h3>
          <button
            type="button"
            className="section-upload-btn"
            onClick={() => imageInputRef.current?.click()}
            disabled={isProcessingFiles || submitting}
          >
            + Upload X-Ray Image
          </button>
        </div>
        <p className="upload-hint-text" style={{ textAlign: 'left', margin: '0 0 0.5rem 0' }}>
          Accepted: PNG, JPG, JPEG. No body-region selection is required. Each image is uploaded and analyzed automatically after selection.
        </p>

        <input
          ref={imageInputRef}
          type="file"
          multiple
          accept="image/png,image/jpeg,.png,.jpg,.jpeg"
          style={{ display: 'none' }}
          onChange={(e) => {
            addImages(e.target.files);
            e.target.value = '';
          }}
          aria-hidden="true"
        />

        <div
          className={`assessment-upload-dropzone ${dragActiveImages ? 'drag-active' : ''}`}
          style={{ padding: '1.25rem 1rem', minHeight: '90px', marginBottom: xrayImages.length > 0 ? '0.75rem' : '1.25rem' }}
          onDragEnter={handleImageDrag}
          onDragLeave={handleImageDrag}
          onDragOver={handleImageDrag}
          onDrop={handleImageDrop}
          onClick={() => !isProcessingFiles && imageInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload X-ray images"
          onKeyDown={(e) => {
            if ((e.key === ' ' || e.key === 'Enter') && !isProcessingFiles) {
              e.preventDefault();
              imageInputRef.current?.click();
            }
          }}
        >
          <p className="upload-primary-text" style={{ margin: 0, fontSize: '0.92rem' }}>
            Drag &amp; drop X-ray images, or <span className="upload-browse-link">Browse images</span>
          </p>
        </div>

        {xrayImages.length > 0 ? (
          <div className="file-list-container" style={{ marginBottom: '1.25rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>
              Selected Images ({xrayImages.length}):
            </span>
            {xrayImages.map((img) => {
              const statusView = getImageStatusPresentation(img);
              return (
                <div key={img.id} className="file-card-item" style={{ display: 'block' }}>
                  <div className="file-card-left" style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                    <div className="file-card-icon-badge badge-img">IMG</div>
                    <div className="file-card-info" style={{ flex: 1, minWidth: 0 }}>
                      <span className="file-card-name" title={img.name}>{img.name}</span>
                      <span className="file-card-meta">
                        {formatFileSize(img.size)} · <span className={statusView.className}>{statusView.label}</span>
                      </span>
                    </div>
                    <div className="file-card-actions">
                      <button type="button" className="btn-card-view" onClick={() => handleOpenPreview(img)} aria-label={`View ${img.name}`}>
                        View
                      </button>
                      {(img.status === 'FAILED') ? (
                        <button type="button" className="btn-card-view" onClick={() => retryImage(img)} disabled={isProcessingFiles || submitting}>
                          Retry
                        </button>
                      ) : null}
                      <button
                        type="button"
                        className="btn-card-remove"
                        onClick={() => handleRemoveImage(img.id)}
                        disabled={submitting || img.status === 'UPLOADING' || img.status === 'DELETING' || img.verifyingAnatomy || deletingImageIds.has(img.id)}
                        title={img.uploadStatus === 'uploaded' ? 'Delete this image and its saved result from the assessment.' : 'Remove selected image'}
                        aria-label={`Remove ${img.name}`}
                      >
                        {deletingImageIds.has(img.id) ? 'Removing...' : 'Remove'}
                      </button>
                    </div>
                  </div>

                  {(img.status === 'UPLOADING' || img.analysisMessage || img.error) ? (
                    <div style={{ margin: '0.65rem 0 0 2.8rem', fontSize: '0.83rem', color: img.status === 'FAILED' || img.status === 'ANALYSIS_UNAVAILABLE' ? '#92400e' : '#475569' }}>
                      {img.analysisMessage || img.error}
                    </div>
                  ) : null}

                  {(img.detectedRegion || img.candidateRegion || img.prediction || img.findings?.length > 0 || (img.uploadStatus === 'uploaded' && img.resultId)) ? (() => {
                    const candidate = String(img.finalCandidate || img.detectedRegion || img.candidateRegion || img.primaryPrediction?.candidate_body_part || 'UNKNOWN').toUpperCase();
                    const isConfirmed = Boolean(img.anatomyUserConfirmed);
                    const confirmedRegion = isConfirmed ? String(img.verificationRegion || img.detectedRegion || candidate).toUpperCase() : null;

                    const decisionState = img.decisionState || (isConfirmed ? 'USER_CONFIRMED' : (
                      (img.classifierGatePassed && !img.isBroadRegion && (typeof img.bodyPartConfidence === 'number' && img.bodyPartConfidence >= 0.60))
                        ? 'HIGH_CONFIDENCE_CANDIDATE'
                        : 'NEEDS_CONFIRMATION'
                    ));

                    const isLowConfidence = !isConfirmed && (
                      decisionState === 'NEEDS_CONFIRMATION' ||
                      decisionState === 'UNABLE_TO_CLASSIFY' ||
                      (typeof img.bodyPartConfidence === 'number' && img.bodyPartConfidence < 0.60)
                    );

                    let displayScore = null;
                    let scoreSource = 'Primary Model';
                    if (img.confidenceSource === 'specialist_refinement' && decisionState === 'HIGH_CONFIDENCE_CANDIDATE' && typeof img.specialistPrediction?.confidence === 'number') {
                      displayScore = (img.specialistPrediction.confidence * 100).toFixed(1);
                      scoreSource = 'Specialist Refinement';
                    } else if (typeof img.primaryPrediction?.confidence === 'number') {
                      displayScore = (img.primaryPrediction.confidence * 100).toFixed(1);
                      scoreSource = 'Primary 33-Class';
                    } else if (typeof img.bodyPartConfidence === 'number') {
                      displayScore = (img.bodyPartConfidence * 100).toFixed(1);
                      scoreSource = 'Primary 33-Class';
                    }

                    return (
                      <div style={{ margin: '0.75rem 0 0 2.8rem', padding: '0.9rem', borderRadius: '12px', border: '1px solid #cbd5e1', background: '#f8fafc', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                        {/* NORMAL CLINICAL VIEW */}
                        {isConfirmed ? (
                          <div style={{ padding: '0.85rem', background: '#ffffff', borderRadius: '10px', border: '1px solid #bbf7d0', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.45rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                              <div style={{ fontSize: '0.95rem', color: '#0f172a' }}>
                                <strong>Confirmed region:</strong>{' '}
                                <span style={{ fontSize: '1.1rem', fontWeight: 800, color: '#15803d', letterSpacing: '0.02em' }}>
                                  {confirmedRegion.replace(/_/g, ' ')}
                                </span>
                              </div>
                              <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.2rem 0.55rem', borderRadius: '6px', background: '#dcfce7', color: '#15803d', border: '1px solid #86efac' }}>
                                Status: {decisionState === 'USER_CORRECTED' ? 'User corrected' : 'User confirmed'}
                              </span>
                            </div>

                            <div style={{ fontSize: '0.84rem', color: '#334155', margin: '0.35rem 0' }}>
                              <strong>Compatible regional model:</strong>{' '}
                              {(img.regionalModelAvailable || img.regionalModelConfigured) ? (
                                <span style={{ color: '#0369a1', fontWeight: 600 }}>Available ({img.regionalModelName || 'Configured'})</span>
                              ) : (
                                <span style={{ color: '#64748b' }}>Anatomical region confirmed. No compatible regional analysis model is currently available.</span>
                              )}
                            </div>

                            {img.prediction ? (
                              <div style={{ marginTop: '0.45rem', padding: '0.45rem 0.65rem', background: '#f0fdf4', borderRadius: '6px', border: '1px solid #bbf7d0', fontSize: '0.84rem', color: '#166534' }}>
                                <strong>Regional Abnormality Result:</strong> {img.prediction}
                              </div>
                            ) : null}

                            {Array.isArray(img.findings) && img.findings.length > 0 ? (
                              <div style={{ marginTop: '0.4rem' }}>
                                <strong style={{ fontSize: '0.82rem', color: '#0f172a' }}>Model Findings:</strong>
                                <ul style={{ margin: '0.2rem 0 0 1.1rem', padding: 0, fontSize: '0.8rem', color: '#334155' }}>
                                  {img.findings.slice(0, 8).map((finding, index) => (
                                    <li key={`${img.id}-finding-${index}`}>{prettyFinding(finding, index)}</li>
                                  ))}
                                </ul>
                              </div>
                            ) : null}

                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.6rem', flexWrap: 'wrap' }}>
                              {(img.regionalModelAvailable || img.regionalModelConfigured) && !img.prediction && (!img.findings || img.findings.length === 0) ? (
                                <button
                                  type="button"
                                  className="btn-card-view"
                                  style={{ padding: '0.35rem 0.85rem', fontSize: '0.82rem', fontWeight: 700, background: '#0284c7', color: '#fff', border: 'none', borderRadius: '6px' }}
                                  onClick={() => handleConfirmAnatomy(img, confirmedRegion)}
                                  disabled={img.verifyingAnatomy || submitting}
                                >
                                  {img.verifyingAnatomy ? 'Running Regional Model...' : 'Run Abnormality Analysis'}
                                </button>
                              ) : null}
                              <button
                                type="button"
                                style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', background: 'transparent', color: '#475569', border: '1px solid #cbd5e1', borderRadius: '6px', cursor: 'pointer' }}
                                onClick={() => updateImage(img.id, { isEditingRegion: !img.isEditingRegion })}
                              >
                                {img.isEditingRegion ? 'Close Region Selector' : 'Change Region'}
                              </button>
                            </div>

                            {img.isEditingRegion ? (
                              <div style={{ marginTop: '0.6rem', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                <label htmlFor={`xray-search-input-${img.id}`} style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, marginBottom: '0.3rem', color: '#0f172a' }}>
                                  Select a new anatomical region (33 body regions):
                                </label>
                                <CardSearchableAnatomySelector
                                  imageId={img.id}
                                  value={img.verificationRegion || ''}
                                  onChange={(canonicalVal) => updateImage(img.id, { verificationRegion: canonicalVal })}
                                  disabled={img.verifyingAnatomy || submitting}
                                  aiPredictedRegion={candidate}
                                  confirmedRegion={confirmedRegion}
                                  isConfirmed={true}
                                />
                                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.5rem' }}>
                                  <button
                                    type="button"
                                    className="btn-card-view"
                                    style={{ padding: '0.35rem 0.8rem', fontSize: '0.8rem', fontWeight: 700 }}
                                    onClick={() => handleConfirmAnatomy(img, img.verificationRegion)}
                                    disabled={!img.verificationRegion || img.verifyingAnatomy || submitting}
                                  >
                                    {img.verifyingAnatomy ? 'Updating...' : 'Confirm New Region'}
                                  </button>
                                  <button
                                    type="button"
                                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', background: 'transparent', color: '#64748b', border: '1px solid #e2e8f0', borderRadius: '6px', cursor: 'pointer' }}
                                    onClick={() => updateImage(img.id, { isEditingRegion: false })}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : isLowConfidence ? (
                          <div style={{ padding: '0.85rem', background: '#ffffff', borderRadius: '10px', border: '1px solid #fde68a', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                              <div style={{ fontSize: '0.92rem', color: '#0f172a' }}>
                                <strong>Predicted region:</strong>{' '}
                                <span style={{ fontSize: '1.05rem', fontWeight: 800, color: '#92400e', letterSpacing: '0.02em' }}>
                                  {candidate.replace(/_/g, ' ')}
                                </span>
                              </div>
                              <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.2rem 0.55rem', borderRadius: '6px', background: '#fef3c7', color: '#b45309', border: '1px solid #fde68a' }}>
                                Status: Needs confirmation
                              </span>
                            </div>

                            <div style={{ fontSize: '0.82rem', color: '#475569', marginBottom: '0.4rem' }}>
                              <strong>AI confidence:</strong>{' '}
                              <span style={{ fontWeight: 700, color: '#d97706' }}>
                                {displayScore ? `${displayScore}%` : 'Low / Uncertain'}
                              </span>{' '}
                              <span style={{ fontSize: '0.74rem', color: '#64748b' }}>({scoreSource})</span>
                            </div>

                            <div style={{ padding: '0.45rem 0.65rem', background: '#fffbeb', borderRadius: '6px', border: '1px solid #fef3c7', fontSize: '0.82rem', color: '#92400e', marginBottom: '0.6rem' }}>
                              We could not identify this image reliably.
                            </div>

                            {!img.isEditingRegion ? (
                              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                                <button
                                  type="button"
                                  className="btn-card-view"
                                  style={{ padding: '0.4rem 0.9rem', fontSize: '0.82rem', fontWeight: 700, background: '#d97706', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
                                  onClick={() => updateImage(img.id, { isEditingRegion: true })}
                                >
                                  Select Correct Region
                                </button>
                              </div>
                            ) : (
                              <div style={{ marginTop: '0.5rem', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                <label htmlFor={`xray-search-input-${img.id}`} style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, marginBottom: '0.3rem', color: '#0f172a' }}>
                                  Select the correct anatomical region (33 body regions):
                                </label>
                                <CardSearchableAnatomySelector
                                  imageId={img.id}
                                  value={img.verificationRegion || ''}
                                  onChange={(canonicalVal) => updateImage(img.id, { verificationRegion: canonicalVal })}
                                  disabled={img.verifyingAnatomy || submitting}
                                  aiPredictedRegion={candidate}
                                  confirmedRegion=""
                                  isConfirmed={false}
                                />
                                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.5rem' }}>
                                  <button
                                    type="button"
                                    className="btn-card-view"
                                    style={{ padding: '0.35rem 0.8rem', fontSize: '0.8rem', fontWeight: 700 }}
                                    onClick={() => handleConfirmAnatomy(img, img.verificationRegion)}
                                    disabled={!img.verificationRegion || img.verifyingAnatomy || submitting}
                                  >
                                    {img.verifyingAnatomy ? 'Saving...' : 'Confirm Selected Region'}
                                  </button>
                                  <button
                                    type="button"
                                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', background: 'transparent', color: '#64748b', border: '1px solid #e2e8f0', borderRadius: '6px', cursor: 'pointer' }}
                                    onClick={() => updateImage(img.id, { isEditingRegion: false })}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        ) : (
                          <div style={{ padding: '0.85rem', background: '#ffffff', borderRadius: '10px', border: '1px solid #cbd5e1', marginBottom: '0.5rem' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.4rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                              <div style={{ fontSize: '0.92rem', color: '#0f172a' }}>
                                <strong>Predicted region:</strong>{' '}
                                <span style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f172a', letterSpacing: '0.02em' }}>
                                  {candidate.replace(/_/g, ' ')}
                                </span>
                              </div>
                              <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.2rem 0.55rem', borderRadius: '6px', background: '#eff6ff', color: '#1d4ed8', border: '1px solid #bfdbfe' }}>
                                Status: Awaiting confirmation
                              </span>
                            </div>

                            <div style={{ fontSize: '0.82rem', color: '#475569', marginBottom: '0.6rem' }}>
                              <strong>AI confidence:</strong>{' '}
                              <span style={{ fontWeight: 700, color: '#0369a1' }}>
                                {displayScore ? `${displayScore}%` : 'N/A'}
                              </span>{' '}
                              <span style={{ fontSize: '0.74rem', color: '#64748b' }}>({scoreSource})</span>
                            </div>

                            {!img.isEditingRegion ? (
                              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                                <button
                                  type="button"
                                  className="btn-card-view"
                                  style={{ padding: '0.4rem 0.95rem', fontSize: '0.82rem', fontWeight: 700, background: '#0284c7', color: '#ffffff', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
                                  onClick={() => handleConfirmAnatomy(img, candidate)}
                                  disabled={img.verifyingAnatomy || submitting}
                                >
                                  {img.verifyingAnatomy ? 'Confirming...' : 'Confirm Region'}
                                </button>
                                <button
                                  type="button"
                                  style={{ padding: '0.4rem 0.85rem', fontSize: '0.82rem', fontWeight: 600, background: 'transparent', color: '#475569', border: '1px solid #cbd5e1', borderRadius: '6px', cursor: 'pointer' }}
                                  onClick={() => updateImage(img.id, { isEditingRegion: true })}
                                  disabled={img.verifyingAnatomy || submitting}
                                >
                                  Change Region
                                </button>
                              </div>
                            ) : (
                              <div style={{ marginTop: '0.5rem', padding: '0.6rem', background: '#f8fafc', borderRadius: '8px', border: '1px solid #cbd5e1' }}>
                                <label htmlFor={`xray-search-input-${img.id}`} style={{ display: 'block', fontSize: '0.78rem', fontWeight: 700, marginBottom: '0.3rem', color: '#0f172a' }}>
                                  Select a different anatomical region (33 body regions):
                                </label>
                                <CardSearchableAnatomySelector
                                  imageId={img.id}
                                  value={img.verificationRegion || ''}
                                  onChange={(canonicalVal) => updateImage(img.id, { verificationRegion: canonicalVal })}
                                  disabled={img.verifyingAnatomy || submitting}
                                  aiPredictedRegion={candidate}
                                  confirmedRegion=""
                                  isConfirmed={false}
                                />
                                <div style={{ display: 'flex', gap: '0.4rem', marginTop: '0.5rem' }}>
                                  <button
                                    type="button"
                                    className="btn-card-view"
                                    style={{ padding: '0.35rem 0.8rem', fontSize: '0.8rem', fontWeight: 700 }}
                                    onClick={() => handleConfirmAnatomy(img, img.verificationRegion)}
                                    disabled={!img.verificationRegion || img.verifyingAnatomy || submitting}
                                  >
                                    {img.verifyingAnatomy ? 'Saving...' : 'Confirm Selected Region'}
                                  </button>
                                  <button
                                    type="button"
                                    style={{ padding: '0.35rem 0.75rem', fontSize: '0.78rem', background: 'transparent', color: '#64748b', border: '1px solid #e2e8f0', borderRadius: '6px', cursor: 'pointer' }}
                                    onClick={() => updateImage(img.id, { isEditingRegion: false })}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* COLLAPSED TECHNICAL DETAILS ACCORDION FOR AUDITING & DEVELOPERS */}
                        <details className="tech-details-accordion" style={{ marginTop: '0.5rem', borderTop: '1px solid #e2e8f0', paddingTop: '0.4rem' }}>
                          <summary style={{ cursor: 'pointer', fontSize: '0.78rem', fontWeight: 600, color: '#64748b', userSelect: 'none' }}>
                            Technical Details
                          </summary>
                          <div style={{ marginTop: '0.5rem', padding: '0.6rem', background: '#ffffff', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.4rem' }}>
                              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#475569' }}>Two-Model Evaluation &amp; Pipeline Metadata</span>
                              <span style={{ fontSize: '0.7rem', padding: '0.1rem 0.4rem', borderRadius: '4px', background: '#e0f2fe', color: '#0369a1', fontWeight: 600 }}>
                                Preprocessing: {img.preprocessingId || img.preprocessingMode || 'letterbox'}
                              </span>
                            </div>

                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.6rem', marginBottom: '0.5rem' }}>
                              {/* PRIMARY MODEL DETAILS */}
                              <div style={{ background: '#f8fafc', borderRadius: '6px', padding: '0.5rem', border: '1px solid #e2e8f0' }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#0284c7' }}>Primary 33-Class (ConvNeXt-Tiny)</span>
                                </div>
                                {img.primaryPrediction ? (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, margin: '0.2rem 0' }}>
                                      <span>{String(img.primaryPrediction.candidate_body_part || 'Unknown').replace(/_/g, ' ')}</span>
                                      <span>{typeof img.primaryPrediction.confidence === 'number' ? `${(img.primaryPrediction.confidence * 100).toFixed(2)}%` : 'N/A'}</span>
                                    </div>
                                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                                      Gate: {(typeof img.primaryPrediction.confidence === 'number' && img.primaryPrediction.confidence >= 0.6 && !img.primaryPrediction.is_broad_region) ? 'Passed (≥60%)' : 'Review Required'}
                                      {typeof img.primaryPrediction.top_two_margin === 'number' ? ` • Margin: +${(img.primaryPrediction.top_two_margin * 100).toFixed(1)}%` : ''}
                                    </div>
                                    {Array.isArray(img.primaryPrediction.top_predictions) && img.primaryPrediction.top_predictions.length > 0 ? (
                                      <div style={{ marginTop: '0.3rem', paddingTop: '0.3rem', borderTop: '1px dashed #e2e8f0' }}>
                                        <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600, marginBottom: '0.15rem' }}>Top Predictions:</div>
                                        {img.primaryPrediction.top_predictions.slice(0, 3).map((item, idx) => (
                                          <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#475569' }}>
                                            <span>{item.label}</span>
                                            <span>{typeof item.probability === 'number' ? `${(item.probability * 100).toFixed(1)}%` : ''}</span>
                                          </div>
                                        ))}
                                      </div>
                                    ) : null}
                                  </div>
                                ) : (
                                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>Primary model output unavailable.</div>
                                )}
                              </div>

                              {/* SPECIALIST MODEL DETAILS */}
                              <div style={{ background: '#f8fafc', borderRadius: '6px', padding: '0.5rem', border: '1px solid #e2e8f0' }}>
                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.2rem' }}>
                                  <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#7c3aed' }}>Specialist Model (DenseNet-121)</span>
                                  <span style={{ fontSize: '0.66rem', fontWeight: 700, padding: '0.05rem 0.35rem', borderRadius: '4px', background: '#f1f5f9', color: '#475569' }}>
                                    {(img.specialistPrediction?.status || 'NOT INVOKED').toUpperCase()}
                                  </span>
                                </div>
                                {img.specialistPrediction && (img.specialistPrediction.candidate_body_part || img.specialistPrediction.confidence) ? (
                                  <div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, margin: '0.2rem 0' }}>
                                      <span>{String(img.specialistPrediction.candidate_body_part || 'Not Applied').replace(/_/g, ' ')}</span>
                                      <span>{typeof img.specialistPrediction.confidence === 'number' ? `${(img.specialistPrediction.confidence * 100).toFixed(2)}%` : 'N/A'}</span>
                                    </div>
                                    <div style={{ fontSize: '0.7rem', color: '#64748b' }}>
                                      {img.specialistPrediction.message || 'Specialist evaluated.'}
                                    </div>
                                    {Array.isArray(img.specialistPrediction.top_predictions) && img.specialistPrediction.top_predictions.length > 0 ? (
                                      <div style={{ marginTop: '0.3rem', paddingTop: '0.3rem', borderTop: '1px dashed #e2e8f0' }}>
                                        <div style={{ fontSize: '0.68rem', color: '#64748b', fontWeight: 600, marginBottom: '0.15rem' }}>Top Predictions:</div>
                                        {img.specialistPrediction.top_predictions.slice(0, 3).map((item, idx) => (
                                          <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#475569' }}>
                                            <span>{item.label}</span>
                                            <span>{typeof item.probability === 'number' ? `${(item.probability * 100).toFixed(1)}%` : ''}</span>
                                          </div>
                                        ))}
                                      </div>
                                    ) : null}
                                  </div>
                                ) : (
                                  <div style={{ fontSize: '0.72rem', color: '#64748b' }}>
                                    {img.specialistPrediction?.message || 'Specialist not invoked (ineligible or non-upper-limb).'}
                                  </div>
                                )}
                              </div>
                            </div>

                            {/* DECISION SUMMARY & SAFETY BOUNDARY */}
                            <div style={{ fontSize: '0.72rem', color: '#475569', background: '#f8fafc', padding: '0.4rem', borderRadius: '5px', border: '1px solid #e2e8f0' }}>
                              <div><strong>Decision Case:</strong> {img.twoModelComparison?.decision_case || 'STANDARD'} • <strong>Source:</strong> {img.selectionSource || 'PRIMARY MODEL'}</div>
                              {img.twoModelComparison?.summary ? <div style={{ marginTop: '0.2rem' }}>{img.twoModelComparison.summary}</div> : null}
                              <div style={{ marginTop: '0.3rem', color: '#d97706' }}>
                                ⚠ Anatomy classification is not disease detection. Regional model will NOT execute before explicit user confirmation.
                              </div>
                            </div>
                          </div>
                        </details>
                      </div>
                    );
                  })() : null}
                </div>
              );
            })}
          </div>
        ) : null}

        {!hasImageStageSettled && xrayImages.length > 0 ? (
          <p role="status" className="upload-hint-text" style={{ margin: '0 0 1rem' }}>
            Image processing is in progress. The optional radiology-report section will appear when the image request returns.
          </p>
        ) : null}

        {/* Stage 2: report upload appears after image analysis, or immediately in report-only mode */}
        {reportStageVisible ? (
          <>
            <div className="section-divider-title">
              <h3>2. Optional Radiology Report</h3>
              <button
                type="button"
                className="section-upload-btn"
                onClick={() => reportInputRef.current?.click()}
                disabled={isProcessingFiles || submitting}
              >
                + Upload Radiology Report
              </button>
            </div>
            <p className="upload-hint-text" style={{ textAlign: 'left', margin: '0 0 0.5rem 0' }}>
              Accepted: PDF. The report is optional. After selection, MediSense AI uploads it, extracts text and displays available report information.
            </p>

            <input
              ref={reportInputRef}
              type="file"
              multiple
              accept="application/pdf,.pdf"
              style={{ display: 'none' }}
              onChange={(e) => {
                addReports(e.target.files);
                e.target.value = '';
              }}
              aria-hidden="true"
            />

            <div
              className={`assessment-upload-dropzone ${dragActiveReports ? 'drag-active' : ''}`}
              style={{ padding: '1.25rem 1rem', minHeight: '90px', marginBottom: xrayReports.length > 0 ? '0.75rem' : '1.25rem' }}
              onDragEnter={handleReportDrag}
              onDragLeave={handleReportDrag}
              onDragOver={handleReportDrag}
              onDrop={handleReportDrop}
              onClick={() => !isProcessingFiles && reportInputRef.current?.click()}
              role="button"
              tabIndex={0}
              aria-label="Upload radiology reports"
              onKeyDown={(e) => {
                if ((e.key === ' ' || e.key === 'Enter') && !isProcessingFiles) {
                  e.preventDefault();
                  reportInputRef.current?.click();
                }
              }}
            >
              <p className="upload-primary-text" style={{ margin: 0, fontSize: '0.92rem' }}>
                Drag &amp; drop a radiology PDF, or <span className="upload-browse-link">Browse reports</span>
              </p>
            </div>

            {xrayReports.length > 0 ? (
              <div className="file-list-container" style={{ marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>
                  Selected Reports ({xrayReports.length}):
                </span>
                {xrayReports.map((rep) => {
                  const statusView = getReportStatusPresentation(rep);
                  return (
                    <div key={rep.id} className="file-card-item" style={{ display: 'block' }}>
                      <div className="file-card-left" style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                        <div className="file-card-icon-badge badge-pdf">PDF</div>
                        <div className="file-card-info" style={{ flex: 1, minWidth: 0 }}>
                          <span className="file-card-name" title={rep.name}>{rep.name}</span>
                          <span className="file-card-meta">
                            {formatFileSize(rep.size)} · <span className={statusView.className}>{statusView.label}</span>
                          </span>
                        </div>
                        <div className="file-card-actions">
                          <button type="button" className="btn-card-view" onClick={() => handleOpenPreview(rep)} aria-label={`View ${rep.name}`}>
                            View
                          </button>
                          {rep.status === 'FAILED' ? (
                            <button type="button" className="btn-card-view" onClick={() => retryReport(rep)} disabled={isProcessingFiles || submitting}>
                              Retry
                            </button>
                          ) : null}
                          <button
                            type="button"
                            className="btn-card-remove"
                            onClick={() => handleRemoveReport(rep.id)}
                            disabled={submitting || rep.status === 'UPLOADING' || rep.status === 'DELETING' || deletingReportIds.has(rep.id)}
                            title={rep.uploadStatus === 'uploaded' ? 'Delete this radiology report and its saved findings from the assessment.' : 'Remove selected report'}
                            aria-label={`Remove ${rep.name}`}
                          >
                            {deletingReportIds.has(rep.id) ? 'Removing...' : 'Remove'}
                          </button>
                        </div>
                      </div>

                      {rep.error ? (
                        <p role="alert" style={{ margin: '0.6rem 0 0 2.8rem', color: '#92400e', fontSize: '0.83rem' }}>{rep.error}</p>
                      ) : null}
                      {rep.regionWarning ? (
                        <p role="note" style={{ margin: '0.6rem 0 0 2.8rem', color: '#92400e', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '0.5rem 0.65rem', fontSize: '0.82rem' }}>
                          {rep.regionWarning}
                        </p>
                      ) : null}
                      {(rep.extractedText || rep.extractedFindings?.length > 0) ? (
                        <div style={{ margin: '0.7rem 0 0 2.8rem', padding: '0.75rem', borderRadius: '10px', border: '1px solid #d1fae5', background: '#f0fdf4' }}>
                          <div style={{ fontWeight: 700, color: '#065f46', marginBottom: '0.35rem' }}>Extracted report information</div>
                          {rep.reportRegion ? <p style={{ margin: '0.2rem 0', fontSize: '0.84rem' }}><strong>Region mentioned by report:</strong> {rep.reportRegion}</p> : null}
                          {rep.extractedFindings?.length > 0 ? (
                            <ul style={{ margin: '0.3rem 0 0 1.1rem', padding: 0, fontSize: '0.84rem' }}>
                              {rep.extractedFindings.slice(0, 10).map((finding, index) => (
                                <li key={`${rep.id}-report-finding-${index}`}>{prettyFinding(finding, index)}</li>
                              ))}
                            </ul>
                          ) : null}
                          {rep.extractedText ? (
                            <details style={{ marginTop: '0.45rem' }}>
                              <summary style={{ cursor: 'pointer', fontSize: '0.83rem', fontWeight: 600 }}>View extracted text</summary>
                              <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: '220px', overflow: 'auto', marginTop: '0.4rem', fontSize: '0.8rem', fontFamily: 'inherit' }}>
                                {rep.extractedText.slice(0, 12000)}{rep.extractedText.length > 12000 ? '\n\n[Text truncated in preview.]' : ''}
                              </pre>
                            </details>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ) : null}
          </>
        ) : null}

        <div className="na-panel-footer" style={{ marginTop: '1.5rem' }}>
          <button
            type="button"
            className="na-continue-btn"
            onClick={onContinue}
            disabled={submitting || isProcessingFiles || !hasAnySelectedInput || (!hasUploadedImage && !hasExtractedReport)}
          >
            <span>{isProcessingFiles ? 'Analyzing selected inputs...' : submitting ? 'Opening combined assessment...' : 'Combine Evidence & Continue'}</span>
            <span className="na-btn-arrow"><ArrowRightIcon size={18} color="#ffffff" /></span>
          </button>

          <div className="na-security-footnote">
            <span className="na-security-icon"><ShieldLockIcon size={15} color="#10b981" /></span>
            <span>Secure Upload · AI-assisted processing · Not a medical diagnosis</span>
          </div>
        </div>
      </section>

      {previewState ? (
        <FilePreviewModal
          file={previewState.file}
          previewUrl={previewState.previewUrl}
          previewType={previewState.previewType}
          onClose={handleClosePreview}
          detectedRegion={previewState.detectedRegion}
          candidateRegion={previewState.candidateRegion}
          bodyPartConfidence={previewState.bodyPartConfidence}
          anatomyVerificationStatus={previewState.anatomyVerificationStatus}
          broadRegionAccepted={previewState.broadRegionAccepted === true}
          anatomyUserConfirmed={previewState.anatomyUserConfirmed === true}
        />
      ) : null}
    </div>
  );
}
