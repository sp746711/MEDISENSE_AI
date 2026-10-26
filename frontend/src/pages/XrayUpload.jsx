import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import FilePreviewModal from '../components/FilePreviewModal';
import XrayRegionSelector from '../components/XrayRegionSelector';
import { uploadXray } from '../services/xrayService';
import { uploadReport } from '../services/reportService';
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

/* ============================================================
   RESPONSE / STATUS HELPERS
   ============================================================ */
function normalizeRegionForComparison(value) {
  if (typeof value !== 'string') return '';
  let key = value.trim().toLowerCase();
  key = key.replace(/^(?:x[- ]?ray|xray|xr)[\s_-]*/, '');
  key = key.replace(/[^a-z0-9]/g, '');
  const aliases = {
    finger: 'fingers',
    digit: 'fingers',
    digits: 'fingers',
    thorax: 'chest',
    thoracic: 'chest',
    cranium: 'skull',
  };
  return aliases[key] || key;
}

// A conservative filename hint catches obvious conflicts such as
// `Clinical Hand X-Ray Radiograph.png` when the user selected Ankle.
// This is only a preliminary metadata check; it does NOT inspect image pixels.
function inferRegionFromFilename(filename) {
  if (typeof filename !== 'string' || !filename.trim()) return null;
  const stem = filename
    .replace(/\.[^.]+$/, '')
    .replace(/[_-]+/g, ' ')
    .toLowerCase();

  const hints = [
    { region: 'fingers', pattern: /\b(finger|fingers|digit|digits|phalanges)\b/ },
    { region: 'wrist', pattern: /\b(wrist|carpal)\b/ },
    { region: 'elbow', pattern: /\b(elbow|olecranon)\b/ },
    { region: 'forearm', pattern: /\b(forearm)\b/ },
    { region: 'hand', pattern: /\b(hand|hands|metacarpal)\b/ },
    { region: 'humerus', pattern: /\b(humerus|humeral)\b/ },
    { region: 'shoulder', pattern: /\b(shoulder|clavicle|scapula)\b/ },
    { region: 'ankle', pattern: /\b(ankle|malleolus)\b/ },
    { region: 'foot', pattern: /\b(foot|feet|toe|toes|tarsal|metatarsal)\b/ },
    { region: 'knee', pattern: /\b(knee|patella)\b/ },
    { region: 'leg', pattern: /\b(leg|tibia|fibula|tibial|fibular)\b/ },
    { region: 'hip', pattern: /\b(hip|femur|femoral)\b/ },
    { region: 'pelvis', pattern: /\b(pelvis|pelvic)\b/ },
    { region: 'skull', pattern: /\b(skull|cranium)\b/ },
    { region: 'head', pattern: /\b(head)\b/ },
    { region: 'chest', pattern: /\b(chest|thorax|thoracic|lung|lungs)\b/ },
    { region: 'spine', pattern: /\b(spine|spinal|vertebra|vertebrae)\b/ },
  ];

  const matches = [...new Set(hints.filter((hint) => hint.pattern.test(stem)).map((hint) => hint.region))];
  // Ambiguous filenames mentioning multiple body parts are not guessed.
  return matches.length === 1 ? matches[0] : null;
}

function getImageStatusPresentation(item) {
  switch (item.status) {
    case 'READY':
      return { className: 'status-ready', label: 'Ready for upload' };
    case 'UPLOADING':
      return { className: 'status-uploading', label: 'Uploading and analyzing...' };
    case 'COMPLETED':
      return { className: 'status-completed', label: 'Image inference completed' };
    case 'ANALYSIS_UNAVAILABLE':
      return {
        className: 'status-failed',
        label: item.error || 'Image uploaded; model analysis unavailable',
      };
    case 'REGION_MISMATCH':
      return {
        className: 'status-failed',
        label: item.error || 'Returned region does not match the selected region',
      };
    case 'FAILED':
    default:
      return { className: 'status-failed', label: item.error || 'Image upload failed' };
  }
}

function getReportStatusPresentation(item) {
  switch (item.status) {
    case 'READY':
      return { className: 'status-ready', label: 'Ready for upload' };
    case 'UPLOADING':
      return { className: 'status-uploading', label: 'Uploading and extracting text...' };
    case 'COMPLETED':
      return { className: 'status-completed', label: 'Text extracted' };
    case 'EXTRACTION_INCOMPLETE':
      return {
        className: 'status-failed',
        label: item.error || 'Report uploaded; structured extraction incomplete',
      };
    case 'UNREADABLE':
      return {
        className: 'status-failed',
        label: item.error || 'Report uploaded; text could not be read',
      };
    case 'REGION_MISMATCH':
      return {
        className: 'status-failed',
        label: item.error || 'Radiology report region does not match the selected region',
      };
    case 'FAILED':
    default:
      return { className: 'status-failed', label: item.error || 'Report upload failed' };
  }
}

/* ============================================================
   MAIN X-RAY & RADIOLOGY UPLOAD COMPONENT (MULTI-FILE)
   ============================================================ */
export default function XrayUpload() {
  const navigate = useNavigate();
  const { assessmentId } = useAssessment();
  const [xrayImages, setXrayImages] = useState([]);
  const [xrayReports, setXrayReports] = useState([]);
  const [region, setRegion] = useState(''); // Initial state must be empty; user must explicitly select
  const [info, setInfo] = useState('');
  const [error, setError] = useState('');
  const [regionMismatchMessage, setRegionMismatchMessage] = useState('');
  const [regionStatusMessage, setRegionStatusMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [dragActiveImages, setDragActiveImages] = useState(false);
  const [dragActiveReports, setDragActiveReports] = useState(false);
  const [previewState, setPreviewState] = useState(null); // { id, file, name, size, previewUrl, previewType }

  const imageInputRef = useRef(null);
  const reportInputRef = useRef(null);
  const previewUrlRef = useRef(null);

  // Clean up object URL when component genuinely unmounts
  useEffect(() => {
    return () => {
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
        previewUrlRef.current = null;
      }
    };
  }, []);

  const ALLOWED_IMAGE_EXTS = ['.png', '.jpg', '.jpeg'];
  const ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/jpg'];

  // Disable continuation immediately for a clear filename/selected-region conflict.
  // This is intentionally conservative and does not claim visual anatomy verification.
  const filenameRegionMismatches = xrayImages
    .map((item) => ({ item, filenameRegion: inferRegionFromFilename(item.name) }))
    .filter(({ filenameRegion }) => (
      Boolean(region) &&
      Boolean(filenameRegion) &&
      normalizeRegionForComparison(filenameRegion) !== normalizeRegionForComparison(region)
    ));
  const hasRegionMismatch = Boolean(regionMismatchMessage) || filenameRegionMismatches.length > 0 ||
    xrayImages.some((item) => item.status === 'REGION_MISMATCH') ||
    xrayReports.some((item) => item.status === 'REGION_MISMATCH');

  const handleImageDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActiveImages(true);
    } else if (e.type === 'dragleave') {
      setDragActiveImages(false);
    }
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveImages(false);
    if (e.dataTransfer.files) {
      addImages(e.dataTransfer.files);
    }
  };

  const handleReportDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActiveReports(true);
    } else if (e.type === 'dragleave') {
      setDragActiveReports(false);
    }
  };

  const handleReportDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActiveReports(false);
    if (e.dataTransfer.files) {
      addReports(e.dataTransfer.files);
    }
  };

  const addImages = (fileList) => {
    if (!fileList || fileList.length === 0) return;

    const accepted = [];
    const rejected = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`;
      const validExtension = ALLOWED_IMAGE_EXTS.includes(ext);
      const validMime = ALLOWED_IMAGE_TYPES.includes((file.type || '').toLowerCase());

      // Accept a known image extension when browsers provide an empty MIME type,
      // but reject files whose extension and declared MIME type both look invalid.
      if (!validExtension || (file.type && !validMime)) {
        rejected.push(file.name);
        continue;
      }

      accepted.push({
        id: `img-${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).substring(2, 9)}`,
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'READY',
        uploadStatus: 'pending',
        analysisStatus: null,
        analysisMessage: null,
        submittedRegion: null,
        error: null,
      });
    }

    setXrayImages((previous) => {
      const existingKeys = new Set(
        previous.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`)
      );
      const additions = accepted.filter((item) => {
        const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
        if (existingKeys.has(key)) return false;
        existingKeys.add(key);
        return true;
      });
      return [...previous, ...additions];
    });

    if (rejected.length > 0) {
      setError(`Skipped unsupported image file(s): ${rejected.join(', ')}. Please use PNG, JPG, or JPEG images.`);
    } else {
      setError('');
    }
  };

  const addReports = (fileList) => {
    if (!fileList || fileList.length === 0) return;

    const accepted = [];
    const rejected = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const ext = `.${(file.name.split('.').pop() || '').toLowerCase()}`;
      const validExtension = ext === '.pdf';
      const validMime = !file.type || file.type.toLowerCase() === 'application/pdf';

      if (!validExtension || !validMime) {
        rejected.push(`${file.name} (not a PDF)`);
        continue;
      }

      // Filename keywords are not reliable proof of document content.
      // Let report processing classify the actual document rather than blocking
      // a valid radiology PDF because its filename happens to contain "CBC", etc.
      accepted.push({
        id: `rep-${file.name}-${file.size}-${file.lastModified}-${Math.random().toString(36).substring(2, 9)}`,
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        status: 'READY',
        uploadStatus: 'pending',
        extractionStatus: null,
        error: null,
      });
    }

    setXrayReports((previous) => {
      const existingKeys = new Set(
        previous.map((item) => `${item.file.name}|${item.file.size}|${item.file.lastModified}`)
      );
      const additions = accepted.filter((item) => {
        const key = `${item.file.name}|${item.file.size}|${item.file.lastModified}`;
        if (existingKeys.has(key)) return false;
        existingKeys.add(key);
        return true;
      });
      return [...previous, ...additions];
    });

    if (rejected.length > 0) {
      setError(`Skipped unsupported report file(s): ${rejected.join(', ')}. Please use PDF radiology reports.`);
    } else {
      setError('');
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

  const handleRemoveImage = (idToRemove) => {
    if (previewState && previewState.id === idToRemove) {
      handleClosePreview();
    }
    setXrayImages((prev) => prev.filter((item) => item.id !== idToRemove));
  };

  const handleRemoveReport = (idToRemove) => {
    if (previewState && previewState.id === idToRemove) {
      handleClosePreview();
    }
    setXrayReports((prev) => prev.filter((item) => item.id !== idToRemove));
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  };

  const onContinue = async () => {
    setError('');
    setInfo('');
    setRegionMismatchMessage('');
    setRegionStatusMessage('');

    if (!assessmentId) {
      setError('No assessment is in progress. Start a new assessment first.');
      return;
    }

    // Goal 1: either source is valid on its own. A radiology PDF is optional.
    if (xrayImages.length === 0 && xrayReports.length === 0) {
      setError('Upload an X-ray image, a radiology report PDF, or both to continue.');
      return;
    }
    if (xrayImages.length > 0 && !region.trim()) {
      setError('Please select the anatomical region for the X-ray image(s).');
      return;
    }

    if (filenameRegionMismatches.length > 0) {
      const { item, filenameRegion } = filenameRegionMismatches[0];
      setRegionMismatchMessage(
        `Possible region mismatch: the selected region is "${region}", but the filename "${item.name}" suggests "${filenameRegion}". Change the selected region or upload the correct image. This preliminary check uses the filename only; it does not verify anatomy from image pixels.`
      );
      return;
    }

    setSubmitting(true);
    let uploadFailed = false;
    let regionMismatchDetected = false;
    let unavailableImageAnalyses = 0;
    let incompleteReports = 0;
    let unreadableReports = 0;
    let successfullyExtractedReports = 0;
    let inlineMismatch = '';
    let inlineRegionNotice = '';

    const selectedRegionKey = normalizeRegionForComparison(region);

    // Upload optional radiology reports first, when the user supplied any.
    // The backend compares the explicitly extracted report region with the
    // selected region when an image is part of this assessment.
    const updatedReports = xrayReports.map((item) => ({ ...item }));
    for (let i = 0; i < updatedReports.length; i++) {
      const item = updatedReports[i];

      if (item.uploadStatus === 'uploaded') {
        const existingRegion = normalizeRegionForComparison(item.reportRegion);
        if (xrayImages.length > 0 && existingRegion && selectedRegionKey && existingRegion !== selectedRegionKey) {
          regionMismatchDetected = true;
          inlineMismatch = `Region mismatch: the selected X-ray region is "${region}", but this report identifies "${item.reportRegion}". Correct the region or report before continuing.`;
          item.status = 'REGION_MISMATCH';
          item.error = inlineMismatch;
          break;
        }
        if (item.status === 'COMPLETED') successfullyExtractedReports += 1;
        else if (item.status === 'EXTRACTION_INCOMPLETE') incompleteReports += 1;
        else if (item.status === 'UNREADABLE') unreadableReports += 1;
        continue;
      }

      item.status = 'UPLOADING';
      item.error = null;
      setXrayReports([...updatedReports]);

      try {
        const response = await uploadReport(
          assessmentId,
          item.file,
          'xray_report',
          xrayImages.length > 0 ? region : null
        );
        const verification = response?.region_verification || response?.ocr_metadata?.region_verification || null;
        const reportRegion = verification?.report_region || response?.report_region || null;
        const verificationStatus = String(verification?.status || '').toUpperCase();

        item.reportRegion = reportRegion;
        item.regionVerificationStatus = verificationStatus || 'UNVERIFIED';
        item.extractionStatus = response?.extraction_status || response?.status || 'unknown';

        if (response?.status === 'region_mismatch' || verificationStatus === 'MISMATCH') {
          regionMismatchDetected = true;
          item.uploadStatus = response?.upload_status === 'rejected' ? 'failed' : 'uploaded';
          item.status = 'REGION_MISMATCH';
          item.error = verification?.message || response?.message || `The report region (${reportRegion || 'unknown'}) does not match the selected region (${region}).`;
          inlineMismatch = item.error;
          setXrayReports([...updatedReports]);
          break;
        }

        // Fail closed on any explicit backend mismatch. A report with no
        // confidently extractable region is unverified, not automatically a match.
        if (xrayImages.length > 0 && reportRegion && selectedRegionKey &&
            normalizeRegionForComparison(reportRegion) !== selectedRegionKey) {
          regionMismatchDetected = true;
          item.uploadStatus = response?.upload_status === 'rejected' ? 'failed' : 'uploaded';
          item.status = 'REGION_MISMATCH';
          item.error = `Region mismatch: selected region is "${region}", but the report says "${reportRegion}".`;
          inlineMismatch = item.error;
          setXrayReports([...updatedReports]);
          break;
        }

        // A 200 response alone is not proof that OCR/extraction succeeded.
        if (item.extractionStatus === 'successful' || response?.status === 'extracted') {
          item.uploadStatus = response?.upload_status === 'rejected' ? 'failed' : 'uploaded';
          item.status = 'COMPLETED';
          item.error = null;
          successfullyExtractedReports += 1;
        } else if (item.extractionStatus === 'incomplete' || response?.status === 'extraction_incomplete') {
          item.uploadStatus = 'uploaded';
          item.status = 'EXTRACTION_INCOMPLETE';
          item.error = response?.message || 'Text was extracted, but structured findings may be incomplete.';
          incompleteReports += 1;
        } else if (item.extractionStatus === 'unreadable' || response?.status === 'unreadable') {
          item.uploadStatus = 'uploaded';
          item.status = 'UNREADABLE';
          item.error = response?.message || 'The report was uploaded but its text could not be read.';
          unreadableReports += 1;
        } else {
          item.uploadStatus = response?.upload_status === 'rejected' ? 'failed' : 'uploaded';
          item.status = 'EXTRACTION_INCOMPLETE';
          item.error = response?.message || 'The report was uploaded, but extraction status was not confirmed.';
          incompleteReports += 1;
        }

        if (xrayImages.length > 0 && verificationStatus === 'UNVERIFIED') {
          inlineRegionNotice = verification?.message || 'The backend could not confidently identify the region in this report. The region has not been marked as verified.';
        }
      } catch (err) {
        uploadFailed = true;
        item.status = 'FAILED';
        item.uploadStatus = 'failed';
        item.error = err?.message || 'Radiology report upload failed.';
      }
      setXrayReports([...updatedReports]);
    }

    setXrayReports([...updatedReports]);
    if (regionMismatchDetected) {
      setSubmitting(false);
      setRegionMismatchMessage(inlineMismatch || 'The backend detected a region mismatch. Correct the selected region or radiology report before continuing.');
      return;
    }
    if (uploadFailed) {
      setSubmitting(false);
      setError('A selected radiology report failed to upload. Retry it or remove it, then continue with the remaining valid input.');
      return;
    }

    // A report-only assessment is permitted. At least one PDF must have yielded
    // usable extracted text/findings if no image is present.
    if (xrayImages.length === 0) {
      setSubmitting(false);
      if (successfullyExtractedReports === 0) {
        setError('No usable radiology-report extraction is available. Upload a readable PDF or add an X-ray image to continue.');
        return;
      }

      const reportNotices = [];
      if (incompleteReports > 0) reportNotices.push(`${incompleteReports} report(s) had incomplete structured extraction`);
      if (unreadableReports > 0) reportNotices.push(`${unreadableReports} report(s) could not be read`);
      setInfo(reportNotices.length ? `${reportNotices.join('; ')}. The successfully extracted report(s) will be included in the assessment.` : 'Radiology report extraction completed.');
      navigate('/assessment/processing');
      return;
    }

    // Image-only is allowed. The backend currently routes by the selected region;
    // without a dedicated anatomy classifier, it cannot prove anatomy from pixels.
    const updatedImages = xrayImages.map((item) => ({ ...item }));
    for (let i = 0; i < updatedImages.length; i++) {
      const item = updatedImages[i];
      if (item.uploadStatus === 'uploaded') {
        const previousRegion = normalizeRegionForComparison(item.submittedRegion);
        if (previousRegion && previousRegion !== selectedRegionKey) {
          regionMismatchDetected = true;
          inlineMismatch = `This image was previously submitted for "${item.submittedRegion}". Start a new assessment to use a different region.`;
          item.status = 'REGION_MISMATCH';
          item.error = inlineMismatch;
          break;
        }
        if (item.status === 'REGION_MISMATCH') {
          regionMismatchDetected = true;
          inlineMismatch = item.error || 'The backend previously detected a region mismatch for this image.';
          break;
        }
        if (item.status === 'ANALYSIS_UNAVAILABLE') unavailableImageAnalyses += 1;
        continue;
      }

      item.status = 'UPLOADING';
      item.error = null;
      setXrayImages([...updatedImages]);
      try {
        const response = await uploadXray(assessmentId, item.file, region);
        const responseRegion = normalizeRegionForComparison(response?.region);
        const responseStatus = String(response?.status || '').trim().toLowerCase();
        const hasPrediction = typeof response?.prediction === 'string' && response.prediction.trim().length > 0;
        const hasFindings = Array.isArray(response?.findings) && response.findings.length > 0;
        const imageVerification = response?.image_anatomy_verification || null;

        item.uploadStatus = 'uploaded';
        item.submittedRegion = region;
        item.analysisStatus = responseStatus || 'unknown';
        item.analysisMessage = response?.message || response?.diagnostic_message || null;
        item.imageAnatomyVerificationStatus = String(imageVerification?.status || 'UNVERIFIED').toUpperCase();

        if (imageVerification?.status && String(imageVerification.status).toUpperCase() === 'MISMATCH') {
          regionMismatchDetected = true;
          inlineMismatch = imageVerification.message || `The backend detected that the image anatomy does not match selected region "${region}".`;
          item.status = 'REGION_MISMATCH';
          item.error = inlineMismatch;
        } else if (responseRegion && selectedRegionKey && responseRegion !== selectedRegionKey) {
          regionMismatchDetected = true;
          inlineMismatch = `The backend returned region "${response.region}", which differs from selected region "${region}".`;
          item.status = 'REGION_MISMATCH';
          item.error = inlineMismatch;
        } else if (responseStatus === 'completed' && (hasPrediction || hasFindings)) {
          item.status = 'COMPLETED';
          item.error = null;
          if (!imageVerification || String(imageVerification.status).toUpperCase() !== 'MATCH') {
            inlineRegionNotice = imageVerification?.message || 'The backend processed this image using the selected region, but the image anatomy is not independently verified by the currently configured model.';
          }
        } else {
          item.status = 'ANALYSIS_UNAVAILABLE';
          item.error = item.analysisMessage || 'Image uploaded, but a usable model result was not returned.';
          unavailableImageAnalyses += 1;
          inlineRegionNotice = imageVerification?.message || 'Image anatomy could not be independently verified by the currently configured backend model.';
        }
      } catch (err) {
        uploadFailed = true;
        item.status = 'FAILED';
        item.uploadStatus = 'failed';
        item.error = err?.message || 'X-ray upload or inference request failed.';
      }
      setXrayImages([...updatedImages]);
      if (regionMismatchDetected || uploadFailed) break;
    }

    setXrayImages([...updatedImages]);
    setSubmitting(false);

    if (regionMismatchDetected) {
      setRegionMismatchMessage(inlineMismatch || 'The backend detected a region mismatch. Correct the image, selected region, or report before continuing.');
      return;
    }
    if (uploadFailed) {
      setError('An X-ray image failed to upload or process. Retry the failed item before continuing.');
      return;
    }

    if (xrayReports.length === 0) {
      inlineRegionNotice = inlineRegionNotice || 'No radiology report was supplied. The selected region is used for model routing; the current backend cannot independently verify image anatomy from pixels.';
    }
    if (inlineRegionNotice) setRegionStatusMessage(inlineRegionNotice);

    const notices = [];
    if (unavailableImageAnalyses > 0) notices.push(`${unavailableImageAnalyses} image(s) did not produce a usable model result`);
    if (incompleteReports > 0) notices.push(`${incompleteReports} report(s) had incomplete structured extraction`);
    if (unreadableReports > 0) notices.push(`${unreadableReports} report(s) could not be read`);
    if (inlineRegionNotice) notices.push('image anatomy verification remains unavailable');

    if (notices.length > 0) {
      setInfo(`${notices.join('; ')}. These limitations should remain visible in the final assessment.`);
    } else if (xrayReports.length > 0) {
      setInfo('X-ray processing and the supplied radiology report processing completed. Review the source-specific findings in the assessment results.');
    } else {
      setInfo('X-ray processing completed. Review the available image findings and region-verification limitation in the assessment results.');
    }
    navigate('/assessment/processing');
  };

  return (
    <div className="assessment-workflow-page">
      {/* 3D Radiology Background Elements */}
      <NeuralNetworkCanvas />
      <ThoracicRibcageVisual />
      <RadiologyFilmScreenVisual />
      <FloatingXrayObjects />

      {/* Central Floating 3D Glass Content Panel */}
      <section className="assessment-glass-panel panel-narrow" aria-label="X-Ray & Radiology Evidence Panel">
        {/* Top Badge */}
        <div className="na-top-badge">
          <span className="na-top-badge-pulse" />
          <SparkleIcon size={14} color="#0284c7" />
          <span>Radiological Vision &amp; Report Analysis · Step 2</span>
        </div>

        {/* Title & Description */}
        <div className="na-header-section">
          <h1 className="na-title">
            X-Ray &amp; Radiology <span className="na-title-highlight">Evidence</span>
          </h1>
          <p className="na-description">
            Upload an X-ray image, a radiology report PDF, or both. A PDF is optional. Select a region when uploading images; the backend checks any identifiable report-region conflict.
          </p>
        </div>

        {/* Progress Indicator 1 — 2 — 3 — 4 */}
        <div className="na-progress-bar" aria-label="Assessment Progress">
          <div className="na-progress-step completed">
            <span className="na-progress-number">
              <CheckIcon size={13} color="#ffffff" />
            </span>
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
            <span>Analysis</span>
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
          <div className="info-banner" style={{ width: '100%', marginBottom: '1rem' }}>
            {info}
          </div>
        ) : null}

        {/* 1. Dynamic, Searchable Anatomical Region & X-Ray Type Selector */}
        <XrayRegionSelector
          selectedRegion={region}
          onSelectRegion={(newRegion) => {
            const hasSavedImage = xrayImages.some((item) => item.uploadStatus === 'uploaded');
            const hasRegionBoundReport = xrayReports.some((item) => item.uploadStatus === 'uploaded' && item.reportRegion);
            if (
              (hasSavedImage || hasRegionBoundReport) &&
              normalizeRegionForComparison(newRegion) !== normalizeRegionForComparison(region)
            ) {
              setError('The region cannot be changed after a related image or region-checked report has been saved to this assessment. Start a new assessment to avoid mixing regions.');
              return;
            }
            setRegion(newRegion);
            setRegionMismatchMessage('');
            setRegionStatusMessage('');
            if (error) setError('');
          }}
          required={xrayImages.length > 0}
        />
        {xrayReports.length > 0 && xrayImages.length === 0 ? (
          <p className="upload-hint-text" role="note">
            The full X-Ray workflow requires both a selected anatomical region and an image. Upload radiology report(s) for the same examination/region; reports are extracted before image inference is requested.
          </p>
        ) : null}


        {/* 2. X-RAY IMAGES SECTION */}
        <div className="section-divider-title">
          <h3>X-Ray Images</h3>
          <button
            type="button"
            className="section-upload-btn"
            onClick={() => imageInputRef.current?.click()}
          >
            + Upload X-Ray Image
          </button>
        </div>
        <p className="upload-hint-text" style={{ textAlign: 'left', margin: '0 0 0.5rem 0' }}>
          Optional input: PNG, JPG, JPEG. Select the body region shown. All images in one batch must use the same selected region. The backend may flag a mismatch when it can identify a conflicting report region.
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

        {/* X-Ray Image Upload Dropzone (Always Visible) */}
        <div
          className={`assessment-upload-dropzone ${dragActiveImages ? 'drag-active' : ''}`}
          style={{ padding: '1.25rem 1rem', minHeight: '90px', marginBottom: xrayImages.length > 0 ? '0.75rem' : '1.25rem' }}
          onDragEnter={handleImageDrag}
          onDragLeave={handleImageDrag}
          onDragOver={handleImageDrag}
          onDrop={handleImageDrop}
          onClick={() => imageInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload X-ray images"
          onKeyDown={(e) => {
            if (e.key === ' ' || e.key === 'Enter') {
              e.preventDefault();
              imageInputRef.current?.click();
            }
          }}
        >
          <p className="upload-primary-text" style={{ margin: 0, fontSize: '0.92rem' }}>
            Drag &amp; drop X-ray images, or <span className="upload-browse-link">Browse images</span>
          </p>
        </div>

        {/* Selected Images List */}
        {xrayImages.length > 0 ? (
          <div className="file-list-container" style={{ marginBottom: '1.25rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>
              Selected Images ({xrayImages.length}):
            </span>
            {xrayImages.map((img) => (
              <div key={img.id} className="file-card-item">
                <div className="file-card-left">
                  <div className="file-card-icon-badge badge-img">IMG</div>
                  <div className="file-card-info">
                    <span className="file-card-name" title={img.name}>
                      ✓ {img.name}
                    </span>
                    <span className="file-card-meta">
                      {formatFileSize(img.size)} ·{' '}
                      {(() => {
                        const statusView = getImageStatusPresentation(img);
                        return <span className={statusView.className}>{statusView.label}</span>;
                      })()}
                    </span>
                  </div>
                </div>

                <div className="file-card-actions">
                  <button
                    type="button"
                    className="btn-card-view"
                    onClick={() => handleOpenPreview(img)}
                    aria-label={`View ${img.name}`}
                  >
                    View
                  </button>
                  <button
                    type="button"
                    className="btn-card-remove"
                    onClick={() => handleRemoveImage(img.id)}
                    disabled={submitting || img.uploadStatus === 'uploaded'}
                    title={img.uploadStatus === 'uploaded' ? 'This file is already saved to the server; start a new assessment to replace it.' : 'Remove selected image'}
                    aria-label={`Remove ${img.name}`}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : null}

        {regionMismatchMessage ? (
          <p role="alert" style={{ margin: '-0.25rem 0 1rem', color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '0.55rem 0.7rem', fontSize: '0.82rem' }}>
            {regionMismatchMessage}
          </p>
        ) : filenameRegionMismatches.length > 0 ? (
          <p role="alert" style={{ margin: '-0.25rem 0 1rem', color: '#b91c1c', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '8px', padding: '0.55rem 0.7rem', fontSize: '0.82rem' }}>
            Possible region mismatch: selected region is <strong>{region}</strong>, but <strong>{filenameRegionMismatches[0].item.name}</strong> suggests <strong>{filenameRegionMismatches[0].filenameRegion}</strong>. Change the selected region or upload the correct image. Continue is disabled until this conflict is resolved.
          </p>
        ) : null}
        {xrayImages.length > 0 && xrayReports.length === 0 && !hasRegionMismatch ? (
          <p role="note" style={{ margin: '-0.25rem 0 1rem', color: '#92400e', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '0.55rem 0.7rem', fontSize: '0.82rem' }}>
            Image-only mode is allowed. The selected region is used for model routing, but the current backend does not independently verify the anatomy in the image pixels.
          </p>
        ) : null}
        {regionStatusMessage ? (
          <p role="status" style={{ margin: '-0.25rem 0 1rem', color: '#92400e', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: '8px', padding: '0.55rem 0.7rem', fontSize: '0.82rem' }}>
            {regionStatusMessage}
          </p>
        ) : null}

        {/* 3. OPTIONAL X-RAY / RADIOLOGY REPORTS SECTION */}
        <div className="section-divider-title">
          <h3>X-Ray / Radiology Reports</h3>
          <button
            type="button"
            className="section-upload-btn"
            onClick={() => reportInputRef.current?.click()}
          >
            + Upload X-Ray Report
          </button>
        </div>
        <p className="upload-hint-text" style={{ textAlign: 'left', margin: '0 0 0.5rem 0' }}>
          Optional input: PDF. You can submit a radiology report without an image.
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

        {/* Radiology Report Upload Dropzone (Always Visible) */}
        <div
          className={`assessment-upload-dropzone ${dragActiveReports ? 'drag-active' : ''}`}
          style={{ padding: '1.25rem 1rem', minHeight: '90px', marginBottom: xrayReports.length > 0 ? '0.75rem' : '1.25rem' }}
          onDragEnter={handleReportDrag}
          onDragLeave={handleReportDrag}
          onDragOver={handleReportDrag}
          onDrop={handleReportDrop}
          onClick={() => reportInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload radiology reports"
          onKeyDown={(e) => {
            if (e.key === ' ' || e.key === 'Enter') {
              e.preventDefault();
              reportInputRef.current?.click();
            }
          }}
        >
          <p className="upload-primary-text" style={{ margin: 0, fontSize: '0.92rem' }}>
            Drag &amp; drop radiology reports, or <span className="upload-browse-link">Browse reports</span>
          </p>
        </div>

        {/* Selected Reports List */}
        {xrayReports.length > 0 ? (
          <div className="file-list-container" style={{ marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 600, color: '#475569' }}>
              Selected Reports ({xrayReports.length}):
            </span>
            {xrayReports.map((rep) => {
              const isPdf = rep.name.toLowerCase().endsWith('.pdf');
              return (
                <div key={rep.id} className="file-card-item">
                  <div className="file-card-left">
                    <div className={`file-card-icon-badge ${isPdf ? 'badge-pdf' : 'badge-img'}`}>
                      {isPdf ? 'PDF' : 'DOC'}
                    </div>
                    <div className="file-card-info">
                      <span className="file-card-name" title={rep.name}>
                        ✓ {rep.name}
                      </span>
                      <span className="file-card-meta">
                        {formatFileSize(rep.size)} ·{' '}
                        {(() => {
                          const statusView = getReportStatusPresentation(rep);
                          return <span className={statusView.className}>{statusView.label}</span>;
                        })()}
                      </span>
                    </div>
                  </div>

                <div className="file-card-actions">
                  <button
                    type="button"
                    className="btn-card-view"
                    onClick={() => handleOpenPreview(rep)}
                    aria-label={`View ${rep.name}`}
                  >
                    View
                  </button>
                  <button
                    type="button"
                    className="btn-card-remove"
                    onClick={() => handleRemoveReport(rep.id)}
                    disabled={submitting || rep.uploadStatus === 'uploaded'}
                    title={rep.uploadStatus === 'uploaded' ? 'This report is already saved to the server; start a new assessment to replace it.' : 'Remove selected report'}
                    aria-label={`Remove ${rep.name}`}
                  >
                    Remove
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      ) : null}

        {/* Panel Footer */}
        <div className="na-panel-footer" style={{ marginTop: '1.5rem' }}>
          <button
            type="button"
            className="na-continue-btn"
            onClick={onContinue}
            disabled={submitting || hasRegionMismatch || (xrayImages.length === 0 && xrayReports.length === 0) || (xrayImages.length > 0 && !region)}
          >
            <span>{submitting ? 'Uploading selected inputs and analyzing...' : 'Continue to Review'}</span>
            <span className="na-btn-arrow">
              <ArrowRightIcon size={18} color="#ffffff" />
            </span>
          </button>

          <div className="na-security-footnote">
            <span className="na-security-icon">
              <ShieldLockIcon size={15} color="#10b981" />
            </span>
            <span>Secure Upload · AI-assisted processing · Not a medical diagnosis</span>
          </div>
        </div>
      </section>

      {/* Independent Verification Preview Modal */}
      {previewState ? (
        <FilePreviewModal
          file={previewState.file}
          previewUrl={previewState.previewUrl}
          previewType={previewState.previewType}
          onClose={handleClosePreview}
        />
      ) : null}
    </div>
  );
}

