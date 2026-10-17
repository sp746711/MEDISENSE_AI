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
    setError('');
    const newItems = [];
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      const ext = '.' + f.name.split('.').pop().toLowerCase();
      const isAllowed = ALLOWED_IMAGE_EXTS.includes(ext) || ALLOWED_IMAGE_TYPES.includes(f.type);
      if (!isAllowed) {
        setError('Please select a PNG, JPG, or JPEG image.');
        return;
      }
      const exists = xrayImages.some(
        (existing) => existing.file.name === f.name && existing.file.size === f.size
      );
      if (!exists) {
        newItems.push({
          id: `img-${f.name}-${f.size}-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          file: f,
          name: f.name,
          size: f.size,
          type: f.type,
          status: 'READY',
          error: null,
        });
      }
    }
    if (newItems.length > 0) {
      setXrayImages((prev) => [...prev, ...newItems]);
      setError('');
    }
  };

  const addReports = (fileList) => {
    if (!fileList || fileList.length === 0) return;
    setError('');
    const newItems = [];
    for (let i = 0; i < fileList.length; i++) {
      const f = fileList[i];
      const ext = '.' + f.name.split('.').pop().toLowerCase();
      if (ext !== '.pdf' && f.type !== 'application/pdf') {
        setError('Please select a PDF radiology report.');
        return;
      }
      const fnameLower = f.name.toLowerCase();
      const isMedicalReport = ['medical_report', 'medical-report', 'blood_report', 'lab_report', 'cbc', 'liver_report', 'kidney_report'].some(
        (m) => fnameLower.includes(m)
      );
      if (isMedicalReport) {
        setError(`This file appears to be a general medical report (${f.name}). Please upload it in the Medical Reports section, not X-Ray / Radiology Reports.`);
        return;
      }
      const exists = xrayReports.some(
        (existing) => existing.file.name === f.name && existing.file.size === f.size
      );
      if (!exists) {
        newItems.push({
          id: `rep-${f.name}-${f.size}-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
          file: f,
          name: f.name,
          size: f.size,
          type: f.type,
          status: 'READY',
          error: null,
        });
      }
    }
    if (newItems.length > 0) {
      setXrayReports((prev) => [...prev, ...newItems]);
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
    if (!assessmentId) return setError('No assessment in progress. Start a new assessment.');

    const totalFiles = xrayImages.length + xrayReports.length;
    if (totalFiles === 0) {
      return setError('Please upload at least one X-ray image or X-ray report before continuing.');
    }

    if (xrayImages.length > 0 && (!region || !region.trim())) {
      return setError('Please select an X-ray body region/type.');
    }

    setSubmitting(true);
    let anyFailed = false;

    // 1. Process Images
    const updatedImages = [...xrayImages];
    for (let i = 0; i < updatedImages.length; i++) {
      const item = updatedImages[i];
      if (item.status === 'COMPLETED') continue;

      item.status = 'UPLOADING';
      setXrayImages([...updatedImages]);

      try {
        await uploadXray(assessmentId, item.file, region);
        item.status = 'COMPLETED';
        item.error = null;
      } catch (err) {
        anyFailed = true;
        item.status = 'FAILED';
        item.error = err.message || 'Image upload failed';
      }
      setXrayImages([...updatedImages]);
    }

    // 2. Process X-Ray / Radiology Reports
    const updatedReports = [...xrayReports];
    for (let i = 0; i < updatedReports.length; i++) {
      const item = updatedReports[i];
      if (item.status === 'COMPLETED') continue;

      item.status = 'UPLOADING';
      setXrayReports([...updatedReports]);

      try {
        await uploadReport(assessmentId, item.file, 'xray_report');
        item.status = 'COMPLETED';
        item.error = null;
      } catch (err) {
        anyFailed = true;
        item.status = 'FAILED';
        item.error = err.message || 'Radiology report upload failed';
      }
      setXrayReports([...updatedReports]);
    }

    setSubmitting(false);

    if (anyFailed) {
      setError('Some X-ray files failed to upload. You can retry failed files or remove them.');
    } else {
      setInfo('All X-ray evidence uploaded successfully.');
      navigate('/assessment/processing');
    }
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
            Upload X-ray images, radiology reports, or both. You can preview every file to verify it before proceeding.
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
            setRegion(newRegion);
            if (error) setError('');
          }}
          required={xrayImages.length > 0}
        />


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
          Accepted: PNG, JPG, JPEG
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
                      {img.status === 'READY' ? (
                        <span className="status-ready">Ready for upload</span>
                      ) : img.status === 'UPLOADING' ? (
                        <span className="status-uploading">Uploading &amp; Analyzing...</span>
                      ) : img.status === 'COMPLETED' ? (
                        <span className="status-completed">✓ Uploaded</span>
                      ) : (
                        <span className="status-failed">✗ {img.error || 'Upload failed'}</span>
                      )}
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
                    disabled={submitting}
                    aria-label={`Remove ${img.name}`}
                  >
                    Remove
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : null}

        {/* 3. X-RAY / RADIOLOGY REPORTS SECTION */}
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
          Accepted: PDF
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
                        {rep.status === 'READY' ? (
                          <span className="status-ready">Ready for OCR/NLP</span>
                        ) : rep.status === 'UPLOADING' ? (
                          <span className="status-uploading">Extracting text &amp; findings...</span>
                        ) : rep.status === 'COMPLETED' ? (
                          <span className="status-completed">✓ Extracted</span>
                        ) : (
                          <span className="status-failed">✗ {rep.error || 'Upload failed'}</span>
                        )}
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
                    disabled={submitting}
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
            disabled={submitting || (xrayImages.length === 0 && xrayReports.length === 0)}
          >
            <span>{submitting ? 'Uploading & Analyzing...' : 'Continue'}</span>
            <span className="na-btn-arrow">
              <ArrowRightIcon size={18} color="#ffffff" />
            </span>
          </button>

          <div className="na-security-footnote">
            <span className="na-security-icon">
              <ShieldLockIcon size={15} color="#10b981" />
            </span>
            <span>Independent Verification · HIPAA-Compliant Vision AI &amp; OCR</span>
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

