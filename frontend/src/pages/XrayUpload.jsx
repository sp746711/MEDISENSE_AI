import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { uploadXray } from '../services/xrayService';
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

function RadiologyUploadIcon({ size = 34, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="4" />
      <path d="M7 9a3 3 0 0 1 3 3v3a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1" />
      <path d="M17 9a3 3 0 0 0-3 3v3a1 1 0 0 0 2 0v-3a1 1 0 0 1 1-1" />
      <line x1="12" y1="7" x2="12" y2="17" strokeWidth="2" />
      <path d="M12 12h.01" />
    </svg>
  );
}

function FileCheckBadgeIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="m9 15 2 2 4-4" />
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

const REGION_OPTIONS = [
  { value: '', label: '-- Select Body Region / Type --' },
  { value: 'chest', label: 'Chest' },
  { value: 'wrist', label: 'Wrist' },
  { value: 'hand', label: 'Hand' },
  { value: 'forearm', label: 'Forearm' },
  { value: 'elbow', label: 'Elbow' },
  { value: 'shoulder', label: 'Shoulder' },
  { value: 'humerus', label: 'Humerus' },
  { value: 'clavicle', label: 'Clavicle' },
  { value: 'knee', label: 'Knee' },
  { value: 'lower leg', label: 'Lower Leg' },
  { value: 'ankle', label: 'Ankle' },
  { value: 'foot', label: 'Foot' },
  { value: 'pelvis', label: 'Pelvis' },
  { value: 'hip', label: 'Hip' },
  { value: 'femur', label: 'Femur' },
  { value: 'spine', label: 'Spine' },
];

/* ============================================================
   MAIN X-RAY UPLOAD COMPONENT
   ============================================================ */
export default function XrayUpload() {
  const navigate = useNavigate();
  const { assessmentId } = useAssessment();
  const [file, setFile] = useState(null);
  const [region, setRegion] = useState('');
  const [info, setInfo] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
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
    if (!region) return setError('Please select the body region for this X-ray.');
    if (!file) return setError('Please select an X-ray image.');
    setSubmitting(true);
    try {
      const result = await uploadXray(assessmentId, file, region);
      setInfo(result.message || 'X-ray uploaded successfully.');
      navigate('/assessment/processing');
    } catch (err) {
      setError(err.message || 'Failed to upload X-ray.');
    } finally {
      setSubmitting(false);
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
      <section className="assessment-glass-panel panel-narrow" aria-label="X-Ray Image Upload Panel">
        {/* Top Badge */}
        <div className="na-top-badge">
          <span className="na-top-badge-pulse" />
          <SparkleIcon size={14} color="#0284c7" />
          <span>Radiological Vision AI · Step 2</span>
        </div>

        {/* Title & Description */}
        <div className="na-header-section">
          <h1 className="na-title">
            X-Ray <span className="na-title-highlight">Upload</span>
          </h1>
          <p className="na-description">
            Upload JPG, JPEG, or PNG radiograph images.
          </p>
          <p className="na-sub-description">
            Supports chest radiographs and musculoskeletal regions. Unsupported body regions return an unavailable status — never a fabricated finding.
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

        {/* Required X-Ray Body Region Selector */}
        <div className="xray-region-selection-box" style={{ width: '100%', marginBottom: '1.25rem' }}>
          <label htmlFor="xray-region-select" style={{ display: 'block', fontSize: '0.88rem', fontWeight: 600, color: '#e2e8f0', marginBottom: '0.45rem' }}>
            Select Body Region / X-Ray Type <span style={{ color: '#ef4444' }}>*</span>
          </label>
          <select
            id="xray-region-select"
            value={region}
            onChange={(e) => setRegion(e.target.value)}
            required
            style={{
              width: '100%',
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(15, 23, 42, 0.85)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              borderRadius: '10px',
              color: '#f8fafc',
              fontSize: '0.95rem',
              outline: 'none',
              cursor: 'pointer',
            }}
          >
            {REGION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value} style={{ backgroundColor: '#0f172a', color: '#f8fafc' }}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Drag & Drop Glass Radiology Upload Zone */}
        <div
          className={`assessment-upload-dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload X-ray image"
          onKeyDown={(e) => {
            if (e.key === ' ' || e.key === 'Enter') {
              e.preventDefault();
              fileInputRef.current?.click();
            }
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png"
            className="hidden-file-input"
            onChange={handleFileInputChange}
            aria-hidden="true"
          />

          <div className="upload-icon-circle upload-icon-xray">
            <RadiologyUploadIcon size={34} color="#ffffff" />
          </div>

          <p className="upload-primary-text">
            Drag &amp; drop your X-ray image here, or <span className="upload-browse-link">Browse files</span>
          </p>
          <p className="upload-hint-text">
            Supports standard JPG, JPEG, and PNG radiograph scans. Images are processed with deep convolutional vision models.
          </p>
        </div>

        {/* Selected File State Display with Preview Thumbnail */}
        {file ? (
          <div className="file-selected-card">
            <div className="file-selected-left">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="X-ray preview"
                  style={{
                    width: '46px',
                    height: '46px',
                    borderRadius: '10px',
                    objectFit: 'cover',
                    border: '1px solid #10b981',
                  }}
                />
              ) : (
                <div className="file-badge-icon">
                  <FileCheckBadgeIcon size={22} color="#10b981" />
                </div>
              )}
              <div className="file-info-text">
                <span className="file-name-label">{file.name}</span>
                <span className="file-meta-label">
                  {formatFileSize(file.size)} · Ready for neural vision inference
                </span>
              </div>
            </div>
            <button
              type="button"
              className="file-change-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              Change image
            </button>
          </div>
        ) : null}

        {/* Panel Footer */}
        <div className="na-panel-footer" style={{ marginTop: '1.5rem' }}>
          <button
            type="button"
            className="na-continue-btn"
            onClick={onContinue}
            disabled={submitting || !file}
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
            <span>HIPAA-Compliant Radiology Inference &amp; Verification</span>
          </div>
        </div>
      </section>
    </div>
  );
}
