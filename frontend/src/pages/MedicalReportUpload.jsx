import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { uploadReport } from '../services/reportService';
import { useAssessment } from '../hooks/useAssessment';
import { INPUT_TYPES } from '../utils/constants';
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

function DocumentUploadIcon({ size = 32, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M12 18v-6" />
      <path d="m9 15 3-3 3 3" />
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
   CANVAS NEURAL NETWORK BACKGROUND
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
        ctx.fillStyle = `rgba(168, 85, 247, ${n.alpha * 0.8})`;
        ctx.shadowColor = '#a855f7';
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
            ctx.strokeStyle = `rgba(168, 85, 247, ${lineAlpha})`;
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
   3D LABORATORY FLASK & BIOMARKER STREAM (LEFT SIDE)
   ============================================================ */
function LabBiomarkerVisual() {
  return (
    <div className="na-left-backdrop" aria-hidden="true">
      <svg width="260" height="640" viewBox="0 0 260 640" fill="none">
        <defs>
          <linearGradient id="flaskGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#c084fc" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.5" />
          </linearGradient>
          <filter id="flaskGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#7c3aed" floodOpacity="0.38" />
          </filter>
        </defs>

        {/* Laboratory Flask outline */}
        <path
          d="M 120 180 L 120 250 L 60 380 Q 50 410 80 410 L 200 410 Q 230 410 220 380 L 160 250 L 160 180 Z"
          fill="none"
          stroke="url(#flaskGrad)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#flaskGlow)"
        />

        {/* Liquid level */}
        <path
          d="M 75 350 Q 140 370 205 350"
          stroke="rgba(168, 85, 247, 0.7)"
          strokeWidth="2"
          fill="none"
        />

        {/* Rising molecular bubbles */}
        <circle cx="110" cy="360" r="4.5" fill="#c084fc" filter="url(#flaskGlow)" />
        <circle cx="150" cy="330" r="6" fill="#38bdf8" filter="url(#flaskGlow)" />
        <circle cx="130" cy="290" r="4" fill="#a855f7" filter="url(#flaskGlow)" />
        <circle cx="140" cy="220" r="3.5" fill="#22d3ee" filter="url(#flaskGlow)" />
        <circle cx="140" cy="150" r="5" fill="#c084fc" filter="url(#flaskGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   DIAGNOSTIC REPORT ANALYTICS VISUAL (RIGHT SIDE)
   ============================================================ */
function DiagnosticDocVisual() {
  return (
    <div className="na-right-backdrop" aria-hidden="true">
      <svg width="300" height="640" viewBox="0 0 300 640" fill="none">
        <defs>
          <linearGradient id="docGrad" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#a855f7" stopOpacity="0.8" />
            <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.65" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.4" />
          </linearGradient>
          <filter id="docGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#a855f7" floodOpacity="0.35" />
          </filter>
        </defs>

        {/* Diagnostic Document Outline */}
        <path
          d="M 120 180 H 250 V 440 H 120 Z"
          rx="12"
          stroke="url(#docGrad)"
          strokeWidth="2.5"
          fill="none"
          filter="url(#docGlow)"
        />

        {/* Document Header Line */}
        <line x1="140" y1="215" x2="200" y2="215" stroke="rgba(168, 85, 247, 0.7)" strokeWidth="3" strokeLinecap="round" />

        {/* Analytical Bars */}
        <rect x="140" y="245" width="85" height="7" rx="3.5" fill="rgba(168, 85, 247, 0.55)" />
        <rect x="140" y="265" width="60" height="7" rx="3.5" fill="rgba(56, 189, 248, 0.6)" />
        <rect x="140" y="285" width="95" height="7" rx="3.5" fill="rgba(168, 85, 247, 0.75)" />

        {/* Chart Wave in lower section */}
        <path
          d="M 140 370 L 165 350 L 190 380 L 215 330 L 235 350"
          stroke="#38bdf8"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          filter="url(#docGlow)"
        />
        <circle cx="215" cy="330" r="4.5" fill="#ffffff" stroke="#7c3aed" strokeWidth="2" filter="url(#docGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   FLOATING LAB / REPORT MEDALLIONS
   ============================================================ */
function FloatingReportObjects() {
  return (
    <div className="na-floating-objects-layer" aria-hidden="true">
      <div className="na-floating-medallion medallion-pos-1" title="Report Analysis">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-2" title="Laboratory Panel">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2" />
          <path d="M8.5 2h7" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-3" title="Biomarker Metric">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M18 20V10" />
          <path d="M12 20V4" />
          <path d="M6 20v-6" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-4" title="Verified AI OCR">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      </div>
    </div>
  );
}

/* ============================================================
   MAIN MEDICAL REPORT UPLOAD COMPONENT
   ============================================================ */
export default function MedicalReportUpload() {
  const navigate = useNavigate();
  const { assessmentId, inputTypes } = useAssessment();
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const nextPath = () => {
    if (inputTypes.includes(INPUT_TYPES.XRAY)) return '/assessment/xray';
    return '/assessment/processing';
  };

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
    if (!file) return setError('Please select a medical report file.');
    setSubmitting(true);
    try {
      const result = await uploadReport(assessmentId, file);
      setMessage(result.message || 'Report uploaded successfully.');
      navigate(nextPath());
    } catch (err) {
      setError(err.message || 'Failed to upload report.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="assessment-workflow-page">
      {/* 3D Laboratory Report Background Elements */}
      <NeuralNetworkCanvas />
      <LabBiomarkerVisual />
      <DiagnosticDocVisual />
      <FloatingReportObjects />

      {/* Central Floating 3D Glass Content Panel */}
      <section className="assessment-glass-panel panel-narrow" aria-label="Medical Report Upload Panel">
        {/* Top Badge */}
        <div className="na-top-badge">
          <span className="na-top-badge-pulse" />
          <SparkleIcon size={14} color="#0284c7" />
          <span>Diagnostic Document Analysis · Step 2</span>
        </div>

        {/* Title & Description */}
        <div className="na-header-section">
          <h1 className="na-title">
            Medical <span className="na-title-highlight">Report</span>
          </h1>
          <p className="na-description">
            Upload PDF, JPG, JPEG, or PNG health reports.
          </p>
          <p className="na-sub-description">
            Supports laboratory blood work, metabolic panels, pathology notes, and radiology summaries. Our OCR extracts biomarker parameters and reference ranges.
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
            <span>Report</span>
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

        {message ? (
          <div className="info-banner" style={{ width: '100%', marginBottom: '1rem' }}>
            {message}
          </div>
        ) : null}

        {/* Drag & Drop Glass Upload Area */}
        <div
          className={`assessment-upload-dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          aria-label="Upload medical report"
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
            accept=".pdf,.jpg,.jpeg,.png"
            className="hidden-file-input"
            onChange={handleFileInputChange}
            aria-hidden="true"
          />

          <div className="upload-icon-circle upload-icon-report">
            <DocumentUploadIcon size={32} color="#ffffff" />
          </div>

          <p className="upload-primary-text">
            Drag &amp; drop your medical report here, or <span className="upload-browse-link">Browse files</span>
          </p>
          <p className="upload-hint-text">
            Supports PDF, JPG, JPEG, and PNG formats (Max size configured by server). Executables are automatically rejected.
          </p>
        </div>

        {/* Selected File State Display */}
        {file ? (
          <div className="file-selected-card">
            <div className="file-selected-left">
              <div className="file-badge-icon">
                <FileCheckBadgeIcon size={22} color="#10b981" />
              </div>
              <div className="file-info-text">
                <span className="file-name-label">{file.name}</span>
                <span className="file-meta-label">
                  {formatFileSize(file.size)} · Ready for extraction
                </span>
              </div>
            </div>
            <button
              type="button"
              className="file-change-btn"
              onClick={() => fileInputRef.current?.click()}
            >
              Change file
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
            <span>{submitting ? 'Uploading & Extracting...' : 'Continue'}</span>
            <span className="na-btn-arrow">
              <ArrowRightIcon size={18} color="#ffffff" />
            </span>
          </button>

          <div className="na-security-footnote">
            <span className="na-security-icon">
              <ShieldLockIcon size={15} color="#10b981" />
            </span>
            <span>HIPAA-Compliant Document Encryption &amp; OCR</span>
          </div>
        </div>
      </section>
    </div>
  );
}
