import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { submitSymptoms, analyzeSymptoms } from '../services/symptomService';
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

function LightbulbIcon({ size = 15, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6" />
      <path d="M10 22h4" />
      <path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5.76.76 1.23 1.52 1.41 2.5" />
    </svg>
  );
}

function AlertShieldIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <line x1="12" y1="8" x2="12" y2="12" />
      <line x1="12" y1="16" x2="12.01" y2="16" />
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
        ctx.fillStyle = `rgba(34, 211, 238, ${n.alpha})`;
        ctx.shadowColor = '#22d3ee';
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
            const lineAlpha = (1 - dist / maxDist) * 0.22;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = `rgba(56, 189, 248, ${lineAlpha})`;
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
   3D ANATOMICAL CARDIAC / ECG VISUAL (LEFT SIDE)
   ============================================================ */
function AnatomicalHeartVisual() {
  return (
    <div className="na-left-backdrop" aria-hidden="true">
      <svg width="260" height="640" viewBox="0 0 260 640" fill="none">
        <defs>
          <linearGradient id="heartGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.75" />
            <stop offset="50%" stopColor="#0ea5e9" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#168bff" stopOpacity="0.5" />
          </linearGradient>
          <linearGradient id="ecgStreamGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.1" />
            <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.1" />
          </linearGradient>
          <filter id="heartGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#22d3ee" floodOpacity="0.4" />
          </filter>
        </defs>

        {/* Vertical streaming ECG rhythm waveform */}
        <path
          d="M 50 40 
             L 50 140 L 40 145 L 65 155 L 30 170 L 75 190 L 50 205 L 50 320 
             L 42 325 L 62 335 L 25 350 L 78 370 L 50 385 L 50 580"
          stroke="url(#ecgStreamGrad)"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* 3D Heart Silhouette with anatomical chambers */}
        <path
          d="M 170 230
             C 140 190, 80 190, 80 260
             C 80 320, 150 380, 180 430
             C 210 380, 280 320, 280 260
             C 280 190, 220 190, 190 230
             Z"
          fill="none"
          stroke="url(#heartGrad)"
          strokeWidth="2.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#heartGlow)"
        />

        {/* Coronary arteries & internal chamber pathways */}
        <path
          d="M 180 240 Q 150 290 170 360"
          stroke="rgba(34, 211, 238, 0.7)"
          strokeWidth="1.8"
          fill="none"
        />
        <path
          d="M 160 300 Q 130 330 140 370"
          stroke="rgba(56, 189, 248, 0.55)"
          strokeWidth="1.5"
          fill="none"
        />
        <path
          d="M 190 290 Q 220 320 210 360"
          stroke="rgba(56, 189, 248, 0.55)"
          strokeWidth="1.5"
          fill="none"
        />

        {/* Pulse nodes */}
        <circle cx="170" cy="230" r="5" fill="#22d3ee" filter="url(#heartGlow)" />
        <circle cx="180" cy="430" r="4" fill="#0ea5e9" filter="url(#heartGlow)" />
        <circle cx="160" cy="300" r="4.5" fill="#ffffff" stroke="#0ea5e9" strokeWidth="1.5" filter="url(#heartGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   HEALTH VITAL SIGNS SILHOUETTE VISUAL (RIGHT SIDE)
   ============================================================ */
function HealthSilhouetteVisual() {
  return (
    <div className="na-right-backdrop" aria-hidden="true">
      <svg width="300" height="640" viewBox="0 0 300 640" fill="none">
        <defs>
          <linearGradient id="vitalGrad" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.8" />
            <stop offset="60%" stopColor="#168bff" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.4" />
          </linearGradient>
          <filter id="vitalGlow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#0ea5e9" floodOpacity="0.4" />
          </filter>
        </defs>

        {/* Torso & Head Clinical Silhouette */}
        <path
          d="M 260 160 
             C 210 160, 200 200, 200 230 
             C 200 260, 220 280, 230 300 
             C 170 320, 140 370, 130 450 
             C 120 530, 110 590, 110 640"
          stroke="url(#vitalGrad)"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
          filter="url(#vitalGlow)"
        />

        {/* Vitals monitor waveform display overlay */}
        <path
          d="M 120 320 H 180 L 190 305 L 200 340 L 210 280 L 220 350 L 230 320 H 290"
          stroke="rgba(34, 211, 238, 0.65)"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        <circle cx="210" cy="280" r="5" fill="#ffffff" stroke="#22d3ee" strokeWidth="2" filter="url(#vitalGlow)" />
        <circle cx="200" cy="230" r="4.5" fill="#38bdf8" filter="url(#vitalGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   FLOATING CARDIAC / SYMPTOMS MEDALLIONS
   ============================================================ */
function FloatingSymptomsObjects() {
  return (
    <div className="na-floating-objects-layer" aria-hidden="true">
      <div className="na-floating-medallion medallion-pos-1" title="Heart Rate">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
          <path d="M3.22 12H9.5l1.5-3 2 6.5 1.5-3.5h6.28" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-2" title="Stethoscope">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M4.5 3v5a4 4 0 0 0 8 0V3" />
          <path d="M8.5 12v3a5 5 0 0 0 10 0v-2" />
          <circle cx="18.5" cy="11" r="2.5" />
          <path d="M3 3h3" />
          <path d="M10 3h3" />
        </svg>
      </div>

      <div className="na-floating-medallion medallion-pos-3" title="Vital Signs">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="10" />
          <polyline points="12 6 12 12 16 14" />
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
   MAIN SYMPTOMS COMPONENT
   ============================================================ */
export default function SymptomsInput() {
  const navigate = useNavigate();
  const { assessmentId, inputTypes } = useAssessment();
  const [text, setText] = useState('');
  const [stage, setStage] = useState('input'); // 'input' | 'followup'
  const [followupQuestions, setFollowupQuestions] = useState([]);
  const [followupAnswers, setFollowupAnswers] = useState({});
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const nextPath = () => {
    if (inputTypes.includes(INPUT_TYPES.MEDICAL_REPORT)) return '/assessment/report';
    if (inputTypes.includes(INPUT_TYPES.XRAY)) return '/assessment/xray';
    return '/assessment/processing';
  };

  const handleInitialSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!assessmentId) return setError('No assessment in progress. Start a new assessment.');
    if (!text.trim()) return setError('Please describe your symptoms.');

    setSubmitting(true);
    try {
      const res = await analyzeSymptoms(assessmentId, text.trim());
      if (res?.followups?.needed && Array.isArray(res?.followups?.questions) && res.followups.questions.length > 0) {
        setFollowupQuestions(res.followups.questions);
        const initialAnswers = {};
        res.followups.questions.forEach((q) => {
          initialAnswers[q.id] = q.options ? q.options[0] : 'No';
        });
        setFollowupAnswers(initialAnswers);
        setStage('followup');
      } else {
        await finalizeAndSubmit(text.trim());
      }
    } catch {
      // In case preview check encounters an issue, proceed directly with submission
      await finalizeAndSubmit(text.trim());
    } finally {
      setSubmitting(false);
    }
  };

  const finalizeAndSubmit = async (finalText) => {
    setSubmitting(true);
    setError('');
    try {
      await submitSymptoms(assessmentId, finalText);
      navigate(nextPath());
    } catch (err) {
      setError(err.message || 'Failed to submit symptoms.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleFollowupConfirm = (e) => {
    e.preventDefault();
    const clarifications = followupQuestions.map((q) => {
      const ans = followupAnswers[q.id] || (q.options ? q.options[0] : 'Not sure');
      return `${q.target}: ${ans}`;
    });

    const combinedText = clarifications.length > 0
      ? `${text.trim()}.\n\nFollow-up responses: ${clarifications.join('. ')}.`
      : text.trim();

    finalizeAndSubmit(combinedText);
  };

  return (
    <div className="assessment-workflow-page">
      {/* 3D Symptoms Background Elements */}
      <NeuralNetworkCanvas />
      <AnatomicalHeartVisual />
      <HealthSilhouetteVisual />
      <FloatingSymptomsObjects />

      {/* Central Floating 3D Glass Content Panel */}
      <section className="assessment-glass-panel panel-narrow" aria-label="Symptoms Assessment Panel">
        {/* Top Clinical Badge */}
        <div className="na-top-badge">
          <span className="na-top-badge-pulse" />
          <SparkleIcon size={14} color="#0284c7" />
          <span>Clinical Symptom Assessment · Step 2</span>
        </div>

        {/* Title & Description */}
        <div className="na-header-section">
          <h1 className="na-title">
            Symptoms <span className="na-title-highlight">Assessment</span>
          </h1>
          <p className="na-description">
            Describe what you are experiencing in your own words.
          </p>
          <p className="na-sub-description">
            Our clinical AI analyzes symptom patterns, severity, potential triggers, and temporal progression to formulate clinical triage insights.
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
            <span>Symptoms</span>
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

        {/* Stage 1: Natural Language Description */}
        {stage === 'input' ? (
          <form onSubmit={handleInitialSubmit} className="symptoms-form-container">
            <div className="symptoms-tip-box">
              <LightbulbIcon size={16} color="#0284c7" />
              <span>
                <strong>Tip:</strong> Mention when symptoms began, what makes them better or worse, and any associated conditions.
              </span>
            </div>

            <div className="symptoms-textarea-wrap">
              <textarea
                className="symptoms-glass-textarea"
                rows={7}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Example: I've had a persistent dry cough and mild fever for 4 days. I also experience slight fatigue in the evenings, but no chest pain..."
                required
                aria-label="Symptom description"
              />
            </div>

            <div className="na-panel-footer">
              <button
                type="submit"
                className="na-continue-btn"
                disabled={submitting}
              >
                <span>{submitting ? 'Analyzing Symptoms...' : 'Continue'}</span>
                <span className="na-btn-arrow">
                  <ArrowRightIcon size={18} color="#ffffff" />
                </span>
              </button>

              <div className="na-security-footnote">
                <span className="na-security-icon">
                  <ShieldLockIcon size={15} color="#10b981" />
                </span>
                <span>Encrypted Clinical NLP &amp; Evidence Processing</span>
              </div>
            </div>
          </form>
        ) : (
          /* Stage 2: Clinical Clarification Follow-up */
          <form onSubmit={handleFollowupConfirm} className="symptoms-form-container">
            <div className="clinical-check-banner">
              <AlertShieldIcon size={20} color="#b45309" />
              <div>
                <strong>Clinical Safety Check:</strong> Please clarify important diagnostic details before our AI finalizes your clinical assessment.
              </div>
            </div>

            <div className="followup-card-group">
              {followupQuestions.map((q, idx) => (
                <div className="followup-question-box" key={q.id}>
                  <h4 className="followup-q-title">
                    {idx + 1}. {q.question}
                  </h4>
                  <div className="choice-row">
                    {(q.options || ['No', 'Yes', 'Not sure']).map((opt) => (
                      <button
                        type="button"
                        key={opt}
                        className={`btn-choice ${(followupAnswers[q.id] || q.options?.[0]) === opt ? 'selected' : ''}`}
                        onClick={() =>
                          setFollowupAnswers((prev) => ({
                            ...prev,
                            [q.id]: opt,
                          }))
                        }
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <div className="na-panel-footer">
              <div style={{ display: 'flex', gap: '0.85rem', flexWrap: 'wrap' }}>
                <button
                  type="button"
                  className="na-secondary-btn"
                  onClick={() => setStage('input')}
                >
                  ← Back to Description
                </button>

                <button
                  type="submit"
                  className="na-continue-btn"
                  disabled={submitting}
                >
                  <span>{submitting ? 'Processing Clinical Data...' : 'Confirm & Continue'}</span>
                  <span className="na-btn-arrow">
                    <ArrowRightIcon size={18} color="#ffffff" />
                  </span>
                </button>
              </div>

              <div className="na-security-footnote">
                <span className="na-security-icon">
                  <ShieldLockIcon size={15} color="#10b981" />
                </span>
                <span>Protected Health Data Processing</span>
              </div>
            </div>
          </form>
        )}
      </section>
    </div>
  );
}
