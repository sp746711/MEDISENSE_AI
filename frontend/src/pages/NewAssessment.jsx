import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { createAssessment } from '../services/assessmentService';
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
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round">
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

// Stethoscope / Clinical Activity Icon for Symptoms
function StethoscopePulseIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 3v5a4 4 0 0 0 8 0V3" />
      <path d="M8.5 12v3a5 5 0 0 0 10 0v-2" />
      <circle cx="18.5" cy="11" r="2.5" />
      <path d="M3 3h3" />
      <path d="M10 3h3" />
    </svg>
  );
}

// Medical Report Document Icon
function MedicalReportDocIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}

// Radiology / Thoracic Chest X-Ray Icon
function RadiologyIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
      <path d="M7 9a3 3 0 0 1 3 3v3a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1" />
      <path d="M17 9a3 3 0 0 0-3 3v3a1 1 0 0 0 2 0v-3a1 1 0 0 1 1-1" />
      <line x1="12" y1="7" x2="12" y2="17" strokeWidth="2.5" />
    </svg>
  );
}

/* ============================================================
   DECORATIVE MEDICAL BOTTOM GRAPHICS FOR CARDS
   ============================================================ */

// Symptoms: Dynamic ECG Waveform with glow
function EcgWaveVisual() {
  return (
    <svg width="100%" height="32" viewBox="0 0 240 32" fill="none" preserveAspectRatio="none">
      <defs>
        <linearGradient id="ecgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" />
          <stop offset="35%" stopColor="#0ea5e9" stopOpacity="0.75" />
          <stop offset="50%" stopColor="#22d3ee" stopOpacity="1" />
          <stop offset="65%" stopColor="#0ea5e9" stopOpacity="0.75" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.2" />
        </linearGradient>
      </defs>
      <path
        d="M0 16 H65 L73 16 L79 9 L85 24 L92 3 L98 28 L104 16 L112 16 L117 11 L122 16 H240"
        stroke="url(#ecgGrad)"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// Medical Report: Subtle document / lab report indicators with soft violet wave
function ReportBarsVisual() {
  return (
    <svg width="100%" height="32" viewBox="0 0 240 32" fill="none" preserveAspectRatio="none">
      <defs>
        <linearGradient id="reportGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#a855f7" stopOpacity="0.2" />
          <stop offset="50%" stopColor="#7c3aed" stopOpacity="0.85" />
          <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.2" />
        </linearGradient>
      </defs>
      {/* Translucent report wave lines */}
      <path
        d="M10 22 C 50 14, 90 28, 130 18 C 170 8, 200 24, 230 16"
        stroke="url(#reportGrad)"
        strokeWidth="1.5"
        strokeDasharray="4 3"
        fill="none"
      />
      <rect x="25" y="10" width="38" height="6" rx="3" fill="url(#reportGrad)" opacity="0.6" />
      <rect x="72" y="8" width="50" height="8" rx="4" fill="url(#reportGrad)" opacity="0.9" />
      <rect x="132" y="10" width="45" height="6" rx="3" fill="url(#reportGrad)" opacity="0.6" />
      <rect x="186" y="9" width="30" height="7" rx="3.5" fill="url(#reportGrad)" opacity="0.8" />
    </svg>
  );
}

// X-Ray: Subtle Thoracic scan / ribcage curves
function XrayChestVisual() {
  return (
    <svg width="100%" height="32" viewBox="0 0 240 32" fill="none" preserveAspectRatio="none">
      <defs>
        <linearGradient id="xrayGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#10b981" stopOpacity="0.15" />
          <stop offset="40%" stopColor="#06b6d4" stopOpacity="0.75" />
          <stop offset="60%" stopColor="#10b981" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.15" />
        </linearGradient>
      </defs>
      <path d="M120 4 V28" stroke="url(#xrayGrad)" strokeWidth="2.2" strokeDasharray="3 2" />
      <path d="M85 8 C102 10 115 12 119 12" stroke="url(#xrayGrad)" strokeWidth="2" strokeLinecap="round" />
      <path d="M155 8 C138 10 125 12 121 12" stroke="url(#xrayGrad)" strokeWidth="2" strokeLinecap="round" />
      <path d="M78 16 C98 18 114 20 119 20" stroke="url(#xrayGrad)" strokeWidth="2" strokeLinecap="round" />
      <path d="M162 16 C142 18 126 20 121 20" stroke="url(#xrayGrad)" strokeWidth="2" strokeLinecap="round" />
      <path d="M82 24 C100 25 114 26 119 26" stroke="url(#xrayGrad)" strokeWidth="1.8" strokeLinecap="round" />
      <path d="M158 24 C140 25 126 26 121 26" stroke="url(#xrayGrad)" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

/* ============================================================
   CANVAS NEURAL NETWORK BACKGROUND (3D DEPTH + SYNAPTIC PULSES)
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

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Create 3D nodes with depth (z: 0.25 - 1.0)
    const nodeCount = 38;
    const nodes = [];
    for (let i = 0; i < nodeCount; i++) {
      const z = 0.25 + Math.random() * 0.75;
      nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        z,
        vx: (Math.random() - 0.5) * 0.28 * z,
        vy: (Math.random() - 0.5) * 0.28 * z,
        baseRadius: 1.4 + z * 2.2,
        alpha: 0.2 + z * 0.45,
        isHub: i % 8 === 0, // Certain nodes act as major synaptic hubs
        pulsePhase: Math.random() * Math.PI * 2,
      });
    }

    // Synaptic signal pulses that travel along connections
    const signals = [];
    const maxSignals = 6;

    const spawnSignal = () => {
      if (signals.length >= maxSignals || nodes.length < 2) return;
      const i = Math.floor(Math.random() * nodes.length);
      // Find a close neighbor
      for (let j = 0; j < nodes.length; j++) {
        if (i === j) continue;
        const dx = nodes[i].x - nodes[j].x;
        const dy = nodes[i].y - nodes[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 145 && Math.abs(nodes[i].z - nodes[j].z) < 0.45) {
          signals.push({
            from: nodes[i],
            to: nodes[j],
            progress: 0,
            speed: 0.005 + Math.random() * 0.007,
            color: Math.random() > 0.3 ? '#22d3ee' : '#a855f7',
          });
          break;
        }
      }
    };

    let frameCount = 0;

    const render = () => {
      frameCount++;
      ctx.clearRect(0, 0, width, height);

      // Spawn occasional synaptic pulses
      if (frameCount % 45 === 0 && !prefersReducedMotion) {
        spawnSignal();
      }

      // Update positions
      if (!prefersReducedMotion) {
        for (let i = 0; i < nodeCount; i++) {
          const n = nodes[i];
          n.x += n.vx;
          n.y += n.vy;
          n.pulsePhase += 0.02;

          if (n.x < -30) n.x = width + 30;
          if (n.x > width + 30) n.x = -30;
          if (n.y < -30) n.y = height + 30;
          if (n.y > height + 30) n.y = -30;
        }
      }

      // Draw neural connections
      const maxDist = 145;
      for (let i = 0; i < nodeCount; i++) {
        for (let j = i + 1; j < nodeCount; j++) {
          const n1 = nodes[i];
          const n2 = nodes[j];
          const zDiff = Math.abs(n1.z - n2.z);
          if (zDiff > 0.4) continue; // Only connect nodes with similar 3D depth

          const dx = n1.x - n2.x;
          const dy = n1.y - n2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDist) {
            const zAvg = (n1.z + n2.z) * 0.5;
            const lineAlpha = (1 - dist / maxDist) * 0.22 * zAvg;

            ctx.beginPath();
            ctx.moveTo(n1.x, n1.y);
            ctx.lineTo(n2.x, n2.y);
            ctx.strokeStyle = `rgba(56, 189, 248, ${lineAlpha})`;
            ctx.lineWidth = 0.8 + zAvg * 0.8;
            ctx.stroke();
          }
        }
      }

      // Draw traveling synaptic signals
      for (let s = signals.length - 1; s >= 0; s--) {
        const sig = signals[s];
        if (!prefersReducedMotion) {
          sig.progress += sig.speed;
        }
        if (sig.progress >= 1) {
          signals.splice(s, 1);
          continue;
        }

        const currX = sig.from.x + (sig.to.x - sig.from.x) * sig.progress;
        const currY = sig.from.y + (sig.to.y - sig.from.y) * sig.progress;

        ctx.beginPath();
        ctx.arc(currX, currY, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = sig.color;
        ctx.shadowColor = sig.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Draw 3D nodes
      for (let i = 0; i < nodeCount; i++) {
        const n = nodes[i];
        const pulse = n.isHub ? Math.sin(n.pulsePhase) * 0.8 : 0;
        const r = Math.max(1, n.baseRadius + pulse);

        // Halo for hub nodes
        if (n.isHub) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, r * 2.4, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(34, 211, 238, ${0.08 * n.z})`;
          ctx.fill();
        }

        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fillStyle = n.isHub
          ? `rgba(255, 255, 255, ${0.9 * n.z})`
          : `rgba(34, 211, 238, ${n.alpha})`;
        ctx.shadowColor = '#0ea5e9';
        ctx.shadowBlur = n.isHub ? 10 : 5;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      if (!prefersReducedMotion) {
        animId = requestAnimationFrame(render);
      }
    };

    render();

    return () => {
      if (animId) cancelAnimationFrame(animId);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return <canvas ref={canvasRef} className="na-neural-canvas" />;
}

/* ============================================================
   3D DNA DOUBLE HELIX VISUAL (LEFT SIDE)
   ============================================================ */
function DnaHelixVisual() {
  const rungs = [];
  const count = 18;
  for (let i = 0; i < count; i++) {
    const y = 35 + i * 34;
    const angle = (i / count) * Math.PI * 3.6;
    const xOffset = Math.sin(angle) * 48;
    const zOffset = Math.cos(angle);
    const x1 = 125 - xOffset;
    const x2 = 125 + xOffset;
    const opacity = 0.4 + (zOffset + 1) * 0.28;
    const rungWidth = 1.6 + (zOffset + 1) * 0.6;
    rungs.push({ i, y, x1, x2, opacity, rungWidth, zOffset });
  }

  return (
    <div className="na-left-backdrop" aria-hidden="true">
      <svg width="250" height="660" viewBox="0 0 250 660" fill="none">
        <defs>
          <linearGradient id="dnaStrand1" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.35" />
            <stop offset="35%" stopColor="#0ea5e9" stopOpacity="0.85" />
            <stop offset="70%" stopColor="#38bdf8" stopOpacity="0.9" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.4" />
          </linearGradient>
          <linearGradient id="dnaStrand2" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#168bff" stopOpacity="0.4" />
            <stop offset="40%" stopColor="#22d3ee" stopOpacity="0.9" />
            <stop offset="75%" stopColor="#0ea5e9" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#168bff" stopOpacity="0.35" />
          </linearGradient>
          <linearGradient id="dnaRungGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#38bdf8" />
            <stop offset="50%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#0ea5e9" />
          </linearGradient>
          <filter id="dnaGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feDropShadow dx="0" dy="0" stdDeviation="6" floodColor="#22d3ee" floodOpacity="0.5" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {rungs.map((r) => (
          <g key={r.i} opacity={r.opacity}>
            <line
              x1={r.x1}
              y1={r.y}
              x2={r.x2}
              y2={r.y}
              stroke="url(#dnaRungGrad)"
              strokeWidth={r.rungWidth}
              strokeLinecap="round"
            />
            <circle
              cx={r.x1}
              cy={r.y}
              r={3.2 + (r.zOffset + 1) * 0.8}
              fill="#22d3ee"
              filter="url(#dnaGlow)"
            />
            <circle
              cx={r.x2}
              cy={r.y}
              r={3.2 + (-r.zOffset + 1) * 0.8}
              fill="#0ea5e9"
              filter="url(#dnaGlow)"
            />
          </g>
        ))}

        <path
          d={'M ' + rungs.map((r, idx) => `${idx === 0 ? '' : 'L '}${r.x1.toFixed(1)} ${r.y}`).join(' ')}
          fill="none"
          stroke="url(#dnaStrand1)"
          strokeWidth="3.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#dnaGlow)"
        />

        <path
          d={'M ' + rungs.map((r, idx) => `${idx === 0 ? '' : 'L '}${r.x2.toFixed(1)} ${r.y}`).join(' ')}
          fill="none"
          stroke="url(#dnaStrand2)"
          strokeWidth="3.2"
          strokeLinecap="round"
          strokeLinejoin="round"
          filter="url(#dnaGlow)"
        />
      </svg>
    </div>
  );
}

/* ============================================================
   AI BRAIN / CRANIAL PROFILE VISUAL (RIGHT SIDE)
   ============================================================ */
function AiBrainVisual() {
  return (
    <div className="na-right-backdrop" aria-hidden="true">
      <svg width="420" height="640" viewBox="0 0 420 640" fill="none">
        <defs>
          <linearGradient id="brainContourGrad" x1="100%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.85" />
            <stop offset="45%" stopColor="#0ea5e9" stopOpacity="0.75" />
            <stop offset="80%" stopColor="#7c3aed" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#ec4899" stopOpacity="0.4" />
          </linearGradient>
          <filter id="brainGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feDropShadow dx="0" dy="0" stdDeviation="8" floodColor="#0ea5e9" floodOpacity="0.5" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Outer radial neural field waves */}
        <circle cx="210" cy="290" r="160" stroke="rgba(34, 211, 238, 0.12)" strokeWidth="1" strokeDasharray="6 4" />
        <circle cx="210" cy="290" r="210" stroke="rgba(22, 139, 255, 0.08)" strokeWidth="1" strokeDasharray="8 6" />
        <circle cx="210" cy="290" r="260" stroke="rgba(124, 58, 237, 0.05)" strokeWidth="1" strokeDasharray="10 8" />

        {/* Anatomical Human Cranium & Brain Outer Profile */}
        <path
          d="M 370 160 
             C 280 120, 170 145, 120 220 
             C 95 260, 95 305, 115 350 
             C 130 385, 160 410, 190 455 
             C 215 495, 230 550, 310 600"
          stroke="url(#brainContourGrad)"
          strokeWidth="2.8"
          strokeLinecap="round"
          fill="none"
          filter="url(#brainGlow)"
        />

        {/* Frontal, Parietal, Occipital & Temporal Cortical Sulci (folds) */}
        <path d="M 155 220 C 190 195, 255 200, 295 235 C 270 270, 215 255, 170 275" stroke="rgba(56, 189, 248, 0.65)" strokeWidth="1.8" fill="none" />
        <path d="M 140 280 C 180 265, 235 285, 270 320 C 245 350, 195 335, 160 355" stroke="rgba(56, 189, 248, 0.6)" strokeWidth="1.8" fill="none" />
        <path d="M 160 360 C 205 350, 260 375, 285 415 C 250 440, 200 420, 175 445" stroke="rgba(56, 189, 248, 0.55)" strokeWidth="1.8" fill="none" />
        <path d="M 200 450 C 240 445, 290 470, 315 515 C 285 535, 245 520, 225 540" stroke="rgba(56, 189, 248, 0.5)" strokeWidth="1.6" fill="none" />

        {/* Neural Network Interconnect Lines */}
        <line x1="155" y1="220" x2="210" y2="245" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="210" y1="245" x2="270" y2="230" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="210" y1="245" x2="195" y2="305" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="195" y1="305" x2="140" y2="280" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="195" y1="305" x2="250" y2="330" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="250" y1="330" x2="295" y2="300" stroke="rgba(34, 211, 238, 0.5)" strokeWidth="1.4" />
        <line x1="250" y1="330" x2="220" y2="390" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="220" y1="390" x2="160" y2="360" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="220" y1="390" x2="280" y2="410" stroke="rgba(34, 211, 238, 0.55)" strokeWidth="1.4" />
        <line x1="280" y1="410" x2="330" y2="390" stroke="rgba(168, 85, 247, 0.5)" strokeWidth="1.4" />
        <line x1="220" y1="390" x2="245" y2="470" stroke="rgba(34, 211, 238, 0.5)" strokeWidth="1.4" />
        <line x1="245" y1="470" x2="290" y2="495" stroke="rgba(34, 211, 238, 0.5)" strokeWidth="1.4" />

        {/* Luminous Synaptic Junction Nodes */}
        <circle cx="155" cy="220" r="5" fill="#22d3ee" filter="url(#brainGlow)" />
        <circle cx="210" cy="245" r="4.5" fill="#38bdf8" filter="url(#brainGlow)" />
        <circle cx="270" cy="230" r="5.5" fill="#22d3ee" filter="url(#brainGlow)" />
        <circle cx="140" cy="280" r="4.5" fill="#38bdf8" filter="url(#brainGlow)" />
        <circle cx="195" cy="305" r="6.5" fill="#ffffff" stroke="#0ea5e9" strokeWidth="2.5" filter="url(#brainGlow)" />
        <circle cx="250" cy="330" r="4.5" fill="#22d3ee" filter="url(#brainGlow)" />
        <circle cx="295" cy="300" r="4" fill="#a855f7" filter="url(#brainGlow)" />
        <circle cx="160" cy="360" r="4.5" fill="#38bdf8" filter="url(#brainGlow)" />
        <circle cx="220" cy="390" r="5.5" fill="#22d3ee" filter="url(#brainGlow)" />
        <circle cx="280" cy="410" r="4.5" fill="#ec4899" filter="url(#brainGlow)" />
        <circle cx="330" cy="390" r="4" fill="#a855f7" filter="url(#brainGlow)" />
        <circle cx="245" cy="470" r="4.5" fill="#22d3ee" filter="url(#brainGlow)" />
        <circle cx="290" cy="495" r="4" fill="#38bdf8" filter="url(#brainGlow)" />
      </svg>
    </div>
  );
}

/* ============================================================
   FLOATING GLASS MEDICAL OBJECTS (MIDGROUND AROUND PANEL)
   ============================================================ */
function FloatingMedicalObjects() {
  return (
    <div className="na-floating-objects-layer" aria-hidden="true">
      {/* 1. Heart / ECG pulse */}
      <div className="na-floating-medallion medallion-pos-1" title="Cardiac Activity">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
          <path d="M3.22 12H9.5l1.5-3 2 6.5 1.5-3.5h6.28" />
        </svg>
      </div>

      {/* 2. Medical Report Document */}
      <div className="na-floating-medallion medallion-pos-2" title="Lab Report">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#7c3aed" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="16" y1="13" x2="8" y2="13" />
          <line x1="16" y1="17" x2="8" y2="17" />
        </svg>
      </div>

      {/* 3. Luminous Medical Cross */}
      <div className="na-floating-medallion medallion-pos-3" title="MediSense AI">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </div>

      {/* 4. Lungs / Thoracic X-Ray */}
      <div className="na-floating-medallion medallion-pos-4" title="Radiology / Thoracic">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M7 9a3 3 0 0 1 3 3v3a1 1 0 0 1-2 0v-3a1 1 0 0 0-1-1" />
          <path d="M17 9a3 3 0 0 0-3 3v3a1 1 0 0 0 2 0v-3a1 1 0 0 1 1-1" />
          <line x1="12" y1="7" x2="12" y2="17" strokeWidth="2.5" />
        </svg>
      </div>

      {/* 5. Radiology Scan Frame */}
      <div className="na-floating-medallion medallion-pos-5" title="Imaging Scan">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#0ea5e9" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="4" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
      </div>
    </div>
  );
}

/* ============================================================
   BOTTOM FLOWING MOLECULAR WAVE & LIGHT TRAILS
   ============================================================ */
function BottomMolecularWaveVisual() {
  return (
    <div className="na-bottom-wave-layer" aria-hidden="true">
      <svg className="na-bottom-wave-svg" viewBox="0 0 1440 180" fill="none" preserveAspectRatio="none">
        <defs>
          <linearGradient id="bottomWaveGrad1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.1" />
            <stop offset="25%" stopColor="#0ea5e9" stopOpacity="0.35" />
            <stop offset="50%" stopColor="#38bdf8" stopOpacity="0.5" />
            <stop offset="75%" stopColor="#0ea5e9" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.1" />
          </linearGradient>
          <linearGradient id="bottomWaveGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.08" />
            <stop offset="40%" stopColor="#22d3ee" stopOpacity="0.25" />
            <stop offset="70%" stopColor="#7c3aed" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#0ea5e9" stopOpacity="0.08" />
          </linearGradient>
          <linearGradient id="bottomWaveFill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.12" />
            <stop offset="50%" stopColor="#e0f2fe" stopOpacity="0.06" />
            <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
          </linearGradient>
          <filter id="waveGlow" x="-10%" y="-20%" width="120%" height="140%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Ambient bottom wave fill */}
        <path
          d="M 0 120 C 240 70, 480 150, 720 100 C 960 50, 1200 130, 1440 90 L 1440 180 L 0 180 Z"
          fill="url(#bottomWaveFill)"
        />

        {/* Primary flowing wave line */}
        <path
          d="M 0 120 C 240 70, 480 150, 720 100 C 960 50, 1200 130, 1440 90"
          stroke="url(#bottomWaveGrad1)"
          strokeWidth="2.5"
          fill="none"
          filter="url(#waveGlow)"
        />

        {/* Secondary harmonized wave line */}
        <path
          d="M 0 145 C 220 110, 500 165, 760 125 C 1020 85, 1240 145, 1440 120"
          stroke="url(#bottomWaveGrad2)"
          strokeWidth="1.8"
          strokeDasharray="8 5"
          fill="none"
        />

        {/* Molecular network nodes along the wave */}
        <circle cx="240" cy="70" r="4" fill="#22d3ee" filter="url(#waveGlow)" />
        <circle cx="480" cy="150" r="3.5" fill="#38bdf8" />
        <circle cx="720" cy="100" r="5" fill="#ffffff" stroke="#0ea5e9" strokeWidth="2" filter="url(#waveGlow)" />
        <circle cx="960" cy="50" r="4.5" fill="#22d3ee" filter="url(#waveGlow)" />
        <circle cx="1200" cy="130" r="3.5" fill="#a855f7" />

        {/* Subtle connecting lines */}
        <line x1="240" y1="70" x2="310" y2="100" stroke="rgba(34, 211, 238, 0.3)" strokeWidth="1" strokeDasharray="3 3" />
        <circle cx="310" cy="100" r="2.5" fill="#38bdf8" opacity="0.7" />
        <line x1="720" y1="100" x2="780" y2="60" stroke="rgba(34, 211, 238, 0.3)" strokeWidth="1" strokeDasharray="3 3" />
        <circle cx="780" cy="60" r="3" fill="#22d3ee" opacity="0.8" />
        <line x1="960" y1="50" x2="1030" y2="85" stroke="rgba(56, 189, 248, 0.3)" strokeWidth="1" strokeDasharray="3 3" />
        <circle cx="1030" cy="85" r="2.5" fill="#38bdf8" opacity="0.7" />
      </svg>
    </div>
  );
}

/* ============================================================
   MAIN NEW ASSESSMENT COMPONENT
   ============================================================ */

const OPTIONS = [
  {
    id: INPUT_TYPES.SYMPTOMS,
    label: 'Symptoms',
    desc: 'Describe symptoms in natural language.',
    cardClass: 'card-symptoms',
    tileClass: 'tile-symptoms',
    icon: <StethoscopePulseIcon size={22} color="#ffffff" />,
    visual: <EcgWaveVisual />,
  },
  {
    id: INPUT_TYPES.MEDICAL_REPORT,
    label: 'Medical Report',
    desc: 'Lab, blood, radiology, or other health reports (PDF/JPG/PNG).',
    cardClass: 'card-report',
    tileClass: 'tile-report',
    icon: <MedicalReportDocIcon size={22} color="#ffffff" />,
    visual: <ReportBarsVisual />,
  },
  {
    id: INPUT_TYPES.XRAY,
    label: 'X-Ray',
    desc: 'Upload an X-ray image for supported regions.',
    cardClass: 'card-xray',
    tileClass: 'tile-xray',
    icon: <RadiologyIcon size={22} color="#ffffff" />,
    visual: <XrayChestVisual />,
  },
];

export default function NewAssessment() {
  const navigate = useNavigate();
  const { setInputTypes, setAssessmentId, setDraft } = useAssessment();
  const [selected, setSelected] = useState([]);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const toggle = (id) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  const handleKeyDown = (e, id) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      toggle(id);
    }
  };

  const continueFlow = async () => {
    setError('');
    if (!selected.length) {
      setError('Select at least one input type.');
      return;
    }
    setSubmitting(true);
    try {
      const assessment = await createAssessment(selected);
      setDraft({
        assessmentId: assessment.assessment_id,
        inputTypes: selected,
        stepIndex: 0,
      });
      setInputTypes(selected);
      setAssessmentId(assessment.assessment_id);

      if (selected.includes(INPUT_TYPES.SYMPTOMS)) {
        navigate('/assessment/symptoms');
      } else if (selected.includes(INPUT_TYPES.MEDICAL_REPORT)) {
        navigate('/assessment/report');
      } else {
        navigate('/assessment/xray');
      }
    } catch (err) {
      setError(err.message || 'Failed to initiate assessment.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="new-assessment-page">
      {/* 3D Medical AI Background System */}
      <NeuralNetworkCanvas />
      <DnaHelixVisual />
      <AiBrainVisual />
      <FloatingMedicalObjects />
      <BottomMolecularWaveVisual />

      {/* Central Floating 3D Glass Assessment Panel (Reference Proportions) */}
      <section className="new-assessment-panel" aria-label="New Health Assessment Form">
        {/* Two-Column Header Arrangement: Left Content + Upper-Right Stepper */}
        <div className="na-panel-header">
          <div className="na-header-left">
            <div className="na-top-badge">
              <span className="na-top-badge-pulse" />
              <SparkleIcon size={14} color="#0284c7" />
              <span>AI-Powered Health Assessment</span>
            </div>

            <h1 className="na-title">
              New Health <span className="na-title-highlight">Assessment</span>
            </h1>

            <p className="na-description">
              Start by selecting the information you have available.
            </p>
            <p className="na-sub-description">
              You can choose one or more options. Our AI will analyze them together for better insights.
            </p>
          </div>

          <div className="na-header-right">
            {/* Upper-Right Progress Stepper 1 — 2 — 3 — 4 */}
            <div className="na-progress-bar" aria-label="Assessment Progress Steps">
              <div className="na-progress-step active">
                <span className="na-progress-number">1</span>
                <span className="na-progress-label">Input</span>
              </div>
              <div className="na-progress-line" />
              <div className="na-progress-step">
                <span className="na-progress-number">2</span>
                <span className="na-progress-label">Analysis</span>
              </div>
              <div className="na-progress-line" />
              <div className="na-progress-step">
                <span className="na-progress-number">3</span>
                <span className="na-progress-label">Review</span>
              </div>
              <div className="na-progress-line" />
              <div className="na-progress-step">
                <span className="na-progress-number">4</span>
                <span className="na-progress-label">Results</span>
              </div>
            </div>
          </div>
        </div>

        {/* Three Input Selection Cards in One Row */}
        <div className="na-cards-grid" role="group" aria-label="Available Input Types">
          {OPTIONS.map((opt) => {
            const isSelected = selected.includes(opt.id);
            return (
              <div
                key={opt.id}
                role="checkbox"
                aria-checked={isSelected}
                tabIndex={0}
                className={`na-input-card ${opt.cardClass} ${isSelected ? 'selected' : ''}`}
                onClick={() => toggle(opt.id)}
                onKeyDown={(e) => handleKeyDown(e, opt.id)}
              >
                {/* Card Top: Icon Tile & Checkbox Indicator */}
                <div className="na-card-top">
                  <div className={`na-icon-tile ${opt.tileClass}`}>
                    {opt.icon}
                  </div>
                  <div className="na-check-indicator" aria-hidden="true">
                    {isSelected ? <CheckIcon size={12} color="#ffffff" /> : null}
                  </div>
                </div>

                {/* Card Content */}
                <h3 className="na-card-title">{opt.label}</h3>
                <p className="na-card-desc">{opt.desc}</p>

                {/* Decorative Bottom Wave / Visual */}
                <div className="na-card-wave-box" aria-hidden="true">
                  {opt.visual}
                </div>
              </div>
            );
          })}
        </div>

        {error ? (
          <div className="na-error-box">
            <ErrorMessage message={error} onDismiss={() => setError('')} />
          </div>
        ) : null}

        {/* Panel Footer: Continue Button in Lower-Left, Security Footnote on Right */}
        <div className="na-panel-footer">
          <button
            type="button"
            className="na-continue-btn"
            onClick={continueFlow}
            disabled={submitting}
          >
            <span>{submitting ? 'Creating Assessment...' : 'Continue'}</span>
            <span className="na-btn-arrow">
              <ArrowRightIcon size={16} color="#ffffff" />
            </span>
          </button>

          <div className="na-security-footnote">
            <span className="na-security-icon">
              <ShieldLockIcon size={15} color="#10b981" />
            </span>
            <span>Your health data is private and secure</span>
          </div>
        </div>
      </section>
    </div>
  );
}
