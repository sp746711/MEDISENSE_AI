import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getDashboardSummary } from '../services/assessmentService';
import { useAuth } from '../hooks/useAuth';
import './Dashboard.css';

// SVG Icons
function HeartPulseIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z" />
      <path d="M3.22 12H9.5l1.5-3 2 6.5 1.5-3.5h6.28" />
    </svg>
  );
}

function CalendarIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <path d="M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01M16 18h.01" />
    </svg>
  );
}

function ClockIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
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

function PlusIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

// Green Document / Report / Check Visual for Total Assessments
function DocumentCheckIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="m9 15 2 2 4-4" />
    </svg>
  );
}

// Recognizable Anatomical Lungs / Chest X-Ray Visual
function LungsXRayIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2v6" />
      <path d="M12 8l-2.5 3" />
      <path d="M12 8l2.5 3" />
      <path d="M9.5 11c-2.4 0-4.8 2-4.8 5.4 0 2.4 2 4.6 4.3 4.6 1.4 0 2.5-1.5 2.5-3.5V11z" />
      <path d="M14.5 11c2.4 0 4.8 2 4.8 5.4 0 2.4-2 4.6-4.3 4.6-1.4 0-2.5-1.5-2.5-3.5V11z" />
    </svg>
  );
}

// Purple Medical Document Visual for Medical Reports
function MedicalReportIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M12 11v6" />
      <path d="M9 14h6" />
    </svg>
  );
}

function SparklineSvg({ color = '#0ea5e9' }) {
  return (
    <svg className="dashboard-stat-sparkline" width="62" height="22" viewBox="0 0 62 22" fill="none">
      <path
        d="M2 17 C 11 17, 15 6, 22 10 C 29 14, 35 4, 43 8 C 50 12, 54 3, 60 5"
        stroke={color}
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function LightbulbIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-1 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
      <path d="M9 18h6" />
      <path d="M10 22h4" />
    </svg>
  );
}

function WaterDropIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z" />
    </svg>
  );
}

function ActivityIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function MoonIcon({ size = 22, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}

function ChevronLeftIcon({ size = 15, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  );
}

function ChevronRightIcon({ size = 15, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  );
}

function HospitalIcon({ size = 24, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 21h18" />
      <path d="M5 21V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2v16" />
      <path d="M9 21v-4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v4" />
      <path d="M10 9h4" />
      <path d="M12 7v4" />
    </svg>
  );
}

function DoctorIcon({ size = 24, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4.5 3h15M12 3v9M7.5 12a4.5 4.5 0 0 0 9 0V3" />
      <path d="M19 14.5a3 3 0 0 1-3 3h-2a3 3 0 0 0-3 3" />
      <circle cx="8" cy="20.5" r="1.5" />
    </svg>
  );
}

function PharmacyIcon({ size = 24, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.5 20.5 3.5 13.5a5 5 0 0 1 7.07-7.07l7 7a5 5 0 0 1-7.07 7.07z" />
      <path d="m8.5 8.5 7 7" />
      <path d="M14 4h6v6" />
    </svg>
  );
}

function CheckCircleIcon({ size = 13, color = '#10b981' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ChartBarIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="20" x2="12" y2="10" />
      <line x1="18" y1="20" x2="18" y2="4" />
      <line x1="6" y1="20" x2="6" y2="16" />
    </svg>
  );
}

// Static General Health Tips
const HEALTH_TIPS = [
  {
    icon: WaterDropIcon,
    title: 'Stay Hydrated',
    desc: 'Drink enough water to support overall health and body function.',
  },
  {
    icon: ActivityIcon,
    title: 'Daily Movement',
    desc: 'Engage in at least 30 minutes of moderate activity each day to boost cardiovascular health.',
  },
  {
    icon: MoonIcon,
    title: 'Consistent Sleep',
    desc: 'Maintain 7-8 hours of quality sleep to enhance immune defense and cognitive recovery.',
  },
];

// Helper to parse date parts for stacked date block
function parseDateParts(isoDate) {
  if (!isoDate) {
    return { day: '02', month: 'OCT', year: '2026' };
  }
  try {
    const d = new Date(isoDate);
    const day = d.toLocaleDateString('en-US', { day: '2-digit' });
    const month = d.toLocaleDateString('en-US', { month: 'short' }).toUpperCase();
    const year = d.getFullYear();
    return { day, month, year };
  } catch {
    return { day: '02', month: 'OCT', year: '2026' };
  }
}

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  // Dynamic Browser Date and Time
  const [currentDateTime, setCurrentDateTime] = useState(() => new Date());
  const [currentTipIndex, setCurrentTipIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentDateTime(new Date());
    }, 30000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const summary = await getDashboardSummary();
        if (active) setData(summary);
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  if (loading) return <LoadingScreen message="Loading dashboard..." />;

  // Format Date & Time dynamically
  const formattedDate = currentDateTime.toLocaleDateString('en-US', {
    weekday: 'long',
    day: '2-digit',
    month: 'long',
    year: 'numeric',
  });

  const formattedTime = currentDateTime.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
  });

  // Carousel handlers
  const handlePrevTip = () => {
    setCurrentTipIndex((prev) => (prev === 0 ? HEALTH_TIPS.length - 1 : prev - 1));
  };

  const handleNextTip = () => {
    setCurrentTipIndex((prev) => (prev === HEALTH_TIPS.length - 1 ? 0 : prev + 1));
  };

  const currentTip = HEALTH_TIPS[currentTipIndex];
  const TipIcon = currentTip.icon;

  // Helper for pathway badge styling
  const renderPathwayBadge = (pathway) => {
    const p = String(pathway || '').toUpperCase();
    let badgeClass = 'pathway-badge-unknown';
    let label = 'Not determined';

    if (p === 'EMERGENCY') {
      badgeClass = 'pathway-badge-emergency';
      label = 'Emergency';
    } else if (p === 'CONSULTATION') {
      badgeClass = 'pathway-badge-consultation';
      label = 'Consultation';
    } else if (p === 'MILD') {
      badgeClass = 'pathway-badge-mild';
      label = 'Mild';
    }

    return (
      <span className={`dashboard-pathway-badge ${badgeClass}`}>
        <span className="pathway-dot" />
        {label}
      </span>
    );
  };

  return (
    <div className="dashboard-page">
      {/* ==================================================
          1. HERO / WELCOME SECTION (3-COLUMN DESKTOP COMPOSITION:
             LEFT GREETING & CTA -> CENTER-RIGHT SMART INSIGHTS -> FAR RIGHT REALISTIC STETHOSCOPE & GLOWING CROSS)
          ================================================== */}
      <section className="dashboard-hero">
        {/* Left: Greeting, user name, date/time, New Assessment CTA */}
        <div className="dashboard-hero-left">
          <div className="dashboard-hero-badge">
            <HeartPulseIcon size={14} color="#0ea5e9" />
            <span>Your Health, Our Priority</span>
          </div>

          <p className="dashboard-hero-welcome">Welcome back,</p>
          <h1 className="dashboard-hero-name">
            {data?.user_name || user?.name || 'Sujan Pradhan'} 👋
          </h1>
          <p className="dashboard-hero-subtitle">
            Track your health, analyze your data and get personalized insights.
          </p>

          <div className="dashboard-hero-datetime">
            <div className="dashboard-hero-date-item">
              <CalendarIcon size={14} color="#0ea5e9" />
              <span>{formattedDate}</span>
            </div>
            <span className="dashboard-hero-divider">|</span>
            <div className="dashboard-hero-date-item">
              <ClockIcon size={14} color="#0ea5e9" />
              <span>{formattedTime}</span>
            </div>
          </div>

          <button
            type="button"
            className="dashboard-hero-cta"
            onClick={() => navigate('/assessment/new')}
          >
            <PlusIcon size={16} color="#ffffff" />
            <span>New Assessment</span>
            <span className="cta-arrow">
              <ArrowRightIcon size={15} color="#ffffff" />
            </span>
          </button>
        </div>

        {/* Center-Right: Smart Insights Glass Card (3D Glassmorphism) */}
        <div className="dashboard-hero-center">
          <div className="dashboard-hero-info-card">
            <div className="dashboard-info-card-header">
              <span className="dashboard-info-badge-dot" />
              <h3 className="dashboard-info-card-title">
                Smarter Insights<br />for a Healthier You
              </h3>
            </div>
            <p className="dashboard-info-card-desc">
              Analyze symptoms, reports and X-rays with AI-powered insights.
            </p>
            <div className="dashboard-info-card-footer">
              <HeartPulseIcon size={13} color="#0ea5e9" />
              <span>MediSense Clinical AI Engine</span>
            </div>
          </div>
        </div>

        {/* Far Right: Volumetric Realistic Medical Stethoscope, Glowing 3D Cross, Layered Wave Lines, Floating Glass Icons */}
        <div className="dashboard-hero-right">
          <div className="dashboard-hero-visual-canvas">
            {/* Photorealistic 3D Medical Illustration SVG */}
            <svg className="dashboard-hero-decorations" viewBox="0 0 360 220" fill="none">
              <defs>
                {/* 3D Realistic Chrome Gradient for Binaural Tubes */}
                <linearGradient id="chromeMetalGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#94a3b8" />
                  <stop offset="25%" stopColor="#f8fafc" />
                  <stop offset="50%" stopColor="#ffffff" />
                  <stop offset="75%" stopColor="#cbd5e1" />
                  <stop offset="100%" stopColor="#475569" />
                </linearGradient>

                {/* Volumetric Cylindrical Gradient for Medical Acoustic Tubing (Heavy Dark Navy/Blue) */}
                <linearGradient id="realisticTubeGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#041e42" />
                  <stop offset="20%" stopColor="#0284c7" />
                  <stop offset="45%" stopColor="#38bdf8" />
                  <stop offset="70%" stopColor="#0284c7" />
                  <stop offset="90%" stopColor="#0369a1" />
                  <stop offset="100%" stopColor="#041e42" />
                </linearGradient>

                {/* Turret & Diaphragm Metallic Shading */}
                <linearGradient id="chestpieceRimGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#ffffff" />
                  <stop offset="30%" stopColor="#e2e8f0" />
                  <stop offset="65%" stopColor="#94a3b8" />
                  <stop offset="100%" stopColor="#1e293b" />
                </linearGradient>

                <linearGradient id="diaphragmFaceGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#e0f2fe" />
                  <stop offset="40%" stopColor="#38bdf8" />
                  <stop offset="85%" stopColor="#0284c7" />
                  <stop offset="100%" stopColor="#0369a1" />
                </linearGradient>

                {/* Luminous Radiant Glow for Medical Cross */}
                <filter id="heroCrossRadiance" x="-60%" y="-60%" width="220%" height="220%">
                  <feGaussianBlur stdDeviation="9" result="blur" />
                  <feDropShadow dx="0" dy="0" stdDeviation="14" floodColor="#38bdf8" floodOpacity="0.75" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>

                {/* Realistic Stethoscope Drop Shadow on Ambient Background */}
                <filter id="tubeShadow" x="-25%" y="-25%" width="150%" height="150%">
                  <feDropShadow dx="0" dy="6" stdDeviation="7" floodColor="#082b61" floodOpacity="0.35" />
                </filter>
              </defs>

              {/* Layered Flowing White/Cyan Waves Creating Depth */}
              <path d="M 0 160 Q 75 105, 170 148 T 360 105" stroke="rgba(56, 189, 248, 0.42)" strokeWidth="2.8" fill="none" />
              <path d="M 25 180 Q 115 125, 215 168 T 360 132" stroke="rgba(14, 165, 233, 0.30)" strokeWidth="2.2" fill="none" />
              <path d="M 65 200 Q 165 152, 265 190 T 360 158" stroke="rgba(255, 255, 255, 0.60)" strokeWidth="1.8" fill="none" />

              {/* Ambient Particle Rings */}
              <circle cx="245" cy="100" r="95" stroke="rgba(56, 189, 248, 0.22)" strokeWidth="1.2" strokeDasharray="5 5" />
              <circle cx="245" cy="100" r="60" stroke="rgba(14, 165, 233, 0.25)" strokeWidth="1" />
              <circle cx="305" cy="42" r="3.5" fill="#38bdf8" opacity="0.9" />
              <circle cx="170" cy="162" r="3" fill="#0ea5e9" opacity="0.8" />

              {/* LARGE GLOWING MEDICAL CROSS (Integrated Behind Stethoscope) */}
              <g filter="url(#heroCrossRadiance)" opacity="0.9">
                <rect x="230" y="20" width="30" height="92" rx="7" fill="#38bdf8" opacity="0.45" />
                <rect x="199" y="51" width="92" height="30" rx="7" fill="#38bdf8" opacity="0.45" />
                <rect x="233" y="23" width="24" height="86" rx="5" fill="#ffffff" opacity="0.82" />
                <rect x="202" y="54" width="86" height="24" rx="5" fill="#ffffff" opacity="0.82" />
              </g>

              {/* REALISTIC PROMINENT VOLUMETRIC MEDICAL STETHOSCOPE */}
              {/* Binaural Metallic Tubes & Tension Spring */}
              <g filter="url(#tubeShadow)">
                {/* Polished Stainless Steel Binaural Headset Arch */}
                <path
                  d="M 222 6 C 208 25, 228 50, 250 52 C 272 50, 292 25, 278 6"
                  stroke="url(#chromeMetalGrad)"
                  strokeWidth="5"
                  strokeLinecap="round"
                  fill="none"
                />
                {/* Silicone Ergonomic Ear Tips with Specular Highlights */}
                <ellipse cx="222" cy="6" rx="4.5" ry="3.5" fill="#0f172a" stroke="#cbd5e1" strokeWidth="1" transform="rotate(-15 222 6)" />
                <ellipse cx="278" cy="6" rx="4.5" ry="3.5" fill="#0f172a" stroke="#cbd5e1" strokeWidth="1" transform="rotate(15 278 6)" />
                {/* Chrome Tension Spring Arch & Turret Yoke */}
                <path d="M 240 46 Q 250 53, 260 46" stroke="url(#chromeMetalGrad)" strokeWidth="3.2" fill="none" />
                <rect x="245" y="50" width="10" height="14" rx="2.5" fill="url(#chromeMetalGrad)" />
              </g>

              {/* Volumetric Acoustic Tubing with Cylindrical Shading & High-Gloss Specular Glint */}
              <g filter="url(#tubeShadow)">
                {/* Main Volumetric Tube (Thick Realistic Rubber) */}
                <path
                  d="M 250 63 C 250 108, 330 110, 325 154 C 320 196, 245 212, 198 200 C 154 189, 124 158, 118 122"
                  stroke="url(#realisticTubeGrad)"
                  strokeWidth="11"
                  strokeLinecap="round"
                  fill="none"
                />
                {/* Specular Highlight Glint Along Top Contour */}
                <path
                  d="M 250 63 C 250 105, 328 107, 323 151 C 318 191, 247 207, 201 196 C 158 185, 128 155, 122 122"
                  stroke="#ffffff"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeOpacity="0.75"
                  fill="none"
                />
              </g>

              {/* Detailed Dual-Head Chestpiece (Heavy Stainless Steel Diaphragm & Bell) */}
              <g transform="translate(118, 122)" filter="url(#tubeShadow)">
                {/* Chrome Outer Housing Rim with Chamfered Edge */}
                <circle cx="0" cy="0" r="22" fill="url(#chestpieceRimGrad)" stroke="#38bdf8" strokeWidth="2.8" />
                {/* Non-chill Protective Ring */}
                <circle cx="0" cy="0" r="18" fill="#041e42" opacity="0.8" />
                {/* Tunable Acoustic Diaphragm Face */}
                <circle cx="0" cy="0" r="15" fill="url(#diaphragmFaceGrad)" stroke="rgba(255, 255, 255, 0.85)" strokeWidth="1.8" />
                {/* Realistic Specular Reflection Crescent */}
                <path d="M -11 -8 A 13 13 0 0 1 9 -11" stroke="#ffffff" strokeWidth="2.8" strokeLinecap="round" fill="none" opacity="0.9" />
                {/* Central Chrome Medallion Core */}
                <circle cx="0" cy="0" r="6" fill="#ffffff" opacity="0.95" />
                <circle cx="0" cy="0" r="3" fill="#0284c7" />
              </g>
            </svg>

            {/* Floating Circular Glass Medical Icons Surrounding Stethoscope */}
            <div className="hero-bubble hero-bubble-heart" title="Health Priority">
              <HeartPulseIcon size={18} color="#ef4444" />
            </div>

            <div className="hero-bubble hero-bubble-doc" title="Triage & Care">
              <DocumentCheckIcon size={16} color="#10b981" />
            </div>

            <div className="hero-bubble hero-bubble-report" title="Clinical Reports">
              <MedicalReportIcon size={16} color="#7c3aed" />
            </div>

            <div className="hero-bubble hero-bubble-chart" title="AI Analytics">
              <ChartBarIcon size={18} color="#0ea5e9" />
            </div>
          </div>
        </div>
      </section>

      {/* Error Message if any */}
      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {/* ==================================================
          2. FOUR STATISTICS CARDS (HORIZONTAL COMPOSITION)
          [ICON]  Title                →
                  Value          ~~~~~~~
                  Description
          ================================================== */}
      <section className="dashboard-stat-grid">
        {/* Card 1: Total Assessments */}
        <div className="dashboard-stat-card stat-card-green">
          <div className="dashboard-stat-left">
            <div className="dashboard-stat-icon-tile dashboard-stat-tile-teal">
              <DocumentCheckIcon size={22} color="#10b981" />
            </div>
          </div>
          <div className="dashboard-stat-body">
            <span className="dashboard-stat-title">Total Assessments</span>
            <div className="dashboard-stat-value">{data?.total_assessments ?? 0}</div>
            <span className="dashboard-stat-desc">All assessments created</span>
          </div>
          <div className="dashboard-stat-right">
            <div className="dashboard-stat-arrow">
              <ArrowRightIcon size={14} color="#10b981" />
            </div>
            <SparklineSvg color="#10b981" />
          </div>
        </div>

        {/* Card 2: X-Rays Processed */}
        <div className="dashboard-stat-card stat-card-blue">
          <div className="dashboard-stat-left">
            <div className="dashboard-stat-icon-tile dashboard-stat-tile-blue">
              <LungsXRayIcon size={22} color="#0ea5e9" />
            </div>
          </div>
          <div className="dashboard-stat-body">
            <span className="dashboard-stat-title">X-Rays Processed</span>
            <div className="dashboard-stat-value">{data?.xrays_processed ?? 0}</div>
            <span className="dashboard-stat-desc">Uploaded and analyzed X-rays</span>
          </div>
          <div className="dashboard-stat-right">
            <div className="dashboard-stat-arrow">
              <ArrowRightIcon size={14} color="#0ea5e9" />
            </div>
            <SparklineSvg color="#0ea5e9" />
          </div>
        </div>

        {/* Card 3: Medical Reports */}
        <div className="dashboard-stat-card stat-card-purple">
          <div className="dashboard-stat-left">
            <div className="dashboard-stat-icon-tile dashboard-stat-tile-purple">
              <MedicalReportIcon size={22} color="#7c3aed" />
            </div>
          </div>
          <div className="dashboard-stat-body">
            <span className="dashboard-stat-title">Medical Reports</span>
            <div className="dashboard-stat-value">{data?.reports_processed ?? 0}</div>
            <span className="dashboard-stat-desc">Uploaded medical reports</span>
          </div>
          <div className="dashboard-stat-right">
            <div className="dashboard-stat-arrow">
              <ArrowRightIcon size={14} color="#7c3aed" />
            </div>
            <SparklineSvg color="#7c3aed" />
          </div>
        </div>

        {/* Card 4: Appointments */}
        <div className="dashboard-stat-card stat-card-orange">
          <div className="dashboard-stat-left">
            <div className="dashboard-stat-icon-tile dashboard-stat-tile-orange">
              <CalendarIcon size={22} color="#f97316" />
            </div>
          </div>
          <div className="dashboard-stat-body">
            <span className="dashboard-stat-title">Appointments</span>
            <div className="dashboard-stat-value">{data?.appointments ?? 0}</div>
            <span className="dashboard-stat-desc">Scheduled appointments</span>
          </div>
          <div className="dashboard-stat-right">
            <div className="dashboard-stat-arrow">
              <ArrowRightIcon size={14} color="#f97316" />
            </div>
            <SparklineSvg color="#f97316" />
          </div>
        </div>
      </section>

      {/* ==================================================
          3. MIDDLE SECTION (RECENT ASSESSMENTS + HEALTH TIPS)
          EXACT POSITION: ONE SHARED ROW, SAME TOP, SAME BOTTOM, SAME HEIGHT!
          RECENT ASSESSMENTS ≈ 66% (2/3), HEALTH TIPS ≈ 34% (1/3)
          ================================================== */}
      <section className="dashboard-middle-grid">
        {/* Recent Assessments Card (occupies ~2/3 of row) */}
        <div className="dashboard-card dashboard-recent-card">
          <div className="dashboard-card-header">
            <div className="dashboard-card-title-group">
              <div className="dashboard-card-title-icon">
                <DocumentCheckIcon size={17} color="#0ea5e9" />
              </div>
              <h2 className="dashboard-card-title">Recent Assessments</h2>
            </div>
            <button
              type="button"
              className="dashboard-view-all-btn"
              onClick={() => navigate('/history')}
            >
              <span>View All</span>
              <ArrowRightIcon size={14} color="#0ea5e9" />
            </button>
          </div>

          {(!data?.recent_assessments || data.recent_assessments.length === 0) ? (
            <div className="dashboard-empty-state">
              <div className="dashboard-empty-icon">
                <DocumentCheckIcon size={22} color="#0ea5e9" />
              </div>
              <p className="dashboard-empty-text">
                No recent assessments yet. Start a new assessment to see your history here.
              </p>
              <button
                type="button"
                className="dashboard-recent-view-btn"
                onClick={() => navigate('/assessment/new')}
              >
                + Start Assessment
              </button>
            </div>
          ) : (
            <div className="dashboard-recent-list">
              {data.recent_assessments.slice(0, 3).map((a) => {
                const { day, month, year } = parseDateParts(a.created_at);
                const inputList = Array.isArray(a.input_types) ? a.input_types : [];
                const inputCount = inputList.length || 1;

                // Title from inputs
                let title = 'Symptoms Assessment';
                if (inputList.includes('xray') && !inputList.includes('symptoms')) {
                  title = 'Chest X-Ray Assessment';
                } else if (inputList.includes('medical_report') && !inputList.includes('symptoms')) {
                  title = 'Medical Report Assessment';
                } else if (inputList.length > 1) {
                  title = 'Comprehensive Assessment';
                }

                return (
                  <div key={a.assessment_id} className="dashboard-recent-item">
                    <div className="dashboard-recent-left">
                      {/* Stacked Date Block matching reference (02 OCT 2026) */}
                      <div className="dashboard-date-tile">
                        <span className="dashboard-date-day">{day}</span>
                        <span className="dashboard-date-month">{month}</span>
                        <span className="dashboard-date-year">{year}</span>
                      </div>

                      <div className="dashboard-recent-info">
                        <div className="dashboard-recent-title">{title}</div>
                        <div className="dashboard-recent-meta">
                          <span className="meta-status-completed">
                            <CheckCircleIcon size={12} color="#10b981" />
                            <span>{a.status || 'Completed'}</span>
                          </span>
                          <span>•</span>
                          <span>
                            {inputCount} input {inputCount === 1 ? 'type' : 'types'}
                          </span>
                          <span>•</span>
                          <span className="meta-report-avail">
                            <MedicalReportIcon size={12} color="#0ea5e9" />
                            <span>Report Available</span>
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="dashboard-recent-right">
                      {renderPathwayBadge(a.pathway)}
                      <button
                        type="button"
                        className="dashboard-recent-view-btn"
                        onClick={() => navigate(`/assessment/${a.assessment_id}`)}
                      >
                        <span>View Details</span>
                        <ArrowRightIcon size={13} color="currentColor" />
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Health Tips Card (occupies ~1/3 of row, shares same row height) */}
        <div className="dashboard-card dashboard-health-tips-card">
          <div className="dashboard-card-header">
            <div className="dashboard-card-title-group">
              <div className="dashboard-card-title-icon">
                <LightbulbIcon size={17} color="#0ea5e9" />
              </div>
              <h2 className="dashboard-card-title">Health Tips</h2>
            </div>
            <div className="dashboard-tip-controls">
              <button
                type="button"
                className="dashboard-tip-nav-btn"
                onClick={handlePrevTip}
                aria-label="Previous tip"
              >
                <ChevronLeftIcon size={13} />
              </button>
              <span className="dashboard-tip-counter">
                {currentTipIndex + 1}/{HEALTH_TIPS.length}
              </span>
              <button
                type="button"
                className="dashboard-tip-nav-btn"
                onClick={handleNextTip}
                aria-label="Next tip"
              >
                <ChevronRightIcon size={13} />
              </button>
            </div>
          </div>

          <div className="dashboard-tip-container">
            <div className="dashboard-tip-main">
              <div className="dashboard-tip-icon-badge">
                <TipIcon size={20} color="#0ea5e9" />
              </div>
              <h4 className="dashboard-tip-title">{currentTip.title}</h4>
              <p className="dashboard-tip-desc">{currentTip.desc}</p>
            </div>

            {/* Progress indicator segments */}
            <div className="dashboard-tip-bottom">
              <div className="dashboard-tip-progress">
                {HEALTH_TIPS.map((_, idx) => (
                  <div
                    key={idx}
                    className={`dashboard-tip-progress-bar ${idx === currentTipIndex ? 'active' : ''}`}
                  />
                ))}
              </div>
              <p className="dashboard-tip-disclaimer">
                General health information • Consult a doctor for medical advice
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ==================================================
          4. BOTTOM CONTEXTUAL CARDS (NO EMPTY GAP ABOVE, 3 EQUAL-WIDTH CARDS):
             - Emergency (Hospital building + Ambulance visual + Red/pink glow)
             - Consult (Realistic Stethoscope visual in lower-right + Blue/cyan glow)
             - Medical Shops (Pharmacy Storefront + Pill visual + Green/mint glow)
          ================================================== */}
      <section className="dashboard-pathway-grid">
        {/* Card 1: Emergency (Hospital & Ambulance Scenic Visual) */}
        <div className="dashboard-pathway-card dashboard-pathway-emergency">
          <svg className="pathway-watermark-canvas" viewBox="0 0 160 140" fill="none">
            {/* Hospital Building & Windows */}
            <rect x="15" y="30" width="75" height="102" rx="3" fill="#ef4444" opacity="0.32" />
            <rect x="42" y="12" width="24" height="20" rx="2" fill="#ef4444" opacity="0.45" />
            {/* Hospital Red Cross */}
            <path d="M 54 16 V 28 M 48 22 H 60" stroke="#ffffff" strokeWidth="2.8" strokeLinecap="round" />
            {/* Illuminated Hospital Windows */}
            <rect x="23" y="40" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="45" y="40" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="67" y="40" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="23" y="60" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="45" y="60" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="67" y="60" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="23" y="80" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="45" y="80" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            <rect x="67" y="80" width="12" height="12" rx="1.5" fill="#ffffff" opacity="0.65" />
            {/* Ambulance Vehicle Silhouette & Flashing Light */}
            <path d="M 85 90 H 132 L 144 106 V 126 H 85 Z" fill="#ef4444" opacity="0.45" />
            <circle cx="100" cy="126" r="7" fill="#ef4444" opacity="0.75" />
            <circle cx="132" cy="126" r="7" fill="#ef4444" opacity="0.75" />
            <rect x="104" y="96" width="16" height="5" rx="1" fill="#ffffff" opacity="0.8" />
            <path d="M 112 92 V 105 M 105 98 H 119" stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" />
            <circle cx="120" cy="86" r="3.5" fill="#ef4444" opacity="0.9" />
          </svg>

          <div>
            <div className="dashboard-pathway-header">
              <div className="dashboard-pathway-icon-tile pathway-icon-emergency">
                <HospitalIcon size={22} color="#ef4444" />
              </div>
              <h3 className="dashboard-pathway-title">Emergency</h3>
            </div>
            <div className="dashboard-pathway-subtitle pathway-subtitle-emergency">
              Urgent medical attention recommended
            </div>
            <p className="dashboard-pathway-desc">
              Find the nearest hospital for immediate care.
            </p>
          </div>
          <div className="dashboard-pathway-actions">
            <button
              type="button"
              className="dashboard-pathway-btn pathway-btn-emergency"
              onClick={() => navigate('/facilities')}
            >
              <span>Find Nearest Hospital</span>
              <ArrowRightIcon size={14} color="#ffffff" />
            </button>
          </div>
        </div>

        {/* Card 2: Consult (REALISTIC STETHOSCOPE VISUAL Toward Lower-Right) */}
        <div className="dashboard-pathway-card dashboard-pathway-consult">
          <svg className="pathway-watermark-canvas" viewBox="0 0 160 140" fill="none">
            <defs>
              <linearGradient id="consultStethGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#38bdf8" />
                <stop offset="45%" stopColor="#0ea5e9" />
                <stop offset="100%" stopColor="#0369a1" />
              </linearGradient>
            </defs>
            {/* Binaural Chrome Headset */}
            <path
              d="M 45 12 C 30 28, 48 56, 76 60 C 104 56, 122 28, 107 12"
              stroke="#0284c7"
              strokeWidth="4"
              strokeLinecap="round"
              fill="none"
              opacity="0.45"
            />
            {/* Spring & Flexible Loop Toward Lower-Right */}
            <path
              d="M 76 60 C 76 90, 142 92, 132 120"
              stroke="url(#consultStethGrad)"
              strokeWidth="6"
              strokeLinecap="round"
              fill="none"
              opacity="0.7"
            />
            {/* Specular Ridge on Tube */}
            <path
              d="M 76 60 C 76 89, 140 91, 131 119"
              stroke="#ffffff"
              strokeWidth="1.8"
              strokeLinecap="round"
              fill="none"
              opacity="0.65"
            />
            {/* Large Diaphragm Chestpiece */}
            <circle cx="132" cy="120" r="18" fill="#0284c7" opacity="0.35" stroke="#38bdf8" strokeWidth="2.6" />
            <circle cx="132" cy="120" r="12" fill="#ffffff" opacity="0.7" stroke="#0ea5e9" strokeWidth="1.6" />
            <circle cx="132" cy="120" r="5" fill="#0284c7" opacity="0.85" />
            {/* Medical Cross Watermark in Upper Area */}
            <path d="M 40 80 V 100 M 30 90 H 50" stroke="#0ea5e9" strokeWidth="2.5" strokeLinecap="round" opacity="0.25" />
          </svg>

          <div>
            <div className="dashboard-pathway-header">
              <div className="dashboard-pathway-icon-tile pathway-icon-consult">
                <DoctorIcon size={22} color="#0284c7" />
              </div>
              <h3 className="dashboard-pathway-title">Consult</h3>
            </div>
            <div className="dashboard-pathway-subtitle pathway-subtitle-consult">
              Medical consultation recommended
            </div>
            <p className="dashboard-pathway-desc">
              Find doctors or nearest hospital for proper evaluation and treatment.
            </p>
          </div>
          <div className="dashboard-pathway-actions">
            <button
              type="button"
              className="dashboard-pathway-btn pathway-btn-consult-primary"
              onClick={() => navigate('/doctors')}
            >
              <span>Find Doctors</span>
              <ArrowRightIcon size={14} color="#ffffff" />
            </button>
            <button
              type="button"
              className="dashboard-pathway-btn pathway-btn-consult-secondary"
              onClick={() => navigate('/facilities')}
            >
              <span>Find Nearest Hospital</span>
              <ArrowRightIcon size={14} color="#0284c7" />
            </button>
          </div>
        </div>

        {/* Card 3: Medical Shops (Storefront & Pharmacy Pill Capsule Visual) */}
        <div className="dashboard-pathway-card dashboard-pathway-shops">
          <svg className="pathway-watermark-canvas" viewBox="0 0 160 140" fill="none">
            {/* Pharmacy Storefront Roof and Pillars */}
            <path d="M 18 48 L 75 16 L 132 48 V 126 H 18 Z" fill="#10b981" opacity="0.28" />
            <rect x="50" y="68" width="42" height="58" rx="3" fill="#10b981" opacity="0.45" />
            {/* Medical Store Cross on Roof */}
            <path d="M 75 30 V 44 M 68 37 H 82" stroke="#ffffff" strokeWidth="3" strokeLinecap="round" />
            {/* Capsule Pill with Division Line */}
            <g transform="translate(98, 70) rotate(-35)">
              <rect x="0" y="0" width="22" height="42" rx="11" fill="#10b981" opacity="0.6" stroke="#10b981" strokeWidth="2" />
              <line x1="0" y1="21" x2="22" y2="21" stroke="#ffffff" strokeWidth="2.5" />
            </g>
          </svg>

          <div>
            <div className="dashboard-pathway-header">
              <div className="dashboard-pathway-icon-tile pathway-icon-shops">
                <PharmacyIcon size={22} color="#10b981" />
              </div>
              <h3 className="dashboard-pathway-title">Medical Shops</h3>
            </div>
            <div className="dashboard-pathway-subtitle pathway-subtitle-shops">
              Mild condition / Self-care guidance
            </div>
            <p className="dashboard-pathway-desc">
              Find nearby medical shops for basic over-the-counter support.
            </p>
          </div>
          <div className="dashboard-pathway-actions">
            <button
              type="button"
              className="dashboard-pathway-btn pathway-btn-shops"
              onClick={() => navigate('/medical-shops')}
            >
              <span>Find Nearby Medical Shops</span>
              <ArrowRightIcon size={14} color="#ffffff" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
