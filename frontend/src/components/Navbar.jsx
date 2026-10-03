import { NavLink, useNavigate } from 'react-router-dom';
import { useState, useRef, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import './Navbar.css';

// Large Strong Luminous Medical Cross Brand Logo
function MediSenseLogo() {
  return (
    <div className="navbar-logo-badge">
      <svg width="30" height="30" viewBox="0 0 36 36" fill="none">
        <defs>
          <linearGradient id="navLogoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#22d3ee" />
            <stop offset="45%" stopColor="#0ea5e9" />
            <stop offset="100%" stopColor="#0369a1" />
          </linearGradient>
          <filter id="navLogoGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="2" stdDeviation="3.5" floodColor="#0ea5e9" floodOpacity="0.55" />
          </filter>
        </defs>
        {/* Anatomical 3D rounded medical cross */}
        <path
          d="M13.5 4C13.5 3.17 14.17 2.5 15 2.5H21C21.83 2.5 22.5 3.17 22.5 4V13.5H32C32.83 13.5 33.5 14.17 33.5 15V21C33.5 21.83 32.83 22.5 32 22.5H22.5V32C22.5 32.83 21.83 33.5 21 33.5H15C14.17 33.5 13.5 32.83 13.5 32V22.5H4C3.17 22.5 2.5 21.83 2.5 21V15C2.5 14.17 3.17 13.5 4 13.5H13.5V4Z"
          fill="url(#navLogoGrad)"
          filter="url(#navLogoGlow)"
        />
        {/* Luminous Inner Highlight */}
        <path
          d="M15 4.5H21V14.5H31V21.5H21V31.5H15V21.5H5V14.5H15V4.5Z"
          fill="none"
          stroke="rgba(255, 255, 255, 0.4)"
          strokeWidth="0.8"
        />
        {/* White ECG / pulse line through cross */}
        <path
          d="M6 18H13L15 13.5L18.5 22.5L20.5 18H30"
          stroke="#ffffff"
          strokeWidth="2.2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

function HomeIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  );
}

function DocumentPlusIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="12" y1="18" x2="12" y2="12" />
      <line x1="9" y1="15" x2="15" y2="15" />
    </svg>
  );
}

function HistoryClockIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <polyline points="12 7 12 12 15 15" />
    </svg>
  );
}

function ChatMessageIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  );
}

function ChevronDownIcon({ size = 13, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function BellIcon({ size = 18, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  );
}

function UserIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}

function SettingsGearIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}

function LogoutIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [profileOpen, setProfileOpen] = useState(false);
  const profileRef = useRef(null);

  // Close profile dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e) {
      if (profileRef.current && !profileRef.current.contains(e.target)) {
        setProfileOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = () => {
    setProfileOpen(false);
    logout();
    navigate('/login');
  };

  const displayName = user?.name || 'Sujan Pradhan';
  const initials = displayName
    .split(' ')
    .filter(Boolean)
    .map((w) => w[0])
    .slice(0, 2)
    .join('')
    .toUpperCase() || 'SP';

  return (
    <header className="navbar">
      {/* Left: Professional Large Glowing Medical Cross Brand */}
      <div className="navbar-brand-group" onClick={() => navigate('/dashboard')}>
        <MediSenseLogo />
        <div className="navbar-brand-text">
          <span className="navbar-brand-title">MediSense AI</span>
          <span className="navbar-brand-subtitle">Your Personal Medical Assistant</span>
        </div>
      </div>

      {/* Center: Main Navigation Links — NO "More" Item */}
      <nav className="navbar-links" aria-label="Main Navigation">
        <NavLink
          to="/dashboard"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <HomeIcon size={16} />
          <span>Dashboard</span>
        </NavLink>

        <NavLink
          to="/assessment/new"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <DocumentPlusIcon size={16} />
          <span>New Assessment</span>
        </NavLink>

        <NavLink
          to="/history"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <HistoryClockIcon size={16} />
          <span>History</span>
        </NavLink>

        <NavLink
          to="/assistant"
          className={({ isActive }) => `navbar-link ${isActive ? 'active' : ''}`}
        >
          <ChatMessageIcon size={16} />
          <span>AI Assistant</span>
        </NavLink>
      </nav>

      {/* Right: Notifications & User Profile with Profile Dropdown */}
      <div className="navbar-right">
        <button
          type="button"
          className="navbar-notif-btn"
          aria-label="Notifications"
          onClick={() => navigate('/history')}
        >
          <BellIcon size={18} />
          <span className="navbar-notif-dot" />
        </button>

        {/* Profile Container with Click-To-Open Dropdown */}
        <div className="navbar-user-menu" ref={profileRef}>
          <button
            type="button"
            className="navbar-user-profile"
            onClick={() => setProfileOpen((prev) => !prev)}
            aria-expanded={profileOpen}
            aria-haspopup="true"
            title="Account Menu"
          >
            <div className="navbar-user-avatar">{initials}</div>
            <span className="navbar-user-name">{displayName}</span>
            <span className="navbar-user-arrow">
              <ChevronDownIcon size={13} />
            </span>
          </button>

          {/* Profile Dropdown: Contains ONLY Profile, Settings, and Logout */}
          {profileOpen ? (
            <div className="profile-dropdown" role="menu">
              <NavLink to="/profile" onClick={() => setProfileOpen(false)}>
                <UserIcon size={16} color="#0ea5e9" />
                <span>Profile</span>
              </NavLink>

              <NavLink to="/settings" onClick={() => setProfileOpen(false)}>
                <SettingsGearIcon size={16} color="#0ea5e9" />
                <span>Settings</span>
              </NavLink>

              <div className="profile-dropdown-divider" />

              <button type="button" className="logout-item" onClick={handleLogout}>
                <LogoutIcon size={16} color="#ef4444" />
                <span>Logout</span>
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
