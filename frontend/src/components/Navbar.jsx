import { NavLink, useNavigate } from 'react-router-dom';
import { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { APP_NAME } from '../utils/constants';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [moreOpen, setMoreOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <header className="navbar">
      <div className="navbar-brand" onClick={() => navigate('/dashboard')}>
        {APP_NAME}
      </div>
      <nav className="navbar-links" aria-label="Main">
        <NavLink to="/dashboard">Dashboard</NavLink>
        <NavLink to="/assessment/new">New Assessment</NavLink>
        <NavLink to="/history">History</NavLink>
        <NavLink to="/assistant">AI Assistant</NavLink>
        <div className="more-menu">
          <button
            type="button"
            className="more-toggle"
            aria-expanded={moreOpen}
            onClick={() => setMoreOpen((v) => !v)}
          >
            More
          </button>
          {moreOpen ? (
            <div className="more-dropdown" role="menu">
              <NavLink to="/doctors" onClick={() => setMoreOpen(false)}>Doctors</NavLink>
              <NavLink to="/facilities" onClick={() => setMoreOpen(false)}>Facilities</NavLink>
              <NavLink to="/medical-shops" onClick={() => setMoreOpen(false)}>Medical Shops</NavLink>
              <NavLink to="/appointments" onClick={() => setMoreOpen(false)}>Appointments</NavLink>
              <NavLink to="/profile" onClick={() => setMoreOpen(false)}>Profile</NavLink>
              <NavLink to="/settings" onClick={() => setMoreOpen(false)}>Settings</NavLink>
              <button type="button" onClick={handleLogout}>Logout</button>
            </div>
          ) : null}
        </div>
      </nav>
      <div className="navbar-user">{user?.name || 'User'}</div>
    </header>
  );
}
