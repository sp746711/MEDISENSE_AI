import { useState } from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { useAuth } from '../hooks/useAuth';
import { isValidEmail, validatePassword } from '../utils/validation';
import { APP_NAME, MEDICAL_DISCLAIMER } from '../utils/constants';

export default function Login() {
  const { login, isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!loading && isAuthenticated) {
    return <Navigate to={location.state?.from || '/dashboard'} replace />;
  }

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!isValidEmail(email)) return setError('Enter a valid email.');
    const pwErr = validatePassword(password);
    if (pwErr) return setError(pwErr);

    setSubmitting(true);
    try {
      await login(email.trim(), password);
      navigate(location.state?.from || '/dashboard', { replace: true });
    } catch (err) {
      setError(err.message || 'Sign in failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>{APP_NAME}</h1>
        <p className="muted">Sign in to continue</p>
        <ErrorMessage message={error} onDismiss={() => setError('')} />
        <form onSubmit={onSubmit} className="stack-form">
          <label>
            Email
            <input
              type="email"
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </label>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? 'Signing in...' : 'Sign In'}
          </button>
        </form>
        <p className="auth-links">
          <button type="button" className="linkish" onClick={() => setError('Password reset is not configured yet.')}>
            Forgot Password
          </button>
          <span>·</span>
          <Link to="/signup">Sign Up</Link>
        </p>
        <p className="disclaimer-inline">{MEDICAL_DISCLAIMER}</p>
      </div>
    </div>
  );
}
