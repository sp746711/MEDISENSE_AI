import { useState } from 'react';
import { Link, Navigate, useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { useAuth } from '../hooks/useAuth';
import { isValidEmail, validatePassword, validateRequired } from '../utils/validation';
import { APP_NAME, MEDICAL_DISCLAIMER } from '../utils/constants';

export default function Signup() {
  const { register, isAuthenticated, loading } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: '',
    email: '',
    password: '',
    confirm_password: '',
    state: '',
    district: '',
    city: '',
  });
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!loading && isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    for (const [key, label] of [
      ['name', 'Name'],
      ['email', 'Email'],
      ['password', 'Password'],
      ['confirm_password', 'Confirm password'],
      ['state', 'State'],
      ['district', 'District'],
    ]) {
      const err = validateRequired(form[key], label);
      if (err) return setError(err);
    }
    if (!isValidEmail(form.email)) return setError('Enter a valid email.');
    const pwErr = validatePassword(form.password);
    if (pwErr) return setError(pwErr);
    if (form.password !== form.confirm_password) {
      return setError('Passwords do not match.');
    }

    setSubmitting(true);
    try {
      await register({
        ...form,
        name: form.name.trim(),
        email: form.email.trim(),
        state: form.state.trim(),
        district: form.district.trim(),
        city: form.city.trim() || null,
      });
      navigate('/dashboard', { replace: true });
    } catch (err) {
      setError(err.message || 'Registration failed.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <h1>Create account</h1>
        <p className="muted">{APP_NAME}</p>
        <ErrorMessage message={error} onDismiss={() => setError('')} />
        <form onSubmit={onSubmit} className="stack-form">
          <label>Name<input value={form.name} onChange={set('name')} required /></label>
          <label>Email<input type="email" value={form.email} onChange={set('email')} required /></label>
          <label>Password<input type="password" value={form.password} onChange={set('password')} required /></label>
          <label>
            Confirm Password
            <input type="password" value={form.confirm_password} onChange={set('confirm_password')} required />
          </label>
          <label>State<input value={form.state} onChange={set('state')} required /></label>
          <label>District<input value={form.district} onChange={set('district')} required /></label>
          <label>
            City/Town (optional)
            <input value={form.city} onChange={set('city')} />
          </label>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>
        <p className="auth-links">
          Already have an account? <Link to="/login">Sign In</Link>
        </p>
        <p className="disclaimer-inline">{MEDICAL_DISCLAIMER}</p>
      </div>
    </div>
  );
}
