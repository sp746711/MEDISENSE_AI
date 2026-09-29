import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import ConfirmationModal from '../components/ConfirmationModal';
import { changePassword } from '../services/userService';
import { useAuth } from '../hooks/useAuth';
import { validatePassword } from '../utils/validation';

export default function Settings() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [passwords, setPasswords] = useState({
    current_password: '',
    new_password: '',
    confirm_password: '',
  });
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [confirmLogout, setConfirmLogout] = useState(false);

  const set = (key) => (e) => setPasswords((p) => ({ ...p, [key]: e.target.value }));

  const onChangePassword = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    const pwErr = validatePassword(passwords.new_password);
    if (pwErr) return setError(pwErr);
    if (passwords.new_password !== passwords.confirm_password) {
      return setError('New passwords do not match.');
    }
    try {
      await changePassword(passwords);
      setSuccess('Password updated.');
      setPasswords({ current_password: '', new_password: '', confirm_password: '' });
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="page narrow">
      <h1>Settings</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {success ? <p className="info-banner">{success}</p> : null}

      <section className="section">
        <h2>Password management</h2>
        <form className="stack-form" onSubmit={onChangePassword}>
          <label>Current password<input type="password" value={passwords.current_password} onChange={set('current_password')} required /></label>
          <label>New password<input type="password" value={passwords.new_password} onChange={set('new_password')} required /></label>
          <label>Confirm new password<input type="password" value={passwords.confirm_password} onChange={set('confirm_password')} required /></label>
          <button type="submit" className="btn btn-primary">Update password</button>
        </form>
      </section>

      <section className="section">
        <h2>Privacy / data</h2>
        <p className="muted">
          Medical uploads are stored privately and scoped to your account. API keys are never exposed in the frontend.
        </p>
      </section>

      <section className="section">
        <h2>AI assistant</h2>
        <p className="muted">
          The assistant explains assessments using limited context and trusted retrieval when configured. It cannot override triage.
        </p>
      </section>

      <section className="section">
        <h2>Account</h2>
        <button type="button" className="btn btn-secondary" onClick={() => setConfirmLogout(true)}>
          Logout
        </button>
      </section>

      <ConfirmationModal
        open={confirmLogout}
        title="Logout?"
        message="You will need to sign in again to access your assessments."
        confirmLabel="Logout"
        onCancel={() => setConfirmLogout(false)}
        onConfirm={() => {
          logout();
          navigate('/login');
        }}
      />
    </div>
  );
}
