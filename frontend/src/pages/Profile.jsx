import { useEffect, useState } from 'react';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getProfile, updateProfile } from '../services/userService';
import { useAuth } from '../hooks/useAuth';

export default function Profile() {
  const { setUser } = useAuth();
  const [form, setForm] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const me = await getProfile();
        setForm({
          name: me.name || '',
          email: me.email || '',
          state: me.state || '',
          district: me.district || '',
          city: me.city || '',
        });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onSave = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    setSaving(true);
    try {
      const updated = await updateProfile({
        name: form.name.trim(),
        state: form.state.trim(),
        district: form.district.trim(),
        city: form.city.trim() || null,
      });
      setUser(updated);
      setSuccess('Profile updated.');
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  if (loading || !form) return <LoadingScreen message="Loading profile..." />;

  return (
    <div className="page narrow">
      <h1>Profile</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {success ? <p className="info-banner">{success}</p> : null}
      <form className="stack-form" onSubmit={onSave}>
        <label>Name<input value={form.name} onChange={set('name')} required /></label>
        <label>Email<input value={form.email} disabled /></label>
        <label>State<input value={form.state} onChange={set('state')} required /></label>
        <label>District<input value={form.district} onChange={set('district')} required /></label>
        <label>City/Town<input value={form.city} onChange={set('city')} /></label>
        <button type="submit" className="btn btn-primary" disabled={saving}>
          {saving ? 'Saving...' : 'Save changes'}
        </button>
      </form>
    </div>
  );
}
