import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import DoctorCard from '../components/DoctorCard';
import ErrorMessage from '../components/ErrorMessage';
import { searchDoctors } from '../services/providerService';
import { useAuth } from '../hooks/useAuth';

export default function Doctors() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    state: user?.state || '',
    district: user?.district || '',
    city: user?.city || '',
    pin: '',
    specialization: '',
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onSearch = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.state.trim() || !form.district.trim()) {
      setError('State and District are required for provider search.');
      return;
    }
    setLoading(true);
    try {
      const params = {
        state: form.state.trim(),
        district: form.district.trim(),
      };
      if (form.city.trim()) params.city = form.city.trim();
      if (form.pin.trim()) params.pin = form.pin.trim();
      if (form.specialization.trim()) params.specialization = form.specialization.trim();
      const data = await searchDoctors(params);
      setResult(data);
      if (data.expanded) {
        setError('Search scope was expanded. Far-away providers are labeled explicitly when expansion occurs.');
      }
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Doctors</h1>
      <p className="muted">
        Provider data comes only from legitimate/authorized sources or imported verified records. Nothing is fabricated.
      </p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      <form className="stack-form search-form" onSubmit={onSearch}>
        <label>State<input value={form.state} onChange={set('state')} required /></label>
        <label>District<input value={form.district} onChange={set('district')} required /></label>
        <label>City/Town (optional)<input value={form.city} onChange={set('city')} /></label>
        <label>PIN (optional)<input value={form.pin} onChange={set('pin')} /></label>
        <label>Specialization (optional)<input value={form.specialization} onChange={set('specialization')} /></label>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Searching healthcare providers...' : 'Search'}
        </button>
      </form>

      {result ? (
        <section className="section">
          <p>{result.message}</p>
          {!result.doctors?.length ? (
            <p className="empty-state">Healthcare provider information is currently unavailable.</p>
          ) : (
            <div className="card-grid">
              {result.doctors.map((d) => (
                <div key={d.doctor_id} onClick={() => navigate(`/doctors/${d.doctor_id}`)} role="button" tabIndex={0}>
                  <DoctorCard doctor={d} />
                </div>
              ))}
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
