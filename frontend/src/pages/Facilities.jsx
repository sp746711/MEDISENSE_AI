import { useState } from 'react';
import FacilityCard from '../components/FacilityCard';
import ErrorMessage from '../components/ErrorMessage';
import { searchFacilities } from '../services/facilityService';
import { useAuth } from '../hooks/useAuth';

export default function Facilities() {
  const { user } = useAuth();
  const [form, setForm] = useState({
    state: user?.state || '',
    district: user?.district || '',
    city: user?.city || '',
    emergency_only: false,
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const set = (key) => (e) => {
    const value = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
    setForm((f) => ({ ...f, [key]: value }));
  };

  const onSearch = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.state.trim() || !form.district.trim()) {
      setError('State and District are required.');
      return;
    }
    setLoading(true);
    try {
      const params = {
        state: form.state.trim(),
        district: form.district.trim(),
        emergency_only: String(form.emergency_only),
      };
      if (form.city.trim()) params.city = form.city.trim();
      setResult(await searchFacilities(params));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Facilities</h1>
      <p className="muted">Facility data is shown only from legitimate sources. Contact information is unavailable through this system unless provided by the source.</p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      <form className="stack-form search-form" onSubmit={onSearch}>
        <label>State<input value={form.state} onChange={set('state')} required /></label>
        <label>District<input value={form.district} onChange={set('district')} required /></label>
        <label>City/Town (optional)<input value={form.city} onChange={set('city')} /></label>
        <label className="checkbox-row">
          <input type="checkbox" checked={form.emergency_only} onChange={set('emergency_only')} />
          Emergency facilities only
        </label>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>
      {result ? (
        <section className="section">
          <p>{result.message}</p>
          {!result.facilities?.length ? (
            <p className="empty-state">Healthcare facility information is currently unavailable.</p>
          ) : (
            <div className="card-grid">
              {result.facilities.map((f) => (
                <FacilityCard key={f.facility_id} facility={f} />
              ))}
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
