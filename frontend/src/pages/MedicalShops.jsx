import { useState } from 'react';
import MedicalShopCard from '../components/MedicalShopCard';
import ErrorMessage from '../components/ErrorMessage';
import { searchMedicalShops } from '../services/medicalShopService';
import { useAuth } from '../hooks/useAuth';

export default function MedicalShops() {
  const { user } = useAuth();
  const [form, setForm] = useState({
    state: user?.state || '',
    district: user?.district || '',
    city: user?.city || '',
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onSearch = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.state.trim() || !form.district.trim()) {
      setError('State and District are required.');
      return;
    }
    setLoading(true);
    try {
      const params = { state: form.state.trim(), district: form.district.trim() };
      if (form.city.trim()) params.city = form.city.trim();
      setResult(await searchMedicalShops(params));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>Medical Shops</h1>
      <p className="muted">
        Used mainly for the Mild pathway. This page does not prescribe medicines or dosages.
      </p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      <form className="stack-form search-form" onSubmit={onSearch}>
        <label>State<input value={form.state} onChange={set('state')} required /></label>
        <label>District<input value={form.district} onChange={set('district')} required /></label>
        <label>City/Town (optional)<input value={form.city} onChange={set('city')} /></label>
        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>
      {result ? (
        <section className="section">
          <p>{result.message || result.disclaimer}</p>
          {!result.medical_shops?.length ? (
            <p className="empty-state">Medical shop information is currently unavailable for this search.</p>
          ) : (
            <div className="card-grid">
              {result.medical_shops.map((s) => (
                <MedicalShopCard key={s.shop_id} shop={s} />
              ))}
            </div>
          )}
        </section>
      ) : null}
    </div>
  );
}
