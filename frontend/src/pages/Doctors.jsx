import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import DoctorCard from '../components/DoctorCard';
import ErrorMessage from '../components/ErrorMessage';
import { searchDoctors } from '../services/providerService';

export default function Doctors() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  const urlSpecialization = searchParams.get('specialization') || '';
  const assessmentId = searchParams.get('assessment_id') || '';

  const [form, setForm] = useState({
    state: '',
    district: '',
    city: '',
    pin: '',
    specialization: urlSpecialization,
  });

  const [page, setPage] = useState(1);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Keep specialization in sync if URL query parameter changes
  useEffect(() => {
    if (urlSpecialization) {
      setForm((f) => ({ ...f, specialization: urlSpecialization }));
    }
  }, [urlSpecialization]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const executeSearch = useCallback(async (targetPage = 1) => {
    setError('');

    if (!form.state.trim() || !form.district.trim()) {
      setError('State and District are required for manual provider search.');
      return;
    }

    setLoading(true);
    try {
      const params = {
        page: targetPage,
        page_size: 10,
        state: form.state.trim(),
        district: form.district.trim(),
      };

      if (form.specialization.trim()) {
        params.specialization = form.specialization.trim();
      }
      if (form.city.trim()) {
        params.city = form.city.trim();
      }
      if (form.pin.trim()) {
        params.pin = form.pin.trim();
      }
      if (assessmentId) {
        params.assessment_id = assessmentId;
      }

      const data = await searchDoctors(params);
      setResult(data);
      setPage(targetPage);
    } catch (err) {
      setError(err.message || 'Unable to retrieve healthcare providers.');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [form, assessmentId]);

  const onSearchSubmit = (e) => {
    e.preventDefault();
    executeSearch(1);
  };

  const handlePageChange = (newPage) => {
    if (newPage < 1) return;
    executeSearch(newPage);
  };

  return (
    <div className="page">
      <h1>Doctors</h1>
      <p className="muted">
        Provider data comes only from legitimate/authorized sources or verified registries. Nothing is fabricated.
      </p>

      {urlSpecialization ? (
        <div
          style={{
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            color: '#1e40af',
            padding: '10px 14px',
            borderRadius: '8px',
            marginBottom: '16px',
            fontSize: '14px',
          }}
        >
          Suggested Specialty from Assessment: <strong>{urlSpecialization}</strong>
          {assessmentId ? ` (Assessment ID: ${assessmentId})` : ''}
          <div style={{ fontSize: '12px', marginTop: '2px', color: '#3b82f6' }}>
            You can modify the specialty filter below if needed.
          </div>
        </div>
      ) : null}

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {/* Search Form */}
      <form className="stack-form search-form" onSubmit={onSearchSubmit}>
        <label>
          Specialization (optional)
          <input
            value={form.specialization}
            onChange={set('specialization')}
            placeholder="e.g. Cardiology, Dermatology, General Physician"
          />
        </label>

        <label>
          State
          <input value={form.state} onChange={set('state')} required placeholder="e.g. Maharashtra" />
        </label>
        <label>
          District
          <input value={form.district} onChange={set('district')} required placeholder="e.g. Mumbai" />
        </label>
        <label>
          City/Town (optional)
          <input value={form.city} onChange={set('city')} placeholder="e.g. Andheri" />
        </label>
        <label>
          PIN (optional)
          <input value={form.pin} onChange={set('pin')} placeholder="e.g. 400053" />
        </label>

        <button type="submit" className="btn btn-primary" disabled={loading}>
          {loading ? 'Searching healthcare providers...' : 'Search'}
        </button>
      </form>

      {/* Search Results */}
      {result ? (
        <section className="section" style={{ marginTop: '24px' }}>
          {result.expanded && result.expansion_message ? (
            <div
              style={{
                backgroundColor: '#fffbeb',
                border: '1px solid #fde68a',
                color: '#92400e',
                padding: '12px 16px',
                borderRadius: '8px',
                marginBottom: '16px',
                fontSize: '14px',
              }}
            >
              <strong>Search Notice:</strong> {result.expansion_message}
            </div>
          ) : null}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h2 style={{ margin: 0, fontSize: '20px' }}>Matching Providers</h2>
            {result.total_count != null ? (
              <span className="muted" style={{ fontSize: '14px' }}>
                Total: {result.total_count}
              </span>
            ) : null}
          </div>

          <p style={{ margin: '0 0 16px 0', color: '#475569' }}>{result.message}</p>

          {!result.doctors?.length ? (
            <div
              className="empty-state"
              style={{
                padding: '32px 16px',
                textAlign: 'center',
                backgroundColor: '#f8fafc',
                borderRadius: '12px',
                border: '1px dashed #cbd5e1',
                color: '#64748b',
              }}
            >
              <p style={{ margin: 0, fontSize: '15px' }}>
                {result.message || 'Healthcare provider information is currently unavailable because no authorized provider data source is configured.'}
              </p>
            </div>
          ) : (
            <>
              <div className="card-grid">
                {result.doctors.map((d, idx) => {
                  const docKey = d.doctor_id || d.external_id || idx;
                  const detailId = d.doctor_id || d.external_id;
                  return (
                    <div
                      key={docKey}
                      onClick={() => {
                        if (detailId) {
                          const target = assessmentId
                            ? `/doctors/${detailId}?assessment_id=${encodeURIComponent(assessmentId)}`
                            : `/doctors/${detailId}`;
                          navigate(target);
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      style={{ cursor: detailId ? 'pointer' : 'default' }}
                    >
                      <DoctorCard doctor={d} />
                    </div>
                  );
                })}
              </div>

              {/* Pagination UI: [ Previous ] [ Page N ] [ Next ] */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '16px',
                  marginTop: '28px',
                }}
              >
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => handlePageChange(page - 1)}
                  disabled={page <= 1 || !result.has_previous || loading}
                >
                  Previous
                </button>
                <span style={{ fontWeight: 600, fontSize: '14px', color: '#334155' }}>
                  Page {result.page || page}
                </span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => handlePageChange(page + 1)}
                  disabled={!result.has_next || loading}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </section>
      ) : null}
    </div>
  );
}
