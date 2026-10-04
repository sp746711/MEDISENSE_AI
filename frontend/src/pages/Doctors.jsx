import { useState, useEffect, useCallback, useMemo } from 'react';
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
  const [entityFilter, setEntityFilter] = useState('all'); // all, doctor, clinic, medical_center, hospital
  const [sortBy, setSortBy] = useState('relevance'); // relevance, distance

  // Keep specialization in sync if URL query parameter changes
  useEffect(() => {
    if (urlSpecialization) {
      setForm((f) => ({ ...f, specialization: urlSpecialization }));
    }
  }, [urlSpecialization]);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const executeSearch = useCallback(
    async (targetPage = 1, overrideSpecialty = null) => {
      setError('');

      const activeSpecialty = overrideSpecialty != null ? overrideSpecialty : form.specialization;

      // In manual search mode (no assessmentId), require at least state and district
      if (!assessmentId && (!form.state.trim() || !form.district.trim())) {
        setError('State and District are required for manual provider search.');
        return;
      }

      setLoading(true);
      try {
        const params = {
          page: targetPage,
          page_size: 10,
        };

        if (activeSpecialty && activeSpecialty.trim()) {
          params.specialization = activeSpecialty.trim();
        }

        if (form.state.trim()) {
          params.state = form.state.trim();
        }
        if (form.district.trim()) {
          params.district = form.district.trim();
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
    },
    [form, assessmentId]
  );

  // Auto-search on mount when navigated from Assessment with assessment_id
  useEffect(() => {
    if (assessmentId) {
      executeSearch(1, urlSpecialization);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assessmentId]);

  const onSearchSubmit = (e) => {
    e.preventDefault();
    executeSearch(1);
  };

  const handlePageChange = (newPage) => {
    if (newPage < 1) return;
    executeSearch(newPage);
  };

  // Filter and sort providers client-side if needed (Requirement 20)
  const filteredDoctors = useMemo(() => {
    if (!result?.doctors) return [];
    let list = [...result.doctors];

    if (entityFilter !== 'all') {
      list = list.filter((d) => (d.entity_type || 'doctor').toLowerCase() === entityFilter.toLowerCase());
    }

    if (sortBy === 'distance') {
      list.sort((a, b) => {
        if (a.distance_km == null && b.distance_km == null) return 0;
        if (a.distance_km == null) return 1;
        if (b.distance_km == null) return -1;
        return a.distance_km - b.distance_km;
      });
    }

    return list;
  }, [result?.doctors, entityFilter, sortBy]);

  return (
    <div className="page">
      <h1>Healthcare Provider Discovery</h1>
      <p className="muted">
        Provider discovery powered by real geographic listings via Google Places API (New) or verified registries.
        Zero synthetic or fabricated data.
      </p>

      {urlSpecialization ? (
        <div
          style={{
            backgroundColor: '#eff6ff',
            border: '1px solid #bfdbfe',
            color: '#1e40af',
            padding: '12px 16px',
            borderRadius: '8px',
            marginBottom: '16px',
            fontSize: '14px',
          }}
        >
          <div style={{ fontWeight: 600, fontSize: '15px' }}>
            Suggested Specialty from Assessment: <span>{urlSpecialization}</span>
          </div>
          {assessmentId ? (
            <div style={{ fontSize: '12px', marginTop: '4px', color: '#2563eb' }}>
              ✓ Assessment location context automatically applied. You can refine filters below if needed.
            </div>
          ) : null}
        </div>
      ) : null}

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {/* Search Form */}
      <form className="stack-form search-form" onSubmit={onSearchSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
          <label>
            Specialization
            <input
              value={form.specialization}
              onChange={set('specialization')}
              placeholder="e.g. General Physician, Cardiologist, Dermatologist"
            />
          </label>

          <label>
            State {assessmentId ? '(optional if from assessment)' : '(required)'}
            <input
              value={form.state}
              onChange={set('state')}
              required={!assessmentId}
              placeholder="e.g. West Bengal, Maharashtra"
            />
          </label>

          <label>
            District {assessmentId ? '(optional if from assessment)' : '(required)'}
            <input
              value={form.district}
              onChange={set('district')}
              required={!assessmentId}
              placeholder="e.g. North 24 Parganas, Mumbai"
            />
          </label>

          <label>
            City/Town (optional)
            <input value={form.city} onChange={set('city')} placeholder="e.g. Barasat, Andheri" />
          </label>

          <label>
            PIN Code (optional)
            <input value={form.pin} onChange={set('pin')} placeholder="e.g. 700124" />
          </label>
        </div>

        <div style={{ marginTop: '8px' }}>
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Searching healthcare providers...' : 'Search Providers'}
          </button>
        </div>
      </form>

      {/* Filter and Sort controls */}
      {result?.doctors?.length ? (
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
            marginTop: '20px',
            padding: '10px 14px',
            backgroundColor: '#f8fafc',
            borderRadius: '8px',
            border: '1px solid #e2e8f0',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#475569' }}>Filter Entity:</span>
            <select
              value={entityFilter}
              onChange={(e) => setEntityFilter(e.target.value)}
              style={{ fontSize: '13px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
            >
              <option value="all">All Types</option>
              <option value="doctor">Doctors Only</option>
              <option value="clinic">Clinics</option>
              <option value="medical_center">Medical Centers</option>
              <option value="hospital">Hospitals</option>
            </select>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: '#475569' }}>Sort By:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              style={{ fontSize: '13px', padding: '4px 8px', borderRadius: '6px', border: '1px solid #cbd5e1' }}
            >
              <option value="relevance">Source Relevance</option>
              <option value="distance">Distance (Nearest)</option>
            </select>
          </div>
        </div>
      ) : null}

      {/* Search Results */}
      {result ? (
        <section className="section" style={{ marginTop: '20px' }}>
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

          {!filteredDoctors.length ? (
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
                {result.message || 'No matching healthcare providers found for this search.'}
              </p>
            </div>
          ) : (
            <>
              <div className="card-grid">
                {filteredDoctors.map((d, idx) => {
                  const docKey = d.doctor_id || d.external_id || idx;
                  const detailId = d.doctor_id || d.external_id;
                  return (
                    <div
                      key={docKey}
                      onClick={() => {
                        if (detailId) {
                          const target = assessmentId
                            ? `/doctors/${encodeURIComponent(detailId)}?assessment_id=${encodeURIComponent(assessmentId)}`
                            : `/doctors/${encodeURIComponent(detailId)}`;
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

              {/* Pagination UI */}
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
