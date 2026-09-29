import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AssessmentCard from '../components/AssessmentCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getDashboardSummary } from '../services/assessmentService';
import { useAuth } from '../hooks/useAuth';

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const summary = await getDashboardSummary();
        if (active) setData(summary);
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, []);

  if (loading) return <LoadingScreen message="Loading dashboard..." />;

  return (
    <div className="page">
      <div className="page-header">
        <div>
          <h1>Welcome, {data?.user_name || user?.name || 'User'}</h1>
          <p className="muted">Your health assessment overview</p>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => navigate('/assessment/new')}>
          + New Assessment
        </button>
      </div>

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      <div className="stat-grid">
        <div className="stat-tile"><span>Total assessments</span><strong>{data?.total_assessments ?? 0}</strong></div>
        <div className="stat-tile"><span>X-rays processed</span><strong>{data?.xrays_processed ?? 0}</strong></div>
        <div className="stat-tile"><span>Medical reports</span><strong>{data?.reports_processed ?? 0}</strong></div>
        <div className="stat-tile"><span>Appointments</span><strong>{data?.appointments ?? 0}</strong></div>
      </div>

      <section className="section">
        <h2>Quick actions</h2>
        <div className="action-row">
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/assessment/new')}>Start New Assessment</button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/history')}>View History</button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>AI Assistant</button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/doctors')}>Doctors</button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/facilities')}>Facilities</button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/appointments')}>Appointments</button>
        </div>
      </section>

      <section className="section">
        <h2>Recent assessments</h2>
        {!data?.recent_assessments?.length ? (
          <p className="empty-state">No assessments yet. Start a new assessment to begin.</p>
        ) : (
          <div className="card-grid">
            {data.recent_assessments.map((a) => (
              <AssessmentCard
                key={a.assessment_id}
                assessment={a}
                onView={() => navigate(`/assessment/${a.assessment_id}`)}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
