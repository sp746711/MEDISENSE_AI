import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AssessmentCard from '../components/AssessmentCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getHistory } from '../services/historyService';

export default function History() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const data = await getHistory();
        if (active) setItems(data.history || []);
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

  if (loading) return <LoadingScreen message="Loading history..." />;

  return (
    <div className="page">
      <h1>History</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {!items.length ? (
        <p className="empty-state">No assessments in history yet.</p>
      ) : (
        <div className="card-grid">
          {items.map((a) => (
            <AssessmentCard
              key={a.assessment_id}
              assessment={a}
              onView={() => navigate(`/assessment/${a.assessment_id}`)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
