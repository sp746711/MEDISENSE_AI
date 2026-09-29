import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AppointmentCard from '../components/AppointmentCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { cancelAppointment, listAppointments } from '../services/appointmentService';

export default function Appointments() {
  const navigate = useNavigate();
  const [items, setItems] = useState([]);
  const [disclaimer, setDisclaimer] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    try {
      const data = await listAppointments();
      setItems(data.appointments || []);
      setDisclaimer(data.disclaimer || '');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onCancel = async (appt) => {
    try {
      await cancelAppointment(appt.appointment_id);
      await load();
    } catch (err) {
      setError(err.message);
    }
  };

  if (loading) return <LoadingScreen message="Loading appointments..." />;

  return (
    <div className="page">
      <div className="page-header">
        <h1>Appointments</h1>
        <button type="button" className="btn btn-primary" onClick={() => navigate('/appointments/new')}>
          Book Demo Appointment
        </button>
      </div>
      <p className="disclaimer-inline">
        {disclaimer ||
          'Demo appointment confirmed. This does not confirm an actual appointment with the healthcare provider.'}
      </p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {!items.length ? (
        <p className="empty-state">No demo appointments yet.</p>
      ) : (
        <div className="card-grid">
          {items.map((a) => (
            <AppointmentCard key={a.appointment_id} appointment={a} onCancel={onCancel} />
          ))}
        </div>
      )}
    </div>
  );
}
