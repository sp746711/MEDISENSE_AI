import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { createAppointment } from '../services/appointmentService';

export default function BookAppointment() {
  const navigate = useNavigate();
  const location = useLocation();
  const [form, setForm] = useState({
    doctor_id: location.state?.doctorId || '',
    appointment_date: '',
    appointment_time: '10:00',
    appointment_type: 'demo',
  });
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onSubmit = async (e) => {
    e.preventDefault();
    setError('');
    if (!form.doctor_id.trim()) {
      setError(
        'A doctor_id from a legitimate provider record is required. Search Doctors first. Fake providers are not allowed.',
      );
      return;
    }
    setSubmitting(true);
    try {
      const data = await createAppointment(form);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page narrow">
      <h1>Book Demo Appointment</h1>
      <p className="disclaimer-inline">
        This creates an in-application demo record only. It does not confirm an actual appointment with a healthcare provider.
      </p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {result ? (
        <div className="info-banner">
          <p>Status: {result.status}</p>
          <p>{result.disclaimer}</p>
          <button type="button" className="btn btn-primary" onClick={() => navigate('/appointments')}>
            View Appointments
          </button>
        </div>
      ) : (
        <form className="stack-form" onSubmit={onSubmit}>
          <label>
            Doctor ID
            <input value={form.doctor_id} onChange={set('doctor_id')} required placeholder="From Doctors search" />
          </label>
          <label>
            Date
            <input type="date" value={form.appointment_date} onChange={set('appointment_date')} required />
          </label>
          <label>
            Time
            <input type="time" value={form.appointment_time} onChange={set('appointment_time')} required />
          </label>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting ? 'Confirming...' : 'Confirm Demo Appointment'}
          </button>
        </form>
      )}
    </div>
  );
}
