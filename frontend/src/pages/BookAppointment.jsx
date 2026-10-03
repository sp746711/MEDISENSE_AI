import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { createAppointment } from '../services/appointmentService';

export default function BookAppointment() {
  const navigate = useNavigate();
  const location = useLocation();

  const doctorId = location.state?.doctorId || '';
  const doctorName = location.state?.doctorName || '';
  const doctorSpecialization = location.state?.doctorSpecialization || '';
  const doctorFacility = location.state?.doctorFacility || '';

  const [form, setForm] = useState({
    doctor_id: doctorId,
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

    if (!form.doctor_id) {
      setError('Please select a legitimate healthcare provider from the Doctors directory first.');
      return;
    }
    if (!form.appointment_date) {
      setError('Please select a date for the demo appointment.');
      return;
    }

    setSubmitting(true);
    try {
      const data = await createAppointment(form);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to schedule demo appointment.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page narrow">
      <h1>Book Demo Appointment</h1>
      <p className="disclaimer-inline">
        Demo appointment confirmed. This does not confirm an actual appointment with the healthcare provider.
      </p>

      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {result ? (
        <div
          className="info-banner"
          style={{
            backgroundColor: '#ecfdf5',
            border: '1px solid #a7f3d0',
            color: '#065f46',
            padding: '20px',
            borderRadius: '12px',
            marginTop: '16px',
          }}
        >
          <h3 style={{ margin: '0 0 8px 0', fontSize: '18px', color: '#047857' }}>
            Demo Appointment Confirmed
          </h3>
          <p style={{ margin: '0 0 8px 0' }}>
            <strong>Status:</strong> {result.status || 'DEMO_CONFIRMED'}
          </p>
          <p style={{ margin: '0 0 16px 0', fontSize: '13px', color: '#065f46' }}>
            Demo appointment confirmed. This does not confirm an actual appointment with the healthcare provider. No real external booking was submitted.
          </p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button type="button" className="btn btn-primary" onClick={() => navigate('/appointments')}>
              View My Appointments
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
              Dashboard
            </button>
          </div>
        </div>
      ) : (
        <>
          {doctorId ? (
            <div
              style={{
                backgroundColor: '#f8fafc',
                border: '1px solid #e2e8f0',
                borderRadius: '10px',
                padding: '14px 18px',
                marginBottom: '20px',
              }}
            >
              <span style={{ fontSize: '12px', textTransform: 'uppercase', color: '#64748b', fontWeight: 600 }}>
                Selected Healthcare Provider
              </span>
              <h3 style={{ margin: '4px 0 2px 0', fontSize: '17px', color: '#0f172a' }}>
                {doctorName || 'Selected Doctor'}
              </h3>
              {doctorSpecialization ? (
                <p style={{ margin: '0 0 2px 0', fontSize: '14px', color: '#2563eb' }}>
                  {doctorSpecialization}
                </p>
              ) : null}
              {doctorFacility ? (
                <p style={{ margin: 0, fontSize: '13px', color: '#64748b' }}>
                  {doctorFacility}
                </p>
              ) : null}
            </div>
          ) : (
            <div
              style={{
                backgroundColor: '#fffbeb',
                border: '1px solid #fde68a',
                color: '#92400e',
                borderRadius: '10px',
                padding: '14px 18px',
                marginBottom: '20px',
                fontSize: '14px',
              }}
            >
              <p style={{ margin: '0 0 10px 0' }}>
                No doctor has been selected yet. Please select a verified doctor first.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => navigate('/doctors')}
                style={{ fontSize: '13px', padding: '6px 14px' }}
              >
                Find Doctors
              </button>
            </div>
          )}

          <form className="stack-form" onSubmit={onSubmit}>
            <label>
              Date
              <input
                type="date"
                value={form.appointment_date}
                onChange={set('appointment_date')}
                required
                min={new Date().toISOString().split('T')[0]}
              />
            </label>
            <label>
              Time
              <input
                type="time"
                value={form.appointment_time}
                onChange={set('appointment_time')}
                required
              />
            </label>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={submitting || !doctorId}
            >
              {submitting ? 'Confirming...' : 'Confirm Demo Appointment'}
            </button>
          </form>
        </>
      )}
    </div>
  );
}
