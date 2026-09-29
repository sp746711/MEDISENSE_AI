export default function AppointmentCard({ appointment, onCancel }) {
  if (!appointment) return null;
  return (
    <article className="entity-card">
      <h3>Demo Appointment</h3>
      <p>
        {appointment.appointment_date} at {appointment.appointment_time}
      </p>
      <p>Status: {appointment.status}</p>
      <p className="disclaimer-inline">{appointment.disclaimer}</p>
      {appointment.status !== 'CANCELLED' && onCancel ? (
        <button type="button" className="btn btn-secondary" onClick={() => onCancel(appointment)}>
          Cancel
        </button>
      ) : null}
    </article>
  );
}
