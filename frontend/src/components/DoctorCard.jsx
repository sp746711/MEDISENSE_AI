export default function DoctorCard({ doctor }) {
  if (!doctor) return null;
  return (
    <article className="entity-card">
      <h3>{doctor.name}</h3>
      <p>{doctor.specialization}</p>
      {doctor.qualification ? <p className="muted">{doctor.qualification}</p> : null}
      {doctor.facility ? <p>{doctor.facility}</p> : null}
      <p className="muted">
        {[doctor.city, doctor.district, doctor.state].filter(Boolean).join(', ')}
      </p>
      <p className="source-tag">Source: {doctor.source || 'unavailable'}</p>
    </article>
  );
}
