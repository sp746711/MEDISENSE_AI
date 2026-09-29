export default function FacilityCard({ facility }) {
  if (!facility) return null;
  return (
    <article className="entity-card">
      <h3>{facility.name}</h3>
      <p>{facility.type}</p>
      <p className="muted">
        {[facility.city, facility.district, facility.state].filter(Boolean).join(', ')}
      </p>
      {facility.emergency_available ? (
        <p>Emergency availability: {facility.emergency_available}</p>
      ) : null}
      <p className="source-tag">Source: {facility.source || 'unavailable'}</p>
    </article>
  );
}
