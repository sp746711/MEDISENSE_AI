import React from 'react';

export default function DoctorCard({ doctor }) {
  if (!doctor) return null;

  const entityType = doctor.entity_type || 'doctor';
  const entityLabels = {
    doctor: 'Doctor',
    clinic: 'Clinic',
    medical_center: 'Medical Center',
    hospital: 'Hospital',
  };
  const entityLabel = entityLabels[entityType.toLowerCase()] || 'Healthcare Provider';

  const isVerified = doctor.verification_status === 'verified' && doctor.last_verified;

  return (
    <article className="entity-card" style={{ padding: '18px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <div>
          <h3 style={{ margin: '0 0 4px 0', fontSize: '18px', fontWeight: 600 }}>{doctor.name}</h3>
          <span
            style={{
              display: 'inline-block',
              fontSize: '11px',
              fontWeight: 600,
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
              padding: '2px 8px',
              borderRadius: '4px',
              backgroundColor: entityType === 'doctor' ? '#e0f2fe' : entityType === 'clinic' ? '#f0fdf4' : '#fef3c7',
              color: entityType === 'doctor' ? '#0369a1' : entityType === 'clinic' ? '#15803d' : '#b45309',
              marginBottom: '6px',
            }}
          >
            {entityLabel}
          </span>
        </div>

        {doctor.distance_km != null ? (
          <span
            style={{
              fontSize: '12px',
              fontWeight: 600,
              backgroundColor: '#eff6ff',
              color: '#1d4ed8',
              padding: '2px 8px',
              borderRadius: '9999px',
              whiteSpace: 'nowrap',
            }}
          >
            {doctor.distance_km} km away
          </span>
        ) : null}
      </div>

      {doctor.specialization ? (
        <p style={{ margin: '0 0 6px 0', fontWeight: 500, color: '#2563eb', fontSize: '14px' }}>
          {doctor.specialization}
        </p>
      ) : null}

      {doctor.qualification ? (
        <p className="muted" style={{ margin: '0 0 6px 0', fontSize: '13px' }}>
          {doctor.qualification}
        </p>
      ) : null}

      {doctor.facility ? (
        <p style={{ margin: '0 0 6px 0', fontSize: '13px' }}>
          <strong>Facility:</strong> {doctor.facility}
        </p>
      ) : null}

      <p className="muted" style={{ margin: '0 0 8px 0', fontSize: '13px', lineHeight: 1.4 }}>
        {[doctor.address, doctor.city, doctor.district, doctor.state].filter(Boolean).join(', ')}
      </p>

      {doctor.registration_number ? (
        <p style={{ margin: '0 0 6px 0', fontSize: '12px', color: '#475569' }}>
          <strong>Reg No:</strong> {doctor.registration_number}
          {doctor.registration_council ? ` (${doctor.registration_council})` : ''}
        </p>
      ) : null}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', margin: '8px 0', fontSize: '13px' }}>
        {doctor.contact ? (
          <span style={{ color: '#0f766e', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <strong>Phone:</strong>{' '}
            <a href={`tel:${doctor.contact}`} onClick={(e) => e.stopPropagation()} style={{ color: '#0f766e' }}>
              {doctor.contact}
            </a>
          </span>
        ) : null}

        {doctor.website ? (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
            <a
              href={doctor.website}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              style={{ color: '#2563eb', textDecoration: 'underline' }}
            >
              Visit Website ↗
            </a>
          </span>
        ) : null}
      </div>

      <div
        style={{
          marginTop: '12px',
          paddingTop: '10px',
          borderTop: '1px solid #f1f5f9',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          fontSize: '11px',
          color: '#64748b',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
          <span>
            Provider listing source: <strong>{doctor.source || 'Google Places'}</strong>
          </span>
          {isVerified ? (
            <span style={{ color: '#059669', fontWeight: 600 }}>
              ✓ Verified: {new Date(doctor.last_verified).toLocaleDateString()}
            </span>
          ) : (
            <span
              style={{
                color: '#64748b',
                backgroundColor: '#f1f5f9',
                padding: '2px 6px',
                borderRadius: '4px',
                fontWeight: 500,
              }}
            >
              Registration Not Verified
            </span>
          )}
        </div>
        {!isVerified ? (
          <span style={{ fontSize: '10.5px', color: '#94a3b8' }}>
            Professional registration verification is not available through this system.
          </span>
        ) : null}
      </div>
    </article>
  );
}
