import React from 'react';

export default function DoctorCard({ doctor }) {
  if (!doctor) return null;

  return (
    <article className="entity-card" style={{ padding: '16px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <h3 style={{ margin: '0 0 6px 0', fontSize: '18px', fontWeight: 600 }}>{doctor.name}</h3>
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

      <p style={{ margin: '0 0 6px 0', fontWeight: 500, color: '#2563eb' }}>
        {doctor.specialization}
      </p>

      {doctor.qualification ? (
        <p className="muted" style={{ margin: '0 0 6px 0', fontSize: '13px' }}>
          {doctor.qualification}
        </p>
      ) : null}

      {doctor.facility ? (
        <p style={{ margin: '0 0 6px 0', fontSize: '14px' }}>
          <strong>Facility:</strong> {doctor.facility}
        </p>
      ) : null}

      <p className="muted" style={{ margin: '0 0 8px 0', fontSize: '13px' }}>
        {[doctor.address, doctor.city, doctor.district, doctor.state].filter(Boolean).join(', ')}
      </p>

      {doctor.registration_number ? (
        <p style={{ margin: '0 0 6px 0', fontSize: '12px', color: '#475569' }}>
          <strong>Reg No:</strong> {doctor.registration_number}
          {doctor.registration_council ? ` (${doctor.registration_council})` : ''}
        </p>
      ) : null}

      {doctor.contact ? (
        <p style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#0f766e' }}>
          <strong>Contact:</strong> {doctor.contact}
        </p>
      ) : null}

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '10px',
          paddingTop: '8px',
          borderTop: '1px solid #f1f5f9',
          fontSize: '12px',
        }}
      >
        <span className="source-tag" style={{ margin: 0 }}>
          Source: {doctor.source || 'Unavailable'}
        </span>
        {doctor.last_verified ? (
          <span style={{ color: '#059669', fontSize: '11px' }}>
            Verified: {new Date(doctor.last_verified).toLocaleDateString()}
          </span>
        ) : null}
      </div>
    </article>
  );
}
