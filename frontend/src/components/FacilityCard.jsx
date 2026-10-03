import React from 'react';

export default function FacilityCard({ facility }) {
  if (!facility) return null;

  const isEmergency = facility.emergency_available === 'yes' || facility.has_emergency === true;

  return (
    <article className="entity-card" style={{ padding: '16px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <h3 style={{ margin: '0 0 6px 0', fontSize: '18px', fontWeight: 600 }}>{facility.name}</h3>
        {facility.distance_km != null ? (
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
            {facility.distance_km} km away
          </span>
        ) : null}
      </div>

      {facility.type ? (
        <p style={{ margin: '0 0 6px 0', color: '#475569', fontSize: '14px', fontWeight: 500 }}>
          {facility.type}
        </p>
      ) : null}

      <p className="muted" style={{ margin: '0 0 8px 0', fontSize: '13px' }}>
        {[facility.address, facility.city, facility.district, facility.state].filter(Boolean).join(', ')}
      </p>

      {isEmergency ? (
        <div style={{ margin: '0 0 8px 0' }}>
          <span
            style={{
              display: 'inline-block',
              backgroundColor: '#fee2e2',
              color: '#b91c1c',
              padding: '3px 8px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 600,
            }}
          >
            Emergency Services Available
          </span>
        </div>
      ) : null}

      {facility.contact ? (
        <p style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#0f766e' }}>
          <strong>Contact:</strong> {facility.contact}
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
          Source: {facility.source || 'Unavailable'}
        </span>
        {facility.last_verified ? (
          <span style={{ color: '#059669', fontSize: '11px' }}>
            Verified: {new Date(facility.last_verified).toLocaleDateString()}
          </span>
        ) : null}
      </div>
    </article>
  );
}
