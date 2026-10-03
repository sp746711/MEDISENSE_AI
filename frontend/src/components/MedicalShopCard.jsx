import React from 'react';

export default function MedicalShopCard({ shop }) {
  if (!shop) return null;

  return (
    <article className="entity-card" style={{ padding: '16px', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '8px' }}>
        <h3 style={{ margin: '0 0 6px 0', fontSize: '18px', fontWeight: 600 }}>{shop.name}</h3>
        {shop.distance_km != null ? (
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
            {shop.distance_km} km away
          </span>
        ) : null}
      </div>

      <p className="muted" style={{ margin: '0 0 6px 0', fontSize: '13px' }}>
        {[shop.address, shop.city, shop.district, shop.state].filter(Boolean).join(', ')}
      </p>

      {shop.contact ? (
        <p style={{ margin: '0 0 8px 0', fontSize: '13px', color: '#0f766e' }}>
          <strong>Contact:</strong> {shop.contact}
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
          Source: {shop.source || 'Unavailable'}
        </span>
        {shop.last_verified ? (
          <span style={{ color: '#059669', fontSize: '11px' }}>
            Verified: {new Date(shop.last_verified).toLocaleDateString()}
          </span>
        ) : null}
      </div>

      <p className="disclaimer-inline" style={{ marginTop: '8px', fontSize: '11px', color: '#94a3b8' }}>
        Listings do not include medication prescribing or dosage advice.
      </p>
    </article>
  );
}
