export default function MedicalShopCard({ shop }) {
  if (!shop) return null;
  return (
    <article className="entity-card">
      <h3>{shop.name}</h3>
      <p className="muted">
        {[shop.city, shop.district, shop.state].filter(Boolean).join(', ')}
      </p>
      {shop.address ? <p>{shop.address}</p> : null}
      <p className="source-tag">Source: {shop.source || 'unavailable'}</p>
      <p className="disclaimer-inline">
        Listings do not include medication prescribing or dosage advice.
      </p>
    </article>
  );
}
