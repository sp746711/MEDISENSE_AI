export default function EvidenceCard({ title, children, emptyText = 'No evidence available.' }) {
  return (
    <section className="evidence-card">
      <h3>{title}</h3>
      {children || <p className="muted">{emptyText}</p>}
    </section>
  );
}
