import { formatPathway, pathwayClass } from '../utils/formatters';

export default function AssessmentStatus({ pathway, status }) {
  return (
    <div className={`assessment-status ${pathwayClass(pathway)}`}>
      <span className="status-dot" aria-hidden="true" />
      <div>
        <strong>{formatPathway(pathway)}</strong>
        {status ? <p className="muted">Status: {status}</p> : null}
      </div>
    </div>
  );
}
