import { formatDate, formatPathway, inputTypesLabel, pathwayClass } from '../utils/formatters';

export default function AssessmentCard({ assessment, onView }) {
  if (!assessment) return null;
  return (
    <article className="assessment-card">
      <div className="assessment-card-top">
        <strong>{formatDate(assessment.created_at)}</strong>
        <span className={`pathway-pill ${pathwayClass(assessment.pathway)}`}>
          {formatPathway(assessment.pathway)}
        </span>
      </div>
      <p>{inputTypesLabel(assessment.input_types)}</p>
      <p className="muted">Status: {assessment.status}</p>
      {onView ? (
        <button type="button" className="btn btn-secondary" onClick={() => onView(assessment)}>
          View
        </button>
      ) : null}
    </article>
  );
}
