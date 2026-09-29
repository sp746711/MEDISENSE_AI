import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getAssessment, getAssessmentResult } from '../services/assessmentService';
import { getSymptoms } from '../services/symptomService';
import { getReports } from '../services/reportService';
import { getXrayResults } from '../services/xrayService';
import { formatDate, inputTypesLabel } from '../utils/formatters';

export default function AssessmentDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [bundle, setBundle] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [assessment, result, symptoms, reports, xray] = await Promise.all([
          getAssessment(id),
          getAssessmentResult(id),
          getSymptoms(id),
          getReports(id),
          getXrayResults(id),
        ]);
        if (active) setBundle({ assessment, result, symptoms, reports, xray });
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [id]);

  if (loading) return <LoadingScreen message="Loading details..." />;
  if (!bundle) {
    return (
      <div className="page">
        <ErrorMessage message={error || 'Assessment not found'} />
      </div>
    );
  }

  const { assessment, result, symptoms, reports, xray } = bundle;

  return (
    <div className="page">
      <h1>Assessment Details</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      <p>Date: {formatDate(assessment.created_at)}</p>
      <p>Inputs: {inputTypesLabel(assessment.input_types)}</p>
      <p>Pathway: {assessment.pathway || 'Not determined'}</p>
      <p>Specialty: {assessment.specialty || '—'}</p>
      <p>Status: {assessment.status}</p>
      <p>Rules version: {assessment.rules_version || '—'}</p>
      <p>Model versions: {JSON.stringify(assessment.model_versions || {})}</p>

      <section className="section">
        <h2>Symptoms</h2>
        {!symptoms.symptoms?.length ? (
          <p className="muted">No symptom records.</p>
        ) : (
          <ul>
            {symptoms.symptoms.map((s) => (
              <li key={s.symptom_id}>
                {s.symptom} — {s.state}
                {s.context ? ` | ${s.context}` : ''}
                <span className="source-tag"> source={s.source}</span>
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="section">
        <h2>Medical Reports</h2>
        {!reports.reports?.length ? (
          <p className="muted">No reports uploaded.</p>
        ) : (
          <ul>
            {reports.reports.map((r) => (
              <li key={r.report_id}>
                {r.file_name} ({r.file_type}) — findings:{' '}
                {r.structured_findings ? JSON.stringify(r.structured_findings) : 'not extracted yet'}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="section">
        <h2>X-Ray</h2>
        {!xray.results?.length ? (
          <p className="muted">No X-ray results.</p>
        ) : (
          <ul>
            {xray.results.map((r) => (
              <li key={r.xray_result_id}>
                status={r.status}; prediction={r.prediction ?? 'none'}; model={r.model_version ?? 'n/a'};{' '}
                {r.message}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="section">
        <h2>Result payload</h2>
        <pre className="code-block">{JSON.stringify(result, null, 2)}</pre>
      </section>

      <button type="button" className="btn btn-secondary" onClick={() => navigate(`/assessment/${id}`)}>
        Back to Result
      </button>
    </div>
  );
}
