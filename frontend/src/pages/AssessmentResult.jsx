import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import AssessmentStatus from '../components/AssessmentStatus';
import EvidenceCard from '../components/EvidenceCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getAssessment, getAssessmentResult } from '../services/assessmentService';
import { formatDate, inputTypesLabel } from '../utils/formatters';
import { MEDICAL_DISCLAIMER } from '../utils/constants';

export default function AssessmentResult() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [assessment, setAssessment] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const [a, r] = await Promise.all([getAssessment(id), getAssessmentResult(id)]);
        if (!active) return;
        setAssessment(a);
        setResult(r);
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

  const downloadPdf = async () => {
    setDownloading(true);
    setError('');
    try {
      const token = localStorage.getItem('medisense_token');
      const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
      const res = await fetch(`${apiBase}/assessments/${id}/report.pdf`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!res.ok) {
        throw new Error('Unable to download PDF report. Please verify assessment status.');
      }
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `MediSense_Report_${id}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloading(false);
    }
  };

  if (loading) return <LoadingScreen message="Loading assessment..." />;

  const pathway = result?.pathway || assessment?.pathway;
  const upper = String(pathway || '').toUpperCase();
  const payload = result?.result || assessment?.result_payload || {};
  const evidence = payload.evidence || {};
  const triage = payload.triage || {};

  const symptomsPresent = evidence.symptoms?.present || [];
  const symptomsAbsent = evidence.symptoms?.absent || [];
  const reportFindings = evidence.report?.abnormal_findings?.concat(evidence.report?.normal_findings || []) || [];
  const xrayData = evidence.xray;

  return (
    <div className="page">
      <h1>Final Health Outcome</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />

      <div className="meta-strip">
        <p><strong>Assessment Date:</strong> {formatDate(assessment?.created_at)}</p>
        <p><strong>Inputs Provided:</strong> {inputTypesLabel(assessment?.input_types)}</p>
      </div>

      <AssessmentStatus pathway={pathway} status={result?.status || assessment?.status} />

      {!pathway ? (
        <p className="info-banner">
          {result?.message ||
            'Available information is insufficient for a specific conclusion, or processing is not complete yet.'}
        </p>
      ) : null}

      <div className="evidence-grid">
        {/* 1. SYMPTOMS */}
        <EvidenceCard title="SYMPTOMS EVIDENCE">
          {symptomsPresent.length === 0 && symptomsAbsent.length === 0 ? (
            <p className="muted">No symptoms provided or structured.</p>
          ) : (
            <div className="symptoms-list">
              {symptomsPresent.map((s, idx) => (
                <div key={idx} className="evidence-pill present">
                  <span className="badge badge-present">PRESENT</span>
                  <strong>{s.symptom}</strong>
                  {s.duration ? <span className="pill-detail">({s.duration})</span> : null}
                  {s.severity ? <span className="pill-detail severity">[{s.severity}]</span> : null}
                </div>
              ))}
              {symptomsAbsent.map((s, idx) => (
                <div key={idx} className="evidence-pill absent">
                  <span className="badge badge-absent">ABSENT</span>
                  <span>{s.symptom}</span>
                </div>
              ))}
            </div>
          )}
        </EvidenceCard>

        {/* 2. MEDICAL REPORT */}
        <EvidenceCard title="MEDICAL REPORT FINDINGS">
          {reportFindings.length === 0 ? (
            <p className="muted">No structured lab parameters found in uploaded report.</p>
          ) : (
            <table className="mini-table">
              <thead>
                <tr>
                  <th>Test Name</th>
                  <th>Value</th>
                  <th>Reference</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {reportFindings.map((f, idx) => (
                  <tr key={idx}>
                    <td>{f.test_name}</td>
                    <td>{f.value} {f.unit || ''}</td>
                    <td>{f.reference_range || '—'}</td>
                    <td>
                      <span className={`badge badge-${(f.interpretation || '').toLowerCase()}`}>
                        {f.interpretation || 'RECORDED'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </EvidenceCard>

        {/* 3. X-RAY ANALYSIS */}
        <EvidenceCard title="X-RAY ANALYSIS">
          {xrayData && xrayData.status !== 'none' ? (
            <div>
              <p><strong>Region:</strong> {(xrayData.region || 'Chest').toUpperCase()}</p>
              <p><strong>Status:</strong> {xrayData.status.toUpperCase()}</p>
              {xrayData.prediction ? (
                <p><strong>Finding:</strong> <span className="highlight">{xrayData.prediction}</span></p>
              ) : (
                <p className="muted">{xrayData.message || 'Automated interpretation for this X-ray type is currently unavailable.'}</p>
              )}
              {xrayData.uncertainty ? (
                <p className="small muted"><strong>Uncertainty:</strong> {xrayData.uncertainty}</p>
              ) : null}
            </div>
          ) : (
            <p className="muted">No X-ray image provided for this assessment.</p>
          )}
        </EvidenceCard>

        {/* 4. WHY THIS PATHWAY */}
        <EvidenceCard title="WHY THIS PATHWAY?">
          <p>{triage.why_this_pathway || 'Selected by backend safety rules based on clinical indicators.'}</p>
          {triage.rules_triggered?.length ? (
            <p className="small muted">Rules triggered: {triage.rules_triggered.join(', ')}</p>
          ) : null}
        </EvidenceCard>

        {/* 5. SUGGESTED SPECIALTY */}
        <EvidenceCard title="SUGGESTED SPECIALTY">
          <p className="highlight-specialty">
            {result?.specialty || assessment?.specialty || 'General Physician / Internal Medicine'}
          </p>
          <p className="small muted">Controlled mapping based on affected clinical domain.</p>
        </EvidenceCard>

        {/* 6. GENERAL GUIDANCE & NEXT STEPS */}
        <EvidenceCard title="RECOMMENDED NEXT STEPS">
          <p><strong>Guidance:</strong> {triage.guidance || 'Consult a qualified physician.'}</p>
          <p><strong>Next Action:</strong> {triage.next_steps || 'Schedule in-person medical evaluation.'}</p>
        </EvidenceCard>
      </div>

      <div className="action-row">
        {upper === 'EMERGENCY' ? (
          <>
            <button type="button" className="btn btn-danger" onClick={() => navigate('/facilities')}>
              Find Emergency Facilities
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {upper === 'CONSULTATION' ? (
          <>
            <button type="button" className="btn btn-primary" onClick={() => navigate('/doctors')}>
              Find Doctors
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/facilities')}>
              Find Facilities
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {upper === 'MILD' ? (
          <>
            <button type="button" className="btn btn-primary" onClick={() => navigate(`/assessment/${id}/details`)}>
              General Guidance
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/medical-shops')}>
              Find Medical Shops
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {!upper || !['EMERGENCY', 'CONSULTATION', 'MILD'].includes(upper) ? (
          <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
            Ask AI Assistant
          </button>
        ) : null}

        <button type="button" className="btn btn-secondary" onClick={() => navigate(`/assessment/${id}/details`)}>
          View Full Details
        </button>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={downloadPdf}
          disabled={downloading}
        >
          {downloading ? 'Generating PDF...' : 'Download Assessment Report'}
        </button>
        <button type="button" className="btn btn-outline" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>

      <p className="disclaimer-inline">{MEDICAL_DISCLAIMER}</p>
    </div>
  );
}
