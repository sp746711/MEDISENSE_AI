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
  const historicalSymptoms =
    payload.feedback?.historical_symptoms ||
    payload.feedback?.historical_context ||
    evidence.symptoms?.historical ||
    [];
  const reportFindings = payload.feedback?.medical_report_findings?.length
    ? payload.feedback.medical_report_findings
    : (evidence.report?.abnormal_findings || [])
        .concat(evidence.report?.normal_findings || [])
        .concat(evidence.report?.unknown_range_findings || []);
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

      {/* DYNAMIC ASSESSMENT SUMMARY */}
      {payload.feedback?.assessment_summary ? (
        <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '4px solid #2563eb', background: '#f8fafc' }}>
          <h3 style={{ marginTop: 0, color: '#1e40af', fontSize: '1.1rem' }}>Clinical Assessment Summary</h3>
          <p style={{ lineHeight: '1.6', margin: 0, color: '#1e293b' }}>{payload.feedback.assessment_summary}</p>
        </div>
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
                  {s.laterality ? <span className="pill-detail">[{s.laterality}-sided]</span> : null}
                  {s.quality ? <span className="pill-detail">[{s.quality}]</span> : null}
                  {s.trigger ? <span className="pill-detail">Trigger: {s.trigger}</span> : null}
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

        {/* 1b. HISTORICAL CONTEXT */}
        {historicalSymptoms.length > 0 ? (
          <EvidenceCard title="HISTORICAL CONTEXT">
            <div className="symptoms-list">
              {historicalSymptoms.map((h, idx) => (
                <div key={idx} className="evidence-pill" style={{ background: '#f8fafc', borderColor: '#cbd5e1' }}>
                  <span className="badge" style={{ background: '#64748b', color: '#fff' }}>HISTORICAL</span>
                  <strong>{h.symptom || h.finding}</strong>
                  <span className="pill-detail">{h.history || h.context || 'Documented occurrence'}</span>
                </div>
              ))}
            </div>
            <p className="small muted" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
              Historical findings are documented for clinical context and excluded from active acute symptom counts.
            </p>
          </EvidenceCard>
        ) : null}

        {/* 2. MEDICAL REPORT */}
        <EvidenceCard title="MEDICAL REPORT FINDINGS">
          {reportFindings.length === 0 ? (
            <p className="muted">{payload.feedback?.medical_report_message || 'Medical Report: Not provided.'}</p>
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
                      <span className={`badge badge-${(f.interpretation || f.status || '').toLowerCase()}`}>
                        {f.interpretation || f.status || 'RECORDED'}
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
          {xrayData && xrayData.received ? (
            <div>
              <p><strong>Region:</strong> {(xrayData.region || 'Not declared').toUpperCase()}</p>
              <p><strong>Status:</strong> {(xrayData.status || '').toUpperCase()}</p>
              {xrayData.prediction ? (
                <p><strong>Finding:</strong> <span className="highlight">{xrayData.prediction}</span></p>
              ) : (
                <p className="muted">{xrayData.description || xrayData.message || 'Automated interpretation for this X-ray type is currently unavailable.'}</p>
              )}
              {xrayData.uncertainty ? (
                <p className="small muted"><strong>Uncertainty:</strong> {xrayData.uncertainty}</p>
              ) : null}
            </div>
          ) : (
            <p className="muted">X-Ray: Not provided.</p>
          )}
        </EvidenceCard>

        {/* 4. WHY THIS PATHWAY */}
        <EvidenceCard title="WHY THIS PATHWAY?">
          <p>{payload.feedback?.why_this_triage_selected || triage.why_this_pathway || 'Selected by backend safety rules based on clinical indicators.'}</p>
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
          <p>
            <strong>Guidance:</strong>{' '}
            {payload.feedback?.recommended_next_step?.guidance || triage.guidance || 'Consult a qualified physician.'}
          </p>
          <p>
            <strong>Next Action:</strong>{' '}
            {payload.feedback?.recommended_next_step?.next_steps || triage.next_steps || 'Schedule in-person medical evaluation.'}
          </p>
        </EvidenceCard>

        {/* 7. SUPPORTING EVIDENCE */}
        {payload.feedback?.supporting_evidence?.length ? (
          <EvidenceCard title="SUPPORTING EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {payload.feedback.supporting_evidence.map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 8. REASSURING EVIDENCE */}
        {payload.feedback?.reassuring_evidence?.length ? (
          <EvidenceCard title="REASSURING EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {payload.feedback.reassuring_evidence.map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#15803d' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 9. SEPARATE / CONTEXTUAL FINDINGS */}
        {payload.feedback?.separate_contextual_findings?.length ? (
          <EvidenceCard title="SEPARATE / CONTEXTUAL FINDINGS">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {payload.feedback.separate_contextual_findings.map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#475569' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 10. CONTRADICTORY EVIDENCE */}
        {payload.feedback?.contradictory_evidence?.length ? (
          <EvidenceCard title="CONTRADICTORY EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {payload.feedback.contradictory_evidence.map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#b91c1c' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 11. UNKNOWN / MISSING INFORMATION */}
        {payload.feedback?.uncertain_missing_info?.length ? (
          <EvidenceCard title="UNKNOWN / MISSING INFORMATION">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {payload.feedback.uncertain_missing_info.map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#64748b' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}
      </div>

      <div className="action-row">
        {upper === 'EMERGENCY' ? (
          <>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => navigate(`/facilities?emergency_only=true&assessment_id=${encodeURIComponent(id)}`)}
            >
              Find Emergency Facilities
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {upper === 'CONSULTATION' ? (
          <>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                const spec = result?.specialty || assessment?.specialty || '';
                const query = spec
                  ? `/doctors?specialization=${encodeURIComponent(spec)}&assessment_id=${encodeURIComponent(id)}`
                  : `/doctors?assessment_id=${encodeURIComponent(id)}`;
                navigate(query);
              }}
            >
              Find Doctors
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => navigate(`/facilities?assessment_id=${encodeURIComponent(id)}`)}
            >
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
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => navigate(`/medical-shops?assessment_id=${encodeURIComponent(id)}`)}
            >
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
