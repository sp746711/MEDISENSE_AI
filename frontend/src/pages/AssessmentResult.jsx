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
  const medicalReportsList =
    payload.feedback?.medical_reports || evidence.report?.medical_reports || [];
  const xrayImagesList =
    payload.feedback?.xray_images || evidence.xray?.images || [];
  const xrayReportsList =
    payload.feedback?.xray_reports || evidence.xray?.reports || [];
  const xrayData = evidence.xray;
  const possibleConditions =
    payload.feedback?.possible_conditions ||
    payload.condition_predictions?.conditions ||
    [];
  const conditionStatus =
    payload.condition_predictions?.status ||
    payload.feedback?.condition_prediction_status;
  const conditionMessage =
    payload.condition_predictions?.message ||
    payload.feedback?.condition_prediction_message;

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
                  {s.onset ? <span className="pill-detail">Onset: {s.onset}</span> : null}
                  {s.character && !['CURRENT', 'PRESENT'].includes(s.character) ? (
                    <span className="pill-detail">{s.character}</span>
                  ) : s.type && !['CURRENT', 'PRESENT', 'NEGATIVE'].includes(s.type) ? (
                    <span className="pill-detail">{s.type}</span>
                  ) : null}
                  {s.mucus ? <span className="pill-detail">{s.mucus}</span> : null}
                  {s.context && s.context !== s.trigger && !s.mucus && s.context.length < 80 ? (
                    <span className="pill-detail">{s.context}</span>
                  ) : null}
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
          {reportFindings.length === 0 && medicalReportsList.length === 0 ? (
            <p className="muted">{payload.feedback?.medical_report_message || 'Medical Report: Not provided.'}</p>
          ) : (
            <div>
              {medicalReportsList.length > 0 ? (
                <div style={{ marginBottom: '0.75rem' }}>
                  <p style={{ margin: '0 0 0.35rem 0', fontSize: '0.85rem', fontWeight: 600, color: '#334155' }}>
                    Uploaded Reports ({medicalReportsList.length}):
                  </p>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.82rem' }}>
                    {medicalReportsList.map((mr, idx) => (
                      <li key={idx} style={{ marginBottom: '0.2rem' }}>
                        <strong>{mr.file_name}</strong>
                        {mr.is_historical ? ' (Historical / Contextual)' : ''}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {reportFindings.length > 0 ? (
                <table className="mini-table">
                  <thead>
                    <tr>
                      <th>Test Name</th>
                      <th>Value</th>
                      <th>Reference</th>
                      <th>Source</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {reportFindings.map((f, idx) => (
                      <tr key={idx}>
                        <td>{f.test_name}</td>
                        <td>{f.value} {f.unit || ''}</td>
                        <td>{f.reference_range || '—'}</td>
                        <td style={{ fontSize: '0.75rem', color: '#64748b' }}>{f.source_file || 'Report'}</td>
                        <td>
                          <span className={`badge badge-${(f.interpretation || f.status || '').toLowerCase()}`}>
                            {f.interpretation || f.status || 'RECORDED'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
            </div>
          )}
        </EvidenceCard>

        {/* 3. X-RAY & RADIOLOGY EVIDENCE */}
        <EvidenceCard title="X-RAY & RADIOLOGY EVIDENCE">
          {(!xrayData || !xrayData.received) && xrayImagesList.length === 0 && xrayReportsList.length === 0 ? (
            <p className="muted">X-Ray: Not provided.</p>
          ) : (
            <div>
              {/* Images */}
              {xrayImagesList.length > 0 ? (
                <div style={{ marginBottom: '1rem' }}>
                  <h4 style={{ margin: '0 0 0.4rem 0', fontSize: '0.92rem', color: '#0369a1' }}>Images:</h4>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.88rem' }}>
                    {xrayImagesList.map((img, idx) => (
                      <li key={idx} style={{ marginBottom: '0.45rem' }}>
                        <strong>{img.file_name}</strong> ({img.region?.toUpperCase()})
                        <div style={{ fontSize: '0.82rem', marginTop: '0.15rem' }}>
                          Status: <span className={`badge badge-${(img.status || '').toLowerCase()}`}>{(img.status || '').toUpperCase()}</span>
                          {img.prediction ? (
                            <span style={{ marginLeft: '0.5rem' }}>
                              Finding: <span className="highlight">{img.prediction}</span>
                            </span>
                          ) : (
                            <span style={{ marginLeft: '0.5rem', color: '#64748b' }}>
                              Reason: automated interpretation unavailable
                            </span>
                          )}
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : xrayData?.received ? (
                <div style={{ marginBottom: '1rem' }}>
                  <p><strong>Region:</strong> {(xrayData.region || 'Not declared').toUpperCase()}</p>
                  <p><strong>Status:</strong> {(xrayData.status || '').toUpperCase()}</p>
                  {xrayData.prediction ? (
                    <p><strong>Finding:</strong> <span className="highlight">{xrayData.prediction}</span></p>
                  ) : (
                    <p className="muted">{xrayData.description || xrayData.message || 'Automated interpretation for this X-ray type is currently unavailable.'}</p>
                  )}
                </div>
              ) : null}

              {/* Radiology Reports */}
              {xrayReportsList.length > 0 ? (
                <div>
                  <h4 style={{ margin: '0 0 0.4rem 0', fontSize: '0.92rem', color: '#0369a1' }}>Radiology Reports:</h4>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.88rem' }}>
                    {xrayReportsList.map((rep, idx) => {
                      const qualFindings = rep.qualitative_findings || rep.structured_findings?.qualitative_findings || [];
                      const narrFindings = rep.narrative_findings || rep.structured_findings?.narrative_findings || [];
                      return (
                        <li key={idx} style={{ marginBottom: '0.65rem' }}>
                          <strong>{rep.file_name}</strong>
                          <div style={{ fontSize: '0.82rem', marginTop: '0.15rem' }}>
                            OCR / NLP: <span className="badge badge-extracted">{rep.status || 'EXTRACTED'}</span>
                            {rep.is_historical ? <span style={{ marginLeft: '0.4rem', color: '#64748b' }}>(Historical report)</span> : null}
                          </div>
                          {qualFindings.length > 0 && (
                            <div style={{ marginTop: '0.3rem', fontSize: '0.84rem' }}>
                              <strong>Extracted Findings:</strong>{' '}
                              {qualFindings.map((q, qIdx) => (
                                <span key={qIdx} style={{ marginRight: '0.4rem', color: q.state === 'ABSENT' ? '#059669' : '#0284c7' }}>
                                  • {q.finding} ({q.state || 'PRESENT'})
                                </span>
                              ))}
                            </div>
                          )}
                          {narrFindings.length > 0 && (
                            <div style={{ marginTop: '0.2rem', fontSize: '0.82rem', color: '#475569' }}>
                              {narrFindings.map((n, nIdx) => (
                                <div key={nIdx}>
                                  <em>{n.section || 'Impression'}:</em> {n.content}
                                </div>
                              ))}
                            </div>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
              ) : null}

              {/* Cross-Modal Concordance */}
              {evidence.xray?.cross_modal_concordance?.length > 0 && (
                <div style={{ marginTop: '0.8rem', padding: '0.6rem', background: '#f8fafc', borderRadius: '4px', border: '1px solid #e2e8f0' }}>
                  <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.85rem', color: '#334155' }}>
                    Cross-Modal Comparison (Image Model vs. Radiology Report):
                  </h5>
                  <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.82rem' }}>
                    {evidence.xray.cross_modal_concordance.map((c, cIdx) => (
                      <li key={cIdx} style={{ marginBottom: '0.25rem' }}>
                        <span style={{ fontWeight: 600, color: c.concordance_status === 'CONCORDANT' ? '#059669' : c.concordance_status === 'DISCORDANT' ? '#dc2626' : '#64748b' }}>
                          [{c.concordance_status}]
                        </span>{' '}
                        {c.details}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </EvidenceCard>

        {/* 4. WHY THIS PATHWAY */}
        <EvidenceCard title="WHY THIS PATHWAY?">
          <p>{payload.feedback?.why_this_triage_selected || triage.why_this_pathway || 'Selected by backend safety rules based on clinical indicators.'}</p>
          {triage.rules_triggered?.length ? (
            <p className="small muted">Rules triggered: {triage.rules_triggered.join(', ')}</p>
          ) : null}
        </EvidenceCard>

        {/* 4b. POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL) */}
        {possibleConditions.length > 0 ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p className="small muted" style={{ marginTop: 0, marginBottom: '0.75rem' }}>
              Probabilistic disease pattern alignment based on verified clinical symptoms. This is an assistive tool for discussion with a doctor and does NOT constitute a diagnosis.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
              {possibleConditions.map((c, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.75rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    background: idx === 0 ? '#eff6ff' : '#ffffff',
                    borderLeft: idx === 0 ? '4px solid #2563eb' : '1px solid #cbd5e1',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                    <strong style={{ fontSize: '1rem', color: '#1e3a8a' }}>
                      #{c.rank || idx + 1} {c.condition}
                    </strong>
                    <span className="badge" style={{ background: idx === 0 ? '#2563eb' : '#64748b', color: '#fff', fontSize: '0.75rem' }}>
                      {c.confidence_tier || 'Possible Alignment'}
                    </span>
                  </div>
                  {c.supporting_symptoms?.length ? (
                    <p style={{ margin: '0 0 0.25rem 0', fontSize: '0.85rem', color: '#334155' }}>
                      <strong>Supporting Symptoms:</strong> {c.supporting_symptoms.join(', ')}
                    </p>
                  ) : null}
                  {c.contradictory_findings?.length ? (
                    <p style={{ margin: '0 0 0.25rem 0', fontSize: '0.85rem', color: '#b91c1c' }}>
                      <strong>Contradictory / Denied:</strong> {c.contradictory_findings.join(', ')}
                    </p>
                  ) : null}
                  {c.unknown_or_missing_information?.length ? (
                    <p style={{ margin: '0 0 0.25rem 0', fontSize: '0.82rem', color: '#64748b' }}>
                      <strong>Unassessed Factors:</strong> {c.unknown_or_missing_information.join(', ')}
                    </p>
                  ) : null}
                  {c.suggested_specialty ? (
                    <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#0369a1' }}>
                      Relevant Specialty: <em>{c.suggested_specialty}</em>
                    </p>
                  ) : null}
                </div>
              ))}
            </div>
          </EvidenceCard>
        ) : conditionStatus === 'unsupported_condition' ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.88rem', color: '#475569', lineHeight: '1.5' }}>
              {conditionMessage || 'Reported symptoms do not show strong alignment with the 24 disease patterns evaluated by this assistive model. A comprehensive clinical evaluation by a healthcare provider is recommended.'}
            </p>
            <p className="small muted" style={{ margin: 0, fontStyle: 'italic' }}>
              Note: The assistive model refuses to guess diagnoses when clinical presentations fall outside its validated training domains.
            </p>
          </EvidenceCard>
        ) : null}

        {/* 5. SUGGESTED SPECIALTY */}
        <EvidenceCard title="SUGGESTED SPECIALTY">
          <p className="highlight-specialty">
            {payload.feedback?.recommended_next_step?.suggested_specialty || result?.specialty || assessment?.specialty || 'General Physician / Internal Medicine'}
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
