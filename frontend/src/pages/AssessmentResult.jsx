import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import AssessmentStatus from '../components/AssessmentStatus';
import EvidenceCard from '../components/EvidenceCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getAssessment, getAssessmentResult } from '../services/assessmentService';
import { inputTypesLabel } from '../utils/formatters';
import { MEDICAL_DISCLAIMER } from '../utils/constants';

// Assessment timestamps are stored by the backend; render them consistently in India Standard Time.
// ISO timestamps with an explicit offset are converted correctly. Invalid/missing values remain clear.
function formatAssessmentDate(value) {
  if (!value) return 'Not available';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Date unavailable';
  return `${new Intl.DateTimeFormat('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: true,
    timeZone: 'Asia/Kolkata',
  }).format(date)} IST`;
}

/** Normalize possibly malformed API payload sections without inventing evidence. */
function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

function displayText(value, fallback = '') {
  if (value === null || value === undefined) return fallback;
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  return fallback;
}

function formatCandidateList(value) {
  return asArray(value)
    .map((item) => (typeof item === 'string' ? item : displayText(item?.symptom || item?.finding || item?.name)))
    .filter(Boolean);
}

function getSourceBadgeStyle(badgeText) {
  const t = String(badgeText || '').toLowerCase();
  if (t.includes('symptom')) return { background: '#e0f2fe', color: '#0369a1', border: '1px solid #bae6fd' };
  if (t.includes('laboratory') || t.includes('lab')) return { background: '#fef3c7', color: '#92400e', border: '1px solid #fde68a' };
  if (t.includes('radiology')) return { background: '#f3e8ff', color: '#6b21a8', border: '1px solid #e9d5ff' };
  if (t.includes('x-ray') || t.includes('xray')) return { background: '#ecfdf5', color: '#065f46', border: '1px solid #a7f3d0' };
  if (t.includes('contradictory')) return { background: '#fee2e2', color: '#991b1b', border: '1px solid #fecaca' };
  return { background: '#f1f5f9', color: '#475569', border: '1px solid #cbd5e1' };
}


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
      window.setTimeout(() => window.URL.revokeObjectURL(url), 1000);
    } catch (err) {
      setError(err.message);
    } finally {
      setDownloading(false);
    }
  };

  if (loading) return <LoadingScreen message="Loading assessment..." />;

  const pathway = result?.pathway || assessment?.pathway || null;
  const upper = String(pathway || '').toUpperCase();
  const payload = asObject(result?.result || assessment?.result_payload);
  const feedback = asObject(payload.feedback);
  const evidence = asObject(payload.evidence);
  const symptomEvidence = asObject(evidence.symptoms);
  const reportEvidence = asObject(evidence.report);
  const xrayData = asObject(evidence.xray);
  const triage = asObject(payload.triage);
  const predictions = asObject(payload.condition_predictions);

  const symptomsPresent = asArray(symptomEvidence.present);
  const symptomsAbsent = asArray(symptomEvidence.absent);
  const symptomsUnknown = asArray(symptomEvidence.unknown);
  const historicalSymptoms = asArray(
    feedback.historical_symptoms?.length
      ? feedback.historical_symptoms
      : feedback.historical_context?.length
        ? feedback.historical_context
        : symptomEvidence.historical
  );
  const feedbackReportFindings = asArray(feedback.medical_report_findings);
  const reportFindings = feedbackReportFindings.length
    ? feedbackReportFindings
    : [
        ...asArray(reportEvidence.abnormal_findings),
        ...asArray(reportEvidence.normal_findings),
        ...asArray(reportEvidence.unknown_range_findings),
      ];
  const medicalReportsList = asArray(
    feedback.medical_reports?.length ? feedback.medical_reports : reportEvidence.medical_reports
  );
  const xrayImagesList = asArray(
    feedback.xray_images?.length ? feedback.xray_images : xrayData.images
  );
  const xrayReportsList = asArray(
    feedback.xray_reports?.length ? feedback.xray_reports : xrayData.reports
  );

  // Display only actual candidate records; do not fabricate missing matches.
  // The system displays up to 24 genuine candidate conditions returned by the backend.
  const feedbackCandidates = asArray(feedback.possible_conditions);
  const predictionCandidates = asArray(predictions.conditions);
  const rawPossibleConditions = feedbackCandidates.length
    ? feedbackCandidates
    : predictionCandidates;
  const possibleConditions = rawPossibleConditions
    .filter((candidate) => candidate && typeof candidate === 'object' &&
      displayText(candidate.condition || candidate.name || candidate.label).trim())
    .slice(0, 24);
  const conditionStatus = predictions.status || feedback.condition_prediction_status;
  const conditionMessage = predictions.message || feedback.condition_prediction_message;

  // Use the recorded input list to distinguish 'not supplied' from 'supplied but not processed'.
  const inputTypes = Array.isArray(assessment?.input_types)
    ? assessment.input_types.map((v) => String(v).toLowerCase())
    : String(assessment?.input_types || '').toLowerCase().split(/[;,|]/).map((v) => v.trim()).filter(Boolean);
  const hasInputType = (...needles) => inputTypes.some((type) => needles.some((needle) => type.includes(needle)));
  const symptomsWereProvided = hasInputType('symptom');
  const medicalReportWasProvided = hasInputType('medical_report', 'medical report', 'lab', 'blood report');
  const xrayWasProvided = hasInputType('xray', 'x-ray', 'radiology', 'image');

  const hasSymptomsInput = symptomsWereProvided || symptomsPresent.length > 0 || symptomsAbsent.length > 0 || symptomsUnknown.length > 0 || historicalSymptoms.length > 0;
  const hasMedicalReportInput = medicalReportWasProvided || reportFindings.length > 0 || medicalReportsList.length > 0;
  const hasXrayImageInput = Boolean(
    xrayImagesList.length > 0 ||
    (xrayData?.images_received && xrayData?.status !== 'NOT_PROVIDED') ||
    (hasInputType('xray', 'image') && !hasInputType('radiology_report'))
  );
  const hasRadiologyReportInput = Boolean(
    xrayReportsList.length > 0 ||
    hasInputType('radiology', 'radiology_report')
  );

  return (
    <div className="page">
      <h1>Final Health Outcome</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />

      <div className="meta-strip">
        <p><strong>Assessment Date:</strong> {formatAssessmentDate(assessment?.created_at)}</p>
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
      {feedback.assessment_summary ? (
        <div className="card" style={{ marginBottom: '1.5rem', borderLeft: '4px solid #2563eb', background: '#f8fafc' }}>
          <h3 style={{ marginTop: 0, color: '#1e40af', fontSize: '1.1rem' }}>Clinical Assessment Summary</h3>
          <p style={{ lineHeight: '1.6', margin: 0, color: '#1e293b' }}>{feedback.assessment_summary}</p>
        </div>
      ) : null}

      <div className="evidence-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 420px), 1fr))', alignItems: 'start', gap: '1rem', width: '100%' }}>
        {/* 1. SYMPTOMS */}
        {hasSymptomsInput ? (
          <EvidenceCard title="SYMPTOMS EVIDENCE">
            {symptomsPresent.length === 0 && symptomsAbsent.length === 0 && symptomsUnknown.length === 0 ? (
              <p className="muted">{symptomsWereProvided ? (feedback.symptom_message || 'Symptoms were submitted, but no structured symptom findings are available in this result. Check processing status or assessment details.') : 'No symptoms were provided for this assessment.'}</p>
            ) : (
              <div className="symptoms-list">
                {symptomsPresent.map((s, idx) => (
                  <div key={idx} className="evidence-pill present">
                    <span className="badge badge-present">PRESENT</span>
                    <strong>{s.symptom || s.finding || 'Unnamed symptom'}</strong>
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
                    <span>{s.symptom || s.finding || 'Unnamed symptom'}</span>
                  </div>
                ))}
                {symptomsUnknown.map((s, idx) => (
                  <div key={`unknown-${idx}`} className="evidence-pill" style={{ background: '#fffbeb', borderColor: '#f59e0b' }}>
                    <span className="badge" style={{ background: '#b45309', color: '#fff' }}>UNKNOWN</span>
                    <span>{s.symptom || s.finding || 'Unanswered symptom'}</span>
                    {s.context ? <span className="pill-detail">{s.context}</span> : null}
                  </div>
                ))}
              </div>
            )}
          </EvidenceCard>
        ) : null}

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
        {hasMedicalReportInput ? (
          <EvidenceCard title="MEDICAL REPORT FINDINGS">
            {reportFindings.length === 0 && medicalReportsList.length === 0 ? (
              <p className="muted">{feedback.medical_report_message || (medicalReportWasProvided ? 'A medical/laboratory report was submitted, but no structured findings are available in this result. Check the upload and extraction status.' : 'No medical/laboratory report was provided for this assessment.')}</p>
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
        ) : null}

        {/* 3a. RADIOLOGY REPORT FINDINGS */}
        {hasRadiologyReportInput && xrayReportsList.length > 0 ? (
          <EvidenceCard title="RADIOLOGY REPORT FINDINGS">
            <div>
              <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.88rem' }}>
                {xrayReportsList.map((rep, idx) => {
                  const qualFindings = rep.qualitative_findings || rep.structured_findings?.qualitative_findings || [];
                  const narrFindings = rep.narrative_findings || rep.structured_findings?.narrative_findings || [];
                  const isUnreadable = rep.status === 'UNREADABLE';
                  const assocStatus = rep.region_verification?.association_status || rep.association_status;
                  return (
                    <li key={idx} style={{ marginBottom: '0.65rem' }}>
                      <strong>{rep.file_name}</strong>
                      <div style={{ fontSize: '0.82rem', marginTop: '0.15rem' }}>
                        OCR / NLP: <span className={`badge badge-${isUnreadable ? 'rejected' : 'extracted'}`}>{rep.status || 'EXTRACTED'}</span>
                        {assocStatus ? (
                          <span style={{ marginLeft: '0.5rem' }}>
                            Report/Image Link: <span className={`badge badge-${assocStatus.toLowerCase()}`}>{assocStatus}</span>
                          </span>
                        ) : null}
                        {rep.is_historical ? <span style={{ marginLeft: '0.4rem', color: '#64748b' }}>(Historical report)</span> : null}
                      </div>
                      {isUnreadable ? (
                        <div style={{ marginTop: '0.3rem', fontSize: '0.82rem', color: '#dc2626' }}>
                          Report text could not be extracted or read. Upload record preserved for clinical review.
                        </div>
                      ) : qualFindings.length === 0 && narrFindings.length === 0 ? (
                        <div style={{ marginTop: '0.3rem', fontSize: '0.82rem', color: '#64748b' }}>
                          No discrete abnormal findings extracted from text; document preserved for clinical review.
                        </div>
                      ) : null}
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
          </EvidenceCard>
        ) : null}

        {/* 3b. X-RAY IMAGE ANALYSIS */}
        {hasXrayImageInput ? (
          <EvidenceCard title="X-RAY IMAGE ANALYSIS">
            {(!xrayData || !xrayData.received) && xrayImagesList.length === 0 ? (
              <p className="muted">{feedback.xray_message || 'An X-ray image was submitted, but detailed image analysis is not available in this result.'}</p>
            ) : (
              <div>
                {xrayImagesList.length > 0 ? (
                  <div>
                    <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.88rem' }}>
                      {xrayImagesList.map((img, idx) => {
                        const isVerified = img.anatomy_independently_verified === true;
                        const rawOutput = img.raw_prediction_for_review || img.raw_model_prediction;
                        return (
                          <li key={idx} style={{ marginBottom: '0.55rem' }}>
                            <strong>{img.file_name}</strong> ({img.region?.toUpperCase()})
                            <div style={{ fontSize: '0.82rem', marginTop: '0.15rem' }}>
                              Status: <span className={`badge badge-${(img.status || '').toLowerCase()}`}>{(img.status || '').toUpperCase()}</span>
                              <span style={{ marginLeft: '0.5rem' }}>
                                Anatomy Verification: <span className={`badge badge-${isVerified ? 'confirmed' : 'unverified'}`}>{isVerified ? 'VERIFIED' : 'UNVERIFIED / PENDING CONFIRMATION'}</span>
                              </span>
                            </div>
                            {img.prediction ? (
                              <div style={{ marginTop: '0.2rem', fontSize: '0.84rem' }}>
                                Accepted Finding: <span className="highlight">{img.prediction}</span>
                              </div>
                            ) : rawOutput ? (
                              <div style={{ marginTop: '0.2rem', fontSize: '0.82rem', color: '#64748b' }}>
                                <em>Raw model candidate (unverified, for audit only):</em> {rawOutput} — <em>excluded from clinical evidence until anatomy is independently verified.</em>
                              </div>
                            ) : (
                              <div style={{ marginTop: '0.2rem', fontSize: '0.82rem', color: '#64748b' }}>
                                Reason: {img.message || img.description || 'Automated interpretation unavailable'}
                              </div>
                            )}
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                ) : (
                  <div>
                    <p><strong>Region:</strong> {(xrayData.region || 'Not declared').toUpperCase()}</p>
                    <p><strong>Status:</strong> {(xrayData.status || '').toUpperCase()}</p>
                    {xrayData.prediction ? (
                      <p><strong>Finding:</strong> <span className="highlight">{xrayData.prediction}</span></p>
                    ) : (
                      <p className="muted">{xrayData.description || xrayData.message || 'Automated interpretation for this X-ray type is currently unavailable.'}</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </EvidenceCard>
        ) : null}

        {/* 3c. RADIOLOGY REPORT & X-RAY COMPARISON */}
        {hasXrayImageInput && hasRadiologyReportInput && asArray(xrayData.cross_modal_concordance).length > 0 ? (
          <EvidenceCard title="RADIOLOGY REPORT & X-RAY COMPARISON">
            <div style={{ padding: '0.6rem', background: '#f8fafc', borderRadius: '4px', border: '1px solid #e2e8f0' }}>
              <h5 style={{ margin: '0 0 0.3rem 0', fontSize: '0.85rem', color: '#334155' }}>
                Cross-Modal Comparison (Image Model vs. Radiology Report):
              </h5>
              <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.82rem' }}>
                {asArray(xrayData.cross_modal_concordance).map((c, cIdx) => (
                  <li key={cIdx} style={{ marginBottom: '0.25rem' }}>
                    <span style={{ fontWeight: 600, color: c.concordance_status === 'CONCORDANT' ? '#059669' : c.concordance_status === 'DISCORDANT' ? '#dc2626' : '#64748b' }}>
                      [{c.concordance_status}]
                    </span>{' '}
                    {c.details}
                  </li>
                ))}
              </ul>
            </div>
          </EvidenceCard>
        ) : null}

        {/* 4. WHY THIS PATHWAY */}
        <EvidenceCard title="WHY THIS PATHWAY?">
          <p>{feedback.why_this_triage_selected || triage.why_this_pathway || 'Selected by backend safety rules based on clinical indicators.'}</p>
          {triage.rules_triggered?.length ? (
            <p className="small muted">Rules triggered: {triage.rules_triggered.join(', ')}</p>
          ) : null}
        </EvidenceCard>

        {/* 4b. POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL) */}
        {possibleConditions.length > 0 ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p className="small muted" style={{ marginTop: 0, marginBottom: '0.75rem' }}>
              These are statistical model candidates for discussion with a qualified clinician, not confirmed diagnoses or clinically validated probabilities. Review the candidate-specific evidence below. The deterministic safety triage is independent and is not overridden by these candidates.
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {possibleConditions.map((c, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '0.85rem',
                    borderRadius: '8px',
                    border: '1px solid #cbd5e1',
                    background: idx === 0 ? '#eff6ff' : '#ffffff',
                    borderLeft: idx === 0 ? '4px solid #2563eb' : '1px solid #cbd5e1',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
                    <strong style={{ fontSize: '1rem', color: '#1e3a8a' }}>
                      #{c.rank || idx + 1} {c.condition || c.name || c.label}
                    </strong>
                    <span className="badge" style={{ background: idx === 0 ? '#2563eb' : '#64748b', color: '#fff', fontSize: '0.75rem' }}>
                      {c.confidence_tier || 'Possible candidate'}
                    </span>
                  </div>

                  {/* Badges strip */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem', margin: '0.35rem 0' }}>
                    <span style={{ fontSize: '0.72rem', padding: '0.15rem 0.45rem', borderRadius: '4px', background: '#f8fafc', color: '#475569', border: '1px solid #e2e8f0', fontWeight: '500' }}>
                      Model Score: {c.model_score || c.probability_score ? Number(c.model_score || c.probability_score).toFixed(4) : 'N/A'} (Assistive statistical score)
                    </span>
                    {asArray(c.evidence_badges).map((badge, bIdx) => (
                      <span key={bIdx} style={{ fontSize: '0.72rem', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: '500', ...getSourceBadgeStyle(badge) }}>
                        {badge}
                      </span>
                    ))}
                  </div>

                  {/* Evidence Summary */}
                  {c.evidence_summary ? (
                    <p style={{ margin: '0.35rem 0 0.5rem 0', fontSize: '0.84rem', color: '#334155', fontStyle: 'italic' }}>
                      {c.evidence_summary}
                    </p>
                  ) : null}

                  {/* Supporting Evidence Breakdown */}
                  {asArray(c.supporting_evidence).length ? (
                    <div style={{ margin: '0.3rem 0', fontSize: '0.84rem' }}>
                      <strong style={{ color: '#166534' }}>Supporting Evidence:</strong>
                      <ul style={{ margin: '0.15rem 0 0.35rem 1.1rem', padding: 0 }}>
                        {c.supporting_evidence.map((item, sIdx) => (
                          <li key={sIdx} style={{ marginBottom: '0.2rem', color: '#15803d' }}>
                            <span style={{ fontWeight: '600' }}>
                              {item.source_type === 'symptom_nlp' ? '[Symptom] ' :
                               item.source_type === 'laboratory_report' ? '[Lab Report] ' :
                               item.source_type === 'radiology_report' ? '[Radiology Report] ' :
                               item.source_type === 'xray_image' ? '[Verified X-ray] ' : ''}
                            </span>
                            {item.finding}
                            {item.source_file ? <span style={{ color: '#64748b', fontSize: '0.78rem' }}> (Source: {item.source_file})</span> : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : c.supporting_symptoms?.length ? (
                    <p style={{ margin: '0 0 0.25rem 0', fontSize: '0.85rem', color: '#334155' }}>
                      <strong>Supporting Symptoms:</strong> {formatCandidateList(c.supporting_symptoms).join(', ')}
                    </p>
                  ) : null}

                  {/* Contradictory Evidence Breakdown */}
                  {asArray(c.contradictory_evidence).length ? (
                    <div style={{ margin: '0.3rem 0', fontSize: '0.84rem' }}>
                      <strong style={{ color: '#b91c1c' }}>Contradictory / Denied Findings:</strong>
                      <ul style={{ margin: '0.15rem 0 0.35rem 1.1rem', padding: 0 }}>
                        {c.contradictory_evidence.map((item, cIdx) => (
                          <li key={cIdx} style={{ marginBottom: '0.2rem', color: '#dc2626' }}>
                            <span style={{ fontWeight: '600' }}>
                              {item.source_type === 'symptom_nlp' ? '[Denied Symptom] ' : '[Lab Finding] '}
                            </span>
                            {item.finding} {item.reason ? `— ${item.reason}` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : c.contradictory_findings?.length ? (
                    <p style={{ margin: '0 0 0.25rem 0', fontSize: '0.85rem', color: '#b91c1c' }}>
                      <strong>Contradictory / Denied:</strong> {formatCandidateList(c.contradictory_findings).join(', ')}
                    </p>
                  ) : null}

                  {/* Reassuring Evidence */}
                  {asArray(c.reassuring_evidence).length ? (
                    <div style={{ margin: '0.3rem 0', fontSize: '0.84rem' }}>
                      <strong style={{ color: '#047857' }}>Reassuring Parameters:</strong>
                      <ul style={{ margin: '0.15rem 0 0.35rem 1.1rem', padding: 0 }}>
                        {c.reassuring_evidence.map((item, rIdx) => (
                          <li key={rIdx} style={{ marginBottom: '0.2rem', color: '#059669' }}>
                            {item.finding} <span style={{ color: '#64748b', fontSize: '0.78rem' }}>({item.reason})</span>
                            {item.source_file ? <span style={{ color: '#64748b', fontSize: '0.75rem' }}> [Source: {item.source_file}]</span> : null}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {/* Contextual Findings */}
                  {asArray(c.contextual_evidence).length ? (
                    <div style={{ margin: '0.25rem 0', fontSize: '0.78rem', color: '#64748b' }}>
                      <strong>Contextual assessment evidence:</strong>{' '}
                      {c.contextual_evidence.map((item, idx) => (
                        <span key={idx}>
                          {idx > 0 ? '; ' : ''}
                          {item.finding || item}
                          {item.source_file ? ` (${item.source_file})` : ''}
                        </span>
                      ))}
                    </div>
                  ) : null}

                  {/* Unassessed / Missing Factors */}
                  {asArray(c.missing_information).length ? (
                    <p style={{ margin: '0.2rem 0 0.25rem 0', fontSize: '0.82rem', color: '#64748b' }}>
                      <strong>Missing / Unassessed Factors:</strong> {c.missing_information.join(' • ')}
                    </p>
                  ) : c.unknown_or_missing_information?.length ? (
                    <p style={{ margin: '0.2rem 0 0.25rem 0', fontSize: '0.82rem', color: '#64748b' }}>
                      <strong>Unassessed Factors:</strong> {formatCandidateList(c.unknown_or_missing_information).join(', ')}
                    </p>
                  ) : null}

                  {/* Suggested Specialty */}
                  {c.suggested_specialty ? (
                    <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#0369a1' }}>
                      Relevant Specialty: <em>{c.suggested_specialty}</em>
                    </p>
                  ) : null}

                  {/* Clinical Disclaimer */}
                  <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.75rem', color: '#94a3b8', fontStyle: 'italic' }}>
                    {c.disclaimer || 'Assistive statistical candidate for clinical consultation; not a confirmed medical diagnosis.'}
                  </p>
                </div>
              ))}
            </div>
          </EvidenceCard>
        ) : conditionStatus === 'unsupported_condition' ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.88rem', color: '#475569', lineHeight: '1.5' }}>
              {conditionMessage || 'Reported symptoms do not show strong alignment with the condition patterns evaluated by this assistive model. A comprehensive clinical evaluation by a healthcare provider is recommended.'}
            </p>
            <p className="small muted" style={{ margin: 0, fontStyle: 'italic' }}>
              Note: The assistive model refuses to guess diagnoses when clinical presentations fall outside its validated training domains.
            </p>
          </EvidenceCard>
        ) : conditionStatus && !['ok', 'success', 'completed'].includes(String(conditionStatus).toLowerCase()) ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p className="small muted" style={{ margin: 0 }}>
              {conditionMessage || 'Condition matching is currently unavailable. No candidate conditions have been inferred.'}
            </p>
          </EvidenceCard>
        ) : rawPossibleConditions.length === 0 ? (
          <EvidenceCard title="POSSIBLE CONDITIONS (ASSISTIVE ML DIFFERENTIAL)">
            <p className="small muted" style={{ margin: 0 }}>
              No displayable candidates were returned by the model for this assessment. This does not establish that no condition is present.
            </p>
          </EvidenceCard>
        ) : null}

        {/* 5. SUGGESTED SPECIALTY */}
        <EvidenceCard title="SUGGESTED SPECIALTY">
          <p className="highlight-specialty">
            {feedback.recommended_next_step?.suggested_specialty || result?.specialty || assessment?.specialty || 'General Physician / Internal Medicine'}
          </p>
          <p className="small muted">Controlled mapping based on affected clinical domain.</p>
        </EvidenceCard>

        {/* 6. GENERAL GUIDANCE & NEXT STEPS */}
        <EvidenceCard title="RECOMMENDED NEXT STEPS">
          <p>
            <strong>Guidance:</strong>{' '}
            {feedback.recommended_next_step?.guidance || triage.guidance || 'Consult a qualified physician.'}
          </p>
          <p>
            <strong>Next Action:</strong>{' '}
            {feedback.recommended_next_step?.next_steps || triage.next_steps || 'Schedule in-person medical evaluation.'}
          </p>
        </EvidenceCard>

        {/* 7. SUPPORTING EVIDENCE */}
        {asArray(feedback.supporting_evidence).length ? (
          <EvidenceCard title="SUPPORTING EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {asArray(feedback.supporting_evidence).map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 8. REASSURING EVIDENCE */}
        {asArray(feedback.reassuring_evidence).length ? (
          <EvidenceCard title="REASSURING EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {asArray(feedback.reassuring_evidence).map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#15803d' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 9. SEPARATE / CONTEXTUAL FINDINGS */}
        {asArray(feedback.separate_contextual_findings).length ? (
          <EvidenceCard title="SEPARATE / CONTEXTUAL FINDINGS">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {asArray(feedback.separate_contextual_findings).map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#475569' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 10. CONTRADICTORY EVIDENCE */}
        {asArray(feedback.contradictory_evidence).length ? (
          <EvidenceCard title="CONTRADICTORY EVIDENCE">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {asArray(feedback.contradictory_evidence).map((item, idx) => (
                <li key={idx} style={{ marginBottom: '0.3rem', fontSize: '0.9rem', color: '#b91c1c' }}>{item}</li>
              ))}
            </ul>
          </EvidenceCard>
        ) : null}

        {/* 11. UNKNOWN / MISSING INFORMATION */}
        {asArray(feedback.uncertain_missing_info).length ? (
          <EvidenceCard title="UNKNOWN / MISSING INFORMATION">
            <ul style={{ paddingLeft: '1.2rem', margin: 0 }}>
              {asArray(feedback.uncertain_missing_info).map((item, idx) => (
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
              Find Nearest Hospital
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
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {upper === 'ROUTINE_CLINIC' || upper === 'ROUTINE' ? (
          <>
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => navigate(`/facilities?assessment_id=${encodeURIComponent(id)}`)}
            >
              Find Clinic / Routine Evaluation
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
        ) : null}

        {upper === 'MILD' ? (
          <>
            <button type="button" className="btn btn-primary" onClick={() => navigate(`/assessment/${id}/details`)}>
              Self-Care Guidance
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

        {!upper || !['EMERGENCY', 'CONSULTATION', 'ROUTINE_CLINIC', 'ROUTINE', 'MILD'].includes(upper) ? (
          <>
            <button type="button" className="btn btn-primary" onClick={() => navigate(`/assessment`)}>
              Review Missing Information
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/assistant')}>
              Ask AI Assistant
            </button>
          </>
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
