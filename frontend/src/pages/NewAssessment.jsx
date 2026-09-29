import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { createAssessment } from '../services/assessmentService';
import { useAssessment } from '../hooks/useAssessment';
import { INPUT_TYPES } from '../utils/constants';

const OPTIONS = [
  { id: INPUT_TYPES.SYMPTOMS, label: 'Symptoms', desc: 'Describe symptoms in natural language' },
  { id: INPUT_TYPES.MEDICAL_REPORT, label: 'Medical Report', desc: 'Lab, blood, radiology, or other health reports (PDF/JPG/PNG)' },
  { id: INPUT_TYPES.XRAY, label: 'X-Ray', desc: 'Upload an X-ray image for supported regions' },
];

export default function NewAssessment() {
  const navigate = useNavigate();
  const { setInputTypes, setAssessmentId, setDraft } = useAssessment();
  const [selected, setSelected] = useState([]);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const toggle = (id) => {
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  const continueFlow = async () => {
    setError('');
    if (!selected.length) {
      setError('Select at least one input type.');
      return;
    }
    setSubmitting(true);
    try {
      const assessment = await createAssessment(selected);
      setDraft({
        assessmentId: assessment.assessment_id,
        inputTypes: selected,
        stepIndex: 0,
      });
      setInputTypes(selected);
      setAssessmentId(assessment.assessment_id);

      if (selected.includes(INPUT_TYPES.SYMPTOMS)) navigate('/assessment/symptoms');
      else if (selected.includes(INPUT_TYPES.MEDICAL_REPORT)) navigate('/assessment/report');
      else navigate('/assessment/xray');
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page narrow">
      <h1>New Health Assessment</h1>
      <p>What would you like to provide?</p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      <div className="option-list">
        {OPTIONS.map((opt) => (
          <label key={opt.id} className={`option-item ${selected.includes(opt.id) ? 'selected' : ''}`}>
            <input
              type="checkbox"
              checked={selected.includes(opt.id)}
              onChange={() => toggle(opt.id)}
            />
            <span>
              <strong>{opt.label}</strong>
              <em>{opt.desc}</em>
            </span>
          </label>
        ))}
      </div>
      <button type="button" className="btn btn-primary" onClick={continueFlow} disabled={submitting}>
        {submitting ? 'Creating...' : 'Continue'}
      </button>
    </div>
  );
}
