import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import { submitSymptoms } from '../services/symptomService';
import { useAssessment } from '../hooks/useAssessment';
import { INPUT_TYPES } from '../utils/constants';

export default function SymptomsInput() {
  const navigate = useNavigate();
  const { assessmentId, inputTypes } = useAssessment();
  const [text, setText] = useState('');
  const [stage, setStage] = useState('input'); // 'input' | 'followup'
  const [chestPain, setChestPain] = useState('No');
  const [breathlessness, setBreathlessness] = useState('No');
  const [duration, setDuration] = useState('3 to 7 days');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const nextPath = () => {
    if (inputTypes.includes(INPUT_TYPES.MEDICAL_REPORT)) return '/assessment/report';
    if (inputTypes.includes(INPUT_TYPES.XRAY)) return '/assessment/xray';
    return '/assessment/processing';
  };

  const handleInitialSubmit = (e) => {
    e.preventDefault();
    setError('');
    if (!assessmentId) return setError('No assessment in progress. Start a new assessment.');
    if (!text.trim()) return setError('Please describe your symptoms.');

    const lower = text.toLowerCase();
    const hasChestPain = lower.includes('chest pain') || lower.includes('chest tightness');
    const hasBreath = lower.includes('breath') || lower.includes('shortness of breath');
    const hasDuration = lower.includes('day') || lower.includes('week') || lower.includes('month');

    // If critical information is already detailed in the natural language text, proceed directly
    if (hasChestPain && hasBreath && hasDuration) {
      finalizeAndSubmit(text.trim());
    } else {
      // Transition to controlled follow-up clarifying questions
      setStage('followup');
    }
  };

  const finalizeAndSubmit = async (finalText) => {
    setSubmitting(true);
    setError('');
    try {
      await submitSymptoms(assessmentId, finalText);
      navigate(nextPath());
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const handleFollowupConfirm = (e) => {
    e.preventDefault();
    const clarifications = [];
    if (chestPain === 'No') clarifications.push("I don't have chest pain");
    else if (chestPain === 'Yes') clarifications.push("I have chest pain");

    if (breathlessness === 'No') clarifications.push("I don't have shortness of breath");
    else if (breathlessness === 'Yes') clarifications.push("I have shortness of breath");

    if (duration) clarifications.push(`Symptoms present for ${duration}`);

    const combinedText = `${text.trim()}. ${clarifications.join('. ')}.`;
    finalizeAndSubmit(combinedText);
  };

  return (
    <div className="page narrow">
      <h1>Symptoms Assessment</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />

      {stage === 'input' ? (
        <>
          <p className="muted">
            Describe symptoms naturally. Example: &quot;I have cough and fever for 3 days. I don&apos;t have chest pain.&quot;
          </p>
          <form onSubmit={handleInitialSubmit} className="stack-form">
            <label>
              Symptom description
              <textarea
                rows={8}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Describe what you are experiencing in detail..."
                required
              />
            </label>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              Continue
            </button>
          </form>
        </>
      ) : (
        <form onSubmit={handleFollowupConfirm} className="stack-form">
          <div className="info-banner">
            <strong>Clinical Safety Check:</strong> Please clarify important missing information before proceeding.
          </div>

          <div className="followup-group">
            <p><strong>1. Do you currently experience chest pain, tightness, or pressure?</strong></p>
            <div className="choice-row">
              {['No', 'Yes', 'Not sure'].map((opt) => (
                <button
                  type="button"
                  key={opt}
                  className={`btn-choice ${chestPain === opt ? 'selected' : ''}`}
                  onClick={() => setChestPain(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="followup-group">
            <p><strong>2. Do you have shortness of breath or difficulty breathing?</strong></p>
            <div className="choice-row">
              {['No', 'Yes', 'Not sure'].map((opt) => (
                <button
                  type="button"
                  key={opt}
                  className={`btn-choice ${breathlessness === opt ? 'selected' : ''}`}
                  onClick={() => setBreathlessness(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="followup-group">
            <p><strong>3. Approximately how long have you had these symptoms?</strong></p>
            <div className="choice-row">
              {['Under 3 days', '3 to 7 days', '1 to 2 weeks', 'More than 2 weeks'].map((opt) => (
                <button
                  type="button"
                  key={opt}
                  className={`btn-choice ${duration === opt ? 'selected' : ''}`}
                  onClick={() => setDuration(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>

          <div className="action-row" style={{ marginTop: '1.5rem' }}>
            <button type="button" className="btn btn-secondary" onClick={() => setStage('input')}>
              Back to Description
            </button>
            <button type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? 'Processing...' : 'Confirm & Continue'}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
