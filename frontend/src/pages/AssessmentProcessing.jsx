import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import LoadingScreen from '../components/LoadingScreen';
import ErrorMessage from '../components/ErrorMessage';
import { processAssessment } from '../services/assessmentService';
import { useAssessment } from '../hooks/useAssessment';
import { INPUT_TYPES } from '../utils/constants';

export default function AssessmentProcessing() {
  const navigate = useNavigate();
  const { assessmentId, inputTypes } = useAssessment();
  const [step, setStep] = useState(0);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  // Dynamically assemble only stages that are actually part of the selected workflow
  const activeSteps = useMemo(() => {
    const steps = ['Validating input'];
    if (inputTypes.includes(INPUT_TYPES.SYMPTOMS)) {
      steps.push('Symptoms NLP');
    }
    if (inputTypes.includes(INPUT_TYPES.MEDICAL_REPORT)) {
      steps.push('Medical Report OCR/NLP');
    }
    if (inputTypes.includes(INPUT_TYPES.XRAY)) {
      steps.push('X-Ray Analysis');
    }
    steps.push('Evidence Processing');
    steps.push('Safety Rules');
    steps.push('Preparing Result');
    return steps;
  }, [inputTypes]);

  useEffect(() => {
    if (!assessmentId) {
      setError('No assessment in progress.');
      return undefined;
    }

    let cancelled = false;
    let timer;

    const run = async () => {
      try {
        // Step through the active workflow stages
        for (let i = 0; i < activeSteps.length - 1; i += 1) {
          if (cancelled) return;
          setStep(i);
          await new Promise((r) => {
            timer = setTimeout(r, 400);
          });
        }

        // Execute real backend pipeline
        const result = await processAssessment(assessmentId);
        if (cancelled) return;

        setStep(activeSteps.length - 1);
        setMessage(result.message || 'Processing complete.');
        
        // Small pause to let user see final result stage
        await new Promise((r) => {
          timer = setTimeout(r, 350);
        });

        if (!cancelled) {
          navigate(`/assessment/${assessmentId}`, { replace: true });
        }
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    };

    run();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [assessmentId, activeSteps, navigate]);

  if (error) {
    return (
      <div className="page narrow">
        <ErrorMessage message={error} />
        <button type="button" className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="page narrow">
      <LoadingScreen message={activeSteps[step] || 'Processing assessment...'} />
      <ul className="process-steps">
        {activeSteps.map((label, idx) => (
          <li key={label} className={idx <= step ? 'active' : ''}>
            {label}
          </li>
        ))}
      </ul>
      {message ? <p className="muted text-center">{message}</p> : null}
    </div>
  );
}
