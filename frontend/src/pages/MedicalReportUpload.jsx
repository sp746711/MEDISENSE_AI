import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import FileUpload from '../components/FileUpload';
import { uploadReport } from '../services/reportService';
import { useAssessment } from '../hooks/useAssessment';
import { INPUT_TYPES } from '../utils/constants';

export default function MedicalReportUpload() {
  const navigate = useNavigate();
  const { assessmentId, inputTypes } = useAssessment();
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const nextPath = () => {
    if (inputTypes.includes(INPUT_TYPES.XRAY)) return '/assessment/xray';
    return '/assessment/processing';
  };

  const onContinue = async () => {
    setError('');
    if (!assessmentId) return setError('No assessment in progress.');
    if (!file) return setError('Please select a report file.');
    setSubmitting(true);
    try {
      const result = await uploadReport(assessmentId, file);
      setMessage(result.message || 'Report uploaded.');
      navigate(nextPath());
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page narrow">
      <h1>Medical Report</h1>
      <p className="muted">Upload PDF, JPG, JPEG, or PNG. Includes laboratory/blood/radiology reports.</p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {message ? <p className="info-banner">{message}</p> : null}
      <FileUpload
        accept=".pdf,.jpg,.jpeg,.png"
        label={file ? file.name : 'Choose medical report'}
        onFile={setFile}
        hint="Maximum size configured by the server. Executable files are rejected."
      />
      <button type="button" className="btn btn-primary" onClick={onContinue} disabled={submitting}>
        {submitting ? 'Uploading...' : 'Continue'}
      </button>
    </div>
  );
}
