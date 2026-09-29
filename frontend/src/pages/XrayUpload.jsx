import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ErrorMessage from '../components/ErrorMessage';
import FileUpload from '../components/FileUpload';
import { uploadXray } from '../services/xrayService';
import { useAssessment } from '../hooks/useAssessment';

export default function XrayUpload() {
  const navigate = useNavigate();
  const { assessmentId } = useAssessment();
  const [file, setFile] = useState(null);
  const [info, setInfo] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const onContinue = async () => {
    setError('');
    if (!assessmentId) return setError('No assessment in progress.');
    if (!file) return setError('Please select an X-ray image.');
    setSubmitting(true);
    try {
      const result = await uploadXray(assessmentId, file);
      setInfo(result.message || 'X-ray uploaded.');
      navigate('/assessment/processing');
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="page narrow">
      <h1>X-Ray Upload</h1>
      <p className="muted">Upload JPG, JPEG, or PNG. Unsupported regions return an unavailable status — never a fabricated finding.</p>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {info ? <p className="info-banner">{info}</p> : null}
      <FileUpload
        accept=".jpg,.jpeg,.png"
        label={file ? file.name : 'Choose X-ray image'}
        onFile={setFile}
      />
      <button type="button" className="btn btn-primary" onClick={onContinue} disabled={submitting}>
        {submitting ? 'Uploading...' : 'Continue'}
      </button>
    </div>
  );
}
