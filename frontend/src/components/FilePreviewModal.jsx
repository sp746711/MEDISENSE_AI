import { useState, useEffect } from 'react';

export default function FilePreviewModal({
  file,
  previewUrl,
  previewType,
  onClose,
  detectedRegion = null,
  candidateRegion = null,
  bodyPartConfidence = null,
  anatomyVerificationStatus = 'unverified',
}) {
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!file && !previewUrl) return null;

  const fileName = file?.name || 'Uploaded Document';
  const fileSize = file?.size
    ? file.size < 1024 * 1024
      ? `${(file.size / 1024).toFixed(1)} KB`
      : `${(file.size / (1024 * 1024)).toFixed(1)} MB`
    : '';

  const isPdf = previewType === 'pdf' || fileName.toLowerCase().endsWith('.pdf');
  const isImage =
    previewType === 'image' ||
    /\.(png|jpe?g|webp)$/i.test(fileName);

  return (
    <div
      className="preview-modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label={`Preview of ${fileName}`}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="preview-modal-container">
        {/* Modal Header */}
        <div className="preview-modal-header">
          <div className="preview-modal-title-group">
            <div className="preview-badge-pill">
              <span className="preview-badge-dot" />
              <span>File Verification Only · Not AI Processing</span>
            </div>
            <h2 className="preview-modal-filename" title={fileName}>
              {fileName}
            </h2>
            {fileSize ? (
              <span className="preview-modal-meta">{fileSize}</span>
            ) : null}
          </div>

          <div className="preview-modal-actions">
            {/* Zoom Controls for Images & Scalable Views */}
            {isImage ? (
              <div className="preview-zoom-group">
                <button
                  type="button"
                  className="preview-zoom-btn"
                  onClick={() => setZoom((z) => Math.max(0.5, z - 0.25))}
                  title="Zoom Out"
                  aria-label="Zoom out"
                >
                  −
                </button>
                <span className="preview-zoom-level">{Math.round(zoom * 100)}%</span>
                <button
                  type="button"
                  className="preview-zoom-btn"
                  onClick={() => setZoom((z) => Math.min(3, z + 0.25))}
                  title="Zoom In"
                  aria-label="Zoom in"
                >
                  +
                </button>
                {zoom !== 1 ? (
                  <button
                    type="button"
                    className="preview-zoom-reset"
                    onClick={() => setZoom(1)}
                  >
                    Reset
                  </button>
                ) : null}
              </div>
            ) : null}

            <button
              type="button"
              className="preview-close-btn"
              onClick={onClose}
              aria-label={`Close preview of ${fileName}`}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Modal Viewer Content */}
        <div className="preview-modal-content">
          {isPdf ? (
            <div className="preview-pdf-wrapper">
              {previewUrl ? (
                <object
                  data={previewUrl}
                  type="application/pdf"
                  className="preview-pdf-object"
                  aria-label={`PDF view for ${fileName}`}
                >
                  <iframe
                    src={previewUrl}
                    title={fileName}
                    className="preview-pdf-iframe"
                  />
                  <div className="preview-unavailable-box">
                    <p>Browser PDF preview unavailable for this document.</p>
                    <a
                      href={previewUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="preview-open-link"
                    >
                      Open PDF in new tab
                    </a>
                  </div>
                </object>
              ) : (
                <div className="preview-unavailable-box">
                  <p>Preview unavailable for this file.</p>
                </div>
              )}
            </div>
          ) : isImage ? (
            <div className="preview-image-wrapper">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt={`Preview of ${fileName}`}
                  className="preview-image-element"
                  style={{ transform: `scale(${zoom})`, transformOrigin: 'top center' }}
                />
              ) : (
                <div className="preview-unavailable-box">
                  <p>Preview unavailable for this image.</p>
                </div>
              )}
            </div>
          ) : (
            <div className="preview-unavailable-box">
              <p>Preview unavailable for this file format.</p>
              <p className="small muted">
                File details: {fileName} ({fileSize || 'Unknown size'})
              </p>
            </div>
          )}

          {isImage && (detectedRegion || candidateRegion) ? (
            <section
              aria-label="Automatic X-ray body-region classification"
              style={{
                margin: '1rem auto',
                padding: '0.85rem 1rem',
                width: 'min(100% - 2rem, 600px)',
                boxSizing: 'border-box',
                border: '1px solid #dbeafe',
                borderRadius: '10px',
                background: '#f8fbff',
                color: '#0f172a',
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: '0.4rem', color: '#075985' }}>
                Automatic body-region detection
              </div>
              {detectedRegion ? (
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Detected region:</strong>{' '}
                  {String(detectedRegion).replace(/_/g, ' ')}
                </p>
              ) : (
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Unverified candidate:</strong>{' '}
                  {String(candidateRegion).replace(/_/g, ' ')}
                </p>
              )}
              {typeof bodyPartConfidence === 'number' && Number.isFinite(bodyPartConfidence) ? (
                <p style={{ margin: '0.25rem 0' }}>
                  <strong>Classifier confidence:</strong>{' '}
                  {(bodyPartConfidence * 100).toFixed(1)}%
                  <span style={{ color: '#64748b' }}> (not a calibrated clinical probability)</span>
                </p>
              ) : null}
              <p style={{ margin: '0.5rem 0 0', fontSize: '0.82rem', color: '#64748b' }}>
                This is an automated image classification, not a confirmed medical finding. Verify the body region before relying on subsequent analysis.
                {String(anatomyVerificationStatus).toLowerCase() !== 'completed' && !detectedRegion
                  ? ' The region has not been verified.'
                  : ''}
              </p>
            </section>
          ) : null}
        </div>

        {/* Modal Footer Note */}
        <div className="preview-modal-footer">
          <span className="preview-footer-note">
            Please verify this is the intended diagnostic file. Medical OCR, NLP, and model evaluation will only run after you click Continue.
          </span>
          <button
            type="button"
            className="preview-done-btn"
            onClick={onClose}
          >
            Done Viewing
          </button>
        </div>
      </div>
    </div>
  );
}
