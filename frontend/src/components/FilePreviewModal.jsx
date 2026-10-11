import { useState, useEffect } from 'react';

export default function FilePreviewModal({ file, previewUrl, previewType, onClose }) {
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
