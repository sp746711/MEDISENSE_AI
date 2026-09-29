export default function FileUpload({
  accept,
  label = 'Choose file',
  onFile,
  disabled = false,
  hint,
}) {
  return (
    <div className="file-upload">
      <label className="file-upload-label">
        <span>{label}</span>
        <input
          type="file"
          accept={accept}
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile?.(file);
          }}
        />
      </label>
      {hint ? <p className="muted">{hint}</p> : null}
    </div>
  );
}
