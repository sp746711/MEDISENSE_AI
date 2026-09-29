export default function ErrorMessage({ message, onDismiss }) {
  if (!message) return null;
  return (
    <div className="error-banner" role="alert">
      <span>{message}</span>
      {onDismiss ? (
        <button type="button" className="linkish" onClick={onDismiss} aria-label="Dismiss">
          Dismiss
        </button>
      ) : null}
    </div>
  );
}
