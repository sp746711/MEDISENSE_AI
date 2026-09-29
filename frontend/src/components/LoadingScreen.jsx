export default function LoadingScreen({ message = 'Loading...' }) {
  return (
    <div className="loading-screen" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
}
