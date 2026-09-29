export default function ChatMessage({ role, content, provider, citations }) {
  return (
    <div className={`chat-message chat-${role}`}>
      <div className="chat-header">
        <span className="chat-role">{role === 'user' ? 'You' : 'MediSense AI Assistant'}</span>
        {provider ? <span className="provider-tag">via {provider}</span> : null}
      </div>
      <p className="chat-content">{content}</p>
      {citations && citations.length > 0 ? (
        <div className="chat-citations">
          <span className="citation-title">Trusted Guidelines / Citations:</span>
          <ul>
            {citations.map((c, i) => (
              <li key={i}>
                <strong>{c.topic}</strong> &mdash; <em>{c.source}</em>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}
