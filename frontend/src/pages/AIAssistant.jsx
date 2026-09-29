import { useEffect, useState } from 'react';
import ChatInput from '../components/ChatInput';
import ChatMessage from '../components/ChatMessage';
import ErrorMessage from '../components/ErrorMessage';
import { chat } from '../services/assistantService';
import { listAssessments } from '../services/assessmentService';
import { formatPathway, inputTypesLabel } from '../utils/formatters';

const QUICK = [
  'Explain my latest assessment.',
  'Why was consultation suggested?',
  'What information did you use?',
  'What information is missing?',
  'Explain my medical report.',
  'Explain my X-ray result.',
  'What did the X-ray model detect?',
  'Why wasn\'t my X-ray interpreted?',
  'Explain this medical term.',
  'What questions should I ask my doctor?',
  'What warning signs should I watch for?',
];

export default function AIAssistant() {
  const [latest, setLatest] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const list = await listAssessments();
        setLatest(list?.[0] || null);
      } catch (err) {
        setError(err.message);
      }
    })();
  }, []);

  const send = async (text) => {
    setError('');
    setMessages((m) => [...m, { role: 'user', content: text }]);
    setBusy(true);
    try {
      const res = await chat({
        message: text,
        assessment_id: latest?.assessment_id || null,
        session_id: sessionId,
      });
      if (res.session_id) setSessionId(res.session_id);
      setMessages((m) => [
        ...m,
        {
          role: 'assistant',
          content: res.answer || res.message || 'AI Assistant is temporarily unavailable.',
          provider: res.provider,
          citations: res.citations || [],
        },
      ]);
    } catch (err) {
      setError(err.message);
      setMessages((m) => [
        ...m,
        { role: 'assistant', content: 'AI Assistant is temporarily unavailable.' },
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="page assistant-page">
      <h1>AI Assistant</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />

      <section className="assistant-context">
        <h2>Latest Assessment Context</h2>
        {!latest ? (
          <p className="muted">No assessment available yet.</p>
        ) : (
          <>
            <p><strong>Status:</strong> {latest.status}</p>
            <p><strong>Pathway:</strong> {formatPathway(latest.pathway)}</p>
            <p><strong>Inputs:</strong> {inputTypesLabel(latest.input_types)}</p>
            <p><strong>Suggested specialty:</strong> {latest.specialty || '—'}</p>
          </>
        )}
      </section>

      <section className="section">
        <h2>Quick actions</h2>
        <div className="chip-row">
          {QUICK.map((q) => (
            <button key={q} type="button" className="chip" disabled={busy} onClick={() => send(q)}>
              {q}
            </button>
          ))}
        </div>
      </section>

      <div className="chat-window">
        {!messages.length ? (
          <p className="muted">Ask a question about your assessment. The assistant will not invent medical findings.</p>
        ) : (
          messages.map((m, idx) => (
            <ChatMessage
              key={`${m.role}-${idx}`}
              role={m.role}
              content={m.content}
              provider={m.provider}
              citations={m.citations}
            />
          ))
        )}
        {busy ? <p className="muted">Generating response...</p> : null}
      </div>
      <ChatInput onSend={send} disabled={busy} />
    </div>
  );
}
