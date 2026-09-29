import { useState } from 'react';

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('');

  const submit = (e) => {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend?.(trimmed);
    setValue('');
  };

  return (
    <form className="chat-input" onSubmit={submit}>
      <input
        type="text"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="Ask about your assessment..."
        disabled={disabled}
        aria-label="Chat message"
      />
      <button type="submit" className="btn btn-primary" disabled={disabled || !value.trim()}>
        Send
      </button>
    </form>
  );
}
