export function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email || '').trim());
}

export function validatePassword(password) {
  if (!password || password.length < 8) {
    return 'Password must be at least 8 characters.';
  }
  return null;
}

export function validateRequired(value, label = 'This field') {
  if (!value || !String(value).trim()) return `${label} is required.`;
  return null;
}
