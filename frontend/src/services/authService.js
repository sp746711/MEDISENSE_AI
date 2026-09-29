import { apiRequest, setAuthToken, clearAuthToken } from './api';

export async function register(payload) {
  const data = await apiRequest('/auth/register', {
    method: 'POST',
    body: payload,
    token: null,
  });
  setAuthToken(data.access_token);
  localStorage.setItem('medisense_user', JSON.stringify(data.user));
  return data;
}

export async function login(payload) {
  const data = await apiRequest('/auth/login', {
    method: 'POST',
    body: payload,
    token: null,
  });
  setAuthToken(data.access_token);
  localStorage.setItem('medisense_user', JSON.stringify(data.user));
  return data;
}

export async function getMe() {
  return apiRequest('/auth/me');
}

export function logout() {
  clearAuthToken();
}

export function getStoredUser() {
  try {
    const raw = localStorage.getItem('medisense_user');
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
