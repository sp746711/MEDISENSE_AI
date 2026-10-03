import {
  apiRequest,
  setAuthToken,
  clearAuthToken,
  setRefreshToken,
  getRefreshToken,
} from './api';

export async function register(payload) {
  const data = await apiRequest('/auth/register', {
    method: 'POST',
    body: payload,
    token: null,
  });
  setAuthToken(data.access_token);
  if (data.refresh_token) {
    setRefreshToken(data.refresh_token);
  }
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
  if (data.refresh_token) {
    setRefreshToken(data.refresh_token);
  }
  localStorage.setItem('medisense_user', JSON.stringify(data.user));
  return data;
}

export async function refreshAccessToken() {
  const refreshToken = getRefreshToken();
  if (!refreshToken) {
    throw new Error('No refresh token available');
  }
  const data = await apiRequest('/auth/refresh', {
    method: 'POST',
    body: { refresh_token: refreshToken },
    token: null,
    _retry: true,
  });
  if (data?.access_token) {
    setAuthToken(data.access_token);
    return data.access_token;
  }
  throw new Error('Failed to refresh access token');
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

export { getRefreshToken, setRefreshToken };

