/**
 * Central API client for MediSense AI.
 * All services should use this module — do not hard-code production URLs.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

function getToken() {
  return localStorage.getItem('medisense_token');
}

export function setAuthToken(token) {
  if (token) localStorage.setItem('medisense_token', token);
  else localStorage.removeItem('medisense_token');
}

export function clearAuthToken() {
  localStorage.removeItem('medisense_token');
  localStorage.removeItem('medisense_user');
}

export class ApiError extends Error {
  constructor(message, status, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

function friendlyMessage(status, detail) {
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
  }
  switch (status) {
    case 401:
      return 'Please sign in again.';
    case 403:
      return 'You do not have permission to perform this action.';
    case 404:
      return 'The requested resource was not found.';
    case 422:
      return 'Please check the information you entered.';
    case 500:
      return 'Something went wrong on the server. Please try again.';
    default:
      return 'Request failed. Please try again.';
  }
}

export async function apiRequest(path, options = {}) {
  const {
    method = 'GET',
    body,
    token = getToken(),
    formData = false,
    headers: customHeaders = {},
  } = options;

  const headers = { ...customHeaders };
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body != null && !formData) {
    headers['Content-Type'] = 'application/json';
  }

  let response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body == null ? undefined : formData ? body : JSON.stringify(body),
    });
  } catch {
    throw new ApiError('Unable to reach the server. Check that the backend is running.', 0);
  }

  if (response.status === 204) return null;

  let data = null;
  const text = await response.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    if (response.status === 401) {
      // Let AuthContext handle redirect; still throw
    }
    const detail = data?.detail ?? data?.message ?? null;
    throw new ApiError(friendlyMessage(response.status, detail), response.status, data);
  }

  return data;
}

export async function checkHealth() {
  return apiRequest('/health', { token: null });
}

export { API_BASE_URL };
