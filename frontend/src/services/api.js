/**
 * Central API client for MediSense AI.
 * All services should use this module — do not hard-code production URLs.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

function getToken() {
  return localStorage.getItem('medisense_token');
}

export function getRefreshToken() {
  return localStorage.getItem('medisense_refresh_token');
}

export function setAuthToken(token) {
  if (token) localStorage.setItem('medisense_token', token);
  else localStorage.removeItem('medisense_token');
}

export function setRefreshToken(token) {
  if (token) localStorage.setItem('medisense_refresh_token', token);
  else localStorage.removeItem('medisense_refresh_token');
}

export function clearAuthToken() {
  localStorage.removeItem('medisense_token');
  localStorage.removeItem('medisense_refresh_token');
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

let refreshPromise = null;

async function executeTokenRefresh() {
  const refreshToken = localStorage.getItem('medisense_refresh_token');
  if (!refreshToken) {
    clearAuthToken();
    throw new ApiError('No refresh token available. Please sign in again.', 401);
  }

  let res;
  try {
    res = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  } catch {
    throw new ApiError('Unable to reach the server to refresh token.', 0);
  }

  let data = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!res.ok) {
    clearAuthToken();
    const detail = data?.detail ?? data?.message ?? 'Session expired. Please sign in again.';
    throw new ApiError(friendlyMessage(res.status, detail), res.status, data);
  }

  if (!data?.access_token) {
    clearAuthToken();
    throw new ApiError('No access token returned from refresh endpoint.', 401);
  }

  setAuthToken(data.access_token);
  return data.access_token;
}

function requestNewToken() {
  if (!refreshPromise) {
    refreshPromise = executeTokenRefresh().finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
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
    // Attempt automatic token refresh on 401
    if (
      response.status === 401 &&
      !options._retry &&
      options.token !== null &&
      path !== '/auth/refresh' &&
      path !== '/auth/login' &&
      path !== '/auth/register'
    ) {
      const refreshToken = localStorage.getItem('medisense_refresh_token');
      if (refreshToken) {
        try {
          const newAccessToken = await requestNewToken();
          // Retry the original request ONCE with new access token
          return await apiRequest(path, {
            ...options,
            _retry: true,
            token: newAccessToken,
          });
        } catch (refreshErr) {
          clearAuthToken();
          throw refreshErr instanceof ApiError
            ? refreshErr
            : new ApiError('Session expired. Please sign in again.', 401);
        }
      } else {
        clearAuthToken();
      }
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
