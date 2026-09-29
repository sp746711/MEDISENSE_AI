import { apiRequest } from './api';

export async function getProfile() {
  return apiRequest('/users/me');
}

export async function updateProfile(payload) {
  return apiRequest('/users/me', { method: 'PUT', body: payload });
}

export async function changePassword(payload) {
  return apiRequest('/users/me/password', { method: 'POST', body: payload });
}
