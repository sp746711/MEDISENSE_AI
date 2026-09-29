import { apiRequest } from './api';

export async function listAppointments() {
  return apiRequest('/appointments');
}

export async function createAppointment(payload) {
  return apiRequest('/appointments', { method: 'POST', body: payload });
}

export async function cancelAppointment(id) {
  return apiRequest(`/appointments/${id}/cancel`, { method: 'PATCH' });
}
