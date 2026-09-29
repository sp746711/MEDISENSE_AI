import { apiRequest } from './api';

export async function chat(payload) {
  return apiRequest('/assistant/chat', { method: 'POST', body: payload });
}

export async function listSessions() {
  return apiRequest('/assistant/sessions');
}
