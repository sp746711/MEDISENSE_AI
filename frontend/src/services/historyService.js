import { apiRequest } from './api';

export async function getHistory() {
  return apiRequest('/history');
}
