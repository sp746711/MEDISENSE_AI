import { apiRequest } from './api';

export async function searchDoctors(params) {
  const q = new URLSearchParams(params).toString();
  return apiRequest(`/doctors?${q}`);
}

export async function getDoctor(id) {
  return apiRequest(`/doctors/${id}`);
}
