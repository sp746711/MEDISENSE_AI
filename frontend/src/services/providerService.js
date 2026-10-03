import { apiRequest } from './api';

function cleanParams(params = {}) {
  const clean = {};
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') {
      clean[k] = v;
    }
  }
  return clean;
}

export async function searchDoctors(params = {}) {
  const q = new URLSearchParams(cleanParams(params)).toString();
  return apiRequest(`/doctors?${q}`);
}

export async function getDoctor(id) {
  return apiRequest(`/doctors/${id}`);
}
