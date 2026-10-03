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

export async function searchMedicalShops(params = {}) {
  const q = new URLSearchParams(cleanParams(params)).toString();
  return apiRequest(`/medical-shops?${q}`);
}
