import { apiRequest } from './api';

export async function searchMedicalShops(params) {
  const q = new URLSearchParams(params).toString();
  return apiRequest(`/medical-shops?${q}`);
}
