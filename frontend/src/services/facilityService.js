import { apiRequest } from './api';

export async function searchFacilities(params) {
  const q = new URLSearchParams(params).toString();
  return apiRequest(`/facilities?${q}`);
}
