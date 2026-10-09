import { apiRequest } from './api';

export async function uploadXray(assessmentId, file, region) {
  const form = new FormData();
  form.append('file', file);
  if (region) {
    form.append('region', region);
  }
  return apiRequest(`/assessments/${assessmentId}/xray`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

export async function getXrayResults(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/xray`);
}
