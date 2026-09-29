import { apiRequest } from './api';

export async function uploadXray(assessmentId, file) {
  const form = new FormData();
  form.append('file', file);
  return apiRequest(`/assessments/${assessmentId}/xray`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

export async function getXrayResults(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/xray`);
}
