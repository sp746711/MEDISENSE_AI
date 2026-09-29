import { apiRequest } from './api';

export async function uploadReport(assessmentId, file) {
  const form = new FormData();
  form.append('file', file);
  return apiRequest(`/assessments/${assessmentId}/reports`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

export async function getReports(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/reports`);
}
