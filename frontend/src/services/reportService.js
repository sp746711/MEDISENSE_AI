import { apiRequest } from './api';

export async function uploadReport(assessmentId, file, reportCategory = 'medical_report') {
  const form = new FormData();
  form.append('file', file);
  if (reportCategory) {
    form.append('report_category', reportCategory);
  }
  return apiRequest(`/assessments/${assessmentId}/reports`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

export async function getReports(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/reports`);
}
