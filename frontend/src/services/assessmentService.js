import { apiRequest } from './api';

export async function createAssessment(inputTypes) {
  return apiRequest('/assessments', {
    method: 'POST',
    body: { input_types: inputTypes },
  });
}

export async function listAssessments() {
  return apiRequest('/assessments');
}

export async function getAssessment(id) {
  return apiRequest(`/assessments/${id}`);
}

export async function processAssessment(id) {
  return apiRequest(`/assessments/${id}/process`, { method: 'POST' });
}

export async function getAssessmentResult(id) {
  return apiRequest(`/assessments/${id}/result`);
}

export async function getDashboardSummary() {
  return apiRequest('/dashboard/summary');
}
