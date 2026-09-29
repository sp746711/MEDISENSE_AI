import { apiRequest } from './api';

export async function submitSymptoms(assessmentId, rawText) {
  return apiRequest(`/assessments/${assessmentId}/symptoms`, {
    method: 'POST',
    body: { raw_text: rawText },
  });
}

export async function getSymptoms(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/symptoms`);
}
