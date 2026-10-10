import { apiRequest } from './api';

export async function submitSymptoms(assessmentId, rawText, followupAnswers = null) {
  const body = { raw_text: rawText };
  if (followupAnswers && Object.keys(followupAnswers).length > 0) {
    body.followup_answers = followupAnswers;
  }
  return apiRequest(`/assessments/${assessmentId}/symptoms`, {
    method: 'POST',
    body,
  });
}

export async function analyzeSymptoms(assessmentId, rawText) {
  return apiRequest(`/assessments/${assessmentId}/symptoms/analyze`, {
    method: 'POST',
    body: { raw_text: rawText },
  });
}

export async function getSymptoms(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/symptoms`);
}
