import { apiRequest } from './api';

/**
 * Upload an X-ray image.
 * The backend automatically detects the body part.
 */
export async function uploadXray(assessmentId, file) {
  if (!assessmentId) {
    throw new Error('An assessment ID is required.');
  }

  if (!file) {
    throw new Error('Please select an X-ray image.');
  }

  const form = new FormData();
  form.append('file', file);

  return apiRequest(`/assessments/${assessmentId}/xray`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

/**
 * Retrieve X-ray results for an assessment.
 */
export async function getXrayResults(assessmentId) {
  if (!assessmentId) {
    throw new Error('An assessment ID is required.');
  }

  return apiRequest(`/assessments/${assessmentId}/xray`);
}

/**
 * Retrieve the X-ray model catalogue.
 */
export async function getXrayCatalogue() {
  return apiRequest('/assessments/xray/catalogue');
}

/**
 * Delete an uploaded X-ray image and its backend record.
 */
export async function deleteXray(assessmentId, xrayResultId) {
  if (!assessmentId || !xrayResultId) {
    throw new Error('Assessment ID and X-ray result ID are required.');
  }

  return apiRequest(
    `/assessments/${assessmentId}/xray/${xrayResultId}`,
    {
      method: 'DELETE',
    }
  );
}