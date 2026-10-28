
import { apiRequest } from './api';

/**
 * Upload one X-ray image.
 * The backend detects the anatomical region from image pixels.
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

  return apiRequest(
    `/assessments/${encodeURIComponent(assessmentId)}/xray`,
    {
      method: 'POST',
      body: form,
      formData: true,
    }
  );
}

/**
 * Retrieve persisted X-ray results for an assessment.
 */
export async function getXrayResults(assessmentId) {
  if (!assessmentId) {
    throw new Error('An assessment ID is required.');
  }

  return apiRequest(
    `/assessments/${encodeURIComponent(assessmentId)}/xray`
  );
}

/**
 * Retrieve supported X-ray regions and model availability.
 */
export async function getXrayCatalogue() {
  return apiRequest('/assessments/xray/catalogue');
}

/**
 * Delete an uploaded X-ray image and its result.
 */
export async function deleteXray(assessmentId, xrayResultId) {
  if (!assessmentId || !xrayResultId) {
    throw new Error(
      'Assessment ID and X-ray result ID are required.'
    );
  }

  return apiRequest(
    `/assessments/${encodeURIComponent(assessmentId)}/xray/${encodeURIComponent(xrayResultId)}`,
    {
      method: 'DELETE',
    }
  );
}

/**
 * Confirm or correct an X-ray's anatomical region.
 * This selects a routing region; it is not a medical diagnosis.
 */
export async function confirmXrayAnatomy(
  assessmentId,
  xrayResultId,
  region
) {
  if (!assessmentId || !xrayResultId) {
    throw new Error(
      'Assessment ID and X-ray result ID are required.'
    );
  }

  if (typeof region !== 'string' || !region.trim()) {
    throw new Error(
      'Choose a specific body region before confirming.'
    );
  }

  return apiRequest(
    `/assessments/${encodeURIComponent(assessmentId)}/xray/${encodeURIComponent(xrayResultId)}/anatomy-verification`,
    {
      method: 'PUT',
      body: {
        region: region.trim(),
      },
    }
  );
}
