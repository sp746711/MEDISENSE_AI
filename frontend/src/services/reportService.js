import { apiRequest } from './api';

/**
 * Upload a medical or radiology report.
 *
 * @param {string|number} assessmentId
 * @param {File} file
 * @param {string} reportCategory
 * @param {string|null} selectedRegion
 * @returns {Promise<object>}
 */
export async function uploadReport(
  assessmentId,
  file,
  reportCategory = 'medical_report',
  selectedRegion = null
) {
  const form = new FormData();

  form.append('file', file);

  if (reportCategory) {
    form.append('report_category', reportCategory);
  }

  if (selectedRegion && String(selectedRegion).trim()) {
    form.append('selected_region', String(selectedRegion).trim());
  }

  return apiRequest(`/assessments/${assessmentId}/reports`, {
    method: 'POST',
    body: form,
    formData: true,
  });
}

/**
 * Retrieve reports associated with an assessment.
 *
 * @param {string|number} assessmentId
 * @returns {Promise<object>}
 */
export async function getReports(assessmentId) {
  return apiRequest(`/assessments/${assessmentId}/reports`);
}