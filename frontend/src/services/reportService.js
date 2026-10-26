import { apiRequest } from './api';

/**
 * Upload a medical report or radiology report.
 */
export async function uploadReport(
  assessmentId,
  file,
  reportCategory = 'medical_report',
  selectedRegion = null
) {
  if (!assessmentId) {
    throw new Error('An assessment ID is required to upload a report.');
  }

  if (!file) {
    throw new Error('Please select a report file to upload.');
  }

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
 * Retrieve reports belonging to an assessment.
 */
export async function getReports(assessmentId) {
  if (!assessmentId) {
    throw new Error('An assessment ID is required to retrieve reports.');
  }

  return apiRequest(`/assessments/${assessmentId}/reports`);
}

/**
 * Delete an uploaded radiology report and its saved findings.
 */
export async function deleteReport(assessmentId, reportId) {
  if (!assessmentId || !reportId) {
    throw new Error('Assessment ID and report ID are required.');
  }

  return apiRequest(
    `/assessments/${assessmentId}/reports/${reportId}`,
    { method: 'DELETE' }
  );
}