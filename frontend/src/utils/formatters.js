export function formatDate(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export function formatPathway(pathway) {
  if (!pathway) return 'Not determined';
  return String(pathway).charAt(0) + String(pathway).slice(1).toLowerCase();
}

export function pathwayClass(pathway) {
  const p = String(pathway || '').toUpperCase();
  if (p === 'EMERGENCY') return 'pathway-emergency';
  if (p === 'CONSULTATION') return 'pathway-consultation';
  if (p === 'MILD') return 'pathway-mild';
  return 'pathway-unknown';
}

export function pathwayLabel(pathway) {
  const p = String(pathway || '').toUpperCase();
  if (p === 'EMERGENCY') return 'Emergency';
  if (p === 'CONSULTATION') return 'Consultation';
  if (p === 'MILD') return 'Mild';
  return 'Not determined';
}

export function inputTypesLabel(types = []) {
  const map = {
    symptoms: 'Symptoms',
    medical_report: 'Medical Report',
    xray: 'X-Ray',
  };
  return (types || []).map((t) => map[t] || t).join(' + ') || 'None';
}
