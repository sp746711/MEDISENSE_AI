/**
 * Backend-Driven X-Ray Body Region Utility.
 *
 * All anatomical regions, canonical identifiers, search aliases, and model capabilities
 * are fetched dynamically from the backend API (/assessments/xray/catalogue).
 * This module provides case-insensitive search and filtering helpers without hardcoding
 * a separate master catalogue.
 */

/**
 * Filter a backend-provided list of body regions based on a user's search query.
 * Matches against region name, canonical identifier, and backend aliases.
 *
 * @param {Array} regions - List of region objects returned from backend
 * @param {string} query - Search string entered by user
 * @returns {Array} Matching region items (empty when query is blank)
 */
export function filterBackendRegions(regions = [], query = '') {
  const cleanQ = (query || '').toLowerCase().trim();
  if (!cleanQ) return [];

  // Split query into tokens for multi-term search
  const tokens = cleanQ.split(/\s+/).filter(Boolean);

  return (regions || []).filter((reg) => {
    const nameNorm = (reg.name || '').toLowerCase();
    const idNorm = (reg.id || '').toLowerCase();
    const canonNorm = (reg.canonical_region || '').toLowerCase();
    const aliases = (reg.aliases || []).map((a) => (a || '').toLowerCase());

    return tokens.every((token) => {
      return (
        nameNorm.includes(token) ||
        idNorm.includes(token) ||
        canonNorm.includes(token) ||
        aliases.some((alias) => alias.includes(token))
      );
    });
  });
}

/**
 * Find a specific region item from the backend-provided list by canonical region or ID.
 */
export function findRegionInList(regions = [], target = '') {
  if (!target) return null;
  const clean = target.toLowerCase().trim();
  return (
    (regions || []).find(
      (r) =>
        (r.canonical_region && r.canonical_region.toLowerCase() === clean) ||
        (r.id && r.id.toLowerCase() === clean) ||
        (r.name && r.name.toLowerCase() === clean)
    ) || null
  );
}
