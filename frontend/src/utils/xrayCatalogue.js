
/**
 * Backend-Driven X-Ray Body Region Utility.
 *
 * Region names, canonical identifiers, aliases, and model capabilities are supplied
 * by the backend catalogue endpoint. This module only normalizes, searches, and
 * resolves those supplied records; it does not invent regions or model support.
 */

function toText(value) {
  return typeof value === 'string' || typeof value === 'number'
    ? String(value)
    : '';
}

function normalizeSearchText(value) {
  return toText(value)
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim()
    .replace(/\s+/g, ' ');
}

function getRegionSearchFields(region) {
  if (!region || typeof region !== 'object' || Array.isArray(region)) {
    return [];
  }

  const aliases = Array.isArray(region.aliases)
    ? region.aliases
    : typeof region.aliases === 'string'
      ? [region.aliases]
      : [];

  return [
    region.name,
    region.id,
    region.canonical_region,
    ...aliases,
  ]
    .map(normalizeSearchText)
    .filter(Boolean);
}

/**
 * Search backend-provided region records.
 * Every query token must match at least one searchable field.
 */
export function filterBackendRegions(regions = [], query = '') {
  if (!Array.isArray(regions)) return [];

  const normalizedQuery = normalizeSearchText(query);
  if (!normalizedQuery) return [];

  const tokens = normalizedQuery.split(' ').filter(Boolean);
  if (!tokens.length) return [];

  return regions.filter((region) => {
    const fields = getRegionSearchFields(region);
    if (!fields.length) return false;

    return tokens.every((token) =>
      fields.some((field) => field.includes(token))
    );
  });
}

/**
 * Find a region by canonical identifier, ID, name, or alias.
 * Exact normalized matches prevent accidental region substitutions.
 */
export function findRegionInList(regions = [], target = '') {
  if (!Array.isArray(regions)) return null;

  const normalizedTarget = normalizeSearchText(target);
  if (!normalizedTarget) return null;

  const validRegions = regions.filter(
    (region) =>
      region &&
      typeof region === 'object' &&
      !Array.isArray(region)
  );

  const exactMatch = validRegions.find((region) =>
    getRegionSearchFields(region).some(
      (field) => field === normalizedTarget
    )
  );

  return exactMatch || null;
}
