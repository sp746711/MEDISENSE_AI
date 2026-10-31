import { useState, useRef, useEffect, useMemo } from 'react';

/**
 * MASTER CANONICAL 33-CLASS ANATOMY CATALOGUE
 * Complete configured catalogue with canonical identifiers, readable titles,
 * broad/specific classifications, comprehensive aliases, and model support status.
 */
export const ALL_33_ANATOMY_CLASSES = [
  {
    canonical: 'ABDOMEN',
    label: 'Abdomen',
    category: 'Abdomen & Pelvis',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Abdomen. Preserves verified anatomy in clinical report.',
    aliases: ['abdomen', 'abdominal', 'belly', 'tummy', 'peritoneal', 'peritoneum', 'gut', 'gi tract', 'stomach area'],
  },
  {
    canonical: 'ANKLE',
    label: 'Ankle',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Ankle. Preserves verified anatomy in clinical report.',
    aliases: ['ankle', 'mortise', 'talus', 'malleolus', 'ankle joint', 'distal fibula'],
  },
  {
    canonical: 'ARM',
    label: 'Arm (Humerus)',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_HUMERUS (DenseNet121)',
    modelNotice: 'Configured with MURA XR_HUMERUS deep learning abnormality pipeline.',
    aliases: ['arm', 'humerus', 'upper arm', 'brachium', 'bicep area'],
  },
  {
    canonical: 'CERVICAL_SPINE',
    label: 'Cervical Spine (C-Spine)',
    category: 'Spine',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Cervical Spine. Preserves verified anatomy in clinical report.',
    aliases: ['cervical spine', 'c-spine', 'cspine', 'cervical', 'neck spine', 'c1-c7', 'cervical vertebrae'],
  },
  {
    canonical: 'CHEST',
    label: 'Chest (Thorax / CXR)',
    category: 'Cardiopulmonary',
    isBroad: false,
    modelSupported: true,
    modelName: 'CheXNet / DenseNet121 (14 Thoracic Conditions)',
    modelNotice: 'Configured with CheXNet deep learning cardiopulmonary screening model.',
    aliases: ['chest', 'thorax', 'lungs', 'pulmonary', 'cxr', 'chest x-ray', 'ribcage', 'heart and lungs'],
  },
  {
    canonical: 'DORSAL_SPINE',
    label: 'Dorsal Spine (Thoracic / T-Spine)',
    category: 'Spine',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Dorsal Spine. Preserves verified anatomy in clinical report.',
    aliases: ['dorsal spine', 'thoracic spine', 't-spine', 'tspine', 'mid back', 'dorsal', 't1-t12', 'thoracic vertebrae'],
  },
  {
    canonical: 'ELBOW',
    label: 'Elbow',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_ELBOW (DenseNet121)',
    modelNotice: 'Configured with MURA XR_ELBOW deep learning abnormality pipeline.',
    aliases: ['elbow', 'olecranon', 'cubital', 'elbow joint', 'radial head', 'trochlea'],
  },
  {
    canonical: 'EXTREMITY',
    label: 'Extremity (Broad Limb)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Please select a specific joint or bone for targeted model execution.',
    aliases: ['extremity', 'limb', 'limbs', 'general limb', 'extremities'],
  },
  {
    canonical: 'FEMUR',
    label: 'Femur (Thigh)',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Femur. Preserves verified anatomy in clinical report.',
    aliases: ['femur', 'thigh', 'femoral shaft', 'thigh bone'],
  },
  {
    canonical: 'FINGER',
    label: 'Finger (Digits)',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_FINGER (DenseNet121)',
    modelNotice: 'Configured with MURA XR_FINGER deep learning abnormality pipeline.',
    aliases: ['finger', 'fingers', 'digits', 'thumb', 'phalanx', 'phalanges', 'mcp joint'],
  },
  {
    canonical: 'FOOT',
    label: 'Foot',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Foot. Preserves verified anatomy in clinical report.',
    aliases: ['foot', 'feet', 'metatarsals', 'tarsals', 'forefoot', 'midfoot', 'hindfoot', 'toes'],
  },
  {
    canonical: 'FOREARM',
    label: 'Forearm (Radius & Ulna)',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_FOREARM (DenseNet121)',
    modelNotice: 'Configured with MURA XR_FOREARM deep learning abnormality pipeline.',
    aliases: ['forearm', 'radius and ulna', 'radius', 'ulna', 'antebrachium'],
  },
  {
    canonical: 'HAND',
    label: 'Hand',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_HAND (DenseNet121)',
    modelNotice: 'Configured with MURA XR_HAND deep learning abnormality pipeline.',
    aliases: ['hand', 'palm', 'metacarpals', 'manus', 'hand bones'],
  },
  {
    canonical: 'HEAD',
    label: 'Head (Broad Cranial)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Select Skull, PNS, or Nasopharynx for specific evaluation.',
    aliases: ['head', 'cranial', 'head x-ray', 'cephalic'],
  },
  {
    canonical: 'HEEL',
    label: 'Heel (Calcaneus)',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Heel. Preserves verified anatomy in clinical report.',
    aliases: ['heel', 'calcaneus', 'os calcis', 'heel bone', 'calcaneal spur'],
  },
  {
    canonical: 'HIP',
    label: 'Hip (Femoral Head / Acetabulum)',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Hip. Preserves verified anatomy in clinical report.',
    aliases: ['hip', 'acetabulum', 'femoral head', 'femoral neck', 'hip joint', 'greater trochanter'],
  },
  {
    canonical: 'KNEE',
    label: 'Knee',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Knee. Preserves verified anatomy in clinical report.',
    aliases: ['knee', 'patella', 'kneecap', 'tibiofemoral', 'knee joint', 'meniscus area'],
  },
  {
    canonical: 'KUB',
    label: 'KUB (Kidneys, Ureters, Bladder)',
    category: 'Abdomen & Pelvis',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for KUB. Preserves verified anatomy in clinical report.',
    aliases: ['kub', 'kidneys ureters bladder', 'urinary plain film', 'plain abdomen', 'abdomen kub'],
  },
  {
    canonical: 'LEG',
    label: 'Leg (Broad Lower Limb)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Select Femur, Knee, Tibia, Ankle, or Foot for specific evaluation.',
    aliases: ['leg', 'lower limb', 'crural', 'leg x-ray'],
  },
  {
    canonical: 'LOWER_EXTREMITY',
    label: 'Lower Extremity (Broad Lower Limb)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Select specific lower extremity bone or joint for targeted model routing.',
    aliases: ['lower extremity', 'lower extremities', 'lower limb', 'lower limbs'],
  },
  {
    canonical: 'LUMBAR_SPINE',
    label: 'Lumbar Spine (L-Spine / Lower Back)',
    category: 'Spine',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Lumbar Spine. Preserves verified anatomy in clinical report.',
    aliases: ['lumbar spine', 'l-spine', 'lspine', 'lumbar', 'lower back', 'low back', 'l1-l5', 'lumbar vertebrae', 'lumbosacral'],
  },
  {
    canonical: 'NASOPHARYNX',
    label: 'Nasopharynx',
    category: 'Head & Neck',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Nasopharynx. Preserves verified anatomy in clinical report.',
    aliases: ['nasopharynx', 'post nasal space', 'adenoids', 'nasopharyngeal airway'],
  },
  {
    canonical: 'NECK',
    label: 'Neck (Soft Tissue Neck)',
    category: 'Head & Neck',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Soft Tissue Neck. Preserves verified anatomy in clinical report.',
    aliases: ['neck', 'soft tissue neck', 'airway', 'throat', 'cervical soft tissue', 'pharynx'],
  },
  {
    canonical: 'PELVIS',
    label: 'Pelvis',
    category: 'Abdomen & Pelvis',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Pelvis. Preserves verified anatomy in clinical report.',
    aliases: ['pelvis', 'pelvic ring', 'pelvic girdle', 'ilium', 'ischium', 'pubis', 'sacrum and pelvis'],
  },
  {
    canonical: 'PNS',
    label: 'PNS (Paranasal Sinuses)',
    category: 'Head & Neck',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Paranasal Sinuses. Preserves verified anatomy in clinical report.',
    aliases: ['pns', 'paranasal sinuses', 'sinus', 'sinuses', 'maxillary sinus', 'frontal sinus', 'sinusitis', 'waters view', 'caldwell view'],
  },
  {
    canonical: 'SHOULDER',
    label: 'Shoulder',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_SHOULDER (DenseNet121)',
    modelNotice: 'Configured with MURA XR_SHOULDER deep learning abnormality pipeline.',
    aliases: ['shoulder', 'glenohumeral joint', 'scapula', 'acromion', 'clavicle area', 'rotator cuff area'],
  },
  {
    canonical: 'SI_JOINT',
    label: 'SI Joint (Sacroiliac Joint)',
    category: 'Spine',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for SI Joint. Preserves verified anatomy in clinical report.',
    aliases: ['si joint', 'sacroiliac joint', 'si joints', 'sacroiliac', 'sij'],
  },
  {
    canonical: 'SKULL',
    label: 'Skull (Cranium)',
    category: 'Head & Neck',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Skull. Preserves verified anatomy in clinical report.',
    aliases: ['skull', 'cranium', 'calvarium', 'brain case', 'skull ap lat', 'towne view'],
  },
  {
    canonical: 'SPINE',
    label: 'Spine (Broad Vertebral Column)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Select Cervical, Dorsal, or Lumbar Spine for targeted assessment.',
    aliases: ['spine', 'vertebral column', 'back', 'spinal column', 'whole spine'],
  },
  {
    canonical: 'TEMPORAL_BONE',
    label: 'Temporal Bone (Mastoid)',
    category: 'Head & Neck',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Temporal Bone. Preserves verified anatomy in clinical report.',
    aliases: ['temporal bone', 'mastoid', 'petrous', 'internal auditory canal', 'temporal bones'],
  },
  {
    canonical: 'TIBIA',
    label: 'Tibia (Shin / Tib-Fib)',
    category: 'Lower Extremity',
    isBroad: false,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Regional abnormality model not configured for Tibia. Preserves verified anatomy in clinical report.',
    aliases: ['tibia', 'shin', 'fibula', 'tib fib', 'tib-fib', 'lower leg bones', 'crus'],
  },
  {
    canonical: 'UPPER_EXTREMITY',
    label: 'Upper Extremity (Broad Upper Limb)',
    category: 'Broad Category',
    isBroad: true,
    modelSupported: false,
    modelName: null,
    modelNotice: 'Broad category. Select Shoulder, Arm, Elbow, Forearm, Wrist, Hand, or Finger for specific model execution.',
    aliases: ['upper extremity', 'upper extremities', 'upper limb', 'upper limbs', 'arm and hand'],
  },
  {
    canonical: 'WRIST',
    label: 'Wrist (Carpals)',
    category: 'Upper Extremity',
    isBroad: false,
    modelSupported: true,
    modelName: 'MURA XR_WRIST (DenseNet121)',
    modelNotice: 'Configured with MURA XR_WRIST deep learning abnormality pipeline.',
    aliases: ['wrist', 'carpal', 'carpals', 'radiocarpal joint', 'scaphoid', 'distal radius', 'wrist joint'],
  },
];

/**
 * Filter the 33 classes by query against canonical name, readable label, category, and aliases.
 */
export function filterAnatomyClasses(query = '') {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return ALL_33_ANATOMY_CLASSES;

  const tokens = q.split(/\s+/).filter(Boolean);
  return ALL_33_ANATOMY_CLASSES.filter((item) => {
    const searchable = [
      item.canonical.toLowerCase(),
      item.canonical.replace(/_/g, ' ').toLowerCase(),
      item.label.toLowerCase(),
      item.category.toLowerCase(),
      ...item.aliases.map((a) => a.toLowerCase()),
    ];
    return tokens.every((token) => searchable.some((text) => text.includes(token)));
  });
}

/**
 * Find class item by canonical name or alias.
 */
export function findAnatomyClass(canonicalOrAlias) {
  if (!canonicalOrAlias) return null;
  const target = String(canonicalOrAlias).trim().toUpperCase().replace(/[ -]+/g, '_');
  const exact = ALL_33_ANATOMY_CLASSES.find((item) => item.canonical === target);
  if (exact) return exact;

  const q = String(canonicalOrAlias).trim().toLowerCase();
  return ALL_33_ANATOMY_CLASSES.find((item) =>
    item.canonical.toLowerCase() === q ||
    item.label.toLowerCase() === q ||
    item.aliases.some((a) => a.toLowerCase() === q)
  ) || null;
}

/**
 * Interactive Searchable Anatomy Selector Component for Image Cards.
 * Features:
 * - Search input with autocomplete suggestions across all 33 full-body classes
 * - Alias and synonym matching (e.g. 'c-spine' -> CERVICAL_SPINE, 'belly' -> ABDOMEN)
 * - Keyboard navigation (Up/Down arrows, Enter to select, Escape to close)
 * - Clear button
 * - Visual model compatibility indicator (Supported vs Unsupported regional model)
 * - Multi-image isolation with unique IDs per image
 */
export default function CardSearchableAnatomySelector({
  imageId,
  value,
  onChange,
  disabled = false,
  aiPredictedRegion = '',
  confirmedRegion = '',
  isConfirmed = false,
}) {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const containerRef = useRef(null);
  const inputRef = useRef(null);

  const selectedClass = useMemo(() => findAnatomyClass(value), [value]);

  const filtered = useMemo(() => filterAnatomyClasses(query), [query]);

  // Click outside listener to close dropdown
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (item) => {
    onChange(item.canonical);
    setQuery('');
    setIsOpen(false);
    setActiveIndex(-1);
  };

  const handleClear = () => {
    onChange('');
    setQuery('');
    setIsOpen(true);
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const handleKeyDown = (e) => {
    if (!isOpen) {
      if (e.key === 'ArrowDown' || e.key === 'Enter') {
        setIsOpen(true);
        e.preventDefault();
      }
      return;
    }

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((prev) => (prev + 1 < filtered.length ? prev + 1 : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((prev) => (prev - 1 >= 0 ? prev - 1 : filtered.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < filtered.length) {
        handleSelect(filtered[activeIndex]);
      } else if (filtered.length > 0) {
        handleSelect(filtered[0]);
      }
    } else if (e.key === 'Escape') {
      e.preventDefault();
      setIsOpen(false);
      setActiveIndex(-1);
    }
  };

  const inputId = `xray-search-input-${imageId}`;
  const listId = `xray-suggestions-${imageId}`;

  return (
    <div
      ref={containerRef}
      className="card-anatomy-selector"
      style={{
        position: 'relative',
        width: '100%',
        maxWidth: '32rem',
        marginTop: '0.45rem',
      }}
    >
      {/* Search Input Bar or Selected Region View */}
      {selectedClass && !isOpen ? (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.55rem 0.85rem',
            background: isConfirmed ? '#f0fdf4' : '#f8fafc',
            border: `1.5px solid ${isConfirmed ? '#86efac' : '#cbd5e1'}`,
            borderRadius: '9px',
            transition: 'all 0.15s ease',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.55rem', overflow: 'hidden' }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                justifyContent: 'center',
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: isConfirmed ? '#16a34a' : '#0284c7',
                color: '#fff',
                fontSize: '11px',
                fontWeight: 700,
                flexShrink: 0,
              }}
            >
              {isConfirmed ? '✓' : '•'}
            </span>
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.72rem', color: '#64748b', fontWeight: 600 }}>
                {isConfirmed ? 'Confirmed Routing Region:' : 'Selected Region (Pending Confirmation):'}
              </span>
              <span style={{ fontSize: '0.88rem', fontWeight: 700, color: '#0f172a' }}>
                {selectedClass.label}{' '}
                <span style={{ fontSize: '0.75rem', fontWeight: 500, color: '#64748b' }}>
                  ({selectedClass.canonical})
                </span>
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flexShrink: 0 }}>
            <button
              type="button"
              onClick={() => {
                setIsOpen(true);
                setTimeout(() => inputRef.current?.focus(), 50);
              }}
              disabled={disabled}
              style={{
                padding: '0.3rem 0.65rem',
                fontSize: '0.76rem',
                fontWeight: 600,
                color: '#0369a1',
                background: '#e0f2fe',
                border: '1px solid #bae6fd',
                borderRadius: '6px',
                cursor: disabled ? 'not-allowed' : 'pointer',
              }}
            >
              Change
            </button>
            <button
              type="button"
              onClick={handleClear}
              disabled={disabled}
              title="Clear selection"
              style={{
                background: 'none',
                border: 'none',
                color: '#94a3b8',
                cursor: disabled ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                padding: '0.2rem 0.4rem',
              }}
            >
              ✕
            </button>
          </div>
        </div>
      ) : (
        <div style={{ position: 'relative' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              border: `1.5px solid ${isOpen ? '#0284c7' : '#cbd5e1'}`,
              borderRadius: '9px',
              background: '#ffffff',
              padding: '0.45rem 0.75rem',
              boxShadow: isOpen ? '0 0 0 3px rgba(2, 132, 199, 0.12)' : 'none',
              transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
            }}
          >
            <span style={{ color: '#94a3b8', marginRight: '0.5rem', display: 'flex', alignItems: 'center' }}>
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
            </span>
            <input
              ref={inputRef}
              id={inputId}
              type="text"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setIsOpen(true);
                setActiveIndex(-1);
              }}
              onFocus={() => setIsOpen(true)}
              onKeyDown={handleKeyDown}
              disabled={disabled}
              placeholder="Search 33 anatomy classes (e.g. Wrist, Hand, Cervical Spine, Chest)..."
              autoComplete="off"
              role="combobox"
              aria-autocomplete="list"
              aria-expanded={isOpen}
              aria-controls={listId}
              style={{
                width: '100%',
                border: 'none',
                outline: 'none',
                fontSize: '0.84rem',
                color: '#0f172a',
                background: 'transparent',
              }}
            />
            {query ? (
              <button
                type="button"
                onClick={() => {
                  setQuery('');
                  setActiveIndex(-1);
                  inputRef.current?.focus();
                }}
                style={{
                  background: 'none',
                  border: 'none',
                  color: '#94a3b8',
                  cursor: 'pointer',
                  fontSize: '13px',
                  padding: '0 0.3rem',
                }}
              >
                ✕
              </button>
            ) : null}
            <button
              type="button"
              onClick={() => setIsOpen(!isOpen)}
              style={{
                background: 'none',
                border: 'none',
                color: '#64748b',
                cursor: 'pointer',
                fontSize: '11px',
                padding: '0 0.2rem',
              }}
              tabIndex={-1}
            >
              {isOpen ? '▲' : '▼'}
            </button>
          </div>

          {/* Autocomplete Dropdown List */}
          {isOpen ? (
            <div
              id={listId}
              role="listbox"
              style={{
                position: 'absolute',
                top: 'calc(100% + 4px)',
                left: 0,
                right: 0,
                maxHeight: '260px',
                overflowY: 'auto',
                background: '#ffffff',
                border: '1px solid #cbd5e1',
                borderRadius: '8px',
                boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)',
                zIndex: 50,
                padding: '0.35rem 0',
              }}
            >
              <div
                style={{
                  padding: '0.3rem 0.75rem',
                  fontSize: '0.72rem',
                  fontWeight: 700,
                  color: '#64748b',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em',
                  borderBottom: '1px solid #f1f5f9',
                }}
              >
                {filtered.length} Supported Body Regions (Catalogue of 33)
              </div>

              {filtered.length > 0 ? (
                filtered.map((item, idx) => {
                  const isActive = idx === activeIndex;
                  const isItemCurrent = value === item.canonical;
                  const isItemPredicted = aiPredictedRegion === item.canonical;

                  return (
                    <div
                      key={item.canonical}
                      role="option"
                      aria-selected={isItemCurrent}
                      onClick={() => handleSelect(item)}
                      onMouseEnter={() => setActiveIndex(idx)}
                      style={{
                        padding: '0.5rem 0.75rem',
                        cursor: 'pointer',
                        background: isActive ? '#f0f9ff' : isItemCurrent ? '#f8fafc' : '#ffffff',
                        borderLeft: isItemCurrent ? '3px solid #0284c7' : '3px solid transparent',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        gap: '0.5rem',
                      }}
                    >
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                          <span style={{ fontSize: '0.84rem', fontWeight: 600, color: '#0f172a' }}>
                            {item.label}
                          </span>
                          <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
                            [{item.canonical}]
                          </span>
                          {isItemPredicted ? (
                            <span
                              style={{
                                fontSize: '0.68rem',
                                padding: '0.1rem 0.4rem',
                                borderRadius: '4px',
                                background: '#fef3c7',
                                color: '#92400e',
                                fontWeight: 700,
                              }}
                            >
                              AI Candidate
                            </span>
                          ) : null}
                          {item.isBroad ? (
                            <span
                              style={{
                                fontSize: '0.68rem',
                                padding: '0.1rem 0.35rem',
                                borderRadius: '4px',
                                background: '#f1f5f9',
                                color: '#475569',
                              }}
                            >
                              Broad Category
                            </span>
                          ) : null}
                        </div>
                        <span style={{ fontSize: '0.72rem', color: '#64748b' }}>
                          {item.category} · {item.aliases.slice(0, 3).join(', ')}
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', flexShrink: 0 }}>
                        {item.modelSupported ? (
                          <span
                            style={{
                              fontSize: '0.7rem',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                              background: '#ecfdf5',
                              color: '#065f46',
                              fontWeight: 600,
                              border: '1px solid #a7f3d0',
                            }}
                          >
                            Model Available
                          </span>
                        ) : (
                          <span
                            style={{
                              fontSize: '0.7rem',
                              padding: '0.15rem 0.45rem',
                              borderRadius: '4px',
                              background: '#f8fafc',
                              color: '#64748b',
                              border: '1px solid #e2e8f0',
                            }}
                          >
                            Preserve Anatomy Only
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div style={{ padding: '0.75rem', fontSize: '0.8rem', color: '#64748b', textAlign: 'center' }}>
                  No matching region found for &ldquo;{query}&rdquo;. All 33 standard classes are supported.
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* Selected Region Status Notice */}
      {selectedClass ? (
        <div
          style={{
            marginTop: '0.4rem',
            padding: '0.45rem 0.65rem',
            borderRadius: '6px',
            fontSize: '0.77rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: selectedClass.modelSupported ? '#ecfdf5' : '#f8fafc',
            border: `1px solid ${selectedClass.modelSupported ? '#a7f3d0' : '#e2e8f0'}`,
            color: selectedClass.modelSupported ? '#065f46' : '#475569',
          }}
        >
          <span style={{ fontSize: '13px' }}>{selectedClass.modelSupported ? '⚡' : 'ℹ️'}</span>
          <span>
            {selectedClass.modelSupported
              ? `Compatible Regional Model: ${selectedClass.modelName} — will execute once region is confirmed.`
              : selectedClass.modelNotice}
          </span>
        </div>
      ) : null}
    </div>
  );
}
