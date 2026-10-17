import { useState, useEffect, useMemo, useRef } from 'react';
import { filterBackendRegions, findRegionInList } from '../utils/xrayCatalogue';
import { getXrayCatalogue } from '../services/xrayService';

function SearchIcon({ size = 16, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
  );
}

function CheckmarkIcon({ size = 14, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function InfoCircleIcon({ size = 14, color = 'currentColor' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
  );
}

export default function XrayRegionSelector({ selectedRegion, onSelectRegion, required = true }) {
  const [regions, setRegions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);

  const containerRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch backend-driven body region catalogue
  useEffect(() => {
    let active = true;

    getXrayCatalogue()
      .then((data) => {
        if (!active) return;
        if (data && Array.isArray(data.regions)) {
          setRegions(data.regions);
        } else if (data && Array.isArray(data.catalogue)) {
          // Fallback extraction if backend only had nested catalogue
          const extracted = [];
          data.catalogue.forEach((main) => {
            (main.subregions || []).forEach((sub) => {
              (sub.xray_types || []).forEach((xt) => {
                extracted.push({
                  id: xt.id,
                  name: sub.name || xt.name,
                  canonical_region: xt.canonical_region,
                  aliases: xt.aliases || [],
                  is_supported: Boolean(xt.model_support?.is_supported),
                  model_status: xt.model_support?.status || 'unsupported',
                  model_notice: xt.model_support?.notice || '',
                });
              });
            });
          });
          setRegions(extracted);
        }
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setFetchError('Unable to load body regions from server. Please refresh.');
        setLoading(false);
      });


    return () => {
      active = false;
    };
  }, []);

  // Close suggestion dropdown if user clicks outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setIsSearching(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Filter matching regions dynamically from backend data based on search input
  const suggestions = useMemo(() => {
    if (!searchQuery.trim()) return [];
    return filterBackendRegions(regions, searchQuery);
  }, [regions, searchQuery]);

  // Find currently selected region details from backend list
  const selectedDetails = useMemo(() => {
    if (!selectedRegion) return null;
    return findRegionInList(regions, selectedRegion);
  }, [regions, selectedRegion]);

  const handleSelect = (item) => {
    onSelectRegion(item.canonical_region, item);
    setSearchQuery('');
    setIsSearching(false);
    setActiveIndex(-1);
  };

  const handleClear = () => {
    onSelectRegion('', null);
    setSearchQuery('');
    setIsSearching(true);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 50);
  };

  const handleKeyDown = (e) => {
    if (!isSearching || suggestions.length === 0) return;

    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIndex((prev) => (prev + 1 < suggestions.length ? prev + 1 : 0));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIndex((prev) => (prev - 1 >= 0 ? prev - 1 : suggestions.length - 1));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (activeIndex >= 0 && activeIndex < suggestions.length) {
        handleSelect(suggestions[activeIndex]);
      } else if (suggestions.length > 0) {
        handleSelect(suggestions[0]);
      }
    } else if (e.key === 'Escape') {
      setIsSearching(false);
      setActiveIndex(-1);
    }
  };

  return (
    <div className="xray-simple-selector-box" ref={containerRef}>
      <div className="xray-selector-header">
        <label htmlFor="xray-search-input" className="xray-selector-label">
          Search body region {required ? <span style={{ color: '#ef4444' }}>*</span> : null}
        </label>
        {loading ? (
          <span className="xray-selector-status-tag">Loading regions...</span>
        ) : null}
      </div>

      {fetchError ? (
        <div className="xray-error-notice" style={{ color: '#b91c1c', fontSize: '0.82rem', marginBottom: '0.5rem' }}>
          {fetchError}
        </div>
      ) : null}

      {/* Selected Region Display Indicator */}
      {selectedRegion && !isSearching ? (
        <div className="xray-selected-simple-banner" id="selected-region-indicator">
          <div className="xray-selected-simple-left">
            <span className="xray-selected-check-icon">
              <CheckmarkIcon size={14} color="#059669" />
            </span>
            <div className="xray-selected-text-col">
              <span className="xray-selected-label-sub">Selected Body Region:</span>
              <strong className="xray-selected-name">
                {selectedDetails ? selectedDetails.name : selectedRegion.toUpperCase()}
              </strong>
            </div>
          </div>

          <button
            type="button"
            className="xray-change-simple-btn"
            id="change-region-btn"
            onClick={handleClear}
            title="Search for a different body region"
          >
            Change Region
          </button>
        </div>
      ) : (
        /* Simple Search Input */
        <div className="xray-search-input-container">
          <span className="xray-search-icon">
            <SearchIcon size={16} />
          </span>

          <input
            ref={inputRef}
            id="xray-search-input"
            type="text"
            className="xray-search-simple-input"
            placeholder="Search body region, e.g. Head, Chest, Spine..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setIsSearching(true);
              setActiveIndex(-1);
            }}
            onFocus={() => {
              if (searchQuery.trim().length > 0) {
                setIsSearching(true);
              }
            }}
            onKeyDown={handleKeyDown}
            autoComplete="off"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={isSearching && suggestions.length > 0}
            aria-controls="xray-suggestions-list"
          />

          {searchQuery ? (
            <button
              type="button"
              className="xray-search-clear-btn"
              onClick={() => {
                setSearchQuery('');
                setActiveIndex(-1);
                inputRef.current?.focus();
              }}
              aria-label="Clear search input"
            >
              ✕
            </button>
          ) : null}

          {/* Small List of Matching Suggestions (Displayed only when user enters search query) */}
          {isSearching && searchQuery.trim().length > 0 ? (
            <ul className="xray-suggestions-dropdown" id="xray-suggestions-list" role="listbox">
              {suggestions.length > 0 ? (
                suggestions.map((item, idx) => (
                  <li
                    key={item.id || item.name}
                    role="option"
                    aria-selected={idx === activeIndex}
                    className={`xray-suggestion-item ${idx === activeIndex ? 'active' : ''}`}
                    onClick={() => handleSelect(item)}
                  >
                    <span className="xray-suggestion-title">{item.name}</span>
                    {item.canonical_region && item.canonical_region !== item.name.toLowerCase() ? (
                      <span className="xray-suggestion-hint">({item.canonical_region})</span>
                    ) : null}
                  </li>
                ))
              ) : (
                <li className="xray-suggestion-empty" id="no-matching-regions">
                  No matching body regions found for &ldquo;{searchQuery}&rdquo;.
                </li>
              )}
            </ul>
          ) : null}
        </div>
      )}

      {/* Model Capability Note for Selected Region */}
      {selectedRegion && selectedDetails ? (
        <div
          className={`xray-model-status-note ${
            selectedDetails.is_supported ? 'note-supported' : 'note-unsupported'
          }`}
          style={{
            marginTop: '0.65rem',
            padding: '0.55rem 0.85rem',
            borderRadius: '8px',
            fontSize: '0.78rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.45rem',
            background: selectedDetails.is_supported ? '#f0fdf4' : '#f8fafc',
            border: `1px solid ${selectedDetails.is_supported ? '#bbf7d0' : '#e2e8f0'}`,
            color: selectedDetails.is_supported ? '#166534' : '#475569',
          }}
        >
          <InfoCircleIcon size={14} color="currentColor" />
          <span>
            {selectedDetails.model_notice ||
              (selectedDetails.is_supported
                ? 'Automated AI vision screening pipeline configured for this region.'
                : 'Automated AI vision inference is currently unavailable for this region. Your upload and any reports will be safely preserved in your assessment report.')}
          </span>
        </div>
      ) : null}
    </div>
  );
}
