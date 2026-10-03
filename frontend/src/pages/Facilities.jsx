import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import FacilityCard from '../components/FacilityCard';
import ErrorMessage from '../components/ErrorMessage';
import LocationMap from '../components/LocationMap';
import NearbyHealthcareMap from '../components/NearbyHealthcareMap';
import { useGeolocation } from '../hooks/useGeolocation';
import { searchFacilities } from '../services/facilityService';

export default function Facilities() {
  const [searchParams] = useSearchParams();

  const urlEmergencyOnly = searchParams.get('emergency_only') === 'true';
  const assessmentId = searchParams.get('assessment_id') || '';

  const [emergencyOnly, setEmergencyOnly] = useState(urlEmergencyOnly);
  const [confirmedCoords, setConfirmedCoords] = useState(null);
  const [selectedPlaceId, setSelectedPlaceId] = useState(null);
  const [showMapModal, setShowMapModal] = useState(false);
  const [page, setPage] = useState(1);
  const [currentRadius, setCurrentRadius] = useState(25);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const {
    coordinates: geoCoords,
    loading: geoLoading,
    error: geoError,
    requestLocation,
  } = useGeolocation();

  // When GPS detects coordinates, open LocationMap modal for user confirmation
  useEffect(() => {
    if (geoCoords && !confirmedCoords) {
      setShowMapModal(true);
    }
  }, [geoCoords, confirmedCoords]);

  useEffect(() => {
    if (urlEmergencyOnly) {
      setEmergencyOnly(true);
    }
  }, [urlEmergencyOnly]);

  const executeSearch = useCallback(
    async (targetPage = 1, coords = confirmedCoords, radiusKm = currentRadius) => {
      setError('');

      // Enforce maximum 2 pages
      if (targetPage > 2 || targetPage < 1) {
        return;
      }

      // Explicit coordinate check - NEVER rely on stale searchMode state
      const isCoords = Boolean(coords && coords.latitude != null && coords.longitude != null);
      if (!isCoords) {
        setError('Please use your current location to discover nearby hospitals.');
        return;
      }

      setLoading(true);
      try {
        const params = {
          latitude: coords.latitude,
          longitude: coords.longitude,
          page: targetPage,
          page_size: 10,
          emergency_only: emergencyOnly ? 'true' : 'false',
        };

        if (radiusKm && radiusKm > 25) {
          params.radius_km = radiusKm;
        }

        if (assessmentId) {
          params.assessment_id = assessmentId;
        }

        const data = await searchFacilities(params);
        setResult(data);
        setPage(targetPage);

        // If search was expanded to 50km, preserve radius context for pagination
        if (data?.expanded) {
          setCurrentRadius(50);
        }
      } catch (err) {
        setError(err.message || 'Unable to retrieve healthcare facilities.');
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [confirmedCoords, emergencyOnly, assessmentId, currentRadius]
  );

  const handleUseCurrentLocation = () => {
    setError('');
    requestLocation();
  };

  const handleConfirmLocationFromMap = (coords) => {
    setConfirmedCoords(coords);
    setShowMapModal(false);
    setCurrentRadius(25); // New location always starts with 25 km
    executeSearch(1, coords, 25);
  };

  const handlePageChange = (newPage) => {
    // Strictly block page > 2 and page < 1
    if (newPage < 1 || newPage > 2) return;
    executeSearch(newPage, confirmedCoords, currentRadius);
  };

  const handleEmergencyToggle = (e) => {
    const val = e.target.checked;
    setEmergencyOnly(val);
    if (confirmedCoords) {
      // Re-query with new filter using confirmed coordinates
      executeSearch(1, confirmedCoords, currentRadius);
    }
  };

  return (
    <div className="page">
      <h1>Nearby Healthcare Facilities</h1>
      <p className="muted">
        Discover hospitals near your confirmed location. Data is sourced directly from OpenStreetMap and Geoapify. Nothing is fabricated.
      </p>

      {urlEmergencyOnly ? (
        <div
          style={{
            backgroundColor: '#fef2f2',
            border: '1px solid #fecaca',
            color: '#991b1b',
            padding: '10px 14px',
            borderRadius: '8px',
            marginBottom: '16px',
            fontSize: '14px',
          }}
        >
          <strong>Emergency Mode:</strong> Prioritizing emergency facilities based on assessment outcome.
          {assessmentId ? ` (Assessment ID: ${assessmentId})` : ''}
        </div>
      ) : null}

      <ErrorMessage
        message={
          error ||
          (geoError
            ? 'Unable to determine your current location. Please allow location access and try again.'
            : '')
        }
        onDismiss={() => setError('')}
      />

      {/* Primary Current Location Discovery Action */}
      <div
        style={{
          backgroundColor: '#f8fafc',
          border: '1px solid #e2e8f0',
          borderRadius: '12px',
          padding: '20px',
          marginBottom: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '14px',
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', alignItems: 'center' }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleUseCurrentLocation}
            disabled={geoLoading || loading}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}
          >
            {geoLoading ? 'Detecting Location...' : '📍 Use My Current Location'}
          </button>

          {confirmedCoords ? (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                backgroundColor: '#ecfdf5',
                border: '1px solid #a7f3d0',
                padding: '8px 14px',
                borderRadius: '8px',
                fontSize: '13px',
                color: '#065f46',
              }}
            >
              <span>
                Confirmed Location: <strong>{confirmedCoords.latitude.toFixed(4)}, {confirmedCoords.longitude.toFixed(4)}</strong>
              </span>
            </div>
          ) : null}
        </div>

        <label className="checkbox-row" style={{ marginTop: '4px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={emergencyOnly}
            onChange={handleEmergencyToggle}
            disabled={urlEmergencyOnly}
          />
          <strong>Emergency facilities only</strong>
        </label>
      </div>

      {/* In-app interactive map modal for location confirmation */}
      {showMapModal ? (
        <LocationMap
          initialLatitude={geoCoords?.latitude}
          initialLongitude={geoCoords?.longitude}
          onConfirmLocation={handleConfirmLocationFromMap}
          onCancel={() => setShowMapModal(false)}
          title="Confirm Your Location for Hospital Search"
          markers={result?.facilities || []}
        />
      ) : null}

      {/* Results Section */}
      {result ? (
        <section className="section" style={{ marginTop: '20px' }}>
          {result.expanded && result.expansion_message ? (
            <div
              style={{
                backgroundColor: '#fffbeb',
                border: '1px solid #fde68a',
                color: '#92400e',
                padding: '12px 16px',
                borderRadius: '8px',
                marginBottom: '16px',
                fontSize: '14px',
              }}
            >
              <strong>Search Notice:</strong> {result.expansion_message}
            </div>
          ) : null}

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
            <h2 style={{ margin: 0, fontSize: '20px' }}>Nearby Hospitals</h2>
            {result.facilities?.length > 0 ? (
              <span className="muted" style={{ fontSize: '14px' }}>
                Showing page {page} of {result.has_next ? '2' : page} (max 20)
              </span>
            ) : null}
          </div>

          <p style={{ margin: '0 0 16px 0', color: '#475569' }}>{result.message}</p>

          {!result.facilities?.length ? (
            <div
              className="empty-state"
              style={{
                padding: '36px 20px',
                textAlign: 'center',
                backgroundColor: '#f8fafc',
                borderRadius: '12px',
                border: '1px dashed #cbd5e1',
                color: '#64748b',
              }}
            >
              <p style={{ margin: 0, fontSize: '15px' }}>{result.message}</p>
            </div>
          ) : (
            <>
              {confirmedCoords ? (
                <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  <div style={{ flex: '1 1 54%', minWidth: '320px', minHeight: '520px', position: 'sticky', top: '20px' }}>
                    <NearbyHealthcareMap
                      userCoords={confirmedCoords}
                      places={result.facilities}
                      placeType="facility"
                      selectedId={selectedPlaceId}
                      onSelectPlace={setSelectedPlaceId}
                    />
                  </div>
                  <div style={{ flex: '1 1 42%', minWidth: '300px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {result.facilities.map((f, idx) => {
                      const fId = String(f.external_id || f.facility_id || `fac-${idx}`);
                      const isSelected = selectedPlaceId === fId;
                      return (
                        <div
                          key={fId}
                          onClick={() => setSelectedPlaceId(fId)}
                          style={{
                            cursor: 'pointer',
                            borderRadius: '12px',
                            transition: 'all 0.2s ease',
                            border: isSelected ? '2px solid #2563eb' : '2px solid transparent',
                            boxShadow: isSelected ? '0 0 0 3px rgba(37, 99, 235, 0.25)' : 'none',
                          }}
                        >
                          <FacilityCard facility={f} />
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="card-grid">
                  {result.facilities.map((f, idx) => (
                    <FacilityCard key={f.facility_id || f.external_id || idx} facility={f} />
                  ))}
                </div>
              )}

              {/* Pagination Controls - Maximum 2 Pages */}
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '16px',
                  marginTop: '28px',
                }}
              >
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => handlePageChange(1)}
                  disabled={page <= 1 || loading}
                >
                  Previous
                </button>
                <span style={{ fontWeight: 600, fontSize: '14px', color: '#334155' }}>
                  Page {page} of {result.has_next ? '2' : page}
                </span>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => handlePageChange(2)}
                  disabled={page >= 2 || !result.has_next || loading}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </section>
      ) : null}
    </div>
  );
}
