import { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import MedicalShopCard from '../components/MedicalShopCard';
import ErrorMessage from '../components/ErrorMessage';
import LocationMap from '../components/LocationMap';
import NearbyHealthcareMap from '../components/NearbyHealthcareMap';
import { useGeolocation } from '../hooks/useGeolocation';
import { searchMedicalShops } from '../services/medicalShopService';

export default function MedicalShops() {
  const [searchParams] = useSearchParams();

  const assessmentId = searchParams.get('assessment_id') || '';

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
        setError('Please use your current location to discover nearby pharmacies.');
        return;
      }

      setLoading(true);
      try {
        const params = {
          latitude: coords.latitude,
          longitude: coords.longitude,
          page: targetPage,
          page_size: 10,
        };

        if (radiusKm && radiusKm > 25) {
          params.radius_km = radiusKm;
        }

        if (assessmentId) {
          params.assessment_id = assessmentId;
        }

        const data = await searchMedicalShops(params);
        setResult(data);
        setPage(targetPage);

        // If search was expanded to 50km, preserve radius context for pagination
        if (data?.expanded) {
          setCurrentRadius(50);
        }
      } catch (err) {
        setError(err.message || 'Unable to retrieve medical shops.');
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [confirmedCoords, assessmentId, currentRadius]
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

  return (
    <div className="page">
      <h1>Nearby Medical Shops & Pharmacies</h1>
      <p className="muted">
        Used primarily for the Mild pathway or general OTC wellness support. This system does not prescribe medicines or provide dosage changes.
      </p>

      {assessmentId ? (
        <div
          style={{
            backgroundColor: '#f0fdf4',
            border: '1px solid #bbf7d0',
            color: '#166534',
            padding: '10px 14px',
            borderRadius: '8px',
            marginBottom: '16px',
            fontSize: '14px',
          }}
        >
          <strong>Self-Care Mode:</strong> Finding pharmacies near your confirmed location based on triage assessment.
          (Assessment ID: {assessmentId})
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
          flexWrap: 'wrap',
          gap: '12px',
          alignItems: 'center',
        }}
      >
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

      {/* In-app interactive map modal for location confirmation */}
      {showMapModal ? (
        <LocationMap
          initialLatitude={geoCoords?.latitude}
          initialLongitude={geoCoords?.longitude}
          onConfirmLocation={handleConfirmLocationFromMap}
          onCancel={() => setShowMapModal(false)}
          title="Confirm Your Location for Pharmacy Search"
          markers={result?.medical_shops || []}
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
            <h2 style={{ margin: 0, fontSize: '20px' }}>Nearby Pharmacies</h2>
            {result.medical_shops?.length > 0 ? (
              <span className="muted" style={{ fontSize: '14px' }}>
                Showing page {page} of {result.has_next ? '2' : page} (max 20)
              </span>
            ) : null}
          </div>

          <p style={{ margin: '0 0 16px 0', color: '#475569' }}>{result.message}</p>

          {!result.medical_shops?.length ? (
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
                      places={result.medical_shops}
                      placeType="shop"
                      selectedId={selectedPlaceId}
                      onSelectPlace={setSelectedPlaceId}
                    />
                  </div>
                  <div style={{ flex: '1 1 42%', minWidth: '300px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
                    {result.medical_shops.map((s, idx) => {
                      const sId = String(s.external_id || s.shop_id || `shop-${idx}`);
                      const isSelected = selectedPlaceId === sId;
                      return (
                        <div
                          key={sId}
                          onClick={() => setSelectedPlaceId(sId)}
                          style={{
                            cursor: 'pointer',
                            borderRadius: '12px',
                            transition: 'all 0.2s ease',
                            border: isSelected ? '2px solid #059669' : '2px solid transparent',
                            boxShadow: isSelected ? '0 0 0 3px rgba(5, 150, 105, 0.25)' : 'none',
                          }}
                        >
                          <MedicalShopCard shop={s} />
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="card-grid">
                  {result.medical_shops.map((s, idx) => (
                    <MedicalShopCard key={s.shop_id || s.external_id || idx} shop={s} />
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
