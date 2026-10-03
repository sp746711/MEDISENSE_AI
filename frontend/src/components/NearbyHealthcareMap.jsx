import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// SVG Icon definition for User's Confirmed Location
const userPinIcon = L.divIcon({
  className: 'custom-user-pin',
  html: `<div style="
    background: #2563eb;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    border: 3px solid white;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4);
    display: flex;
    align-items: center;
    justify-content: center;
  "><div style="
    width: 10px;
    height: 10px;
    background: white;
    border-radius: 50%;
    transform: rotate(45deg);
  "></div></div>`,
  iconSize: [32, 32],
  iconAnchor: [16, 32],
  popupAnchor: [0, -32],
});

// SVG Icon for Healthcare Facilities (Hospitals)
const hospitalPinIcon = L.divIcon({
  className: 'custom-hospital-pin',
  html: `<div style="
    background: #dc2626;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    border: 2px solid white;
    box-shadow: 0 3px 10px rgba(220,38,38,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
  "><span style="
    transform: rotate(45deg);
    font-weight: 800;
    font-size: 14px;
    line-height: 1;
  ">H</span></div>`,
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

// SVG Icon for Medical Shops / Pharmacies
const pharmacyPinIcon = L.divIcon({
  className: 'custom-pharmacy-pin',
  html: `<div style="
    background: #059669;
    color: white;
    width: 30px;
    height: 30px;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    border: 2px solid white;
    box-shadow: 0 3px 10px rgba(5,150,105,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
  "><span style="
    transform: rotate(45deg);
    font-weight: 800;
    font-size: 13px;
    line-height: 1;
  ">Rx</span></div>`,
  iconSize: [30, 30],
  iconAnchor: [15, 30],
  popupAnchor: [0, -30],
});

export default function NearbyHealthcareMap({
  userCoords,
  places = [],
  placeType = 'facility', // 'facility' | 'shop'
  selectedId = null,
  onSelectPlace = () => {},
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef(new Map());
  const onSelectPlaceRef = useRef(onSelectPlace);
  useEffect(() => {
    onSelectPlaceRef.current = onSelectPlace;
  });

  const initialCoordsRef = useRef({
    lat: userCoords?.latitude || 20.5937,
    lng: userCoords?.longitude || 78.9629,
  });

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    const initialLat = initialCoordsRef.current.lat;
    const initialLng = initialCoordsRef.current.lng;

    const map = L.map(mapContainerRef.current, {
      center: [initialLat, initialLng],
      zoom: 13,
      zoomControl: true,
    });
    mapInstanceRef.current = map;

    // Real OpenStreetMap tiles + OpenStreetMap & Geoapify attribution
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors | Powered by <a href="https://www.geoapify.com/" target="_blank" rel="noopener noreferrer">Geoapify</a>',
    }).addTo(map);

    const handleResize = () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.invalidateSize();
      }
    };
    window.addEventListener('resize', handleResize);

    // Initial size invalidation
    const timer = setTimeout(handleResize, 200);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('resize', handleResize);
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Update markers and bounds when userCoords or places change
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    // Clear existing place markers
    markersRef.current.forEach((marker) => {
      marker.remove();
    });
    markersRef.current.clear();

    const bounds = L.latLngBounds();

    // 1. Add User marker if coordinates provided
    if (userCoords?.latitude != null && userCoords?.longitude != null) {
      const userLatLng = [userCoords.latitude, userCoords.longitude];
      const userMarker = L.marker(userLatLng, { icon: userPinIcon }).addTo(map);
      userMarker.bindPopup(
        `<div style="font-size: 13px; font-family: sans-serif;">
          <strong style="color: #1e40af;">Your Confirmed Location</strong><br/>
          <span style="color: #64748b; font-size: 12px;">${userCoords.latitude.toFixed(4)}, ${userCoords.longitude.toFixed(4)}</span>
        </div>`
      );
      bounds.extend(userLatLng);
    }

    // 2. Add healthcare place markers
    places.forEach((place, idx) => {
      if (place.latitude == null || place.longitude == null) return;

      const latLng = [place.latitude, place.longitude];
      const icon = placeType === 'shop' ? pharmacyPinIcon : hospitalPinIcon;
      const marker = L.marker(latLng, { icon }).addTo(map);

      const placeId = place.external_id || place.place_id || place.facility_id || place.shop_id || `place-${idx}`;
      markersRef.current.set(String(placeId), marker);

      // Construct popup content strictly with source-provided information
      let popupHtml = '';
      if (placeType === 'facility') {
        const emergencyBadge =
          place.emergency_available === 'yes' || place.has_emergency === true
            ? `<div style="color: #dc2626; font-weight: 700; font-size: 12px; margin-top: 4px;">Emergency Services Available</div>`
            : '';

        popupHtml = `
          <div style="font-size: 13px; line-height: 1.4; font-family: sans-serif; min-width: 180px;">
            <strong style="font-size: 14px; color: #0f172a;">${place.name || 'Hospital'}</strong><br/>
            ${place.type ? `<span style="color: #475569; font-size: 12px;">Type: ${place.type}</span><br/>` : ''}
            ${place.address ? `<div style="color: #64748b; font-size: 12px; margin-top: 2px;">${place.address}</div>` : ''}
            ${place.distance_km != null ? `<div style="color: #1d4ed8; font-weight: 600; font-size: 12px; margin-top: 4px;">${place.distance_km} km away</div>` : ''}
            ${emergencyBadge}
            ${place.contact ? `<div style="color: #0f766e; font-size: 12px; margin-top: 4px;">Contact: ${place.contact}</div>` : ''}
            <div style="font-size: 11px; color: #94a3b8; margin-top: 6px; border-top: 1px solid #f1f5f9; padding-top: 4px;">
              Source: ${place.source || 'Geoapify / OpenStreetMap'}
            </div>
          </div>
        `;
      } else {
        // Medical Shop / Pharmacy popup
        popupHtml = `
          <div style="font-size: 13px; line-height: 1.4; font-family: sans-serif; min-width: 180px;">
            <strong style="font-size: 14px; color: #0f172a;">${place.name || 'Medical Shop / Pharmacy'}</strong><br/>
            ${place.address ? `<div style="color: #64748b; font-size: 12px; margin-top: 2px;">${place.address}</div>` : ''}
            ${place.distance_km != null ? `<div style="color: #1d4ed8; font-weight: 600; font-size: 12px; margin-top: 4px;">${place.distance_km} km away</div>` : ''}
            ${place.contact ? `<div style="color: #0f766e; font-size: 12px; margin-top: 4px;">Contact: ${place.contact}</div>` : ''}
            <div style="font-size: 11px; color: #94a3b8; margin-top: 6px; border-top: 1px solid #f1f5f9; padding-top: 4px;">
              Source: ${place.source || 'Geoapify / OpenStreetMap'}
            </div>
          </div>
        `;
      }

      marker.bindPopup(popupHtml);

      // Marker click synchronization
      marker.on('click', () => {
        onSelectPlaceRef.current(String(placeId));
      });

      bounds.extend(latLng);
    });

    // Fit map bounds to encompass user and returned places
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [35, 35], maxZoom: 15 });
    } else if (userCoords?.latitude != null && userCoords?.longitude != null) {
      map.setView([userCoords.latitude, userCoords.longitude], 13);
    }
  }, [userCoords, places, placeType]);

  // Synchronize map focus when selectedId changes from result card click
  useEffect(() => {
    if (!selectedId) return;
    const map = mapInstanceRef.current;
    const marker = markersRef.current.get(String(selectedId));
    if (map && marker) {
      map.panTo(marker.getLatLng(), { animate: true, duration: 0.5 });
      marker.openPopup();
    }
  }, [selectedId]);

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        width: '100%',
        height: '100%',
        minHeight: '480px',
        borderRadius: '12px',
        overflow: 'hidden',
        boxShadow: '0 4px 16px rgba(0,0,0,0.08)',
        border: '1px solid #e2e8f0',
        backgroundColor: '#f8fafc',
      }}
    >
      <div
        ref={mapContainerRef}
        style={{
          width: '100%',
          flex: '1 1 auto',
          minHeight: '440px',
        }}
      />
      <div
        style={{
          padding: '6px 12px',
          backgroundColor: '#f8fafc',
          borderTop: '1px solid #e2e8f0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '11px',
          color: '#64748b',
        }}
      >
        <span>
          {places.length} nearby {placeType === 'shop' ? 'pharmacies' : 'hospitals'} mapped
        </span>
        <span>
          Powered by <a href="https://www.geoapify.com/" target="_blank" rel="noopener noreferrer" style={{ color: '#2563eb', textDecoration: 'none' }}>Geoapify</a> | &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer" style={{ color: '#2563eb', textDecoration: 'none' }}>OpenStreetMap</a>
        </span>
      </div>
    </div>
  );
}
