import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// SVG Icon definition to avoid Vite/Webpack image asset resolution issues
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
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
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

const placePinIcon = L.divIcon({
  className: 'custom-place-pin',
  html: `<div style="
    background: #059669;
    color: white;
    width: 28px;
    height: 28px;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    border: 2px solid white;
    box-shadow: 0 3px 8px rgba(0,0,0,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
  "><div style="
    width: 8px;
    height: 8px;
    background: white;
    border-radius: 50%;
    transform: rotate(45deg);
  "></div></div>`,
  iconSize: [28, 28],
  iconAnchor: [14, 28],
  popupAnchor: [0, -28],
});

export default function LocationMap({
  initialLatitude,
  initialLongitude,
  onConfirmLocation,
  onCancel,
  markers = [],
  title = "Select Your Location",
}) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const userMarkerRef = useRef(null);

  const [selectedCoords, setSelectedCoords] = useState({
    latitude: initialLatitude || 20.5937,
    longitude: initialLongitude || 78.9629,
  });

  useEffect(() => {
    if (!mapContainerRef.current) return;

    const lat = initialLatitude || 20.5937;
    const lng = initialLongitude || 78.9629;
    const zoom = initialLatitude && initialLongitude ? 14 : 5;

    const map = L.map(mapContainerRef.current, {
      center: [lat, lng],
      zoom: zoom,
      zoomControl: true,
    });
    mapInstanceRef.current = map;

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    }).addTo(map);

    if (initialLatitude && initialLongitude) {
      const marker = L.marker([lat, lng], {
        icon: userPinIcon,
        draggable: true,
      }).addTo(map);

      marker.bindPopup("<b>Your Current Location</b><br>You can drag or click to adjust.").openPopup();
      userMarkerRef.current = marker;

      marker.on('dragend', (e) => {
        const pos = e.target.getLatLng();
        setSelectedCoords({ latitude: pos.lat, longitude: pos.lng });
      });
    }

    map.on('click', (e) => {
      const { lat: clickLat, lng: clickLng } = e.latlng;
      setSelectedCoords({ latitude: clickLat, longitude: clickLng });
      if (userMarkerRef.current) {
        userMarkerRef.current.setLatLng([clickLat, clickLng]);
      } else {
        const marker = L.marker([clickLat, clickLng], {
          icon: userPinIcon,
          draggable: true,
        }).addTo(map);
        userMarkerRef.current = marker;
        marker.on('dragend', (evt) => {
          const pos = evt.target.getLatLng();
          setSelectedCoords({ latitude: pos.lat, longitude: pos.lng });
        });
      }
    });

    // Add legitimate markers if coordinates exist
    if (markers && markers.length > 0) {
      markers.forEach((item) => {
        if (item.latitude != null && item.longitude != null) {
          const m = L.marker([item.latitude, item.longitude], { icon: placePinIcon }).addTo(map);
          const popupContent = `
            <div style="font-size: 13px;">
              <strong>${item.name || 'Healthcare Facility'}</strong><br/>
              ${item.type ? `<span>Type: ${item.type}</span><br/>` : ''}
              ${item.address ? `<span>${item.address}</span><br/>` : ''}
              ${item.emergency_available === 'yes' || item.has_emergency ? '<span style="color: #dc2626; font-weight: bold;">[Emergency Services Available]</span><br/>' : ''}
              ${item.source ? `<small style="color: #6b7280;">Source: ${item.source}</small>` : ''}
            </div>
          `;
          m.bindPopup(popupContent);
        }
      });
    }

    // Force map size invalidation after mount
    setTimeout(() => {
      map.invalidateSize();
    }, 150);

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, [initialLatitude, initialLongitude, markers]);

  const handleConfirm = () => {
    if (onConfirmLocation) {
      onConfirmLocation(selectedCoords);
    }
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.75)',
        backdropFilter: 'blur(4px)',
        zIndex: 9999,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '16px',
          width: '100%',
          maxWidth: '750px',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <div
          style={{
            padding: '16px 20px',
            borderBottom: '1px solid #e2e8f0',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600, color: '#0f172a' }}>
              {title}
            </h3>
            <p style={{ margin: '4px 0 0', fontSize: '13px', color: '#64748b' }}>
              Confirm your location on the map to find nearby healthcare providers
            </p>
          </div>
          <button
            onClick={onCancel}
            type="button"
            style={{
              background: 'transparent',
              border: 'none',
              fontSize: '22px',
              cursor: 'pointer',
              color: '#94a3b8',
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>

        {/* Map Container */}
        <div
          ref={mapContainerRef}
          style={{
            height: '380px',
            width: '100%',
            backgroundColor: '#e2e8f0',
          }}
        />

        {/* Location Details and Action Buttons */}
        <div
          style={{
            padding: '16px 20px',
            backgroundColor: '#f8fafc',
            borderTop: '1px solid #e2e8f0',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '12px',
          }}
        >
          <div style={{ fontSize: '13px', color: '#475569' }}>
            <span>Latitude: <strong>{selectedCoords.latitude.toFixed(5)}</strong></span>
            <span style={{ marginLeft: '12px' }}>Longitude: <strong>{selectedCoords.longitude.toFixed(5)}</strong></span>
          </div>

          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              type="button"
              onClick={onCancel}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                border: '1px solid #cbd5e1',
                backgroundColor: '#ffffff',
                color: '#475569',
                fontSize: '14px',
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleConfirm}
              style={{
                padding: '8px 20px',
                borderRadius: '8px',
                border: 'none',
                backgroundColor: '#2563eb',
                color: '#ffffff',
                fontSize: '14px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Use This Location
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
