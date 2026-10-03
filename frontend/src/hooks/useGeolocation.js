import { useState, useCallback } from 'react';

export function useGeolocation() {
  const [coordinates, setCoordinates] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [permissionDenied, setPermissionDenied] = useState(false);

  const requestLocation = useCallback(() => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser. Please use manual location search.');
      return;
    }

    setLoading(true);
    setError(null);
    setPermissionDenied(false);

    navigator.geolocation.getCurrentPosition(
      (position) => {
        setCoordinates({
          latitude: position.coords.latitude,
          longitude: position.coords.longitude,
        });
        setLoading(false);
        setError(null);
        setPermissionDenied(false);
      },
      (err) => {
        setLoading(false);
        if (err.code === 1) { // PERMISSION_DENIED
          setPermissionDenied(true);
          setError('Location access was not granted. You can continue with manual location search.');
        } else if (err.code === 2) { // POSITION_UNAVAILABLE
          setError('Location information is currently unavailable. Please enter your location manually.');
        } else if (err.code === 3) { // TIMEOUT
          setError('Location request timed out. Please try again or use manual location search.');
        } else {
          setError('Unable to retrieve location. Please use manual location search.');
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 15000,
        maximumAge: 0,
      }
    );
  }, []);

  const clearLocation = useCallback(() => {
    setCoordinates(null);
    setError(null);
    setPermissionDenied(false);
    setLoading(false);
  }, []);

  return {
    latitude: coordinates?.latitude ?? null,
    longitude: coordinates?.longitude ?? null,
    coordinates,
    loading,
    error,
    permissionDenied,
    requestLocation,
    clearLocation,
  };
}
