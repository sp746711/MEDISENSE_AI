import { createContext, useCallback, useEffect, useMemo, useState } from 'react';
import * as authService from '../services/authService';
import { ApiError } from '../services/api';

export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(authService.getStoredUser());
  const [token, setToken] = useState(() => localStorage.getItem('medisense_token'));
  const [loading, setLoading] = useState(
    Boolean(localStorage.getItem('medisense_token') || localStorage.getItem('medisense_refresh_token'))
  );
  const [error, setError] = useState(null);

  const refreshUser = useCallback(async () => {
    const accessToken = localStorage.getItem('medisense_token');
    const refreshToken = localStorage.getItem('medisense_refresh_token');
    if (!accessToken && !refreshToken) {
      setUser(null);
      setToken(null);
      setLoading(false);
      return null;
    }
    try {
      const me = await authService.getMe();
      setUser(me);
      setToken(localStorage.getItem('medisense_token'));
      localStorage.setItem('medisense_user', JSON.stringify(me));
      setError(null);
      return me;
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        authService.logout();
        setUser(null);
        setToken(null);
      }
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = useCallback(async (email, password) => {
    setError(null);
    const data = await authService.login({ email, password });
    setToken(data.access_token);
    setUser(data.user);
    return data;
  }, []);

  const register = useCallback(async (payload) => {
    setError(null);
    const data = await authService.register(payload);
    setToken(data.access_token);
    setUser(data.user);
    return data;
  }, []);

  const logout = useCallback(() => {
    authService.logout();
    setUser(null);
    setToken(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      token,
      loading,
      error,
      isAuthenticated: Boolean(token && user),
      login,
      register,
      logout,
      refreshUser,
      setUser,
    }),
    [user, token, loading, error, login, register, logout, refreshUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
