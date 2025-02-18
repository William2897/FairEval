import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import axios from 'axios';

// Create an axios instance with the configuration
const api = axios.create({
  baseURL: '/api',  // Use relative URL
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest'
  }
});

// Add request interceptor to get CSRF token from cookie
api.interceptors.request.use((config) => {
  const csrfToken = document.cookie.split('; ')
    .find(row => row.startsWith('csrftoken='))
    ?.split('=')[1];
  
  if (csrfToken) {
    config.headers['X-CSRFToken'] = csrfToken;
  }
  return config;
});

interface User {
  id: string;
  username: string;  // This is the professor_id for academic users
  email: string;
  first_name: string;
  last_name: string;
  role?: {
    role: string;
    discipline: string;
  };
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  authStateChanged: number; // Add this to trigger re-renders
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    const savedUser = localStorage.getItem('user');
    return savedUser ? JSON.parse(savedUser) : null;
  });
  const [authStateChanged, setAuthStateChanged] = useState(0);
  const queryClient = useQueryClient();

  useEffect(() => {
    // Check authentication on mount
    const checkAuth = async () => {
      try {
        const response = await api.get('/auth/me/');
        if (response.data.authenticated) {
          setUser(response.data);
          localStorage.setItem('user', JSON.stringify(response.data));
          setAuthStateChanged(prev => prev + 1); // Increment counter on auth change
        } else {
          setUser(null);
          localStorage.removeItem('user');
          queryClient.clear(); // Clear all cached data
          setAuthStateChanged(prev => prev + 1);
        }
      } catch (error) {
        setUser(null);
        localStorage.removeItem('user');
        queryClient.clear(); // Clear all cached data
        setAuthStateChanged(prev => prev + 1);
      }
    };
    checkAuth();
  }, [queryClient]);

  const login = useCallback(async (username: string, password: string) => {
    try {
      const response = await api.post('/auth/login/', { 
        username, 
        password 
      });

      const userData = response.data.user;
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      queryClient.clear(); // Clear previous user's cached data
      setAuthStateChanged(prev => prev + 1); // Increment counter on login
    } catch (error) {
      console.error('Login error:', error);
      throw new Error('Login failed');
    }
  }, [queryClient]);

  const logout = useCallback(async () => {
    try {
      await api.post('/auth/logout/');
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      localStorage.removeItem('user');
      // Clear all cookies
      document.cookie.split(';').forEach(cookie => {
        document.cookie = cookie
          .replace(/^ +/, '')
          .replace(/=.*/, `=;expires=${new Date().toUTCString()};path=/`);
      });
      queryClient.clear(); // Clear all cached data
      setAuthStateChanged(prev => prev + 1); // Increment counter on logout
    }
  }, [queryClient]);

  const value = {
    isAuthenticated: !!user,
    user,
    login,
    logout,
    authStateChanged, // Include in context value
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}