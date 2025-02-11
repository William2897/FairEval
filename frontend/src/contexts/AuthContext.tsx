import { createContext, useContext, useState, useCallback, useEffect } from 'react';
import axios from 'axios';

interface User {
  id: string;
  username: string;
  email: string;
  first_name: string;
  last_name: string;
  role: 'professor' | 'academic_admin';
}

interface AuthContextType {
  isAuthenticated: boolean;
  user: User | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    const savedUser = localStorage.getItem('user');
    return savedUser ? JSON.parse(savedUser) : null;
  });

  // Set up axios defaults
  useEffect(() => {
    if (user) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${localStorage.getItem('token')}`;
    } else {
      delete axios.defaults.headers.common['Authorization'];
    }
  }, [user]);

  const login = useCallback(async (username: string, password: string) => {
    try {
      // Get CSRF token first
      await axios.get('/api/csrf/');
      
      // Perform login
      const response = await axios.post('/api/auth/login/', { 
        username, 
        password 
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        withCredentials: true
      });

      const userData = {
        ...response.data.user,
        role: response.data.user.groups[0]
      };

      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
      localStorage.setItem('token', response.data.token);
      
      axios.defaults.headers.common['Authorization'] = `Bearer ${response.data.token}`;
    } catch (error) {
      console.error('Login error:', error);
      throw new Error('Login failed');
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await axios.post('/api/auth/logout/', {}, {
        withCredentials: true
      });
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setUser(null);
      localStorage.removeItem('user');
      localStorage.removeItem('token');
      delete axios.defaults.headers.common['Authorization'];
    }
  }, []);

  const value = {
    isAuthenticated: !!user,
    user,
    login,
    logout,
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