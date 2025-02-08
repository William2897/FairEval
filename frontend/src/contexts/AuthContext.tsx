import { createContext, useContext, useState, useCallback } from 'react';
import axios from 'axios';

interface AuthContextType {
  isAuthenticated: boolean;
  user: any | null;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<any | null>(null);

  const login = useCallback(async (username: string, password: string) => {
    try {
      const response = await axios.post('/api/auth/login/', { username, password });
      setUser(response.data);
    } catch (error) {
      throw new Error('Login failed');
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
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