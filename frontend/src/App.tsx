import { RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import router from './routes';
import { AuthProvider } from './contexts/AuthContext';
import { useAuth } from './contexts/AuthContext';
import { useEffect } from 'react';
import axios, { AxiosError } from 'axios';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        if (error instanceof AxiosError && (error.response?.status === 401 || error.response?.status === 403)) {
          return false;
        }
        return failureCount < 3;
      },
    },
  },
});

function AppContent() {
  const { isAuthenticated, logout } = useAuth();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await axios.get('/api/auth/me/');
        // If we get here, we're authenticated
        if (!response.data.authenticated) {
          logout();
        }
      } catch (error) {
        // If we get a 401/403, we're not authenticated
        if (error instanceof AxiosError && (error.response?.status === 401 || error.response?.status === 403)) {
          logout();
        }
      }
    };
    checkAuth();
  }, [isAuthenticated, logout]);

  return <RouterProvider router={router} />;
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
      <ReactQueryDevtools />
    </QueryClientProvider>
  );
}