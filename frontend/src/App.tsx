import { RouterProvider } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import router from './routes';
import { AuthProvider, useAuth } from './contexts/AuthContext';
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
      staleTime: 0, // Ensure data is always validated
      gcTime: 1000 * 60 * 5, // Cache for 5 minutes before garbage collection
    },
  },
});

function AppContent() {
  const { isAuthenticated, logout } = useAuth();

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const response = await axios.get('/api/auth/me/');
        if (!response.data.authenticated) {
          await logout();
          queryClient.clear(); // Clear all queries when logging out
        }
      } catch (error) {
        if (error instanceof AxiosError && (error.response?.status === 401 || error.response?.status === 403)) {
          await logout();
          queryClient.clear(); // Clear all queries when logging out
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