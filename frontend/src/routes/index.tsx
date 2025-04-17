import React from 'react';
import { createBrowserRouter, Navigate, Outlet } from 'react-router-dom';
import Login from '../pages/Login';
import Dashboard from '../pages/Dashboard';
import Evaluations from '../pages/Evaluations';
import Settings from '../pages/Settings';
import Layout from '../components/Layout';
import NotFound from '../pages/NotFound';
import SentimentAnalysis from '../pages/SentimentAnalysis';
import { useAuth } from '../contexts/AuthContext';

interface PrivateRouteProps {
  children: React.ReactNode;
}

function PrivateRoute({ children }: PrivateRouteProps) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

// AdminRoute component that checks for ADMIN role
interface AdminRouteProps {
  children: React.ReactNode;
}

function AdminRoute({ children }: AdminRouteProps) {
  const { isAuthenticated, user } = useAuth();
  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }
  return user?.role?.role === 'ADMIN' ? <>{children}</> : <Navigate to="/" />;
}

const router = createBrowserRouter([
  {
    path: '/',
    element: (
      <Layout>
        <PrivateRoute>
          <Outlet />
        </PrivateRoute>
      </Layout>
    ),
    children: [
      {
        index: true,
        element: <Dashboard />,
      },
      {
        path: 'evaluations',
        element: <AdminRoute><Evaluations /></AdminRoute>,
      },
      {
        path: 'sentiment',
        element: <SentimentAnalysis />,
      },
      {
        path: 'settings',
        element: <Settings />,
      }
    ],
  },
  {
    path: '/login',
    element: <Login />,
  },
  {
    path: '*',
    element: <NotFound />,
  }
]);

export default router;