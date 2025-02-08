import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import Login from '../pages/Login';
import Dashboard from '../pages/Dashboard';
import Evaluations from '../pages/Evaluations';
import Courses from '../pages/Courses';
import Departments from '../pages/Departments';
import NotFound from '../pages/NotFound';

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      
      <Route path="/" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
      <Route path="/evaluations" element={<PrivateRoute><Evaluations /></PrivateRoute>} />
      <Route path="/courses" element={<PrivateRoute><Courses /></PrivateRoute>} />
      <Route path="/departments" element={<PrivateRoute><Departments /></PrivateRoute>} />
      
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

export default AppRoutes;