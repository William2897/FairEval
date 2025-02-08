import { Link } from 'react-router-dom';
import { Home } from 'lucide-react';

function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-indigo-600">404</h1>
        <p className="text-xl text-gray-600 mt-4">Page not found</p>
        <Link
          to="/"
          className="inline-flex items-center space-x-2 mt-8 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700"
        >
          <Home size={20} />
          <span>Return Home</span>
        </Link>
      </div>
    </div>
  );
}

export default NotFound;