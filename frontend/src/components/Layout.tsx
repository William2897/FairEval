import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { 
  LayoutDashboard, 
  FileText, 
  Menu,
  X,
  LogOut,
  BarChart,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';

function Layout({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1024);
  const { logout, user } = useAuth();
  const location = useLocation();

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 1024;
      setIsMobile(mobile);
      if (!mobile) {
        setIsSidebarOpen(true);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial check
    
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Define navigation items based on user role
  const navItems = [
    { to: '/', icon: <LayoutDashboard size={20} />, label: 'Dashboard' },
    // Only show Evaluations tab for ADMIN users
    ...(user?.role?.role === 'ADMIN' ? [
      { to: '/evaluations', icon: <FileText size={20} />, label: 'Evaluations' }
    ] : []),
    { to: '/sentiment', icon: <BarChart size={20} />, label: 'Sentiment Analysis' },
    { to: '/settings', icon: <Settings size={20} />, label: 'Settings' }
  ];

  const toggleSidebar = () => {
    setIsSidebarOpen(prev => !prev);
  };

  return (
    <div className="min-h-screen bg-gray-100 relative">
      {/* Mobile overlay */}
      {isMobile && isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40"
          onClick={toggleSidebar}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 h-screen bg-indigo-800 text-white z-50
          transition-all duration-300 ease-in-out
          ${isSidebarOpen ? 'w-64' : 'w-16'}
          ${isMobile ? (isSidebarOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'}
        `}
      >
        <div className="flex flex-col h-full p-4">
          <div className="flex items-center justify-between mb-8">
            <h1 className={`text-2xl font-bold transition-opacity duration-300 
              ${!isSidebarOpen ? 'opacity-0 w-0 overflow-hidden' : 'opacity-100'}`}>
              FairEval
            </h1>
            {!isMobile && (
              <button
                onClick={toggleSidebar}
                className="text-gray-300 hover:text-white"
              >
                {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
              </button>
            )}
          </div>

          <nav className="space-y-2 flex-1">
            {navItems.map((item) => (
              <Link
                key={item.to}
                to={item.to}
                className={`
                  flex items-center rounded-lg transition-colors
                  ${!isSidebarOpen ? 'justify-center px-2' : 'px-4'} 
                  py-2
                  ${location.pathname === item.to
                    ? 'bg-indigo-700 text-white'
                    : 'text-gray-300 hover:bg-indigo-700 hover:text-white'
                  }
                `}
              >
                <div className="flex-shrink-0 w-5 h-5">
                  {item.icon}
                </div>
                <span 
                  className={`ml-2 whitespace-nowrap transition-opacity duration-300
                    ${!isSidebarOpen ? 'opacity-0 w-0 overflow-hidden' : 'opacity-100'}
                  `}
                >
                  {item.label}
                </span>
              </Link>
            ))}
          </nav>

          <button
            onClick={logout}
            className={`
              flex items-center text-gray-300 hover:text-white transition-colors
              ${!isSidebarOpen ? 'justify-center px-2' : 'px-4'} 
              py-2 rounded-lg hover:bg-indigo-700
            `}
          >
            <div className="flex-shrink-0 w-5 h-5">
              <LogOut size={20} />
            </div>
            <span 
              className={`ml-2 whitespace-nowrap transition-opacity duration-300
                ${!isSidebarOpen ? 'opacity-0 w-0 overflow-hidden' : 'opacity-100'}
              `}
            >
              Logout
            </span>
          </button>
        </div>
      </aside>

      {/* Main content */}
      <div 
        className={`
          transition-all duration-300 
          ${isSidebarOpen ? 'lg:ml-64' : 'lg:ml-16'}
          ${isMobile ? 'ml-0' : ''}
        `}
      >
        {/* Top bar */}
        <header className="bg-white shadow-sm">
          <div className="flex items-center justify-between px-4 py-3">
            <button
              onClick={toggleSidebar}
              className="text-gray-600 lg:hidden"
            >
              {isSidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4">
          {children}
        </main>
      </div>
    </div>
  );
}

export default Layout;