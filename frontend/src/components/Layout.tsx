import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { 
  LayoutDashboard, 
  FileText, 
  GraduationCap, 
  Building2,
  Menu,
  X,
  LogOut
} from 'lucide-react';

interface NavItemProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  isActive: boolean;
}

function NavItem({ to, icon, label, isActive }: NavItemProps) {
  return (
    <Link
      to={to}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
        isActive 
          ? 'bg-indigo-700 text-white' 
          : 'text-gray-300 hover:bg-indigo-700 hover:text-white'
      }`}
    >
      {icon}
      <span>{label}</span>
    </Link>
  );
}

function Layout({ children }: { children: React.ReactNode }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const { logout } = useAuth();
  const location = useLocation();

  const navItems = [
    { to: '/', icon: <LayoutDashboard size={20} />, label: 'Dashboard' },
    { to: '/evaluations', icon: <FileText size={20} />, label: 'Evaluations' },
    { to: '/courses', icon: <GraduationCap size={20} />, label: 'Courses' },
    { to: '/departments', icon: <Building2 size={20} />, label: 'Departments' },
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 z-40 h-screen transition-transform ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } bg-indigo-800 text-white w-64 p-4`}
      >
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-bold">FairEval</h1>
          <button
            onClick={() => setIsSidebarOpen(false)}
            className="lg:hidden text-gray-300 hover:text-white"
          >
            <X size={24} />
          </button>
        </div>

        <nav className="space-y-2">
          {navItems.map((item) => (
            <NavItem
              key={item.to}
              {...item}
              isActive={location.pathname === item.to}
            />
          ))}
        </nav>

        <button
          onClick={logout}
          className="absolute bottom-4 left-4 flex items-center space-x-2 text-gray-300 hover:text-white"
        >
          <LogOut size={20} />
          <span>Logout</span>
        </button>
      </aside>

      {/* Main content */}
      <div className={`${isSidebarOpen ? 'lg:ml-64' : ''}`}>
        {/* Top bar */}
        <header className="bg-white shadow-sm">
          <div className="flex items-center justify-between px-4 py-3">
            <button
              onClick={() => setIsSidebarOpen(true)}
              className={`text-gray-600 lg:hidden ${isSidebarOpen ? 'hidden' : ''}`}
            >
              <Menu size={24} />
            </button>
          </div>
        </header>

        {/* Page content */}
        <main className="p-4">{children}</main>
      </div>
    </div>
  );
}

export default Layout;