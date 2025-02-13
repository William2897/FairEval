import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, FileSpreadsheet, Settings as SettingsIcon } from 'lucide-react';

export function Navigation() {
  const location = useLocation();

  const links = [
    {
      to: '/',
      icon: LayoutDashboard,
      label: 'Dashboard',
      description: 'Overview and discipline analysis'
    },
    {
      to: '/evaluations',
      icon: FileSpreadsheet,
      label: 'Evaluations',
      description: 'View and manage evaluations'
    },
    {
      to: '/settings',
      icon: SettingsIcon,
      label: 'Settings',
      description: 'Account and preferences'
    }
  ];

  return (
    <nav className="space-y-1">
      {links.map(({ to, icon: Icon, label, description }) => {
        const isActive = location.pathname === to;
        
        return (
          <Link
            key={to}
            to={to}
            className={`
              flex items-center px-3 py-2 text-sm font-medium rounded-md
              ${isActive
                ? 'bg-indigo-100 text-indigo-700'
                : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
              }
            `}
          >
            <Icon size={20} className="mr-3 flex-shrink-0" />
            <div>
              <div className="font-medium">{label}</div>
              <div className="text-xs text-gray-500">{description}</div>
            </div>
          </Link>
        );
      })}
    </nav>
  );
}