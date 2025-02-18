import { useAuth } from '../contexts/AuthContext';

function Settings() {
  const { user } = useAuth();

  return (
    <div className="space-y-8 p-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Account Settings</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Name</label>
            <div className="mt-1 text-gray-900">
              {user?.first_name} {user?.last_name}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700">Role</label>
            <div className="mt-1 text-gray-900">
              {user?.role?.role === 'ACADEMIC' ? 'Professor' : 'Administrator'}
            </div>
          </div>

          {user?.role?.discipline && (
            <div>
              <label className="block text-sm font-medium text-gray-700">Discipline</label>
              <div className="mt-1 text-gray-900">
                {user.role.discipline}
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Preferences</h2>
        <p className="text-gray-600">Added in a future update.</p>
      </div>
    </div>
  );
}

export default Settings;