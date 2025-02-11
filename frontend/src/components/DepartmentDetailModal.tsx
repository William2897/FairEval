import { X } from 'lucide-react';

interface Department {
  id: number;
  name: string;
  discipline: string;
  sub_discipline: string | null;
}

interface DepartmentDetailModalProps {
  department: Department;
  onClose: () => void;
}

export function DepartmentDetailModal({ department, onClose }: DepartmentDetailModalProps) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg w-full max-w-2xl p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900">{department.name}</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <X size={24} />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Discipline</h3>
            <p className="text-gray-600">{department.discipline}</p>
          </div>

          {department.sub_discipline && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900">Specialization</h3>
              <p className="text-gray-600">{department.sub_discipline}</p>
            </div>
          )}

          {/* Additional sections can be added here as needed */}
        </div>

        <div className="mt-6 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}