import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, Plus, Users, Star } from 'lucide-react';
import { useState } from 'react';
import axios from 'axios';
import { DepartmentDetailModal } from '../components/DepartmentDetailModal';

interface Department {
  id: number;
  name: string;
  discipline: string;
  sub_discipline: string | null;
}

interface DepartmentStats {
  professorCount: number;
  avgRating: number;
}

function Departments() {
  const [selectedDept, setSelectedDept] = useState<Department | null>(null);

  const { data: departments, isLoading } = useQuery<Department[]>({
    queryKey: ['departments'],
    queryFn: async () => {
      const { data } = await axios.get('/api/departments/');
      return data;
    },
  });

  const { data: stats } = useQuery<Record<number, DepartmentStats>>({
    queryKey: ['departmentStats', departments],
    enabled: !!departments,
    queryFn: async () => {
      const { data } = await axios.get('/api/departments/stats/');
      return data;
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Academic Departments</h1>
        <div className="flex space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search departments..."
              className="pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
            <Plus size={20} />
            <span>Add Department</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {departments?.map((department) => (
          <div 
            key={department.id} 
            className="bg-white rounded-lg shadow-md p-6"
            onClick={() => setSelectedDept(department)}
          >
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {department.name}
                </h3>
                <p className="text-sm text-gray-500">{department.discipline}</p>
                {department.sub_discipline && (
                  <p className="text-sm text-gray-500">
                    Specialization: {department.sub_discipline}
                  </p>
                )}
              </div>
            </div>
            
            {stats && stats[department.id] && (
              <div className="mt-4 grid grid-cols-2 gap-4">
                <div className="flex items-center space-x-2 text-gray-600">
                  <Users size={20} />
                  <span>{stats[department.id].professorCount} Faculty</span>
                </div>
                <div className="flex items-center space-x-2 text-gray-600">
                  <Star size={20} />
                  <span>Avg Rating: {stats[department.id].avgRating.toFixed(1)}</span>
                </div>
              </div>
            )}

            <button className="mt-4 text-sm text-indigo-600 hover:text-indigo-800">
              View Details →
            </button>
          </div>
        ))}
      </div>

      {selectedDept && (
        <DepartmentDetailModal
          department={selectedDept}
          onClose={() => setSelectedDept(null)}
        />
      )}
    </div>
  );
}

export default Departments