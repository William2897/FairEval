import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, Plus } from 'lucide-react';

interface Course {
  id: number;
  code: string;
  title: string;
  department: string;
  description: string;
}

function Courses() {
  const { data: courses, isLoading } = useQuery<Course[]>({
    queryKey: ['courses'],
    queryFn: async () => {
      // TODO: Implement API call
      return [];
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
        <h1 className="text-2xl font-bold text-gray-900">Courses</h1>
        <div className="flex space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search courses..."
              className="pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700">
            <Plus size={20} />
            <span>Add Course</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {courses?.map((course) => (
          <div key={course.id} className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-start">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{course.code}</h3>
                <p className="text-sm text-gray-500">{course.department}</p>
              </div>
            </div>
            <h4 className="mt-2 font-medium text-gray-900">{course.title}</h4>
            <p className="mt-2 text-sm text-gray-600 line-clamp-3">{course.description}</p>
            <button className="mt-4 text-sm text-indigo-600 hover:text-indigo-800">
              View Details →
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Courses