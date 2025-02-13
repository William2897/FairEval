import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, Filter, AlertOctagon } from 'lucide-react';
import { useState } from 'react';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { DataUploadForm } from '../components/DataUploadForm';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';

interface Rating {
  id: number;
  professor: number;
  professor_name: string;
  avg_rating: number;
  flag_status: string | null;
  class_name: string | null;
  helpful_rating: number | null;
  clarity_rating: number | null;
  difficulty_rating: number | null;
  is_online: boolean;
  is_for_credit: boolean;
  created_at: string;
}

function Evaluations() {
  const { user } = useAuth();
  const [selectedProfessorId, setSelectedProfessorId] = useState<number | null>(null);
  
  const { data: ratings, isLoading } = useQuery<Rating[]>({
    queryKey: ['ratings'],
    queryFn: async () => {
      const { data } = await axios.get('/api/ratings/');
      // Ensure we return an array even if the API response is wrapped
      return Array.isArray(data) ? data : data?.results || [];
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
        <h1 className="text-2xl font-bold text-gray-900">Course Evaluations</h1>
        <div className="flex space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search evaluations..."
              className="pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button className="flex items-center space-x-2 px-4 py-2 bg-white border rounded-md hover:bg-gray-50">
            <Filter size={20} />
            <span>Filter</span>
          </button>
        </div>
      </div>

      {/* Admin Section */}
      {user?.role.role === 'ADMIN' && (
        <>
          <DataUploadForm />

          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Institution-wide Topic Analysis</h2>
            <TopicModelVisualization />
          </div>
        </>
      )}

      <div className="grid gap-6">
        <div className="bg-white shadow-md rounded-lg overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Class
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Professor
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Overall Rating
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Helpful
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Clarity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Difficulty
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {ratings?.map((rating) => (
                <tr 
                  key={rating.id} 
                  className={`hover:bg-gray-50 cursor-pointer ${
                    selectedProfessorId === rating.professor ? 'bg-indigo-50' : ''
                  }`}
                  onClick={() => setSelectedProfessorId(rating.professor)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    {rating.class_name || 'N/A'}
                    {rating.flag_status && (
                      <div className="ml-2 inline-flex items-center" title={rating.flag_status}>
                        <AlertOctagon size={16} className="text-yellow-500" />
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">{rating.professor_name}</td>
                  <td className="px-6 py-4 whitespace-nowrap">{rating.avg_rating.toFixed(2)}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {rating.helpful_rating?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {rating.clarity_rating?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {rating.difficulty_rating?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex space-x-2">
                      {rating.is_online && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                          Online
                        </span>
                      )}
                      {rating.is_for_credit && (
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          For Credit
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {selectedProfessorId && (
          <>
            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Comment Summaries
              </h2>
              <CommentSummaryDisplay professorId={selectedProfessorId.toString()} />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Professor-Specific Topics
              </h2>
              <TopicModelVisualization professorId={selectedProfessorId.toString()} />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Improvement Recommendations
              </h2>
              <RecommendationDisplay professorId={selectedProfessorId.toString()} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Evaluations;