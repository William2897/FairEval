import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, Filter } from 'lucide-react';
import { useState } from 'react';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { DataUploadForm } from '../components/DataUploadForm';
import { DepartmentGenderBiasDashboard } from '../components/DepartmentGenderBiasDashboard';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { useAuth } from '../contexts/AuthContext';

interface Evaluation {
  id: number;
  course: string;
  professor: string;
  semester: string;
  year: number;
  teaching_effectiveness: number;
  course_content: number;
  workload_fairness: number;
  sentiment_score: number;
}

function Evaluations() {
  const { user } = useAuth();
  const [selectedProfessorId, setSelectedProfessorId] = useState<string | null>(null);
  const { data: evaluations, isLoading } = useQuery<Evaluation[]>({
    queryKey: ['evaluations'],
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
        <h1 className="text-2xl font-bold text-gray-900">Evaluations</h1>
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
      {user?.role === 'admin' && (
        <>
          <DataUploadForm />
          
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Bias Analysis</h2>
            <DepartmentGenderBiasDashboard />
          </div>

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
                  Course
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Professor
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Semester
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Teaching Score
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sentiment
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {evaluations?.map((evaluation) => (
                <tr 
                  key={evaluation.id} 
                  className={`hover:bg-gray-50 cursor-pointer ${
                    selectedProfessorId === evaluation.id.toString() ? 'bg-indigo-50' : ''
                  }`}
                  onClick={() => setSelectedProfessorId(evaluation.id.toString())}
                >
                  <td className="px-6 py-4 whitespace-nowrap">{evaluation.course}</td>
                  <td className="px-6 py-4 whitespace-nowrap">{evaluation.professor}</td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {evaluation.semester} {evaluation.year}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {evaluation.teaching_effectiveness.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {evaluation.sentiment_score.toFixed(2)}
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
              <CommentSummaryDisplay professorId={selectedProfessorId} />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Professor-Specific Topics
              </h2>
              <TopicModelVisualization professorId={selectedProfessorId} />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Improvement Recommendations
              </h2>
              <RecommendationDisplay professorId={selectedProfessorId} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Evaluations;