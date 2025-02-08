import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';
import ProfessorSentimentDashboard from '../components/ProfessorSentimentDashboard';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { useAuth } from '../contexts/AuthContext';

function Dashboard() {
  const { user } = useAuth();
  const { data: stats, isLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      // TODO: Implement API call
      return {
        evaluationCount: 150,
        averageScores: [
          { semester: 'Fall 2024', score: 4.2 },
          { semester: 'Spring 2024', score: 4.1 },
          { semester: 'Fall 2023', score: 3.9 },
          { semester: 'Spring 2023', score: 4.0 },
        ],
      };
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
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900">Total Evaluations</h3>
          <p className="text-3xl font-bold text-indigo-600 mt-2">
            {stats?.evaluationCount}
          </p>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-medium text-gray-900 mb-4">
          Average Scores by Semester
        </h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={stats?.averageScores}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="semester" />
              <YAxis domain={[0, 5]} />
              <Tooltip />
              <Bar dataKey="score" fill="#4f46e5" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Show sentiment dashboard only for professors */}
      {user?.role === 'professor' && (
        <>
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Sentiment Analysis</h2>
            <ProfessorSentimentDashboard professorId={user.id} />
          </div>
          
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Comment Summaries</h2>
            <CommentSummaryDisplay professorId={user.id} />
          </div>

          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Topic Analysis</h2>
            <TopicModelVisualization professorId={user.id} />
          </div>

          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Recommendations</h2>
            <RecommendationDisplay professorId={user.id} />
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;