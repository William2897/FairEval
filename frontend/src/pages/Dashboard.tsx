import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { DisciplineAnalysisDashboard } from '../components/DisciplineAnalysisDashboard';
import { useAuth } from '../contexts/AuthContext';

interface DashboardStats {
  evaluationCount: number;
  metrics?: {
    avg_rating: number;
    avg_helpful: number;
    avg_clarity: number;
    avg_difficulty: number;
    trend: number;
  };
}

function Dashboard() {
  const { user } = useAuth();
  const { data: stats, isLoading, isError } = useQuery<DashboardStats>({
    queryKey: ['dashboard-stats'],
    queryFn: async () => {
      if (user?.role?.role === 'ACADEMIC') {
        const metrics = await axios.get(`/api/professors/${user.username}/metrics/`);
        return {
          evaluationCount: metrics.data.total_ratings,
          metrics: metrics.data
        };
      } else {
        const { data } = await axios.get('/api/ratings/stats/');
        return data;
      }
    },
    retry: 2,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });

  if (isError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-600">
          Failed to load dashboard data. Please try refreshing the page.
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 w-48 bg-gray-200 rounded"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="bg-gray-100 h-32 rounded-lg"></div>
          ))}
        </div>
        <div className="bg-gray-100 h-64 rounded-lg"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 p-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        </div>
        {user && (
          <div className="text-right">
            <p className="text-lg font-medium text-gray-900">
              {user.first_name} {user.last_name}
            </p>
            <p className="text-sm text-gray-500">
              {user.role?.role === 'ACADEMIC' ? 'Professor' : 'Administrator'}
            </p>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900">Total Evaluations</h3>
          <p className="text-3xl font-bold text-indigo-600 mt-2">
            {stats?.evaluationCount || 0}
          </p>
        </div>
        {stats?.metrics && (
          <>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-medium text-gray-900">Average Rating</h3>
              <p className="text-3xl font-bold text-indigo-600 mt-2">
                {stats.metrics.avg_rating.toFixed(2)}
              </p>
              <p className={`text-sm ${stats.metrics.trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {stats.metrics.trend > 0 ? '↑' : '↓'} {Math.abs(stats.metrics.trend).toFixed(2)} vs last month
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-medium text-gray-900">Clarity Score</h3>
              <p className="text-3xl font-bold text-indigo-600 mt-2">
                {stats.metrics.avg_clarity.toFixed(2)}
              </p>
            </div>
            <div className="bg-white p-6 rounded-lg shadow">
              <h3 className="text-lg font-medium text-gray-900">Helpfulness Score</h3>
              <p className="text-3xl font-bold text-indigo-600 mt-2">
                {stats.metrics.avg_helpful.toFixed(2)}
              </p>
            </div>
          </>
        )}
      </div>

      {/* Discipline Analysis Section */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Discipline Analysis</h2>
        <DisciplineAnalysisDashboard />
      </div>

      {/* Show topic analysis and recommendations for professors */}
      {user?.role?.role === 'ACADEMIC' && (
        <>
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Topic Analysis</h2>
            <TopicModelVisualization professorId={user.username} />
          </div>

          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-4">Recommendations</h2>
            <RecommendationDisplay professorId={user.username} />
          </div>
        </>
      )}
    </div>
  );
}

export default Dashboard;