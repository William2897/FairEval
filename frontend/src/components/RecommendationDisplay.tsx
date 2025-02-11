import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2 } from 'lucide-react';

interface Recommendation {
  id: number;
  text: string;
  priority: 'high' | 'medium' | 'low';
  impact_score: number;
  supporting_ratings: number;
  category: string;
}

interface RatingMetrics {
  avg_rating: number;
  helpful_rating: number;
  clarity_rating: number;
  difficulty_rating: number;
}

interface RecommendationData {
  categories: {
    teaching_effectiveness: {
      score: number;
      recommendations: Recommendation[];
    };
    clarity: {
      score: number;
      recommendations: Recommendation[];
    };
    workload: {
      score: number;
      recommendations: Recommendation[];
    };
  };
  overall_metrics: RatingMetrics;
  total_ratings: number;
  last_updated: string;
}

interface RecommendationDisplayProps {
  professorId: string;
  className?: string;
}

const PriorityBadge: React.FC<{ priority: Recommendation['priority'] }> = ({ priority }) => {
  const colors = {
    high: 'bg-red-100 text-red-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-green-100 text-green-800',
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[priority]}`}>
      {priority.charAt(0).toUpperCase() + priority.slice(1)} Priority
    </span>
  );
};

const MetricCard: React.FC<{
  label: string;
  value: number;
  type: 'rating' | 'difficulty';
}> = ({ label, value, type }) => (
  <div className="text-center p-4 bg-gray-50 rounded-lg">
    <div className="text-2xl font-bold text-gray-900">
      {value.toFixed(1)}
    </div>
    <div className="text-sm text-gray-600">{label}</div>
    {type === 'rating' && (
      <div className="mt-2 h-1 bg-gray-200 rounded-full">
        <div 
          className="h-1 bg-indigo-600 rounded-full" 
          style={{ width: `${(value / 5) * 100}%` }}
        />
      </div>
    )}
  </div>
);

const RecommendationCard: React.FC<{
  recommendation: Recommendation;
}> = ({ recommendation }) => (
  <div className="p-4 rounded-lg bg-gray-50">
    <div className="flex items-start space-x-3">
      <div className="flex-1">
        <div className="flex items-center justify-between mb-2">
          <PriorityBadge priority={recommendation.priority} />
          <span className="text-sm text-gray-500">
            {recommendation.supporting_ratings} supporting ratings
          </span>
        </div>
        <p className="text-gray-700">{recommendation.text}</p>
        <div className="mt-2">
          <div className="text-sm text-gray-500">
            Potential Impact: {recommendation.impact_score.toFixed(1)}/10
          </div>
          <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
            <div
              className="bg-indigo-600 rounded-full h-1.5"
              style={{ width: `${(recommendation.impact_score / 10) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  </div>
);

export const RecommendationDisplay: React.FC<RecommendationDisplayProps> = ({
  professorId,
  className = '',
}) => {
  const { data, isLoading, isError, error } = useQuery<RecommendationData>({
    queryKey: ['recommendations', professorId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/professors/${professorId}/recommendations/`);
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

  if (isError) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        Error loading recommendations: {(error as Error).message}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-yellow-50 text-yellow-700 p-4 rounded-lg">
        No recommendation data available.
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Overall Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Performance Overview</h2>
        <div className="grid grid-cols-4 gap-4">
          <MetricCard
            label="Overall Rating"
            value={data.overall_metrics.avg_rating}
            type="rating"
          />
          <MetricCard
            label="Helpfulness"
            value={data.overall_metrics.helpful_rating}
            type="rating"
          />
          <MetricCard
            label="Clarity"
            value={data.overall_metrics.clarity_rating}
            type="rating"
          />
          <MetricCard
            label="Difficulty"
            value={data.overall_metrics.difficulty_rating}
            type="difficulty"
          />
        </div>
      </div>

      {/* Recommendations by Category */}
      {Object.entries(data.categories).map(([category, { score, recommendations }]) => (
        <div key={category} className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900 capitalize">
              {category.replace('_', ' ')}
            </h3>
            <div className={`px-3 py-1 rounded-full text-sm font-medium
              ${score > 4 ? 'bg-green-100 text-green-800' :
                score > 2 ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'}`}
            >
              Score: {score.toFixed(1)}/5
            </div>
          </div>

          <div className="space-y-4">
            {recommendations.map((rec) => (
              <RecommendationCard key={rec.id} recommendation={rec} />
            ))}
          </div>
        </div>
      ))}

      <div className="text-sm text-gray-500 text-right">
        Based on {data.total_ratings} ratings • Last updated {new Date(data.last_updated).toLocaleDateString()}
      </div>
    </div>
  );
};