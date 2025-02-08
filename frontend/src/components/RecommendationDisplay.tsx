import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2, ChevronDown, ChevronUp, CheckCircle2, AlertTriangle, Info } from 'lucide-react';

interface Recommendation {
  id: number;
  text: string;
  priority: 'high' | 'medium' | 'low';
  impact_score: number;
  evidence_count: number;
}

interface RecommendationCategory {
  name: string;
  description: string;
  recommendations: Recommendation[];
  improvement_score: number;
}

interface RecommendationData {
  categories: RecommendationCategory[];
  overall_score: number;
  last_updated: string;
  total_reviews_analyzed: number;
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

const CategoryCard: React.FC<{
  category: RecommendationCategory;
}> = ({ category }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="bg-white rounded-lg shadow">
      <div
        className="p-4 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-gray-900">
              {category.name}
            </h3>
            <span className={`px-2 py-1 rounded-full text-xs font-medium
              ${category.improvement_score > 7
                ? 'bg-red-100 text-red-800'
                : category.improvement_score > 4
                ? 'bg-yellow-100 text-yellow-800'
                : 'bg-green-100 text-green-800'
              }`}
            >
              Score: {category.improvement_score.toFixed(1)}
            </span>
          </div>
          {isExpanded ? (
            <ChevronUp size={20} className="text-gray-500" />
          ) : (
            <ChevronDown size={20} className="text-gray-500" />
          )}
        </div>
        
        <p className="text-sm text-gray-600 mt-1">
          {category.description}
        </p>
      </div>

      {isExpanded && (
        <div className="border-t border-gray-100 p-4 space-y-4">
          {category.recommendations.map((rec) => (
            <div
              key={rec.id}
              className="flex items-start space-x-3 p-3 rounded-md bg-gray-50"
            >
              {rec.priority === 'high' ? (
                <AlertTriangle className="text-red-500 mt-0.5" size={18} />
              ) : rec.priority === 'medium' ? (
                <Info className="text-yellow-500 mt-0.5" size={18} />
              ) : (
                <CheckCircle2 className="text-green-500 mt-0.5" size={18} />
              )}
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <PriorityBadge priority={rec.priority} />
                  <span className="text-xs text-gray-500">
                    Based on {rec.evidence_count} reviews
                  </span>
                </div>
                <p className="text-gray-700">{rec.text}</p>
                <div className="mt-2">
                  <div className="text-xs text-gray-500">
                    Potential Impact Score: {rec.impact_score.toFixed(1)}/10
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5 mt-1">
                    <div
                      className="bg-indigo-600 rounded-full h-1.5"
                      style={{ width: `${(rec.impact_score / 10) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

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

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">
            Improvement Recommendations
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Based on {data.total_reviews_analyzed} reviews • Last updated {new Date(data.last_updated).toLocaleDateString()}
          </p>
        </div>
        <div className={`px-3 py-1 rounded-full text-sm font-medium
          ${data.overall_score > 7
            ? 'bg-red-100 text-red-800'
            : data.overall_score > 4
            ? 'bg-yellow-100 text-yellow-800'
            : 'bg-green-100 text-green-800'
          }`}
        >
          Overall Score: {data.overall_score.toFixed(1)}/10
        </div>
      </div>

      <div className="grid gap-6">
        {data.categories
          .sort((a, b) => b.improvement_score - a.improvement_score)
          .map((category) => (
            <CategoryCard
              key={category.name}
              category={category}
            />
          ))}
      </div>
    </div>
  );
}; 