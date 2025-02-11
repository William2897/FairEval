import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface SentimentSummary {
  total_comments: number;
  sentiment_breakdown: {
    positive: number;
    negative: number;
  };
  recent_sentiments: Array<{
    comment: string;
    processed_comment: string;
    sentiment: number;
    created_at: string;
  }>;
}

interface CommentSummaryDisplayProps {
  professorId: string;
  className?: string;
}

const fetchSentimentSummary = async (professorId: string): Promise<SentimentSummary> => {
  const { data } = await axios.get(`/api/professors/${professorId}/sentiment-summary/`);
  return data;
};

export const CommentSummaryDisplay: React.FC<CommentSummaryDisplayProps> = ({ 
  professorId,
  className = ''
}) => {
  const { 
    data: summary,
    isLoading,
    isError,
    error
  } = useQuery({
    queryKey: ['sentimentSummary', professorId],
    queryFn: () => fetchSentimentSummary(professorId)
  });

  if (isLoading) return <div className="animate-pulse">Loading...</div>;

  if (isError) {
    return (
      <div className="text-red-600">
        Error loading summaries: {(error as Error).message}
      </div>
    );
  }

  if (!summary) {
    return <div>No data available</div>;
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Sentiment Distribution */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Overall Sentiment Distribution
        </h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">
              {((summary.sentiment_breakdown.positive / summary.total_comments) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Positive</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-600">
              {((summary.sentiment_breakdown.negative / summary.total_comments) * 100).toFixed(1)}%
            </div>
            <div className="text-sm text-gray-600">Negative</div>
          </div>
        </div>
      </div>

      {/* Recent Comments */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Comments
        </h3>
        <div className="space-y-4">
          {summary.recent_sentiments.map((sentiment, index) => (
            <div 
              key={index}
              className={`p-4 rounded-lg ${
                sentiment.sentiment > 0 ? 'bg-green-50' :
                sentiment.sentiment < 0 ? 'bg-red-50' : 'bg-gray-50'
              }`}
            >
              <p className="text-gray-700">{sentiment.comment}</p>
              <div className="mt-2 flex justify-between items-center text-sm">
                <span className="text-gray-500">
                  {new Date(sentiment.created_at).toLocaleDateString()}
                </span>
                <span className={`font-medium ${
                  sentiment.sentiment > 0 ? 'text-green-600' :
                  sentiment.sentiment < 0 ? 'text-red-600' : 'text-gray-600'
                }`}>
                  Sentiment Score: {sentiment.sentiment}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};