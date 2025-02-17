import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { SmileIcon, MehIcon, FrownIcon } from 'lucide-react';
import { CommentTabs } from './CommentTabs';

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

  if (isLoading) return (
    <div className="animate-pulse flex space-x-4 p-4">
      <div className="flex-1 space-y-4 py-1">
        <div className="h-4 bg-gray-200 rounded w-3/4"></div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-200 rounded"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
        </div>
      </div>
    </div>
  );

  if (isError) {
    return (
      <div className="text-red-600 bg-red-50 p-4 rounded-lg">
        Error loading summaries: {(error as Error).message}
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="text-gray-500 bg-gray-50 p-4 rounded-lg text-center">
        No data available
      </div>
    );
  }

  const positivePercentage = ((summary.sentiment_breakdown.positive / summary.total_comments) * 100).toFixed(1);
  const negativePercentage = ((summary.sentiment_breakdown.negative / summary.total_comments) * 100).toFixed(1);
  const neutralPercentage = (100 - parseFloat(positivePercentage) - parseFloat(negativePercentage)).toFixed(1);

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Sentiment Distribution with Emojis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">
          Overall Sentiment Distribution
        </h3>
        <div className="grid grid-cols-3 gap-6">
          <div className="flex flex-col items-center p-4 bg-green-50 rounded-lg">
            <SmileIcon size={32} className="text-green-600 mb-2" />
            <div className="text-2xl font-bold text-green-600">
              {positivePercentage}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Positive</div>
            <div className="text-xs text-gray-500 mt-1">
              {summary.sentiment_breakdown.positive} comments
            </div>
          </div>
          <div className="flex flex-col items-center p-4 bg-gray-50 rounded-lg">
            <MehIcon size={32} className="text-gray-600 mb-2" />
            <div className="text-2xl font-bold text-gray-600">
              {neutralPercentage}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Neutral</div>
            <div className="text-xs text-gray-500 mt-1">
              {summary.total_comments - (summary.sentiment_breakdown.positive + summary.sentiment_breakdown.negative)} comments
            </div>
          </div>
          <div className="flex flex-col items-center p-4 bg-red-50 rounded-lg">
            <FrownIcon size={32} className="text-red-600 mb-2" />
            <div className="text-2xl font-bold text-red-600">
              {negativePercentage}%
            </div>
            <div className="text-sm text-gray-600 mt-1">Negative</div>
            <div className="text-xs text-gray-500 mt-1">
              {summary.sentiment_breakdown.negative} comments
            </div>
          </div>
        </div>
      </div>

      {/* Comments Tabs */}
      <CommentTabs comments={summary.recent_sentiments} />
    </div>
  );
};