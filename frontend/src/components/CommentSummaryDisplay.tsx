import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { SmileIcon, FrownIcon } from 'lucide-react';
import { CommentTabs } from './CommentTabs';

interface SentimentSummary {
  total_comments: number;
  sentiment_breakdown: {
    positive: number;
    negative: number;
  };
  comments: Array<{
    comment: string;
    processed_comment: string;
    sentiment: number;
    created_at: string;
  }>;
  total_pages: number;
}

interface CommentSummaryDisplayProps {
  professorId: string;
  className?: string;
}

const fetchSentimentSummary = async (professorId: string, page: number = 1): Promise<SentimentSummary> => {
  const { data } = await axios.get(`/api/professors/${professorId}/sentiment-summary/?page=${page}`);
  return data;
};

export const CommentSummaryDisplay: React.FC<CommentSummaryDisplayProps> = ({ 
  professorId,
  className = ''
}) => {
  const [currentPage, setCurrentPage] = useState(1);
  const { 
    data: summary,
    isLoading,
    isError,
    error
  } = useQuery({
    queryKey: ['sentimentSummary', professorId, currentPage],
    queryFn: () => fetchSentimentSummary(professorId, currentPage)
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

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Sentiment Distribution with Emojis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">
          Overall Sentiment Distribution
        </h3>
        <div className="grid grid-cols-2 gap-6">
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
      <CommentTabs comments={summary.comments} />

      {/* Pagination */}
      {summary.total_pages > 1 && (
        <div className="flex justify-center space-x-2 mt-4">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="px-4 py-2 text-sm font-medium text-gray-700">
            Page {currentPage} of {summary.total_pages}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(summary.total_pages, p + 1))}
            disabled={currentPage === summary.total_pages}
            className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};