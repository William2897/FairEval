import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface CommentSummary {
  positive_summary: string;
  negative_summary: string;
}

interface CommentSummaryDisplayProps {
  professorId: string;
  className?: string;
}

const fetchCommentSummary = async (professorId: string): Promise<CommentSummary> => {
  const { data } = await axios.get(`/api/professors/${professorId}/comment_summary/`);
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
    queryKey: ['commentSummary', professorId],
    queryFn: () => fetchCommentSummary(professorId)
  });

  if (isLoading) {
    return (
      <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
        <p className="text-red-600">
          Error loading comment summaries: {(error as Error).message}
        </p>
      </div>
    );
  }

  return (
    <div className={`p-4 bg-white rounded-lg shadow ${className}`}>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-green-600 mb-2">
            Positive Feedback Summary
          </h3>
          <p className="text-gray-700">{summary?.positive_summary}</p>
        </div>
        
        <div>
          <h3 className="text-lg font-semibold text-red-600 mb-2">
            Areas for Improvement
          </h3>
          <p className="text-gray-700">{summary?.negative_summary}</p>
        </div>
      </div>
    </div>
  );
}; 