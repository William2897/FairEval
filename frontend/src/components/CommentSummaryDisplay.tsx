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
    id: number;
    comment: string;
    processed_comment: string;
    sentiment: number;
    created_at: string;
    bias_tag: string | null;
    bias_interpretation: string | null;
    stereotype_bias_score: number | null;
    objective_focus_percentage: number | null;
  }>;
  total_pages: number;
  message?: string;
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

  // Handle case where there are no comments
  if (summary.total_comments === 0 || summary.message?.includes('No sentiment data')) {
    return (
      <div className="text-gray-600 bg-gray-50 p-6 rounded-lg text-center">
        <p className="text-lg font-medium">No comments available for analysis</p>
        <p className="text-sm mt-2">This professor doesn't have any student comments to analyze yet.</p>
      </div>
    );
  }
  const positivePercentage = ((summary.sentiment_breakdown.positive / summary.total_comments) * 100).toFixed(1);
  const negativePercentage = ((summary.sentiment_breakdown.negative / summary.total_comments) * 100).toFixed(1);

  // Calculate bias tag distributions
  const calculateBiasDistribution = (comments: SentimentSummary['comments'], sentiment: number) => {
    // Filter by sentiment
    const filteredComments = comments.filter(c => c.sentiment === sentiment);
    if (filteredComments.length === 0) return {};
    
    // Count occurrences of each bias tag
    const tagCounts: Record<string, number> = {};
    filteredComments.forEach(comment => {
      const tag = comment.bias_tag || 'UNKNOWN';
      tagCounts[tag] = (tagCounts[tag] || 0) + 1;
    });
    
    // Calculate percentages
    const tagPercentages: Record<string, string> = {};
    const total = filteredComments.length;
    
    Object.entries(tagCounts).forEach(([tag, count]) => {
      tagPercentages[tag] = ((count / total) * 100).toFixed(1);
    });
    
    return { counts: tagCounts, percentages: tagPercentages };
  };
  
  // Get bias tag distribution for positive and negative comments
  const positiveBiasDistribution = calculateBiasDistribution(summary.comments, 1);
  const negativeBiasDistribution = calculateBiasDistribution(summary.comments, 0);
  
  // Helper function to get label for bias tag
  const getBiasTagLabel = (tag: string): string => {
    switch (tag) {
      case 'POS_BIAS_M': return 'Male Stereotype Focus';
      case 'POS_BIAS_F': return 'Female Stereotype Focus';
      case 'NEG_BIAS_M': return 'Male Negative Bias';
      case 'NEG_BIAS_F': return 'Female Negative Bias';
      case 'OBJECTIVE': case 'OBJECTIVE_M_LEAN': case 'OBJECTIVE_F_LEAN': return 'Objective Focus';
      case 'NEUTRAL': return 'Neutral Pattern';
      case 'UNKNOWN': return 'Analysis Error';
      default: return 'No Bias Tag';
    }
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Sentiment Distribution with Emojis */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-6">
          Overall Sentiment Distribution
        </h3>
        <div className="grid grid-cols-2 gap-6">
          <div className="flex flex-col p-4 bg-green-50 rounded-lg">
            <div className="flex items-center justify-center mb-4">
              <SmileIcon size={32} className="text-green-600 mr-2" />
              <div className="text-2xl font-bold text-green-600">
                {positivePercentage}%
              </div>
            </div>
            <div className="text-sm text-gray-600 mt-1 text-center">Positive</div>
            <div className="text-xs text-gray-500 mt-1 text-center mb-4">
              {summary.sentiment_breakdown.positive} comments
            </div>
            
            {/* Positive Bias Pattern Summary */}
            <div className="mt-2 border-t border-green-100 pt-3">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Bias Pattern Summary:</h4>
              {Object.keys(positiveBiasDistribution.percentages || {}).length > 0 ? (
                <ul className="text-xs space-y-1">
                  {Object.entries(positiveBiasDistribution.percentages || {}).map(([tag, percentage]) => (
                    <li key={tag} className="flex justify-between">
                      <span>{getBiasTagLabel(tag)}:</span>
                      <span className="font-medium">{percentage}%</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-500">No bias pattern data available</p>
              )}
            </div>
          </div>
          <div className="flex flex-col p-4 bg-red-50 rounded-lg">
            <div className="flex items-center justify-center mb-4">
              <FrownIcon size={32} className="text-red-600 mr-2" />
              <div className="text-2xl font-bold text-red-600">
                {negativePercentage}%
              </div>
            </div>
            <div className="text-sm text-gray-600 mt-1 text-center">Negative</div>
            <div className="text-xs text-gray-500 mt-1 text-center mb-4">
              {summary.sentiment_breakdown.negative} comments
            </div>
            
            {/* Negative Bias Pattern Summary */}
            <div className="mt-2 border-t border-red-100 pt-3">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Bias Pattern Summary:</h4>
              {Object.keys(negativeBiasDistribution.percentages || {}).length > 0 ? (
                <ul className="text-xs space-y-1">
                  {Object.entries(negativeBiasDistribution.percentages || {}).map(([tag, percentage]) => (
                    <li key={tag} className="flex justify-between">
                      <span>{getBiasTagLabel(tag)}:</span>
                      <span className="font-medium">{percentage}%</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-xs text-gray-500">No bias pattern data available</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Comments Tabs */}
      <CommentTabs comments={summary.comments || []} />

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