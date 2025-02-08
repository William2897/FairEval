import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Loader2, ChevronDown, ChevronUp } from 'lucide-react';

interface TopicKeyword {
  word: string;
  weight: number;
}

interface Topic {
  id: number;
  name: string;
  description: string;
  distribution: number;
  keywords: TopicKeyword[];
  sentiment_score: number;
}

interface TopicModelData {
  topics: Topic[];
  total_documents: number;
  time_period: string;
}

interface TopicModelVisualizationProps {
  professorId?: string;
  className?: string;
}

const TopicKeywordList: React.FC<{
  keywords: TopicKeyword[];
  className?: string;
}> = ({ keywords, className = '' }) => (
  <div className={`flex flex-wrap gap-2 ${className}`}>
    {keywords.map((keyword, index) => (
      <span
        key={keyword.word}
        className={`px-2 py-1 rounded-full text-sm
          ${index < 3 ? 'bg-indigo-100 text-indigo-800' : 'bg-gray-100 text-gray-800'}`}
      >
        {keyword.word}
        <span className="text-xs ml-1 opacity-60">
          ({keyword.weight.toFixed(2)})
        </span>
      </span>
    ))}
  </div>
);

const TopicCard: React.FC<{
  topic: Topic;
  maxDistribution: number;
}> = ({ topic, maxDistribution }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const distributionPercentage = (topic.distribution / maxDistribution) * 100;

  return (
    <div className="bg-white rounded-lg shadow p-4 hover:shadow-md transition-shadow">
      <div 
        className="cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between mb-2">
          <h4 className="text-lg font-semibold text-gray-900">
            {topic.name}
          </h4>
          {isExpanded ? (
            <ChevronUp size={20} className="text-gray-500" />
          ) : (
            <ChevronDown size={20} className="text-gray-500" />
          )}
        </div>
        
        <div className="mb-3">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">Distribution</span>
            <span className="font-medium">{(topic.distribution * 100).toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-indigo-600 rounded-full h-2"
              style={{ width: `${distributionPercentage}%` }}
            />
          </div>
        </div>

        <div className="flex items-center gap-2 mb-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium
            ${topic.sentiment_score > 0 ? 'bg-green-100 text-green-800' : 
              topic.sentiment_score < 0 ? 'bg-red-100 text-red-800' : 
              'bg-gray-100 text-gray-800'}`}>
            Sentiment: {topic.sentiment_score.toFixed(2)}
          </span>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-3">
          <p className="text-gray-600 text-sm">
            {topic.description}
          </p>
          <div>
            <h5 className="text-sm font-medium text-gray-700 mb-2">
              Top Keywords
            </h5>
            <TopicKeywordList keywords={topic.keywords} />
          </div>
        </div>
      )}
    </div>
  );
};

export const TopicModelVisualization: React.FC<TopicModelVisualizationProps> = ({ 
  professorId,
  className = ''
}) => {
  const [view, setView] = useState<'chart' | 'cards'>('cards');
  
  const { data, isLoading, isError, error } = useQuery<TopicModelData>({
    queryKey: ['topics', professorId],
    queryFn: async () => {
      const url = professorId 
        ? `/api/professors/${professorId}/topics/`
        : '/api/topics/';
      const { data } = await axios.get(url);
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
        Error loading topic data: {(error as Error).message}
      </div>
    );
  }

  const maxDistribution = Math.max(...data.topics.map(t => t.distribution));

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-xl font-semibold text-gray-900">
            Topic Analysis
          </h2>
          <p className="text-sm text-gray-500 mt-1">
            Based on {data.total_documents} reviews from {data.time_period}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setView('chart')}
            className={`px-3 py-1 rounded-md text-sm ${
              view === 'chart'
                ? 'bg-indigo-100 text-indigo-700'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Chart View
          </button>
          <button
            onClick={() => setView('cards')}
            className={`px-3 py-1 rounded-md text-sm ${
              view === 'cards'
                ? 'bg-indigo-100 text-indigo-700'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            Card View
          </button>
        </div>
      </div>

      {view === 'chart' ? (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={data.topics}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis
                  tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                />
                <Tooltip
                  formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, 'Distribution']}
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null;
                    const topic = payload[0].payload as Topic;
                    return (
                      <div className="bg-white shadow-lg rounded-lg p-3 border">
                        <p className="font-semibold">{topic.name}</p>
                        <p className="text-sm text-gray-600 mt-1">
                          Distribution: {(topic.distribution * 100).toFixed(1)}%
                        </p>
                        <div className="mt-2">
                          <p className="text-xs font-medium text-gray-500">Top Keywords:</p>
                          <p className="text-sm">
                            {topic.keywords.slice(0, 3).map(k => k.word).join(', ')}
                          </p>
                        </div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="distribution" fill="#4f46e5">
                  {data.topics.map((topic, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={topic.sentiment_score > 0 ? '#4f46e5' : '#6366f1'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : (
        <div className="grid gap-6 md:grid-cols-2">
          {data.topics.map((topic) => (
            <TopicCard
              key={topic.id}
              topic={topic}
              maxDistribution={maxDistribution}
            />
          ))}
        </div>
      )}
    </div>
  );
}; 