import { useQuery } from '@tanstack/react-query';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Loader2 } from 'lucide-react';

interface SentimentData {
  sentiment_counts: {
    positive: number;
    negative: number;
  };
  vader_scores: {
    compound: number;
    positive: number;
    negative: number;
  };
  top_words: {
    positive: Array<{ word: string; count: number }>;
    negative: Array<{ word: string; count: number }>;
  };
  recent_sentiments: Array<{
    comment: string;
    processed_comment: string;
    sentiment: number;
    created_at: string;
  }>;
}

interface Props {
  professorId: string;
}

function WordGrid({ words }: { words: Array<{ word: string; count: number }> }) {
  return (
    <div className="grid grid-cols-3 gap-2">
      {words.map(({ word, count }) => (
        <div
          key={word}
          className="p-2 rounded-lg text-center"
          style={{
            fontSize: `${Math.max(0.8 + (count / 10) * 0.4, 1)}rem`,
            backgroundColor: `rgba(79, 70, 229, ${Math.min(count / 20, 0.2)})`,
          }}
        >
          {word}
        </div>
      ))}
    </div>
  );
}

function ProfessorSentimentDashboard({ professorId }: Props) {
  const { data, isLoading, error } = useQuery<SentimentData>({
    queryKey: ['professor-sentiment', professorId],
    queryFn: async () => {
      const response = await fetch(`/api/professors/${professorId}/sentiment-analysis/`);
      if (!response.ok) {
        throw new Error('Failed to fetch sentiment data');
      }
      return response.json();
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 text-red-600 p-4 rounded-lg">
        Failed to load sentiment analysis data
      </div>
    );
  }

  const sentimentData = [
    { name: 'Positive', count: data.sentiment_counts.positive },
    { name: 'Negative', count: data.sentiment_counts.negative },
  ];

  const vaderData = [
    { name: 'Compound', score: data.vader_scores.compound },
    { name: 'Positive', score: data.vader_scores.positive },
    { name: 'Negative', score: data.vader_scores.negative },
  ];

  return (
    <div className="space-y-8">
      {/* Sentiment Distribution */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Comment Sentiment Distribution
        </h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sentimentData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar
                dataKey="count"
                fill="#4f46e5"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* VADER Scores */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          VADER Sentiment Scores
        </h2>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={vaderData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[-1, 1]} />
              <Tooltip />
              <Bar
                dataKey="score"
                fill="#4f46e5"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Sentiments */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Recent Sentiment Analysis
        </h2>
        <div className="space-y-4">
          {data?.recent_sentiments.map((sentiment, index) => (
            <div 
              key={index}
              className={`p-4 rounded-lg ${
                sentiment.sentiment > 0 
                  ? 'bg-green-50' 
                  : sentiment.sentiment < 0 
                  ? 'bg-red-50' 
                  : 'bg-gray-50'
              }`}
            >
              <p className="text-gray-700">{sentiment.comment}</p>
              <div className="mt-2 flex justify-between items-center text-sm">
                <span className="text-gray-500">
                  {new Date(sentiment.created_at).toLocaleDateString()}
                </span>
                <span className={`font-medium ${
                  sentiment.sentiment > 0 
                    ? 'text-green-600' 
                    : sentiment.sentiment < 0 
                    ? 'text-red-600' 
                    : 'text-gray-600'
                }`}>
                  Sentiment: {sentiment.sentiment}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Word Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Positive Words */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Top Positive Words
          </h2>
          <WordGrid words={data.top_words.positive} />
        </div>

        {/* Negative Words */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Top Negative Words
          </h2>
          <WordGrid words={data.top_words.negative} />
        </div>
      </div>
    </div>
  );
}

export default ProfessorSentimentDashboard;