import { useAuth } from '../contexts/AuthContext';
import ProfessorSentimentDashboard from '../components/ProfessorSentimentDashboard';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { WordCloudVisualization } from '../components/WordCloudVisualization';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface WordData {
  positive: Array<{ word: string; count: number }>;
  negative: Array<{ word: string; count: number }>;
}

function SentimentAnalysis() {
  const { user } = useAuth();

  const { data: wordData } = useQuery<WordData>({
    queryKey: ['word-frequencies', user?.username],
    queryFn: async () => {
      const { data } = await axios.get(`/api/professors/${user?.username}/word-frequencies/`);
      return data;
    },
    enabled: !!user?.username
  });

  if (!user) {
    return null;
  }

  return (
    <div className="space-y-8 p-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Sentiment Analysis</h1>
      </div>

      <div className="space-y-8">
        {user?.role?.role === 'ACADEMIC' && (
          <>
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">Overall Sentiment Analysis</h2>
              <ProfessorSentimentDashboard professorId={user.username} />
            </div>
            
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">Comment Analysis</h2>
              <CommentSummaryDisplay professorId={user.username} />
            </div>

            {wordData && (
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Term Frequency Analysis</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <WordCloudVisualization
                    words={wordData.positive}
                    title="Positive Terms"
                    colorScheme="positive"
                  />
                  <WordCloudVisualization
                    words={wordData.negative}
                    title="Negative Terms"
                    colorScheme="negative"
                  />
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default SentimentAnalysis;