import { useAuth } from '../contexts/AuthContext';
import ProfessorSentimentDashboard from '../components/ProfessorSentimentDashboard';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { WordCloudVisualization } from '../components/WordCloudVisualization';
import { GenderSentimentVisualization } from '../components/GenderSentimentVisualization';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

interface WordCloudData {
  vader: {
    positive: Array<{ word: string; count: number }>;
    negative: Array<{ word: string; count: number }>;
  };
  lexicon: {
    positive: Array<{ word: string; count: number }>;
    negative: Array<{ word: string; count: number }>;
  };
}

interface GenderAnalysisData {
  gender_analysis: {
    positive_terms: Array<{
      term: string;
      male_freq: number;
      female_freq: number;
      bias: 'Male' | 'Female';
    }>;
    negative_terms: Array<{
      term: string;
      male_freq: number;
      female_freq: number;
      bias: 'Male' | 'Female';
    }>;
  };
}

function SentimentAnalysis() {
  const { user } = useAuth();

  const { data: wordCloudData } = useQuery<WordCloudData>({
    queryKey: ['word-clouds'],
    queryFn: async () => {
      const { data } = await axios.get(`/api/professors/institution/word_clouds/`);
      return data;
    },
    enabled: !!user?.username && user?.role?.role === 'ADMIN' // Only fetch for admin users
  });

  const { data: genderData } = useQuery<GenderAnalysisData>({
    queryKey: ['gender-analysis'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/institution/sentiment-analysis/');
      return data;
    },
    enabled: !!user?.username && user?.role?.role === 'ADMIN' // Only fetch for admin users
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
          </>
        )}

        {user?.role?.role === 'ADMIN' && (
          <>
            {genderData?.gender_analysis && (
              <div>
                <h2 className="text-xl font-bold text-gray-900 mb-4">Gender-Based Sentiment Analysis (Institution-wide)</h2>
                <GenderSentimentVisualization
                  positiveTerms={genderData.gender_analysis.positive_terms}
                  negativeTerms={genderData.gender_analysis.negative_terms}
                  title="Gender-Based Term Analysis"
                />
              </div>
            )}

            {wordCloudData && (
              <>
                <div>
                  <h2 className="text-xl font-bold text-gray-900 mb-4">VADER Sentiment Word Clouds (Institution-wide)</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <WordCloudVisualization
                      words={wordCloudData.vader.positive}
                      title="VADER Positive Terms"
                      colorScheme="positive"
                    />
                    <WordCloudVisualization
                      words={wordCloudData.vader.negative}
                      title="VADER Negative Terms"
                      colorScheme="negative"
                    />
                  </div>
                </div>

                <div>
                  <h2 className="text-xl font-bold text-gray-900 mb-4">Opinion Lexicon Word Clouds (Institution-wide)</h2>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <WordCloudVisualization
                      words={wordCloudData.lexicon.positive}
                      title="Lexicon Positive Terms"
                      colorScheme="positive"
                    />
                    <WordCloudVisualization
                      words={wordCloudData.lexicon.negative}
                      title="Lexicon Negative Terms"
                      colorScheme="negative"
                    />
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default SentimentAnalysis;