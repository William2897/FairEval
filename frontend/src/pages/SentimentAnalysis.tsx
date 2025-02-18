import { useAuth } from '../contexts/AuthContext';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { WordCloudVisualization } from '../components/WordCloudVisualization';
import { GenderSentimentVisualization } from '../components/GenderSentimentVisualization';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Tab, TabPanel, TabPanels, TabGroup, TabList } from '@headlessui/react';

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
    vader: {
      positive_terms: Array<{
        term: string;
        male_freq: number;
        female_freq: number;
        male_rel_freq: number;
        female_rel_freq: number;
        bias: 'Male' | 'Female';
        total_freq: number;
      }>;
      negative_terms: Array<{
        term: string;
        male_freq: number;
        female_freq: number;
        male_rel_freq: number;
        female_rel_freq: number;
        bias: 'Male' | 'Female';
        total_freq: number;
      }>;
    };
    lexicon: {
      positive_terms: Array<{
        term: string;
        male_freq: number;
        female_freq: number;
        male_rel_freq: number;
        female_rel_freq: number;
        bias: 'Male' | 'Female';
        total_freq: number;
      }>;
      negative_terms: Array<{
        term: string;
        male_freq: number;
        female_freq: number;
        male_rel_freq: number;
        female_rel_freq: number;
        bias: 'Male' | 'Female';
        total_freq: number;
      }>;
    };
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
    enabled: !!user?.username && user?.role?.role === 'ADMIN'
  });

  const { data: genderData } = useQuery<GenderAnalysisData>({
    queryKey: ['gender-analysis'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/institution/sentiment-analysis/');
      return data;
    },
    enabled: !!user?.username && user?.role?.role === 'ADMIN'
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
              <h2 className="text-xl font-bold text-gray-900 mb-4">Comment Analysis</h2>
              <CommentSummaryDisplay professorId={user.username} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-4">Recommendations</h2>
              <RecommendationDisplay professorId={user.username} />
            </div>
          </>
        )}

        {user?.role?.role === 'ADMIN' && wordCloudData && genderData?.gender_analysis && (
          <div className="space-y-8">
            <TabGroup>
              <TabList className="flex space-x-1 rounded-xl bg-indigo-100 p-1 mb-6">
                <Tab
                  className={({ selected }) =>
                    `w-full rounded-lg py-2.5 text-sm font-medium leading-5 ${
                      selected
                        ? 'bg-white text-indigo-700 shadow'
                        : 'text-gray-700 hover:bg-white/[0.12] hover:text-indigo-600'
                    }`
                  }
                >
                  Word Cloud Analysis
                </Tab>
                <Tab
                  className={({ selected }) =>
                    `w-full rounded-lg py-2.5 text-sm font-medium leading-5 ${
                      selected
                        ? 'bg-white text-indigo-700 shadow'
                        : 'text-gray-700 hover:bg-white/[0.12] hover:text-indigo-600'
                    }`
                  }
                >
                  Gender-Based Analysis
                </Tab>
              </TabList>

              <TabPanels>
                <TabPanel>
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

                  <div className="mt-8">
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
                </TabPanel>

                <TabPanel>
                  <div className="space-y-8">
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Gender-Based Sentiment Analysis (Institution-wide)</h2>
                    
                    <div>
                      <h3 className="text-lg font-semibold text-gray-800 mb-4">VADER Analysis</h3>
                      <div className="space-y-8">
                        <GenderSentimentVisualization
                          terms={genderData.gender_analysis.vader.positive_terms}
                          title="Top Biased Positive Terms (VADER)"
                        />
                        <GenderSentimentVisualization
                          terms={genderData.gender_analysis.vader.negative_terms}
                          title="Top Biased Negative Terms (VADER)"
                        />
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold text-gray-800 mb-4">Opinion Lexicon Analysis</h3>
                      <div className="space-y-8">
                        <GenderSentimentVisualization
                          terms={genderData.gender_analysis.lexicon.positive_terms}
                          title="Top Biased Positive Terms (Lexicon)"
                        />
                        <GenderSentimentVisualization
                          terms={genderData.gender_analysis.lexicon.negative_terms}
                          title="Top Biased Negative Terms (Lexicon)"
                        />
                      </div>
                    </div>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-gray-900 mb-4">Topic Analysis</h2>
                    <TopicModelVisualization professorId={user.username} />
                  </div>
                </TabPanel>
              </TabPanels>
            </TabGroup>
          </div>
        )}
      </div>
    </div>
  );
}

export default SentimentAnalysis;