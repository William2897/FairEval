import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2, AlertCircle, Info, Lightbulb, HelpCircle } from 'lucide-react';
import { BiasExplainer } from './BiasExplainer';

interface BiasAnalysisResult {
  professor_id: string;
  discipline: string;
  analysis_results: {
    overall_bias_score: number;
    positive_comments_bias_score: number;
    negative_comments_bias_score: number;
    comment_count: number;
    positive_count: number;
    negative_count: number;
    top_male_terms: [string, number][];
    top_female_terms: [string, number][];
  };
  recommendations: {
    text: string;
    priority: 'high' | 'medium' | 'low';
    impact_score: number;
    supporting_evidence?: any;
  }[];
}

interface DisciplineData {
  discipline: string;
  gender: string;
  avg_rating: number;
}

interface ProfessorBiasDashboardProps {
  professorId: string;
  className?: string;
}

export const ProfessorBiasDashboard: React.FC<ProfessorBiasDashboardProps> = ({ 
  professorId,
  className = ''
}) => {
  const [showExplainer, setShowExplainer] = useState(false);

  // Tooltip component for consistent styling
  const Tooltip = ({ children, text }: { children: React.ReactNode, text: string }) => {
    return (
      <div className="relative group inline-block">
        {children}
        <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 text-sm font-medium text-white bg-gray-900 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 w-64 shadow-lg pointer-events-none">
          {text}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 border-t-4 border-l-4 border-r-4 border-gray-900 border-r-transparent border-l-transparent"></div>
        </div>
      </div>
    );
  };

  // Fetch bias analysis data from our new endpoint
  const { data: biasAnalysis, isLoading, isError, error } = useQuery<BiasAnalysisResult>({
    queryKey: ['professorBiasAnalysis', professorId],
    queryFn: async () => {
      const { data } = await axios.get(`/api/sentiment-explainability/${professorId}/professor_bias_analysis/`);
      return data;
    },
    enabled: !!professorId,
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

  // Fetch discipline-specific gender data for context
  const { data: disciplineData } = useQuery<DisciplineData[]>({
    queryKey: ['gender-discipline-data', biasAnalysis?.discipline],
    queryFn: async () => {
      if (!biasAnalysis?.discipline) return null;
      const { data } = await axios.get('/api/professors/gender_discipline_heatmap/');
      return data.filter((item: DisciplineData) => item.discipline === biasAnalysis.discipline);
    },
    enabled: !!biasAnalysis?.discipline,
  });

  const disciplineContext = useMemo(() => {
    if (!disciplineData) return null;
    
    const maleData = disciplineData.find((d: DisciplineData) => d.gender === 'Male');
    const femaleData = disciplineData.find((d: DisciplineData) => d.gender === 'Female');
    
    if (!maleData || !femaleData) return null;
    
    return {
      male_avg: maleData.avg_rating,
      female_avg: femaleData.avg_rating,
      gap: maleData.avg_rating - femaleData.avg_rating
    };
  }, [disciplineData]);

  const getBiasLevelClass = (score: number) => {
    const absScore = Math.abs(score);
    if (absScore < 0.1) return 'bg-green-50 text-green-800';
    if (absScore < 0.3) return score > 0 ? 'bg-blue-50 text-blue-800' : 'bg-pink-50 text-pink-800';
    return score > 0 ? 'bg-blue-100 text-blue-900' : 'bg-pink-100 text-pink-900';
  };

  const getBiasLabel = (score: number) => {
    const absScore = Math.abs(score);
    if (absScore < 0.1) return 'Balanced language usage';
    if (absScore < 0.3) {
      return score > 0 
        ? 'Slight male-associated language bias' 
        : 'Slight female-associated language bias';
    }
    return score > 0 
      ? 'Significant male-associated language bias' 
      : 'Significant female-associated language bias';
  };
  
  if (isError) {
    return (
      <div className={`bg-red-50 text-red-700 p-4 rounded-lg flex items-start ${className}`}>
        <AlertCircle size={20} className="mr-2 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-medium">Error analyzing comments</p>
          <p className="text-sm">{(error as Error).message || 'Unknown error occurred'}</p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className={`flex justify-center p-8 ${className}`} role="status">
        <Loader2 size={32} className="animate-spin text-indigo-600" />
      </div>
    );
  }

  if (!biasAnalysis?.analysis_results) {
    return (
      <div className={`bg-yellow-50 text-yellow-700 p-4 rounded-lg ${className}`}>
        No bias analysis data available. This could be due to insufficient comments for analysis.
      </div>
    );
  }

  const analysisResults = biasAnalysis.analysis_results;

  return (
    <div className={`space-y-8 ${className}`}>
      {/* Overall Bias Score Card */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Gender Bias Analysis Summary
          <Tooltip text="This analysis uses our LSTM model to identify potential gender bias patterns in student comments about your teaching.">
            <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
          </Tooltip>
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <div className={`rounded-lg p-4 ${getBiasLevelClass(analysisResults.overall_bias_score)}`}>
            <div className="font-semibold mb-1">
              Overall Bias Score
              <Tooltip text="Measures how language patterns in comments may reflect gendered expectations. Positive values suggest male-associated language patterns, negative suggest female-associated patterns.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
            <div className="text-2xl font-bold">
              {analysisResults.overall_bias_score.toFixed(2)}
            </div>
            <div className="text-sm mt-1">{getBiasLabel(analysisResults.overall_bias_score)}</div>
          </div>
          
          <div className={`rounded-lg p-4 ${getBiasLevelClass(analysisResults.positive_comments_bias_score)}`}>
            <div className="font-semibold mb-1">
              Positive Comments Bias
              <Tooltip text="Shows gender bias specifically in positive evaluations. Research shows different language is often used when praising male vs. female professors.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
            <div className="text-2xl font-bold">
              {analysisResults.positive_comments_bias_score.toFixed(2)}
            </div>
            <div className="text-sm mt-1">{getBiasLabel(analysisResults.positive_comments_bias_score)}</div>
          </div>
          
          <div className={`rounded-lg p-4 ${getBiasLevelClass(analysisResults.negative_comments_bias_score)}`}>
            <div className="font-semibold mb-1">
              Negative Comments Bias
              <Tooltip text="Shows gender bias specifically in critical evaluations. Different standards are often applied when criticizing professors of different genders.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
            <div className="text-2xl font-bold">
              {analysisResults.negative_comments_bias_score.toFixed(2)}
            </div>
            <div className="text-sm mt-1">{getBiasLabel(analysisResults.negative_comments_bias_score)}</div>
          </div>
        </div>
        
        {disciplineContext && (
          <div className="p-4 bg-indigo-50 rounded-lg mb-4">
            <div className="flex items-start">
              <Info size={20} className="mr-2 text-indigo-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-indigo-700">
                  <span className="font-medium">
                    Discipline Context
                    <Tooltip text="This provides context about how your discipline compares to the institutional average. Some disciplines show larger gender disparities in ratings.">
                      <HelpCircle className="inline-block ml-1 cursor-help text-indigo-600 opacity-70" size={14} />
                    </Tooltip>:
                  </span> In {biasAnalysis.discipline}, there is a
                  <strong className={disciplineContext.gap > 0 ? ' text-blue-700' : ' text-pink-700'}>
                    {' '}{Math.abs(disciplineContext.gap).toFixed(2)} rating gap 
                  </strong> favoring {disciplineContext.gap > 0 ? 'male' : 'female'} professors.
                </p>
                <p className="text-sm text-indigo-600 mt-1">
                  Male avg: {disciplineContext.male_avg.toFixed(2)}, Female avg: {disciplineContext.female_avg.toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        )}
        
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="flex items-start">
            <Info size={20} className="mr-2 text-gray-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-gray-600">
                Analysis based on {biasAnalysis.analysis_results.comment_count} comments 
                ({biasAnalysis.analysis_results.positive_count} positive, {biasAnalysis.analysis_results.negative_count} negative) 
                in the {biasAnalysis.discipline} discipline.
              </p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Gendered Terms Analysis */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Gendered Language Analysis
          <Tooltip text="These are terms that the LSTM model identified as having significant gender associations in your student evaluations.">
            <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
          </Tooltip>
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border rounded-lg p-4">
            <h3 className="text-lg font-medium text-blue-800 mb-3">
              Male-Associated Terms
              <Tooltip text="Terms that are more commonly associated with male professors or receive higher attention when evaluating men. These often focus on personality, humor, or entertainment value.">
                <HelpCircle className="inline-block ml-1 cursor-help text-blue-600 opacity-70" size={14} />
              </Tooltip>
            </h3>
            <div className="flex flex-wrap gap-2">
              {biasAnalysis.analysis_results.top_male_terms.map(([term], index) => (
                <div 
                  key={`male-${term}`} 
                  className="px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-sm"
                  style={{ opacity: 1 - (index * 0.05) }}
                >
                  {term}
                </div>
              ))}
              {biasAnalysis.analysis_results.top_male_terms.length === 0 && (
                <p className="text-gray-500 italic">No significant male-associated terms identified</p>
              )}
            </div>
          </div>
          
          <div className="border rounded-lg p-4">
            <h3 className="text-lg font-medium text-pink-800 mb-3">
              Female-Associated Terms
              <Tooltip text="Terms that are more commonly associated with female professors or receive higher attention when evaluating women. These often focus on competence, organization, and teaching skills.">
                <HelpCircle className="inline-block ml-1 cursor-help text-pink-600 opacity-70" size={14} />
              </Tooltip>
            </h3>
            <div className="flex flex-wrap gap-2">
              {biasAnalysis.analysis_results.top_female_terms.map(([term], index) => (
                <div 
                  key={`female-${term}`} 
                  className="px-3 py-1 bg-pink-50 text-pink-700 rounded-full text-sm"
                  style={{ opacity: 1 - (index * 0.05) }}
                >
                  {term}
                </div>
              ))}
              {biasAnalysis.analysis_results.top_female_terms.length === 0 && (
                <p className="text-gray-500 italic">No significant female-associated terms identified</p>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Recommendations */}
      {biasAnalysis.recommendations.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            AI-Generated Recommendations
            <Tooltip text="These recommendations are generated based on the bias patterns detected in your student evaluations and research on gender bias in higher education.">
              <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
            </Tooltip>
          </h2>
          
          <div className="space-y-4">
            {biasAnalysis.recommendations.map((rec, index) => (
              <div 
                key={`rec-${index}`} 
                className={`p-4 rounded-lg flex ${
                  rec.priority === 'high' ? 'bg-amber-50' :
                  rec.priority === 'medium' ? 'bg-blue-50' : 'bg-gray-50'
                }`}
              >
                <Lightbulb 
                  size={24} 
                  className={`mr-4 flex-shrink-0 ${
                    rec.priority === 'high' ? 'text-amber-500' :
                    rec.priority === 'medium' ? 'text-blue-500' : 'text-gray-500'
                  }`} 
                />
                <div>
                  <p className={`font-medium ${
                    rec.priority === 'high' ? 'text-amber-800' :
                    rec.priority === 'medium' ? 'text-blue-800' : 'text-gray-800'
                  }`}>
                    {rec.text}
                  </p>
                  {rec.supporting_evidence && rec.supporting_evidence.length > 0 && (
                    <div className="mt-2 text-sm text-gray-600">
                      Supporting evidence: {rec.supporting_evidence.map((item: string | [string, any]) => 
                        typeof item === 'string' ? item : item[0]
                      ).join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Interactive Explainer Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Interactive Comment Analysis
          <Tooltip text="This tool allows you to analyze individual comments to understand how our AI model detects gender bias in language patterns.">
            <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
          </Tooltip>
        </h2>
        
        {showExplainer ? (
          <div>
            <button 
              onClick={() => setShowExplainer(false)}
              className="mb-4 px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
            >
              Hide Explainer
            </button>
            <BiasExplainer />
          </div>
        ) : (
          <div className="text-center py-8">
            <p className="text-gray-600 mb-4">
              Analyze specific comments to understand how the AI model identifies gender bias
              using attention mechanisms.
            </p>
            <button 
              onClick={() => setShowExplainer(true)}
              className="px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700"
            >
              Open Interactive Explainer
            </button>
          </div>
        )}
      </div>
    </div>
  );
};