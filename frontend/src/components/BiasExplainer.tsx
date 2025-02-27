import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2, AlertCircle, Info } from 'lucide-react';

interface DisciplineData {
  discipline_ratings: Array<{
    discipline: string;
  }>;
}

interface AttentionData {
  prediction: string;
  confidence: number;
  tokens: string[];
  attention: number[];
  gender_bias: {
    bias_score: number;
    male_attention_total: number;
    female_attention_total: number;
    neutral_attention_total: number;
    male_attention_pct: number;
    female_attention_pct: number;
    neutral_attention_pct: number;
    male_terms: [number, string, number][];
    female_terms: [number, string, number][];
    top_male_terms: [number, string, number][];
    top_female_terms: [number, string, number][];
    top_neutral_terms: [number, string, number][];
  };
  discipline_context?: {
    discipline: string;
    gender_rating_gap: number;
    male_avg_rating: number;
    female_avg_rating: number;
    correlation: {
      alignment: string;
      explanation: string;
    }
  }
}

interface BiasExplainerProps {
  className?: string;
}

export const BiasExplainer: React.FC<BiasExplainerProps> = ({ 
  className = ''
}) => {
  const [comment, setComment] = useState('');
  const [discipline, setDiscipline] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [disciplines, setDisciplines] = useState<string[]>([]);
  
  // Fetch available disciplines
  useQuery({
    queryKey: ['disciplines'],
    queryFn: async () => {
      try {
        const response = await axios.get<DisciplineData>('/api/professors/discipline_stats/');
        const uniqueDisciplines = [...new Set(
          response.data.discipline_ratings.map(d => d.discipline)
        )];
        setDisciplines(uniqueDisciplines);
        return uniqueDisciplines;
      } catch (error) {
        console.error('Error fetching disciplines:', error);
        return [];
      }
    }
  });
  
  // Get CSRF token from cookies
  const getCsrfToken = () => {
    const name = 'csrftoken=';
    const decodedCookie = decodeURIComponent(document.cookie);
    const cookieArray = decodedCookie.split(';');
    
    for (let i = 0; i < cookieArray.length; i++) {
      let cookie = cookieArray[i].trim();
      if (cookie.indexOf(name) === 0) {
        return cookie.substring(name.length, cookie.length);
      }
    }
    return '';
  };
  
  // Use useMutation with proper types
  const { mutate, data: explanation, isPending, isError, error } = useMutation<AttentionData, Error>({
    mutationFn: async () => {
      // Get CSRF token
      const csrfToken = getCsrfToken();
      
      // Create a properly configured request
      const response = await axios({
        method: 'POST',
        url: '/api/sentiment-explainability/explain_comment/',
        data: { 
          comment, 
          discipline: discipline || undefined 
        },
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': csrfToken,
          'X-Requested-With': 'XMLHttpRequest'
        },
        withCredentials: true
      });
      
      return response.data;
    }
  });

  const analyzeComment = () => {
    if (!comment.trim()) return;
    setIsAnalyzing(true);
    
    mutate(undefined, {
      onSettled: () => {
        setIsAnalyzing(false);
      },
      onError: (error) => {
        console.error("Analysis error:", error);
      }
    });
  };

  const getBiasLabel = (score: number) => {
    if (score > 0.3) return 'Strong male-biased language';
    if (score > 0.1) return 'Moderate male-biased language';
    if (score < -0.3) return 'Strong female-biased language';
    if (score < -0.1) return 'Moderate female-biased language';
    return 'Neutral language';
  };
  
  const getBiasColor = (score: number) => {
    if (score > 0.3) return 'bg-blue-100 text-blue-800';
    if (score > 0.1) return 'bg-blue-50 text-blue-600';
    if (score < -0.3) return 'bg-pink-100 text-pink-800';
    if (score < -0.1) return 'bg-pink-50 text-pink-600';
    return 'bg-gray-100 text-gray-800';
  };
  
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <h2 className="text-xl font-semibold text-gray-900 mb-6">Gender Bias Analysis in Comments</h2>
      
      <div className="space-y-4 mb-6">
        <div>
          <label htmlFor="comment" className="block text-sm font-medium text-gray-700 mb-1">
            Enter a comment to analyze for gender bias:
          </label>
          <textarea 
            id="comment"
            className="w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 resize-none"
            rows={5}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Enter a student comment to analyze for gender bias..."
          />
        </div>
        
        <div>
          <label htmlFor="discipline" className="block text-sm font-medium text-gray-700 mb-1">
            Academic Discipline (optional):
          </label>
          <select
            id="discipline"
            className="w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
            value={discipline}
            onChange={(e) => setDiscipline(e.target.value)}
          >
            <option value="">-- Select a discipline for context --</option>
            {disciplines.map(d => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
          <p className="mt-1 text-xs text-gray-500">
            Selecting a discipline provides context about gender rating patterns
          </p>
        </div>
        
        <div className="flex justify-center">
          <button 
            onClick={analyzeComment}
            disabled={isAnalyzing || !comment.trim() || isPending}
            className="px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {(isAnalyzing || isPending) && <Loader2 size={18} className="animate-spin mr-2" />}
            Analyze Comment
          </button>
        </div>
      </div>
      
      {isPending && (
        <div className="flex justify-center p-8">
          <Loader2 size={32} className="animate-spin text-indigo-600" />
        </div>
      )}
      
      {isError && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg flex items-start">
          <AlertCircle size={20} className="mr-2 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Error analyzing comment</p>
            <p className="text-sm">{(error as Error).message || 'Unknown error occurred'}</p>
          </div>
        </div>
      )}
      
      {explanation && !isPending && (
        <div className="space-y-6">
          {/* Sentiment prediction */}
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              explanation.prediction === 'Positive' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {explanation.prediction} ({(explanation.confidence * 100).toFixed(1)}%)
            </div>
            
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getBiasColor(explanation.gender_bias.bias_score)}`}>
              {getBiasLabel(explanation.gender_bias.bias_score)}
            </div>
          </div>
          
          {/* Attention visualization */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Attention Visualization</h3>
            <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
              <div className="flex flex-wrap gap-2 min-w-[640px]">
                {explanation.tokens.map((token, i: number) => {
                  // Determine if token is gendered
                  const isMaleTerm = explanation.gender_bias.male_terms.some(t => t[0] === i);
                  const isFemaleTerm = explanation.gender_bias.female_terms.some(t => t[0] === i);
                  
                  // Calculate color intensity based on attention weight
                  const attentionWeight = explanation.attention[i];
                  const fontWeight = attentionWeight > 0.05 ? 'font-medium' : 'font-normal';
                  
                  // Determine background color for token
                  let bgColor = `rgba(209, 213, 219, ${attentionWeight * 3})`; // gray
                  let textColor = 'text-gray-900';
                  
                  if (isMaleTerm) {
                    bgColor = `rgba(59, 130, 246, ${attentionWeight * 3})`; // blue
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-blue-800';
                  }
                  if (isFemaleTerm) {
                    bgColor = `rgba(236, 72, 153, ${attentionWeight * 3})`; // pink
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-pink-800';
                  }
                  
                  return (
                    <div key={`${token}-${i}`} className="flex flex-col items-center">
                      <div 
                        className={`px-2 py-1 rounded ${textColor} ${fontWeight}`}
                        style={{ backgroundColor: bgColor }}
                      >
                        {token}
                      </div>
                      <div className="text-xs mt-1 text-gray-500">{attentionWeight.toFixed(2)}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
          
          {/* Gender attention distribution */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">Gender Attention Distribution</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-lg bg-blue-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-blue-800">Male Terms</h4>
                  <span className="text-xl font-bold text-blue-600">{explanation.gender_bias.male_attention_pct.toFixed(1)}%</span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.male_attention_pct}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm">
                  <strong>Top terms:</strong> {
                    explanation.gender_bias.top_male_terms.length > 0 
                      ? explanation.gender_bias.top_male_terms.map((t: [number, string, number]) => t[1]).join(', ') 
                      : 'None'
                  }
                </div>
              </div>
              
              <div className="p-4 rounded-lg bg-pink-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-pink-800">Female Terms</h4>
                  <span className="text-xl font-bold text-pink-600">{explanation.gender_bias.female_attention_pct.toFixed(1)}%</span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-pink-600 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.female_attention_pct}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm">
                  <strong>Top terms:</strong> {
                    explanation.gender_bias.top_female_terms.length > 0 
                      ? explanation.gender_bias.top_female_terms.map((t: [number, string, number]) => t[1]).join(', ') 
                      : 'None'
                  }
                </div>
              </div>
              
              <div className="p-4 rounded-lg bg-gray-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-gray-800">Neutral Terms</h4>
                  <span className="text-xl font-bold text-gray-600">{explanation.gender_bias.neutral_attention_pct.toFixed(1)}%</span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-gray-500 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.neutral_attention_pct}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm text-gray-600">
                  <strong>Top terms:</strong> {
                    explanation.gender_bias.top_neutral_terms.length > 0 
                      ? explanation.gender_bias.top_neutral_terms.slice(0, 3).map(t => t[1]).join(', ')
                      : 'None'
                  }
                </div>
              </div>
            </div>
          </div>
          
          {/* Discipline context if available */}
          {explanation.discipline_context && (
            <div className="mt-6 border-t border-gray-200 pt-4">
              <div className="flex items-start space-x-2">
                <Info size={20} className="flex-shrink-0 mt-1 text-indigo-500" />
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Disciplinary Context</h3>
                  <p className="mb-2">
                    In <strong>{explanation.discipline_context.discipline}</strong>, there is a 
                    <strong className={explanation.discipline_context.gender_rating_gap > 0 
                      ? ' text-blue-600' 
                      : ' text-pink-600'
                    }>
                      {' '}{Math.abs(explanation.discipline_context.gender_rating_gap).toFixed(2)} rating gap 
                    </strong> favoring {explanation.discipline_context.gender_rating_gap > 0 ? 'male' : 'female'} professors 
                    (Male: {explanation.discipline_context.male_avg_rating.toFixed(2)}, 
                    Female: {explanation.discipline_context.female_avg_rating.toFixed(2)})
                  </p>
                  <div className={`p-3 rounded-lg ${
                    explanation.discipline_context.correlation.alignment === 'aligned' ? 'bg-yellow-50' : 'bg-blue-50'
                  }`}>
                    <p className={
                      explanation.discipline_context.correlation.alignment === 'aligned' ? 'text-yellow-800' : 'text-blue-800'
                    }>
                      <strong>Finding:</strong> {explanation.discipline_context.correlation.explanation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {/* Bias implications */}
          <div className="mt-6 border-t border-gray-200 pt-4">
            <h3 className="text-lg font-medium text-gray-900 mb-3">Implications & Recommendations</h3>
            
            {Math.abs(explanation.gender_bias.bias_score) > 0.2 ? (
              <div className="space-y-3">
                <div className={`p-4 rounded-lg ${
                  explanation.gender_bias.bias_score > 0 ? 'bg-blue-50' : 'bg-pink-50'
                }`}>
                  <p className={
                    explanation.gender_bias.bias_score > 0 ? 'text-blue-800' : 'text-pink-800'
                  }>
                    <strong>Detected Bias:</strong> This comment shows significant 
                    {explanation.gender_bias.bias_score > 0 ? ' male' : ' female'}-oriented language patterns 
                    in sentiment evaluation.
                  </p>
                </div>
                
                <div className="bg-indigo-50 p-4 rounded-lg">
                  <p className="text-indigo-800 font-medium mb-2">Recommendations:</p>
                  <ul className="list-disc pl-5 space-y-1 text-indigo-800">
                    <li>Consider if these language patterns reflect genuine teaching qualities or potential bias</li>
                    <li>Be aware that attention focuses on gendered terms may influence sentiment assessments</li>
                    {explanation.discipline_context && (
                      <li>
                        {explanation.discipline_context.correlation.alignment === 'aligned' 
                          ? 'This comment reflects broader gender-based evaluation patterns in your discipline' 
                          : 'This comment shows language patterns that run counter to typical evaluation patterns in your discipline'
                        }
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="bg-green-50 p-4 rounded-lg">
                <p className="text-green-800">
                  <strong>Balanced Analysis:</strong> This comment shows relatively balanced language 
                  without significant gender-based attention patterns.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};