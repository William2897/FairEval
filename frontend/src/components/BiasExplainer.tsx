import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2, AlertCircle, Info, HelpCircle } from 'lucide-react';

interface DisciplineData {
  discipline_ratings: Array<{
    discipline: string;
  }>;
}

// Updated interface to match our new structure
interface AttentionData {
  prediction: string;
  confidence: number;
  tokens: string[];
  attention: number[];
  gender_bias: {
    bias_score: number;
    descriptor_bias_score: number;
    explicit_gender_bias: number;
    category_attention: {
      personality_entertainment: number;
      competence: number;
      explicit_male: number;
      explicit_female: number;
      other: number;
    };
    category_attention_pct: {
      personality_entertainment: number;
      competence: number;
      explicit_male: number;
      explicit_female: number;
      other: number;
    };
    descriptor_categories: {
      personality_entertainment: [number, string, number][];
      competence: [number, string, number][];
      explicit_male: [number, string, number][];
      explicit_female: [number, string, number][];
      other: [number, string, number][];
    };
    top_terms_by_category: {
      personality_entertainment: [number, string, number][];
      competence: [number, string, number][];
      explicit_male: [number, string, number][];
      explicit_female: [number, string, number][];
      other: [number, string, number][];
    };
    interpretation: string[];
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

  // Function to get descriptor bias label
  const getDescriptorBiasLabel = (score: number) => {
    if (score > 0.3) return 'Strong focus on personality/entertainment';
    if (score > 0.1) return 'Moderate focus on personality/entertainment';
    if (score < -0.3) return 'Strong focus on competence/qualifications';
    if (score < -0.1) return 'Moderate focus on competence/qualifications';
    return 'Balanced descriptors';
  };
  
  // Function to get descriptor bias color
  const getDescriptorBiasColor = (score: number) => {
    if (score > 0.3) return 'bg-purple-100 text-purple-800';
    if (score > 0.1) return 'bg-purple-50 text-purple-600';
    if (score < -0.3) return 'bg-orange-100 text-orange-800';
    if (score < -0.1) return 'bg-orange-50 text-orange-600';
    return 'bg-gray-100 text-gray-800';
  };

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
  
  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      <h2 className="text-xl font-semibold text-gray-900 mb-6">
        Gender Bias Analysis in Comments
        <Tooltip text="This tool analyzes how language used in student evaluations might reflect gender bias through our LSTM model's attention patterns.">
          <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
        </Tooltip>
      </h2>
      
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
          <div className="flex items-center space-x-4 flex-wrap gap-2">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              explanation.prediction === 'Positive' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {explanation.prediction} ({(explanation.confidence * 100).toFixed(1)}%)
              <Tooltip text="The overall sentiment prediction of our LSTM model, based on the text content.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
            
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getBiasColor(explanation.gender_bias.bias_score)}`}>
              {getBiasLabel(explanation.gender_bias.bias_score)}
              <Tooltip text="Overall gender bias score combines both descriptor patterns and explicit gendered terms. Positive values indicate male-associated patterns; negative values indicate female-associated patterns.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>

            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getDescriptorBiasColor(explanation.gender_bias.descriptor_bias_score)}`}>
              {getDescriptorBiasLabel(explanation.gender_bias.descriptor_bias_score)}
              <Tooltip text="Research shows evaluations of male professors often focus on personality/entertainment, while female professors are evaluated more on competence/qualifications.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
          </div>
          
          {/* Attention visualization */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              Attention Visualization
              <Tooltip text="This shows which words the model pays most attention to when making its prediction. Words are color-coded by category and sized by attention weight.">
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>
            <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
              <div className="flex flex-wrap gap-2 min-w-[640px]">
                {explanation.tokens.map((token, i: number) => {
                  // Determine token category from descriptor_categories
                  const isPersonalityTerm = explanation.gender_bias.descriptor_categories.personality_entertainment.some(t => t[0] === i);
                  const isCompetenceTerm = explanation.gender_bias.descriptor_categories.competence.some(t => t[0] === i);
                  const isMaleTerm = explanation.gender_bias.descriptor_categories.explicit_male.some(t => t[0] === i);
                  const isFemaleTerm = explanation.gender_bias.descriptor_categories.explicit_female.some(t => t[0] === i);
                  
                  // Calculate color intensity based on attention weight
                  const attentionWeight = explanation.attention[i];
                  const fontWeight = attentionWeight > 0.05 ? 'font-medium' : 'font-normal';
                  
                  // Determine background color for token
                  let bgColor = `rgba(209, 213, 219, ${attentionWeight * 3})`; // gray
                  let textColor = 'text-gray-900';
                  
                  if (isPersonalityTerm) {
                    bgColor = `rgba(147, 51, 234, ${attentionWeight * 3})`; // purple
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-purple-800';
                  }
                  if (isCompetenceTerm) {
                    bgColor = `rgba(249, 115, 22, ${attentionWeight * 3})`; // orange
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-orange-800';
                  }
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
          
          {/* Category attention distribution */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              Descriptor Categories Distribution
              <Tooltip text="This breaks down the attention paid to different types of descriptors used in the evaluation. Research shows gendered patterns in which aspects of teaching are emphasized.">
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Personality/Entertainment Terms */}
              <div className="p-4 rounded-lg bg-purple-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-purple-800">
                    Personality/Entertainment
                    <Tooltip text="Terms describing personality traits, humor, entertainment value, and rapport. These are typically more emphasized in male professor evaluations.">
                      <HelpCircle className="inline-block ml-1 cursor-help text-purple-600 opacity-70" size={14} />
                    </Tooltip>
                  </h4>
                  <span className="text-xl font-bold text-purple-600">
                    {explanation.gender_bias.category_attention_pct.personality_entertainment.toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-purple-600 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.category_attention_pct.personality_entertainment}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm">
                  <strong>Top terms:</strong> {
                    explanation.gender_bias.top_terms_by_category.personality_entertainment?.length > 0 
                      ? explanation.gender_bias.top_terms_by_category.personality_entertainment.map((t) => t[1]).join(', ') 
                      : 'None'
                  }
                </div>
              </div>
              
              {/* Competence Terms */}
              <div className="p-4 rounded-lg bg-orange-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-orange-800">
                    Competence
                    <Tooltip text="Terms describing professional competence, expertise, organization, and teaching skill. These are typically more emphasized in female professor evaluations.">
                      <HelpCircle className="inline-block ml-1 cursor-help text-orange-600 opacity-70" size={14} />
                    </Tooltip>
                  </h4>
                  <span className="text-xl font-bold text-orange-600">
                    {explanation.gender_bias.category_attention_pct.competence.toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-orange-600 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.category_attention_pct.competence}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm">
                  <strong>Top terms:</strong> {
                    explanation.gender_bias.top_terms_by_category.competence?.length > 0 
                      ? explanation.gender_bias.top_terms_by_category.competence.map((t) => t[1]).join(', ') 
                      : 'None'
                  }
                </div>
              </div>
              
              {/* Explicit Gendered Terms */}
              <div className="p-4 rounded-lg bg-gray-50">
                <div className="flex justify-between items-center">
                  <h4 className="font-medium text-gray-800">
                    Explicit Gender Terms
                    <Tooltip text="Explicitly gendered words like pronouns (he/she) or gendered nouns. High attention on these terms can indicate gender may be influencing the evaluation.">
                      <HelpCircle className="inline-block ml-1 cursor-help text-gray-600 opacity-70" size={14} />
                    </Tooltip>
                  </h4>
                  <span className="text-xl font-bold text-gray-600">
                    {(explanation.gender_bias.category_attention_pct.explicit_male + 
                      explanation.gender_bias.category_attention_pct.explicit_female).toFixed(1)}%
                  </span>
                </div>
                <div className="mt-2">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 h-2.5 rounded-full" 
                      style={{ width: `${explanation.gender_bias.category_attention_pct.explicit_male}%` }}
                    />
                    <div 
                      className="bg-pink-600 h-2.5 rounded-full mt-1" 
                      style={{ width: `${explanation.gender_bias.category_attention_pct.explicit_female}%` }}
                    />
                  </div>
                </div>
                <div className="mt-3 text-sm text-gray-600">
                  <div><strong className="text-blue-800">Male:</strong> {
                    explanation.gender_bias.top_terms_by_category.explicit_male?.length > 0 
                      ? explanation.gender_bias.top_terms_by_category.explicit_male.map((t) => t[1]).join(', ')
                      : 'None'
                  }</div>
                  <div><strong className="text-pink-800">Female:</strong> {
                    explanation.gender_bias.top_terms_by_category.explicit_female?.length > 0 
                      ? explanation.gender_bias.top_terms_by_category.explicit_female.map((t) => t[1]).join(', ')
                      : 'None'
                  }</div>
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
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Disciplinary Context
                    <Tooltip text="This shows how your discipline compares to institutional averages in terms of gender rating gaps, and whether the language in this comment aligns with those patterns.">
                      <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
                    </Tooltip>
                  </h3>
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
          
          {/* AI Interpretation */}
          <div className="mt-6 border-t border-gray-200 pt-4">
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              AI Interpretation
              <Tooltip text="The model analyzes the overall patterns and provides interpretations of what the language patterns might indicate about potential bias.">
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>
            
            {explanation.gender_bias.interpretation && explanation.gender_bias.interpretation.length > 0 ? (
              <div className="space-y-3">
                {explanation.gender_bias.interpretation.map((interpretation, index) => (
                  <div key={index} className="bg-indigo-50 p-4 rounded-lg">
                    <p className="text-indigo-800">{interpretation}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-800">
                  No specific pattern interpretation available.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};