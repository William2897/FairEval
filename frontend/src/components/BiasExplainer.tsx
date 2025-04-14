import React, { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2, AlertCircle, Info, HelpCircle, CheckCircle } from 'lucide-react';

interface DisciplineData {
  discipline_ratings: Array<{
    discipline: string;
  }>;
}

// Updated interface reflecting backend changes
interface AttentionData {
  prediction: string;
  confidence: number;
  tokens: string[];
  attention: number[];
  gender_bias: {
    stereotype_bias_score: number;
    original_descriptor_bias_score: number;
    category_attention: {
      intellect_achievement: number;
      entertainment_authority: number;
      competence_organization: number;
      warmth_nurturing: number;
      male_negative: number;
      female_negative: number;
      objective_pedagogical: number; // ADDED
      other: number;
    };
    category_attention_pct: {
      intellect_achievement: number;
      entertainment_authority: number;
      competence_organization: number;
      warmth_nurturing: number;
      male_negative: number;
      female_negative: number;
      objective_pedagogical: number; // ADDED
      other: number;
    };
    descriptor_categories: {
      intellect_achievement: [number, string, number][];
      entertainment_authority: [number, string, number][];
      competence_organization: [number, string, number][];
      warmth_nurturing: [number, string, number][];
      male_negative: [number, string, number][];
      female_negative: [number, string, number][];
      objective_pedagogical: [number, string, number][]; // ADDED
      other: [number, string, number][];
    };
    top_terms_by_category: {
      intellect_achievement: [number, string, number][];
      entertainment_authority: [number, string, number][];
      competence_organization: [number, string, number][];
      warmth_nurturing: [number, string, number][];
      male_negative: [number, string, number][];
      female_negative: [number, string, number][];
      objective_pedagogical: [number, string, number][]; // ADDED
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
    };
  };
}

interface BiasExplainerProps {
  className?: string;
}

export const BiasExplainer: React.FC<BiasExplainerProps> = ({
  className = ''
}) => {
  const [comment, setComment] = useState('');
  const [discipline, setDiscipline] = useState('');
  const [selectedGender, setSelectedGender] = useState<'Male' | 'Female' | ''>('');
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

  // Get CSRF token
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

  // API Mutation
  const { mutate, data: explanation, isPending, isError, error } = useMutation<AttentionData, Error>({
    mutationFn: async () => {
      const csrfToken = getCsrfToken();
      const response = await axios({
        method: 'POST',
        url: '/api/sentiment-explainability/explain_comment/',
        data: {
          comment,
          discipline: discipline || undefined,
          gender: selectedGender // Send selected gender
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
    if (!comment.trim() || !selectedGender) return; // Require gender
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

  // Descriptor Focus Label (uses original score balancing non-objective terms)
  const getDescriptorFocusLabel = (score: number) => {
    if (score > 0.3) return 'Strong focus on Personality/Intellect over Competence/Warmth';
    if (score > 0.1) return 'Moderate focus on Personality/Intellect over Competence/Warmth';
    if (score < -0.3) return 'Strong focus on Competence/Warmth over Personality/Intellect';
    if (score < -0.1) return 'Moderate focus on Competence/Warmth over Personality/Intellect';
    return 'Balanced focus between descriptor types';
  };

  const getDescriptorFocusColor = (score: number) => {
    if (score > 0.3) return 'bg-purple-100 text-purple-800';
    if (score > 0.1) return 'bg-purple-50 text-purple-600';
    if (score < -0.3) return 'bg-orange-100 text-orange-800';
    if (score < -0.1) return 'bg-orange-50 text-orange-600';
    return 'bg-gray-100 text-gray-800';
  };

  // Tooltip component
  const Tooltip = ({ children, text }: { children: React.ReactNode, text: string }) => {
    return (
      <div className="relative group inline-block">
        {children}
        <div className="absolute z-10 bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-3 py-2 text-sm font-medium text-white bg-gray-900 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 w-64 shadow-lg pointer-events-none">
          {text}
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 w-2 h-2 rotate-45 bg-gray-900 -mt-1"></div> {/* Arrow */}
        </div>
      </div>
    );
  };

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 ${className}`}>
      {/* Title */}
      <h2 className="text-xl font-semibold text-gray-900 mb-6">
        Gender Bias Analysis in Comments
        <Tooltip text="Analyzes language for potential gender bias patterns considering the specified gender context and objective pedagogical terms.">
          <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
        </Tooltip>
      </h2>

      {/* Input Section */}
      <div className="space-y-4 mb-6">
        {/* Comment Textarea */}
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

        {/* Gender Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Select the gender of the person being evaluated: <span className="text-red-600">*</span>
          </label>
          <fieldset className="mt-2">
            <legend className="sr-only">Gender Selection</legend>
            <div className="space-x-4 flex">
              <div className="flex items-center">
                <input
                  id="gender-male"
                  name="gender-selection"
                  type="radio"
                  value="Male"
                  checked={selectedGender === 'Male'}
                  onChange={(e) => setSelectedGender(e.target.value as 'Male' | 'Female')}
                  className="h-4 w-4 text-indigo-600 border-gray-300 focus:ring-indigo-500"
                />
                <label htmlFor="gender-male" className="ml-2 block text-sm font-medium text-gray-700">
                  Male
                </label>
              </div>
              <div className="flex items-center">
                <input
                  id="gender-female"
                  name="gender-selection"
                  type="radio"
                  value="Female"
                  checked={selectedGender === 'Female'}
                  onChange={(e) => setSelectedGender(e.target.value as 'Male' | 'Female')}
                  className="h-4 w-4 text-indigo-600 border-gray-300 focus:ring-indigo-500"
                />
                <label htmlFor="gender-female" className="ml-2 block text-sm font-medium text-gray-700">
                  Female
                </label>
              </div>
            </div>
          </fieldset>
          {!selectedGender && comment.trim() && (
             <p className="mt-1 text-xs text-red-600">Please select a gender.</p>
          )}
        </div>

        {/* Discipline Selection */}
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

        {/* Analyze Button */}
        <div className="flex justify-center">
          <button
            onClick={analyzeComment}
            disabled={isAnalyzing || !comment.trim() || !selectedGender || isPending}
            className="px-4 py-2 bg-indigo-600 text-white font-medium rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {(isAnalyzing || isPending) && <Loader2 size={18} className="animate-spin mr-2" />}
            Analyze Comment
          </button>
        </div>
      </div>

      {/* Loading Indicator */}
      {isPending && (
        <div className="flex justify-center p-8">
          <Loader2 size={32} className="animate-spin text-indigo-600" />
        </div>
      )}

      {/* Error Message */}
      {isError && (
        <div className="bg-red-50 text-red-700 p-4 rounded-lg flex items-start">
          <AlertCircle size={20} className="mr-2 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium">Error analyzing comment</p>
            <p className="text-sm">{(error as Error).message || 'Unknown error occurred'}</p>
          </div>
        </div>
      )}

      {/* Results Display */}
      {explanation && !isPending && (
        <div className="space-y-6 mt-6">
          {/* Top Labels */}
          <div className="flex items-center space-x-4 flex-wrap gap-2">
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${
              explanation.prediction === 'Positive' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
            }`}>
              {explanation.prediction} ({(explanation.confidence * 100).toFixed(1)}%)
              <Tooltip text="Overall sentiment prediction.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
            {/* Use original_descriptor_bias_score for this broad label */}
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${getDescriptorFocusColor(explanation.gender_bias.original_descriptor_bias_score)}`}>
              {getDescriptorFocusLabel(explanation.gender_bias.original_descriptor_bias_score)}
              <Tooltip text="Indicates the balance between male-associated (personality/intellect) and female-associated (competence/warmth) descriptors, excluding objective terms.">
                <HelpCircle className="inline-block ml-1 cursor-help text-inherit opacity-70" size={14} />
              </Tooltip>
            </div>
          </div>

          {/* Gender Bias Assessment (Interpretation) */}
          <div className="mt-4">
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              Gender Bias Assessment
              <Tooltip text={`Model's interpretation of potential bias based on language patterns, objective terms, and the selected gender (${selectedGender}).`}>
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>
            {explanation.gender_bias.interpretation && explanation.gender_bias.interpretation.length > 0 ? (
              <div className="space-y-2">
                {explanation.gender_bias.interpretation.map((interpretation, index) => {
                  let bgColor = 'bg-gray-100'; // Default for undetermined/neutral pattern
                  let textColor = 'text-gray-800';
                  let Icon = Info;

                  if (interpretation.toLowerCase().includes('objective focus')) {
                    bgColor = 'bg-teal-50'; textColor = 'text-teal-800'; Icon = CheckCircle;
                  } else if (interpretation.toLowerCase().includes('negative bias')) {
                    bgColor = 'bg-red-50'; textColor = 'text-red-800'; Icon = AlertCircle;
                  } else if (interpretation.toLowerCase().includes('positive bias') || interpretation.toLowerCase().includes('stereotypical praise') || interpretation.toLowerCase().includes('stereotypes')) {
                    bgColor = 'bg-yellow-50'; textColor = 'text-yellow-800'; Icon = Info;
                  } else if (interpretation.toLowerCase().includes('focus on male-associated') || interpretation.toLowerCase().includes('focus on female-associated')) {
                    bgColor = 'bg-blue-50'; textColor = 'text-blue-800'; Icon = Info;
                  }

                  return (
                    <div key={index} className={`p-3 rounded-lg flex items-start ${bgColor}`}>
                      <Icon size={18} className={`mr-2 flex-shrink-0 mt-0.5 ${textColor}`} />
                      <p className={`${textColor} text-sm`}>{interpretation}</p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-800 text-sm">No specific bias interpretation generated.</p>
              </div>
            )}
          </div>

          {/* Attention visualization */}
          <div>
            <h3 className="text-lg font-medium text-gray-900 mb-3">
              Attention Visualization
              <Tooltip text="Word importance for sentiment prediction. Color indicates descriptor category (Objective=Teal). Size reflects attention weight.">
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>
            <div className="bg-gray-50 rounded-lg p-4 overflow-x-auto">
              <div className="flex flex-wrap gap-2 min-w-[640px]">
                {explanation.tokens.map((token, i: number) => {
                  const isObjectiveTerm = explanation.gender_bias.descriptor_categories.objective_pedagogical?.some(t => t[0] === i);
                  const isIntellectTerm = explanation.gender_bias.descriptor_categories.intellect_achievement?.some(t => t[0] === i);
                  const isEntertainmentTerm = explanation.gender_bias.descriptor_categories.entertainment_authority?.some(t => t[0] === i);
                  const isCompetenceTerm = explanation.gender_bias.descriptor_categories.competence_organization?.some(t => t[0] === i);
                  const isWarmthTerm = explanation.gender_bias.descriptor_categories.warmth_nurturing?.some(t => t[0] === i);
                  const isMaleNegativeTerm = explanation.gender_bias.descriptor_categories.male_negative?.some(t => t[0] === i);
                  const isFemaleNegativeTerm = explanation.gender_bias.descriptor_categories.female_negative?.some(t => t[0] === i);

                  const attentionWeight = explanation.attention[i];
                  const fontWeight = attentionWeight > 0.05 ? 'font-medium' : 'font-normal';
                  const opacity = Math.min(1, attentionWeight * 6); // Scale opacity

                  let bgColor = `rgba(209, 213, 219, ${opacity})`; // gray default
                  let textColor = 'text-gray-900';

                  // Check Objective first
                  if (isObjectiveTerm) {
                    bgColor = `rgba(20, 184, 166, ${opacity})`; // teal-500
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-teal-800';
                  } else if (isIntellectTerm) {
                    bgColor = `rgba(99, 102, 241, ${opacity})`; // indigo
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-indigo-800';
                  } else if (isEntertainmentTerm) {
                    bgColor = `rgba(168, 85, 247, ${opacity})`; // purple
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-purple-800';
                  } else if (isCompetenceTerm) {
                    bgColor = `rgba(249, 115, 22, ${opacity})`; // orange
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-orange-800';
                  } else if (isWarmthTerm) {
                    bgColor = `rgba(236, 72, 153, ${opacity})`; // pink
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-pink-800';
                  } else if (isMaleNegativeTerm) {
                    bgColor = `rgba(100, 116, 139, ${opacity})`; // slate
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-slate-800';
                  } else if (isFemaleNegativeTerm) {
                    bgColor = `rgba(244, 63, 94, ${opacity})`; // rose
                    textColor = attentionWeight > 0.1 ? 'text-white' : 'text-rose-800';
                  }

                  return (
                    <div key={`${token}-${i}`} className="flex flex-col items-center">
                      <div
                        className={`px-2 py-1 rounded ${textColor} ${fontWeight} transition-colors duration-150`}
                        style={{ backgroundColor: bgColor }}
                        title={`Attention: ${attentionWeight.toFixed(3)}`}
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
              <Tooltip text="Breakdown of model attention across different descriptor types. Objective terms relate to pedagogy/structure.">
                <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
              </Tooltip>
            </h3>

            {/* Objective Category Card */}
            <div className="p-4 rounded-lg bg-teal-50 mb-4 border border-teal-200">
                <div className="flex justify-between items-center">
                    <h4 className="font-medium text-teal-800">
                        Objective Pedagogical Terms
                        <Tooltip text="Focus on course structure, clarity, feedback, materials etc. High focus suggests less gendered language.">
                            <HelpCircle className="inline-block ml-1 cursor-help text-teal-600 opacity-70" size={14} />
                        </Tooltip>
                    </h4>
                    <span className="text-xl font-bold text-teal-600">
                        {explanation.gender_bias.category_attention_pct.objective_pedagogical?.toFixed(1) ?? '0.0'}%
                    </span>
                </div>
                <div className="mt-2">
                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                        className="bg-teal-600 h-2.5 rounded-full"
                        style={{ width: `${explanation.gender_bias.category_attention_pct.objective_pedagogical ?? 0}%` }}
                    />
                    </div>
                </div>
                <div className="mt-3 text-sm">
                    <strong>Top terms:</strong> {
                    explanation.gender_bias.top_terms_by_category.objective_pedagogical?.length > 0
                        ? explanation.gender_bias.top_terms_by_category.objective_pedagogical.map((t) => t[1]).join(', ')
                        : 'None'
                    }
                </div>
            </div>

            {/* Stereotypical Categories Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
               {/* Intellect/Achievement (Male Assoc) */}
               <div className="p-4 rounded-lg bg-indigo-50">
                   <div className="flex justify-between items-center">
                     <h4 className="font-medium text-indigo-800">
                       Intellect & Achievement
                       <Tooltip text="Terms describing intelligence, expertise.">
                         <HelpCircle className="inline-block ml-1 cursor-help text-indigo-600 opacity-70" size={14} />
                       </Tooltip>
                     </h4>
                     <span className="text-xl font-bold text-indigo-600">
                       {explanation.gender_bias.category_attention_pct.intellect_achievement?.toFixed(1) ?? '0.0'}%
                     </span>
                   </div>
                     <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-indigo-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.intellect_achievement ?? 0}%` }}/></div></div>
                    <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.intellect_achievement?.length > 0? explanation.gender_bias.top_terms_by_category.intellect_achievement.map((t) => t[1]).join(', '): 'None'}</div>
               </div>

               {/* Entertainment/Authority (Male Assoc) */}
               <div className="p-4 rounded-lg bg-purple-50">
                   <div className="flex justify-between items-center">
                     <h4 className="font-medium text-purple-800">
                       Entertainment & Authority
                       <Tooltip text="Terms describing humor, authority, confidence.">
                         <HelpCircle className="inline-block ml-1 cursor-help text-purple-600 opacity-70" size={14} />
                       </Tooltip>
                     </h4>
                     <span className="text-xl font-bold text-purple-600">
                       {explanation.gender_bias.category_attention_pct.entertainment_authority?.toFixed(1) ?? '0.0'}%
                     </span>
                   </div>
                     <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-purple-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.entertainment_authority ?? 0}%` }}/></div></div>
                     <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.entertainment_authority?.length > 0? explanation.gender_bias.top_terms_by_category.entertainment_authority.map((t) => t[1]).join(', '): 'None'}</div>
               </div>

               {/* Competence/Organization (Female Assoc) */}
               <div className="p-4 rounded-lg bg-orange-50">
                   <div className="flex justify-between items-center">
                     <h4 className="font-medium text-orange-800">
                       Competence & Organization
                       <Tooltip text="Terms describing preparedness, process, reliability (excluding objective terms like 'clear').">
                         <HelpCircle className="inline-block ml-1 cursor-help text-orange-600 opacity-70" size={14} />
                       </Tooltip>
                     </h4>
                     <span className="text-xl font-bold text-orange-600">
                       {explanation.gender_bias.category_attention_pct.competence_organization?.toFixed(1) ?? '0.0'}%
                     </span>
                   </div>
                     <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-orange-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.competence_organization ?? 0}%` }}/></div></div>
                    <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.competence_organization?.length > 0? explanation.gender_bias.top_terms_by_category.competence_organization.map((t) => t[1]).join(', '): 'None'}</div>
               </div>

               {/* Warmth/Nurturing (Female Assoc) */}
               <div className="p-4 rounded-lg bg-pink-50">
                   <div className="flex justify-between items-center">
                     <h4 className="font-medium text-pink-800">
                       Warmth & Nurturing
                       <Tooltip text="Terms describing caring, supportiveness, approachability.">
                         <HelpCircle className="inline-block ml-1 cursor-help text-pink-600 opacity-70" size={14} />
                       </Tooltip>
                     </h4>
                     <span className="text-xl font-bold text-pink-600">
                       {explanation.gender_bias.category_attention_pct.warmth_nurturing?.toFixed(1) ?? '0.0'}%
                     </span>
                   </div>
                    <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-pink-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.warmth_nurturing ?? 0}%` }}/></div></div>
                   <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.warmth_nurturing?.length > 0? explanation.gender_bias.top_terms_by_category.warmth_nurturing.map((t) => t[1]).join(', '): 'None'}</div>
               </div>
            </div>

            {/* Negative descriptors grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
               {/* Male-Associated Negative */}
               <div className="p-4 rounded-lg bg-slate-50">
                    <div className="flex justify-between items-center">
                      <h4 className="font-medium text-slate-800">
                        Male-Associated Negative Terms
                        <Tooltip text="Negative terms often used for male professors (e.g., boring, harsh, arrogant).">
                          <HelpCircle className="inline-block ml-1 cursor-help text-slate-600 opacity-70" size={14} />
                        </Tooltip>
                      </h4>
                      <span className="text-xl font-bold text-slate-600">
                        {explanation.gender_bias.category_attention_pct.male_negative?.toFixed(1) ?? '0.0'}%
                      </span>
                    </div>
                     <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-slate-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.male_negative ?? 0}%` }}/></div></div>
                    <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.male_negative?.length > 0? explanation.gender_bias.top_terms_by_category.male_negative.map((t) => t[1]).join(', '): 'None'}</div>
               </div>
               {/* Female-Associated Negative */}
               <div className="p-4 rounded-lg bg-rose-50">
                    <div className="flex justify-between items-center">
                      <h4 className="font-medium text-rose-800">
                        Female-Associated Negative Terms
                        <Tooltip text="Negative terms often used for female professors (e.g., unprofessional, emotional, strict).">
                          <HelpCircle className="inline-block ml-1 cursor-help text-rose-600 opacity-70" size={14} />
                        </Tooltip>
                      </h4>
                      <span className="text-xl font-bold text-rose-600">
                        {explanation.gender_bias.category_attention_pct.female_negative?.toFixed(1) ?? '0.0'}%
                      </span>
                    </div>
                    <div className="mt-2"><div className="w-full bg-gray-200 rounded-full h-2.5"><div className="bg-rose-600 h-2.5 rounded-full" style={{ width: `${explanation.gender_bias.category_attention_pct.female_negative ?? 0}%` }}/></div></div>
                   <div className="mt-3 text-sm"><strong>Top terms:</strong> {explanation.gender_bias.top_terms_by_category.female_negative?.length > 0? explanation.gender_bias.top_terms_by_category.female_negative.map((t) => t[1]).join(', '): 'None'}</div>
               </div>
            </div>
          </div>

          {/* Discipline context */}
          {explanation.discipline_context && (
            <div className="mt-6 border-t border-gray-200 pt-4">
              <div className="flex items-start space-x-2">
                <Info size={20} className="flex-shrink-0 mt-1 text-indigo-500" />
                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Disciplinary Context
                    <Tooltip text="How the balance of stereotypical descriptor language in this comment compares to typical gender rating gaps in the selected discipline.">
                      <HelpCircle className="inline-block ml-2 cursor-help text-gray-400" size={16} />
                    </Tooltip>
                  </h3>
                  <p className="mb-2 text-sm">
                    In <strong>{explanation.discipline_context.discipline}</strong>, the average rating gap is
                    <strong className={explanation.discipline_context.gender_rating_gap >= 0
                      ? ' text-blue-600'
                      : ' text-pink-600'
                    }>
                      {' '}{explanation.discipline_context.gender_rating_gap.toFixed(2)} points
                    </strong> ({explanation.discipline_context.gender_rating_gap >= 0 ? 'favoring male' : 'favoring female'} professors).
                    <span className="text-xs block text-gray-500">
                      (Avg Male: {explanation.discipline_context.male_avg_rating.toFixed(2)}, Avg Female: {explanation.discipline_context.female_avg_rating.toFixed(2)})
                    </span>
                  </p>
                  <div className={`p-2 rounded-lg ${
                     explanation.discipline_context.correlation.alignment === 'aligned' ? 'bg-yellow-50 text-yellow-800' : 'bg-green-50 text-green-800'
                  }`}>
                    <p className="text-sm">
                       <strong>Stereotype Focus vs. Rating Gap:</strong> {explanation.discipline_context.correlation.explanation}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};