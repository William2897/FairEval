// CommentTabs.tsx
import React, { useState, useMemo } from 'react';
import { Tab, TabGroup, TabList, TabPanels, TabPanel } from '@headlessui/react';
import { SmileIcon, FrownIcon, Info, Filter, ArrowUpDown } from 'lucide-react';

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
// --- UPDATE Comment Interface ---
interface Comment {
  id: number; // Add ID for key prop
  comment: string;
  processed_comment: string;
  sentiment: number;
  created_at: string;
  bias_tag: string | null; // e.g., 'POS_BIAS_M', 'NEG_BIAS_F', 'OBJECTIVE', 'NEUTRAL', etc.
  bias_interpretation: string | null;
  stereotype_bias_score: number | null;
  objective_focus_percentage: number | null;
  // Add other fields if needed from the .values() call
}
// --- END UPDATE ---

interface CommentTabsProps {
  comments: Comment[] | undefined;
  className?: string;
}

// Define filter options type
type FilterOption = 'all' | 'POS_BIAS_M' | 'POS_BIAS_F' | 'NEG_BIAS_M' | 'NEG_BIAS_F' | 'OBJECTIVE' | 'NEUTRAL' | 'UNKNOWN';
// Define sort options type
type SortOption = 'none' | 'stereotype_asc' | 'stereotype_desc' | 'objective_asc' | 'objective_desc' | 'date_asc' | 'date_desc';

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

// --- Helper to get Tag styling ---
const getBiasTagStyle = (tag: string | null): { bg: string; text: string; label: string } => {
    switch (tag) {
      case 'POS_BIAS_M':
        return { bg: 'bg-blue-100', text: 'text-blue-800', label: 'Male Stereotype Focus' };
      case 'POS_BIAS_F':
        return { bg: 'bg-pink-100', text: 'text-pink-800', label: 'Female Stereotype Focus' };
      case 'NEG_BIAS_M':
        return { bg: 'bg-slate-100', text: 'text-slate-800', label: 'Male Negative Bias' };
      case 'NEG_BIAS_F':
        return { bg: 'bg-rose-100', text: 'text-rose-800', label: 'Female Negative Bias' };
      case 'OBJECTIVE':
      case 'OBJECTIVE_M_LEAN':
      case 'OBJECTIVE_F_LEAN':
        return { bg: 'bg-teal-100', text: 'text-teal-800', label: 'Objective Focus' };
      case 'NEUTRAL':
        return { bg: 'bg-gray-100', text: 'text-gray-800', label: 'Neutral Pattern' };
      case 'UNKNOWN':
         return { bg: 'bg-orange-100', text: 'text-orange-800', label: 'Analysis Error' };
      default:
        return { bg: 'bg-gray-50', text: 'text-gray-500', label: 'No Bias Tag' };
    }
  };
// --- END Helper ---

export const CommentTabs: React.FC<CommentTabsProps> = ({
    comments = [],
    className = ''
}) => {
  // Add state for filtering and sorting
  const [filterOption, setFilterOption] = useState<FilterOption>('all');
  const [sortOption, setSortOption] = useState<SortOption>('none');
  
  // Ensure comments is an array before filtering
  const safeComments = Array.isArray(comments) ? comments : [];
  
  // Filter and sort comments based on selected options
  const filteredAndSortedComments = useMemo(() => {
    // First, filter the comments
    let filtered = [...safeComments];
    if (filterOption !== 'all') {
      filtered = filtered.filter(c => c.bias_tag === filterOption);
    }
    
    // Then sort the filtered comments
    switch (sortOption) {
      case 'stereotype_asc':
        return filtered.sort((a, b) => (a.stereotype_bias_score ?? 0) - (b.stereotype_bias_score ?? 0));
      case 'stereotype_desc':
        return filtered.sort((a, b) => (b.stereotype_bias_score ?? 0) - (a.stereotype_bias_score ?? 0));
      case 'objective_asc':
        return filtered.sort((a, b) => (a.objective_focus_percentage ?? 0) - (b.objective_focus_percentage ?? 0));
      case 'objective_desc':
        return filtered.sort((a, b) => (b.objective_focus_percentage ?? 0) - (a.objective_focus_percentage ?? 0));
      case 'date_asc':
        return filtered.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      case 'date_desc':
        return filtered.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      default:
        return filtered;
    }
  }, [safeComments, filterOption, sortOption]);
  
  const positiveComments = filteredAndSortedComments.filter(c => c.sentiment === 1);
  const negativeComments = filteredAndSortedComments.filter(c => c.sentiment === 0);

  const renderComment = (comment: Comment, index: number) => {
    const { bg, text, label } = getBiasTagStyle(comment.bias_tag);
    const biasInterpretationText = comment.bias_interpretation || 'No detailed interpretation available.';

    return (
        <div key={comment.id || index} className={`p-4 ${comment.sentiment === 1 ? 'bg-green-50' : 'bg-red-50'} rounded-lg border ${comment.sentiment === 1 ? 'border-green-100' : 'border-red-100'}`}>
            <p className="text-gray-800 text-sm">{comment.comment}</p>
            <div className="mt-2 pt-2 border-t border-gray-200 flex justify-between items-center text-xs">
            <span className="text-gray-500">
                {new Date(comment.created_at).toLocaleDateString()}
            </span>
            {/* --- Conditional Rendering for Academic User --- */}
              <Tooltip text={biasInterpretationText}>
                  <span className={`px-2 py-0.5 rounded-full font-medium ${bg} ${text}`}>
                  <Info size={12} className="inline mr-1" /> {label}
                  </span>
              </Tooltip>
            {/* --- End Conditional Rendering --- */}
            </div>
      </div>
    );
  };

  // Define filter options for the dropdown
  const filterOptions = [
    { value: 'all', label: 'Show All' },
    { value: 'POS_BIAS_M', label: 'Male Stereotype Focus' },
    { value: 'POS_BIAS_F', label: 'Female Stereotype Focus' },
    { value: 'NEG_BIAS_M', label: 'Male Negative Bias' },
    { value: 'NEG_BIAS_F', label: 'Female Negative Bias' },
    { value: 'OBJECTIVE', label: 'Objective Focus' },
    { value: 'NEUTRAL', label: 'Neutral Pattern' },
    { value: 'UNKNOWN', label: 'Analysis Error' }
  ];
  
  // Define sort options
  const sortOptions = [
    { value: 'none', label: 'Default Order' },
    { value: 'stereotype_asc', label: 'Stereotype Score: Female to Male' },
    { value: 'stereotype_desc', label: 'Stereotype Score: Male to Female' },
    { value: 'objective_asc', label: 'Objective Focus: Low to High' },
    { value: 'objective_desc', label: 'Objective Focus: High to Low' },
    { value: 'date_asc', label: 'Date: Oldest First' },
    { value: 'date_desc', label: 'Date: Newest First' }
  ];

  return (
    <div className={`${className} bg-white rounded-lg shadow-md p-6`}>
      {/* Filter and Sort Controls */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 space-y-4 md:space-y-0">
        {/* Filter dropdown */}
        <div className="flex items-center space-x-2">
          <Filter size={18} className="text-gray-500" />
          <label htmlFor="filter-bias" className="text-sm font-medium text-gray-700">Filter by bias tag:</label>
          <select
            id="filter-bias"
            value={filterOption}
            onChange={(e) => setFilterOption(e.target.value as FilterOption)}
            className="rounded-md border-gray-300 py-1 pl-3 pr-10 text-base focus:border-indigo-500 focus:outline-none focus:ring-indigo-500"
          >
            {filterOptions.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
        
        {/* Sort dropdown */}
        <div className="flex items-center space-x-2">
          <ArrowUpDown size={18} className="text-gray-500" />
          <label htmlFor="sort-comments" className="text-sm font-medium text-gray-700">Sort comments:</label>
          <select
            id="sort-comments"
            value={sortOption}
            onChange={(e) => setSortOption(e.target.value as SortOption)}
            className="rounded-md border-gray-300 py-1 pl-3 pr-10 text-base focus:border-indigo-500 focus:outline-none focus:ring-indigo-500"
          >
            {sortOptions.map(option => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </div>
      </div>
      
      {/* Active filters display */}
      {filterOption !== 'all' && (
        <div className="mb-4 px-4 py-2 bg-blue-50 border border-blue-200 rounded-md">
          <p className="text-sm text-blue-800">
            <span className="font-medium">Active filter:</span> {filterOptions.find(option => option.value === filterOption)?.label}
            <button 
              onClick={() => setFilterOption('all')} 
              className="ml-2 text-blue-600 hover:text-blue-800 underline"
            >
              Clear
            </button>
          </p>
        </div>
      )}

      <TabGroup>
        <TabList className="flex space-x-1 rounded-xl bg-indigo-100 p-1">
          {/* Positive Tab */}
          <Tab
            className={({ selected }) =>
              classNames( /* ... (existing styles) ... */
                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                'flex items-center justify-center gap-2',
                'focus:outline-none focus:ring-2 ring-offset-2 ring-offset-blue-400 ring-white ring-opacity-60',
                 selected
                  ? 'bg-white text-green-700 shadow'
                  : 'text-gray-700 hover:bg-white/[0.60] hover:text-green-600'
              )
            }
          >
             {/* ... (existing icon logic) ... */}
             <SmileIcon size={16} className="mr-1"/> Positive Comments ({positiveComments.length})
          </Tab>
          {/* Negative Tab */}
          <Tab
            className={({ selected }) =>
              classNames( /* ... (existing styles) ... */
                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                'flex items-center justify-center gap-2',
                'focus:outline-none focus:ring-2 ring-offset-2 ring-offset-blue-400 ring-white ring-opacity-60',
                selected
                 ? 'bg-white text-red-700 shadow'
                 : 'text-gray-700 hover:bg-white/[0.60] hover:text-red-600'
              )
            }
          >
            {/* ... (existing icon logic) ... */}
            <FrownIcon size={16} className="mr-1"/> Negative Comments ({negativeComments.length})
          </Tab>
        </TabList>
        <TabPanels className="mt-4 max-h-[600px] overflow-y-auto p-1"> {/* Added scroll */}
          {/* Positive Panel */}          <TabPanel className="space-y-3 rounded-xl">
            {positiveComments.length > 0 ? (
                positiveComments.map(renderComment)
            ) : (
              <div className="text-center py-8 text-gray-500 italic">
                {filterOption !== 'all' ? 'No positive comments match your filter.' : 'No positive comments found.'}
              </div>
            )}
          </TabPanel>
          {/* Negative Panel */}
          <TabPanel className="space-y-3 rounded-xl">
             {negativeComments.length > 0 ? (
                negativeComments.map(renderComment)
            ) : (
              <div className="text-center py-8 text-gray-500 italic">
                {filterOption !== 'all' ? 'No negative comments match your filter.' : 'No negative comments found.'}
              </div>
            )}
          </TabPanel>
        </TabPanels>
      </TabGroup>

      {/* Comment analytics summary */}
      {(positiveComments.length > 0 || negativeComments.length > 0) && (
        <div className="mt-6 pt-4 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row justify-between text-sm text-gray-600">
            <div className="mb-2 sm:mb-0">
              <span className="font-medium">Bias distribution:</span> {filterOption === 'all' ? 'All comments' : 'Filtered comments'}
            </div>
            {sortOption !== 'none' && (
              <div>
                <span className="font-medium">Sort method:</span> {sortOptions.find(option => option.value === sortOption)?.label}
              </div>
            )}
          </div>
          {filterOption === 'all' && safeComments.length > 0 && (
            <div className="mt-2 text-xs text-gray-500">
              <p>Tip: Use filters to isolate specific bias patterns and sorting to identify trends.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Remember to create or import the Tooltip component used above.