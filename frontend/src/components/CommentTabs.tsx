// CommentTabs.tsx
import React from 'react';
import { Tab, TabGroup, TabList, TabPanels, TabPanel } from '@headlessui/react';
import { SmileIcon, FrownIcon, Info } from 'lucide-react';

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
  // Ensure comments is an array before filtering
  const safeComments = Array.isArray(comments) ? comments : [];
  const positiveComments = safeComments.filter(c => c.sentiment === 1);
  const negativeComments = safeComments.filter(c => c.sentiment === 0);

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


  return (
    <div className={`${className} bg-white rounded-lg shadow-md p-6`}>
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
          {/* Positive Panel */}
          <TabPanel className="space-y-3 rounded-xl">
            {positiveComments.length > 0 ? (
                positiveComments.map(renderComment)
            ) : (
              <div className="text-center py-8 text-gray-500 italic">
                No positive comments found.
              </div>
            )}
          </TabPanel>
          {/* Negative Panel */}
          <TabPanel className="space-y-3 rounded-xl">
             {negativeComments.length > 0 ? (
                negativeComments.map(renderComment)
            ) : (
              <div className="text-center py-8 text-gray-500 italic">
                No negative comments found.
              </div>
            )}
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

// Remember to create or import the Tooltip component used above.