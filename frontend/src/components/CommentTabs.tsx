import React from 'react';
import { Tab, TabGroup, TabList, TabPanels, TabPanel } from '@headlessui/react';
import { SmileIcon, FrownIcon, ThumbsUpIcon, ThumbsDownIcon } from 'lucide-react';

interface Comment {
  comment: string;
  processed_comment: string;
  sentiment: number;
  created_at: string;
  vader_compound?: number;
}

interface CommentTabsProps {
  comments: Comment[];
  className?: string;
}

function classNames(...classes: string[]) {
  return classes.filter(Boolean).join(' ');
}

export const CommentTabs: React.FC<CommentTabsProps> = ({ comments, className = '' }) => {
  const positiveComments = comments.filter(c => c.sentiment > 0);
  const negativeComments = comments.filter(c => c.sentiment < 0);

  return (
    <div className={`${className} bg-white rounded-lg shadow-md p-6`}>
      <TabGroup>
        <TabList className="flex space-x-1 rounded-xl bg-indigo-100 p-1">
          <Tab
            className={({ selected }) =>
              classNames(
                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                'flex items-center justify-center gap-2',
                selected
                  ? 'bg-white text-green-700 shadow'
                  : 'text-gray-700 hover:bg-white/[0.12] hover:text-green-600'
              )
            }
          >
            {({ selected }) => (
              <>
                {selected ? <SmileIcon size={20} /> : <ThumbsUpIcon size={20} />}
                Positive Comments ({positiveComments.length})
              </>
            )}
          </Tab>
          <Tab
            className={({ selected }) =>
              classNames(
                'w-full rounded-lg py-2.5 text-sm font-medium leading-5',
                'flex items-center justify-center gap-2',
                selected
                  ? 'bg-white text-red-700 shadow'
                  : 'text-gray-700 hover:bg-white/[0.12] hover:text-red-600'
              )
            }
          >
            {({ selected }) => (
              <>
                {selected ? <FrownIcon size={20} /> : <ThumbsDownIcon size={20} />}
                Negative Comments ({negativeComments.length})
              </>
            )}
          </Tab>
        </TabList>
        <TabPanels className="mt-4">
          <TabPanel className="space-y-4">
            {positiveComments.map((comment, idx) => (
              <div key={idx} className="p-4 bg-green-50 rounded-lg">
                <p className="text-gray-700">{comment.comment}</p>
                <div className="mt-2 flex justify-between items-center text-sm">
                  <span className="text-gray-500">
                    {new Date(comment.created_at).toLocaleDateString()}
                  </span>
                  <div className="flex items-center gap-2">
                    <SmileIcon size={16} className="text-green-600" />
                    <span className="font-medium text-green-600">
                      Score: {comment.sentiment.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            {positiveComments.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No positive comments yet
              </div>
            )}
          </TabPanel>
          <TabPanel className="space-y-4">
            {negativeComments.map((comment, idx) => (
              <div key={idx} className="p-4 bg-red-50 rounded-lg">
                <p className="text-gray-700">{comment.comment}</p>
                <div className="mt-2 flex justify-between items-center text-sm">
                  <span className="text-gray-500">
                    {new Date(comment.created_at).toLocaleDateString()}
                  </span>
                  <div className="flex items-center gap-2">
                    <FrownIcon size={16} className="text-red-600" />
                    <span className="font-medium text-red-600">
                      Score: {comment.sentiment.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
            {negativeComments.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No negative comments yet
              </div>
            )}
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};