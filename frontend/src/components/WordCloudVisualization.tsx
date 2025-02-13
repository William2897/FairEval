import React from 'react';
import ReactWordcloud, { Scale } from 'react-wordcloud';
import "tippy.js/dist/tippy.css";
import "tippy.js/animations/scale.css";

interface WordCloudData {
  text: string;
  value: number;
}

interface WordCloudProps {
  words: Array<{ word: string; count: number }>;
  title: string;
  colorScheme?: 'positive' | 'negative';
}

export const WordCloudVisualization: React.FC<WordCloudProps> = ({ words, title, colorScheme = 'positive' }) => {
  const wordcloudData: WordCloudData[] = words.map(({ word, count }) => ({
    text: word,
    value: count,
  }));

  const options = {
    colors: colorScheme === 'positive' 
      ? ['#047857', '#059669', '#10B981', '#34D399', '#6EE7B7']  // green shades
      : ['#991B1B', '#DC2626', '#EF4444', '#F87171', '#FCA5A5'], // red shades
    enableTooltip: true,
    deterministic: true,
    fontFamily: 'Inter',
    fontSizes: [20, 60] as [number, number], // Fix the type to be a tuple of exactly 2 numbers
    fontStyle: 'normal',
    fontWeight: 'normal',
    padding: 1,
    rotations: 0,
    rotationAngles: [0, 0] as [number, number], // Fixed to be a tuple of 2 numbers
    scale: 'sqrt' as Scale,
    spiral: 'archimedean' as 'archimedean' | 'rectangular',
    transitionDuration: 1000,
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">{title}</h2>
      <div style={{ height: '400px' }}>
        {wordcloudData.length > 0 ? (
          <ReactWordcloud words={wordcloudData} options={options} />
        ) : (
          <div className="h-full flex items-center justify-center text-gray-500">
            No words to display
          </div>
        )}
      </div>
    </div>
  );
};