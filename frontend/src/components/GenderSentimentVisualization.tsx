import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface Term {
  term: string;
  male_freq: number;
  female_freq: number;
  bias: 'Male' | 'Female';
  total_freq: number;
  male_rel_freq: number;
  female_rel_freq: number;
}

interface GenderSentimentVisualizationProps {
  terms: Term[];
  title: string;
}

export const GenderSentimentVisualization: React.FC<GenderSentimentVisualizationProps> = ({
  terms,
  title
}) => {
  // Separate male and female biased terms
  const maleBiased = terms
    .filter(term => term.bias === 'Male')
    .sort((a, b) => b.male_freq - a.male_freq)
    .slice(0, 10);

  const femaleBiased = terms
    .filter(term => term.bias === 'Female')
    .sort((a, b) => b.female_freq - a.female_freq)
    .slice(0, 10);

  const transformDataForChart = (terms: Term[]) => {
    return terms.map(term => ({
      term: term.term,
      Male: term.bias === 'Male' ? term.male_freq : 0,
      Female: term.bias === 'Female' ? term.female_freq : 0
    }));
  };

  return (
    <div className="w-full">
      <h3 className="text-lg font-semibold mb-4">{title}</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="h-[400px]">
          <h4 className="text-md font-medium mb-2">Male-Biased Terms</h4>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={transformDataForChart(maleBiased)}
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="term" type="category" />
              <Tooltip />
              <Legend />
              <Bar dataKey="Male" fill="#2563eb" name="Male Frequency" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="h-[400px]">
          <h4 className="text-md font-medium mb-2">Female-Biased Terms</h4>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={transformDataForChart(femaleBiased)}
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="term" type="category" />
              <Tooltip />
              <Legend />
              <Bar dataKey="Female" fill="#f97316" name="Female Frequency" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};