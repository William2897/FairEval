import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface GenderTermData {
  term: string;
  male_freq: number;
  female_freq: number;
  bias: 'Male' | 'Female';
}

interface Props {
  positiveTerms: GenderTermData[];
  negativeTerms: GenderTermData[];
  title?: string;
}

export const GenderSentimentVisualization: React.FC<Props> = ({ 
  positiveTerms, 
  negativeTerms, 
  title = "Gender-Based Term Analysis" 
}) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-md">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">{title}</h2>
      
      {/* Positive Terms */}
      <div>
        <h3 className="text-lg font-medium text-gray-800 mb-3">Positive Terms by Gender</h3>
        <div className="h-64 mb-8">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={positiveTerms}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="term" type="category" />
              <Tooltip />
              <Bar dataKey="male_freq" name="Male" fill="#3B82F6" />
              <Bar dataKey="female_freq" name="Female" fill="#EC4899" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Negative Terms */}
      <div>
        <h3 className="text-lg font-medium text-gray-800 mb-3">Negative Terms by Gender</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={negativeTerms}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="term" type="category" />
              <Tooltip />
              <Bar dataKey="male_freq" name="Male" fill="#3B82F6" />
              <Bar dataKey="female_freq" name="Female" fill="#EC4899" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};