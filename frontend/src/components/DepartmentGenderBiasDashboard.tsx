import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Loader2, Info } from 'lucide-react';

interface BiasKeyword {
  word: string;
  frequency: number;
  bias_score: number;
}

interface DepartmentBias {
  department_name: string;
  male_avg_rating: number;
  female_avg_rating: number;
  rating_difference: number;
  p_value: number;
  sample_size_male: number;
  sample_size_female: number;
  significant: boolean;
}

interface GenderBiasData {
  department_biases: DepartmentBias[];
  male_biased_keywords: BiasKeyword[];
  female_biased_keywords: BiasKeyword[];
  overall_bias_score: number;
}

const formatPValue = (p: number): string => {
  if (p < 0.001) return 'p < 0.001';
  return `p = ${p.toFixed(3)}`;
};

const BiasKeywordList: React.FC<{
  title: string;
  keywords: BiasKeyword[];
  className?: string;
}> = ({ title, keywords, className = '' }) => (
  <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
    <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
    <div className="grid grid-cols-2 gap-4">
      {keywords.map((keyword) => (
        <div
          key={keyword.word}
          className="flex items-center justify-between p-2 bg-gray-50 rounded"
        >
          <span className="font-medium">{keyword.word}</span>
          <div className="text-sm text-gray-500">
            <span className="mr-2">Score: {keyword.bias_score.toFixed(2)}</span>
            <span>({keyword.frequency}×)</span>
          </div>
        </div>
      ))}
    </div>
  </div>
);

export const DepartmentGenderBiasDashboard: React.FC = () => {
  const [sortBy, setSortBy] = useState<'difference' | 'significance'>('difference');
  
  const { data, isLoading, isError, error } = useQuery<GenderBiasData>({
    queryKey: ['genderBias'],
    queryFn: async () => {
      const { data } = await axios.get('/api/departments/bias/gender/');
      return data;
    },
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        Error loading gender bias data: {(error as Error).message}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        No data available
      </div>
    );
  }

  const sortedDepartments = [...data.department_biases].sort((a, b) => {
    if (sortBy === 'significance') {
      return a.p_value - b.p_value;
    }
    return Math.abs(b.rating_difference) - Math.abs(a.rating_difference);
  });

  return (
    <div className="space-y-8">
      {/* Overall Bias Score */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">
            Gender Bias Analysis Dashboard
          </h2>
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-500">Overall Bias Score:</span>
            <span className={`text-lg font-bold ${
              Math.abs(data.overall_bias_score) > 0.1 
                ? 'text-red-600' 
                : 'text-green-600'
            }`}>
              {data.overall_bias_score.toFixed(3)}
            </span>
            <div title="Positive values indicate bias favoring male professors, negative values indicate bias favoring female professors">
              <Info 
                size={16} 
                className="text-gray-400 cursor-help"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Department Bias Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-lg font-semibold text-gray-900">
            Rating Differences by Department
          </h3>
          <div className="flex items-center space-x-4">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'difference' | 'significance')}
              className="border rounded-md px-3 py-1 text-sm"
            >
              <option value="difference">Sort by Difference</option>
              <option value="significance">Sort by Significance</option>
            </select>
          </div>
        </div>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={sortedDepartments} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={['dataMin', 'dataMax']} />
              <YAxis dataKey="department_name" type="category" width={150} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const data = payload[0].payload as DepartmentBias;
                  return (
                    <div className="bg-white shadow-lg rounded-lg p-3 border">
                      <p className="font-semibold">{data.department_name}</p>
                      <p>Difference: {data.rating_difference.toFixed(2)}</p>
                      <p>{formatPValue(data.p_value)}</p>
                      <p className="text-sm text-gray-500">
                        Male: {data.sample_size_male} reviews
                        <br />
                        Female: {data.sample_size_female} reviews
                      </p>
                    </div>
                  );
                }}
              />
              <Bar dataKey="rating_difference" fill="#4f46e5">
                {sortedDepartments.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.significant ? '#4f46e5' : '#94a3b8'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Detailed Statistics Table */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Department
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Male Avg
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Female Avg
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Difference
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Significance
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Sample Size
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {sortedDepartments.map((dept) => (
              <tr key={dept.department_name} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  {dept.department_name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {dept.male_avg_rating.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {dept.female_avg_rating.toFixed(2)}
                </td>
                <td className={`px-6 py-4 whitespace-nowrap font-medium ${
                  Math.abs(dept.rating_difference) > 0.5 
                    ? 'text-red-600' 
                    : 'text-gray-900'
                }`}>
                  {dept.rating_difference > 0 ? '+' : ''}
                  {dept.rating_difference.toFixed(2)}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    dept.significant 
                      ? 'bg-yellow-100 text-yellow-800' 
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {formatPValue(dept.p_value)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  M: {dept.sample_size_male} / F: {dept.sample_size_female}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Biased Keywords Analysis */}
      <div className="grid md:grid-cols-2 gap-6">
        <BiasKeywordList
          title="Male-Associated Keywords"
          keywords={data.male_biased_keywords}
        />
        <BiasKeywordList
          title="Female-Associated Keywords"
          keywords={data.female_biased_keywords}
        />
      </div>
    </div>
  );
};