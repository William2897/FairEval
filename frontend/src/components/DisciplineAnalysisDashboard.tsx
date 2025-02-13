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
  Treemap,
  Cell
} from 'recharts';
import { Loader2 } from 'lucide-react';
import { GenderAnalysisDashboard } from './GenderAnalysisDashboard';

interface DisciplineStats {
  discipline: string;
  sub_discipline: string | null;
  avg_rating: number;
  total_ratings: number;
  professor_count: number;
}

interface GenderStats {
  discipline: string;
  sub_discipline: string | null;
  gender: string;
  avg_rating: number;
  total_ratings: number;
}

interface TukeyResults {
  discipline: string;
  anova: {
    f_stat: number;
    p_value: number;
  };
  tukey: Array<{
    group1: string;
    group2: string;
    meandiff: number;
    lower: number;
    upper: number;
    reject: boolean;
  }> | null;
  summary: Array<{
    discipline: string;
    gender: string;
    mean: number;
    count: number;
  }>;
}

interface DisciplineData {
  discipline_ratings: DisciplineStats[];
  gender_distribution: GenderStats[];
  statistical_tests: TukeyResults[];
}

// Remove unused interface TreemapDataPoint

export const DisciplineAnalysisDashboard: React.FC = () => {
  const { data, isLoading, error } = useQuery<DisciplineData>({
    queryKey: ['discipline-analysis'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/discipline_stats/');
      return data;
    }
  });

  const [selectedView, setSelectedView] = useState<'disciplines' | 'subdisciplines'>('disciplines');
  const [activeTab, setActiveTab] = useState<'ratings' | 'treemap' | 'gender' | 'stats'>('ratings');

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        Failed to load discipline analysis data
      </div>
    );
  }

  // Aggregate ratings: if subdisciplines is selected, filter for entries with a sub_discipline;
  // otherwise (for disciplines) include all records.
  const aggregatedData = data.discipline_ratings
    .filter(d => selectedView === 'subdisciplines' ? d.sub_discipline : true)
    .reduce((acc, curr) => {
      const key = selectedView === 'disciplines' ? curr.discipline : curr.sub_discipline || '';
      if (!acc[key]) {
        acc[key] = {
          discipline: curr.discipline,
          sub_discipline: curr.sub_discipline,
          total_ratings: curr.total_ratings,
          sum_ratings: curr.avg_rating * curr.total_ratings, // weighted sum
          name: key // For chart consistency
        };
      } else {
        acc[key].total_ratings += curr.total_ratings;
        acc[key].sum_ratings += curr.avg_rating * curr.total_ratings;
      }
      return acc;
    }, {} as Record<string, any>);

  // Compute weighted average and sort by average rating descending
  const sortedRatings = Object.values(aggregatedData)
    .map(item => ({
      ...item,
      avg_rating: item.sum_ratings / item.total_ratings
    }))
    .sort((a, b) => b.avg_rating - a.avg_rating);

  // Prepare data for discipline treemap
  const prepareDisciplineTreemap = () => {
    const aggregatedData = data.discipline_ratings.reduce((acc, curr) => {
      const key = curr.discipline;
      if (!acc[key]) {
        acc[key] = {
          discipline: curr.discipline,
          total_ratings: curr.total_ratings,
          sum_ratings: curr.avg_rating * curr.total_ratings,
        };
      } else {
        acc[key].total_ratings += curr.total_ratings;
        acc[key].sum_ratings += curr.avg_rating * curr.total_ratings;
      }
      return acc;
    }, {} as Record<string, any>);

    return Object.entries(aggregatedData).map(([key, value]) => ({
      name: key,
      value: value.total_ratings,
      avg: value.sum_ratings / value.total_ratings,
      displayName: `${key}\n${(value.sum_ratings / value.total_ratings).toFixed(1)}\n(${value.total_ratings})`
    }));
  };

  // Prepare data for sub-discipline treemap
  const prepareSubDisciplineTreemap = () => {
    return data.discipline_ratings
      .filter(d => d.sub_discipline)
      .map(d => ({
        name: d.sub_discipline!,
        value: d.total_ratings,
        avg: d.avg_rating,
        displayName: `${d.sub_discipline}\n${d.avg_rating.toFixed(1)}\n(${d.total_ratings})`
      }));
  };

  return (
    <div className="space-y-4">
      {/* Tab Navigation */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-4" aria-label="Tabs">
          {[
            { id: 'ratings', label: 'Rating Distribution' },
            { id: 'treemap', label: 'Treemap Analysis' },
            { id: 'gender', label: 'Gender Analysis' },
            { id: 'stats', label: 'Statistical Tests' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`py-2 px-4 text-sm font-medium border-b-2 ${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'ratings' && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900">
              Rating Distribution by {selectedView === 'disciplines' ? 'Discipline' : 'Sub-discipline'}
            </h2>
            <div className="flex space-x-2">
              <button
                onClick={() => setSelectedView('disciplines')}
                className={`px-3 py-1 rounded-md ${
                  selectedView === 'disciplines' 
                    ? 'bg-indigo-100 text-indigo-700' 
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Disciplines
              </button>
              <button
                onClick={() => setSelectedView('subdisciplines')}
                className={`px-3 py-1 rounded-md ${
                  selectedView === 'subdisciplines'
                    ? 'bg-indigo-100 text-indigo-700'
                    : 'text-gray-600 hover:bg-gray-100'
                }`}
              >
                Sub-disciplines
              </button>
            </div>
          </div>
          
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sortedRatings}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name"
                  angle={-45} 
                  textAnchor="end"
                  height={100}
                  interval={0}
                  fontSize={11}
                />
                <YAxis 
                  domain={[0, 5]} 
                  ticks={[0, 1, 2, 3, 4, 5]}
                />
                <Tooltip 
                  formatter={(value: number) => [value.toFixed(2), 'Average Rating']}
                  labelFormatter={(label: string) => {
                    const item = sortedRatings.find(d => d.name === label);
                    return `${label} (${item?.total_ratings || 0} ratings)`;
                  }}
                />
                <Bar dataKey="avg_rating" fill="#4f46e5" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === 'treemap' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Rating Distribution Treemap Analysis
          </h2>
          <div className="space-y-8"> {/* Changed from grid to vertical spacing */}
            {/* Discipline Treemap */}
            <div>
              <h3 className="text-lg font-medium mb-4">By Discipline</h3>
              <div className="h-[500px]"> {/* Made each treemap taller */}
                <ResponsiveContainer width="100%" height="100%">
                  <Treemap
                    data={prepareDisciplineTreemap()}
                    dataKey="value"
                    stroke="#fff"
                  >
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white shadow-lg rounded-lg p-3 border">
                            <p className="font-semibold">{data.name}</p>
                            <p className="text-sm text-gray-600">Average Rating: {data.avg.toFixed(2)}</p>
                            <p className="text-sm text-gray-600">Total Ratings: {data.value}</p>
                          </div>
                        );
                      }}
                    />
                    {prepareDisciplineTreemap().map((entry, index) => {
                      const colorHue = Math.min(entry.avg * 40, 150);
                      const colorLightness = 45 + (entry.avg * 5);
                      return (
                        <Cell 
                          key={`cell-${index}`}
                          fill={`hsl(${colorHue}, 80%, ${colorLightness}%)`}
                        >
                          <text
                            x="50%"
                            y="50%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fill="#fff"
                            fontSize={11}
                            style={{
                              whiteSpace: 'pre',
                              textShadow: '0px 0px 3px rgba(0,0,0,0.5)'
                            }}
                          >
                            {entry.displayName}
                          </text>
                        </Cell>
                      );
                    })}
                  </Treemap>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Sub-discipline Treemap */}
            <div>
              <h3 className="text-lg font-medium mb-4">By Sub-discipline</h3>
              <div className="h-[500px]">
                <ResponsiveContainer width="100%" height="100%">
                  <Treemap
                    data={prepareSubDisciplineTreemap()}
                    dataKey="value"
                    stroke="#fff"
                  >
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const data = payload[0].payload;
                        return (
                          <div className="bg-white shadow-lg rounded-lg p-3 border">
                            <p className="font-semibold">{data.name}</p>
                            <p className="text-sm text-gray-600">Average Rating: {data.avg.toFixed(2)}</p>
                            <p className="text-sm text-gray-600">Total Ratings: {data.value}</p>
                          </div>
                        );
                      }}
                    />
                    {prepareSubDisciplineTreemap().map((entry, index) => {
                      const colorHue = Math.min(entry.avg * 40, 150);
                      const colorLightness = 45 + (entry.avg * 5);
                      return (
                        <Cell 
                          key={`cell-${index}`}
                          fill={`hsl(${colorHue}, 80%, ${colorLightness}%)`}
                        >
                          <text
                            x="50%"
                            y="50%"
                            textAnchor="middle"
                            dominantBaseline="middle"
                            fill="#fff"
                            fontSize={11}
                            style={{
                              whiteSpace: 'pre',
                              textShadow: '0px 0px 3px rgba(0,0,0,0.5)'
                            }}
                          >
                            {entry.displayName}
                          </text>
                        </Cell>
                      );
                    })}
                  </Treemap>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'gender' && (
        <div>
          <GenderAnalysisDashboard />
        </div>
      )}

      {activeTab === 'stats' && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-6">
            Statistical Analysis Results
          </h2>
          <div className="space-y-6">
            {data.statistical_tests.map((test) => (
              <div key={test.discipline} className="border-b pb-4">
                <h3 className="font-medium text-lg mb-2">{test.discipline}</h3>
                <p className="text-sm text-gray-600 mb-2">
                  ANOVA Results: F={test.anova.f_stat.toFixed(2)}, 
                  p={test.anova.p_value < 0.001 ? '< 0.001' : test.anova.p_value.toFixed(3)}
                </p>
                {test.tukey && (
                  <div className="mt-2">
                    <h4 className="font-medium mb-1">Tukey's HSD Test Results:</h4>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="bg-gray-50">
                            <th className="px-4 py-2">Comparison</th>
                            <th className="px-4 py-2">Mean Diff</th>
                            <th className="px-4 py-2">CI Lower</th>
                            <th className="px-4 py-2">CI Upper</th>
                            <th className="px-4 py-2">Significant</th>
                          </tr>
                        </thead>
                        <tbody>
                          {test.tukey.map((result, idx) => (
                            <tr key={idx} className="border-t">
                              <td className="px-4 py-2">
                                {result.group1} vs {result.group2}
                              </td>
                              <td className="px-4 py-2">
                                {result.meandiff.toFixed(2)}
                              </td>
                              <td className="px-4 py-2">
                                {result.lower.toFixed(2)}
                              </td>
                              <td className="px-4 py-2">
                                {result.upper.toFixed(2)}
                              </td>
                              <td className="px-4 py-2">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                                  result.reject 
                                    ? 'bg-green-100 text-green-800'
                                    : 'bg-gray-100 text-gray-800'
                                }`}>{result.reject ? 'Yes' : 'No'}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};