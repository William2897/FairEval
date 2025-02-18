import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2 } from 'lucide-react';

interface GenderDistribution {
  discipline?: string;
  sub_discipline?: string;
  total: number;
  female_count: number;
  male_count: number;
  female_percent: number;
  male_percent: number;
  rating?: number;
}

interface GenderData {
  disciplines: {
    top: GenderDistribution[];
    bottom: GenderDistribution[];
  };
  sub_disciplines: {
    top: GenderDistribution[];
    bottom: GenderDistribution[];
  };
  total_stats?: {
    total: number;
    female_count: number;
    male_count: number;
    female_percent: number;
    male_percent: number;
  };
}

export const GenderAnalysisDashboard: React.FC = () => {
  const { data, isLoading, error } = useQuery<GenderData>({
    queryKey: ['gender-distribution'],
    queryFn: async () => {
      try {
        const response = await axios.get('/api/professors/gender_distribution/');
        console.log('API Response:', response.data); // Debug log

        // Transform and validate the data
        const transformDistribution = (items: GenderDistribution[]) => {
          return items.map(item => {
            console.log('Processing item:', item); // Debug log
            return {
              ...item,
              total: Number(item.total) || 0,
              female_count: Number(item.female_count) || 0,
              male_count: Number(item.male_count) || 0,
              female_percent: Number(item.female_percent) || 0,
              male_percent: Number(item.male_percent) || 0,
            };
          });
        };

        const transformedData = {
          ...response.data,
          disciplines: {
            top: transformDistribution(response.data.disciplines.top),
            bottom: transformDistribution(response.data.disciplines.bottom),
          },
          sub_disciplines: {
            top: transformDistribution(response.data.sub_disciplines.top),
            bottom: transformDistribution(response.data.sub_disciplines.bottom),
          },
          total_stats: response.data.total_stats ? {
            total: Number(response.data.total_stats.total) || 0,
            female_count: Number(response.data.total_stats.female_count) || 0,
            male_count: Number(response.data.total_stats.male_count) || 0,
            female_percent: Number(response.data.total_stats.female_percent) || 0,
            male_percent: Number(response.data.total_stats.male_percent) || 0,
          } : undefined
        };

        console.log('Transformed data:', transformedData); // Debug log
        return transformedData;
      } catch (error) {
        console.error('API Error:', error);
        throw error;
      }
    }
  });

  if (error) {
    console.error('Query Error:', error); // Debug log
  }

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
        Failed to load gender distribution data
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Overall Stats Card */}
      {data.total_stats && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Overall Gender Distribution
          </h2>
          <div className="flex flex-col space-y-4">
            <div className="text-sm text-gray-600">Total Faculty: {data.total_stats.total}</div>
            <div className="flex flex-col space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-pink-600 h-3 rounded-full"
                    style={{ width: `${data.total_stats.female_percent}%` }}
                  />
                </div>
                <span className="text-sm text-pink-600 whitespace-nowrap min-w-[120px]">
                  Female: {data.total_stats.female_count} ({data.total_stats.female_percent.toFixed(1)}%)
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-blue-600 h-3 rounded-full"
                    style={{ width: `${data.total_stats.male_percent}%` }}
                  />
                </div>
                <span className="text-sm text-blue-600 whitespace-nowrap min-w-[120px]">
                  Male: {data.total_stats.male_count} ({data.total_stats.male_percent.toFixed(1)}%)
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Disciplines Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">
            Discipline Gender Distribution Analysis
          </h2>
        </div>

        {/* Top Disciplines */}
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Top Performing Disciplines</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Discipline</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Faculty</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gender Distribution</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.disciplines.top.map((analysis) => (
                  <tr key={analysis.discipline} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {analysis.discipline}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.rating?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.total}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-col space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-pink-600 h-2 rounded-full"
                              style={{ width: `${analysis.female_percent}%` }}
                            />
                          </div>
                          <span className="text-xs text-pink-600 whitespace-nowrap">
                            {analysis.female_count} ({analysis.female_percent?.toFixed(1) || 0}%)
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${analysis.male_percent}%` }}
                            />
                          </div>
                          <span className="text-xs text-blue-600 whitespace-nowrap">
                            {analysis.male_count} ({analysis.male_percent?.toFixed(1) || 0}%)
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Bottom Disciplines */}
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Bottom Performing Disciplines</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Discipline</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Faculty</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gender Distribution</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.disciplines.bottom.map((analysis) => (
                  <tr key={`discipline-bottom-${analysis.discipline}`} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {analysis.discipline}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.rating?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.total}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-col space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-pink-600 h-2 rounded-full"
                              style={{ width: `${analysis.female_percent}%` }}
                            />
                          </div>
                          <span className="text-xs text-pink-600 whitespace-nowrap">
                            {analysis.female_count} ({analysis.female_percent?.toFixed(1) || 0}%)
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${analysis.male_percent}%` }}
                            />
                          </div>
                          <span className="text-xs text-blue-600 whitespace-nowrap">
                            {analysis.male_count} ({analysis.male_percent?.toFixed(1) || 0}%)
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Sub-Disciplines Section */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900">
            Sub-Discipline Gender Distribution Analysis
          </h2>
        </div>

        {/* Top Sub-Disciplines */}
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Top Performing Sub-Disciplines</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Sub-Discipline</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Faculty</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gender Distribution</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.sub_disciplines.top.map((analysis, index) => (
                  <tr key={`subdiscipline-top-${analysis.sub_discipline || `unnamed-${index}`}`} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {analysis.sub_discipline || 'Unnamed Sub-discipline'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.rating?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.total || 0}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-col space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-pink-600 h-2 rounded-full"
                              style={{ width: `${analysis.female_percent || 0}%` }}
                            />
                          </div>
                          <span className="text-xs text-pink-600 whitespace-nowrap">
                            {analysis.female_count || 0} ({(analysis.female_percent || 0).toFixed(1)}%)
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${analysis.male_percent || 0}%` }}
                            />
                          </div>
                          <span className="text-xs text-blue-600 whitespace-nowrap">
                            {analysis.male_count || 0} ({(analysis.male_percent || 0).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Bottom Sub-Disciplines */}
        <div className="mt-8">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Bottom Performing Sub-Disciplines</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Sub-Discipline</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rating</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Total Faculty</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gender Distribution</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {data.sub_disciplines.bottom.map((analysis, index) => (
                  <tr key={`subdiscipline-bottom-${analysis.sub_discipline || `unnamed-${index}`}`} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {analysis.sub_discipline || 'Unnamed Sub-discipline'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.rating?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {analysis.total || 0}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-col space-y-1">
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-pink-600 h-2 rounded-full"
                              style={{ width: `${analysis.female_percent || 0}%` }}
                            />
                          </div>
                          <span className="text-xs text-pink-600 whitespace-nowrap">
                            {analysis.female_count || 0} ({(analysis.female_percent || 0).toFixed(1)}%)
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full"
                              style={{ width: `${analysis.male_percent || 0}%` }}
                            />
                          </div>
                          <span className="text-xs text-blue-600 whitespace-nowrap">
                            {analysis.male_count || 0} ({(analysis.male_percent || 0).toFixed(1)}%)
                          </span>
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};