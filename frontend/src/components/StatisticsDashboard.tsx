import React, { useState, useMemo } from 'react';
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
  ReferenceLine,
  Label,
  ErrorBar,
  Cell
} from 'recharts';
import { Loader2 } from 'lucide-react';

interface TukeyComparison {
  group1: string;
  group2: string;
  meandiff: number;
  lower: number;
  upper: number;
  p_adj: number;
  reject: boolean;
}

interface TukeyData {
  discipline_comparisons: TukeyComparison[];
  sub_discipline_comparisons: TukeyComparison[];
}

interface FormattedComparison extends TukeyComparison {
  comparison: string;
  errorUpper: number;
  errorLower: number;
}

interface HeatmapData {
  gender: string;
  discipline: string;
  avg_rating: number;
}

export const StatisticsDashboard: React.FC = () => {
  const [activeView, setActiveView] = useState<'discipline' | 'sub-discipline'>('discipline');
  
  const { data: tukeyData, isLoading: tukeyLoading, error: tukeyError } = useQuery<TukeyData>({
    queryKey: ['tukey-analysis'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/tukey_analysis/');
      return data;
    }
  });

  const { data: heatmapData, isLoading: heatmapLoading } = useQuery<HeatmapData[]>({
    queryKey: ['gender-discipline-heatmap'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/gender_discipline_heatmap/');
      return data;
    }
  });

  const processedHeatmapData = useMemo(() => {
    if (!heatmapData) return { cells: [], disciplines: [], genders: [] };

    const disciplines = Array.from(new Set(heatmapData.map(d => d.discipline))).sort();
    const genders = Array.from(new Set(heatmapData.map(d => d.gender))).sort();
    
    // Create a lookup table for quick access to ratings
    const ratingLookup = new Map(
      heatmapData.map(d => [`${d.gender}-${d.discipline}`, d.avg_rating])
    );

    return {
      disciplines,
      genders,
      cells: genders.flatMap(gender =>
        disciplines.map(discipline => ({
          gender,
          discipline,
          rating: ratingLookup.get(`${gender}-${discipline}`) || 0
        }))
      )
    };
  }, [heatmapData]);

  if (tukeyLoading || heatmapLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  if (tukeyError || !tukeyData) {
    return (
      <div className="bg-red-50 text-red-700 p-4 rounded-lg">
        Failed to load statistical analysis data
      </div>
    );
  }

  // Transform data for visualization
  const formatComparisons = (comparisons: TukeyComparison[]): FormattedComparison[] => {
    return comparisons.map(comp => ({
      ...comp,
      comparison: `${comp.group1} vs ${comp.group2}`,
      errorUpper: comp.upper - comp.meandiff,
      errorLower: comp.meandiff - comp.lower
    })).sort((a, b) => a.meandiff - b.meandiff);
  };

  const disciplineData = formatComparisons(tukeyData?.discipline_comparisons ?? []);
  const subDisciplineData = formatComparisons(tukeyData?.sub_discipline_comparisons ?? []);

  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;
    const data = payload[0].payload as FormattedComparison;
    return (
      <div className="bg-white p-3 border rounded-lg shadow-lg">
        <p className="font-semibold">{data.comparison}</p>
        <p>Mean Difference: {data.meandiff.toFixed(3)}</p>
        <p>p-value: {data.p_adj.toFixed(3)}</p>
        <p>95% CI: [{data.lower.toFixed(3)}, {data.upper.toFixed(3)}]</p>
      </div>
    );
  };

  const renderBarChart = (data: FormattedComparison[], title: string, isSubDiscipline: boolean = false) => (
    <div className={`bg-white rounded-lg shadow p-6 mb-6 ${isSubDiscipline ? 'w-full max-w-[1600px] mx-auto' : ''}`}>
      <h2 className="text-xl font-semibold text-gray-900 mb-6">{title}</h2>
      <div className={`${isSubDiscipline ? 'h-[1600px]' : 'h-[600px]'}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            layout="vertical"
            margin={isSubDiscipline ? 
              { top: 20, right: 160, bottom: 20, left: 300 } : 
              { top: 20, right: 120, bottom: 20, left: 200 }
            }
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" domain={['auto', 'auto']}>
              <Label 
                value="Mean Difference in Ratings" 
                offset={-10} 
                position="insideBottom"
                style={{ fontSize: isSubDiscipline ? '14px' : '12px' }}
              />
            </XAxis>
            <YAxis
              type="category"
              dataKey="comparison"
              tick={{ 
                fontSize: isSubDiscipline ? 14 : 12,
                width: isSubDiscipline ? 280 : 200 
              }}
              width={isSubDiscipline ? 280 : 200}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine x={0} stroke="#666" />
            <Bar
              dataKey="meandiff"
              fill="#3b82f6"
              stroke="#000000"
              fillOpacity={0.8}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.meandiff < 0 ? '#ef4444' : '#3b82f6'}
                />
              ))}
              <ErrorBar
                dataKey="errorUpper"
                direction="x"
                stroke="#666"
                strokeWidth={1}
                width={4}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  const getRatingColor = (rating: number) => {
    // Calculate color based on rating (1-5 scale)
    const normalizedRating = (rating - 1) / 4; // Convert 1-5 to 0-1
    const hue = normalizedRating * 120; // 0 = red (0), 1 = green (120)
    return `hsl(${hue}, 70%, 50%)`;
  };

  return (
    <div className="space-y-6">
      {/* Heatmap Section */}
      <div className="bg-white rounded-lg shadow p-6 mb-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">
          Average Discipline Rating by Gender
        </h2>
        <div className="relative overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="px-4 py-2 bg-gray-50"></th>
                {processedHeatmapData.disciplines.map(discipline => (
                  <th 
                    key={discipline}
                    className="px-4 py-2 bg-gray-50 text-sm font-medium text-gray-700"
                    style={{ minWidth: '120px' }}
                  >
                    {discipline}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {processedHeatmapData.genders.map(gender => (
                <tr key={gender}>
                  <th className="px-4 py-2 bg-gray-50 text-sm font-medium text-gray-700 text-left">
                    {gender}
                  </th>
                  {processedHeatmapData.disciplines.map(discipline => {
                    const cell = processedHeatmapData.cells.find(
                      c => c.gender === gender && c.discipline === discipline
                    );
                    const rating = cell?.rating || 0;
                    return (
                      <td
                        key={`${gender}-${discipline}`}
                        className="px-4 py-2 text-center"
                        style={{
                          backgroundColor: getRatingColor(rating),
                          color: 'black',
                          fontWeight: 400,
                        }}
                      >
                        {rating.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tukey Analysis Section */}
      <div className="flex justify-center space-x-4 mb-6">
        <button
          onClick={() => setActiveView('discipline')}
          className={`px-4 py-2 rounded-md ${
            activeView === 'discipline'
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Discipline Analysis
        </button>
        <button
          onClick={() => setActiveView('sub-discipline')}
          className={`px-4 py-2 rounded-md ${
            activeView === 'sub-discipline'
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
          }`}
        >
          Sub-discipline Analysis
        </button>
      </div>

      {activeView === 'discipline' && renderBarChart(
        disciplineData,
        "Tukey HSD Analysis - Gender Comparisons by Discipline",
        false
      )}

      {activeView === 'sub-discipline' && renderBarChart(
        subDisciplineData,
        "Tukey HSD Analysis - Gender Comparisons by Sub-Discipline",
        true
      )}
    </div>
  );
};