import React from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import { Loader2 } from 'lucide-react';

interface GenderDistribution {
  total: number;
  female_count: number;
  male_count: number;
  female_percent: number;
  male_percent: number;
}

interface DisciplineDistribution extends GenderDistribution {
  discipline: string;
}

interface SubDisciplineDistribution extends GenderDistribution {
  sub_discipline: string;
}

interface DistributionData {
  disciplines: {
    top: DisciplineDistribution[];
    bottom: DisciplineDistribution[];
  };
  sub_disciplines: {
    top: SubDisciplineDistribution[];
    bottom: SubDisciplineDistribution[];
  };
}

const DistributionTable: React.FC<{
  data: (DisciplineDistribution | SubDisciplineDistribution)[];
  nameKey: 'discipline' | 'sub_discipline';
  title: string;
}> = ({ data, nameKey, title }) => (
  <div className="bg-white rounded-lg shadow-md overflow-hidden">
    <div className="px-6 py-4 bg-gray-50 border-b">
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
    </div>
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              {nameKey === 'discipline' ? 'Discipline' : 'Sub-discipline'}
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Total
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Female
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Male
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Female %
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Male %
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {data.map((item, index) => {
            // TypeScript type guard
            const name = nameKey === 'discipline' 
              ? (item as DisciplineDistribution).discipline 
              : (item as SubDisciplineDistribution).sub_discipline;
              
            return (
              <tr key={name} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {name}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {item.total}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <span className="px-2 py-1 rounded-full bg-pink-100 text-pink-800">
                    {item.female_count}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <span className="px-2 py-1 rounded-full bg-blue-100 text-blue-800">
                    {item.male_count}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-pink-600 h-2.5 rounded-full"
                      style={{ width: `${item.female_percent}%` }}
                    />
                  </div>
                  <span className="text-pink-600 font-medium ml-2">
                    {item.female_percent.toFixed(1)}%
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                    <div
                      className="bg-blue-600 h-2.5 rounded-full"
                      style={{ width: `${item.male_percent}%` }}
                    />
                  </div>
                  <span className="text-blue-600 font-medium ml-2">
                    {item.male_percent.toFixed(1)}%
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  </div>
);

export const GenderDistributionDashboard: React.FC = () => {
  const { data, isLoading, error } = useQuery<DistributionData>({
    queryKey: ['gender-distribution'],
    queryFn: async () => {
      const { data } = await axios.get('/api/professors/gender_distribution/');
      return data;
    }
  });

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
      <div className="grid md:grid-cols-2 gap-8">
        <DistributionTable
          data={data.disciplines.top}
          nameKey="discipline"
          title="Top 3 Rated Disciplines"
        />
        <DistributionTable
          data={data.disciplines.bottom}
          nameKey="discipline"
          title="Bottom 3 Rated Disciplines"
        />
      </div>
      
      <div className="grid md:grid-cols-2 gap-8">
        <DistributionTable
          data={data.sub_disciplines.top}
          nameKey="sub_discipline"
          title="Top 10 Rated Sub-disciplines"
        />
        <DistributionTable
          data={data.sub_disciplines.bottom}
          nameKey="sub_discipline"
          title="Bottom 10 Rated Sub-disciplines"
        />
      </div>
    </div>
  );
};