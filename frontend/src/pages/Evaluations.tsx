import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, Filter, AlertOctagon, ChevronUp, ChevronDown, X, ChevronLeft, ChevronRight } from 'lucide-react';
import { useState, useMemo, useEffect } from 'react';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { DataUploadForm } from '../components/DataUploadForm';
import { TopicModelVisualization } from '../components/TopicModelVisualization';
import { RecommendationDisplay } from '../components/RecommendationDisplay';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';

interface Rating {
  id: number;
  professor: number;
  professor_name: string;
  discipline: string;
  sub_discipline: string | null;
  avg_rating: number;
  flag_status: string | null;
  helpful_rating: number | null;
  clarity_rating: number | null;
  difficulty_rating: number | null;
  is_online: boolean;
  is_for_credit: boolean;
  created_at: string;
}

interface FilterState {
  search: string;
  ratingMin: number;
  isOnline: boolean | null;
  isForCredit: boolean | null;
}

interface PaginatedResponse {
  count: number;
  next: string | null;
  previous: string | null;
  results: Rating[];
}

function Evaluations() {
  const { user } = useAuth();
  const [selectedProfessorId, setSelectedProfessorId] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{ key: keyof Rating; direction: 'asc' | 'desc' }>({ 
    key: 'created_at', 
    direction: 'desc' 
  });
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    search: '',
    ratingMin: 0,
    isOnline: null,
    isForCredit: null
  });
  
  const { data: paginatedData, isLoading } = useQuery<PaginatedResponse>({
    queryKey: ['ratings', currentPage, filters, sortConfig],
    queryFn: async () => {
      const params = new URLSearchParams({
        page: currentPage.toString(),
        ...(filters.search && { search: filters.search }),
        ...(filters.ratingMin > 0 && { min_rating: filters.ratingMin.toString() }),
        ...(filters.isOnline !== null && { is_online: filters.isOnline.toString() }),
        ...(filters.isForCredit !== null && { is_for_credit: filters.isForCredit.toString() }),
        ...(sortConfig && { ordering: `${sortConfig.direction === 'desc' ? '-' : ''}${sortConfig.key}` })
      });
      
      const { data } = await axios.get(`/api/ratings/?${params}`);
      return data;
    },
  });

  // Reset to first page when filters or sort change
  useEffect(() => {
    setCurrentPage(1);
  }, [filters, sortConfig]);

  const ratings = paginatedData?.results || [];
  const totalPages = Math.ceil((paginatedData?.count || 0) / 50);

  const filteredAndSortedRatings = useMemo(() => {
    if (!ratings) return [];

    let filtered = ratings.filter(rating => {
      const searchMatch = 
        (rating.professor_name?.toLowerCase().includes(filters.search.toLowerCase()) ||
        rating.discipline?.toLowerCase().includes(filters.search.toLowerCase()) ||
        rating.sub_discipline?.toLowerCase().includes(filters.search.toLowerCase()));
      const ratingMatch = rating.avg_rating >= filters.ratingMin;
      const onlineMatch = filters.isOnline === null || rating.is_online === filters.isOnline;
      const creditMatch = filters.isForCredit === null || rating.is_for_credit === filters.isForCredit;
      
      return searchMatch && ratingMatch && onlineMatch && creditMatch;
    });

    return filtered.sort((a, b) => {
      const aValue = a[sortConfig.key];
      const bValue = b[sortConfig.key];
      
      if (aValue === bValue) return 0;
      if (aValue === null) return 1;
      if (bValue === null) return -1;
      
      const comparison = aValue < bValue ? -1 : 1;
      return sortConfig.direction === 'desc' ? -comparison : comparison;
    });
  }, [ratings, sortConfig, filters]);

  const handleSort = (key: keyof Rating) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  };

  const SortIndicator = ({ column }: { column: keyof Rating }) => {
    if (sortConfig.key !== column) return null;
    return sortConfig.direction === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin" size={32} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold text-gray-900">Course Evaluations</h1>
        <div className="flex space-x-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search by professor or discipline..."
              value={filters.search}
              onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
              className="pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>
          <button 
            onClick={() => setShowFilters(!showFilters)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-md ${
              showFilters 
                ? 'bg-indigo-100 text-indigo-700 border-indigo-200' 
                : 'bg-white border text-gray-700 hover:bg-gray-50'
            } border`}
          >
            <Filter size={20} />
            <span>Filters</span>
          </button>
        </div>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <div className="bg-white p-4 rounded-lg shadow border">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-medium text-gray-900">Filter Options</h3>
            <button onClick={() => setShowFilters(false)} className="text-gray-500 hover:text-gray-700">
              <X size={20} />
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Minimum Rating
              </label>
              <select
                value={filters.ratingMin}
                onChange={(e) => setFilters(prev => ({ ...prev, ratingMin: Number(e.target.value) }))}
                className="w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value={0}>All Ratings</option>
                <option value={3}>3.0+</option>
                <option value={4}>4.0+</option>
                <option value={4.5}>4.5+</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Course Type
              </label>
              <select
                value={filters.isOnline === null ? '' : filters.isOnline.toString()}
                onChange={(e) => setFilters(prev => ({ 
                  ...prev, 
                  isOnline: e.target.value === '' ? null : e.target.value === 'true'
                }))}
                className="w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="">All Courses</option>
                <option value="true">Online Only</option>
                <option value="false">In-Person Only</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Credit Status
              </label>
              <select
                value={filters.isForCredit === null ? '' : filters.isForCredit.toString()}
                onChange={(e) => setFilters(prev => ({ 
                  ...prev, 
                  isForCredit: e.target.value === '' ? null : e.target.value === 'true'
                }))}
                className="w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500"
              >
                <option value="">All Types</option>
                <option value="true">For Credit Only</option>
                <option value="false">Not For Credit Only</option>
              </select>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-6">
        <div className="bg-white shadow-md rounded-lg overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {[
                    { key: 'discipline', label: 'Discipline' },
                    { key: 'sub_discipline', label: 'Sub-discipline' },
                    { key: 'professor_name', label: 'Professor' },
                    { key: 'avg_rating', label: 'Overall Rating' },
                    { key: 'helpful_rating', label: 'Helpful' },
                    { key: 'clarity_rating', label: 'Clarity' },
                    { key: 'difficulty_rating', label: 'Difficulty' },
                    { key: 'created_at', label: 'Date' }
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      onClick={() => handleSort(key as keyof Rating)}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    >
                      <div className="flex items-center space-x-1">
                        <span>{label}</span>
                        <SortIndicator column={key as keyof Rating} />
                      </div>
                    </th>
                  ))}
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredAndSortedRatings.map((rating) => (
                  <tr 
                    key={rating.id} 
                    className={`hover:bg-gray-50 cursor-pointer transition-colors ${
                      selectedProfessorId === rating.professor ? 'bg-indigo-50' : ''
                    }`}
                    onClick={() => setSelectedProfessorId(
                      selectedProfessorId === rating.professor ? null : rating.professor
                    )}
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        {rating.discipline}
                        {rating.flag_status && (
                          <div className="ml-2" title={rating.flag_status}>
                            <AlertOctagon size={16} className="text-yellow-500" />
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">{rating.sub_discipline || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{rating.professor_name}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <div className="w-16 text-sm font-medium">{rating.avg_rating.toFixed(2)}</div>
                        <div className="flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className="h-full bg-indigo-600 rounded-full"
                            style={{ width: `${(rating.avg_rating / 5) * 100}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">{rating.helpful_rating?.toFixed(2) || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{rating.clarity_rating?.toFixed(2) || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap">{rating.difficulty_rating?.toFixed(2) || 'N/A'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(rating.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex flex-wrap gap-2">
                        {rating.is_online && (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                            Online
                          </span>
                        )}
                        {rating.is_for_credit && (
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            For Credit
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {/* Pagination Controls */}
          <div className="px-6 py-4 flex items-center justify-between border-t border-gray-200">
            <div className="flex-1 flex justify-between items-center">
              <div>
                <p className="text-sm text-gray-700">
                  Showing page <span className="font-medium">{currentPage}</span> of{' '}
                  <span className="font-medium">{totalPages}</span>
                </p>
                <p className="text-sm text-gray-700">
                  Total entries: <span className="font-medium">{paginatedData?.count || 0}</span>
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                  className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="h-5 w-5" />
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                  className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:bg-gray-100 disabled:cursor-not-allowed"
                >
                  Next
                  <ChevronRight className="h-5 w-5" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {selectedProfessorId && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Comment Analysis
              </h2>
              <CommentSummaryDisplay professorId={selectedProfessorId.toString()} />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Improvement Recommendations
              </h2>
              <RecommendationDisplay professorId={selectedProfessorId.toString()} />
            </div>

            <div className="lg:col-span-2 bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Topic Analysis
              </h2>
              <TopicModelVisualization professorId={selectedProfessorId.toString()} />
            </div>
          </div>
        )}

        {/* Admin Section */}
        {user?.role?.role === 'ADMIN' && !selectedProfessorId && (
          <>
            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Data Management</h2>
              <DataUploadForm />
            </div>

            <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Institution-wide Topic Analysis</h2>
              <TopicModelVisualization />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Evaluations;