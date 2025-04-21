import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Loader2, Search, AlertOctagon, ChevronUp, ChevronDown, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';
import { useState, useEffect, useMemo, useCallback } from 'react';
import { CommentSummaryDisplay } from '../components/CommentSummaryDisplay';
import { DataUploadForm } from '../components/DataUploadForm';
import { useAuth } from '../contexts/AuthContext';
import { useDebounce } from '../hooks/useDebounce';
import axios, { AxiosError } from 'axios'; // Import AxiosError for better typing

// --- Interface definitions remain the same ---
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
  comment: string | null;
}

interface PaginatedResponse {
  count: number;
  next: string | null;
  previous: string | null;
  results: Rating[];
}

// Helper to get CSRF token (ensure this is robust)
const getCsrfToken = () => {
  const name = 'csrftoken=';
  const decodedCookie = decodeURIComponent(document.cookie);
  const cookieArray = decodedCookie.split(';');
  for (let i = 0; i < cookieArray.length; i++) {
    let cookie = cookieArray[i].trim();
    if (cookie.indexOf(name) === 0) {
      return cookie.substring(name.length, cookie.length);
    }
  }
  console.warn('CSRF token not found in cookies.');
  return '';
};


function Evaluations() {
  const { user } = useAuth();
  const [selectedProfessorId, setSelectedProfessorId] = useState<number | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortConfig, setSortConfig] = useState<{ key: keyof Rating; direction: 'asc' | 'desc' }>({
    key: 'created_at',
    direction: 'desc'
  });
  const [searchTerm, setSearchTerm] = useState('');
  const [deleteConfirmation, setDeleteConfirmation] = useState<{showModal: boolean; ratingId: number | null}>({
    showModal: false,
    ratingId: null
  });
  const [isConfirmDeleteLoading, setIsConfirmDeleteLoading] = useState<boolean>(false); // Separate loading state for confirmation button
  const [bulkDeleteConfirmation, setBulkDeleteConfirmation] = useState<boolean>(false); // State for bulk delete modal
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set()); // State for selected row IDs
  const [successNotification, setSuccessNotification] = useState<string | null>(null); // Store success message
  const [rowsBeingDeleted, setRowsBeingDeleted] = useState<Set<number>>(new Set()); // Track rows being deleted
  const queryClient = useQueryClient();

  // Debounce the search term
  const debouncedSearchTerm = useDebounce(searchTerm, 400);

  // Querykeys now use debouncedSearchTerm instead of debouncedFilters
  const queryKey = ['ratings', currentPage, debouncedSearchTerm, sortConfig];

  const { data: paginatedData, isLoading, isFetching } = useQuery<PaginatedResponse, Error, PaginatedResponse>({
    queryKey: queryKey,
    queryFn: async () => {
      // Create params object with only necessary parameters
      const paramsObj: Record<string, string> = {
        page: currentPage.toString(),
        limit: '50', // Explicitly set limit
        // Only include search if debouncedSearchTerm is not empty
        ...(debouncedSearchTerm && { search: debouncedSearchTerm }),
        ...(sortConfig && { ordering: `${sortConfig.direction === 'desc' ? '-' : ''}${sortConfig.key}` })
      };

      // Filter out empty parameters to keep the URL clean
      const params = new URLSearchParams(
        Object.entries(paramsObj).filter(([_, value]) => value !== undefined && value !== '')
      );

      const { data } = await axios.get<PaginatedResponse>(`/api/ratings/?${params}`);
      return data;
    },
    placeholderData: previous => previous, // Modern equivalent of keepPreviousData
  });

  const showSuccess = (message: string) => {
    setSuccessNotification(message);
    setTimeout(() => {
      setSuccessNotification(null);
    }, 8000); // Increased from 3000ms to 8000ms (8 seconds) for better visibility
  }

// --- Single Delete Mutation ---
const deleteMutation = useMutation({
  mutationFn: async (ratingId: number) => {
    console.log("Attempting to delete rating ID:", ratingId);
    
    // Add this row ID to the set of rows being deleted
    setRowsBeingDeleted(prev => {
      const newSet = new Set(prev);
      newSet.add(ratingId);
      return newSet;
    });
    
    // Optimistic UI update - invalidate queries immediately to hide the row
    // This makes the UI feel instantaneous even if the backend is slow
    queryClient.invalidateQueries({ queryKey: ['ratings'] });
    queryClient.invalidateQueries({ queryKey: ['professors'] });
    queryClient.invalidateQueries({ queryKey: ['departmentStats'] });
    queryClient.invalidateQueries({ queryKey: ['sentiment-analysis'] });
    queryClient.invalidateQueries({ queryKey: ['discipline-stats'] });
    queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
    // Additional invalidations for professor-specific views to ensure they reflect deleted data
    queryClient.invalidateQueries({ queryKey: ['professorBiasAnalysis'] });
    queryClient.invalidateQueries({ queryKey: ['sentimentSummary'] });
    
    const csrfToken = getCsrfToken();
    if (!csrfToken) throw new Error("CSRF Token not found");
    
    // Simple axios request without timeout - let it complete naturally
    return axios.delete(`/api/ratings/${ratingId}/`, {
      headers: {
        'X-CSRFToken': csrfToken,
        'X-Requested-With': 'XMLHttpRequest'
      },
      withCredentials: true
    });
  },
  onSuccess: (_, variables) => {
    console.log("Successfully deleted rating ID:", variables);
    
    // Update selection state if the deleted row was selected
    setSelectedIds(prev => { 
        const newSet = new Set(prev);
        newSet.delete(variables);
        return newSet;
    });
    
    // Remove this ID from the rowsBeingDeleted set
    setRowsBeingDeleted(prev => {
      const newSet = new Set(prev);
      newSet.delete(variables);
      return newSet;
    });
    
    // Show success notification
    showSuccess(`Evaluation #${variables} deleted successfully.`);
    console.log("Delete success UI updated.");
    
    // With optimistic updates, we don't need to refresh data here
    // as we already invalidated the query in mutationFn
  },
  onError: (error: AxiosError | Error, variables) => {
    console.error('Error deleting evaluation ID:', variables, error);
    const axiosError = error as AxiosError;
    
    const errorMessage = (axiosError.response?.data as any)?.detail || 
                         (axiosError.response?.data as any)?.error || 
                         axiosError.message || 
                         'Failed to delete evaluation';
    
    // Remove this ID from the rowsBeingDeleted set
    setRowsBeingDeleted(prev => {
      const newSet = new Set(prev);
      newSet.delete(variables);
      return newSet;
    });
    
    // Show error notification
    alert(`Error deleting evaluation #${variables}: ${errorMessage}`);
    
    // Refresh data to restore the row if deletion actually failed on the backend
    // This ensures UI is consistent with the server state
    queryClient.invalidateQueries({ queryKey: ['ratings'] });
  }
});

  // --- Bulk Delete Mutation ---
  const bulkDeleteMutation = useMutation({
    mutationFn: async (ids: number[]) => {
      console.log("Attempting to bulk delete rating IDs:", ids);
      if (ids.length === 0) {
        console.warn("Bulk delete called with empty ID list.");
        return; // Or throw an error
      }
      const csrfToken = getCsrfToken();
      if (!csrfToken) throw new Error("CSRF Token not found");

      // *** IMPORTANT: Adjust this URL and body structure based on your backend API ***
      return axios.delete(`/api/ratings/bulk_delete/`, {
        headers: {
          'X-CSRFToken': csrfToken,
          'X-Requested-With': 'XMLHttpRequest',
          'Content-Type': 'application/json', // Important for sending JSON body
        },
        withCredentials: true,
        data: { ids: ids } // Send IDs in the request body
      });
    },
    onSuccess: (data, variables) => {
      console.log("Successfully bulk deleted rating IDs:", variables);
      queryClient.invalidateQueries({ queryKey: ['ratings'] });
      queryClient.invalidateQueries({ queryKey: ['professors'] });
      queryClient.invalidateQueries({ queryKey: ['departmentStats'] });
      queryClient.invalidateQueries({ queryKey: ['sentiment-analysis'] });
      queryClient.invalidateQueries({ queryKey: ['discipline-stats'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] }); // Add this to update the Dashboard count
      // Additional invalidations for professor-specific views
      queryClient.invalidateQueries({ queryKey: ['professorBiasAnalysis'] });
      queryClient.invalidateQueries({ queryKey: ['sentimentSummary'] });
      queryClient.invalidateQueries({ queryKey: [''] });
      console.log("Ratings query invalidated after bulk delete.");

      setBulkDeleteConfirmation(false); // Close bulk delete modal
      setSelectedIds(new Set()); // Clear selection
      showSuccess(`${variables.length} evaluation(s) deleted successfully.`);
      console.log("Bulk delete success UI updated.");
    },
    onError: (error: AxiosError | Error, variables) => {
      console.error('Error bulk deleting evaluations:', variables, error);
      const axiosError = error as AxiosError;
      const errorMessage = (axiosError.response?.data as any)?.detail || (axiosError.response?.data as any)?.error || axiosError.message || 'Failed to bulk delete evaluations';
      alert(`Error: ${errorMessage}`);
      // Ensure modal closes even on error
      setBulkDeleteConfirmation(false);
      console.log("Bulk delete error UI updated (modal closed).");
    }
  });


  const handleDeleteClick = (e: React.MouseEvent, ratingId: number) => {
    e.stopPropagation();
    setDeleteConfirmation({ showModal: true, ratingId });
  };

  const confirmDelete = () => {
    if (deleteConfirmation.ratingId !== null) {
        const ratingIdToDelete = deleteConfirmation.ratingId;
        // First set the confirmation button to loading state
        setIsConfirmDeleteLoading(true);
        
        // Then close the modal immediately
        setTimeout(() => {
          setDeleteConfirmation({ showModal: false, ratingId: null });
          // Reset the confirmation loading state
          setIsConfirmDeleteLoading(false);
          // Then trigger the mutation
          deleteMutation.mutate(ratingIdToDelete);
        }, 10); // Tiny delay to allow the spinner to be visible briefly
    } else {
       console.error("Confirm delete called with null ratingId");
       setDeleteConfirmation({ showModal: false, ratingId: null });
    }
  };

  const cancelDelete = () => {
    setDeleteConfirmation({ showModal: false, ratingId: null });
  };

  // --- Bulk Delete Handlers ---
  const handleBulkDeleteClick = () => {
    if (selectedIds.size > 0) {
      setBulkDeleteConfirmation(true);
    }
  };

  const confirmBulkDelete = () => {
    if (selectedIds.size > 0) {
        const idsToDelete = Array.from(selectedIds);
        // The actual mutation call triggers onSuccess/onError where state is handled
        bulkDeleteMutation.mutate(idsToDelete);
    } else {
        console.error("Confirm bulk delete called with no IDs selected.");
        setBulkDeleteConfirmation(false); // Close modal just in case
    }
  };

  const cancelBulkDelete = () => {
    setBulkDeleteConfirmation(false);
  };

  // --- Selection Handlers ---
  const handleSelectAll = (e: React.ChangeEvent<HTMLInputElement>) => {
    const isChecked = e.target.checked;
    if (isChecked) {
      // Select all IDs currently visible on the page
      const currentPageIds = ratings.map(r => r.id);
      setSelectedIds(new Set(currentPageIds));
    } else {
      setSelectedIds(new Set());
    }
  };


  // Reset selection when page changes or data reloads significantly
  useEffect(() => {
    setSelectedIds(new Set());
  }, [currentPage, debouncedSearchTerm, sortConfig, paginatedData?.results]); // Reset if data changes


  // Reset to first page when search or sort change
  useEffect(() => {
    setCurrentPage(1);
  }, [debouncedSearchTerm, sortConfig]);

  // Optimized memoized values
  const ratings = useMemo(() => paginatedData?.results || [], [paginatedData?.results]);
  const totalCount = useMemo(() => paginatedData?.count || 0, [paginatedData?.count]);
  const itemsPerPage = 50; // Match the 'limit' param used in useQuery
  const totalPages = useMemo(() => 
    totalCount > 0 ? Math.ceil(totalCount / itemsPerPage) : 1, 
    [totalCount]
  );

  const isAllSelected = useMemo(() => 
    ratings.length > 0 && 
    selectedIds.size === ratings.length && 
    ratings.every((r: Rating) => selectedIds.has(r.id)),
    [ratings, selectedIds]
  );

  // Effect to prefetch the next page when we're close to it
  useEffect(() => {
    // Only prefetch if we're not on the last page
    if (currentPage < totalPages) {
      const nextPageParams = new URLSearchParams({
        page: (currentPage + 1).toString(),
        limit: '50'
      });

      // Add only search and sort parameters to the prefetch query
      if (debouncedSearchTerm) nextPageParams.set('search', debouncedSearchTerm);
      if (sortConfig) nextPageParams.set('ordering', `${sortConfig.direction === 'desc' ? '-' : ''}${sortConfig.key}`);

      // Prefetch the next page
      queryClient.prefetchQuery({
        queryKey: ['ratings', currentPage + 1, debouncedSearchTerm, sortConfig],
        queryFn: async () => {
          const { data } = await axios.get<PaginatedResponse>(`/api/ratings/?${nextPageParams}`);
          return data;
        }
      });
    }
  }, [currentPage, debouncedSearchTerm, sortConfig, totalPages, queryClient]);

  // Optimized event handlers with useCallback
  const handleSort = useCallback((key: keyof Rating) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'asc' ? 'desc' : 'asc'
    }));
  }, []);

  const handleSelectRow = useCallback((e: React.ChangeEvent<HTMLInputElement>, id: number) => {
    const isChecked = e.target.checked;
    setSelectedIds(prev => {
      const newSet = new Set(prev);
      if (isChecked) {
        newSet.add(id);
      } else {
        newSet.delete(id);
      }
      return newSet;
    });
  }, []);

  const SortIndicator = useCallback(({ column }: { column: keyof Rating }) => {
    if (sortConfig.key !== column) return null;
    return sortConfig.direction === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />;
  }, [sortConfig]);

  // Use `isLoading` for initial load, `isFetching` for background updates
  const showInitialLoader = isLoading && !paginatedData;
  const showRefetchOverlay = isFetching && paginatedData;


  if (showInitialLoader) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="animate-spin text-indigo-600" size={32} />
      </div>
    );
  }

  return (
    <div className="space-y-6">
       {/* Success Notification */}
       {successNotification && (
        <div className="fixed top-4 right-4 bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded shadow-md z-[100]" role="alert"> {/* Increased z-index */}
          <div className="flex items-center">
             <svg className="fill-current h-6 w-6 text-green-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M2.93 17.07A10 10 0 1 1 17.07 2.93 10 10 0 0 1 2.93 17.07zm12.73-1.41A8 8 0 1 0 4.34 4.34a8 8 0 0 0 11.32 11.32zM9 11V9h2v6H9v-4zm0-6h2v2H9V5z"/></svg>
             <div>
                 <p className="font-bold">Success!</p>
                 <p className="text-sm">{successNotification}</p>
             </div>
          </div>
        </div>
      )}

      {/* --- Header Section --- */}
       <div className="flex flex-wrap justify-between items-center gap-4"> {/* Flex-wrap for responsiveness */}
        <h1 className="text-2xl font-bold text-gray-900">Course Evaluations</h1>
        <div className="flex items-center space-x-2 sm:space-x-4 flex-wrap gap-2"> {/* Flex-wrap + gap */}
           {/* Bulk Delete Button */}
           {selectedIds.size > 0 && (
              <button
                onClick={handleBulkDeleteClick}
                disabled={bulkDeleteMutation.isPending}
                className="flex items-center space-x-2 px-3 py-2 rounded-md bg-red-600 text-white hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {bulkDeleteMutation.isPending ? (
                    <Loader2 className="animate-spin h-5 w-5" />
                 ) : (
                    <Trash2 size={16} />
                 )}
                <span>Delete ({selectedIds.size})</span>
              </button>
            )}
          {/* Search Input */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              placeholder="Search..." // Shorter placeholder
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 w-48 sm:w-auto" // Responsive width
              title="Search by Professor Name, Discipline, or Sub-discipline" // Added tooltip
            />
          </div>
        </div>
      </div>

      <div className="grid gap-6">
        <div className="bg-white shadow-md rounded-lg overflow-hidden">
          <div className="overflow-x-auto relative">
            {/* Refetching Overlay */}
            {showRefetchOverlay && (
              <div className="absolute inset-0 bg-white bg-opacity-60 flex items-center justify-center z-10">
                <Loader2 className="animate-spin text-indigo-600" size={24} />
              </div>
            )}
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {/* Select All Checkbox Header */}
                  <th scope="col" className="px-4 py-3 text-center">
                    <input
                      type="checkbox"
                      className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                      checked={isAllSelected}
                      onChange={handleSelectAll}
                      aria-label="Select all evaluations on this page"
                      disabled={ratings.length === 0} // Disable if no ratings
                    />
                  </th>
                  {/* --- Other Headers --- */}
                  {[
                    { key: 'discipline', label: 'Discipline' },
                    { key: 'sub_discipline', label: 'Sub-discipline' },
                    { key: 'professor_name', label: 'Professor' },
                    { key: 'avg_rating', label: 'Overall Rating' },
                    { key: 'helpful_rating', label: 'Helpful' },
                    { key: 'clarity_rating', label: 'Clarity' },
                    { key: 'difficulty_rating', label: 'Difficulty' },
                    { key: 'comment', label: 'Comment' },
                    { key: 'created_at', label: 'Date' }
                  ].map(({ key, label }) => (
                    <th
                      key={key}
                      scope="col"
                      onClick={() => handleSort(key as keyof Rating)}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                       aria-sort={sortConfig.key === key ? (sortConfig.direction === 'asc' ? 'ascending' : 'descending') : 'none'}
                    >
                      <div className="flex items-center space-x-1">
                        <span>{label}</span>
                        <SortIndicator column={key as keyof Rating} />
                      </div>
                    </th>
                  ))}
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Type
                  </th>
                   <th scope="col" className="px-6 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {ratings.map((rating) => {
                   const isSelected = selectedIds.has(rating.id);
                   const isBeingDeleted = rowsBeingDeleted.has(rating.id);
                   return (
                      <tr
                        key={rating.id}
                        className={`transition-colors duration-150 ${
                          isSelected ? 'bg-indigo-50' : 'hover:bg-gray-50'
                        } ${isBeingDeleted ? 'opacity-50' : ''}`} // Dim row being deleted
                        // onClick={() => setSelectedProfessorId( // Allow row click even with checkbox
                        //  selectedProfessorId === rating.professor ? null : rating.professor
                        // )}
                        // aria-selected={isSelected} // Accessibility
                      >
                        {/* Row Checkbox */}
                         <td className="px-4 py-4 whitespace-nowrap text-center">
                           <input
                              type="checkbox"
                              className="h-4 w-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
                              checked={isSelected}
                              onChange={(e) => handleSelectRow(e, rating.id)}
                              onClick={(e) => e.stopPropagation()} // Prevent row click when clicking checkbox
                              aria-labelledby={`rating-professor-${rating.id}`}
                           />
                         </td>

                        {/* --- Other Data Cells (no changes needed) --- */}
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <span id={`rating-professor-${rating.id}`}>{rating.discipline}</span> {/* Added ID for aria-labelledby */}
                            {rating.flag_status && (
                              <div className="ml-2" title={rating.flag_status}>
                                <AlertOctagon size={16} className="text-yellow-500" />
                              </div>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">{rating.sub_discipline || <span className="text-gray-400">N/A</span>}</td>
                        <td className="px-6 py-4 whitespace-nowrap" onClick={(e) => { e.stopPropagation(); setSelectedProfessorId(selectedProfessorId === rating.professor ? null : rating.professor); }} style={{cursor: 'pointer'}}> {/* Make professor name clickable */}
                          <span className="hover:text-indigo-600 hover:underline">{rating.professor_name}</span>
                          {selectedProfessorId === rating.professor && <span className="ml-1 text-indigo-600 text-xs">(Selected)</span>}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="w-10 text-sm font-medium">{rating.avg_rating.toFixed(2)}</div>
                            <div className="ml-2 flex-1 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-indigo-600 rounded-full"
                                style={{ width: `${(rating.avg_rating / 5) * 100}%` }}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">{rating.helpful_rating?.toFixed(1) || <span className="text-gray-400">N/A</span>}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{rating.clarity_rating?.toFixed(1) || <span className="text-gray-400">N/A</span>}</td>
                        <td className="px-6 py-4 whitespace-nowrap">{rating.difficulty_rating?.toFixed(1) || <span className="text-gray-400">N/A</span>}</td>
                        <td className="px-6 py-4 max-w-xs">
                          <div className="truncate text-sm" title={rating.comment || ""}>
                            {rating.comment || <span className="text-gray-400">No comment</span>}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {new Date(rating.created_at).toLocaleDateString()}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex flex-col sm:flex-row flex-wrap gap-1">
                            {rating.is_online && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800">
                                Online
                              </span>
                            )}
                             {!rating.is_online && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700">
                                In-Person
                              </span>
                            )}
                            {rating.is_for_credit && (
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
                                For Credit
                              </span>
                            )}
                            {!rating.is_for_credit && (
                               <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                                Not Credit
                              </span>
                            )}
                          </div>
                        </td>
                        {/* Single Delete Button */}
                         <td className="px-6 py-4 whitespace-nowrap text-center">
                          <button
                            onClick={(e) => handleDeleteClick(e, rating.id)}
                            className={`text-red-600 hover:text-red-900 disabled:text-gray-400 disabled:cursor-not-allowed p-1 rounded hover:bg-red-100 transition-colors duration-150 ${isBeingDeleted ? 'opacity-0 cursor-default' : ''}`} // Hide button when deleting this row
                            disabled={isBeingDeleted || bulkDeleteMutation.isPending} // Only disable this button if this specific row is being deleted or bulk delete is happening
                            aria-label={`Delete evaluation by ${rating.professor_name}`}
                          >
                             <Trash2 size={16} />
                          </button>
                           {/* Show spinner in place of button when deleting this specific row */}
                           {isBeingDeleted && (
                             <div className="flex justify-center items-center h-full">
                               <Loader2 className="animate-spin h-4 w-4 text-red-600" />
                             </div>
                           )}
                        </td>
                      </tr>
                   );
                })}
                 {ratings.length === 0 && !isFetching && ( // Show only if not fetching and no ratings
                    <tr>
                        <td colSpan={11} className="text-center py-10 text-gray-500"> {/* Adjusted colspan */}
                            No evaluations found matching your criteria.
                        </td>
                    </tr>
                 )}
              </tbody>
            </table>
          </div>

          {/* --- Pagination Controls (no changes needed) --- */}
           <div className="px-6 py-4 flex items-center justify-between border-t border-gray-200 bg-gray-50">
            <div className="flex-1 flex justify-between items-center sm:flex-row flex-col gap-4">
              <div>
                <p className="text-sm text-gray-700">
                   Page <span className="font-medium">{currentPage}</span> of{' '}
                  <span className="font-medium">{totalPages}</span>.
                   Total: <span className="font-medium">{totalCount}</span> entries.
                   {selectedIds.size > 0 && ` (${selectedIds.size} selected)`}
                </p>
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1 || totalPages === 0 || isFetching}
                  className="relative inline-flex items-center px-3 py-1.5 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <ChevronLeft className="h-5 w-5 mr-1" />
                  Previous
                </button>
                <button
                  onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages || totalPages === 0 || isFetching}
                  className="relative inline-flex items-center px-3 py-1.5 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                  <ChevronRight className="h-5 w-5 ml-1" />
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* --- Selected Professor Details and Admin Section (no changes needed) --- */}
        {selectedProfessorId && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in"> {/* Added animation */}
            {/* ... Comment Analysis ... */}
            {/* ... Improvement Recommendations ... */}
             <div className="bg-white shadow-md rounded-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">
                Comment Analysis
              </h2>
              <CommentSummaryDisplay professorId={selectedProfessorId.toString()} />
            </div>
          </div>
        )}
        {user?.role?.role === 'ADMIN' && !selectedProfessorId && (
          <div className="bg-white shadow-md rounded-lg p-6 animate-fade-in"> {/* Added animation */}
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Data Management</h2>
            <DataUploadForm />
          </div>
        )}
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirmation.showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]" 
            aria-labelledby="modal-title" 
            role="dialog" 
            aria-modal="true">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-sm w-full mx-4 transform transition-all duration-300 scale-100"  
              onClick={(e) => e.stopPropagation()}>
            <h3 id="modal-title" className="text-lg font-semibold mb-4 text-gray-900">Confirm Deletion</h3>
            <p className="mb-6 text-gray-600">Are you sure you want to delete this evaluation? This action cannot be undone.</p>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={cancelDelete}
                disabled={deleteMutation.isPending}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmDelete}
                disabled={isConfirmDeleteLoading}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center min-w-[80px]"
              >
                {isConfirmDeleteLoading ? (
                  <Loader2 className="animate-spin h-5 w-5 text-white" />
                ) : (
                  'Delete'
                )}
              </button>
            </div>
          </div>
        </div>
      )}

       {/* --- Bulk Delete Confirmation Modal --- */}
      {bulkDeleteConfirmation && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[70]" aria-labelledby="bulk-modal-title" role="dialog" aria-modal="true">
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-sm w-full mx-4 transform transition-all duration-300 scale-100" onClick={(e) => e.stopPropagation()}>
            <h3 id="bulk-modal-title" className="text-lg font-semibold mb-4 text-gray-900">Confirm Bulk Deletion</h3>
            <p className="mb-6 text-gray-600">Are you sure you want to delete the selected <span className='font-medium'>{selectedIds.size}</span> evaluation(s)? This action cannot be undone.</p>
            <div className="flex justify-end space-x-3">
              <button
                type="button"
                onClick={cancelBulkDelete}
                disabled={bulkDeleteMutation.isPending}
                className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-opacity-50 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmBulkDelete}
                disabled={bulkDeleteMutation.isPending}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center min-w-[120px]" // Wider button
              >
                {bulkDeleteMutation.isPending ? (
                   <Loader2 className="animate-spin h-5 w-5 text-white" />
                ) : (
                   `Delete ${selectedIds.size} Item(s)`
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Evaluations;