import React, { useState, useRef } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Upload, X, Check, AlertCircle, Eye, ArrowRight } from 'lucide-react';
import axios from 'axios';
import Papa from 'papaparse';
import { ParseResult, ParseError } from 'papaparse';

// Helper function to get CSRF token from cookies
const getCSRFToken = (): string => {
  const name = 'csrftoken=';
  const decodedCookie = decodeURIComponent(document.cookie);
  const cookieArray = decodedCookie.split(';');
  
  for (let i = 0; i < cookieArray.length; i++) {
    let cookie = cookieArray[i].trim();
    if (cookie.indexOf(name) === 0) {
      return cookie.substring(name.length, cookie.length);
    }
  }
  return '';
};

interface UploadResponse {
  message: string;
  task_id: string;
  success?: boolean;
  recordsProcessed?: number;
  errors?: string[];
}

interface CSVPreviewData {
  headers: string[];
  rows: Array<Record<string, string>>;
}

export const DataUploadForm: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed' | 'error'>('idle');
  const [processedRecords, setProcessedRecords] = useState<number>(0);
  const [previewData, setPreviewData] = useState<CSVPreviewData | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  // Check the status of a processing task
  const checkTaskStatus = async (taskId: string) => {
    try {
      const { data } = await axios.get(`/api/ratings/upload_status/?task_id=${taskId}`);
      
      if (data.status === 'SUCCESS') {
        setProcessingStatus('completed');
        // If the result contains processed records information
        if (data.result && typeof data.result === 'object') {
          // Try to access the count from processed_records first (from process_evaluation_data_task)
          // then fallback to processed (from process_rating_upload)
          setProcessedRecords(data.result.processed_records || data.result.processed || 0);
        }        // Refresh relevant data
        queryClient.invalidateQueries({ queryKey: ['ratings'] });
        queryClient.invalidateQueries({ queryKey: ['professors'] });
        queryClient.invalidateQueries({ queryKey: ['departmentStats'] });
        queryClient.invalidateQueries({ queryKey: ['sentiment-analysis'] });
        queryClient.invalidateQueries({ queryKey: ['discipline-stats'] });
        queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] }); // Add this to update the Dashboard count
      } else if (data.status === 'FAILURE') {
        setProcessingStatus('error');
      } else {
        // Still processing, check again in a few seconds
        setTimeout(() => checkTaskStatus(taskId), 3000);
      }
    } catch (error) {
      console.error('Error checking task status:', error);
      setProcessingStatus('error');
    }
  };

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const csrfToken = getCSRFToken();
      
      const { data } = await axios.post<UploadResponse>('/api/ratings/upload/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'X-CSRFToken': csrfToken,
        },
        withCredentials: true,
      });
      return data;
    },
    onSuccess: (data) => {
      if (data.task_id) {
        setProcessingStatus('processing');
        // Start checking the task status
        checkTaskStatus(data.task_id);
      }
    },
  });

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const file = e.dataTransfer.files?.[0];
    if (file && file.type === 'text/csv') {
      setSelectedFile(file);
      parseCSVPreview(file);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      parseCSVPreview(file);
    }
  };

  const parseCSVPreview = (file: File) => {
    Papa.parse(file, {
      header: true,
      preview: 5, // Show only the first 5 rows
      skipEmptyLines: true,
      complete: (results: ParseResult<Record<string, string>>) => {
        if (results.data && results.data.length > 0) {
          const headers = results.meta.fields || [];
          setPreviewData({
            headers,
            rows: results.data
          });
        }
      },
      error: (error: ParseError) => {
        console.error('Error parsing CSV:', error);
        setPreviewData(null);
      }
    });
  };

  const validateCSV = (file: File): boolean => {
    // Only allow .csv files
    if (!file.name.endsWith('.csv')) {
      alert('Please upload a CSV file');
      return false;
    }
    return true;
  };

  const handlePreview = () => {
    setShowPreview(true);
  };

  const handleUpload = async () => {
    if (selectedFile && validateCSV(selectedFile)) {
      setShowPreview(false);
      uploadMutation.mutate(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewData(null);
    setShowPreview(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-900">Upload Evaluation Data</h2>
        <div className="text-sm text-gray-500">
          <a href="/api/ratings/template" className="text-indigo-600 hover:text-indigo-800">
            Download Template
          </a>
        </div>
      </div>
      
      {!showPreview ? (
        <>
          <div
            className={`relative border-2 border-dashed rounded-lg p-6 transition-colors
              ${dragActive ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300'}
              ${uploadMutation.isError ? 'border-red-300' : ''}
              ${uploadMutation.isSuccess ? 'border-green-300' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
            />

            <div className="text-center">
              <Upload 
                className={`mx-auto h-12 w-12 ${
                  dragActive ? 'text-indigo-500' : 'text-gray-400'
                }`}
              />
              
              <p className="mt-2 text-sm text-gray-600">
                Drag and drop your CSV file here, or{' '}
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="text-indigo-600 hover:text-indigo-500 font-medium"
                  type="button"
                >
                  browse
                </button>
              </p>
              <p className="mt-1 text-xs text-gray-500">
                Only CSV files are supported
              </p>
            </div>

            {selectedFile && (
              <div className="mt-4 flex items-center justify-between bg-gray-50 p-3 rounded-md">
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">{selectedFile.name}</span>
                  <span className="text-xs text-gray-500">
                    ({(selectedFile.size / 1024).toFixed(1)} KB)
                  </span>
                </div>
                <button
                  onClick={clearFile}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <X size={16} />
                </button>
              </div>
            )}
          </div>

          {/* Status Messages */}
          {uploadMutation.isError && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md flex items-center space-x-2">
              <AlertCircle size={16} />
              <span>Upload failed: {uploadMutation.error?.message}</span>
            </div>
          )}

          {processingStatus === 'processing' && (
            <div className="mt-4 p-3 bg-yellow-50 text-yellow-700 rounded-md flex items-center space-x-2">
              <Upload className="animate-spin" size={16} />
              <span>Processing uploaded data...</span>
            </div>
          )}

          {processingStatus === 'completed' && (
            <div className="mt-4 p-3 bg-green-50 text-green-700 rounded-md flex items-center space-x-2">
              <Check size={16} />
              <span>Processing completed! {processedRecords} evaluations processed.</span>
            </div>
          )}

          {processingStatus === 'error' && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md flex items-center space-x-2">
              <AlertCircle size={16} />
              <span>Processing failed. Please try again.</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="mt-6 flex justify-end space-x-3">
            {selectedFile && previewData && (
              <button
                onClick={handlePreview}
                className="flex items-center space-x-2 px-4 py-2 rounded-md text-indigo-600 border border-indigo-600 hover:bg-indigo-50"
              >
                <Eye size={16} />
                <span>Preview Data</span>
              </button>
            )}
            
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploadMutation.isPending}
              className={`
                flex items-center space-x-2 px-4 py-2 rounded-md text-white
                ${!selectedFile || uploadMutation.isPending
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700'
                }
              `}
            >
              {uploadMutation.isPending ? (
                <>
                  <Upload className="animate-spin" size={16} />
                  <span>Uploading...</span>
                </>
              ) : (
                <>
                  <Upload size={16} />
                  <span>Upload</span>
                </>
              )}
            </button>
          </div>
        </>
      ) : (
        <div className="mt-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Data Preview</h3>
            <span className="text-sm text-gray-500">Showing first 5 rows</span>
          </div>
          
          <div className="overflow-x-auto max-h-96 border rounded-lg">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50 sticky top-0">
                <tr>
                  {previewData?.headers.map((header, index) => (
                    <th 
                      key={index}
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {previewData?.rows.map((row, rowIndex) => (
                  <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    {previewData.headers.map((header, cellIndex) => (
                      <td 
                        key={`${rowIndex}-${cellIndex}`}
                        className="px-6 py-4 whitespace-nowrap text-sm text-gray-500"
                      >
                        {row[header]}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="mt-6 flex justify-end space-x-3">
            <button
              onClick={() => setShowPreview(false)}
              className="flex items-center space-x-2 px-4 py-2 rounded-md text-gray-600 border border-gray-300 hover:bg-gray-50"
            >
              <X size={16} />
              <span>Back</span>
            </button>
            
            <button
              onClick={handleUpload}
              className="flex items-center space-x-2 px-4 py-2 rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              <ArrowRight size={16} />
              <span>Proceed to Upload</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};