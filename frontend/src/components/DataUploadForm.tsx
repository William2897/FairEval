import React, { useState, useRef } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Upload, X, Check, AlertCircle } from 'lucide-react';
import axios from 'axios';

interface UploadResponse {
  success: boolean;
  message: string;
  recordsProcessed?: number;
}

export const DataUploadForm: React.FC = () => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const { data } = await axios.post<UploadResponse>('/api/admin/upload-evaluations', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return data;
    },
    onSuccess: (data) => {
      // Invalidate relevant queries to trigger refetch
      queryClient.invalidateQueries({ queryKey: ['evaluations'] });
      queryClient.invalidateQueries({ queryKey: ['dashboard-stats'] });
      queryClient.invalidateQueries({ queryKey: ['commentSummary'] });
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
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (selectedFile) {
      uploadMutation.mutate(selectedFile);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">Upload Evaluation Data</h2>
      
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

      {uploadMutation.isSuccess && (
        <div className="mt-4 p-3 bg-green-50 text-green-700 rounded-md flex items-center space-x-2">
          <Check size={16} />
          <span>
            Upload successful! {uploadMutation.data.recordsProcessed} records processed.
            Dashboard data will refresh automatically.
          </span>
        </div>
      )}

      {/* Upload Button */}
      <div className="mt-6 flex justify-end">
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
    </div>
  );
}; 