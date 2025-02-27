import { render, screen, waitFor } from '@testing-library/react';
import { ProfessorBiasDashboard } from '../ProfessorBiasDashboard';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock lucide-react icons
jest.mock('lucide-react', () => ({
  Loader2: () => <div data-testid="loader-icon" />,
  AlertCircle: () => <div data-testid="alert-icon" />,
  Info: () => <div data-testid="info-icon" />,
  Lightbulb: () => <div data-testid="lightbulb-icon" />,
}));

// Create a fresh query client for each test
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      gcTime: 0,
    },
  },
});

// Wrap component with providers needed for testing
const renderWithClient = (ui: React.ReactElement) => {
  const testQueryClient = createTestQueryClient();
  return {
    ...render(
      <QueryClientProvider client={testQueryClient}>
        {ui}
      </QueryClientProvider>
    ),
  };
};

describe('ProfessorBiasDashboard Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders loading state initially', () => {
    // Mock the API response with a delay to ensure we see loading state
    mockedAxios.get.mockImplementationOnce(() => 
      new Promise(resolve => setTimeout(() => resolve({ data: {} }), 100))
    );
    
    renderWithClient(<ProfessorBiasDashboard professorId="testprof123" />);
    
    // Check loading state
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('displays professor bias analysis results', async () => {
    // Mock the API response for bias analysis
    mockedAxios.get.mockImplementationOnce(() => Promise.resolve({
      data: {
        professor_id: 'testprof123',
        discipline: 'Computer Science',
        analysis_results: {
          overall_bias_score: 0.32,
          positive_comments_bias_score: 0.38,
          negative_comments_bias_score: 0.12,
          comment_count: 50,
          positive_count: 35,
          negative_count: 15,
          top_male_terms: [['he', 0.25], ['his', 0.18], ['male', 0.12]],
          top_female_terms: [['she', 0.05], ['her', 0.03]]
        },
        recommendations: [
          {
            text: "Student feedback shows potential bias toward male-associated language.",
            priority: 'high',
            impact_score: 8.5,
            supporting_evidence: ['he', 'his', 'male']
          }
        ]
      }
    }));
    
    // Mock the API response for discipline gender data
    mockedAxios.get.mockImplementationOnce(() => Promise.resolve({
      data: [
        { discipline: 'Computer Science', gender: 'Male', avg_rating: 4.2 },
        { discipline: 'Computer Science', gender: 'Female', avg_rating: 3.9 }
      ]
    }));
    
    renderWithClient(<ProfessorBiasDashboard professorId="testprof123" />);
    
    // Wait for results to load
    await waitFor(() => {
      const summary = screen.getByText('Gender Bias Analysis Summary');
      expect(summary).toBeInTheDocument();
      
      // Check specific sections for values
      const overallBiasSection = summary.parentElement?.querySelector('.grid-cols-1.md\\:grid-cols-3')?.children[0];
      expect(overallBiasSection).not.toBeUndefined();
      expect(overallBiasSection).toHaveTextContent('0.32');
      expect(overallBiasSection).toHaveTextContent(/significant male-associated language bias/i);

      // Check other sections exist
      expect(screen.getByText('Male-Associated Terms')).toBeInTheDocument();
      expect(screen.getByText('AI-Generated Recommendations')).toBeInTheDocument();
      expect(screen.getByText(/Student feedback shows potential bias/i)).toBeInTheDocument();

      // Verify specific terms appear
      expect(screen.getByText('he')).toBeInTheDocument();
      expect(screen.getByText('his')).toBeInTheDocument();
      expect(screen.getByText('male')).toBeInTheDocument();
    });
  });
  
  it('handles error state correctly', async () => {
    // Mock an API error
    mockedAxios.get.mockRejectedValueOnce(new Error('Network error'));
    
    renderWithClient(<ProfessorBiasDashboard professorId="testprof123" />);
    
    // Wait for error state
    await waitFor(() => {
      expect(screen.getByText(/Error analyzing comments/i)).toBeInTheDocument();
      expect(screen.getByText(/Network error/i)).toBeInTheDocument();
    });
  });
});