import '@testing-library/jest-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BiasExplainer } from '../BiasExplainer';
import axios from 'axios';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Create a query client for testing
const createTestQueryClient = () => new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
    },
  },
});

describe('BiasExplainer Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    jest.clearAllMocks();
  });

  test('renders the bias explainer component correctly', () => {
    const queryClient = createTestQueryClient();
    
    render(
      <QueryClientProvider client={queryClient}>
        <BiasExplainer />
      </QueryClientProvider>
    );
    
    // Check if the component renders
    expect(screen.getByText('Gender Bias Analysis in Comments')).toBeInTheDocument();
    expect(screen.getByLabelText(/enter a comment to analyze for gender bias/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/academic discipline/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyze comment/i })).toBeInTheDocument();
  });

  test('fetches and displays disciplines correctly', async () => {
    // Mock the API response for disciplines
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        discipline_ratings: [
          { discipline: 'Computer Science' },
          { discipline: 'Physics' },
          { discipline: 'Mathematics' }
        ]
      }
    });
    
    const queryClient = createTestQueryClient();
    
    render(
      <QueryClientProvider client={queryClient}>
        <BiasExplainer />
      </QueryClientProvider>
    );
    
    // Wait for disciplines to load
    await waitFor(() => {
      const selectElement = screen.getByLabelText(/academic discipline/i) as HTMLSelectElement;
      expect(selectElement.options.length).toBeGreaterThan(1);
      expect(selectElement.options[1].text).toBe('Computer Science');
    });
  });

  test('submits comment for analysis and displays results', async () => {
    // Mock the API response for disciplines
    mockedAxios.get.mockResolvedValueOnce({
      data: {
        discipline_ratings: [
          { discipline: 'Computer Science' },
          { discipline: 'Physics' }
        ]
      }
    });
    
    // Mock the API response for analysis
    mockedAxios.post.mockResolvedValueOnce({
      data: {
        prediction: 'Positive',
        confidence: 0.92,
        tokens: ['the', 'professor', 'is', 'great', 'he', 'explains', 'well'],
        attention: [0.1, 0.2, 0.05, 0.15, 0.3, 0.1, 0.1],
        gender_bias: {
          bias_score: 0.4,
          male_attention_total: 0.3,
          female_attention_total: 0,
          neutral_attention_total: 0.7,
          male_attention_pct: 30,
          female_attention_pct: 0,
          neutral_attention_pct: 70,
          male_terms: [[4, 'he', 0.3]],
          female_terms: [],
          top_male_terms: [[4, 'he', 0.3]],
          top_female_terms: [],
          top_neutral_terms: [[1, 'professor', 0.2]]
        }
      }
    });
    
    const queryClient = createTestQueryClient();
    
    render(
      <QueryClientProvider client={queryClient}>
        <BiasExplainer />
      </QueryClientProvider>
    );
    
    // Enter a comment
    const commentInput = screen.getByLabelText(/enter a comment to analyze for gender bias/i);
    fireEvent.change(commentInput, { target: { value: 'The professor is great. He explains well.' } });
    
    // Submit for analysis
    const analyzeButton = screen.getByRole('button', { name: /analyze comment/i });
    fireEvent.click(analyzeButton);
    
    // Wait for results to display
    await waitFor(() => {
      expect(screen.getByText('Male Terms')).toBeInTheDocument();
      expect(screen.getByText('30.0%')).toBeInTheDocument(); // Male attention percentage
      expect(screen.getByText('Strong male-biased language')).toBeInTheDocument();
    });
  });
});