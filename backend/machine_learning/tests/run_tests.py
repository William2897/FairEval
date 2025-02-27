import unittest
import sys
import os

# Add parent directory to path so tests can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test modules
from .test_lstm_attention import TestLSTMAttention

if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add tests
    test_suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestLSTMAttention))
    
    # Run tests
    result = unittest.TextTestRunner().run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())