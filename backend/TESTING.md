# Testing the Enhanced LSTM with Attention for Gender Bias Analysis

This document provides instructions for running tests on the enhanced LSTM model with attention mechanisms for gender bias analysis.

## Prerequisites

- Python 3.8+
- Django 4.0+
- PyTorch 1.12+
- Node.js 14+
- npm 6+

## Running Backend Tests

### 1. Unit Tests for LSTM Attention Model

```bash
docker-compose exec backend python -m machine_learning.tests.run_tests

### 2. API tests 
docker-compose exec backend python manage.py test api.tests.test_sentiment_explainability_api

### 3. End-to-end tests
docker-compose exec backend python manage.py test e2e_tests.test_gender_bias_analysis
docker-compose exec backend python manage.py test api.tests data_processing.tests machine_learning.tests e2e_tests 

## Running Frontend Tests
docker-compose exec frontend npm test -- --testPathPattern=src/components/__tests__/BiasExplainer.test.tsx

docker-compose exec frontend npm test -- --testPathPattern=src/components/__tests__/ProfessorBiasDashboard.test.tsx


## Test Coverage
docker-compose exec backend coverage run --source='.' manage.py test

docker-compose exec backend coverage report

docker-compose exec backend coverage html  # Generates an HTML report in htmlcov/

docker-compose exec frontend npm test -- --coverage

