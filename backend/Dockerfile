# Use PyTorch GPU base image
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install python-Levenshtein to remove fuzzywuzzy warning
RUN pip install python-Levenshtein

# Create non-root user earlier
RUN useradd -m faireval
RUN mkdir -p /home/faireval/nltk_data
RUN chown -R faireval:faireval /home/faireval/nltk_data

# Switch to non-root user before downloading NLTK data
USER faireval

# Download NLTK data to user's home directory
RUN python -c "import nltk; nltk.data.path.append('/home/faireval/nltk_data'); nltk.download(['vader_lexicon', 'opinion_lexicon', 'punkt', 'stopwords', 'averaged_perceptron_tagger'], download_dir='/home/faireval/nltk_data')"

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Switch back to root for remaining setup
USER root

# Create volume mount points with proper permissions
RUN mkdir -p /app/staticfiles /app/frontend/dist && \
    chmod -R 755 /app/staticfiles /app/frontend/dist

# Copy project
COPY . .

# Set final ownership and user
RUN chown -R faireval:faireval /app
USER faireval

# Command to run on container start
CMD ["gunicorn", "faireval.wsgi:application", "--bind", "0.0.0.0:8000"]