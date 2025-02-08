FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Run as non-root user
RUN useradd -m faireval
RUN chown -R faireval:faireval /app
USER faireval

# Command to run on container start
CMD ["gunicorn", "faireval.wsgi:application", "--bind", "0.0.0.0:8000"]