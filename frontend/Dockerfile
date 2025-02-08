# Use Node.js base image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy project files
COPY . .

# Set environment variables
ENV VITE_APP_API_URL=http://localhost:8000/api

# Start development server (using Vite)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]