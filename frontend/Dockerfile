# Use Node.js base image
FROM node:20-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies with legacy peer deps
RUN npm install --legacy-peer-deps

# Copy project files
COPY . .

# Build with explicit static file paths
ENV VITE_APP_API_URL=http://localhost:8000/api
ENV PUBLIC_URL=/static

# Build for production
CMD ["npm", "run", "build"]