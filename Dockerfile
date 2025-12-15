FROM python:3.11-slim

LABEL maintainer="Climate Research Team"
LABEL description="Climate Data ETL Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for frontend build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libeccodes-dev \
    libeccodes0 \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy and build Vue frontend
COPY web_api/frontend/package*.json ./web_api/frontend/
WORKDIR /app/web_api/frontend
RUN npm install

COPY web_api/frontend/ ./
RUN npm run build

# Save the dist folder
RUN cp -r dist /tmp/frontend-dist

# Back to app root
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
COPY README.md .
COPY web_api/ ./web_api/
COPY dagster_project/ ./dagster_project/

# Restore the built frontend
RUN cp -r /tmp/frontend-dist ./web_api/frontend/dist

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DATA_RAW_DIR=/app/data/raw
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV LOGS_DIR=/app/logs

# Volume mounts
VOLUME ["/app/data", "/app/logs", "/app/.env"]

# Default command
CMD ["python", "-m", "src.data_transformation.pipeline", "--help"]
