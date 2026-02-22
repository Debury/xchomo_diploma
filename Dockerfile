FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="Climate Research Team"
LABEL description="Climate Data ETL Pipeline"

# Set working directory
WORKDIR /app

# Install Python 3.11, system dependencies, and Node.js for frontend build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libeccodes-dev \
    libeccodes0 \
    ca-certificates \
    curl \
    unzip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is available for python3.11
RUN python3.11 -m ensurepip --upgrade && python3.11 -m pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Install ONNX Runtime + Optimum separately to avoid dependency resolution conflicts with dagster
RUN python3.11 -m pip install --no-cache-dir "onnxruntime>=1.17.0" "optimum[onnxruntime]>=1.17.0"

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
RUN python3.11 -m pip install -e .

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
