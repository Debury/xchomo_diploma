FROM python:3.11-slim

LABEL maintainer="Climate Research Team"
LABEL description="Climate Data ETL Pipeline"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_RAW_DIR=/app/data/raw
ENV DATA_PROCESSED_DIR=/app/data/processed
ENV LOGS_DIR=/app/logs

# Volume mounts
VOLUME ["/app/data", "/app/logs", "/app/.env"]

# Default command
CMD ["python", "-m", "src.data_transformation.pipeline", "--help"]
