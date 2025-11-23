# System Architecture

## Overview

The Climate Data ETL Pipeline is designed as a modular, scalable system for processing climate data from various sources.

## Architecture Diagram (All 4 Phases)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Climate Data ETL Pipeline - Complete System            │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────────────┐
                              │   Web App Interface      │
                              │  (User submits URL)      │
                              └────────────┬─────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   Data Sources (URLs)    │
                              │  • ERA5 reanalysis       │
                              │  • CMIP6 climate models  │
                              │  • Weather stations      │
                              │  • Air pollution data    │
                              └────────────┬─────────────┘
                                           │
                      ┌────────────────────▼───────────────┐
                      │  PHASE 1: Data Acquisition         │
                      │  • Download from URL               │
                      │  • Auto-detect format              │
                      │  • Validate data                   │
                      └────────────────┬───────────────────┘
                                       │
                          ┌────────────▼─────────┐
                          │  Raw Data Storage    │
                          │    (data/raw/)       │
                          └────────────┬─────────┘
                                       │
                      ┌────────────────▼───────────────┐
                      │ PHASE 2: Transformation        │
                      │  • Ingestion (multi-format)    │
                      │  • Metadata extraction         │
                      │  • Statistics computation      │
                      │  • Validation                  │
                      └────────────────┬───────────────┘
                                       │
                     ┌─────────────────▼──────────────┐
                     │  Processed Data Storage        │
                     │   (data/processed/)            │
                     └─────────────────┬──────────────┘
                                       │
                      ┌────────────────▼───────────────┐
                      │  PHASE 3: Embeddings           │
                      │  • Text generation             │
                      │  • Embedding generation        │
                      │  • Qdrant vector storage       │
                      │  • Semantic search ready       │
                      └────────────────┬───────────────┘
                                       │
                          ┌────────────▼─────────┐
                          │  Vector Database │
                          │    (ChromaDB)    │
                          └──────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                 PHASE 4: Orchestration & Web UI                         │
│                                                                           │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐   │
│  │  Dagster Daemon │    │    Dagit UI      │    │   FastAPI       │   │
│  │  • Schedules    │    │  • DAG Viz       │    │  • 6 Endpoints  │   │
│  │  • Sensors      │    │  • Run History   │    │  • OpenAPI Docs │   │
│  │  • Job Queue    │    │  • Monitoring    │    │  • CORS         │   │
│  └────────┬────────┘    └────────┬─────────┘    └────────┬────────┘   │
│           │                      │                        │             │
│           └──────────────────────┼────────────────────────┘             │
│                                  │                                       │
│                     ┌────────────▼────────────┐                         │
│                     │  Dagster Repository     │                         │
│                     │  • 8 Ops                │                         │
│                     │  • 4 Jobs               │                         │
│                     │  • 3 Schedules          │                         │
│                     │  • 3 Sensors            │                         │
│                     └─────────────────────────┘                         │
│                                                                           │
│         Orchestrates: Phases 1, 2, 3 (Full ETL + Embeddings)            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Acquisition Module (`src/data_acquisition/`) - PHASE 1

**Purpose**: Download and fetch climate data from external sources

**Components**:
- `era5_downloader.py`: ERA5 data retrieval from Copernicus CDS
- `visualizer.py`: Data visualization and quality checks

**Key Features**:
- API authentication and connection management
- Retry logic for failed downloads
- Progress tracking
- Automatic data validation

### 2. Data Transformation Module (`src/data_transformation/`) - PHASE 2

**Purpose**: Process, transform, and standardize climate data

**Components**:
- `ingestion.py`: Multi-format data loading (NetCDF, CSV, JSON, GeoTIFF)
- `transformations.py`: Data transformations (unit conversion, aggregation, normalization)
- `export.py`: Multi-format export with compression
- `pipeline.py`: Orchestration and workflow management

**Processing Stages**:
1. **Ingestion**: Load data from multiple formats
2. **Standardization**: Rename dimensions and variables
3. **Transformation**: Apply unit conversions and aggregations
4. **Normalization**: Scale data for analysis
5. **Validation**: Quality checks and range verification
6. **Export**: Save to multiple formats

### 3. Embeddings Module (`src/embeddings/`) - PHASE 3

**Purpose**: Generate vector embeddings for semantic search

**Components**:
- `generator.py`: Embedding generation using sentence transformers
- `database.py`: ChromaDB integration and management
- `pipeline.py`: Embedding pipeline orchestration
- `search.py`: Semantic search functionality

**Key Features**:
- Sentence transformer models (all-MiniLM-L6-v2)
- ChromaDB persistent storage
- Batch processing for efficiency
- Similarity search with cosine distance
- Metadata filtering capabilities

### 4. Orchestration Layer (`dagster_project/`) - PHASE 4

**Purpose**: Workflow orchestration and automation

**Components**:
- `ops/`: 8 Dagster operations
  - `data_acquisition_ops.py`: Download and validation
  - `transformation_ops.py`: Ingestion, transformation, export
  - `embedding_ops.py`: Embedding generation and search
- `jobs.py`: 4 complete workflow jobs
  - `daily_etl_job`: Daily data acquisition + transformation
  - `embedding_job`: Embedding generation pipeline
  - `complete_pipeline_job`: End-to-end all phases
  - `validation_job`: Data quality checks
- `schedules.py`: Automated execution
  - 3 Schedules: Daily ETL (2am), Daily embeddings (4am), Weekly complete (Sunday 3am)
  - 3 Sensors: New data detector, quality checker, config monitor
- `resources.py`: Configurable resources (Logger, ConfigLoader, DataPaths)
- `repository.py`: Dagster definitions

**Key Features**:
- DAG-based workflow visualization
- Automated scheduling with cron
- Event-driven sensors
- Run history and monitoring
- Resource dependency management

### 5. REST API Service (`web_api/`) - PHASE 4

**Purpose**: HTTP API for external system integration

**Components**:
- `main.py`: FastAPI application

**Endpoints**:
- `GET /`: API information
- `GET /health`: Health check with Dagster connectivity
- `GET /jobs`: List available jobs
- `POST /jobs/{name}/run`: Trigger job execution
- `GET /runs/{run_id}/status`: Get run status
- `GET /runs`: List recent runs

**Key Features**:
- OpenAPI/Swagger documentation
- CORS middleware
- Pydantic data validation
- Async request handling
- GraphQL integration with Dagster

### 6. Utilities Module (`src/utils/`)

**Purpose**: Shared utilities and helpers

**Components**:
- `logger.py`: Centralized logging configuration
- `config_loader.py`: Configuration management with environment variables

## Data Flow

### Manual Execution Flow

1. **Phase 1 - Download**:
   ```
   CDS API → ERA5 Downloader → Raw Data (NetCDF) → data/raw/
   ```

2. **Phase 2 - Transformation**:
   ```
   data/raw/ → Ingestion → Transformations → Validation → Export → data/processed/
   ```

3. **Phase 3 - Embeddings**:
   ```
   data/processed/ → Embedding Generator → ChromaDB → Vector Storage
   ```

### Orchestrated Execution Flow (Phase 4)

1. **Automated Daily ETL Job** (runs at 2 AM):
   ```
   Dagster Schedule → download_era5_data op → validate_downloaded_data op
                   → ingest_data op → transform_data op → export_data op
   ```

2. **Automated Embedding Job** (runs at 4 AM):
   ```
   Dagster Schedule → generate_embeddings op → store_embeddings op → test_semantic_search op
   ```

3. **Sensor-Driven Flow**:
   ```
   New file in data/processed/ → new_processed_data_sensor triggers
                                → embedding_job executes automatically
   ```

4. **API-Triggered Flow**:
   ```
   HTTP POST /jobs/daily_etl_job/run → Dagster GraphQL → Job Execution
                                     → GET /runs/{run_id}/status for monitoring
   ```

### Output Formats

- **NetCDF**: For scientific analysis (compressed)
- **Parquet**: For efficient querying and analytics
- **CSV**: For simple data inspection
- **Vector Embeddings**: In ChromaDB for semantic search

## Configuration Management

The system uses a hierarchical configuration approach:

1. **Default Configuration**: `config/pipeline_config.yaml`
2. **Environment Variables**: `.env` file
3. **Runtime Parameters**: Command-line arguments

Priority: Runtime > Environment > Configuration File > Defaults

## Error Handling

- **Retry Logic**: Automatic retry for transient failures
- **Graceful Degradation**: Continue processing other files if one fails
- **Comprehensive Logging**: All errors logged with context
- **Validation Checks**: Data quality verification at each stage

## Scalability

- **Parallel Processing**: Multiple files processed concurrently
- **Memory Management**: Streaming for large datasets
- **Chunking**: Process data in manageable chunks
- **Caching**: Intermediate results cached for efficiency

## Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmark critical operations
- **Data Validation Tests**: Verify output quality

## Deployment

### Local Development
- **Direct Python**: `make run-all` for phases 1-3
- **Dagit UI**: `make dagit` for orchestration interface
- **FastAPI**: `make api` for REST service

### Docker Deployment
- **Single Container**: `make docker-run` (ETL only)
- **Multi-Service**: `make dagster-all` (Full stack)
  - PostgreSQL database
  - Dagster daemon (schedules/sensors)
  - Dagit UI (port 3000)
  - FastAPI service (port 8000)
  - Climate ETL pipeline

### Cloud Deployment
- **AWS**: ECS/EKS with RDS PostgreSQL
- **GCP**: Cloud Run with Cloud SQL
- **Azure**: Container Instances with Azure Database
- **CI/CD**: GitHub Actions for automated testing and deployment

## Security

- **API Keys**: Stored in environment variables
- **Data Access**: File system permissions
- **Logging**: No sensitive data in logs
- **Dependencies**: Regular security updates
- **CORS**: Configured for FastAPI endpoints
- **Authentication**: Ready for OAuth2/JWT integration

## Technology Stack

### Core ETL (Phases 1-3)
- **Python 3.8+**: Core language
- **xarray**: N-dimensional labeled arrays
- **pandas**: Data manipulation
- **netCDF4**: NetCDF file handling
- **cdsapi**: ERA5 data access
- **sentence-transformers**: Embedding generation
- **chromadb**: Vector database

### Orchestration & API (Phase 4)
- **Dagster Core**: Workflow orchestration
- **Dagit**: Web UI for DAG visualization
- **FastAPI**: REST API framework
- **uvicorn**: ASGI server
- **Pydantic**: Data validation
- **PostgreSQL**: Run history storage
- **Docker Compose**: Multi-service orchestration

## Key Metrics

- **8 Dagster Ops**: Modular workflow components
- **4 Complete Jobs**: End-to-end pipelines
- **3 Schedules**: Automated daily/weekly execution
- **3 Sensors**: Event-driven automation
- **6 API Endpoints**: REST interface
- **100+ Tests**: Comprehensive test coverage
- **4 Phases Complete**: Full production system
