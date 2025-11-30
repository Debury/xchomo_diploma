# Climate Data RAG Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready climate data RAG (Retrieval-Augmented Generation) system with memory-safe multi-format processing, Ollama LLM integration, and vector search.

## 📋 Overview

**Core Features:**
- **Memory-Safe Raster Pipeline**: Streams NetCDF/GRIB/HDF5/GeoTIFF/CSV without loading entire files into RAM
- **LLM-Powered RAG**: Ollama integration for intelligent climate Q&A
- **Dynamic Source Management**: Web UI for adding/managing climate data sources
- **Vector Search**: Qdrant for semantic search over climate embeddings
- **Orchestration**: Dagster for automated ETL workflows
- **REST API**: FastAPI with interactive docs

**Test Coverage**: 21% (focused on raster pipeline)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Dagit UI (3000)                        │
│               Dagster Workflow Orchestration                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Dynamic Source ETL Job                         │
│  1. Download climate file from source URL                   │
│  2. Auto-detect format (NetCDF/GRIB/HDF5/GeoTIFF/CSV)       │
│  3. Stream raster in memory-safe chunks                     │
│  4. Generate statistical embeddings                         │
│  5. Store in Qdrant vector DB                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────┐    ┌──────────────────┐    ┌────────────┐
│  Raster Pipeline │───▶│  Text Embedder   │───▶│   Qdrant   │
│  (Memory-Safe)   │    │  (MiniLM-L6-v2)  │    │  (Vectors) │
└──────────────────┘    └──────────────────┘    └────────────┘
                                                       │
                                                       ▼
┌─────────────────────────────────────────────────────────────┐
│               FastAPI Web Service (8000)                    │
│  • /rag/chat - Q&A with Ollama LLM + vector context        │
│  • /sources - Manage climate data sources (CRUD)           │
│  • /docs - Interactive API documentation                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     ┌────────────────┐
                     │  Ollama (3b)   │
                     │  LLM Service   │
                     └────────────────┘
```

## 📁 Project Structure

```
xchomo_diploma/
├── src/
│   ├── embeddings/
│   │   └── raster_pipeline.py      # Memory-safe multi-format loader
│   ├── llm/
│   │   └── ollama_client.py        # LLM integration
│   ├── sources/
│   │   └── SourceManager.py        # Dynamic source CRUD
│   └── utils/
├── dagster_project/
│   ├── dynamic_jobs.py             # Source-driven ETL job
│   ├── repository.py
│   └── workspace.yaml
├── web_api/
│   └── main.py                     # FastAPI endpoints
├── tests/
│   ├── test_raster_pipeline_flow.py   # Multi-format tests
│   └── test_web_api.py
├── config/
│   ├── pipeline_config.yaml
│   └── era5_config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── EMBEDDING_REFACTORING.md
│   └── FLEXIBLE_EMBEDDINGS_QUICKSTART.md
├── docker-compose.yml              # 7 services
└── requirements.txt
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB RAM (4GB minimum for small datasets)

### Installation

```bash
cd xchomo_diploma

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start all services
docker-compose up -d

# Wait for Ollama to pull model (first time only, ~2min)
docker-compose logs -f ollama
```

### Access Points

- **Dagit UI**: http://localhost:3000
- **REST API**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📖 Usage

### 1. Add a Climate Data Source

```bash
curl -X POST http://localhost:8000/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_id": "gistemp_global",
    "url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "format": "csv",
    "description": "NASA GISTEMP global temperature anomalies",
    "tags": ["temperature", "global"],
    "variables": ["Jan", "Feb", "Mar"]
  }'
```

### 2. Trigger ETL Job

Go to **Dagit UI** (http://localhost:3000) and run `dynamic_source_etl_job`:
- Auto-detects format (NetCDF/GRIB/HDF5/GeoTIFF/CSV)
- Streams data in chunks (no OOM errors)
- Generates embeddings and stores in Qdrant

### 3. Query with RAG

```bash
curl -X POST http://localhost:8000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the global temperature trend?",
    "use_llm": true,
    "top_k": 5
  }'
```

Response:
```json
{
  "answer": "Based on GISTEMP data, global temperature anomalies show an upward trend of approximately 0.8°C since 1880...",
  "sources": ["gistemp_global"],
  "llm_used": true
}
```

### 4. Test Multi-Format Pipeline

```bash
# Run comprehensive format tests (NetCDF, GeoTIFF, CSV)
docker-compose exec web-api pytest tests/test_raster_pipeline_flow.py -vv

# Check coverage
docker-compose exec web-api pytest --cov=src.embeddings.raster_pipeline
```

## 🔑 Key Components

### Raster Pipeline (`src/embeddings/raster_pipeline.py`)

**Purpose**: Memory-safe loading and embedding generation for climate rasters

**Supported Formats**:
- NetCDF (`.nc`, `.nc4`) - uses xarray+dask chunks
- GRIB (`.grib`, `.grib2`) - via cfgrib
- HDF5 (`.h5`, `.hdf5`) - h5netcdf backend
- GeoTIFF (`.tif`, `.tiff`) - rasterio windowed reading
- ASCII Grid (`.asc`) - rasterio
- CSV (`.csv`) - pandas chunked reading
- Zarr (`.zarr`) - xarray chunks

**Key Functions**:
```python
# Auto-detect format and load in chunks
data_iter = load_raster_auto(
    file_path="temperature_2024.nc",
    max_chunk_size=1000,  # Max 1000 cells per chunk
    variables=["temperature"]
)

# Generate embeddings from chunks
embeddings = raster_to_embeddings(
    data_iter,
    normalization="zscore",  # or "minmax"
    pooling_strategy="mean"
)
```

**Memory Safety**:
- Never loads full rasters into RAM
- Auto-reduces chunk size on `MemoryError`
- Yields chunks one at a time (generator pattern)

### Ollama Client (`src/llm/ollama_client.py`)

**Purpose**: LLM integration for RAG answer generation

**Features**:
- Health checks and model auto-pull
- Climate-specific system prompt
- Context injection from vector search
- Fallback to template-based answers
- 120s timeout for model loading

**Usage**:
```python
from src.llm.ollama_client import OllamaClient

client = OllamaClient(base_url="http://ollama:11434")

# Generate RAG answer
answer = client.generate_rag_answer(
    query="What is the temperature trend?",
    context_hits=[...],  # Top-k results from Qdrant
    temperature=0.7
)
```

### Dynamic Jobs (`dagster_project/dynamic_jobs.py`)

**Process**:
1. Fetch climate file from source URL
2. Auto-detect format (no manual specification)
3. Load raster in memory-safe chunks
4. Extract statistics (mean/std/min/max/percentiles)
5. Convert stats to text descriptions
6. Generate semantic embeddings (MiniLM-L6-v2)
7. Store in Qdrant with metadata

**No OOM errors**: Uses chunked loading throughout pipeline

## 🧪 Testing

```bash
# Run all tests
docker-compose exec web-api pytest -v

# Multi-format pipeline tests
docker-compose exec web-api pytest tests/test_raster_pipeline_flow.py -vv

# RAG endpoint tests
docker-compose exec web-api pytest tests/test_web_api.py -k rag

# Coverage report
docker-compose exec web-api pytest --cov=src --cov-report=html
```

**Test Files**:
- `test_raster_pipeline_flow.py` - NetCDF, GeoTIFF, CSV auto-detection
- `test_web_api.py` - RAG endpoint, source management
- `test_embeddings.py` - Embedding generation

## 🐳 Docker Services

```yaml
services:
  dagit:       # Dagster UI (3000)
  dagster-daemon:  # Workflow scheduler
  web-api:     # FastAPI (8000)
  postgres:    # Dagster storage
  qdrant:      # Vector DB (6333/6334)
  ollama:      # LLM service (11434)
```

## 📊 Sample Datasets

The curated Excel registry `Kopie souboru D1.1.xlsx` lists climate sources:
- **ERA5**: European reanalysis (1940-present)
- **GISTEMP**: NASA global temperature anomalies
- **CERRA**: European regional reanalysis
- **Open-Meteo**: Air quality API

All sources include access URLs, coverage metadata, and format info.

## 🔧 Configuration

### Pipeline Config (`config/pipeline_config.yaml`)

```yaml
raster_pipeline:
  max_chunk_size: 1000
  normalization: "zscore"
  pooling_strategy: "mean"

embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  vector_dim: 384

qdrant:
  collection_name: "climate_embeddings"
  distance: "Cosine"
```

### Ollama Config (docker-compose.yml)

```yaml
ollama:
  image: ollama/ollama:latest
  environment:
    OLLAMA_MODEL: "llama3.2:3b"
  ports:
    - "11434:11434"
```

## 🐛 Troubleshooting

### OOM Errors on 4GB VM

**Solution**: Use smaller test files or upgrade to 8GB RAM

```bash
# Generate small test NetCDF (10x10x10 grid)
docker-compose exec web-api python -c "
import xarray as xr
import numpy as np
data = xr.Dataset({
    'temperature': (['time', 'lat', 'lon'], np.random.randn(10, 10, 10))
}).to_netcdf('/app/data/raw/small_test.nc')
"
```

### Ollama Timeout

**Symptom**: First request times out (model loading)

**Solution**: Increased timeout to 120s + warmup command in docker-compose:

```yaml
ollama:
  command: >
    sh -c "ollama serve & 
           sleep 10 && 
           ollama run llama3.2:3b 'Hello' &&
           wait"
```

### Format Detection Fails

**Check**: `load_raster_auto()` logs detection attempts:

```bash
docker-compose logs web-api | grep "Trying format"
```

**Supported extensions**: `.nc`, `.nc4`, `.grib`, `.grib2`, `.h5`, `.hdf5`, `.tif`, `.tiff`, `.asc`, `.csv`, `.zarr`

## 📝 License

MIT License - see [LICENSE](LICENSE) file

## 👤 Author

Climate Data RAG Pipeline - Thesis Project

## 📧 Contact

For issues, open a GitHub issue or contact the author.

---

**Status**: ✅ Production Ready  
**Version**: 5.0.0  
**Last Updated**: January 2025
