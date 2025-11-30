# Climate Embeddings Package

Clean, modular structure for climate data RAG with powerful embedding models.

## 📁 Structure

```
src/climate_embeddings/
├── loaders/              # Data loading for all formats
│   ├── detect_format.py  # Auto-detect file format
│   ├── raster_pipeline.py # Memory-safe raster loading
│   └── __init__.py
│
├── embeddings/           # Embedding models
│   ├── text_embeddings.py # BGE, GTE, MPNet models
│   └── __init__.py
│
├── index/                # Vector search
│   ├── vector_index.py   # In-memory index with filtering
│   └── __init__.py
│
├── rag/                  # RAG pipeline
│   ├── rag_pipeline.py   # Query → Retrieve → Generate
│   └── __init__.py
│
├── io/                   # Save/load utilities
│   ├── metadata.py
│   └── __init__.py
│
├── config/               # Configuration
│   ├── defaults.py
│   └── __init__.py
│
├── cli/                  # Command-line interface
│   ├── main.py
│   └── __init__.py
│
└── __init__.py          # Main package exports
```

## 🚀 Quick Start

### Generate Embeddings

```bash
python -m climate_embeddings.cli.main generate data.nc -o embeddings.jsonl
```

### Build Index

```bash
python -m climate_embeddings.cli.main build-index embeddings.jsonl --dim 384 -o index.pkl
```

### Ask Question

```bash
python -m climate_embeddings.cli.main ask "What is the temperature trend?" --index index.pkl
```

## 💻 Python API

```python
from climate_embeddings import (
    load_raster_auto,
    raster_to_embeddings,
    get_text_embedder,
    VectorIndex,
    RAGPipeline,
)

# Load and embed climate data
result = load_raster_auto("temperature.nc")
embeddings = raster_to_embeddings(result)

# Build index
index = VectorIndex(dim=384)
text_embedder = get_text_embedder("bge-large")

for emb in embeddings:
    index.add(emb["vector"], emb["metadata"])

# RAG pipeline
from src.llm.ollama_client import OllamaClient
rag = RAGPipeline(index, text_embedder, OllamaClient())

answer = rag.ask("What is the global temperature anomaly?")
print(answer)
```

## 📊 Supported Formats

- **NetCDF** (.nc, .nc4) - xarray + dask chunks
- **GRIB** (.grib, .grb2) - cfgrib engine
- **HDF5** (.h5, .hdf5) - h5netcdf
- **GeoTIFF** (.tif) - rasterio windows
- **ASCII Grid** (.asc) - rasterio
- **CSV** (.csv) - pandas chunks
- **Zarr** - dask arrays
- **ZIP** - auto-extract and load

## 🤖 Text Embedding Models

- `bge-large` - BAAI/bge-large-en-v1.5 (1024-dim, SOTA)
- `gte-large` - Alibaba-NLP/gte-large (1024-dim)
- `mpnet` - all-mpnet-base-v2 (768-dim)
- `minilm` - all-MiniLM-L6-v2 (384-dim, fast)

## 🧪 Testing

```bash
# Test all components
pytest tests/test_rag_components.py -v

# Test raster pipeline
pytest tests/test_raster_pipeline_flow.py -v
```

## 🔄 Integration

This package is integrated with:
- **Dagster** (`dagster_project/`) - workflow orchestration
- **FastAPI** (`web_api/`) - REST API
- **Qdrant** - vector database (via existing VectorDatabase)
- **Ollama** (`src/llm/`) - LLM for answer generation
