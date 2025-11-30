"""Configuration defaults."""

# Text embedding models
DEFAULT_TEXT_MODEL = "BAAI/bge-large-en-v1.5"
AVAILABLE_MODELS = {
    "bge-large": "BAAI/bge-large-en-v1.5",
    "gte-large": "Alibaba-NLP/gte-large",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# Raster processing
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_NORMALIZATION = "zscore"

# Vector search
DEFAULT_TOP_K = 5
DEFAULT_METRIC = "cosine"

# RAG
DEFAULT_TEMPERATURE = 0.7

# Spatial chunking targets (in pixels)
DEFAULT_CHUNK_SHAPE = {"y": 256, "x": 256}

# Memory safety limit for chunks (MB)
CHUNK_MEMORY_LIMIT_MB = 64