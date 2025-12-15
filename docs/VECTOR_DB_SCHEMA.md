# Vector Database Schema Documentation

## Overview

This document describes the normalized, structured schema for storing climate data chunks in the vector database (Qdrant). The schema is designed to be:

- **Universal**: Works for any dataset (ERA5, NOAA GSOM, NetCDF, GeoTIFF, CSV, etc.)
- **Filterable**: All fields are structured and can be filtered/queried efficiently
- **RAG-friendly**: Metadata is structured for easy LLM consumption
- **Scalable**: Optimized for large collections with efficient indexing

## Schema Structure

### ClimateChunkMetadata

The core metadata structure stored in the vector database:

```python
{
    # Dataset identification
    "dataset_name": "Sample Data",           # Human-readable dataset name
    "source_id": "Sample Data",              # Unique source identifier
    "variable": "TMIN",                      # Variable name (e.g., "TMAX", "TMIN", "HTDD")
    
    # Variable metadata (for dynamic LLM understanding)
    "long_name": "Minimum temperature",     # Human-readable variable name
    "standard_name": "air_temperature",      # CF standard name
    "unit": "°F",                           # Unit of measurement
    
    # Temporal information
    "time_start": "2016-01-01",             # ISO format start time
    "time_end": "2016-01-31",               # ISO format end time
    "temporal_frequency": "daily",           # e.g., "daily", "monthly", "hourly"
    
    # Spatial information (bounding box)
    "latitude_min": 35.43,
    "latitude_max": 35.43,
    "longitude_min": -82.54,
    "longitude_max": -82.54,
    
    # Statistics (computed from chunk data)
    "stats_mean": 35.25,                     # Mean value
    "stats_std": 7.63,                      # Standard deviation
    "stats_min": 25.00,                     # Minimum value
    "stats_max": 44.00,                     # Maximum value
    "stats_p10": 28.00,                     # 10th percentile
    "stats_median": 35.00,                  # Median
    "stats_p90": 42.00,                     # 90th percentile
    "stats_range": 19.00,                   # Range (max - min)
    
    # Grid/spatial metadata
    "grid_shape": [1, 1],                   # [height, width] or [n_lat, n_lon]
    "resolution_deg": 0.1,                  # Spatial resolution in degrees
    
    # File/chunk identification
    "file_path": "datasets/sample/2016_01.csv",
    "chunk_index": 1,
    "row_count": 31                         # For CSV/time-series data
}
```

## Key Design Decisions

### 1. Structured Metadata (Not Text Strings)

**Before (Bad):**
```
"text_content": "Climate variable: Htdd (HTDD) | Dataset: Sample Data | Time: 2016-01 | Latitude: 35.43° to 35.43° | Longitude: -82.54° to -82.54° | Mean: 317.00 | Standard deviation: 150.08 | Range: 144.00 to 517.00"
```

**After (Good):**
```json
{
    "dataset_name": "Sample Data",
    "variable": "HTDD",
    "latitude_min": 35.43,
    "latitude_max": 35.43,
    "longitude_min": -82.54,
    "longitude_max": -82.54,
    "stats_mean": 317.00,
    "stats_std": 150.08,
    "stats_min": 144.00,
    "stats_max": 517.00
}
```

**Benefits:**
- ✅ Filterable by any field (dataset, variable, time, location, stats)
- ✅ Efficient indexing in vector DB
- ✅ LLM can easily parse structured data
- ✅ No parsing errors from text strings
- ✅ Scalable to millions of chunks

### 2. Dynamic Text Generation

Text descriptions are **NOT stored** in the database. Instead, they are generated dynamically from structured metadata when needed:

1. **For Embedding Generation**: Text is generated from metadata → embedded → stored as vector
2. **For LLM Context**: Text is generated from metadata → formatted for LLM

This keeps the database clean and ensures consistency.

### 3. Normalized Field Names

All fields use consistent naming:
- `latitude_min` / `latitude_max` (not `lat_min` / `lat_max` or `min_lat` / `max_lat`)
- `longitude_min` / `longitude_max` (not `lon_min` / `lon_max` or `min_lon` / `max_lon`)
- `time_start` / `time_end` (not `start_date` / `end_date`)
- `stats_mean`, `stats_std`, `stats_min`, `stats_max` (flattened from nested `stats` dict)

### 4. Flattened Statistics

Statistics are flattened for Qdrant compatibility:
- `stats_mean` instead of `stats.mean`
- `stats_std` instead of `stats.std`
- etc.

## Usage Examples

### Creating Normalized Metadata

```python
from src.climate_embeddings.schema import ClimateChunkMetadata

# From raw chunk metadata
raw_meta = {
    "variable": "TMIN",
    "lat_min": 35.43,
    "lon_min": -82.54,
    "time_start": "2016-01-01",
    "unit": "°F"
}

stats_vector = [35.25, 7.63, 25.00, 44.00, 28.00, 35.00, 42.00, 19.00]

normalized = ClimateChunkMetadata.from_chunk_metadata(
    raw_metadata=raw_meta,
    stats_vector=stats_vector,
    source_id="Sample Data",
    dataset_name="Sample Data"
)

# Convert to dict for storage
payload = normalized.to_dict()
```

### Generating Human-Readable Text

```python
from src.climate_embeddings.schema import generate_human_readable_text

# Generate text for embedding or LLM
text = generate_human_readable_text(payload, verbosity="medium")
# Output: "Climate variable: Minimum temperature (TMIN) | Dataset: Sample Data | Time: 2016-01-01 | Geographic coordinates: Latitude: 35.43°N Longitude: -82.54°E | Mean: 35.25 °F | Value range: 25.00 to 44.00 °F"
```

### Filtering in Qdrant

```python
# Filter by dataset
filter_dict = {"dataset_name": "Sample Data"}

# Filter by variable
filter_dict = {"variable": "TMIN"}

# Filter by time range
filter_dict = {
    "time_start": {"$gte": "2016-01-01"},
    "time_end": {"$lte": "2016-12-31"}
}

# Filter by location (bounding box)
filter_dict = {
    "latitude_min": {"$gte": 35.0},
    "latitude_max": {"$lte": 36.0},
    "longitude_min": {"$gte": -83.0},
    "longitude_max": {"$lte": -82.0}
}

# Combined filters
filter_dict = {
    "dataset_name": "Sample Data",
    "variable": "TMIN",
    "stats_mean": {"$gte": 30.0}
}
```

## Migration from Old Schema

If you have existing data with `text_content` stored:

1. **Backward Compatibility**: The code still handles `text_content` if present
2. **Re-indexing**: For best results, re-process your data sources to use the new schema
3. **Gradual Migration**: New chunks use the new schema, old chunks still work

## Benefits Summary

| Aspect | Old (Text String) | New (Structured) |
|--------|-------------------|------------------|
| **Filtering** | ❌ Not possible | ✅ Filter by any field |
| **Indexing** | ❌ Text search only | ✅ Indexed fields |
| **LLM Parsing** | ❌ Error-prone | ✅ Structured, easy to parse |
| **Scalability** | ❌ Poor | ✅ Excellent |
| **Consistency** | ❌ Format varies | ✅ Normalized schema |
| **Query Performance** | ❌ Slow | ✅ Fast with indexes |

## Future Enhancements

- [ ] Add payload schema validation in Qdrant
- [ ] Support for nested metadata (if Qdrant version supports it)
- [ ] Automatic schema migration tools
- [ ] Schema versioning for backward compatibility


