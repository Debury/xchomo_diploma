"""
Normalized schema for climate data chunks in vector database.

This module defines the standard structure for storing climate data chunks
in vector databases (Qdrant, Weaviate, etc.) with structured, filterable metadata.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ClimateChunkMetadata:
    """
    Normalized metadata schema for climate data chunks.
    
    This schema is universal and works for any dataset (ERA5, NOAA GSOM, NetCDF, GeoTIFF, etc.)
    All fields are optional to support diverse data sources.
    """
    # Dataset identification
    dataset_name: str  # e.g., "Sample Data", "ERA5", "NOAA GSOM"
    source_id: str  # Unique source identifier
    variable: str  # Variable name (e.g., "TMAX", "TMIN", "HTDD", "2m_temperature")
    
    # Variable metadata (for dynamic LLM understanding)
    long_name: Optional[str] = None  # Human-readable variable name
    standard_name: Optional[str] = None  # CF standard name
    unit: Optional[str] = None  # Unit of measurement
    
    # Temporal information
    time_start: Optional[str] = None  # ISO format: "2016-01-01"
    time_end: Optional[str] = None  # ISO format: "2016-01-31"
    temporal_frequency: Optional[str] = None  # e.g., "daily", "monthly", "hourly"
    
    # Spatial information (bounding box)
    latitude_min: Optional[float] = None
    latitude_max: Optional[float] = None
    longitude_min: Optional[float] = None
    longitude_max: Optional[float] = None
    
    # Statistics (computed from chunk data)
    stats: Optional[Dict[str, float]] = None  # {"mean": 317.0, "std": 150.08, "min": 144.0, "max": 517.0, ...}
    
    # Grid/spatial metadata
    grid_shape: Optional[list] = None  # [height, width] or [n_lat, n_lon]
    resolution_deg: Optional[float] = None  # Spatial resolution in degrees
    
    # File/chunk identification
    file_path: Optional[str] = None
    chunk_index: Optional[int] = None
    row_count: Optional[int] = None  # For CSV/time-series data
    
    # Additional metadata (flexible for dataset-specific fields)
    additional_metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing None values for cleaner storage."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "stats" and isinstance(value, dict):
                    # Flatten stats for Qdrant (nested dicts not well supported)
                    for stat_key, stat_val in value.items():
                        result[f"stats_{stat_key}"] = stat_val
                elif key == "additional_metadata" and isinstance(value, dict):
                    # Merge additional metadata into main dict
                    result.update(value)
                else:
                    result[key] = value
        return result
    
    @classmethod
    def from_chunk_metadata(
        cls,
        raw_metadata: Dict[str, Any],
        stats_vector: Optional[list] = None,
        source_id: str = "unknown",
        dataset_name: Optional[str] = None
    ) -> "ClimateChunkMetadata":
        """
        Create normalized metadata from raw chunk metadata.
        
        Args:
            raw_metadata: Raw metadata from chunk (from raster_pipeline)
            stats_vector: Statistical vector [mean, std, min, max, p10, median, p90, range]
            source_id: Source identifier
            dataset_name: Dataset name (falls back to source_id if not provided)
        """
        # Extract stats
        stats = None
        if stats_vector and len(stats_vector) >= 8:
            stats = {
                "mean": float(stats_vector[0]),
                "std": float(stats_vector[1]),
                "min": float(stats_vector[2]),
                "max": float(stats_vector[3]),
                "p10": float(stats_vector[4]),
                "median": float(stats_vector[5]),
                "p90": float(stats_vector[6]),
                "range": float(stats_vector[7])
            }
        
        # Normalize spatial fields (handle both lat_min/lat_max and min_lat/max_lat)
        lat_min = raw_metadata.get("lat_min") or raw_metadata.get("min_lat")
        lat_max = raw_metadata.get("lat_max") or raw_metadata.get("max_lat")
        lon_min = raw_metadata.get("lon_min") or raw_metadata.get("min_lon")
        lon_max = raw_metadata.get("lon_max") or raw_metadata.get("max_lon")
        
        # Normalize time fields
        time_start = raw_metadata.get("time_start") or raw_metadata.get("start_date")
        time_end = raw_metadata.get("time_end") or raw_metadata.get("end_date")
        
        # Extract grid shape if available
        grid_shape = None
        if "shape" in raw_metadata:
            try:
                if isinstance(raw_metadata["shape"], (list, tuple)):
                    grid_shape = list(raw_metadata["shape"])
                elif isinstance(raw_metadata["shape"], str):
                    # Try to parse string representation
                    import ast
                    grid_shape = ast.literal_eval(raw_metadata["shape"])
            except:
                pass
        
        # Collect additional metadata (fields not in standard schema)
        # This preserves ALL custom metadata from any source dynamically
        standard_fields = {
            "dataset_name", "source_id", "variable", "long_name", "standard_name", "unit", "units",
            "time_start", "time_end", "temporal_frequency", "lat_min", "lat_max", "lon_min", "lon_max",
            "min_lat", "max_lat", "min_lon", "max_lon", "start_date", "end_date",
            "file_path", "chunk_index", "row_count", "shape", "format", "source",
            "station_id", "station_name", "station_names", "station_count",
            "description", "title", "institution", "source_description"
        }
        additional = {}
        for k, v in raw_metadata.items():
            if k not in standard_fields and v is not None:
                # Handle nested dicts (like additional_attrs, geotiff_tags)
                if isinstance(v, dict):
                    for nested_k, nested_v in v.items():
                        additional[f"{k}_{nested_k}"] = nested_v
                else:
                    additional[k] = v
        
        return cls(
            dataset_name=dataset_name or source_id,
            source_id=source_id,
            variable=raw_metadata.get("variable", "unknown"),
            long_name=raw_metadata.get("long_name"),
            standard_name=raw_metadata.get("standard_name"),
            unit=raw_metadata.get("unit") or raw_metadata.get("units"),
            time_start=str(time_start) if time_start else None,
            time_end=str(time_end) if time_end else None,
            temporal_frequency=raw_metadata.get("temporal_frequency"),
            latitude_min=float(lat_min) if lat_min is not None else None,
            latitude_max=float(lat_max) if lat_max is not None else None,
            longitude_min=float(lon_min) if lon_min is not None else None,
            longitude_max=float(lon_max) if lon_max is not None else None,
            stats=stats,
            grid_shape=grid_shape,
            resolution_deg=raw_metadata.get("resolution_deg") or raw_metadata.get("resolution"),
            file_path=raw_metadata.get("file_path") or raw_metadata.get("source"),
            chunk_index=raw_metadata.get("chunk_index"),
            row_count=raw_metadata.get("row_count"),
            additional_metadata=additional if additional else None
        )


def generate_human_readable_text(metadata: Dict[str, Any], verbosity: str = "medium") -> str:
    """
    Generate human-readable text description from structured metadata.
    OPTIMIZED FOR PRECISION: Provides detailed, accurate information for expert users.
    
    This is used for:
    1. Embedding generation (text embedding model needs text input)
    2. LLM context formatting (after retrieval)
    
    This text is NOT stored in the database - it's generated dynamically.
    
    Args:
        metadata: Structured metadata dict (from ClimateChunkMetadata.to_dict())
        verbosity: "low", "medium", or "high" - controls detail level
    
    Returns:
        Human-readable text description optimized for RAG with precise data for experts
    """
    parts = []
    
    # Variable information (most important for RAG)
    variable = metadata.get("variable", "unknown")
    long_name = metadata.get("long_name")
    standard_name = metadata.get("standard_name")
    unit = metadata.get("unit", "")
    
    # DYNAMIC: Build variable description ONLY from available metadata
    # NO HARDCODED MAPPINGS - rely on what's in the data itself
    var_desc_parts = []
    
    # Use long_name if available (from dataset metadata)
    if long_name:
        var_desc_parts.append(long_name)
    
    # Use standard_name if available (CF convention standard names)
    if standard_name and standard_name != long_name:
        var_desc_parts.append(standard_name)
    
    # Build final variable description - ONLY use what's in metadata
    if var_desc_parts:
        var_desc = ", ".join(var_desc_parts)
        parts.append(f"Climate variable: {var_desc} ({variable})")
        parts.append(f"Variable code: {variable} | Meaning: {var_desc_parts[0]}")
    elif long_name:
        parts.append(f"Climate variable: {long_name} ({variable})")
        parts.append(f"Variable code: {variable} | Meaning: {long_name}")
    else:
        # If no metadata available, just use variable name
        # LLM during RAG query will infer meaning from context
        parts.append(f"Climate variable: {variable}")
    
    if standard_name and verbosity in ["medium", "high"]:
        parts.append(f"Standard name: {standard_name}")
    
    if unit:
        parts.append(f"Unit: {unit}")
    
    # Dataset
    dataset = metadata.get("dataset_name") or metadata.get("source_id", "unknown")
    parts.append(f"Dataset: {dataset}")
    
    # Station/Location information (CRITICAL for CSV data)
    station_name = metadata.get("station_name")
    station_id = metadata.get("station_id")
    if station_name:
        parts.append(f"Station: {station_name}")
    elif station_id:
        parts.append(f"Station ID: {station_id}")
    
    # Temporal - PRECISE formatting for experts
    time_start = metadata.get("time_start")
    if time_start:
        try:
            from datetime import datetime
            dt_start = datetime.fromisoformat(str(time_start).replace('Z', '+00:00'))
            
            time_end = metadata.get("time_end")
            if time_end and time_end != time_start:
                dt_end = datetime.fromisoformat(str(time_end).replace('Z', '+00:00'))
                # Calculate number of days
                days = (dt_end - dt_start).days + 1
                
                if verbosity == "low":
                    time_str = f"{dt_start.strftime('%B %Y')} to {dt_end.strftime('%B %Y')}"
                else:
                    time_str = f"{time_start} to {time_end}"
                
                parts.append(f"Time period: {time_str} ({days} days)")
                
                # Add temporal frequency if available
                freq = metadata.get("temporal_frequency")
                if freq:
                    parts.append(f"Frequency: {freq}")
            else:
                if verbosity == "low":
                    time_str = dt_start.strftime("%B %Y")
                else:
                    time_str = time_start
                parts.append(f"Time: {time_str}")
        except:
            # Fallback if parsing fails
            time_str = str(time_start)
            time_end = metadata.get("time_end")
            if time_end and time_end != time_start:
                parts.append(f"Time period: {time_str} to {time_end}")
            else:
                parts.append(f"Time: {time_str}")
    
    # Row count (for CSV data - shows data volume)
    row_count = metadata.get("row_count")
    if row_count and verbosity in ["medium", "high"]:
        parts.append(f"Data points: {row_count}")
    
    # Spatial (clearly marked as coordinates, not temperature!)
    lat_min = metadata.get("latitude_min")
    lat_max = metadata.get("latitude_max")
    lon_min = metadata.get("longitude_min")
    lon_max = metadata.get("longitude_max")
    
    spatial_parts = []
    if lat_min is not None and lat_max is not None:
        if lat_min == lat_max:
            spatial_parts.append(f"Latitude: {lat_min:.4f}°N")
        else:
            spatial_parts.append(f"Latitude: {lat_min:.4f}° to {lat_max:.4f}°N")
    
    if lon_min is not None and lon_max is not None:
        if lon_min == lon_max:
            spatial_parts.append(f"Longitude: {lon_min:.4f}°E")
        else:
            spatial_parts.append(f"Longitude: {lon_min:.4f}° to {lon_max:.4f}°E")
    
    if spatial_parts:
        parts.append(f"Geographic coordinates: {' '.join(spatial_parts)}")
    
    # Statistics (CRITICAL for RAG - PRECISE formatting for experts)
    stats_parts = []
    unit_str = f" {unit}" if unit else ""
    
    if "stats_mean" in metadata:
        mean_val = metadata['stats_mean']
        # Use appropriate precision based on unit
        if unit and unit.lower() in ['c', '°c', 'f', '°f', 'k']:
            stats_parts.append(f"Mean: {mean_val:.2f}{unit_str}")
        else:
            stats_parts.append(f"Mean: {mean_val:.2f}{unit_str}")
    
    if verbosity in ["medium", "high"] and "stats_std" in metadata:
        std_val = metadata['stats_std']
        stats_parts.append(f"Std dev: {std_val:.2f}{unit_str}")
    
    if "stats_min" in metadata and "stats_max" in metadata:
        min_val = metadata['stats_min']
        max_val = metadata['stats_max']
        stats_parts.append(f"Range: [{min_val:.2f}, {max_val:.2f}]{unit_str}")
    
    if verbosity in ["medium", "high"] and "stats_median" in metadata:
        median_val = metadata['stats_median']
        stats_parts.append(f"Median: {median_val:.2f}{unit_str}")
    
    if verbosity == "high":
        if "stats_p10" in metadata:
            p10_val = metadata['stats_p10']
            stats_parts.append(f"P10: {p10_val:.2f}{unit_str}")
        if "stats_p90" in metadata:
            p90_val = metadata['stats_p90']
            stats_parts.append(f"P90: {p90_val:.2f}{unit_str}")
        if "stats_range" in metadata:
            range_val = metadata['stats_range']
            stats_parts.append(f"Range width: {range_val:.2f}{unit_str}")
    
    if stats_parts:
        parts.append("Statistics: " + " | ".join(stats_parts))
    
    # Grid resolution (if available)
    if verbosity in ["medium", "high"] and metadata.get("resolution_deg"):
        parts.append(f"Grid resolution: {metadata['resolution_deg']}°")
    
    return " | ".join(parts)

