"""
Normalized schema for climate data chunks in vector database.

This module defines the standard structure for storing climate data chunks
in vector databases (Qdrant, Weaviate, etc.) with structured, filterable metadata.
"""

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


def _sanitize_number(v):
    """Replace NaN/Infinity with None so Qdrant JSON serialization doesn't fail."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


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
    
    # Catalog metadata (from D1.1.xlsx Excel catalog)
    hazard_type: Optional[str] = None  # e.g., "Mean surface temperature"
    data_type: Optional[str] = None  # "Reanalysis data", "Model", "Gridded observations"
    spatial_coverage: Optional[str] = None  # "Regional", "Global"
    region_country: Optional[str] = None  # "Europe", "Global", "Iberian Peninsula"
    temporal_coverage_text: Optional[str] = None  # "1984-Present", "Pre-industrial-2100"
    impact_sector: Optional[str] = None  # "Health, Energy, Agriculture"
    access_type: Optional[str] = None  # "Open", "Open (upon registration)"
    catalog_source: Optional[str] = None  # "D1.1.xlsx"
    location_name: Optional[str] = None  # Enriched location from Excel Region/Country

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
                        sanitized = _sanitize_number(stat_val)
                        if sanitized is not None:
                            result[f"stats_{stat_key}"] = sanitized
                elif key == "additional_metadata" and isinstance(value, dict):
                    # Merge additional metadata into main dict
                    for k, v in value.items():
                        sanitized = _sanitize_number(v)
                        if sanitized is not None:
                            result[k] = sanitized
                else:
                    sanitized = _sanitize_number(value)
                    if sanitized is not None:
                        result[key] = sanitized
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
            "description", "title", "institution", "source_description",
            # Catalog metadata fields (from D1.1.xlsx)
            "hazard_type", "data_type", "spatial_coverage", "region_country",
            "temporal_coverage_text", "impact_sector", "access_type", "catalog_source",
            "location_name", "is_metadata_only", "catalog_row_index",
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

    Produces a natural-language narrative that embeds well for semantic search.
    All content is derived dynamically from whatever metadata fields are present.

    Used for:
    1. Embedding generation (text embedding model needs text input)
    2. LLM context formatting (after retrieval)
    """
    parts = []

    # ── Dataset identity ──────────────────────────────────────────────────
    dataset = metadata.get("dataset_name") or metadata.get("source_id", "unknown")
    variable = metadata.get("variable", "unknown")
    long_name = metadata.get("long_name")
    standard_name = metadata.get("standard_name")
    unit = metadata.get("unit", "")

    # Build a rich opening sentence from available fields
    var_label = long_name or standard_name or variable
    hazard = metadata.get("hazard_type")
    data_type = metadata.get("data_type")
    region = metadata.get("region_country") or metadata.get("location_name")
    spatial_cov = metadata.get("spatial_coverage")
    impact = metadata.get("impact_sector")

    # Opening: "Dataset X provides <variable> (<hazard>) data over <region>."
    opener_parts = [f"{dataset}"]
    if data_type:
        opener_parts.append(f"({data_type})")
    opener_parts.append(f"— {var_label}")
    if var_label != variable:
        opener_parts.append(f"[{variable}]")
    if unit:
        opener_parts.append(f"in {unit}")
    parts.append(" ".join(opener_parts))

    # Hazard / application context — critical for semantic matching
    if hazard:
        # Strip catalog_ prefix if present for cleaner text
        clean_hazard = hazard.replace("catalog_", "").replace("_", " ")
        parts.append(f"Climate hazard context: {clean_hazard}")
    if impact:
        parts.append(f"Relevant to: {impact}")

    # User-provided keywords for semantic enrichment
    keywords = metadata.get("keywords")
    if keywords and isinstance(keywords, list):
        parts.append(f"Keywords: {', '.join(str(k) for k in keywords)}")

    # Standard name adds CF-convention searchability
    if standard_name and standard_name != long_name and standard_name != variable:
        parts.append(f"CF standard name: {standard_name}")

    # ── Spatial context ───────────────────────────────────────────────────
    spatial_desc_parts = []
    if region:
        spatial_desc_parts.append(region)
    if spatial_cov and spatial_cov.lower() != (region or "").lower():
        spatial_desc_parts.append(f"{spatial_cov} coverage")

    lat_min = metadata.get("latitude_min")
    lat_max = metadata.get("latitude_max")
    lon_min = metadata.get("longitude_min")
    lon_max = metadata.get("longitude_max")
    if lat_min is not None and lat_max is not None:
        if lat_min == lat_max:
            spatial_desc_parts.append(f"lat {lat_min:.2f}°")
        else:
            spatial_desc_parts.append(f"lat {lat_min:.2f}° to {lat_max:.2f}°")
    if lon_min is not None and lon_max is not None:
        if lon_min == lon_max:
            spatial_desc_parts.append(f"lon {lon_min:.2f}°")
        else:
            spatial_desc_parts.append(f"lon {lon_min:.2f}° to {lon_max:.2f}°")
    if spatial_desc_parts:
        parts.append("Coverage: " + ", ".join(spatial_desc_parts))

    # Station for point observations
    station = metadata.get("station_name") or metadata.get("station_id")
    if station:
        parts.append(f"Station: {station}")

    # ── Temporal context ──────────────────────────────────────────────────
    time_start = metadata.get("time_start")
    time_end = metadata.get("time_end")
    temporal_text = metadata.get("temporal_coverage_text")
    freq = metadata.get("temporal_frequency")

    if time_start:
        t_str = str(time_start)[:10]
        if time_end and str(time_end)[:10] != t_str:
            t_str += f" to {str(time_end)[:10]}"
            try:
                dt_s = datetime.fromisoformat(str(time_start).replace('Z', '+00:00'))
                dt_e = datetime.fromisoformat(str(time_end).replace('Z', '+00:00'))
                days = (dt_e - dt_s).days + 1
                t_str += f" ({days} days)"
            except Exception:
                pass
        time_line = f"Period: {t_str}"
        if freq:
            time_line += f", {freq}"
        parts.append(time_line)
    elif temporal_text:
        parts.append(f"Temporal range: {temporal_text}")

    # ── Statistics ────────────────────────────────────────────────────────
    unit_str = f" {unit}" if unit else ""
    stats_items = []
    if "stats_mean" in metadata:
        stats_items.append(f"mean={metadata['stats_mean']:.2f}{unit_str}")
    if "stats_min" in metadata and "stats_max" in metadata:
        stats_items.append(f"range=[{metadata['stats_min']:.2f}, {metadata['stats_max']:.2f}]{unit_str}")
    if verbosity in ("medium", "high") and "stats_std" in metadata:
        stats_items.append(f"std={metadata['stats_std']:.2f}{unit_str}")
    if verbosity in ("medium", "high") and "stats_median" in metadata:
        stats_items.append(f"median={metadata['stats_median']:.2f}{unit_str}")
    if stats_items:
        parts.append("Statistics: " + ", ".join(stats_items))

    # Row count for tabular data
    row_count = metadata.get("row_count")
    if row_count and verbosity in ("medium", "high"):
        parts.append(f"Data points: {row_count}")

    # Grid resolution
    if verbosity in ("medium", "high") and metadata.get("resolution_deg"):
        parts.append(f"Resolution: {metadata['resolution_deg']}°")

    # Access type
    if metadata.get("access_type"):
        parts.append(f"Access: {metadata['access_type']}")

    # Dataset-level attributes from file (title, institution, source, etc.)
    # These come from NetCDF/GRIB global attrs, stored as dataset_* keys
    ds_desc_parts = []
    for key, val in metadata.items():
        if key.startswith("dataset_") and isinstance(val, str) and len(val) > 3:
            label = key.replace("dataset_", "").replace("_", " ").capitalize()
            # Skip very long attrs (e.g. history) to keep text focused
            if len(val) <= 200:
                ds_desc_parts.append(f"{label}: {val}")
    if ds_desc_parts:
        parts.append("File metadata: " + "; ".join(ds_desc_parts[:5]))

    return ". ".join(parts)


def build_dataset_summary(
    chunk_metadatas: list,
    entry_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build a single summary metadata dict from all chunks of a dataset.

    Aggregates spatial bounds, temporal range, variables, and statistics
    across all chunks into one record that represents the whole dataset.
    This summary chunk is embedded alongside data chunks so that broad
    queries ("global warming", "European drought") match the dataset
    overview rather than a random spatial tile.

    Args:
        chunk_metadatas: list of per-chunk metadata dicts (from to_dict())
        entry_meta: catalog entry fields (hazard_type, impact_sector, etc.)

    Returns:
        A single metadata dict ready for embedding + upsert.
    """
    if not chunk_metadatas:
        return {}

    # Collect unique variables and their long names
    variables = set()
    long_names = set()
    units = set()
    for m in chunk_metadatas:
        v = m.get("variable")
        if v:
            variables.add(v)
        ln = m.get("long_name")
        if ln:
            long_names.add(ln)
        u = m.get("unit")
        if u:
            units.add(u)

    # Aggregate spatial bounds (full extent)
    lat_mins = [m["latitude_min"] for m in chunk_metadatas if m.get("latitude_min") is not None]
    lat_maxs = [m["latitude_max"] for m in chunk_metadatas if m.get("latitude_max") is not None]
    lon_mins = [m["longitude_min"] for m in chunk_metadatas if m.get("longitude_min") is not None]
    lon_maxs = [m["longitude_max"] for m in chunk_metadatas if m.get("longitude_max") is not None]

    # Aggregate temporal range
    time_starts = [m["time_start"] for m in chunk_metadatas if m.get("time_start")]
    time_ends = [m["time_end"] for m in chunk_metadatas if m.get("time_end")]

    # Aggregate statistics (global mean of means, global min/max)
    means = [m["stats_mean"] for m in chunk_metadatas if m.get("stats_mean") is not None]
    mins = [m["stats_min"] for m in chunk_metadatas if m.get("stats_min") is not None]
    maxs = [m["stats_max"] for m in chunk_metadatas if m.get("stats_max") is not None]

    first = chunk_metadatas[0]
    summary = {
        "dataset_name": first.get("dataset_name", "unknown"),
        "source_id": first.get("source_id", "unknown"),
        "variable": ", ".join(sorted(variables)) if variables else first.get("variable", "unknown"),
        "is_dataset_summary": True,
        "chunk_count": len(chunk_metadatas),
    }

    if long_names:
        summary["long_name"] = ", ".join(sorted(long_names))
    if units:
        summary["unit"] = ", ".join(sorted(units))

    # Spatial extent
    if lat_mins:
        summary["latitude_min"] = min(lat_mins)
    if lat_maxs:
        summary["latitude_max"] = max(lat_maxs)
    if lon_mins:
        summary["longitude_min"] = min(lon_mins)
    if lon_maxs:
        summary["longitude_max"] = max(lon_maxs)

    # Temporal extent
    if time_starts:
        summary["time_start"] = min(time_starts)
    if time_ends:
        summary["time_end"] = max(time_ends)

    # Aggregated statistics
    if means:
        summary["stats_mean"] = sum(means) / len(means)
    if mins:
        summary["stats_min"] = min(mins)
    if maxs:
        summary["stats_max"] = max(maxs)

    # Copy all entry metadata dynamically
    for key, val in entry_meta.items():
        if val and key not in summary:
            summary[key] = val

    # Resolution from first chunk
    if first.get("resolution_deg"):
        summary["resolution_deg"] = first["resolution_deg"]

    return summary
