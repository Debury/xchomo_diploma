"""
Text generation module for creating RAG-friendly descriptions from climate data chunks.

This module generates human-readable text descriptions from raster chunks with
proper formatting of spatial, temporal, and statistical information.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Variable name mapping for readability
VARIABLE_NAME_MAP = {
    "2m_temperature": "2-meter temperature",
    "temperature_2m": "2-meter temperature",
    "t2m": "2-meter temperature",
    "total_precipitation": "total precipitation",
    "tp": "total precipitation",
    "precipitation": "precipitation",
    "10m_wind_speed": "10-meter wind speed",
    "wind_speed": "wind speed",
    "surface_pressure": "surface pressure",
    "msl": "mean sea level pressure",
    "geopotential": "geopotential",
    "z": "geopotential",
    "band_1": "satellite image band 1",
    "air": "air temperature",
    "gray": "terrain intensity",
    "pm25": "PM2.5 particulate matter",
    "pm10": "PM10 particulate matter",
    "no2": "nitrogen dioxide",
    "o3": "ozone",
    "co": "carbon monoxide",
}


def format_temporal_info(metadata: Dict[str, Any]) -> List[str]:
    """Extract and format temporal information from metadata."""
    parts = []
    
    # Try various time field names
    time_fields = ["time_start", "time", "date", "timestamp", "datetime"]
    time_value = None
    
    for field in time_fields:
        if field in metadata:
            time_value = metadata[field]
            break
    
    if time_value:
        time_str = str(time_value)
        parts.append(f"Time: {time_str}")
        
        # Try to extract date and format nicely
        try:
            # Handle ISO format
            if "T" in time_str:
                date_part = time_str.split("T")[0]
            else:
                date_part = time_str.split()[0]
            
            # Try parsing as date
            try:
                dt_obj = datetime.strptime(date_part, "%Y-%m-%d")
                month_name = dt_obj.strftime("%B %Y")
                parts.append(f"Date: {month_name}")
            except ValueError:
                # Try other formats
                try:
                    dt_obj = datetime.strptime(date_part, "%Y/%m/%d")
                    month_name = dt_obj.strftime("%B %Y")
                    parts.append(f"Date: {month_name}")
                except ValueError:
                    pass
        except Exception:
            pass
    
    return parts


def format_spatial_info(metadata: Dict[str, Any]) -> List[str]:
    """Extract and format spatial information from metadata."""
    parts = []
    
    # Check for bounding box
    has_lat = "lat_min" in metadata or "lat_max" in metadata
    has_lon = "lon_min" in metadata or "lon_max" in metadata
    
    if has_lat and has_lon:
        lat_min = metadata.get("lat_min")
        lat_max = metadata.get("lat_max")
        lon_min = metadata.get("lon_min")
        lon_max = metadata.get("lon_max")
        
        if lat_min is not None and lat_max is not None:
            parts.append(f"Latitude: {lat_min:.2f}° to {lat_max:.2f}°")
        
        if lon_min is not None and lon_max is not None:
            parts.append(f"Longitude: {lon_min:.2f}° to {lon_max:.2f}°")
    
    # Check for single point coordinates
    elif "latitude" in metadata or "lat" in metadata:
        lat = metadata.get("latitude") or metadata.get("lat")
        lon = metadata.get("longitude") or metadata.get("lon")
        if lat is not None and lon is not None:
            parts.append(f"Location: {lat:.2f}°N, {lon:.2f}°E")
    
    return parts


def format_statistics(
    stats_vector: List[float],
    metadata: Dict[str, Any],
    verbosity: str = "medium"
) -> List[str]:
    """
    Format statistical information from a stats vector.
    
    Args:
        stats_vector: Statistical vector [mean, std, min, max, p10, median, p90, range]
        metadata: Additional metadata
        verbosity: "low", "medium", or "high"
    
    Returns:
        List of formatted statistic strings
    """
    parts = []
    
    if not stats_vector or len(stats_vector) < 8:
        return parts
    
    mean = stats_vector[0]
    std = stats_vector[1]
    min_val = stats_vector[2]
    max_val = stats_vector[3]
    p10 = stats_vector[4]
    median = stats_vector[5]
    p90 = stats_vector[6]
    range_val = stats_vector[7]
    
    # Get unit if available
    unit = metadata.get("unit", metadata.get("units", ""))
    unit_str = f" {unit}" if unit else ""
    
    if verbosity == "low":
        parts.append(f"Mean: {mean:.2f}{unit_str}")
        parts.append(f"Range: {min_val:.2f} to {max_val:.2f}{unit_str}")
    elif verbosity == "medium":
        parts.append(f"Mean: {mean:.2f}{unit_str}")
        parts.append(f"Standard deviation: {std:.2f}{unit_str}")
        parts.append(f"Range: {min_val:.2f} to {max_val:.2f}{unit_str}")
        parts.append(f"Median: {median:.2f}{unit_str}")
        parts.append(f"90th percentile: {p90:.2f}{unit_str}")
    else:  # high
        parts.append(f"Mean: {mean:.2f}{unit_str}")
        parts.append(f"Standard deviation: {std:.2f}{unit_str}")
        parts.append(f"Minimum: {min_val:.2f}{unit_str}")
        parts.append(f"Maximum: {max_val:.2f}{unit_str}")
        parts.append(f"Median: {median:.2f}{unit_str}")
        parts.append(f"10th percentile: {p10:.2f}{unit_str}")
        parts.append(f"90th percentile: {p90:.2f}{unit_str}")
        parts.append(f"Range: {range_val:.2f}{unit_str}")
    
    return parts


def generate_text_description(
    metadata: Dict[str, Any],
    stats_vector: Optional[List[float]] = None,
    verbosity: str = "medium",
    include_sample_values: bool = False,
    include_coordinates: bool = True,
    include_statistics: bool = True,
    include_attributes: bool = False,
) -> str:
    """
    Generate a comprehensive text description from chunk metadata and statistics.
    
    This function creates RAG-friendly text descriptions that include:
    - Variable name and type
    - Temporal information (if available)
    - Spatial information (if available)
    - Statistical summaries
    - Optional attributes and sample values
    
    Args:
        metadata: Chunk metadata dictionary
        stats_vector: Statistical vector [mean, std, min, max, p10, median, p90, range]
        verbosity: "low", "medium", or "high" - controls detail level
        include_sample_values: Whether to include sample data values
        include_coordinates: Whether to include spatial coordinates
        include_statistics: Whether to include statistical summaries
        include_attributes: Whether to include variable attributes
    
    Returns:
        Formatted text description string
    """
    parts = []
    
    # Variable information
    variable = metadata.get("variable", "unknown")
    readable_var = VARIABLE_NAME_MAP.get(variable.lower(), variable.replace("_", " ").title())
    parts.append(f"Climate variable: {readable_var} ({variable})")
    
    # Dataset/source information
    source_id = metadata.get("source_id", metadata.get("source", "unknown"))
    if source_id and source_id != "unknown":
        parts.append(f"Dataset: {source_id}")
    
    # Temporal information
    if "time" in str(metadata).lower() or "date" in str(metadata).lower():
        temporal_parts = format_temporal_info(metadata)
        parts.extend(temporal_parts)
    
    # Spatial information
    if include_coordinates:
        spatial_parts = format_spatial_info(metadata)
        parts.extend(spatial_parts)
    
    # Statistical information
    if include_statistics and stats_vector:
        stat_parts = format_statistics(stats_vector, metadata, verbosity)
        parts.extend(stat_parts)
    
    # Variable attributes (if high verbosity)
    if include_attributes and verbosity == "high":
        long_name = metadata.get("long_name")
        standard_name = metadata.get("standard_name")
        if long_name:
            parts.append(f"Long name: {long_name}")
        if standard_name:
            parts.append(f"Standard name: {standard_name}")
    
    # Sample values (if requested and available)
    if include_sample_values and "sample_values" in metadata:
        sample_str = str(metadata["sample_values"])
        if len(sample_str) < 100:  # Only include if not too long
            parts.append(f"Sample values: {sample_str}")
    
    # Join all parts with clear separators
    return " | ".join(parts)


def generate_batch_descriptions(
    metadata_list: List[Dict[str, Any]],
    stats_vectors: Optional[List[List[float]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Generate text descriptions for a batch of chunks.
    
    Args:
        metadata_list: List of metadata dictionaries
        stats_vectors: Optional list of statistical vectors
        config: Optional configuration dict with text_generation settings
    
    Returns:
        List of text description strings
    """
    if config is None:
        config = {}
    
    text_config = config.get("text_generation", {})
    verbosity = text_config.get("verbosity", "medium")
    include_sample_values = text_config.get("include_sample_values", False)
    include_coordinates = text_config.get("include_coordinates", True)
    include_statistics = text_config.get("include_statistics", True)
    include_attributes = text_config.get("include_attributes", False)
    
    descriptions = []
    
    for i, metadata in enumerate(metadata_list):
        stats_vec = stats_vectors[i] if stats_vectors and i < len(stats_vectors) else None
        
        desc = generate_text_description(
            metadata=metadata,
            stats_vector=stats_vec,
            verbosity=verbosity,
            include_sample_values=include_sample_values,
            include_coordinates=include_coordinates,
            include_statistics=include_statistics,
            include_attributes=include_attributes,
        )
        
        descriptions.append(desc)
    
    return descriptions

