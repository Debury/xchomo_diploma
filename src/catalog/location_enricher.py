"""
Location enrichment for catalog entries.

Strategy (ordered by reliability — advisor-approved):
1. Use Region/Country from Excel (most reliable)
2. Use station_name if present in data
3. Use spatial_coverage from Excel for global/regional descriptions
4. For point coords, use curated lookup table of known stations
5. For bounding boxes, describe geographic zone

NEVER calls external reverse-geocoding APIs.
"""

import logging
import re
from typing import Optional

from src.catalog.excel_reader import CatalogEntry

logger = logging.getLogger(__name__)

# Curated mapping of known bounding boxes to geographic zone descriptions
_BBOX_ZONES = [
    # (lat_min, lat_max, lon_min, lon_max, description)
    (35.0, 72.0, -25.0, 45.0, "Europe"),
    (25.0, 50.0, -130.0, -60.0, "North America"),
    (55.0, 72.0, -180.0, 180.0, "Arctic / High Latitudes"),
    (-60.0, -10.0, -80.0, -35.0, "South America"),
    (0.0, 40.0, -20.0, 55.0, "North Africa / Mediterranean"),
    (10.0, 55.0, 60.0, 150.0, "Asia"),
    (-50.0, -10.0, 110.0, 180.0, "Oceania / Australia"),
    (-90.0, 90.0, -180.0, 180.0, "Global"),
]

# Known station locations (lat, lon) -> location name
_KNOWN_STATIONS = {
    "Berlin-Tempelhof": "Berlin, Germany",
    "De Bilt": "De Bilt, Netherlands",
    "Wien Hohe Warte": "Vienna, Austria",
    "Praha-Klementinum": "Prague, Czech Republic",
}


def enrich_location(entry: CatalogEntry) -> str:
    """
    Derive a human-readable location description for a catalog entry.

    Uses the advisor-approved strategy: Excel metadata first, never geocode.

    Args:
        entry: A CatalogEntry from the Excel catalog.

    Returns:
        A location description string (e.g. "Europe", "Global", "Iberian Peninsula").
    """
    # 1. Region/Country from Excel — most reliable
    if entry.region_country:
        return entry.region_country.strip()

    # 2. Spatial coverage from Excel
    if entry.spatial_coverage:
        coverage = entry.spatial_coverage.strip()
        if coverage.lower() == "global":
            return "Global"
        if coverage.lower() == "regional":
            # Try to infer from dataset name
            return _infer_region_from_dataset(entry.dataset_name) or "Regional"
        return coverage

    return "Unknown"


def describe_bbox(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    """
    Describe a geographic bounding box as a named zone.

    Uses curated lookup table — no external API calls.
    """
    # Check if it's approximately global
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span > 150 and lon_span > 300:
        return "Global"

    # Find best matching zone
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    for z_lat_min, z_lat_max, z_lon_min, z_lon_max, desc in _BBOX_ZONES:
        if z_lat_min <= center_lat <= z_lat_max and z_lon_min <= center_lon <= z_lon_max:
            if desc != "Global":
                return desc

    return "Global"


def lookup_station(station_name: str) -> Optional[str]:
    """Look up a station name in the curated station table."""
    return _KNOWN_STATIONS.get(station_name)


def _infer_region_from_dataset(dataset_name: Optional[str]) -> Optional[str]:
    """Try to infer region from dataset name patterns."""
    if not dataset_name:
        return None
    name = dataset_name.lower()

    region_patterns = {
        "euro": "Europe",
        "europe": "Europe",
        "e-obs": "Europe",
        "cerra": "Europe",
        "iberia": "Iberian Peninsula",
        "safran": "France",
        "noa-gr": "Greece",
        "cy-obs": "Cyprus",
        "sloclim": "Slovenia",
        "apgd": "Alps",
        "arcis": "Arctic",
        "chirps": "Global (tropics)",
        "med-cordex": "Mediterranean",
    }

    for pattern, region in region_patterns.items():
        if pattern in name:
            return region

    return None
