"""
Connection tester for data sources.

Tests reachability, detects content type, and measures latency.
"""

import logging
import time
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

# Known portal domains and their detection rules
PORTAL_DOMAINS = {
    "cds.climate.copernicus.eu": "CDS",
    "cds-beta.climate.copernicus.eu": "CDS",
    "esgf-node.llnl.gov": "ESGF",
    "esgf-data.dkrz.de": "ESGF",
    "esgf.ceda.ac.uk": "ESGF",
    "cmems-catalogue.eu": "MARINE",
    "nrt.cmems-du.eu": "MARINE",
    "my.cmems-du.eu": "MARINE",
    "marine.copernicus.eu": "MARINE",
    "earthdata.nasa.gov": "NASA",
    "disc.gsfc.nasa.gov": "NASA",
    "opendap.nccs.nasa.gov": "NASA",
    "data.giss.nasa.gov": "NASA",
    "www.ncei.noaa.gov": "NOAA",
    "downloads.psl.noaa.gov": "NOAA",
}

# Format detection from content-type
CONTENT_TYPE_FORMATS = {
    "application/x-netcdf": "netcdf",
    "application/netcdf": "netcdf",
    "application/x-hdf": "hdf5",
    "application/x-hdf5": "hdf5",
    "image/tiff": "geotiff",
    "image/geotiff": "geotiff",
    "text/csv": "csv",
    "application/zip": "zip",
    "application/gzip": "gz",
    "application/x-gzip": "gz",
    "application/grib": "grib",
    "application/octet-stream": None,  # Need to check extension
}

# Format detection from file extension
EXT_FORMATS = {
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".hdf": "hdf5",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".csv": "csv",
    ".zip": "zip",
    ".gz": "gz",
    ".grib": "grib",
    ".grib2": "grib",
    ".grb": "grib",
}


def test_connection(url: str, timeout: int = 15) -> Dict[str, Any]:
    """
    Test connectivity to a data source URL.

    Returns:
        {
            "reachable": bool,
            "content_type": str or None,
            "detected_format": str or None,
            "detected_portal": str or None,
            "suggested_auth": str or None,
            "latency_ms": float or None,
            "error": str or None,
            "status_code": int or None,
        }
    """
    result = {
        "reachable": False,
        "content_type": None,
        "detected_format": None,
        "detected_portal": None,
        "suggested_auth": None,
        "latency_ms": None,
        "error": None,
        "status_code": None,
    }

    if not url:
        result["error"] = "Empty URL"
        return result

    # Check for known portals
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    for domain, portal in PORTAL_DOMAINS.items():
        if domain in hostname:
            result["detected_portal"] = portal
            break

    # Portal-specific checks
    if result["detected_portal"] == "CDS":
        result["suggested_auth"] = "api_key"
        result["reachable"] = True  # CDS requires API, HEAD won't work
        result["detected_format"] = "netcdf"
        return result

    if result["detected_portal"] in ("NASA",) and "earthdata" in hostname:
        result["suggested_auth"] = "bearer_token"

    # Try HTTP HEAD first (fast), fall back to GET with stream
    start = time.time()
    try:
        resp = requests.head(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "ClimateRAG/1.0"},
        )
        latency = (time.time() - start) * 1000

        result["status_code"] = resp.status_code
        result["latency_ms"] = round(latency, 1)
        result["reachable"] = resp.status_code < 400

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        result["content_type"] = content_type

        # Detect format from content-type
        if content_type in CONTENT_TYPE_FORMATS:
            result["detected_format"] = CONTENT_TYPE_FORMATS[content_type]

        # If content-type is generic, try extension
        if not result["detected_format"]:
            path = parsed.path.lower()
            for ext, fmt in EXT_FORMATS.items():
                if path.endswith(ext):
                    result["detected_format"] = fmt
                    break

        # Check for auth-required responses
        if resp.status_code == 401:
            result["suggested_auth"] = "api_key"
            result["reachable"] = False
        elif resp.status_code == 403:
            result["suggested_auth"] = "bearer_token"
            result["reachable"] = False

        # HTML response likely means portal page, not data
        if "text/html" in content_type and result["reachable"]:
            result["error"] = "URL returns HTML (likely a portal page, not direct data)"
            result["suggested_auth"] = result["suggested_auth"] or "registration"

    except requests.exceptions.Timeout:
        result["error"] = f"Connection timed out after {timeout}s"
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection failed: {str(e)[:200]}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)[:200]}"

    return result


def _extract_dataset_name_from_url(url: str, portal: Optional[str] = None) -> Optional[str]:
    """Extract a human-readable dataset name from a URL."""
    parsed = urlparse(url)
    path = parsed.path.strip("/")

    # CDS: /api/v2/resources/reanalysis-era5-land → "reanalysis-era5-land"
    if portal == "CDS" or "cds.climate.copernicus.eu" in (parsed.hostname or ""):
        parts = path.split("/")
        if parts:
            slug = parts[-1]
            # Clean up CDS slug: "sis-agrometeorological-indicators" → "SIS Agrometeorological Indicators"
            return slug.replace("-", " ").replace("_", " ").title()

    # NOAA: /Datasets/ncep.reanalysis2/Monthlies/pressure/air.mon.mean.nc → "NCEP Reanalysis2"
    if portal == "NOAA" or "psl.noaa.gov" in (parsed.hostname or ""):
        if "/Datasets/" in parsed.path or "/datasets/" in path:
            dataset_part = path.split("atasets/")[-1].split("/")[0]
            return dataset_part.replace(".", " ").replace("_", " ").title()

    # NASA: extract from path
    if portal == "NASA":
        parts = path.split("/")
        for p in reversed(parts):
            if p and not p.startswith(".") and len(p) > 3:
                return p.replace("_", " ").replace("-", " ").title()

    # Marine Copernicus
    if portal == "MARINE":
        parts = path.split("/")
        if parts:
            return parts[-1].replace("-", " ").replace("_", " ").upper()

    # Generic: use filename without extension
    if path:
        filename = path.split("/")[-1]
        name = filename.rsplit(".", 1)[0] if "." in filename else filename
        if name and len(name) > 2:
            return name.replace("_", " ").replace("-", " ").title()

    return None


def _match_existing_dataset(extracted_name: str, existing_datasets: list) -> Optional[Dict]:
    """Fuzzy match an extracted name against existing datasets."""
    if not extracted_name or not existing_datasets:
        return None

    extracted_lower = extracted_name.lower()
    extracted_tokens = set(extracted_lower.replace("-", " ").replace("_", " ").split())

    best_match = None
    best_score = 0

    for ds in existing_datasets:
        ds_name = (ds.get("dataset_name") or "").lower()
        ds_tokens = set(ds_name.replace("-", " ").replace("_", " ").split())

        # Exact match
        if extracted_lower == ds_name:
            return ds

        # Token overlap score
        if ds_tokens and extracted_tokens:
            overlap = len(extracted_tokens & ds_tokens)
            score = overlap / max(len(extracted_tokens), len(ds_tokens))
            # Also check substring containment
            if ds_name in extracted_lower or extracted_lower in ds_name:
                score = max(score, 0.7)
            if score > best_score and score >= 0.4:
                best_score = score
                best_match = ds

    return best_match


def analyze_url(url: str, existing_datasets: list = None) -> Dict[str, Any]:
    """
    Analyze a URL to auto-detect format, portal, auth, and dataset grouping.
    Combines URL parsing with a connection test.

    Args:
        url: The URL to analyze
        existing_datasets: List of existing dataset dicts (from Qdrant) for grouping
    """
    conn = test_connection(url)

    # Enhance with URL-based detection
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Try to detect format from URL path if not detected from headers
    if not conn["detected_format"]:
        for ext, fmt in EXT_FORMATS.items():
            if ext in path:
                conn["detected_format"] = fmt
                break

    # Extract dataset name from URL
    portal = conn["detected_portal"]
    extracted_name = _extract_dataset_name_from_url(url, portal)

    # Match against existing datasets
    matched_dataset = None
    if extracted_name and existing_datasets:
        matched_dataset = _match_existing_dataset(extracted_name, existing_datasets)

    result = {
        "url": url,
        "reachable": conn["reachable"],
        "content_type": conn["content_type"],
        "format": conn["detected_format"],
        "portal": conn["detected_portal"],
        "suggested_auth": conn["suggested_auth"],
        "latency_ms": conn["latency_ms"],
        "error": conn["error"],
        "suggested_name": extracted_name,
        "matched_dataset": None,
    }

    if matched_dataset:
        result["matched_dataset"] = {
            "dataset_name": matched_dataset.get("dataset_name"),
            "source_id": matched_dataset.get("source_id"),
            "chunk_count": matched_dataset.get("chunk_count", 0),
        }

    return result
