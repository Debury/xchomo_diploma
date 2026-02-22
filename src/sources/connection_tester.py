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


def analyze_url(url: str) -> Dict[str, Any]:
    """
    Analyze a URL to auto-detect format, portal, and suggested auth.
    Combines URL parsing with a connection test.
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

    return {
        "url": url,
        "reachable": conn["reachable"],
        "content_type": conn["content_type"],
        "format": conn["detected_format"],
        "portal": conn["detected_portal"],
        "suggested_auth": conn["suggested_auth"],
        "latency_ms": conn["latency_ms"],
        "error": conn["error"],
    }
