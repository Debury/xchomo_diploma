"""
Classify catalog entries into processing phases (0-4).

Phase 0: Metadata-only (embed Excel metadata, no data download)
Phase 1: Direct download, open access (URLs with file extensions)
Phase 2: Registration-required downloads
Phase 3: API-based portals (CDS, ESGF, etc.)
Phase 4: Manual / contact-required access
"""

import logging
import re
from typing import Dict, List
from urllib.parse import urlparse

from src.catalog.excel_reader import CatalogEntry

logger = logging.getLogger(__name__)

# File extensions we can download and process directly
_DIRECT_EXTENSIONS = {
    ".nc", ".nc4", ".netcdf",
    ".grib", ".grib2", ".grb", ".grb2",
    ".tif", ".tiff", ".geotiff",
    ".csv", ".tsv",
    ".hdf", ".hdf5", ".h5", ".he5",
    ".zarr",
    ".zip", ".gz", ".tar",
}

# Domains that host API portals requiring special adapters
_PORTAL_DOMAINS = {
    "cds.climate.copernicus.eu": "CDS",
    "climate.copernicus.eu": "CDS",
    "esgf-node.llnl.gov": "ESGF",
    "pcmdi.llnl.gov": "ESGF",
    "esgf-data.dkrz.de": "ESGF",
    "psl.noaa.gov": "NOAA",
    "www.ncei.noaa.gov": "NOAA",
    "disc.gsfc.nasa.gov": "NASA",
    "earthdata.nasa.gov": "NASA",
    "podaac.jpl.nasa.gov": "NASA",
    "gpm.nasa.gov": "NASA",
    # Additional portals discovered during Phase 1 audit
    "rda.ucar.edu": "NCAR",
    "publitheque.meteo.fr": "METEO",
    "ogimet.com": "WMO",
    "cost-g.org": "PORTAL",
    "land.copernicus.eu": "CDS",
    "mareografico.it": "PORTAL",
    "scia.isprambiente.it": "PORTAL",
    "meteo.data.gouv.fr": "PORTAL",
    "meteo.gr": "PORTAL",
    "esgf-node.ipsl.upmc.fr": "ESGF",
}


def _detect_format_from_url(url: str) -> str | None:
    """Check if a URL points to a directly downloadable file."""
    if not url:
        return None
    parsed = urlparse(url.strip())
    path = parsed.path.lower()
    for ext in _DIRECT_EXTENSIONS:
        if path.endswith(ext):
            return ext.lstrip(".")
    return None


def _get_portal(url: str) -> str | None:
    """Check if URL belongs to a known data portal."""
    if not url:
        return None
    parsed = urlparse(url.strip())
    domain = parsed.netloc.lower()
    for portal_domain, portal_name in _PORTAL_DOMAINS.items():
        if domain == portal_domain or domain.endswith("." + portal_domain):
            return portal_name
    return None


def classify_source(entry: CatalogEntry) -> int:
    """
    Classify a catalog entry into a processing phase.

    Returns:
        0 = metadata-only (always applied first)
        1 = direct download, open access
        2 = registration-required download
        3 = API-based portal
        4 = manual / contact-required
    """
    access = (entry.access or "").lower().strip()
    link = (entry.link or "").strip()

    # No link at all → metadata only
    if not link:
        return 4

    # Contact-required
    if "contact" in access:
        return 4

    # Check if it's an API portal (CDS, ESGF, etc.)
    portal = _get_portal(link)
    if portal:
        if "registration" in access or "upon registration" in access:
            return 3
        # Even open-access portals often need API adapters
        return 3

    # Check if URL has a direct file extension
    fmt = _detect_format_from_url(link)

    if fmt:
        # Has a recognizable file extension
        if "registration" in access:
            return 2
        return 1

    # URL exists but no direct extension and not a known portal
    if "registration" in access:
        return 2

    if access == "open":
        # Open access but no file extension — likely a web page / data browser
        return 1

    # Fallback
    return 4


def classify_all(entries: List[CatalogEntry]) -> Dict[int, List[CatalogEntry]]:
    """
    Classify all catalog entries and group by phase.

    Returns:
        Dictionary mapping phase number (0-4) to list of entries.
        Note: Phase 0 includes ALL entries (metadata embedding).
    """
    grouped: Dict[int, List[CatalogEntry]] = {0: list(entries), 1: [], 2: [], 3: [], 4: []}

    for entry in entries:
        phase = classify_source(entry)
        grouped[phase].append(entry)

    for phase, items in sorted(grouped.items()):
        logger.info(f"Phase {phase}: {len(items)} sources")

    return grouped
