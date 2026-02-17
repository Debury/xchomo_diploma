"""
Read climate dataset catalog from Excel file (D1.1.xlsx).

Parses the 234-source Excel catalog with columns: Hazard, Dataset, Type,
Spatial/Temporal coverage, Access type, Link, Impact sector, etc.
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)

# Column name mapping from Excel headers to our field names
_COLUMN_MAP = {
    "Hazard": "hazard",
    "Dataset": "dataset_name",
    "Type (reanalysis/observations/models)": "data_type",
    "Spatial coverage": "spatial_coverage",
    "Region/Country": "region_country",
    "Spatial resolution (finest)": "spatial_resolution",
    "Temporal coverage": "temporal_coverage",
    "Temporal resolution (finest)": "temporal_resolution",
    "Bias corrected version available": "bias_corrected",
    "Access": "access",
    "Link": "link",
    "Impact sector": "impact_sector",
    "Notes": "notes",
}


@dataclass
class CatalogEntry:
    """A single climate dataset entry from the Excel catalog."""

    row_index: int
    hazard: Optional[str] = None
    dataset_name: Optional[str] = None
    data_type: Optional[str] = None
    spatial_coverage: Optional[str] = None
    region_country: Optional[str] = None
    spatial_resolution: Optional[str] = None
    temporal_coverage: Optional[str] = None
    temporal_resolution: Optional[str] = None
    bias_corrected: Optional[str] = None
    access: Optional[str] = None
    link: Optional[str] = None
    impact_sector: Optional[str] = None
    notes: Optional[str] = None

    @property
    def source_id(self) -> str:
        """Generate a unique source identifier from dataset name and row index."""
        name = (self.dataset_name or "unknown").strip()
        # Clean name for use as ID
        clean = name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        return f"catalog_{clean}_{self.row_index}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = asdict(self)
        result["source_id"] = self.source_id
        return {k: v for k, v in result.items() if v is not None}


def _clean_value(val: Any) -> Optional[str]:
    """Clean a cell value — convert NaN to None, strip whitespace."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def read_catalog(excel_path: str | Path) -> List[CatalogEntry]:
    """
    Read the climate dataset catalog from an Excel file.

    Args:
        excel_path: Path to the Excel file (e.g. "Kopie souboru D1.1.xlsx")

    Returns:
        List of CatalogEntry objects, one per row.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel catalog not found: {excel_path}")

    df = pd.read_excel(excel_path, engine="openpyxl")
    logger.info(f"Read {len(df)} rows from {excel_path.name}")

    # Forward-fill the Hazard column (it's merged across rows in the Excel)
    if "Hazard" in df.columns:
        df["Hazard"] = df["Hazard"].ffill()

    entries: List[CatalogEntry] = []
    for idx, row in df.iterrows():
        kwargs: Dict[str, Any] = {"row_index": int(idx)}

        for excel_col, field_name in _COLUMN_MAP.items():
            if excel_col in df.columns:
                kwargs[field_name] = _clean_value(row[excel_col])

        entry = CatalogEntry(**kwargs)

        # Skip rows that have no dataset name at all
        if entry.dataset_name is None:
            logger.debug(f"Skipping row {idx}: no dataset name")
            continue

        entries.append(entry)

    logger.info(f"Parsed {len(entries)} catalog entries from {len(df)} rows")
    return entries
