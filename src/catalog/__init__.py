"""Catalog module for batch processing climate data sources from Excel catalog."""

from src.catalog.excel_reader import CatalogEntry, read_catalog
from src.catalog.phase_classifier import classify_source, classify_all
from src.catalog.metadata_pipeline import process_metadata_only, process_metadata_batch

__all__ = [
    "CatalogEntry",
    "read_catalog",
    "classify_source",
    "classify_all",
    "process_metadata_only",
    "process_metadata_batch",
]
