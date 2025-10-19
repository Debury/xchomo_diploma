"""
Source management module
Handles dynamic climate data sources for ETL pipeline
"""

from .source_store import ClimateDataSource, SourceStore, get_source_store

__all__ = [
    'ClimateDataSource',
    'SourceStore',
    'get_source_store'
]
