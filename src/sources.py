from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ClimateDataSource:
    source_id: str
    url: str
    is_active: bool = True
    format: Optional[str] = None
    variables: Optional[List[str]] = None
    spatial_bbox: Optional[List[float]] = None

class SourceStore:
    """Mock database for demonstration. Replace with SQL/SQLite in prod."""
    def __init__(self):
        self._sources = {}

    def create_source(self, data: dict):
        obj = ClimateDataSource(**data)
        self._sources[obj.source_id] = obj
        return obj

    def get_source(self, source_id):
        return self._sources.get(source_id)

    def get_all_sources(self, active_only=True):
        return [s for s in self._sources.values() if not active_only or s.is_active]

    def update_processing_status(self, source_id, status, error_message=None):
        print(f"[DB] Source {source_id} status -> {status}")

_STORE = SourceStore()
def get_source_store():
    return _STORE