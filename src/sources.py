from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ClimateDataSource:
    """
    Internal representation of a data source.
    Matches the fields sent by the API.
    """
    source_id: str
    url: str
    is_active: bool = True
    format: Optional[str] = None
    variables: Optional[List[str]] = None
    spatial_bbox: Optional[List[float]] = None
    time_range: Optional[Dict[str, str]] = None
    transformations: Optional[List[str]] = None
    aggregation_method: Optional[str] = "mean"
    output_resolution: Optional[float] = None
    embedding_model: Optional[str] = "BAAI/bge-large-en-v1.5"
    chunk_size: Optional[int] = 512
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Tracking fields (optional, often handled by DB)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_status: str = "pending"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return self.__dict__

class SourceStore:
    """
    In-memory database for sources. 
    (In production, replace this with SQLAlchemy/Postgres logic)
    """
    def __init__(self, db_url: str = None):
        self._sources: Dict[str, ClimateDataSource] = {}

    def create_source(self, data: Dict[str, Any]) -> ClimateDataSource:
        """Create and store a new source."""
        # SAFETY FIX: Only pass arguments that the dataclass actually accepts
        # This prevents 'unexpected keyword argument' errors
        valid_keys = ClimateDataSource.__dataclass_fields__.keys()
        clean_data = {k: v for k, v in data.items() if k in valid_keys}
        
        obj = ClimateDataSource(**clean_data)
        self._sources[obj.source_id] = obj
        return obj

    def get_source(self, source_id: str) -> Optional[ClimateDataSource]:
        return self._sources.get(source_id)

    def get_all_sources(self, active_only: bool = True) -> List[ClimateDataSource]:
        if active_only:
            return [s for s in self._sources.values() if s.is_active]
        return list(self._sources.values())

    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[ClimateDataSource]:
        if source_id not in self._sources:
            return None
        
        source = self._sources[source_id]
        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)
        return source

    def update_processing_status(self, source_id: str, status: str, error_message: str = None):
        if source_id in self._sources:
            self._sources[source_id].processing_status = status
            if error_message:
                self._sources[source_id].error_message = error_message
            print(f"[SourceStore] Updated {source_id} -> {status}")
            return True
        return False

    def delete_source(self, source_id: str) -> bool:
        """Soft delete (set inactive)."""
        if source_id in self._sources:
            self._sources[source_id].is_active = False
            return True
        return False

    def hard_delete_source(self, source_id: str) -> bool:
        """Permanently remove."""
        if source_id in self._sources:
            del self._sources[source_id]
            return True
        return False

    def get_sources_by_tags(self, tags: List[str]) -> List[ClimateDataSource]:
        results = []
        for source in self._sources.values():
            if source.tags and any(t in source.tags for t in tags):
                results.append(source)
        return results

# Singleton instance
_STORE = SourceStore()

def get_source_store(db_url: str = None) -> SourceStore:
    return _STORE