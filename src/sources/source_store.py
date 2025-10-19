"""
Database models for dynamic climate data sources
Phase 5: Source-driven ETL pipeline with Vector DB integration
"""

from sqlalchemy import Column, Integer, String, DateTime, JSON, Boolean, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

Base = declarative_base()


class ClimateDataSource(Base):
    """
    Climate data source configuration for dynamic ETL
    
    Each source represents a dataset that will be:
    1. Downloaded/ingested
    2. Transformed (units, dimensions, aggregations)
    3. Embedded into vector database
    4. Made available for RAG queries
    """
    __tablename__ = 'climate_sources'
    
    # Primary identification
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # Source location
    url = Column(String(2048), nullable=False)
    format = Column(String(50), nullable=False)  # netcdf, grib, csv, parquet, etc.
    
    # Data specification
    variables = Column(JSON, nullable=False)  # ["temperature_2m", "precipitation"]
    time_range = Column(JSON, nullable=True)  # {"start": "2024-01-01", "end": "2024-12-31"}
    spatial_bbox = Column(JSON, nullable=True)  # [south, west, north, east]
    
    # Processing configuration
    transformations = Column(JSON, nullable=True)  # ["convert_units", "aggregate_daily"]
    aggregation_method = Column(String(50), default="mean")  # mean, sum, min, max
    output_resolution = Column(Float, nullable=True)  # Spatial resolution in degrees
    
    # Vector DB configuration
    embedding_model = Column(String(255), default="all-MiniLM-L6-v2")
    chunk_size = Column(Integer, default=512)
    collection_name = Column(String(255), nullable=True)
    
    # Metadata
    description = Column(String(1024), nullable=True)
    tags = Column(JSON, nullable=True)  # ["ERA5", "temperature", "Europe"]
    
    # Status tracking
    is_active = Column(Boolean, default=True, index=True)
    last_processed = Column(DateTime, nullable=True)
    processing_status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(String(2048), nullable=True)
    
    # Audit fields
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    created_by = Column(String(255), nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "url": self.url,
            "format": self.format,
            "variables": self.variables,
            "time_range": self.time_range,
            "spatial_bbox": self.spatial_bbox,
            "transformations": self.transformations,
            "aggregation_method": self.aggregation_method,
            "output_resolution": self.output_resolution,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "collection_name": self.collection_name,
            "description": self.description,
            "tags": self.tags,
            "is_active": self.is_active,
            "last_processed": self.last_processed.isoformat() if self.last_processed else None,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by
        }
    
    def __repr__(self):
        return f"<ClimateDataSource(source_id='{self.source_id}', format='{self.format}', status='{self.processing_status}')>"


class SourceStore:
    """
    Database manager for climate data sources
    Handles CRUD operations and provides interface for Dagster resources
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize source store with database connection
        
        Args:
            database_url: SQLAlchemy database URL. If None, uses environment variable or SQLite
        """
        if database_url is None:
            database_url = os.getenv(
                "CLIMATE_DB_URL",
                "sqlite:///data/climate_sources.db"
            )
        
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
    
    def create_source(self, source_data: Dict[str, Any]) -> ClimateDataSource:
        """Create a new climate data source"""
        session = self.SessionLocal()
        try:
            source = ClimateDataSource(**source_data)
            session.add(source)
            session.commit()
            session.refresh(source)
            return source
        finally:
            session.close()
    
    def get_source(self, source_id: str) -> Optional[ClimateDataSource]:
        """Get source by ID"""
        session = self.SessionLocal()
        try:
            return session.query(ClimateDataSource).filter(
                ClimateDataSource.source_id == source_id
            ).first()
        finally:
            session.close()
    
    def get_all_sources(self, active_only: bool = True) -> List[ClimateDataSource]:
        """Get all sources, optionally filtered by active status"""
        session = self.SessionLocal()
        try:
            query = session.query(ClimateDataSource)
            if active_only:
                query = query.filter(ClimateDataSource.is_active == True)
            return query.order_by(ClimateDataSource.created_at.desc()).all()
        finally:
            session.close()
    
    def get_pending_sources(self) -> List[ClimateDataSource]:
        """Get sources that are pending processing"""
        session = self.SessionLocal()
        try:
            return session.query(ClimateDataSource).filter(
                ClimateDataSource.is_active == True,
                ClimateDataSource.processing_status == "pending"
            ).all()
        finally:
            session.close()
    
    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[ClimateDataSource]:
        """Update an existing source"""
        session = self.SessionLocal()
        try:
            source = session.query(ClimateDataSource).filter(
                ClimateDataSource.source_id == source_id
            ).first()
            
            if source:
                for key, value in updates.items():
                    if hasattr(source, key):
                        setattr(source, key, value)
                session.commit()
                session.refresh(source)
            
            return source
        finally:
            session.close()
    
    def update_processing_status(
        self, 
        source_id: str, 
        status: str, 
        error_message: Optional[str] = None
    ) -> bool:
        """Update processing status of a source"""
        updates = {
            "processing_status": status,
            "last_processed": datetime.utcnow() if status == "completed" else None
        }
        if error_message:
            updates["error_message"] = error_message
        
        result = self.update_source(source_id, updates)
        return result is not None
    
    def delete_source(self, source_id: str) -> bool:
        """Delete a source (soft delete by setting is_active=False)"""
        session = self.SessionLocal()
        try:
            source = session.query(ClimateDataSource).filter(
                ClimateDataSource.source_id == source_id
            ).first()
            
            if source:
                source.is_active = False
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def hard_delete_source(self, source_id: str) -> bool:
        """Permanently delete a source from database"""
        session = self.SessionLocal()
        try:
            result = session.query(ClimateDataSource).filter(
                ClimateDataSource.source_id == source_id
            ).delete()
            session.commit()
            return result > 0
        finally:
            session.close()
    
    def get_sources_by_tags(self, tags: List[str]) -> List[ClimateDataSource]:
        """Get sources that match any of the given tags"""
        session = self.SessionLocal()
        try:
            sources = session.query(ClimateDataSource).filter(
                ClimateDataSource.is_active == True
            ).all()
            
            # Filter by tags in Python (since JSON querying varies by DB)
            matching_sources = []
            for source in sources:
                if source.tags and any(tag in source.tags for tag in tags):
                    matching_sources.append(source)
            
            return matching_sources
        finally:
            session.close()


# Initialize global store instance
_store_instance = None

def get_source_store(database_url: Optional[str] = None) -> SourceStore:
    """Get or create global SourceStore instance"""
    global _store_instance
    if _store_instance is None:
        _store_instance = SourceStore(database_url)
    return _store_instance
