"""
SQLAlchemy models for the climate_app database.

Four tables:
- sources: replaces shelve-based SourceStore
- source_credentials: per-source auth credentials
- source_schedules: per-source cron scheduling
- processing_runs: processing history tracking
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime,
    JSON, ForeignKey, Index, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(255), unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    format = Column(String(50), nullable=True)
    variables = Column(JSON, nullable=True)
    spatial_bbox = Column(JSON, nullable=True)
    time_range = Column(JSON, nullable=True)
    transformations = Column(JSON, nullable=True)
    aggregation_method = Column(String(50), default="mean")
    output_resolution = Column(Float, nullable=True)
    embedding_model = Column(String(255), default="BAAI/bge-large-en-v1.5")
    chunk_size = Column(Integer, default=512)
    description = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)
    keywords = Column(JSON, nullable=True)
    custom_metadata = Column(JSON, nullable=True)

    # Processing state
    processing_status = Column(String(50), default="pending", nullable=False)
    error_message = Column(Text, nullable=True)

    # Auth & portal
    auth_method = Column(String(50), nullable=True)
    portal = Column(String(50), nullable=True)
    catalog_row_index = Column(Integer, nullable=True)

    # User-supplied taxonomy (hazard/region/sector/spatial). Added 2026-04 after
    # an audit found they were accepted by the API and shown in the UI but
    # silently dropped before hitting PostgreSQL — so RAG filtering by these
    # facets never worked for non-catalog sources.
    hazard_type = Column(String(100), nullable=True)
    region_country = Column(String(255), nullable=True)
    spatial_coverage = Column(Text, nullable=True)
    impact_sector = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_processed_at = Column(DateTime, nullable=True)
    next_scheduled_at = Column(DateTime, nullable=True)

    # Relationships
    credentials = relationship("SourceCredential", back_populates="source", cascade="all, delete-orphan")
    schedule = relationship("SourceSchedule", back_populates="source", uselist=False, cascade="all, delete-orphan")
    runs = relationship("ProcessingRun", back_populates="source", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "url": self.url,
            "is_active": self.is_active,
            "format": self.format,
            "variables": self.variables,
            "spatial_bbox": self.spatial_bbox,
            "time_range": self.time_range,
            "transformations": self.transformations,
            "aggregation_method": self.aggregation_method,
            "output_resolution": self.output_resolution,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "description": self.description,
            "tags": self.tags,
            "processing_status": self.processing_status,
            "error_message": self.error_message,
            "auth_method": self.auth_method,
            "portal": self.portal,
            "catalog_row_index": self.catalog_row_index,
            "hazard_type": self.hazard_type,
            "region_country": self.region_country,
            "spatial_coverage": self.spatial_coverage,
            "impact_sector": self.impact_sector,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_processed": self.last_processed_at.isoformat() if self.last_processed_at else None,
        }


class SourceCredential(Base):
    __tablename__ = "source_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(
        String(255),
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    credential_key = Column(String(255), nullable=False)
    credential_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    source = relationship("Source", back_populates="credentials")

    __table_args__ = (
        Index("ix_source_cred_key", "source_id", "credential_key", unique=True),
    )


class SourceSchedule(Base):
    __tablename__ = "source_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(
        String(255),
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    cron_expression = Column(String(100), nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    last_triggered_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    source = relationship("Source", back_populates="schedule")

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "cron_expression": self.cron_expression,
            "is_enabled": self.is_enabled,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
        }


class DatasetSchedule(Base):
    """Dataset-level schedule — triggers ALL sources under a dataset_name."""
    __tablename__ = "dataset_schedules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    dataset_name = Column(String(255), nullable=False, index=True)
    cron_expression = Column(String(100), nullable=False)
    is_enabled = Column(Boolean, default=True, nullable=False)
    last_triggered_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "dataset_name": self.dataset_name,
            "cron_expression": self.cron_expression,
            "is_enabled": self.is_enabled,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
        }


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(
        String(255),
        ForeignKey("sources.source_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dagster_run_id = Column(String(255), nullable=True)
    job_name = Column(String(255), nullable=True)
    phase = Column(Integer, nullable=True)
    status = Column(String(50), nullable=False, default="started")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    chunks_processed = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    trigger_type = Column(String(50), default="manual")  # manual, schedule, sensor

    source = relationship("Source", back_populates="runs")

    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "dagster_run_id": self.dagster_run_id,
            "job_name": self.job_name,
            "phase": self.phase,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "chunks_processed": self.chunks_processed,
            "error_message": self.error_message,
            "trigger_type": self.trigger_type,
        }


class CatalogProgress(Base):
    """Tracks batch catalog processing progress per (source_id, phase) pair.

    Replaces the old JSON-file-based progress tracking with proper DB rows.
    """
    __tablename__ = "catalog_progress"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(String(255), nullable=False, index=True)
    dataset_name = Column(String(255), nullable=True)
    phase = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    error = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("source_id", "phase", name="uq_catalog_progress_source_phase"),
    )
