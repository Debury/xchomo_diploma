"""
PostgreSQL-backed SourceStore — drop-in replacement for the shelve-based store.

Same public API as the original SourceStore in src/sources.py, plus new methods
for scheduling and processing history.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from src.database.connection import get_db_session
from src.database.models import Source, SourceCredential, SourceSchedule, ProcessingRun
from src.sources import ClimateDataSource

logger = logging.getLogger(__name__)


class SourceStore:
    """
    PostgreSQL-backed persistent store for climate data sources.
    Thread-safe via SQLAlchemy session-per-request pattern.
    """

    # ------------------------------------------------------------------ #
    # Core CRUD (same API as shelve-based store)
    # ------------------------------------------------------------------ #

    def create_source(self, data: Dict[str, Any]) -> ClimateDataSource:
        """Create a new source. Returns a ClimateDataSource DTO."""
        with get_db_session() as session:
            # Map dataclass fields to model columns
            model_fields = {c.key for c in Source.__table__.columns}
            clean = {k: v for k, v in data.items() if k in model_fields and k != "id"}
            clean.setdefault("created_at", datetime.utcnow())
            clean.setdefault("updated_at", datetime.utcnow())

            source = Source(**clean)
            session.add(source)
            session.flush()
            return _to_dto(source)

    def get_source(self, source_id: str) -> Optional[ClimateDataSource]:
        with get_db_session() as session:
            row = session.query(Source).filter(Source.source_id == source_id).first()
            return _to_dto(row) if row else None

    def get_all_sources(self, active_only: bool = True) -> List[ClimateDataSource]:
        with get_db_session() as session:
            q = session.query(Source)
            if active_only:
                q = q.filter(Source.is_active == True)
            return [_to_dto(r) for r in q.all()]

    def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[ClimateDataSource]:
        with get_db_session() as session:
            row = session.query(Source).filter(Source.source_id == source_id).first()
            if not row:
                return None
            model_fields = {c.key for c in Source.__table__.columns}
            for key, value in updates.items():
                if key in model_fields and key not in ("id", "source_id"):
                    setattr(row, key, value)
            row.updated_at = datetime.utcnow()
            session.flush()
            return _to_dto(row)

    def update_processing_status(self, source_id: str, status: str, error_message: str = None) -> bool:
        with get_db_session() as session:
            row = session.query(Source).filter(Source.source_id == source_id).first()
            if not row:
                return False
            row.processing_status = status
            if error_message:
                row.error_message = error_message
            if status == "completed":
                row.last_processed_at = datetime.utcnow()
                row.error_message = None
            row.updated_at = datetime.utcnow()
            logger.info(f"[SourceStore] Updated {source_id} -> {status}")
            return True

    def delete_source(self, source_id: str) -> bool:
        """Soft delete (set is_active=False)."""
        with get_db_session() as session:
            row = session.query(Source).filter(Source.source_id == source_id).first()
            if not row:
                return False
            row.is_active = False
            row.updated_at = datetime.utcnow()
            return True

    def hard_delete_source(self, source_id: str) -> bool:
        """Permanently delete a source and all related data."""
        with get_db_session() as session:
            row = session.query(Source).filter(Source.source_id == source_id).first()
            if not row:
                return False
            session.delete(row)
            return True

    def get_sources_by_tags(self, tags: List[str]) -> List[ClimateDataSource]:
        with get_db_session() as session:
            results = []
            for row in session.query(Source).filter(Source.is_active == True).all():
                if row.tags and any(t in row.tags for t in tags):
                    results.append(_to_dto(row))
            return results

    # ------------------------------------------------------------------ #
    # New: scheduling & freshness
    # ------------------------------------------------------------------ #

    def get_sources_due_for_update(self) -> List[ClimateDataSource]:
        """Get sources whose next_scheduled_at is in the past."""
        with get_db_session() as session:
            now = datetime.utcnow()
            rows = (
                session.query(Source)
                .join(SourceSchedule)
                .filter(
                    Source.is_active == True,
                    SourceSchedule.is_enabled == True,
                    SourceSchedule.next_run_at <= now,
                )
                .all()
            )
            return [_to_dto(r) for r in rows]

    # ------------------------------------------------------------------ #
    # New: processing history
    # ------------------------------------------------------------------ #

    def record_processing_run(
        self,
        source_id: str,
        job_name: str = None,
        dagster_run_id: str = None,
        phase: int = None,
        trigger_type: str = "manual",
    ) -> int:
        """Start a processing run record, returns run ID."""
        with get_db_session() as session:
            run = ProcessingRun(
                source_id=source_id,
                job_name=job_name,
                dagster_run_id=dagster_run_id,
                phase=phase,
                status="started",
                started_at=datetime.utcnow(),
                trigger_type=trigger_type,
            )
            session.add(run)
            session.flush()
            return run.id

    def complete_processing_run(
        self,
        run_id: int,
        status: str = "completed",
        chunks_processed: int = None,
        error_message: str = None,
    ):
        """Mark a processing run as completed or failed."""
        with get_db_session() as session:
            run = session.query(ProcessingRun).get(run_id)
            if not run:
                return
            run.status = status
            run.completed_at = datetime.utcnow()
            if run.started_at:
                run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
            run.chunks_processed = chunks_processed
            run.error_message = error_message

    def get_source_history(self, source_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get processing history for a source."""
        with get_db_session() as session:
            runs = (
                session.query(ProcessingRun)
                .filter(ProcessingRun.source_id == source_id)
                .order_by(ProcessingRun.started_at.desc())
                .limit(limit)
                .all()
            )
            return [r.to_dict() for r in runs]

    # ------------------------------------------------------------------ #
    # New: credentials
    # ------------------------------------------------------------------ #

    def set_credential(self, source_id: Optional[str], key: str, value: str):
        """Set a credential for a source (or global if source_id is None)."""
        with get_db_session() as session:
            existing = (
                session.query(SourceCredential)
                .filter(
                    SourceCredential.source_id == source_id,
                    SourceCredential.credential_key == key,
                )
                .first()
            )
            if existing:
                existing.credential_value = value
            else:
                cred = SourceCredential(
                    source_id=source_id,
                    credential_key=key,
                    credential_value=value,
                )
                session.add(cred)

    def get_credentials(self, source_id: Optional[str]) -> Dict[str, str]:
        """Get all credentials for a source (or global)."""
        with get_db_session() as session:
            creds = (
                session.query(SourceCredential)
                .filter(SourceCredential.source_id == source_id)
                .all()
            )
            return {c.credential_key: c.credential_value for c in creds}

    # ------------------------------------------------------------------ #
    # New: schedule CRUD
    # ------------------------------------------------------------------ #

    def set_schedule(self, source_id: str, cron_expression: str, is_enabled: bool = True) -> Dict[str, Any]:
        """Create or update a schedule for a source."""
        from croniter import croniter

        next_run = croniter(cron_expression, datetime.utcnow()).get_next(datetime)

        with get_db_session() as session:
            sched = session.query(SourceSchedule).filter(SourceSchedule.source_id == source_id).first()
            if sched:
                sched.cron_expression = cron_expression
                sched.is_enabled = is_enabled
                sched.next_run_at = next_run
                sched.updated_at = datetime.utcnow()
            else:
                sched = SourceSchedule(
                    source_id=source_id,
                    cron_expression=cron_expression,
                    is_enabled=is_enabled,
                    next_run_at=next_run,
                )
                session.add(sched)
            session.flush()
            return sched.to_dict()

    def get_schedule(self, source_id: str) -> Optional[Dict[str, Any]]:
        with get_db_session() as session:
            sched = session.query(SourceSchedule).filter(SourceSchedule.source_id == source_id).first()
            return sched.to_dict() if sched else None

    def delete_schedule(self, source_id: str) -> bool:
        with get_db_session() as session:
            sched = session.query(SourceSchedule).filter(SourceSchedule.source_id == source_id).first()
            if not sched:
                return False
            session.delete(sched)
            return True

    def get_due_schedules(self) -> List[Dict[str, Any]]:
        """Get all enabled schedules whose next_run_at is in the past."""
        with get_db_session() as session:
            now = datetime.utcnow()
            schedules = (
                session.query(SourceSchedule)
                .filter(
                    SourceSchedule.is_enabled == True,
                    SourceSchedule.next_run_at <= now,
                )
                .all()
            )
            return [s.to_dict() for s in schedules]

    def advance_schedule(self, source_id: str):
        """Advance a schedule's next_run_at to the next occurrence."""
        from croniter import croniter

        with get_db_session() as session:
            sched = session.query(SourceSchedule).filter(SourceSchedule.source_id == source_id).first()
            if not sched:
                return
            sched.last_triggered_at = datetime.utcnow()
            sched.next_run_at = croniter(sched.cron_expression, datetime.utcnow()).get_next(datetime)
            sched.updated_at = datetime.utcnow()


def _to_dto(row: Source) -> ClimateDataSource:
    """Convert a SQLAlchemy Source row to a ClimateDataSource dataclass."""
    return ClimateDataSource(
        source_id=row.source_id,
        url=row.url,
        is_active=row.is_active,
        format=row.format,
        variables=row.variables,
        spatial_bbox=row.spatial_bbox,
        time_range=row.time_range,
        transformations=row.transformations,
        aggregation_method=row.aggregation_method,
        output_resolution=row.output_resolution,
        embedding_model=row.embedding_model,
        chunk_size=row.chunk_size,
        description=row.description,
        tags=row.tags,
        keywords=row.keywords,
        custom_metadata=row.custom_metadata,
        created_at=row.created_at.isoformat() if row.created_at else None,
        updated_at=row.updated_at.isoformat() if row.updated_at else None,
        processing_status=row.processing_status or "pending",
        error_message=row.error_message,
    )
