"""
Batch orchestrator for processing catalog entries across all phases.

Supports resume (via JSON state file), dry-run, and phase filtering.
"""

import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.catalog.excel_reader import CatalogEntry, read_catalog
from src.catalog.phase_classifier import classify_source, classify_all
from src.catalog.metadata_pipeline import process_metadata_batch, process_metadata_only
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# Persistent file logger — writes to logs/catalog_pipeline.log (Docker volume mount)
catalog_logger = setup_logger("catalog_pipeline", "logs/catalog_pipeline.log", "INFO")

DEFAULT_PROGRESS_PATH = Path("data/catalog_progress.json")


@dataclass
class SourceProgress:
    """Progress state for a single source."""
    source_id: str
    dataset_name: str
    phase: int
    status: str = "pending"  # pending | processing | completed | failed | skipped
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class BatchProgress:
    """Overall batch progress state."""
    total: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    current_phase: Optional[int] = None
    current_source: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    sources: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def save(self, path: Path = DEFAULT_PROGRESS_PATH):
        """Persist state to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path = DEFAULT_PROGRESS_PATH) -> "BatchProgress":
        """Load state from JSON, or return empty state."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            sources = data.pop("sources", {})
            progress = cls(**{k: v for k, v in data.items() if k != "sources"})
            progress.sources = sources
            return progress
        except Exception as e:
            logger.warning(f"Failed to load progress from {path}: {e}")
            return cls()

    def mark_started(self, source_id: str, dataset_name: str, phase: int):
        self.current_source = source_id
        self.current_phase = phase
        self.sources[source_id] = {
            "dataset_name": dataset_name,
            "phase": phase,
            "status": "processing",
            "started_at": datetime.now().isoformat(),
        }
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self, source_id: str):
        if source_id in self.sources:
            self.sources[source_id]["status"] = "completed"
            self.sources[source_id]["completed_at"] = datetime.now().isoformat()
        self.processed += 1
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, source_id: str, error: str):
        if source_id in self.sources:
            self.sources[source_id]["status"] = "failed"
            self.sources[source_id]["error"] = error
            self.sources[source_id]["completed_at"] = datetime.now().isoformat()
        self.failed += 1
        self.updated_at = datetime.now().isoformat()

    def is_completed(self, source_id: str) -> bool:
        entry = self.sources.get(source_id, {})
        return entry.get("status") == "completed"

    def get_summary(self) -> Dict[str, Any]:
        pending = self.total - self.processed - self.failed - self.skipped
        return {
            "total": self.total,
            "processed": self.processed,
            "failed": self.failed,
            "skipped": self.skipped,
            "pending": max(0, pending),
            "current_phase": self.current_phase,
            "current_source": self.current_source,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }


def run_batch_pipeline(
    excel_path: str,
    phases: Optional[List[int]] = None,
    dry_run: bool = False,
    resume: bool = True,
    progress_path: Path = DEFAULT_PROGRESS_PATH,
) -> Dict[str, Any]:
    """
    Run the batch catalog processing pipeline.

    Args:
        excel_path: Path to the Excel catalog file.
        phases: Which phases to process (default: [0] for metadata-only).
        dry_run: If True, only classify and report — no processing.
        resume: If True, skip already-completed sources.
        progress_path: Path to the progress JSON file.

    Returns:
        Summary dict with processing results.
    """
    if phases is None:
        phases = [0]

    catalog_logger.info(f"=== Batch pipeline started | phases={phases} | excel={excel_path} ===")

    # Read catalog
    entries = read_catalog(excel_path)
    grouped = classify_all(entries)

    if dry_run:
        summary = {phase: len(items) for phase, items in grouped.items()}
        logger.info(f"Dry run — phase distribution: {summary}")
        return {"dry_run": True, "phases": summary, "total": len(entries)}

    # Load or create progress state
    progress = BatchProgress.load(progress_path) if resume else BatchProgress()
    progress.total = len(entries)
    progress.started_at = progress.started_at or datetime.now().isoformat()

    results: Dict[str, Any] = {"phases_run": phases, "total": len(entries)}

    for phase in sorted(phases):
        if phase not in grouped:
            continue

        phase_entries = grouped[phase] if phase > 0 else entries  # Phase 0 = all entries
        progress.current_phase = phase
        logger.info(f"Starting Phase {phase}: {len(phase_entries)} entries")

        if phase == 0:
            results["phase_0"] = _run_phase_0(phase_entries, progress, progress_path, resume)
        elif phase in (1, 2):
            results[f"phase_{phase}"] = _run_phase_download(
                phase_entries, phase, progress, progress_path, resume
            )
        elif phase == 3:
            results["phase_3"] = _run_phase_portal(phase_entries, progress, progress_path, resume)
        elif phase == 4:
            # Phase 4 is metadata-only for manual sources
            results["phase_4"] = _run_phase_0(phase_entries, progress, progress_path, resume)

        progress.save(progress_path)

    results["summary"] = progress.get_summary()
    catalog_logger.info(
        f"=== Batch pipeline finished | processed={progress.processed} "
        f"failed={progress.failed} skipped={progress.skipped} total={progress.total} ==="
    )
    return results


def _run_phase_0(
    entries: List[CatalogEntry],
    progress: BatchProgress,
    progress_path: Path,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 0: embed metadata only."""
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader("config/pipeline_config.yaml").load()
    embedder = TextEmbedder()
    db = VectorDatabase(config=config)

    to_process = []
    for entry in entries:
        if resume and progress.is_completed(entry.source_id):
            progress.skipped += 1
            continue
        to_process.append(entry)

    if not to_process:
        logger.info("Phase 0: nothing to process (all completed)")
        return {"processed": 0, "failed": 0, "skipped": len(entries)}

    logger.info(f"Phase 0: processing {len(to_process)} entries (skipped {len(entries) - len(to_process)})")

    processed = 0
    failed = 0

    for entry in to_process:
        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", 0)
        catalog_logger.info(f"Phase 0: started {entry.dataset_name} ({entry.source_id})")
        try:
            ok = process_metadata_only(entry, embedder, db)
            if ok:
                progress.mark_completed(entry.source_id)
                processed += 1
                catalog_logger.info(f"Phase 0: completed {entry.dataset_name}")
            else:
                progress.mark_failed(entry.source_id, "process_metadata_only returned False")
                failed += 1
                catalog_logger.warning(f"Phase 0: returned False for {entry.dataset_name}")
        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e))
            failed += 1
            catalog_logger.error(f"Phase 0: FAILED {entry.dataset_name}: {e}\n{tb}")

        # Save progress periodically (every 10 entries)
        if (processed + failed) % 10 == 0:
            progress.save(progress_path)

    progress.save(progress_path)
    return {"processed": processed, "failed": failed, "skipped": len(entries) - len(to_process)}


def _run_phase_download(
    entries: List[CatalogEntry],
    phase: int,
    progress: BatchProgress,
    progress_path: Path,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 1/2: download + process + embed."""
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto, raster_to_embeddings
    from src.climate_embeddings.schema import ClimateChunkMetadata, generate_human_readable_text
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    import requests as http_requests
    import tempfile

    config = ConfigLoader("config/pipeline_config.yaml").load()
    embedder = TextEmbedder()
    db = VectorDatabase(config=config)

    processed = 0
    failed = 0
    skipped = 0

    for entry in entries:
        if resume and progress.is_completed(entry.source_id):
            skipped += 1
            continue

        if not entry.link:
            skipped += 1
            continue

        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", phase)

        try:
            # Download to temp file
            url = entry.link.strip()
            logger.info(f"Phase {phase}: downloading {entry.dataset_name} from {url}")

            resp = http_requests.get(url, timeout=120, stream=True,
                                     headers={"User-Agent": "ClimateRAG/1.0"})
            resp.raise_for_status()

            # Detect extension from URL or content-type
            from src.climate_embeddings.loaders.detect_format import detect_format_from_url
            fmt = detect_format_from_url(url)
            ext = f".{fmt}" if fmt else ".nc"

            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                for chunk in resp.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name

            # Load and process
            result = load_raster_auto(tmp_path)
            embeddings_data = raster_to_embeddings(result)

            # Generate text embeddings and store
            for j, emb_data in enumerate(embeddings_data):
                meta = ClimateChunkMetadata.from_chunk_metadata(
                    raw_metadata=emb_data["metadata"],
                    stats_vector=emb_data["vector"],
                    source_id=entry.source_id,
                    dataset_name=entry.dataset_name,
                )
                meta_dict = meta.to_dict()

                # Add catalog metadata
                meta_dict["catalog_source"] = "D1.1.xlsx"
                if entry.hazard:
                    meta_dict["hazard_type"] = entry.hazard
                if entry.impact_sector:
                    meta_dict["impact_sector"] = entry.impact_sector
                if entry.region_country:
                    meta_dict["location_name"] = entry.region_country

                text = generate_human_readable_text(meta_dict)
                text_embedding = embedder.embed_documents([text])[0]

                point_id = f"{entry.source_id}_chunk_{j}"
                db.add_embeddings(
                    ids=[point_id],
                    embeddings=[text_embedding.tolist()],
                    metadatas=[meta_dict],
                )

            progress.mark_completed(entry.source_id)
            processed += 1
            logger.info(f"Phase {phase}: completed {entry.dataset_name} ({len(embeddings_data)} chunks)")

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e))
            failed += 1
            catalog_logger.error(f"Phase {phase}: FAILED {entry.dataset_name}: {e}\n{tb}")

        if (processed + failed) % 5 == 0:
            progress.save(progress_path)

    progress.save(progress_path)
    return {"processed": processed, "failed": failed, "skipped": skipped}


def _run_phase_portal(
    entries: List[CatalogEntry],
    progress: BatchProgress,
    progress_path: Path,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 3: API portal downloads (CDS, ESGF, etc.)."""
    from src.catalog.portal_adapters import get_adapter

    processed = 0
    failed = 0
    skipped = 0

    for entry in entries:
        if resume and progress.is_completed(entry.source_id):
            skipped += 1
            continue

        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", 3)

        try:
            adapter = get_adapter(entry)
            if adapter is None:
                logger.warning(f"No adapter for {entry.dataset_name} ({entry.link})")
                progress.mark_failed(entry.source_id, "No portal adapter available")
                failed += 1
                continue

            ok = adapter.download_and_process(entry)
            if ok:
                progress.mark_completed(entry.source_id)
                processed += 1
            else:
                progress.mark_failed(entry.source_id, "Adapter returned False")
                failed += 1

        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e))
            failed += 1
            catalog_logger.error(f"Phase 3: FAILED {entry.dataset_name}: {e}\n{tb}")

        if (processed + failed) % 5 == 0:
            progress.save(progress_path)

    progress.save(progress_path)
    return {"processed": processed, "failed": failed, "skipped": skipped}


def get_progress(progress_path: Path = DEFAULT_PROGRESS_PATH) -> Dict[str, Any]:
    """Get current batch progress (for API endpoint)."""
    progress = BatchProgress.load(progress_path)
    return progress.get_summary()


def retry_failed(
    excel_path: str,
    progress_path: Path = DEFAULT_PROGRESS_PATH,
) -> Dict[str, Any]:
    """Re-run all failed sources."""
    progress = BatchProgress.load(progress_path)
    failed_ids = {
        sid for sid, info in progress.sources.items()
        if info.get("status") == "failed"
    }

    if not failed_ids:
        return {"message": "No failed sources to retry", "count": 0}

    # Reset failed entries to pending
    for sid in failed_ids:
        progress.sources[sid]["status"] = "pending"
        progress.sources[sid].pop("error", None)
    progress.failed -= len(failed_ids)
    progress.save(progress_path)

    logger.info(f"Reset {len(failed_ids)} failed sources to pending for retry")

    # Re-run the pipeline with resume=True (will pick up the reset entries)
    return run_batch_pipeline(
        excel_path=excel_path,
        phases=[0, 1, 2, 3],
        resume=True,
        progress_path=progress_path,
    )
