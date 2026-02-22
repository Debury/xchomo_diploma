"""
Batch orchestrator for processing catalog entries across all phases.

Supports resume (via PostgreSQL state), dry-run, and phase filtering.
"""

import gc
import logging
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from src.catalog.excel_reader import CatalogEntry, read_catalog
from src.catalog.phase_classifier import classify_source, classify_all
from src.catalog.metadata_pipeline import process_metadata_batch, process_metadata_only
from src.database.connection import get_db_session
from src.database.models import CatalogProgress as CatalogProgressRow
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# Persistent file logger — writes to logs/catalog_pipeline.log (Docker volume mount)
catalog_logger = setup_logger("catalog_pipeline", "logs/catalog_pipeline.log", "INFO")

# Direct download URLs to override portal/landing page links in the Excel catalog.
# Keys are dataset names (matching CatalogEntry.dataset_name).
# Values are actual file URLs that return data when fetched with HTTP GET.
DIRECT_DOWNLOAD_URLS = {
    "WorldClim - Historical climate data": "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_tavg.zip",
    "WorldClim - Future climate data": "https://geodata.ucdavis.edu/cmip6/10m/ACCESS-CM2/ssp245/wc2.1_10m_tmin_ACCESS-CM2_ssp245_2041-2060.tif",
    "GISTEMP": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "CRU": "https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/cruts.2503051245.v4.09/tmp/cru_ts4.09.2021.2024.tmp.dat.nc.gz",
    "CHIRPS": "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/chirps-v2.0.2020.tif",
    # Figshare Aridity removed: AWS WAF blocks automated downloads (Phase 0 metadata-only)
    "SPEI-GD": "https://zenodo.org/api/records/8060268/files/30days.zip/content",
    "SLOCLIM": "https://zenodo.org/api/records/4108543/files/sloclim_pcp.nc/content",
    "SPEIbase": "https://digital.csic.es/bitstream/10261/364137/1/spei01.nc",
    "Iberia01": "https://digital.csic.es/bitstream/10261/183071/1/Iberia01_v1.0_DD_010reg_aa3d_pr.nc",
    "STEAD": "https://digital.csic.es/bitstream/10261/177655/14/tmax_pen.nc",
    "SPREAD": "https://digital.csic.es/bitstream/10261/141218/11/SPREAD_pen_pcp.nc",
    "Standardized Evapotranspiration Deficit Index (SEDI)": "https://digital.csic.es/bitstream/10261/160091/1/SEDI.zip",
    "Combined Drought Indicator": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DROUGHTOBS/Drought_Observatories_datasets/EDO_Combined_Drought_Indicator/ver1-4-0/cdinx_m_euu_20190101_20191221_t.nc",
    "SPI-MARSMet": "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DROUGHTOBS/Drought_Observatories_datasets/EDO_MARSMet_Standardized_Precipitation_Index_SPI3/ver1-0-0/spm03_m_euu_20040101_20041221_t.nc",
    "ISI-MIP": "https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere_composition/co2/historical/co2_historical_annual_1850_2014.txt",
    # --- Phase 1 audit overrides ---
    "E-OBS": "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.25deg_reg_ensemble/tg_ens_mean_0.25deg_reg_2011-2025_v32.0e.nc",
    "NOAAN": "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/ytd/12/1880-2023.csv",
    "GSFC-NASA": "https://earth.gsfc.nasa.gov/sites/default/files/geo/gsfc.glb_.200204_202505_rl06v2.0_obp-ice6gd_halfdegree.nc",
    "ROCIO_IBEB": "https://www.aemet.es/documentos/es/serviciosclimaticos/cambio_climat/datos_diarios/dato_observacional/rejilla_5km/v2/Serie_AEMET_v2_pcp_2020_netcdf.tar.gz",
    # --- NOAA PSL datasets (Phase 3→1 via direct NetCDF, verified anonymous) ---
    "CPC": "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.2023.nc",
    "GPCC": "https://downloads.psl.noaa.gov/Datasets/gpcc/full_v2020/precip.mon.total.0.25x0.25.v2020.nc",
    "GPCP": "https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
    "NOAA-NCEP/NCAR Reanalysis 1": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/air.sig995.2023.nc",
    "NCEP-NCAR2": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/surface/mslp.2023.nc",
    # --- Other Phase 3/4→1 overrides (verified anonymous) ---
    "HadlSST": "https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz",
    "SPI-GPCC": "https://opendata.dwd.de/climate_environment/GPCC/GPCC_DI/2023/GPCC_DI_202301.nc.gz",
    "CSR GRACE": "https://download.csr.utexas.edu/outgoing/grace/RL0603_mascons/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
    "CMIP6": "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/latest/tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc",
    # Hydro-JULES: CEDA DAP requires auth (removed from overrides, stays Phase 2)
    # --- ISI-MIP alias ---
    "ISIMIP": "https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere_composition/co2/historical/co2_historical_annual_1850_2014.txt",
    # CERES-EBAF: requires NASA Earthdata login (Phase 3 NASA adapter)
    # EURO-CORDEX: ESGF OpenDAP (Phase 3 ESGF adapter)
    # MED-CORDEX: ESGF OpenDAP (Phase 3 ESGF adapter)
    # ERA5, ERA5 Land, CERRA: CDS API (Phase 3 CDS adapter)
    # IMERG, MERRA-2: NASA adapter (Phase 3)
    # HWE-DB: portal only (meteo.gr), no direct downloads
    # NOA-GR: auth required (meteosearch.meteo.gr), registration closed
}

# Datasets that should NOT be processed in Phase 1 (auth required, no direct downloads, portals).
# These will be skipped during _run_phase_download() with a log message.
SKIP_PHASE1 = {
    "NOA-GR",           # auth required, portal down
    "HWE-DB",           # portal only (meteo.gr)
    "CERES-EBAF",       # NASA Earthdata login required
    "JPL GRACE",        # NASA Earthdata login required
    "COST-g",           # No direct downloads (GFC format, unsupported)
    "G3P",              # registration required (g3p.eu portal)
    "Rete Mareografica Italiana",  # Italian portal, no direct downloads
    "MED-CORDEX",       # ESGF auth required
    "EURO-CORDEX",      # ESGF auth required
    "RMI-ISPRA",        # ISPRA portal, no direct downloads
    "STEAD",            # 2GB file, server drops connection at ~1GB consistently (Phase 0 metadata available)
    "SPEI-GD",          # 21GB file, too large for current pipeline (Phase 0 metadata available)
}


class BatchProgress:
    """Overall batch progress state backed by PostgreSQL.

    Each (source_id, phase) pair is a row in the ``catalog_progress`` table.
    Every ``mark_*`` method commits immediately so progress is never lost.
    """

    def __init__(self, total: int = 0):
        self.total = total
        self.skipped = 0
        self.current_phase: Optional[int] = None
        self.current_source: Optional[str] = None
        self.started_at: Optional[str] = None

    # ------------------------------------------------------------------
    # Core mutation helpers
    # ------------------------------------------------------------------

    def mark_started(self, source_id: str, dataset_name: str, phase: int):
        self.current_source = source_id
        self.current_phase = phase
        with get_db_session() as session:
            row = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id, phase=phase)
                .first()
            )
            if row is None:
                row = CatalogProgressRow(
                    source_id=source_id,
                    dataset_name=dataset_name,
                    phase=phase,
                    status="processing",
                    started_at=datetime.utcnow(),
                )
                session.add(row)
            else:
                row.status = "processing"
                row.dataset_name = dataset_name
                row.error = None
                row.started_at = datetime.utcnow()
                row.completed_at = None

    def mark_completed(self, source_id: str, phase: Optional[int] = None):
        p = phase if phase is not None else (self.current_phase or 0)
        with get_db_session() as session:
            row = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id, phase=p)
                .first()
            )
            if row:
                row.status = "completed"
                row.completed_at = datetime.utcnow()

    def mark_failed(self, source_id: str, error: str, phase: Optional[int] = None):
        p = phase if phase is not None else (self.current_phase or 0)
        with get_db_session() as session:
            row = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id, phase=p)
                .first()
            )
            if row:
                row.status = "failed"
                row.error = error
                row.completed_at = datetime.utcnow()

    def is_completed(self, source_id: str, phase: Optional[int] = None) -> bool:
        with get_db_session() as session:
            if phase is not None:
                row = (
                    session.query(CatalogProgressRow)
                    .filter_by(source_id=source_id, phase=phase)
                    .first()
                )
                return row is not None and row.status == "completed"
            # Any phase completed?
            rows = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id, status="completed")
                .first()
            )
            return rows is not None

    def get_overall_status(self, source_id: str, target_phase: int) -> str:
        """Get display status considering target phase."""
        with get_db_session() as session:
            rows = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id)
                .all()
            )
            if not rows:
                return "pending"

            # Copy data within session to avoid DetachedInstanceError
            phases = {r.phase: r.status for r in rows}

        if phases.get(target_phase) == "completed":
            return "completed"
        if phases.get(target_phase) == "failed":
            return "failed"
        if phases.get(target_phase) == "processing":
            return "processing"
        if target_phase > 0 and phases.get(0) == "completed":
            return "metadata_only"
        return "pending"

    # ------------------------------------------------------------------
    # Bulk helpers — reduce DB round-trips from O(n) to O(1)
    # ------------------------------------------------------------------

    def get_completed_set(self, phase: int) -> set:
        """Return set of all completed source_ids for a phase in one query."""
        with get_db_session() as session:
            rows = (
                session.query(CatalogProgressRow.source_id)
                .filter_by(phase=phase, status="completed")
                .all()
            )
        return {r.source_id for r in rows}

    def mark_started_bulk(self, entries: List[CatalogEntry], phase: int):
        """Insert/update multiple rows to 'processing' in one transaction."""
        if not entries:
            return
        with get_db_session() as session:
            # Fetch existing rows for this phase in one query
            source_ids = [e.source_id for e in entries]
            existing = (
                session.query(CatalogProgressRow)
                .filter(
                    CatalogProgressRow.source_id.in_(source_ids),
                    CatalogProgressRow.phase == phase,
                )
                .all()
            )
            existing_map = {r.source_id: r for r in existing}

            now = datetime.utcnow()
            for entry in entries:
                row = existing_map.get(entry.source_id)
                if row is None:
                    session.add(CatalogProgressRow(
                        source_id=entry.source_id,
                        dataset_name=entry.dataset_name or "unknown",
                        phase=phase,
                        status="processing",
                        started_at=now,
                    ))
                else:
                    row.status = "processing"
                    row.dataset_name = entry.dataset_name or "unknown"
                    row.error = None
                    row.started_at = now
                    row.completed_at = None

    def mark_completed_bulk(self, source_ids: List[str], phase: int):
        """Mark multiple rows as completed in one UPDATE."""
        if not source_ids:
            return
        with get_db_session() as session:
            now = datetime.utcnow()
            session.query(CatalogProgressRow).filter(
                CatalogProgressRow.source_id.in_(source_ids),
                CatalogProgressRow.phase == phase,
            ).update(
                {"status": "completed", "completed_at": now},
                synchronize_session="fetch",
            )

    def mark_failed_bulk(self, source_id_error_pairs: List[tuple], phase: int):
        """Mark multiple rows as failed in one transaction.

        Args:
            source_id_error_pairs: list of (source_id, error_message) tuples
        """
        if not source_id_error_pairs:
            return
        with get_db_session() as session:
            now = datetime.utcnow()
            ids = [sid for sid, _ in source_id_error_pairs]
            rows = (
                session.query(CatalogProgressRow)
                .filter(
                    CatalogProgressRow.source_id.in_(ids),
                    CatalogProgressRow.phase == phase,
                )
                .all()
            )
            error_map = dict(source_id_error_pairs)
            for row in rows:
                row.status = "failed"
                row.error = error_map.get(row.source_id, "unknown error")
                row.completed_at = now

    def mark_interrupted(self) -> int:
        """Reset all 'processing' rows back to 'pending' (crash recovery)."""
        with get_db_session() as session:
            count = (
                session.query(CatalogProgressRow)
                .filter_by(status="processing")
                .update({"status": "pending", "completed_at": None})
            )
        return count

    def get_summary(self) -> Dict[str, Any]:
        """Build summary dict compatible with the old JSON-based format."""
        from sqlalchemy import func

        with get_db_session() as session:
            rows = (
                session.query(
                    CatalogProgressRow.phase,
                    CatalogProgressRow.status,
                    func.count().label("cnt"),
                )
                .group_by(CatalogProgressRow.phase, CatalogProgressRow.status)
                .all()
            )

            # Per-phase breakdown
            phase_counts: Dict[str, Dict[str, int]] = {}
            for phase_val, status, cnt in rows:
                p = str(phase_val)
                if p not in phase_counts:
                    phase_counts[p] = {"completed": 0, "failed": 0, "total": 0}
                phase_counts[p]["total"] += cnt
                if status == "completed":
                    phase_counts[p]["completed"] += cnt
                elif status == "failed":
                    phase_counts[p]["failed"] += cnt

            # Unique source statuses — query all rows grouped by source
            all_rows = session.query(CatalogProgressRow).all()

            # Copy data within session to avoid DetachedInstanceError
            source_phases: Dict[str, Dict[int, str]] = {}
            for r in all_rows:
                source_phases.setdefault(r.source_id, {})[r.phase] = r.status

        unique_processed = 0
        unique_failed = 0
        unique_metadata_only = 0
        for sid, phases in source_phases.items():
            has_data_phase = any(
                p != 0 and st == "completed" for p, st in phases.items()
            )
            has_failed = any(
                p != 0 and st == "failed" for p, st in phases.items()
            )
            if has_data_phase:
                unique_processed += 1
            elif has_failed:
                unique_failed += 1
            elif phases.get(0) == "completed":
                unique_metadata_only += 1

        effective_total = max(self.total, len(source_phases))
        effective_pending = max(
            0,
            effective_total - unique_processed - unique_failed
            - unique_metadata_only - self.skipped,
        )

        return {
            "total": effective_total,
            "processed": unique_processed,
            "failed": unique_failed,
            "metadata_only": unique_metadata_only,
            "skipped": self.skipped,
            "pending": effective_pending,
            "current_phase": self.current_phase,
            "current_source": self.current_source,
            "started_at": self.started_at,
            "updated_at": datetime.utcnow().isoformat(),
            "phases": phase_counts,
        }

    def get_source_info(self, source_id: str) -> Dict[str, Any]:
        """Get per-source progress info (used by /catalog/{row_index})."""
        with get_db_session() as session:
            rows = (
                session.query(CatalogProgressRow)
                .filter_by(source_id=source_id)
                .all()
            )
            if not rows:
                return {}
            # Copy data within session to avoid DetachedInstanceError
            phases = {str(r.phase): r.status for r in rows}
            errors = [r.error for r in rows if r.error]
            last_status = rows[-1].status
        return {
            "status": last_status,
            "phases": phases,
            "error": errors[-1] if errors else None,
        }


def run_batch_pipeline(
    excel_path: str,
    phases: Optional[List[int]] = None,
    dry_run: bool = False,
    resume: bool = True,
) -> Dict[str, Any]:
    """
    Run the batch catalog processing pipeline.

    Args:
        excel_path: Path to the Excel catalog file.
        phases: Which phases to process (default: [0] for metadata-only).
        dry_run: If True, only classify and report — no processing.
        resume: If True, skip already-completed sources.

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

    # Create progress tracker (state lives in PostgreSQL)
    if not resume:
        # Clear existing progress when not resuming
        with get_db_session() as session:
            session.query(CatalogProgressRow).delete()

    progress = BatchProgress(total=len(entries))
    progress.started_at = datetime.utcnow().isoformat()

    results: Dict[str, Any] = {"phases_run": phases, "total": len(entries)}

    for phase in sorted(phases):
        if phase not in grouped:
            continue

        phase_entries = grouped[phase] if phase > 0 else entries  # Phase 0 = all entries
        progress.current_phase = phase
        logger.info(f"Starting Phase {phase}: {len(phase_entries)} entries")

        if phase == 0:
            results["phase_0"] = _run_phase_0(phase_entries, progress, resume)
        elif phase in (1, 2):
            results[f"phase_{phase}"] = _run_phase_download(
                phase_entries, phase, progress, resume
            )
        elif phase == 3:
            results["phase_3"] = _run_phase_portal(phase_entries, progress, resume)
        elif phase == 4:
            # Phase 4 is metadata-only for manual sources
            results["phase_4"] = _run_phase_0(phase_entries, progress, resume)

    results["summary"] = progress.get_summary()
    catalog_logger.info(
        f"=== Batch pipeline finished | "
        f"skipped={progress.skipped} total={progress.total} ==="
    )
    return results


def _run_phase_0(
    entries: List[CatalogEntry],
    progress: BatchProgress,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 0: embed metadata only (batched).

    Uses bulk DB operations and batch embedding to minimize round-trips:
    - 1 DB query to get completed set (instead of 233 individual checks)
    - 1 DB transaction to mark all as started
    - ~4 batch embed + upsert calls (batch_size=64)
    - 1 DB transaction each for completed/failed
    """
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader("config/pipeline_config.yaml").load()
    embedder = TextEmbedder()
    db = VectorDatabase(config=config)

    # 1 DB query instead of 233 individual is_completed() calls
    if resume:
        completed = progress.get_completed_set(phase=0)
        to_process = [e for e in entries if e.source_id not in completed]
        skipped = len(entries) - len(to_process)
        progress.skipped += skipped
    else:
        to_process = list(entries)
        skipped = 0

    if not to_process:
        logger.info("Phase 0: nothing to process (all completed)")
        return {"processed": 0, "failed": 0, "skipped": len(entries)}

    logger.info(f"Phase 0: processing {len(to_process)} entries (skipped {skipped})")
    catalog_logger.info(f"Phase 0: batch processing {len(to_process)} entries")

    # 1 DB transaction to mark all as started
    progress.mark_started_bulk(to_process, phase=0)

    # Batch embed + upsert — RTX 5090 handles 233 texts in <0.2s,
    # so use large batches to minimize Qdrant upsert round-trips
    result = process_metadata_batch(to_process, embedder, db, batch_size=256)

    # 1 DB transaction each for completed/failed
    progress.mark_completed_bulk(result["succeeded_ids"], phase=0)
    progress.mark_failed_bulk(result["failed_entries"], phase=0)

    processed = result["processed"]
    failed = result["failed"]

    catalog_logger.info(
        f"Phase 0: batch complete — processed={processed}, failed={failed}, skipped={skipped}"
    )

    return {"processed": processed, "failed": failed, "skipped": skipped}


# --- Retry & resource guard settings ---
MAX_RETRIES = 3
RETRY_BACKOFF = [5, 10, 20]  # seconds between attempts

MEMORY_WARN_PCT = 85   # trigger GC + wait
MEMORY_ABORT_PCT = 90  # stop batch if still above after GC

RASTER_TIMEOUT_SEC = 30 * 60  # 30 minutes for load + embed


def _check_memory_pressure() -> bool:
    """Return True if it's safe to continue, False if batch should stop.

    On high memory (>85 %): force GC, wait 30 s, re-check.
    If still >90 % after GC → return False (caller should stop).
    """
    try:
        import psutil
    except ImportError:
        return True  # psutil not installed — skip guard

    mem = psutil.virtual_memory()
    if mem.percent <= MEMORY_WARN_PCT:
        return True

    catalog_logger.warning(
        f"Memory pressure high ({mem.percent:.0f}% used). "
        f"Running gc.collect() and waiting 30 s …"
    )
    gc.collect()
    time.sleep(30)

    mem = psutil.virtual_memory()
    if mem.percent > MEMORY_ABORT_PCT:
        catalog_logger.error(
            f"Memory still critical ({mem.percent:.0f}% used) after GC. "
            f"Stopping batch to prevent OOM."
        )
        return False

    catalog_logger.info(f"Memory dropped to {mem.percent:.0f}% after GC — continuing.")
    return True


FORMAT_TO_EXT = {
    "netcdf": ".nc", "grib": ".grib", "hdf5": ".h5",
    "geotiff": ".tif", "csv": ".csv", "zip": ".zip", "gz": ".gz",
    "ascii": ".asc", "zarr": ".zarr", "tar": ".tar",
}

# Size used when HEAD request fails or Content-Length is missing (sorts last)
_UNKNOWN_SIZE = 10 * 1024**3  # 10 GB


def _prefetch_sizes(entries: List[CatalogEntry], phase: int) -> List[CatalogEntry]:
    """Sort entries by download file size (smallest first) using HEAD requests.

    Does a HEAD request per unique (dataset_name, url) pair to read
    Content-Length.  Entries whose size is unknown sort to the end.
    """
    import requests as http_requests

    # Build unique URLs to probe
    url_map: Dict[str, str] = {}  # dedup_key -> url
    entry_keys: Dict[str, str] = {}  # source_id -> dedup_key
    for entry in entries:
        url = DIRECT_DOWNLOAD_URLS.get(entry.dataset_name, entry.link or "").strip()
        if not url:
            continue
        dedup_key = f"{entry.dataset_name}||{url}"
        url_map[dedup_key] = url
        entry_keys[entry.source_id] = dedup_key

    # HEAD requests (parallel with ThreadPoolExecutor for speed)
    size_cache: Dict[str, int] = {}

    def _head(dedup_key_url):
        dk, url = dedup_key_url
        try:
            resp = http_requests.head(
                url, timeout=10, allow_redirects=True,
                headers={"User-Agent": "ClimateRAG/1.0"},
            )
            cl = resp.headers.get("Content-Length")
            return dk, int(cl) if cl else _UNKNOWN_SIZE
        except Exception:
            return dk, _UNKNOWN_SIZE

    unique_items = list(url_map.items())
    catalog_logger.info(
        f"Phase {phase}: prefetching sizes for {len(unique_items)} unique URLs …"
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        for dk, size in executor.map(_head, unique_items):
            size_cache[dk] = size

    # Log size distribution
    known = {dk: s for dk, s in size_cache.items() if s != _UNKNOWN_SIZE}
    if known:
        sizes_mb = sorted(s / 1024**2 for s in known.values())
        catalog_logger.info(
            f"Phase {phase}: size prefetch done — "
            f"{len(known)} known, {len(size_cache) - len(known)} unknown | "
            f"range: {sizes_mb[0]:.1f} MB … {sizes_mb[-1]:.1f} MB"
        )
    else:
        catalog_logger.info(f"Phase {phase}: no Content-Length headers returned")

    # Sort entries by size (smallest first, unknown last)
    def _sort_key(entry):
        dk = entry_keys.get(entry.source_id)
        return size_cache.get(dk, _UNKNOWN_SIZE) if dk else _UNKNOWN_SIZE

    sorted_entries = sorted(entries, key=_sort_key)
    return sorted_entries

def _run_phase_download(
    entries: List[CatalogEntry],
    phase: int,
    progress: BatchProgress,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 1/2: download + process + embed."""
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto
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

    # Track datasets already processed in this run to avoid re-downloading
    # the same file for duplicate rows (e.g. SLOCLIM appears on 3 rows for
    # different hazards but uses the exact same download URL).
    datasets_done_this_run: set = set()

    # Prefetch completed set — 1 DB query instead of per-entry is_completed()
    completed_set = progress.get_completed_set(phase=phase) if resume else set()

    # Pre-filter entries that will definitely be skipped, so we only HEAD-request
    # URLs we actually intend to download.
    download_candidates = []
    pre_skipped = 0
    for entry in entries:
        if resume and entry.source_id in completed_set:
            pre_skipped += 1
            continue
        if not entry.link:
            pre_skipped += 1
            continue
        if entry.dataset_name in SKIP_PHASE1:
            pre_skipped += 1
            continue
        download_candidates.append(entry)

    # Sort by file size (smallest first) — HEAD requests on remaining candidates
    if download_candidates:
        download_candidates = _prefetch_sizes(download_candidates, phase)

    # Re-combine: we'll iterate download_candidates and track skipped count separately
    skipped += pre_skipped

    # Disable HNSW indexing during bulk ingestion (rebuild once at the end)
    db.disable_indexing()
    try:
      for entry in download_candidates:
        # Deduplicate: same dataset_name + same download URL = same data.
        # Mark all duplicate rows as completed (they share embeddings via Phase 0).
        download_url = DIRECT_DOWNLOAD_URLS.get(entry.dataset_name, entry.link).strip()
        dedup_key = f"{entry.dataset_name}||{download_url}"
        if dedup_key in datasets_done_this_run:
            catalog_logger.info(
                f"Phase {phase}: skipping duplicate {entry.dataset_name} "
                f"(row {entry.row_index}, already processed same URL)"
            )
            progress.mark_started(entry.source_id, entry.dataset_name or "unknown", phase)
            progress.mark_completed(entry.source_id, phase=phase)
            skipped += 1
            continue

        # --- Memory guard ---
        if not _check_memory_pressure():
            catalog_logger.error("Batch stopped due to memory pressure. Remaining entries stay pending.")
            break

        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", phase)

        # --- Per-entry retry loop ---
        entry_succeeded = False
        last_error = ""

        for attempt in range(1, MAX_RETRIES + 1):
            tmp_path = None
            try:
                if attempt > 1:
                    backoff = RETRY_BACKOFF[min(attempt - 2, len(RETRY_BACKOFF) - 1)]
                    catalog_logger.info(
                        f"Phase {phase}: retry {attempt}/{MAX_RETRIES} for "
                        f"{entry.dataset_name} after {backoff}s backoff"
                    )
                    time.sleep(backoff)
                else:
                    catalog_logger.info(
                        f"Phase {phase}: attempt {attempt}/{MAX_RETRIES} for {entry.dataset_name}"
                    )

                # Use direct download URL override if available, otherwise use catalog link
                url = DIRECT_DOWNLOAD_URLS.get(entry.dataset_name, entry.link).strip()
                if url != entry.link.strip():
                    catalog_logger.info(f"Phase {phase}: using override URL for {entry.dataset_name}")

                # --- Disk space guard ---
                import shutil
                try:
                    free_bytes = shutil.disk_usage("/app/data").free
                except OSError:
                    free_bytes = shutil.disk_usage(".").free
                if free_bytes < 5 * 1024**3:  # Less than 5 GB free
                    catalog_logger.error(
                        f"Disk space critically low ({free_bytes / 1e9:.1f} GB free). Stopping downloads."
                    )
                    return {"processed": processed, "failed": failed, "skipped": skipped}

                # Download to temp file
                logger.info(f"Phase {phase}: downloading {entry.dataset_name} from {url}")

                try:
                    resp = http_requests.get(url, timeout=(30, 600), stream=True,
                                             headers={"User-Agent": "ClimateRAG/1.0"})
                    resp.raise_for_status()
                except http_requests.exceptions.SSLError:
                    catalog_logger.warning(
                        f"Phase {phase}: SSL verification failed for {entry.dataset_name}, "
                        "retrying without verification"
                    )
                    resp = http_requests.get(url, timeout=(30, 600), stream=True, verify=False,
                                             headers={"User-Agent": "ClimateRAG/1.0"})
                    resp.raise_for_status()

                # Check Content-Type before downloading body
                content_type = resp.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    raise ValueError(
                        f"Server returned HTML (Content-Type: {content_type}). "
                        "URL is likely a portal page, not a direct file download."
                    )

                # Detect extension from URL — map format name to proper file extension
                from src.climate_embeddings.loaders.detect_format import detect_format_from_url
                fmt = detect_format_from_url(url)
                ext = FORMAT_TO_EXT.get(fmt, ".nc")

                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name

                # --- Post-download content validation: detect HTML login pages ---
                with open(tmp_path, "rb") as f:
                    head_bytes = f.read(512)
                head_text = head_bytes.decode("utf-8", errors="ignore").lower().strip()
                if any(tag in head_text for tag in ["<!doctype", "<html", "<head", "<body", "<?xml"]):
                    raise ValueError(
                        "Downloaded file contains HTML/XML (likely a login/redirect page, not data). "
                        "This dataset probably requires authentication."
                    )

                # --- Load raster with timeout ---
                def _load_raster():
                    return load_raster_auto(tmp_path)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_load_raster)
                    try:
                        raster_result = future.result(timeout=RASTER_TIMEOUT_SEC)
                    except FuturesTimeoutError:
                        future.cancel()
                        raise TimeoutError(
                            f"Raster loading timed out after {RASTER_TIMEOUT_SEC // 60} min "
                            f"for {entry.dataset_name}"
                        )

                # --- Stream chunks through async embed + upsert pipeline ---
                # Overlap GPU embedding with Qdrant network I/O using a background thread.
                UPSERT_BATCH_SIZE = 2000
                batch_ids: list = []
                batch_texts: list = []
                batch_metadatas: list = []
                total_chunks = 0
                upsert_executor = ThreadPoolExecutor(max_workers=1)
                pending_upsert_future = None

                def _wait_for_pending_upsert():
                    """Wait for previous async upsert to complete (backpressure)."""
                    nonlocal pending_upsert_future
                    if pending_upsert_future is not None:
                        try:
                            pending_upsert_future.result(timeout=300)
                        except Exception as upsert_err:
                            catalog_logger.error(f"Async upsert failed: {upsert_err}")
                        pending_upsert_future = None

                def _flush_batch():
                    """Embed all texts on GPU, then submit upsert to background thread."""
                    nonlocal pending_upsert_future
                    if not batch_ids:
                        return
                    t0 = time.time()
                    vecs = embedder.embed_documents(batch_texts)
                    embed_time = time.time() - t0

                    # Wait for any previous upsert to finish before starting next
                    _wait_for_pending_upsert()

                    # Capture batch data for the background thread
                    upsert_ids = list(batch_ids)
                    upsert_vecs = [v.tolist() for v in vecs]
                    upsert_metas = list(batch_metadatas)
                    chunks_so_far = total_chunks

                    def _do_upsert():
                        t1 = time.time()
                        db.add_embeddings(
                            ids=upsert_ids,
                            embeddings=upsert_vecs,
                            metadatas=upsert_metas,
                        )
                        catalog_logger.info(
                            f"Phase {phase}: {entry.dataset_name} — "
                            f"upserted batch ({chunks_so_far} chunks so far, "
                            f"embed={embed_time:.1f}s, upsert={time.time() - t1:.1f}s)"
                        )

                    pending_upsert_future = upsert_executor.submit(_do_upsert)

                    batch_ids.clear()
                    batch_texts.clear()
                    batch_metadatas.clear()

                for chunk in raster_result.chunk_iterator:
                    data = chunk.data
                    valid = data[np.isfinite(data)]
                    if valid.size == 0:
                        continue

                    # 8-dim stats vector
                    mn = float(np.min(valid))
                    mx = float(np.max(valid))
                    stats = [
                        float(np.mean(valid)), float(np.std(valid)),
                        mn, mx,
                        float(np.percentile(valid, 10)),
                        float(np.percentile(valid, 50)),
                        float(np.percentile(valid, 90)),
                        mx - mn,
                    ]

                    meta = ClimateChunkMetadata.from_chunk_metadata(
                        raw_metadata=chunk.metadata,
                        stats_vector=stats,
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

                    batch_ids.append(f"{entry.source_id}_chunk_{total_chunks}")
                    batch_texts.append(text)
                    batch_metadatas.append(meta_dict)
                    total_chunks += 1

                    if len(batch_ids) >= UPSERT_BATCH_SIZE:
                        _flush_batch()

                # Flush remaining
                _flush_batch()
                _wait_for_pending_upsert()
                upsert_executor.shutdown(wait=False)

                if total_chunks == 0:
                    raise ValueError(
                        f"Loader produced 0 data chunks from {url} — file may be corrupt, "
                        f"unsupported format, or xarray engine failed (check logs for details)"
                    )

                progress.mark_completed(entry.source_id, phase=phase)
                processed += 1
                entry_succeeded = True
                datasets_done_this_run.add(dedup_key)
                catalog_logger.info(
                    f"Phase {phase}: completed {entry.dataset_name} "
                    f"({total_chunks} chunks, attempt {attempt}/{MAX_RETRIES})"
                )

                # Update shelve SourceStore status
                try:
                    from src.sources import get_source_store
                    store = get_source_store()
                    store.update_processing_status(entry.source_id, "completed")
                except Exception as store_err:
                    catalog_logger.warning(f"Could not update SourceStore for {entry.source_id}: {store_err}")

                break  # Success — exit retry loop

            except Exception as e:
                last_error = str(e)
                tb_str = traceback.format_exc()
                catalog_logger.warning(
                    f"Phase {phase}: attempt {attempt}/{MAX_RETRIES} FAILED for "
                    f"{entry.dataset_name}: {e}\n{tb_str}"
                )

            finally:
                # Always clean up temp file
                if tmp_path:
                    Path(tmp_path).unlink(missing_ok=True)

        # All retries exhausted
        if not entry_succeeded:
            progress.mark_failed(entry.source_id, last_error, phase=phase)
            failed += 1
            catalog_logger.error(
                f"Phase {phase}: FAILED {entry.dataset_name} after {MAX_RETRIES} attempts: {last_error}"
            )

    finally:
        # Re-enable HNSW indexing — triggers single optimized index build
        db.enable_indexing()

    return {"processed": processed, "failed": failed, "skipped": skipped}


def _run_phase_portal(
    entries: List[CatalogEntry],
    progress: BatchProgress,
    resume: bool,
) -> Dict[str, int]:
    """Run Phase 3: API portal downloads (CDS, ESGF, etc.)."""
    from src.catalog.portal_adapters import get_adapter

    processed = 0
    failed = 0
    skipped = 0

    # Prefetch completed set — 1 DB query instead of per-entry is_completed()
    completed_set = progress.get_completed_set(phase=3) if resume else set()

    for entry in entries:
        if resume and entry.source_id in completed_set:
            skipped += 1
            continue

        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", 3)

        try:
            adapter = get_adapter(entry)
            if adapter is None:
                logger.warning(f"No adapter for {entry.dataset_name} ({entry.link})")
                progress.mark_failed(entry.source_id, "No portal adapter available", phase=3)
                failed += 1
                continue

            ok = adapter.download_and_process(entry)
            if ok:
                progress.mark_completed(entry.source_id, phase=3)
                processed += 1
            else:
                progress.mark_failed(entry.source_id, "Adapter returned False", phase=3)
                failed += 1

        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e), phase=3)
            failed += 1
            catalog_logger.error(f"Phase 3: FAILED {entry.dataset_name}: {e}\n{tb}")

    return {"processed": processed, "failed": failed, "skipped": skipped}


def get_progress() -> Dict[str, Any]:
    """Get current batch progress (for API endpoint)."""
    progress = BatchProgress()
    return progress.get_summary()


def retry_failed(excel_path: str) -> Dict[str, Any]:
    """Re-run all failed sources."""
    with get_db_session() as session:
        failed_rows = (
            session.query(CatalogProgressRow)
            .filter_by(status="failed")
            .all()
        )
        if not failed_rows:
            return {"message": "No failed sources to retry", "count": 0}

        failed_count = len(failed_rows)
        for row in failed_rows:
            row.status = "pending"
            row.error = None
            row.completed_at = None

    logger.info(f"Reset {failed_count} failed rows to pending for retry")

    return run_batch_pipeline(
        excel_path=excel_path,
        phases=[0, 1, 2, 3],
        resume=True,
    )
