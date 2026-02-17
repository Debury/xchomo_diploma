"""
Batch orchestrator for processing catalog entries across all phases.

Supports resume (via JSON state file), dry-run, and phase filtering.
"""

import json
import logging
import os
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
}


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
        if source_id not in self.sources:
            self.sources[source_id] = {
                "dataset_name": dataset_name,
                "phases": {},
            }
        self.sources[source_id]["phase"] = phase
        self.sources[source_id].setdefault("phases", {})
        self.sources[source_id]["phases"][str(phase)] = "processing"
        self.sources[source_id]["started_at"] = datetime.now().isoformat()
        # Keep legacy "status" field in sync with current phase
        self.sources[source_id]["status"] = "processing"
        self.updated_at = datetime.now().isoformat()

    def mark_completed(self, source_id: str, phase: Optional[int] = None):
        if source_id in self.sources:
            p = str(phase if phase is not None else self.sources[source_id].get("phase", 0))
            self.sources[source_id].setdefault("phases", {})
            self.sources[source_id]["phases"][p] = "completed"
            self.sources[source_id]["completed_at"] = datetime.now().isoformat()
            self.sources[source_id]["status"] = "completed"
        self.processed += 1
        self.updated_at = datetime.now().isoformat()

    def mark_failed(self, source_id: str, error: str, phase: Optional[int] = None):
        if source_id in self.sources:
            p = str(phase if phase is not None else self.sources[source_id].get("phase", 0))
            self.sources[source_id].setdefault("phases", {})
            self.sources[source_id]["phases"][p] = "failed"
            self.sources[source_id]["error"] = error
            self.sources[source_id]["completed_at"] = datetime.now().isoformat()
            self.sources[source_id]["status"] = "failed"
        self.failed += 1
        self.updated_at = datetime.now().isoformat()

    def is_completed(self, source_id: str, phase: Optional[int] = None) -> bool:
        """Check if a specific phase is completed for a source."""
        entry = self.sources.get(source_id, {})
        if phase is not None:
            phases = entry.get("phases", {})
            return phases.get(str(phase)) == "completed"
        # Legacy: check overall status
        return entry.get("status") == "completed"

    def get_overall_status(self, source_id: str, target_phase: int) -> str:
        """
        Get display status considering target phase.

        Returns:
            "completed" — target phase is done
            "metadata_only" — only Phase 0 done, target > 0
            "processing" / "failed" / "pending" — current state
        """
        entry = self.sources.get(source_id, {})
        if not entry:
            return "pending"

        phases = entry.get("phases", {})

        # Migrate legacy entries: old format has "status" but no "phases" dict
        if not phases and entry.get("status") == "completed":
            old_phase = entry.get("phase", 0)
            phases = {str(old_phase): "completed"}

        # Check if target phase is done
        if phases.get(str(target_phase)) == "completed":
            return "completed"

        # Check if target phase failed
        if phases.get(str(target_phase)) == "failed":
            return "failed"

        # Check if target phase is processing
        if phases.get(str(target_phase)) == "processing":
            return "processing"

        # Phase 0 done but target phase > 0 — metadata only
        if target_phase > 0 and phases.get("0") == "completed":
            return "metadata_only"

        # Legacy fallback
        return entry.get("status", "pending")

    def get_summary(self) -> Dict[str, Any]:
        pending = self.total - self.processed - self.failed - self.skipped
        # Build per-phase breakdown from sources dict
        phase_counts: Dict[str, Dict[str, int]] = {}
        for sid, info in self.sources.items():
            phases = info.get("phases", {})
            # Determine which phase this source belongs to
            source_phase = str(info.get("phase", 0))
            for p, status in phases.items():
                if p not in phase_counts:
                    phase_counts[p] = {"completed": 0, "failed": 0, "total": 0}
                phase_counts[p]["total"] += 1
                if status == "completed":
                    phase_counts[p]["completed"] += 1
                elif status == "failed":
                    phase_counts[p]["failed"] += 1
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
            "phases": phase_counts,
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
        if resume and progress.is_completed(entry.source_id, phase=0):
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
                progress.mark_completed(entry.source_id, phase=0)
                processed += 1
                catalog_logger.info(f"Phase 0: completed {entry.dataset_name}")
            else:
                progress.mark_failed(entry.source_id, "process_metadata_only returned False", phase=0)
                failed += 1
                catalog_logger.warning(f"Phase 0: returned False for {entry.dataset_name}")
        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e), phase=0)
            failed += 1
            catalog_logger.error(f"Phase 0: FAILED {entry.dataset_name}: {e}\n{tb}")

        # Save progress periodically (every 10 entries)
        if (processed + failed) % 10 == 0:
            progress.save(progress_path)

    progress.save(progress_path)
    return {"processed": processed, "failed": failed, "skipped": len(entries) - len(to_process)}


FORMAT_TO_EXT = {
    "netcdf": ".nc", "grib": ".grib", "hdf5": ".h5",
    "geotiff": ".tif", "csv": ".csv", "zip": ".zip", "gz": ".gz",
    "ascii": ".asc", "zarr": ".zarr", "tar": ".tar",
}

MAX_DOWNLOAD_SIZE_MB = int(os.getenv("MAX_DOWNLOAD_SIZE_MB", "2000"))


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
        if resume and progress.is_completed(entry.source_id, phase=phase):
            skipped += 1
            continue

        if not entry.link:
            skipped += 1
            continue

        # Skip datasets known to require auth or have no direct downloads
        if entry.dataset_name in SKIP_PHASE1:
            catalog_logger.info(f"Phase {phase}: skipping {entry.dataset_name} (in SKIP_PHASE1)")
            skipped += 1
            continue

        progress.mark_started(entry.source_id, entry.dataset_name or "unknown", phase)

        tmp_path = None
        try:
            # Use direct download URL override if available, otherwise use catalog link
            url = DIRECT_DOWNLOAD_URLS.get(entry.dataset_name, entry.link).strip()
            if url != entry.link.strip():
                catalog_logger.info(f"Phase {phase}: using override URL for {entry.dataset_name}")

            # --- Size guard: HEAD request to check Content-Length ---
            try:
                head_resp = http_requests.head(url, timeout=30, allow_redirects=True,
                                               headers={"User-Agent": "ClimateRAG/1.0"})
                content_length = int(head_resp.headers.get("Content-Length", 0))
                max_size = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024
                if content_length > max_size:
                    msg = f"File too large: {content_length / 1e9:.1f} GB (limit: {max_size / 1e9:.1f} GB)"
                    progress.mark_failed(entry.source_id, msg, phase=phase)
                    catalog_logger.warning(f"Skipping {entry.dataset_name}: {msg}")
                    failed += 1
                    continue
            except Exception as head_err:
                catalog_logger.warning(f"HEAD check failed for {entry.dataset_name}: {head_err} — proceeding anyway")

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
                progress.save(progress_path)
                break

            # Download to temp file
            logger.info(f"Phase {phase}: downloading {entry.dataset_name} from {url}")

            resp = http_requests.get(url, timeout=600, stream=True,
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
                bytes_written = 0
                max_bytes = MAX_DOWNLOAD_SIZE_MB * 1024 * 1024
                for chunk in resp.iter_content(chunk_size=8192):
                    bytes_written += len(chunk)
                    if bytes_written > max_bytes:
                        raise ValueError(
                            f"Download exceeded {MAX_DOWNLOAD_SIZE_MB} MB limit "
                            f"(streamed {bytes_written / 1e6:.0f} MB so far)"
                        )
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

            # Load and process
            result = load_raster_auto(tmp_path)
            embeddings_data = raster_to_embeddings(result)

            if not embeddings_data:
                raise ValueError(
                    f"Loader produced 0 data chunks from {url} — file may be corrupt, "
                    f"unsupported format, or xarray engine failed (check logs for details)"
                )

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

            progress.mark_completed(entry.source_id, phase=phase)
            processed += 1
            catalog_logger.info(f"Phase {phase}: completed {entry.dataset_name} ({len(embeddings_data)} chunks)")

            # Update shelve SourceStore status
            try:
                from src.sources import get_source_store
                store = get_source_store()
                store.update_processing_status(entry.source_id, "completed")
            except Exception as store_err:
                catalog_logger.warning(f"Could not update SourceStore for {entry.source_id}: {store_err}")

        except Exception as e:
            tb = traceback.format_exc()
            progress.mark_failed(entry.source_id, str(e), phase=phase)
            failed += 1
            catalog_logger.error(f"Phase {phase}: FAILED {entry.dataset_name}: {e}\n{tb}")

        finally:
            # Always clean up temp file
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

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
        if resume and progress.is_completed(entry.source_id, phase=3):
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
    failed_ids = set()
    for sid, info in progress.sources.items():
        phases = info.get("phases", {})
        if any(v == "failed" for v in phases.values()) or info.get("status") == "failed":
            failed_ids.add(sid)

    if not failed_ids:
        return {"message": "No failed sources to retry", "count": 0}

    # Reset failed phases to pending
    for sid in failed_ids:
        progress.sources[sid]["status"] = "pending"
        progress.sources[sid].pop("error", None)
        phases = progress.sources[sid].get("phases", {})
        for p, status in list(phases.items()):
            if status == "failed":
                phases[p] = "pending"
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
