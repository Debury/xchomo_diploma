"""Reusable multi-file ingester for pre-extracted raster directories.

The original SPEI-GD bespoke ingest script lived under
``data/uploads/spei_gd/ingest.py`` — it iterated 40 NetCDFs, chunked
each via the catalog raster pipeline, embedded on GPU, and upserted to
Qdrant. The pattern (resume marker + heartbeat + flush in BATCH=1000)
is generic enough that any future zip-of-rasters upload should use the
same code path. This module is that code path.

Usage from a CLI shim or from ``/sources/upload`` after a zip extract::

    from src.catalog.multi_file_ingester import ingest_directory
    ingest_directory(
        directory="/app/data/uploads/<id>/extracted",
        source_id="catalog_FOO_manual",
        dataset_name="FOO",
        hazard_type="Drought",
        glob_pattern="*.nc",
    )

Resume support: ``done.txt`` lives next to the directory; re-running
the same call picks up where the previous run stopped.

Liveness for the Catalog UI: ``ingest_state.json`` carries pid +
heartbeat so the badge can flip from ⏳ Running to ⚠ Killed when the
process dies. Catalog UI's "Resume" button just calls back into the
same script that wrote the state — no special dispatch needed.
"""
import glob
import json
import logging
import os
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_BATCH = 1000
STATE_PATH = "/app/data/ingest_state.json"


def _load_done(done_path: str) -> set:
    if not os.path.exists(done_path):
        return set()
    with open(done_path) as f:
        return {line.strip() for line in f if line.strip()}


def _mark_done(done_path: str, name: str) -> None:
    with open(done_path, "a") as f:
        f.write(name + "\n")


def _update_state(
    *,
    source_id: str,
    dataset_name: str,
    done_count: int,
    total_count: int,
    script_path: str,
    finished: bool = False,
) -> None:
    """Write per-source ingest progress to the shared state file.

    Includes pid + heartbeat so the Catalog UI can decide between
    "running" and "killed" without the user telling it. See
    ``web_api/routes/catalog.py`` for the consumer side.
    """
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH) as f:
                state = json.load(f) or {}
        else:
            state = {}
        state[source_id] = {
            "dataset_name": dataset_name,
            "done_files": done_count,
            "total_files": total_count,
            "updated_at": datetime.utcnow().isoformat(),
            "script": script_path,
            "pid": os.getpid(),
            "finished": finished,
        }
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log.warning(f"could not update {STATE_PATH}: {e}")


def extract_zip_if_needed(zip_path: str, target_dir: str) -> str:
    """Extract ``zip_path`` into ``target_dir`` once. Idempotent.

    A flag file ``.extracted`` marks completion so re-running on a
    persistent volume doesn't re-extract a 22 GB zip every restart.
    Returns ``target_dir`` so callers can chain into ``ingest_directory``.
    """
    target = Path(target_dir)
    flag = target / ".extracted"
    if flag.exists():
        log.info(f"zip already extracted at {target} — skipping")
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    log.info(f"extracting {zip_path} → {target} (this may take a while)")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target)
    flag.touch()
    log.info(f"extracted in {time.time() - t0:.0f}s")
    return str(target)


def ingest_directory(
    *,
    directory: str,
    source_id: str,
    dataset_name: str,
    hazard_type: str = "",
    glob_pattern: str = "*.nc",
    catalog_source: str = "manual_upload",
    script_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH,
    done_marker: Optional[str] = None,
    zip_path: Optional[str] = None,
) -> dict:
    """Ingest every file matching ``glob_pattern`` under ``directory`` to
    Qdrant, tagging chunks with the supplied source_id / dataset_name.

    Returns a summary dict with counts so a caller can log or surface it.
    Does not raise on per-file errors — failed files are skipped and
    counted in ``failed_files``; the run continues so one bad file
    doesn't block 39 good ones.
    """
    # Lazy imports — heavy ML deps shouldn't load just because someone
    # imports this module from the API process.
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto
    from src.climate_embeddings.schema import (
        ClimateChunkMetadata,
        generate_human_readable_text,
    )
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    # If a zip path is supplied, extract it (idempotently) before globbing.
    # This is what makes resume robust across container restarts — the
    # `.extracted` flag avoids re-extracting on every retry.
    if zip_path:
        directory = extract_zip_if_needed(zip_path, directory)

    config = ConfigLoader("config/pipeline_config.yaml").load()
    embedder = TextEmbedder()
    db = VectorDatabase(config=config)

    if done_marker is None:
        done_marker = str(Path(directory).parent / "done.txt")
    if script_path is None:
        # Best-effort guess for the resume button — the caller usually
        # passes its own __file__ in.
        script_path = "<unknown>"

    done = _load_done(done_marker)
    if done:
        log.info(f"resume: {len(done)} files already done; skipping")

    files = sorted(glob.glob(str(Path(directory) / "**" / glob_pattern), recursive=True))
    if not files:
        # Try non-recursive too in case glob_pattern already includes path
        files = sorted(glob.glob(str(Path(directory) / glob_pattern)))
    log.info(f"found {len(files)} candidate files under {directory}")

    ids: list = []
    texts: list = []
    metas: list = []
    total_chunks = 0
    failed_files = 0

    def flush() -> None:
        nonlocal ids, texts, metas
        if not ids:
            return
        t0 = time.time()
        vecs = embedder.embed_documents(texts)
        embed_t = time.time() - t0
        t0 = time.time()
        db.add_embeddings(
            ids=ids,
            embeddings=[v.tolist() for v in vecs],
            metadatas=metas,
        )
        upsert_t = time.time() - t0
        log.info(
            f"upserted {len(ids)} chunks (running total {total_chunks}, "
            f"embed={embed_t:.1f}s, upsert={upsert_t:.1f}s)"
        )
        ids = []
        texts = []
        metas = []

    _update_state(
        source_id=source_id, dataset_name=dataset_name,
        done_count=len(done), total_count=len(files),
        script_path=script_path,
    )

    for fi, path in enumerate(files, 1):
        name = os.path.basename(path)
        if name in done:
            log.info(f"[{fi}/{len(files)}] skip {name} (already done)")
            continue

        log.info(
            f"[{fi}/{len(files)}] loading {name} "
            f"({os.path.getsize(path)/1e6:.0f} MB)"
        )
        try:
            raster = load_raster_auto(path)
        except Exception as e:
            log.error(f"  load failed for {name}: {e}")
            failed_files += 1
            continue

        file_chunks = 0
        try:
            for c in raster.chunk_iterator:
                valid = c.data[np.isfinite(c.data)]
                if valid.size == 0:
                    continue
                mn, mx = float(np.min(valid)), float(np.max(valid))
                stats = [
                    float(np.mean(valid)), float(np.std(valid)), mn, mx,
                    float(np.percentile(valid, 10)),
                    float(np.percentile(valid, 50)),
                    float(np.percentile(valid, 90)),
                    mx - mn,
                ]
                meta = ClimateChunkMetadata.from_chunk_metadata(
                    raw_metadata=c.metadata, stats_vector=stats,
                    source_id=source_id, dataset_name=dataset_name,
                ).to_dict()
                meta["catalog_source"] = catalog_source
                meta["hazard_type"] = hazard_type
                meta["source_file"] = name
                text = generate_human_readable_text(meta)
                ids.append(str(uuid.uuid4()))
                texts.append(text)
                metas.append(meta)
                total_chunks += 1
                file_chunks += 1
                if len(ids) >= batch_size:
                    flush()
        except Exception as e:
            log.error(f"  chunking failed for {name}: {e}")
            failed_files += 1
            # Still flush whatever we did manage to chunk before failure
            flush()
            continue

        # Flush this file's tail before marking it done — a kill mid-flush
        # mustn't leave a partially-ingested file marked complete.
        flush()
        _mark_done(done_marker, name)
        done.add(name)
        _update_state(
            source_id=source_id, dataset_name=dataset_name,
            done_count=len(done), total_count=len(files),
            script_path=script_path,
        )
        log.info(f"  → {file_chunks} chunks from {name} (file marked done)")

    _update_state(
        source_id=source_id, dataset_name=dataset_name,
        done_count=len(done), total_count=len(files),
        script_path=script_path, finished=True,
    )
    log.info(
        f"DONE {dataset_name}: {total_chunks} chunks total "
        f"({failed_files} files failed)"
    )
    return {
        "dataset_name": dataset_name,
        "total_chunks": total_chunks,
        "files_processed": len(done),
        "files_total": len(files),
        "failed_files": failed_files,
    }
