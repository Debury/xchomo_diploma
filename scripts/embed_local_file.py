"""One-shot: embed a local raster file as a manual catalog source.

Used to populate Qdrant for catalog datasets that aren't reachable by HTTP
download (e.g. MSWEP — public but only via Google Drive). Mirrors the
chunk → embed → upsert flow in batch_orchestrator._run_phase_download
but skips the download step entirely.

Usage (from inside the web-api container):
    python scripts/embed_local_file.py \
        --path /app/data/raw/mswep_nrt_3hourly_2024.nc \
        --source-id catalog_MSWEP_manual \
        --dataset-name MSWEP \
        --hazard "Mean precipitation"
"""

import argparse
import logging
import sys
import time
import uuid

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_local_file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Local raster file path")
    ap.add_argument("--source-id", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--hazard", default=None)
    ap.add_argument("--impact-sector", default=None)
    args = ap.parse_args()

    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto
    from src.climate_embeddings.schema import (
        ClimateChunkMetadata, generate_human_readable_text,
    )
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    config = ConfigLoader("config/pipeline_config.yaml").load()
    embedder = TextEmbedder()
    db = VectorDatabase(config=config)

    log.info(f"Loading raster {args.path} …")
    raster_result = load_raster_auto(args.path)
    log.info("Raster loaded — streaming chunks now")

    BATCH = 2000
    batch_ids: list = []
    batch_texts: list = []
    batch_metas: list = []
    total = 0
    all_chunk_metas: list = []

    def flush() -> None:
        nonlocal batch_ids, batch_texts, batch_metas
        if not batch_ids:
            return
        t0 = time.time()
        vecs = embedder.embed_documents(batch_texts)
        log.info(f"Embedded {len(batch_ids)} chunks in {time.time()-t0:.1f}s")
        t0 = time.time()
        db.add_embeddings(
            ids=batch_ids,
            embeddings=[v.tolist() for v in vecs],
            metadatas=batch_metas,
        )
        log.info(f"Upserted {len(batch_ids)} chunks in {time.time()-t0:.1f}s "
                 f"(running total: {total})")
        batch_ids = []
        batch_texts = []
        batch_metas = []

    for chunk in raster_result.chunk_iterator:
        data = chunk.data
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            continue

        mn, mx = float(np.min(valid)), float(np.max(valid))
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
            source_id=args.source_id,
            dataset_name=args.dataset_name,
        )
        meta_dict = meta.to_dict()
        meta_dict["catalog_source"] = "D1.1.xlsx"
        if args.hazard:
            meta_dict["hazard_type"] = args.hazard
        if args.impact_sector:
            meta_dict["impact_sector"] = args.impact_sector

        text = generate_human_readable_text(meta_dict)
        batch_ids.append(str(uuid.uuid4()))
        batch_texts.append(text)
        batch_metas.append(meta_dict)
        all_chunk_metas.append(meta_dict)
        total += 1

        if len(batch_ids) >= BATCH:
            flush()

    flush()
    log.info(f"DONE — wrote {total} chunks for {args.source_id} (dataset={args.dataset_name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
