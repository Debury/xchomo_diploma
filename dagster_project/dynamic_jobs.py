import sys
from pathlib import Path
from datetime import datetime
import requests
import traceback

# PATH MAGIC: Ensure Docker finds /app/src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dagster import job, op, Out, OpExecutionContext
from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource

# Clean Imports from the new structure
from src.climate_embeddings.loaders import load_raster_auto, raster_to_embeddings, detect_format_from_url
from src.climate_embeddings.embeddings import TextEmbedder
from src.embeddings.database import VectorDatabase
from src.sources import get_source_store

@op(
    description="Full Pipeline",
    out=Out(list),
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context):
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    # Init
    store = get_source_store()
    sources = store.get_all_sources()
    
    if not sources:
        logger.info("No sources found.")
        return []

    text_embedder = TextEmbedder() # BGE-Large
    vector_db = VectorDatabase()   # Qdrant
    results = []

    for source in sources:
        try:
            # 1. Download
            output_dir = data_paths.get_raw_path()
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{source.source_id}_{timestamp}.dat"
            filepath = output_dir / filename
            
            # Simple download
            if not filepath.exists():
                logger.info(f"Downloading {source.url}...")
                with requests.get(source.url, stream=True) as r:
                    r.raise_for_status()
                    with open(filepath, 'wb') as f:
                        for chunk in r.iter_content(8192): f.write(chunk)

            # 2. Load & Embed Stats
            raster_result = load_raster_auto(filepath)
            stat_embeddings = raster_to_embeddings(raster_result)
            
            if not stat_embeddings:
                continue

            # 3. Semantic Embedding
            texts = []
            valid_chunks = []
            for item in stat_embeddings:
                meta = item['metadata']
                stats = item['vector']
                desc = f"Variable: {meta.get('variable')} | Stats: Mean={stats[0]:.2f}, Max={stats[3]:.2f}"
                texts.append(desc)
                valid_chunks.append(item)

            semantic_vectors = text_embedder.embed_documents(texts)

            # 4. Store
            ids, embs, metas, docs = [], [], [], []
            for i, (sem_vec, stat_item, txt) in enumerate(zip(semantic_vectors, valid_chunks, texts)):
                ids.append(f"{source.source_id}_{i}")
                embs.append(sem_vec.tolist())
                docs.append(txt)
                metas.append({**stat_item['metadata'], "source_id": source.source_id})

            vector_db.add_embeddings(ids, embs, metas, docs)
            results.append(source.source_id)
            
        except Exception as e:
            logger.error(f"Error processing {source.source_id}: {e}")
            logger.error(traceback.format_exc())

    return results

@job(resource_defs={
    "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
    "logger": LoggerResource(log_file="logs/dagster.log"),
    "data_paths": DataPathResource(raw_data_dir="data/raw", processed_data_dir="data/processed", embeddings_dir="db")
})
def dynamic_source_etl_job():
    process_all_sources()