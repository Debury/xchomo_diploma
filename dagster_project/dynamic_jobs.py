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
    description="Complete pipeline: download → process → embeddings → Qdrant (memory-safe)",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline")
    logger.info("=" * 80)
    
    # Init Store
    store = get_source_store()
    sources = store.get_all_sources(active_only=True)
    logger.info(f"Found {len(sources)} active source(s)")
    
    if not sources:
        return []

    # Initialize models
    logger.info("Initializing Embedder (BGE-Large)...")
    text_embedder = TextEmbedder()
    vector_db = VectorDatabase()

    results = []

    for source in sources:
        source_id = source.source_id
        logger.info(f"\nSOURCE: {source_id}")
        
        try:
            # Set status to processing
            store.update_processing_status(source_id, "processing")

            # 1. Download
            format_hint = source.format or detect_format_from_url(source.url)
            output_dir = data_paths.get_raw_path()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = {'netcdf': 'nc', 'geotiff': 'tif'}.get(format_hint, 'dat')
            filepath = output_dir / f"{source_id}_{timestamp}.{ext}"

            # Only download if not exists or force
            if not filepath.exists():
                logger.info(f"Downloading {source.url}...")
                response = requests.get(source.url, stream=True, timeout=120)
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                logger.info("File already exists, skipping download.")

            # 2. Load & Embed Stats
            logger.info("Loading & Chunking...")
            raster_result = load_raster_auto(
                filepath,
                chunks="auto",
                variables=source.variables,
                bbox=tuple(source.spatial_bbox) if source.spatial_bbox else None
            )
            
            stat_embeddings = raster_to_embeddings(
                raster_result,
                normalization="zscore"
            )
            
            if not stat_embeddings:
                logger.warning(f"No data generated for {source_id}")
                store.update_processing_status(source_id, "failed", error_message="No valid data found in file")
                results.append({"source_id": source_id, "status": "empty"})
                continue

            # 3. Generate Semantic Vectors & Store
            logger.info(f"Embedding {len(stat_embeddings)} chunks...")
            
            text_descriptions = []
            for item in stat_embeddings:
                meta = item["metadata"]
                v = item["vector"]
                desc = f"Variable: {meta.get('variable')} | Time: {meta.get('time_start')} | Stats: Mean={v[0]:.2f}, Max={v[3]:.2f}"
                text_descriptions.append(desc)

            semantic_vectors = text_embedder.embed_documents(text_descriptions)
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, (sem_vec, stat_item, desc) in enumerate(zip(semantic_vectors, stat_embeddings, text_descriptions)):
                uid = f"{source_id}_{timestamp}_{i}"
                ids.append(uid)
                embeddings.append(sem_vec.tolist())
                documents.append(desc)
                
                # Convert stats to python floats
                v = stat_item["vector"]
                full_meta = {
                    **stat_item["metadata"],
                    "source_id": source_id,
                    "timestamp": timestamp,
                    # Flatten essential stats for filtering
                    "stat_mean": float(v[0]),
                    "stat_max": float(v[3]),
                }
                metadatas.append(full_meta)

            vector_db.add_embeddings(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            # --- CRITICAL: Update status to COMPLETED ---
            logger.info(f"Updating status for {source_id} to COMPLETED")
            store.update_processing_status(source_id, "completed")
            
            results.append({"source_id": source_id, "status": "success", "count": len(ids)})
            
        except Exception as e:
            logger.error(f"Failed {source_id}: {e}")
            logger.error(traceback.format_exc())
            # Update status to FAILED
            store.update_processing_status(source_id, "failed", error_message=str(e))
            results.append({"source_id": source_id, "status": "failed", "error": str(e)})

    return results