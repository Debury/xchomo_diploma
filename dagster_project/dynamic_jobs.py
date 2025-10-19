"""
Dynamic Source-Driven ETL Jobs for Phase 5

Complete pipeline: Download → Process → Generate Embeddings → Store in ChromaDB
"""

from dagster import job, op, Out, OpExecutionContext
from typing import Dict, Any, List

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource


@op(
    description="Complete pipeline: download → process → embeddings → ChromaDB",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    """Process all active sources with embeddings generation."""
    from src.sources import get_source_store
    from dagster_project.ops.dynamic_source_ops import detect_format_from_url
    import requests
    from datetime import datetime
    from pathlib import Path
    
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline")
    logger.info("=" * 80)
    
    store = get_source_store()
    sources = store.get_all_sources(active_only=True)
    
    logger.info(f"Found {len(sources)} active source(s)")
    
    results = []
    
    for source in sources:
        source_id = source.source_id
        logger.info(f"\n{'='*70}")
        logger.info(f"SOURCE: {source_id}")
        logger.info(f"{'='*70}")
        
        try:
            store.update_processing_status(source_id, "processing")
            
            format = source.format or detect_format_from_url(source.url)
            logger.info(f"Format: {format} | URL: {source.url[:60]}...")
            
            # STEP 1: DOWNLOAD
            logger.info(f"\n[1/4] DOWNLOADING...")
            output_dir = data_paths.get_raw_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.replace('netcdf', 'nc')
            filename = f"{source_id}_{timestamp}.{ext}"
            filepath = output_dir / filename
            
            response = requests.get(source.url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            file_size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"✓ Downloaded {file_size_mb:.2f} MB")
            
            # STEP 2: PROCESS
            logger.info(f"\n[2/4] PROCESSING...")
            
            if format == "netcdf":
                import xarray as xr
                
                ds = xr.open_dataset(filepath)
                logger.info(f"Variables: {list(ds.data_vars)} | Dims: {dict(ds.dims)}")
                
                # Transform
                for var in ds.data_vars:
                    if 'temperature' in var.lower() or var in ['t2m', 'air', 'tas']:
                        if ds[var].attrs.get('units') == 'K':
                            logger.info(f"Converting {var}: K → °C")
                            ds[var] = ds[var] - 273.15
                            ds[var].attrs['units'] = 'degC'
                
                # Save processed
                processed_dir = data_paths.get_processed_path()
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_file = processed_dir / f"{source_id}_processed.nc"
                ds.to_netcdf(processed_file)
                logger.info(f"✓ Saved: {processed_file.name}")
                
                # STEP 3: GENERATE EMBEDDINGS
                logger.info(f"\n[3/4] GENERATING EMBEDDINGS...")
                
                from src.embeddings.generator import EmbeddingGenerator
                generator = EmbeddingGenerator()
                
                embeddings_data = []
                
                # Process up to 5 variables
                for var in list(ds.data_vars)[:5]:
                    var_data = ds[var]
                    
                    # Calculate spatial mean
                    if 'time' in var_data.dims:
                        spatial_mean = var_data.mean(dim='time')
                    else:
                        spatial_mean = var_data
                    
                    # Sample data points
                    if spatial_mean.size > 100:
                        step = max(1, int(spatial_mean.size ** 0.5) // 10)
                        sampled = spatial_mean.values.flatten()[::step][:100]
                    else:
                        sampled = spatial_mean.values.flatten()
                    
                    # Create descriptive text
                    text = f"Climate variable '{var}': "
                    text += f"mean={float(sampled.mean()):.2f}, "
                    text += f"std={float(sampled.std()):.2f}, "
                    text += f"range=[{float(sampled.min()):.2f}, {float(sampled.max()):.2f}]. "
                    text += f"Units: {ds[var].attrs.get('units', 'unknown')}. "
                    
                    if 'long_name' in ds[var].attrs:
                        text += f"{ds[var].attrs['long_name']}."
                    
                    # Generate embedding (expects list of strings, returns numpy array)
                    embedding_array = generator.generate_embeddings([text])
                    embedding = embedding_array[0].tolist()  # Convert first element to list
                    
                    embeddings_data.append({
                        'id': f"{source_id}_{var}_{timestamp}",
                        'embedding': embedding,
                        'metadata': {
                            'source_id': source_id,
                            'variable': var,
                            'text': text,
                            'units': ds[var].attrs.get('units', 'unknown'),
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                    logger.info(f"  ✓ {var}: embedding shape {len(embedding)}")
                
                # STEP 4: STORE IN CHROMADB
                logger.info(f"\n[4/4] STORING IN CHROMADB...")
                
                from src.embeddings.database import VectorDatabase
                db = VectorDatabase()
                
                # Extract separate lists for ids, embeddings, metadatas, documents
                ids = [item['id'] for item in embeddings_data]
                embeddings = [item['embedding'] for item in embeddings_data]
                metadatas = [item['metadata'] for item in embeddings_data]
                documents = [item['metadata']['text'] for item in embeddings_data]
                
                db.add_embeddings(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
                logger.info(f"✓ Stored {len(embeddings_data)} embeddings in ChromaDB")
                
                result = {
                    "source_id": source_id,
                    "status": "success",
                    "raw_file": str(filepath),
                    "processed_file": str(processed_file),
                    "format": format,
                    "variables": list(ds.data_vars),
                    "file_size_mb": file_size_mb,
                    "embeddings_count": len(embeddings_data)
                }
                
                ds.close()
                
            elif format == "csv":
                import pandas as pd
                
                df = pd.read_csv(filepath)
                logger.info(f"Rows: {len(df)} | Columns: {len(df.columns)}")
                
                # Save processed
                processed_dir = data_paths.get_processed_path()
                processed_dir.mkdir(parents=True, exist_ok=True)
                processed_file = processed_dir / f"{source_id}_processed.parquet"
                df.to_parquet(processed_file)
                logger.info(f"✓ Saved: {processed_file.name}")
                
                # Generate embeddings for numeric columns
                logger.info(f"\n[3/4] GENERATING EMBEDDINGS...")
                
                from src.embeddings.generator import EmbeddingGenerator
                generator = EmbeddingGenerator()
                
                embeddings_data = []
                
                for col in df.select_dtypes(include=['number']).columns[:10]:
                    text = f"Column '{col}': mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
                    text += f"min={df[col].min():.2f}, max={df[col].max():.2f}"
                    
                    # Generate embedding (expects list of strings, returns numpy array)
                    embedding_array = generator.generate_embeddings([text])
                    embedding = embedding_array[0].tolist()  # Convert first element to list
                    
                    embeddings_data.append({
                        'id': f"{source_id}_{col}_{timestamp}",
                        'embedding': embedding,
                        'metadata': {
                            'source_id': source_id,
                            'variable': col,
                            'text': text,
                            'timestamp': datetime.now().isoformat()
                        }
                    })
                
                logger.info(f"\n[4/4] STORING IN CHROMADB...")
                from src.embeddings.database import VectorDatabase
                db = VectorDatabase()
                
                # Extract separate lists for ids, embeddings, metadatas, documents
                ids = [item['id'] for item in embeddings_data]
                embeddings = [item['embedding'] for item in embeddings_data]
                metadatas = [item['metadata'] for item in embeddings_data]
                documents = [item['metadata']['text'] for item in embeddings_data]
                
                db.add_embeddings(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
                logger.info(f"✓ Stored {len(embeddings_data)} embeddings")
                
                result = {
                    "source_id": source_id,
                    "status": "success",
                    "format": "parquet",
                    "embeddings_count": len(embeddings_data)
                }
                
            else:
                logger.info(f"⚠ Unsupported format: {format}")
                result = {
                    "source_id": source_id,
                    "status": "unsupported_format",
                    "format": format
                }
            
            store.update_processing_status(source_id, "completed")
            logger.info(f"\n✓ COMPLETE: {source_id}")
            
        except Exception as e:
            logger.error(f"\n✗ FAILED: {source_id} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            store.update_processing_status(source_id, "failed", error_message=str(e))
            result = {
                "source_id": source_id,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PIPELINE COMPLETE: {len(results)} sources processed")
    logger.info("=" * 80)
    
    return results


@job(
    description="Dynamic ETL: all active sources → download → process → embeddings → ChromaDB",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_dynamic_etl.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={"pipeline": "dynamic_etl", "phase": "5"}
)
def dynamic_source_etl_job():
    """
    🚀 Complete Dynamic Source ETL Pipeline
    
    For each active source:
    1. 📥 Download from URL
    2. ⚙️ Process & transform data
    3. 🧠 Generate embeddings
    4. 💾 Store in ChromaDB
    
    Usage:
        POST /sources/{source_id}/trigger
    """
    process_all_sources()


__all__ = ["dynamic_source_etl_job"]
