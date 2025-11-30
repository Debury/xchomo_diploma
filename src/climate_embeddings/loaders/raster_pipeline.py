import logging
import zipfile
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Optional, Union
import numpy as np
import xarray as xr
import dask.array as da
import rasterio

logger = logging.getLogger(__name__)

@dataclass
class RasterChunk:
    data: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class RasterLoadResult:
    chunk_iterator: Iterator[RasterChunk]
    source_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

# --- PUBLIC API ---
def load_raster_auto(path: Union[str, Path], **kwargs) -> RasterLoadResult:
    """Main entry point used by Dagster."""
    path = Path(path)
    
    if path.suffix == ".zip":
        iterator = _load_zip(path, **kwargs)
    else:
        iterator = _load_single_file(path, **kwargs)
        
    return RasterLoadResult(chunk_iterator=iterator, source_path=path, metadata={"format": path.suffix})

def raster_to_embeddings(source: RasterLoadResult, **kwargs) -> list:
    """Converts pixel chunks into stats vectors."""
    results = []
    if not source.chunk_iterator: return []
    
    for chunk in source.chunk_iterator:
        data = chunk.data
        valid = data[np.isfinite(data)]
        if valid.size == 0:
            stats = np.zeros(8, dtype="float32")
        else:
            # [mean, std, min, max, p10, p50, p90, trend_placeholder]
            stats = np.array([
                np.mean(valid), np.std(valid), np.min(valid), np.max(valid),
                np.percentile(valid, 10), np.percentile(valid, 50), np.percentile(valid, 90), 0.0
            ], dtype="float32")
        
        results.append({"vector": stats.tolist(), "metadata": chunk.metadata})
    return results

# --- INTERNAL HELPERS ---
def _load_zip(path, **kwargs):
    # Simplified zip streaming logic
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(tmpdir)
        for f in Path(tmpdir).rglob("*"):
            if f.suffix in {'.nc', '.tif', '.grib', '.csv'}:
                yield from _load_single_file(f, **kwargs)

def _load_single_file(path: Path, **kwargs) -> Iterator[RasterChunk]:
    # xarray loader logic
    try:
        ds = xr.open_dataset(path, chunks="auto")
        # Filter for numeric vars only
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        
        for var in numeric_vars:
            da_var = ds[var]
            # Chunk spatially (Time: All, Y: 128, X: 128)
            chunking = {d: 128 for d in da_var.dims if "time" not in str(d).lower()}
            da_chunked = da_var.chunk(chunking)
            
            for slices in da.core.slices_from_chunks(da_chunked.chunks):
                # Metadata extraction
                meta = {"variable": str(var), "source": str(path)}
                for dim, sl in zip(da_var.dims, slices):
                    if dim in da_var.coords:
                        vals = da_var.coords[dim][sl].values
                        if len(vals) > 0:
                            meta[f"{dim}_start"] = str(vals.min())
                            meta[f"{dim}_end"] = str(vals.max())

                # Compute
                try:
                    data = da_chunked[slices].compute().values
                    # Fast NaN fill
                    if np.isnan(data).any():
                        data[np.isnan(data)] = np.nanmean(data)
                    yield RasterChunk(data=data, metadata=meta)
                except Exception as e:
                    logger.warning(f"Chunk error: {e}")
    except Exception as e:
        logger.error(f"Failed to open {path}: {e}")