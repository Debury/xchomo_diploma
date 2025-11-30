import logging
import zipfile
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Union
import numpy as np
import xarray as xr
import dask.array as da

logger = logging.getLogger(__name__)

@dataclass
class RasterChunk:
    """
    Represents a specific slice of Space and Time.
    If 'data' is 128x128 pixels, this represents the state of that region
    at a specific timestamp (not an average over time).
    """
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
    """
    Converts raw pixel grids into dense statistical vectors.
    Because we now slice Time=1, these stats represent the exact values 
    for a specific time step, not a yearly average.
    """
    results = []
    if not source.chunk_iterator: return []
    
    for chunk in source.chunk_iterator:
        data = chunk.data
        # We only care about valid pixels
        valid = data[np.isfinite(data)]
        
        if valid.size == 0:
            # Empty ocean/mask tile, skip or return zeros
            continue
            
        # These 8 numbers describe the distribution of the variable 
        # within this specific tile at this specific time.
        stats = np.array([
            np.mean(valid), 
            np.std(valid), 
            np.min(valid), 
            np.max(valid),
            np.percentile(valid, 10), 
            np.percentile(valid, 50), 
            np.percentile(valid, 90), 
            0.0 # Trend placeholder (0 for single timestep)
        ], dtype="float32")
        
        results.append({"vector": stats.tolist(), "metadata": chunk.metadata})
    return results

# --- INTERNAL HELPERS ---
def _load_zip(path, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(tmpdir)
        for f in Path(tmpdir).rglob("*"):
            if f.suffix in {'.nc', '.tif', '.grib', '.csv', '.nc4'}:
                yield from _load_single_file(f, **kwargs)

def _load_single_file(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """
    Loads a file and slices it into granular Spatiotemporal Chunks.
    CRITICAL CHANGE: We now slice Time=1 to preserve daily/hourly detail.
    """
    try:
        # Open dataset lazily
        ds = xr.open_dataset(path, chunks="auto", engine="h5netcdf")
        
        # Filter for numeric vars only
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        
        for var in numeric_vars:
            da_var = ds[var]
            
            # --- THE FIX FOR DETAILED ANSWERS ---
            # We enforce that any dimension named "time" has a chunk size of 1.
            # This ensures we generate a separate vector for every single time step.
            chunking_plan = {}
            for dim in da_var.dims:
                dim_name = str(dim).lower()
                if "time" in dim_name or "date" in dim_name:
                    # ONE chunk per time step -> High Resolution
                    chunking_plan[dim] = 1 
                elif "lat" in dim_name or "y" in dim_name:
                    # Spatial chunking (keep reasonably small for location queries)
                    chunking_plan[dim] = 100 
                elif "lon" in dim_name or "x" in dim_name:
                    chunking_plan[dim] = 100
                else:
                    # Unknown dim (e.g. depth), keep whole
                    chunking_plan[dim] = -1 

            # Apply the chunking plan
            da_chunked = da_var.chunk(chunking_plan)
            
            # Use Dask's logic to iterate over every block in the grid
            for slices in da.core.slices_from_chunks(da_chunked.chunks):
                # 1. Extract Metadata (Timestamp, Lat/Lon Box)
                meta = {"variable": str(var), "source": str(path)}
                
                for dim, sl in zip(da_var.dims, slices):
                    if dim in da_var.coords:
                        vals = da_var.coords[dim][sl].values
                        if len(vals) > 0:
                            # If time chunk is 1, start==end, which effectively gives us the exact timestamp
                            meta[f"{dim}_start"] = str(vals.min())
                            meta[f"{dim}_end"] = str(vals.max())

                # 2. Compute the actual data for this slice
                try:
                    # Bring just this small 100x100x1 block into memory
                    data = da_chunked[slices].compute().values
                    
                    # Handle NaNs (e.g. land mask) without crashing
                    if np.isnan(data).all():
                        continue 
                        
                    yield RasterChunk(data=data, metadata=meta)
                    
                except Exception as e:
                    logger.warning(f"Error reading chunk {slices}: {e}")
                    continue
                    
    except Exception as e:
        logger.error(f"Failed to open {path}: {e}")