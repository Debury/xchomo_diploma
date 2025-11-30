import logging
import zipfile
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, Union, Optional
import numpy as np
import xarray as xr
import dask.array as da
import rasterio

logger = logging.getLogger(__name__)

@dataclass
class RasterChunk:
    """
    Represents a specific slice of Space and Time.
    """
    data: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class RasterLoadResult:
    chunk_iterator: Iterator[RasterChunk]
    source_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==============================================================================
# PUBLIC API
# ==============================================================================

def load_raster_auto(path: Union[str, Path], **kwargs) -> RasterLoadResult:
    """
    Main entry point. Automatically selects the correct loader based on extension.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    
    logger.info(f"Selecting loader for: {path.name} (Suffix: {suffix})")
    
    if suffix == ".zip":
        iterator = _load_zip(path, **kwargs)
    elif suffix in {'.tif', '.tiff'}:
        iterator = _load_geotiff(path, **kwargs)
    elif suffix in {'.nc', '.nc4', '.hdf', '.h5'}:
        iterator = _load_netcdf(path, **kwargs)
    else:
        # Fallback: Try NetCDF loader for unknown extensions (e.g. .dat)
        logger.warning(f"Unknown extension '{suffix}', attempting NetCDF load...")
        iterator = _load_netcdf(path, **kwargs)
        
    return RasterLoadResult(
        chunk_iterator=iterator, 
        source_path=path, 
        metadata={"format": suffix}
    )

def raster_to_embeddings(source: RasterLoadResult, **kwargs) -> list:
    """
    Converts raw pixel grids into dense statistical vectors.
    """
    results = []
    if not source.chunk_iterator: return []
    
    for chunk in source.chunk_iterator:
        data = chunk.data
        # We only care about valid pixels
        valid = data[np.isfinite(data)]
        
        if valid.size == 0:
            continue
            
        # 8-dim stats vector
        stats = np.array([
            float(np.mean(valid)), 
            float(np.std(valid)), 
            float(np.min(valid)), 
            float(np.max(valid)),
            float(np.percentile(valid, 10)), 
            float(np.percentile(valid, 50)), 
            float(np.percentile(valid, 90)), 
            0.0 # Trend placeholder
        ], dtype="float32")
        
        results.append({"vector": stats.tolist(), "metadata": chunk.metadata})
    return results

# ==============================================================================
# INTERNAL LOADERS
# ==============================================================================

def _load_zip(path, **kwargs):
    """Extracts ZIP and recurses on supported files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(tmpdir)
            
            # Recursively find files
            found = False
            for f in Path(tmpdir).rglob("*"):
                # Ignore MacOS metadata
                if f.name.startswith("__MACOSX") or f.name.startswith("."): 
                    continue
                    
                if f.suffix.lower() in {'.nc', '.nc4', '.h5'}:
                    found = True
                    yield from _load_netcdf(f, **kwargs)
                elif f.suffix.lower() in {'.tif', '.tiff'}:
                    found = True
                    yield from _load_geotiff(f, **kwargs)
            
            if not found:
                logger.warning(f"No supported files found inside zip: {path.name}")
                
        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {path}")

def _load_netcdf(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """
    Loader for NetCDF/HDF5 (Time-Aware).
    Enforces Time=1 chunking for granular RAG answers.
    """
    try:
        # Open dataset lazily
        ds = xr.open_dataset(path, chunks="auto", engine="h5netcdf")
        
        # Filter for numeric vars only
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        
        for var in numeric_vars:
            da_var = ds[var]
            
            # --- CHUNKING STRATEGY ---
            chunking = {}
            for dim in da_var.dims:
                d = str(dim).lower()
                if "time" in d or "date" in d:
                    # CRITICAL: One chunk per time step -> High Resolution
                    chunking[dim] = 1 
                elif "lat" in d or "y" in d or "lon" in d or "x" in d:
                    chunking[dim] = 100 # Spatial chunks
                else:
                    chunking[dim] = -1

            try:
                da_chunked = da_var.chunk(chunking)
            except Exception:
                # Fallback if dask chunking fails
                da_chunked = da.from_array(da_var, chunks=chunking)
            
            # Iterate
            for slices in da.core.slices_from_chunks(da_chunked.chunks):
                meta = {"variable": str(var), "source": str(path.name)}
                
                # Extract coords (Timestamp)
                for dim, sl in zip(da_var.dims, slices):
                    if dim in da_var.coords:
                        vals = da_var.coords[dim][sl].values
                        if len(vals) > 0:
                            # Start==End when chunksize is 1, giving exact timestamp
                            meta[f"{dim}_start"] = str(vals.min())
                            # Clean up numpy datetime string for prettier metadata
                            if "T" in meta[f"{dim}_start"]:
                                meta["time_start"] = meta[f"{dim}_start"]

                try:
                    data = da_chunked[slices].compute().values
                    if np.isnan(data).all(): continue
                    yield RasterChunk(data=data, metadata=meta)
                except Exception as e:
                    continue
                    
    except Exception as e:
        logger.error(f"NetCDF Load Error {path}: {e}")

def _load_geotiff(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """Loader for GeoTIFFs (Spatial Only)."""
    try:
        with rasterio.open(path) as src:
            # Iterate over 256x256 windows
            # Use smaller windows if image is small
            w = 256
            if src.width < 512 or src.height < 512:
                w = 64
                
            for ji, window in src.block_windows(1):
                # Read band 1 by default
                try:
                    data = src.read(1, window=window)
                except Exception:
                    continue
                
                # Convert to float for stats
                data = data.astype('float32')
                
                # Handle NoData
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                # Skip empty
                if np.isnan(data).all():
                    continue

                bounds = src.window_bounds(window)
                meta = {
                    "variable": "band_1",
                    "source": str(path.name),
                    "lat_min": bounds[1],
                    "lat_max": bounds[3],
                    "lon_min": bounds[0],
                    "lon_max": bounds[2],
                    "time_start": "static_image",
                }
                
                yield RasterChunk(data=data, metadata=meta)
                
    except Exception as e:
        logger.error(f"GeoTIFF Load Error {path}: {e}")