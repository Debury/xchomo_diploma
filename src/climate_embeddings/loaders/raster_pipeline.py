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
        iterator = _load_xarray_generic(path, engine="h5netcdf", **kwargs)
    elif suffix in {'.grib', '.grib2', '.grb'}:
        # Requires 'cfgrib' installed
        iterator = _load_xarray_generic(path, engine="cfgrib", **kwargs)
    else:
        # Fallback: Try NetCDF loader for unknown extensions
        logger.warning(f"Unknown extension '{suffix}', attempting generic Xarray load...")
        # Defaulting to None lets xarray guess, or fallback to h5netcdf
        iterator = _load_xarray_generic(path, engine=None, **kwargs)
        
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
            
        # 8-dim stats vector describing the distribution
        # Indices: 0=Mean, 1=Std, 2=Min, 3=Max, 4=P10, 5=Median, 6=P90, 7=Range
        mn = float(np.min(valid))
        mx = float(np.max(valid))
        
        stats = np.array([
            float(np.mean(valid)), 
            float(np.std(valid)), 
            mn, 
            mx,
            float(np.percentile(valid, 10)), 
            float(np.percentile(valid, 50)), 
            float(np.percentile(valid, 90)), 
            mx - mn 
        ], dtype="float32")
        
        results.append({"vector": stats.tolist(), "metadata": chunk.metadata})
    return results

# ==============================================================================
# INTERNAL LOADERS
# ==============================================================================

def _load_zip(path, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(tmpdir)
            
            found = False
            for f in Path(tmpdir).rglob("*"):
                if f.name.startswith("__MACOSX") or f.name.startswith("."): 
                    continue
                    
                if f.suffix.lower() in {'.nc', '.nc4', '.h5'}:
                    found = True
                    yield from _load_xarray_generic(f, engine="h5netcdf", **kwargs)
                elif f.suffix.lower() in {'.grib', '.grib2'}:
                    found = True
                    yield from _load_xarray_generic(f, engine="cfgrib", **kwargs)
                elif f.suffix.lower() in {'.tif', '.tiff'}:
                    found = True
                    yield from _load_geotiff(f, **kwargs)
            
            if not found:
                logger.warning(f"No supported files found inside zip: {path.name}")
                
        except zipfile.BadZipFile:
            logger.error(f"Invalid ZIP file: {path}")

def _load_xarray_generic(path: Path, engine: Optional[str] = None, **kwargs) -> Iterator[RasterChunk]:
    """
    Generic Loader for NetCDF/GRIB using Xarray.
    Enforces Time=1 chunking for granular RAG answers.
    """
    try:
        # Open dataset lazily
        # GRIB files might have multiple datasets (hypercubes), filter_by_keys might be needed in complex cases
        # For general purpose, we try to open the standard dataset.
        ds = xr.open_dataset(path, chunks="auto", engine=engine)
        
        # Filter for numeric vars only
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        
        for var in numeric_vars:
            da_var = ds[var]
            
            # --- CHUNKING STRATEGY ---
            chunking = {}
            for dim in da_var.dims:
                d = str(dim).lower()
                if "time" in d or "date" in d or "step" in d:
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
        logger.error(f"Xarray Load Error ({engine}) for {path}: {e}")

def _load_geotiff(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """Loader for GeoTIFFs (Spatial Only)."""
    try:
        with rasterio.open(path) as src:
            w = 256
            if src.width < 512 or src.height < 512:
                w = 64
                
            for ji, window in src.block_windows(1):
                try:
                    data = src.read(1, window=window).astype('float32')
                except Exception:
                    continue
                
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
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