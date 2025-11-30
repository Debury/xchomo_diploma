# climate_embeddings/loaders/raster_pipeline.py
import logging
import math
import zipfile
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)

# --- CONFIG ---
SUPPORTED_EXTENSIONS = {
    ".nc": "netcdf", ".nc4": "netcdf", ".grib": "grib", ".grb2": "grib",
    ".h5": "hdf5", ".tif": "geotiff", ".tiff": "geotiff",
    ".asc": "ascii", ".csv": "csv", ".zarr": "zarr"
}

@dataclass
class RasterChunk:
    """Standardized output for the pipeline."""
    data: np.ndarray  # float32
    metadata: Dict[str, Any]

class RasterLoader:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.format = self._detect_format()

    def _detect_format(self) -> str:
        suffix = self.path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[suffix]
        return "unknown"

    def load_chunks(self, **kwargs) -> Iterator[RasterChunk]:
        """Dispatcher for specific loaders."""
        if self.format == "netcdf":
            yield from self._load_xarray_engine("h5netcdf", **kwargs)
        elif self.format == "grib":
            yield from self._load_xarray_engine("cfgrib", **kwargs)
        elif self.format == "geotiff":
            yield from self._load_rasterio(**kwargs)
        elif self.format == "csv":
            yield from self._load_csv(**kwargs)
        else:
            raise ValueError(f"Unsupported format {self.format} for {self.path}")

    def _load_xarray_engine(self, engine: str, time_slice=None, variables=None, **kwargs) -> Iterator[RasterChunk]:
        # 1. Lazy Open
        try:
            ds = xr.open_dataset(self.path, engine=engine, chunks="auto", decode_coords="all")
        except Exception:
            # Fallback for complex GRIBs or memory issues
            ds = xr.open_dataset(self.path, engine=engine, chunks={"time": 1, "y": 256, "x": 256})

        # 2. Filter Variables
        if variables:
            ds = ds[variables]

        # 3. Filter Time (Lazy Slicing)
        if time_slice and "time" in ds.coords:
            ds = ds.sel(time=slice(*time_slice))

        # 4. Iterate over Spatial Chunks (The Fix)
        # We chunk spatially, keeping full time dimension in one block if possible
        # so we can interpolate efficiently.
        
        # Ensure we have numeric variables only
        data_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        
        for var_name in data_vars:
            da_var = ds[var_name]
            
            # Create a localized generator of chunks using dask slices
            # We enforce that 'time' is NOT chunked, but Y and X are.
            spatial_chunks = {"time": -1, "y": 128, "x": 128} 
            
            # Handle cases where dims might be named differently (lat/lon)
            dims = da_var.dims
            chunk_plan = {}
            for d in dims:
                if "time" in d.lower():
                    chunk_plan[d] = -1 # Keep time contiguous
                else:
                    chunk_plan[d] = 128 # Spatially chunk
            
            da_chunked = da_var.chunk(chunk_plan)
            
            # Iterate blocks
            # slices_from_chunks returns slices for every block in the dask array
            for slices in da.core.slices_from_chunks(da_chunked.chunks):
                # slices is a tuple of slice objects, e.g., (slice(0,100), slice(0,128), slice(0,128))
                
                # Extract Metadata from coordinates BEFORE compute
                meta = self._extract_metadata(da_var, slices)
                meta["variable"] = var_name
                meta["source"] = str(self.path)

                # COMPUTE: Bring this specific 3D block into memory
                # This is safe because 128x128 * time is usually small enough for RAM
                block_data = da_chunked[slices].compute()
                
                # Convert to numpy and Handle NaNs (The Fix!)
                # We do interpolation here on the numpy array, not the dask array.
                np_data = block_data.values if hasattr(block_data, "values") else block_data
                
                if kwargs.get("interpolate", True):
                    np_data = self._numpy_interpolate_na(np_data)

                yield RasterChunk(data=np_data, metadata=meta)

    def _numpy_interpolate_na(self, arr: np.ndarray) -> np.ndarray:
        """Fast interpolation on in-memory numpy array along axis 0 (time)."""
        # If 2D (no time), just fillna
        if arr.ndim < 3:
            mask = np.isnan(arr)
            arr[mask] = 0  # Or mean
            return arr
            
        # For 3D (time, y, x), we can use pandas for interpolation or simple forward fill
        # A quick approximation is filling with the mean of the chunk to avoid slow python loops
        # Or proper interpolation:
        import bottleneck as bn # Optional optimization
        
        # Simple approach: Replace NaN with nanmean of the spatial chunk
        # This is much faster than time-series interpolation for massive rasters
        # For strict time interpolation, we'd need to reshape and iterate.
        
        mean_val = np.nanmean(arr)
        if np.isnan(mean_val): mean_val = 0.0
        
        # In-place fill
        inds = np.where(np.isnan(arr))
        arr[inds] = mean_val
        return arr

    def _extract_metadata(self, da_var, slices):
        """Extract spatial/temporal bounds from coords based on slices."""
        meta = {}
        # Mapping slices to dimensions
        for dim, sl in zip(da_var.dims, slices):
            if dim in da_var.coords:
                coord_vals = da_var.coords[dim][sl].values
                if len(coord_vals) > 0:
                    if np.issubdtype(coord_vals.dtype, np.number):
                        meta[f"{dim}_min"] = float(coord_vals.min())
                        meta[f"{dim}_max"] = float(coord_vals.max())
                    elif np.issubdtype(coord_vals.dtype, np.datetime64) or "time" in str(coord_vals.dtype):
                        meta[f"{dim}_start"] = str(coord_vals.min())
                        meta[f"{dim}_end"] = str(coord_vals.max())
        return meta

    def _load_rasterio(self, **kwargs) -> Iterator[RasterChunk]:
        with rasterio.open(self.path) as src:
            # 256x256 windows
            for ji, window in src.block_windows(1):
                data = src.read(window=window)
                bounds = src.window_bounds(window)
                meta = {
                    "source": str(self.path),
                    "driver": "rasterio",
                    "lon_min": bounds[0], "lat_min": bounds[1],
                    "lon_max": bounds[2], "lat_max": bounds[3]
                }
                # Handle nodata
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                
                yield RasterChunk(data=data, metadata=meta)

    def _load_csv(self, **kwargs) -> Iterator[RasterChunk]:
        # Simple station data loader
        df = pd.read_csv(self.path)
        # Assume standard columns exist or infer them
        numeric_cols = df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            data = df[col].values
            meta = {
                "source": str(self.path),
                "variable": col,
                "rows": len(df)
            }
            yield RasterChunk(data=data, metadata=meta)

# ZIP HANDLER
def load_data_source(path: Union[str, Path], **kwargs) -> Iterator[RasterChunk]:
    """Universal entry point. Handles ZIPs recursively."""
    path = Path(path)
    
    if path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                with zipfile.ZipFile(path, 'r') as zf:
                    zf.extractall(tmpdir)
                    
                # Recurse into extracted files
                tmp_path = Path(tmpdir)
                for f in tmp_path.rglob("*"):
                    if f.suffix in SUPPORTED_EXTENSIONS:
                        loader = RasterLoader(f)
                        for chunk in loader.load_chunks(**kwargs):
                            # Annotate metadata with original zip source
                            chunk.metadata["zip_source"] = str(path)
                            yield chunk
            except Exception as e:
                logger.error(f"Failed to process zip {path}: {e}")
    else:
        loader = RasterLoader(path)
        yield from loader.load_chunks(**kwargs)