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
import pandas as pd

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
    elif suffix in {'.csv', '.tsv'}:
        iterator = _load_csv(path, **kwargs)
    elif suffix in {'.tif', '.tiff'}:
        iterator = _load_geotiff(path, **kwargs)
    elif suffix in {'.nc', '.nc4', '.hdf', '.h5'}:
        iterator = _load_xarray_generic(path, engine="h5netcdf", **kwargs)
    elif suffix in {'.grib', '.grib2', '.grb'}:
        # Requires 'cfgrib' installed
        iterator = _load_xarray_generic(path, engine="cfgrib", **kwargs)
    else:
        # Fallback: Try CSV first (common), then NetCDF
        logger.warning(f"Unknown extension '{suffix}', trying CSV loader first...")
        try:
            iterator = _load_csv(path, **kwargs)
        except Exception as csv_err:
            logger.warning(f"CSV load failed: {csv_err}, attempting generic Xarray load...")
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
                elif f.suffix.lower() in {'.csv', '.tsv'}:
                    found = True
                    yield from _load_csv(f, **kwargs)
            
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

def _load_csv(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """
    Loader for CSV/TSV files using pandas.
    Handles time series data, station data, and gridded CSV formats.
    """
    try:
        # Try to detect delimiter
        suffix = path.suffix.lower()
        delimiter = '\t' if suffix == '.tsv' else ','
        
        # Read CSV in chunks for memory efficiency
        chunk_size = kwargs.get('chunk_size', 10000)
        
        logger.info(f"Loading CSV file: {path.name}")
        
        # First, read a small sample to understand structure
        sample_df = pd.read_csv(path, nrows=100, delimiter=delimiter, low_memory=False)
        
        # Identify numeric columns (excluding metadata columns)
        numeric_cols = []
        metadata_cols = []
        time_cols = []
        spatial_cols = []
        
        for col in sample_df.columns:
            col_lower = str(col).lower()
            col_str = str(col)
            
            # Skip attribute columns (usually end with _ATTRIBUTES or _ATTR)
            if '_attributes' in col_lower or col_lower.endswith('_attr') or col_str.endswith('_ATTRIBUTES'):
                metadata_cols.append(col)
                continue
            
            # Identify metadata columns
            if col_lower in ['station', 'name', 'elevation']:
                metadata_cols.append(col)
                continue
            
            # Identify time columns
            if 'date' in col_lower or 'time' in col_lower or col_lower == 'date':
                time_cols.append(col)
                continue
            
            # Identify spatial columns
            if 'lat' in col_lower or 'lon' in col_lower or col_lower in ['latitude', 'longitude']:
                spatial_cols.append(col)
                continue
            
            # Check if numeric (but not already identified as metadata/spatial/time)
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(sample_df[col], errors='coerce')
                if not numeric_series.isna().all():  # Has at least some numeric values
                    # Check if it's mostly numeric (at least 50% valid numbers)
                    valid_ratio = numeric_series.notna().sum() / len(numeric_series)
                    if valid_ratio > 0.5:
                        numeric_cols.append(col)
            except:
                pass
        
        logger.info(f"Found {len(numeric_cols)} numeric columns, {len(time_cols)} time columns, {len(spatial_cols)} spatial columns")
        
        if not numeric_cols:
            logger.warning(f"No numeric columns found in CSV: {path.name}")
            return
        
        # Read full CSV in chunks
        chunk_iter = pd.read_csv(
            path, 
            delimiter=delimiter,
            chunksize=chunk_size,
            low_memory=False,
            iterator=True
        )
        
        for chunk_idx, df_chunk in enumerate(chunk_iter):
            # Process each numeric column as a separate variable
            for var_col in numeric_cols:
                # Extract numeric values, handling missing data
                values = pd.to_numeric(df_chunk[var_col], errors='coerce').values
                values = values[~np.isnan(values)]  # Remove NaN
                
                if len(values) == 0:
                    continue
                
                # Convert to 2D array for consistency with raster pipeline
                # Reshape to (1, n) to match expected format
                data = values.reshape(1, -1).astype('float32')
                
                # Extract metadata
                meta = {
                    "variable": str(var_col),
                    "source": str(path.name),
                    "format": "csv",
                }
                
                # Extract time information if available
                if time_cols:
                    time_col = time_cols[0]
                    if time_col in df_chunk.columns:
                        time_values = df_chunk[time_col].dropna()
                        if len(time_values) > 0:
                            meta["time_start"] = str(time_values.iloc[0])
                            if len(time_values) > 1:
                                meta["time_end"] = str(time_values.iloc[-1])
                
                # Extract spatial information if available
                if spatial_cols:
                    lat_col = next((c for c in spatial_cols if 'lat' in str(c).lower()), None)
                    lon_col = next((c for c in spatial_cols if 'lon' in str(c).lower()), None)
                    
                    if lat_col and lat_col in df_chunk.columns:
                        lat_values = pd.to_numeric(df_chunk[lat_col], errors='coerce').dropna()
                        if len(lat_values) > 0:
                            meta["lat_min"] = float(lat_values.min())
                            meta["lat_max"] = float(lat_values.max())
                    
                    if lon_col and lon_col in df_chunk.columns:
                        lon_values = pd.to_numeric(df_chunk[lon_col], errors='coerce').dropna()
                        if len(lon_values) > 0:
                            meta["lon_min"] = float(lon_values.min())
                            meta["lon_max"] = float(lon_values.max())
                
                # Add station info if available
                if 'STATION' in df_chunk.columns:
                    stations = df_chunk['STATION'].dropna().unique()
                    if len(stations) > 0:
                        meta["station_id"] = str(stations[0])
                        if len(stations) > 1:
                            meta["station_count"] = len(stations)
                
                # Add chunk index
                meta["chunk_index"] = chunk_idx
                meta["row_count"] = len(df_chunk)
                
                yield RasterChunk(data=data, metadata=meta)
        
        logger.info(f"Successfully processed CSV: {path.name}")
        
    except Exception as e:
        logger.error(f"CSV Load Error for {path}: {e}")
        raise

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