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
            
            # Extract ALL variable attributes from NetCDF/GRIB (no hardcoding)
            var_attrs = {}
            if hasattr(da_var, 'attrs'):
                var_attrs = dict(da_var.attrs)
            elif var in ds.data_vars and hasattr(ds[var], 'attrs'):
                var_attrs = dict(ds[var].attrs)
            
            # Also get dataset-level attributes
            ds_attrs = {}
            if hasattr(ds, 'attrs'):
                ds_attrs = dict(ds.attrs)
            
            # Iterate
            for slices in da.core.slices_from_chunks(da_chunked.chunks):
                meta = {
                    "variable": str(var),
                    "source": str(path.name),
                    "format": "netcdf" if suffix in {'.nc', '.nc4', '.hdf', '.h5'} else "grib"
                }
                
                # Store ALL variable attributes (no hardcoding - works for ANY dataset)
                for attr_key, attr_val in var_attrs.items():
                    # Skip internal/technical attributes
                    if attr_key.startswith('_') or attr_key in ['_FillValue', 'missing_value', 'fill_value']:
                        continue
                    # Store everything else with original key name
                    try:
                        # Convert to string if not primitive type
                        if isinstance(attr_val, (int, float, str, bool)):
                            meta[attr_key] = attr_val
                        else:
                            meta[attr_key] = str(attr_val)
                    except:
                        meta[attr_key] = str(attr_val)
                
                # Store ALL dataset-level attributes
                for attr_key, attr_val in ds_attrs.items():
                    if attr_key.startswith('_'):
                        continue
                    try:
                        if isinstance(attr_val, (int, float, str, bool)):
                            meta[f"dataset_{attr_key}"] = attr_val
                        else:
                            meta[f"dataset_{attr_key}"] = str(attr_val)
                    except:
                        meta[f"dataset_{attr_key}"] = str(attr_val)
                
                # Extract ALL coordinate dimensions (no hardcoding - detect by value, not name)
                for dim, sl in zip(da_var.dims, slices):
                    if dim in da_var.coords:
                        vals = da_var.coords[dim][sl].values
                        if len(vals) > 0:
                            # Store dimension info
                            meta[f"{dim}_start"] = str(vals.min())
                            if len(vals) > 1:
                                meta[f"{dim}_end"] = str(vals.max())
                            
                            # Try to detect time by attempting to parse as datetime
                            try:
                                from datetime import datetime
                                pd.to_datetime(str(vals.min()))
                                # If successful, it's likely a time dimension
                                meta["time_start"] = str(vals.min())
                                if len(vals) > 1:
                                    meta["time_end"] = str(vals.max())
                            except:
                                # Not a time dimension - check if numeric and in spatial ranges
                                try:
                                    numeric_vals = vals[~np.isnan(vals)] if hasattr(vals, '__iter__') else [vals]
                                    if len(numeric_vals) > 0:
                                        min_val = float(np.min(numeric_vals))
                                        max_val = float(np.max(numeric_vals))
                                        # Latitude range
                                        if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                                            meta["lat_min"] = min_val
                                            meta["lat_max"] = max_val
                                        # Longitude range
                                        elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                                            meta["lon_min"] = min_val
                                            meta["lon_max"] = max_val
                                except:
                                    pass

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
        
        # COMPLETELY DYNAMIC: No hardcoded column names
        # Classify columns purely by data type and content, not by name
        numeric_cols = []
        non_numeric_cols = []
        
        for col in sample_df.columns:
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(sample_df[col], errors='coerce')
                if not numeric_series.isna().all():  # Has at least some numeric values
                    # Check if it's mostly numeric (at least 50% valid numbers)
                    valid_ratio = numeric_series.notna().sum() / len(numeric_series)
                    if valid_ratio > 0.5:
                        numeric_cols.append(col)
                    else:
                        non_numeric_cols.append(col)
                else:
                    non_numeric_cols.append(col)
            except:
                # If conversion fails, treat as non-numeric metadata
                non_numeric_cols.append(col)
        
        logger.info(f"Found {len(numeric_cols)} numeric columns (variables), {len(non_numeric_cols)} non-numeric columns (metadata)")
        
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
                
                # Extract metadata - COMPLETELY DYNAMIC: Store ALL non-numeric columns as metadata
                # No assumptions about column names - works for ANY dataset structure
                meta = {
                    "variable": str(var_col),
                    "source": str(path.name),
                    "format": "csv",
                }
                
                # Store ALL non-numeric columns as metadata (no hardcoded names)
                for meta_col in non_numeric_cols:
                    if meta_col in df_chunk.columns:
                        # Get unique values
                        meta_values = df_chunk[meta_col].dropna().unique()
                        if len(meta_values) > 0:
                            val = meta_values[0]
                            # Try to detect if it looks like a date/time (heuristic, not hardcoded)
                            val_str = str(val)
                            try:
                                # Try parsing as date
                                from datetime import datetime
                                pd.to_datetime(val_str)
                                # If successful, it's likely a time column
                                meta["time_start"] = val_str
                                time_vals = df_chunk[meta_col].dropna()
                                if len(time_vals) > 1:
                                    meta["time_end"] = str(time_vals.iloc[-1])
                            except:
                                # Not a date - store with original column name
                                if len(meta_values) == 1:
                                    meta[str(meta_col)] = val_str
                                else:
                                    meta[str(meta_col)] = val_str
                                    meta[f"{meta_col}_count"] = len(meta_values)
                                    if len(meta_values) <= 10:
                                        meta[f"{meta_col}_all"] = [str(v) for v in meta_values]
                
                # For numeric columns that aren't the variable itself, check if they might be spatial
                # Detect by value range, not by name
                for col in numeric_cols:
                    if col == var_col:
                        continue
                    try:
                        col_values = pd.to_numeric(df_chunk[col], errors='coerce').dropna()
                        if len(col_values) > 0:
                            min_val = float(col_values.min())
                            max_val = float(col_values.max())
                            # Latitude range: -90 to 90
                            if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                                meta["lat_min"] = min_val
                                meta["lat_max"] = max_val
                            # Longitude range: -180 to 180
                            elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                                meta["lon_min"] = min_val
                                meta["lon_max"] = max_val
                            # Store any other numeric metadata
                            else:
                                meta[f"{col}_min"] = min_val
                                meta[f"{col}_max"] = max_val
                                if len(col_values) == 1:
                                    meta[col] = min_val
                    except:
                        pass
                
                # Add chunk index
                meta["chunk_index"] = chunk_idx
                meta["row_count"] = len(df_chunk)
                
                yield RasterChunk(data=data, metadata=meta)
        
        logger.info(f"Successfully processed CSV: {path.name}")
        
    except Exception as e:
        logger.error(f"CSV Load Error for {path}: {e}")
        raise

def _load_geotiff(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """Loader for GeoTIFFs (Spatial Only). Dynamically extracts metadata from tags."""
    try:
        with rasterio.open(path) as src:
            w = 256
            if src.width < 512 or src.height < 512:
                w = 64
            
            # Extract ALL metadata from GeoTIFF tags (no hardcoding - store everything)
            geotiff_meta = {}
            if src.tags():
                geotiff_meta.update(dict(src.tags()))
            if src.tags(1):  # Band 1 tags
                geotiff_meta.update(dict(src.tags(1)))
            
            # Try to find variable name from any tag that might contain it
            # No hardcoded names - just look for descriptive tags
            variable_name = "band_1"  # Default fallback
            for key in geotiff_meta.keys():
                key_lower = str(key).lower()
                if any(word in key_lower for word in ['description', 'name', 'variable', 'title', 'label']):
                    val = geotiff_meta[key]
                    if val and str(val).strip():
                        variable_name = str(val).strip()
                        break
                
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
                    "variable": variable_name,
                    "source": str(path.name),
                    "format": "geotiff",
                    "lat_min": bounds[1],
                    "lat_max": bounds[3],
                    "lon_min": bounds[0],
                    "lon_max": bounds[2],
                    "time_start": "static_image",  # GeoTIFFs are typically static
                }
                
                # Store ALL GeoTIFF tags as metadata (no filtering - preserve everything)
                for key, val in geotiff_meta.items():
                    # Skip internal/technical tags
                    if key.startswith('_') or key in ['TIFFTAG_SAMPLESPERPIXEL', 'TIFFTAG_BITSPERSAMPLE']:
                        continue
                    try:
                        if isinstance(val, (int, float, str, bool)):
                            meta[f"geotiff_{key}"] = val
                        else:
                            meta[f"geotiff_{key}"] = str(val)
                    except:
                        meta[f"geotiff_{key}"] = str(val)
                
                yield RasterChunk(data=data, metadata=meta)
                
    except Exception as e:
        logger.error(f"GeoTIFF Load Error {path}: {e}")