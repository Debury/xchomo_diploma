import gzip
import logging
import shutil
import subprocess
import tarfile
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
# MAGIC BYTE DETECTION
# ==============================================================================

def _detect_format_from_magic(path: Path) -> Optional[str]:
    """Detect file format from header magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if len(header) < 4:
            return None
        # NetCDF classic: starts with 'CDF\x01' or 'CDF\x02'
        if header[:3] == b"CDF":
            return "netcdf"
        # HDF5 / NetCDF-4: starts with '\x89HDF\r\n\x1a\n'
        if header[:8] == b"\x89HDF\r\n\x1a\n":
            return "hdf5"
        # GRIB: starts with 'GRIB'
        if header[:4] == b"GRIB":
            return "grib"
        # TIFF / GeoTIFF: 'II*\x00' (little-endian) or 'MM\x00*' (big-endian)
        if header[:4] in (b"II*\x00", b"MM\x00*"):
            return "geotiff"
        # Gzip: starts with '\x1f\x8b'
        if header[:2] == b"\x1f\x8b":
            return "gzip"
        return None
    except Exception:
        return None

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
    elif suffix == ".tar":
        iterator = _load_tar(path, **kwargs)
    elif suffix in {'.csv', '.tsv', '.txt'}:
        iterator = _load_csv(path, **kwargs)
    elif suffix in {'.tif', '.tiff'}:
        iterator = _load_geotiff(path, **kwargs)
    elif suffix in {'.h5', '.he5', '.hdf5'}:
        # HDF5 files: use h5netcdf (handles groups better than netcdf4)
        iterator = _load_xarray_generic(path, engine="h5netcdf", **kwargs)
    elif suffix in {'.nc', '.nc4', '.hdf'}:
        iterator = _load_xarray_generic(path, engine="netcdf4", **kwargs)
    elif suffix in {'.grib', '.grib2', '.grb'}:
        # Requires 'cfgrib' installed
        iterator = _load_xarray_generic(path, engine="cfgrib", **kwargs)
    elif suffix == '.gz':
        iterator = _load_gzip(path, **kwargs)
    else:
        # Fallback: detect format from file header magic bytes
        detected = _detect_format_from_magic(path)
        if detected == "netcdf":
            logger.info(f"Magic bytes detected NetCDF for '{suffix}', using xarray")
            iterator = _load_xarray_generic(path, engine="netcdf4", **kwargs)
        elif detected == "hdf5":
            logger.info(f"Magic bytes detected HDF5 for '{suffix}', using xarray")
            iterator = _load_xarray_generic(path, engine="netcdf4", **kwargs)
        elif detected == "geotiff":
            logger.info(f"Magic bytes detected GeoTIFF for '{suffix}', using rasterio")
            iterator = _load_geotiff(path, **kwargs)
        elif detected == "grib":
            logger.info(f"Magic bytes detected GRIB for '{suffix}', using cfgrib")
            iterator = _load_xarray_generic(path, engine="cfgrib", **kwargs)
        else:
            # No magic match — try xarray first (handles most scientific formats), then CSV
            logger.warning(f"Unknown extension '{suffix}', trying xarray first...")
            try:
                iterator = _load_xarray_generic(path, engine=None, **kwargs)
            except Exception as xr_err:
                # Only try CSV if file appears to be plausible text (not binary/HTML)
                with open(path, "rb") as f:
                    probe = f.read(512)
                probe_text = probe.decode("utf-8", errors="ignore").lower()
                if any(tag in probe_text for tag in ["<html", "<!doctype", "<head", "<body"]):
                    raise ValueError(f"File appears to be HTML, not data: {path.name}") from xr_err
                try:
                    probe.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    raise ValueError(f"File appears to be binary, not CSV: {path.name}") from xr_err
                logger.warning(f"Xarray load failed: {xr_err}, attempting CSV loader...")
                iterator = _load_csv(path, **kwargs)
        
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

MAX_DECOMPRESSED_SIZE_MB = 3000  # 3 GB limit for decompressed files

def _load_gzip(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """Decompress a .gz file and load the inner file via load_raster_auto."""
    # Determine inner suffix: e.g. file.nc.gz → .nc
    inner_suffix = Path(path.stem).suffix.lower() or ".nc"
    tmp_decompressed = None
    try:
        max_bytes = MAX_DECOMPRESSED_SIZE_MB * 1024 * 1024
        with tempfile.NamedTemporaryFile(suffix=inner_suffix, delete=False) as tmp:
            tmp_decompressed = tmp.name
            bytes_written = 0
            with gzip.open(path, "rb") as gz_in:
                while True:
                    buf = gz_in.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    bytes_written += len(buf)
                    if bytes_written > max_bytes:
                        raise ValueError(
                            f"Decompressed size exceeds {MAX_DECOMPRESSED_SIZE_MB} MB limit "
                            f"({bytes_written / 1e6:.0f} MB so far) — file too large for processing"
                        )
                    tmp.write(buf)
        decompressed_size = Path(tmp_decompressed).stat().st_size
        logger.info(f"Decompressed {path.name} → {inner_suffix} ({decompressed_size / 1e6:.1f} MB)")

        # Check if decompressed file is actually a tar archive (e.g. from .tar.gz)
        if tarfile.is_tarfile(tmp_decompressed):
            logger.info(f"Decompressed file is a tar archive, extracting...")
            yield from _load_tar(Path(tmp_decompressed), **kwargs)
        else:
            result = load_raster_auto(tmp_decompressed, **kwargs)
            yield from result.chunk_iterator
    finally:
        if tmp_decompressed:
            Path(tmp_decompressed).unlink(missing_ok=True)


def _load_tar(path: Path, **kwargs) -> Iterator[RasterChunk]:
    """Extract a tar archive and load supported files inside."""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with tarfile.open(path, "r:*") as tf:
                tf.extractall(tmpdir)
        except Exception as e:
            logger.error(f"Failed to extract tar {path.name}: {e}")
            return

        found = False
        for f in sorted(Path(tmpdir).rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            if f.suffix.lower() in {'.nc', '.nc4', '.h5'}:
                found = True
                yield from _load_xarray_generic(f, engine="netcdf4", **kwargs)
            elif f.suffix.lower() in {'.tif', '.tiff'}:
                found = True
                yield from _load_geotiff(f, **kwargs)
            elif f.suffix.lower() in {'.csv', '.tsv', '.txt'}:
                found = True
                yield from _load_csv(f, **kwargs)

        if not found:
            logger.warning(f"No supported files found inside tar: {path.name}")


def _extract_zip(path, tmpdir):
    """Extract ZIP file, falling back to system unzip if Python can't handle the compression."""
    try:
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(tmpdir)
        return True
    except NotImplementedError:
        logger.info(f"Python zipfile can't handle compression in {path.name}, trying system unzip")
        for cmd in [["unzip", "-o", "-q", str(path), "-d", tmpdir],
                    ["7z", "x", str(path), f"-o{tmpdir}", "-y"]]:
            try:
                subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        logger.error(f"No system tool could extract {path.name} (tried unzip, 7z)")
        return False


def _load_zip(path, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            if not _extract_zip(path, tmpdir):
                return

            found = False
            for f in Path(tmpdir).rglob("*"):
                if f.name.startswith("__MACOSX") or f.name.startswith("."):
                    continue

                if f.suffix.lower() in {'.nc', '.nc4', '.h5'}:
                    found = True
                    yield from _load_xarray_generic(f, engine="netcdf4", **kwargs)
                elif f.suffix.lower() in {'.grib', '.grib2'}:
                    found = True
                    yield from _load_xarray_generic(f, engine="cfgrib", **kwargs)
                elif f.suffix.lower() in {'.tif', '.tiff'}:
                    found = True
                    yield from _load_geotiff(f, **kwargs)
                elif f.suffix.lower() in {'.csv', '.tsv', '.txt'}:
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
    Tries engine fallback: requested engine → netcdf4 → h5netcdf → scipy → auto.
    """
    try:
        # Try requested engine first, then fallback chain
        engines_to_try = [engine] if engine else [None]
        if engine and engine != "cfgrib":
            # Add fallbacks for NetCDF files
            for fallback in ["netcdf4", "h5netcdf", "scipy", None]:
                if fallback != engine and fallback not in engines_to_try:
                    engines_to_try.append(fallback)

        ds = None
        last_err = None
        for eng in engines_to_try:
            try:
                ds = xr.open_dataset(path, chunks="auto", engine=eng)
                if eng != engine:
                    logger.info(f"Opened {path.name} with fallback engine '{eng}' (primary '{engine}' failed)")
                break
            except Exception as e:
                # Retry with decode_times=False for non-standard time units
                if "decode time" in str(e).lower() or "calendar" in str(e).lower():
                    try:
                        ds = xr.open_dataset(path, chunks="auto", engine=eng, decode_times=False)
                        logger.info(f"Opened {path.name} with engine '{eng}' + decode_times=False")
                        break
                    except Exception as e2:
                        last_err = e2
                        logger.debug(f"Engine '{eng}' + decode_times=False also failed: {e2}")
                        continue
                last_err = e
                logger.debug(f"Engine '{eng}' failed for {path.name}: {e}")
                continue

        if ds is None:
            raise last_err or RuntimeError(f"All xarray engines failed for {path.name}")

        logger.info(f"Opened {path.name}: dims={dict(ds.dims)}, vars={list(ds.data_vars)}")

        # HDF5 files may have data in groups (e.g. IMERG uses /Grid)
        # If we got 0 data vars, try opening with common group names
        if not list(ds.data_vars) and path.suffix.lower() in {'.h5', '.hdf', '.hdf5', '.he5'}:
            ds.close()
            for group in ["Grid", "HDFEOS/GRIDS", "data", "Data"]:
                try:
                    ds = xr.open_dataset(path, chunks="auto", engine="h5netcdf", group=group)
                    if list(ds.data_vars):
                        logger.info(f"Opened {path.name} with group='{group}': vars={list(ds.data_vars)}")
                        break
                    ds.close()
                except Exception:
                    continue

        # Filter for numeric vars only
        numeric_vars = [v for v in ds.data_vars if np.issubdtype(ds[v].dtype, np.number)]
        if not numeric_vars:
            logger.warning(f"No numeric variables found in {path.name}. All vars: {list(ds.data_vars)}, dtypes: {[(v, ds[v].dtype) for v in ds.data_vars]}")

        for var in numeric_vars:
            da_var = ds[var]
            
            # --- CHUNKING STRATEGY ---
            chunking = {}
            for dim in da_var.dims:
                d = str(dim).lower()
                if "time" in d or "date" in d or "step" in d:
                    # ~Seasonal aggregation: 90 timesteps per chunk
                    chunking[dim] = 90
                elif "lat" in d or "y" in d or "lon" in d or "x" in d:
                    # ~10° per chunk at 0.25° resolution (reduces chunk count significantly)
                    chunking[dim] = 40
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
            
            # --- Pre-load strategy: load entire variable into RAM once ---
            # This avoids 10-100K individual dask.compute() calls (huge overhead).
            # Falls back to per-chunk compute if the variable doesn't fit in memory.
            var_data_np = None
            try:
                var_nbytes = da_var.nbytes

                # Dynamic threshold: use at most 30% of *available* RAM,
                # capped at 2 GB absolute, so future large datasets degrade
                # gracefully instead of OOM-crashing.
                max_preload = 6 * 1024**3  # 6 GB absolute cap
                try:
                    import psutil
                    avail = psutil.virtual_memory().available
                    max_preload = min(max_preload, int(avail * 0.3))
                except ImportError:
                    pass  # psutil not installed — use absolute cap

                if var_nbytes < max_preload:
                    logger.info(
                        f"Pre-loading {var} ({var_nbytes / 1e6:.0f} MB) into RAM for fast slicing"
                    )
                    var_data_np = da_var.values  # single read, decompresses everything
                else:
                    logger.info(
                        f"Variable {var} ({var_nbytes / 1e6:.0f} MB) exceeds pre-load limit "
                        f"({max_preload / 1e6:.0f} MB avail), using per-chunk dask compute"
                    )
            except MemoryError:
                logger.warning(f"MemoryError pre-loading {var}, falling back to per-chunk")
                var_data_np = None
            except Exception as preload_err:
                logger.warning(f"Pre-load failed for {var}: {preload_err}, falling back to per-chunk")

            # --- Build metadata and yield chunks ---
            # When var_data_np is None (variable too large for pre-load),
            # batch-compute slices via dask.compute() to amortize I/O overhead.
            DASK_BATCH_SIZE = 500  # slices per dask.compute() call

            all_slices = list(da.core.slices_from_chunks(da_chunked.chunks))

            def _build_meta(slices):
                """Build metadata dict for a single chunk slice."""
                meta = {
                    "variable": str(var),
                    "source": str(path.name),
                    "format": "netcdf" if path.suffix.lower() in {'.nc', '.nc4', '.hdf', '.h5'} else "grib"
                }

                # Store ALL variable attributes (no hardcoding - works for ANY dataset)
                for attr_key, attr_val in var_attrs.items():
                    if attr_key.startswith('_') or attr_key in ['_FillValue', 'missing_value', 'fill_value']:
                        continue
                    try:
                        if isinstance(attr_val, (int, float, str, bool)):
                            meta[attr_key] = attr_val
                        else:
                            meta[attr_key] = str(attr_val)
                    except Exception:
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
                    except Exception:
                        meta[f"dataset_{attr_key}"] = str(attr_val)

                # Extract ALL coordinate dimensions
                for dim, sl in zip(da_var.dims, slices):
                    if dim in da_var.coords:
                        vals = da_var.coords[dim][sl].values
                        if len(vals) > 0:
                            meta[f"{dim}_start"] = str(vals.min())
                            if len(vals) > 1:
                                meta[f"{dim}_end"] = str(vals.max())

                            try:
                                from datetime import datetime
                                pd.to_datetime(str(vals.min()))
                                meta["time_start"] = str(vals.min())
                                if len(vals) > 1:
                                    meta["time_end"] = str(vals.max())
                            except Exception:
                                try:
                                    numeric_vals = vals[~np.isnan(vals)] if hasattr(vals, '__iter__') else [vals]
                                    if len(numeric_vals) > 0:
                                        min_val = float(np.min(numeric_vals))
                                        max_val = float(np.max(numeric_vals))
                                        if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                                            meta["lat_min"] = min_val
                                            meta["lat_max"] = max_val
                                        elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                                            meta["lon_min"] = min_val
                                            meta["lon_max"] = max_val
                                except Exception:
                                    pass
                return meta

            if var_data_np is not None:
                # Fast path: pre-loaded numpy array — slice directly
                for slices in all_slices:
                    try:
                        data = var_data_np[slices]
                        if np.isnan(data).all():
                            continue
                        yield RasterChunk(data=data, metadata=_build_meta(slices))
                    except MemoryError:
                        logger.error(f"MemoryError computing chunk for var={var} in {path.name} — skipping remaining chunks")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to compute chunk for var={var} in {path.name}: {e}")
                        continue
            else:
                # Batched dask compute: compute N slices in one call to amortize
                # disk seek + decompress overhead across hundreds of chunks.
                import dask
                for batch_start in range(0, len(all_slices), DASK_BATCH_SIZE):
                    batch_slices = all_slices[batch_start:batch_start + DASK_BATCH_SIZE]
                    try:
                        # Single dask.compute() for entire batch — one I/O pass
                        delayed_arrays = [da_chunked[sl] for sl in batch_slices]
                        computed = dask.compute(*delayed_arrays)
                        for sl, arr in zip(batch_slices, computed):
                            try:
                                data = arr.values if hasattr(arr, 'values') else np.asarray(arr)
                                if np.isnan(data).all():
                                    continue
                                yield RasterChunk(data=data, metadata=_build_meta(sl))
                            except Exception as e:
                                logger.warning(f"Failed processing chunk for var={var} in {path.name}: {e}")
                                continue
                        del computed  # free batch memory
                    except MemoryError:
                        logger.error(f"MemoryError in batch dask compute for var={var} in {path.name} — skipping remaining")
                        return
                    except Exception as e:
                        logger.warning(f"Batch dask compute failed for var={var} in {path.name}: {e}, falling back to per-chunk")
                        # Fallback: compute individually for this batch
                        for sl in batch_slices:
                            try:
                                data = da_chunked[sl].compute().values
                                if np.isnan(data).all():
                                    continue
                                yield RasterChunk(data=data, metadata=_build_meta(sl))
                            except Exception as inner_e:
                                logger.warning(f"Per-chunk fallback failed for var={var}: {inner_e}")
                                continue

            # Free pre-loaded data after processing all chunks for this variable
            del var_data_np
                    
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

        # Detect comment lines and auto-sniff delimiter from actual content
        comment_char = None
        try:
            with open(path, 'r', errors='replace') as f:
                first_lines = [f.readline() for _ in range(10)]
            # Check for common comment prefixes
            for ch in ('#', '//'):
                if any(line.startswith(ch) for line in first_lines if line.strip()):
                    comment_char = ch[0]
                    break
            # Auto-detect delimiter: if no commas found in data lines, try whitespace
            data_lines = [l for l in first_lines if l.strip() and (comment_char is None or not l.startswith(comment_char))]
            if data_lines and suffix not in ('.tsv',):
                sample_line = data_lines[0]
                if ',' not in sample_line and ('\t' in sample_line or '  ' in sample_line):
                    delimiter = r'\s+'  # whitespace-delimited
        except Exception:
            pass

        # First, read a small sample to understand structure
        read_kwargs = dict(nrows=100, delimiter=delimiter, low_memory=False)
        if comment_char:
            read_kwargs['comment'] = comment_char
        if delimiter == r'\s+':
            read_kwargs.pop('delimiter')
            read_kwargs.pop('low_memory', None)  # python engine doesn't support low_memory
            read_kwargs['sep'] = r'\s+'
            read_kwargs['engine'] = 'python'
        try:
            sample_df = pd.read_csv(path, **read_kwargs)
        except pd.errors.ParserError:
            # Fallback: try with python engine and error_bad_lines handling
            read_kwargs['engine'] = 'python'
            read_kwargs.pop('low_memory', None)  # python engine doesn't support low_memory
            read_kwargs['on_bad_lines'] = 'skip'
            if 'delimiter' in read_kwargs:
                read_kwargs['sep'] = read_kwargs.pop('delimiter')
            sample_df = pd.read_csv(path, **read_kwargs)
        
        # COMPLETELY DYNAMIC: No hardcoded column names
        # Classify columns purely by data type and content, not by name
        # CRITICAL: Latitude/longitude are numeric but should be treated as metadata
        numeric_cols = []
        non_numeric_cols = []
        spatial_cols = []  # Track latitude/longitude columns separately
        
        for col in sample_df.columns:
            col_lower = str(col).lower()
            # CRITICAL: Check if column is latitude/longitude by name first
            if 'lat' in col_lower and 'lon' not in col_lower:
                # Latitude column - treat as metadata, not variable
                spatial_cols.append(col)
                non_numeric_cols.append(col)
                continue
            elif 'lon' in col_lower:
                # Longitude column - treat as metadata, not variable
                spatial_cols.append(col)
                non_numeric_cols.append(col)
                continue
            
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
            except Exception:
                # If conversion fails, treat as non-numeric metadata
                non_numeric_cols.append(col)
        
        logger.info(f"Found {len(numeric_cols)} numeric columns (variables), {len(non_numeric_cols)} non-numeric columns (metadata)")
        
        if not numeric_cols:
            logger.warning(f"No numeric columns found in CSV: {path.name}")
            return
        
        # Detect if this is time-series data (has a time/date column)
        has_time_column = False
        time_col = None
        for col in non_numeric_cols:
            try:
                # Try to parse first value as date
                test_val = sample_df[col].dropna().iloc[0] if len(sample_df[col].dropna()) > 0 else None
                if test_val:
                    pd.to_datetime(str(test_val))
                    has_time_column = True
                    time_col = col
                    break
            except Exception:
                pass
        
        # BEST PRACTICE: For time-series CSV, create ONE embedding per variable (or variable + station combination)
        # This is more efficient and provides better context for RAG queries
        if has_time_column:
            logger.info(f"Detected time-series data, using variable-based chunking (one embedding per variable)")
            
            # Read entire CSV (reuse detected settings)
            full_kwargs = dict(delimiter=delimiter, low_memory=False)
            if comment_char:
                full_kwargs['comment'] = comment_char
            if delimiter == r'\s+':
                full_kwargs.pop('delimiter')
                full_kwargs.pop('low_memory', None)  # python engine doesn't support low_memory
                full_kwargs['sep'] = r'\s+'
                full_kwargs['engine'] = 'python'
            try:
                df_full = pd.read_csv(path, **full_kwargs)
            except pd.errors.ParserError:
                full_kwargs['engine'] = 'python'
                full_kwargs.pop('low_memory', None)  # python engine doesn't support low_memory
                full_kwargs['on_bad_lines'] = 'skip'
                if 'delimiter' in full_kwargs:
                    full_kwargs['sep'] = full_kwargs.pop('delimiter')
                df_full = pd.read_csv(path, **full_kwargs)
            
            # Detect if there are station/location columns (non-numeric metadata that might identify locations)
            station_cols = []
            for meta_col in non_numeric_cols:
                if meta_col == time_col:
                    continue
                # If this column has multiple unique values, it might be stations/locations
                unique_vals = df_full[meta_col].dropna().unique()
                if 1 < len(unique_vals) <= 100:  # Reasonable number of stations
                    station_cols.append(meta_col)
            
            # Process each variable
            for var_col in numeric_cols:
                # If we have station columns, create one embedding per variable + station combination
                if station_cols:
                    for station_col in station_cols:
                        stations = df_full[station_col].dropna().unique()
                        for station in stations:
                            # Filter data for this variable + station
                            df_filtered = df_full[df_full[station_col] == station].copy()
                            
                            # Extract values for this variable + station
                            values = pd.to_numeric(df_filtered[var_col], errors='coerce').values
                            values = values[~np.isnan(values)]
                            
                            if len(values) == 0:
                                continue
                            
                            # Convert to 2D array
                            data = values.reshape(1, -1).astype('float32')
                            
                            # Extract metadata
                            meta = {
                                "variable": str(var_col),
                                "source": str(path.name),
                                "format": "csv",
                                str(station_col): str(station),
                            }
                            
                            # Extract time range
                            if time_col and time_col in df_filtered.columns:
                                time_vals = df_filtered[time_col].dropna()
                                if len(time_vals) > 0:
                                    meta["time_start"] = str(time_vals.iloc[0])
                                    meta["time_end"] = str(time_vals.iloc[-1])
                            
                            # Store other metadata and detect spatial info
                            for meta_col in non_numeric_cols:
                                if meta_col == time_col or meta_col == station_col:
                                    continue
                                meta_values = df_filtered[meta_col].dropna().unique()
                                if len(meta_values) == 1:
                                    meta[str(meta_col)] = str(meta_values[0])
                                    
                                    # DYNAMIC: Check if this metadata column is latitude/longitude by name
                                    meta_col_lower = str(meta_col).lower()
                                    if 'lat' in meta_col_lower and 'lon' not in meta_col_lower:
                                        try:
                                            lat_val = float(meta_values[0])
                                            if -90 <= lat_val <= 90:
                                                meta["lat_min"] = lat_val
                                                meta["lat_max"] = lat_val
                                        except Exception:
                                            pass
                                    elif 'lon' in meta_col_lower:
                                        try:
                                            lon_val = float(meta_values[0])
                                            if -180 <= lon_val <= 180:
                                                meta["lon_min"] = lon_val
                                                meta["lon_max"] = lon_val
                                        except Exception:
                                            pass
                            
                            # Also check numeric columns for spatial info (CRITICAL: use filtered data, not full dataset)
                            for col in numeric_cols:
                                if col == var_col:
                                    continue
                                col_lower = str(col).lower()
                                # Check by column name first (more reliable) - CRITICAL: use df_filtered, not df_full
                                if 'lat' in col_lower and 'lon' not in col_lower:
                                    try:
                                        col_values = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
                                        if len(col_values) > 0:
                                            min_val = float(col_values.min())
                                            max_val = float(col_values.max())
                                            # CRITICAL: If all values are the same (single station), use that value
                                            if min_val == max_val:
                                                meta["lat_min"] = min_val
                                                meta["lat_max"] = min_val
                                            elif -90 <= min_val <= 90 and -90 <= max_val <= 90:
                                                # Multiple stations - use range
                                                meta["lat_min"] = min_val
                                                meta["lat_max"] = max_val
                                    except Exception:
                                        pass
                                elif 'lon' in col_lower:
                                    try:
                                        col_values = pd.to_numeric(df_filtered[col], errors='coerce').dropna()
                                        if len(col_values) > 0:
                                            min_val = float(col_values.min())
                                            max_val = float(col_values.max())
                                            # CRITICAL: If all values are the same (single station), use that value
                                            if min_val == max_val:
                                                meta["lon_min"] = min_val
                                                meta["lon_max"] = min_val
                                            elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                                                # Multiple stations - use range
                                                meta["lon_min"] = min_val
                                                meta["lon_max"] = max_val
                                    except Exception:
                                        pass
                            
                            meta["row_count"] = len(df_filtered)
                            
                            yield RasterChunk(data=data, metadata=meta)
                else:
                    # No station columns - create one embedding per variable for entire dataset
                    values = pd.to_numeric(df_full[var_col], errors='coerce').values
                    values = values[~np.isnan(values)]
                    
                    if len(values) == 0:
                        continue
                    
                    # Convert to 2D array
                    data = values.reshape(1, -1).astype('float32')
                    
                    # Extract metadata
                    meta = {
                        "variable": str(var_col),
                        "source": str(path.name),
                        "format": "csv",
                    }
                    
                    # Extract time range for entire dataset
                    if time_col and time_col in df_full.columns:
                        time_vals = df_full[time_col].dropna()
                        if len(time_vals) > 0:
                            meta["time_start"] = str(time_vals.iloc[0])
                            meta["time_end"] = str(time_vals.iloc[-1])
                    
                    # Store all metadata
                    for meta_col in non_numeric_cols:
                        if meta_col == time_col:
                            continue
                        meta_values = df_full[meta_col].dropna().unique()
                        if len(meta_values) == 1:
                            meta[str(meta_col)] = str(meta_values[0])
                        elif len(meta_values) <= 10:
                            meta[f"{meta_col}_all"] = [str(v) for v in meta_values]
                            meta[f"{meta_col}_count"] = len(meta_values)
                    
                    # Extract spatial info
                    for col in numeric_cols:
                        if col == var_col:
                            continue
                        try:
                            col_values = pd.to_numeric(df_full[col], errors='coerce').dropna()
                            if len(col_values) > 0:
                                min_val = float(col_values.min())
                                max_val = float(col_values.max())
                                if -90 <= min_val <= 90 and -90 <= max_val <= 90:
                                    meta["lat_min"] = min_val
                                    meta["lat_max"] = max_val
                                elif -180 <= min_val <= 180 and -180 <= max_val <= 180:
                                    meta["lon_min"] = min_val
                                    meta["lon_max"] = max_val
                        except Exception:
                            pass
                    
                    meta["row_count"] = len(df_full)
                    
                    yield RasterChunk(data=data, metadata=meta)
        else:
            # Non-time-series: use original chunking strategy (larger chunks)
            logger.info("Non-time-series data, using variable-based chunking")
            iter_kwargs = dict(
                delimiter=delimiter, chunksize=chunk_size,
                low_memory=False, iterator=True
            )
            if comment_char:
                iter_kwargs['comment'] = comment_char
            if delimiter == r'\s+':
                iter_kwargs.pop('delimiter')
                iter_kwargs.pop('low_memory', None)  # python engine doesn't support low_memory
                iter_kwargs['sep'] = r'\s+'
                iter_kwargs['engine'] = 'python'
            chunk_iter = pd.read_csv(path, **iter_kwargs)
            
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
                                except Exception:
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
                        except Exception:
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
                    except Exception:
                        meta[f"geotiff_{key}"] = str(val)
                
                yield RasterChunk(data=data, metadata=meta)
                
    except Exception as e:
        logger.error(f"GeoTIFF Load Error {path}: {e}")