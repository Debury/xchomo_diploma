from __future__ import annotations

import gzip
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas optional in some tests
    gpd = None

try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:  # pragma: no cover - rasterio optional in some tests
    rasterio = None
    Affine = None

try:
    import xarray as xr
except ImportError:  # pragma: no cover - xarray optional in some tests
    xr = None

from src.embeddings.generator import EmbeddingGenerator

CSV_MISSING_TOKENS = {"***", "***", "...", ".."}


SUPPORTED_FORMATS = {
    "netcdf",
    "grib",
    "hdf5",
    "geotiff",
    "esri_grid",
    "ascii_grid",
    "csv",
    "tsv",
    "txt",
    "json",
    "shapefile",
}


@dataclass
class FormatProcessResult:
    embeddings: List[Dict[str, object]]
    processed_file: Optional[Path]
    artifacts: Dict[str, object]


def _infer_format_from_suffix(file_path: Path) -> Optional[str]:
    """Guess format from file suffix chain (e.g., .tif.gz -> geotiff)."""

    suffix_map = {
        ".nc": "netcdf",
        ".nc4": "netcdf",
        ".cdf": "netcdf",
        ".grib": "grib",
        ".grb": "grib",
        ".grb2": "grib",
        ".h5": "hdf5",
        ".hdf5": "hdf5",
        ".tif": "geotiff",
        ".tiff": "geotiff",
        ".asc": "ascii_grid",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "txt",
        ".json": "json",
        ".geojson": "json",
        ".zip": "zip",
        ".gz": "gzip",
    }

    suffixes = [s.lower() for s in file_path.suffixes]
    for suffix in reversed(suffixes):
        if suffix in suffix_map:
            return suffix_map[suffix]
    return None


def _decompress_gzip(file_path: Path) -> Path:
    """Decompress .gz files next to the original and return the new path."""

    target_path = file_path.with_suffix("")
    with gzip.open(file_path, "rb") as source, open(target_path, "wb") as dest:
        shutil.copyfileobj(source, dest)
    return target_path


def prepare_file_for_processing(
    download_path: Path,
    detected_format: str,
    logger,
) -> Tuple[Path, str]:
    """Normalize downloaded assets (e.g., unzip shapefiles) before processing."""

    declared_format = (detected_format or "").lower()
    inferred_format = _infer_format_from_suffix(download_path)

    if inferred_format == "gzip":
        decompressed_path = _decompress_gzip(download_path)
        logger.info("Decompressed gzip archive %s -> %s", download_path.name, decompressed_path.name)
        download_path = decompressed_path
        inferred_format = _infer_format_from_suffix(download_path)

    canonical_format = declared_format or inferred_format or ""
    if inferred_format and canonical_format != inferred_format and inferred_format != "gzip":
        logger.warning(
            "Declared format '%s' mismatches %s; using '%s' inferred from file extension",
            declared_format or "unknown",
            download_path.name,
            inferred_format,
        )
        canonical_format = inferred_format

    if canonical_format == "zip":
        extract_dir = download_path.parent / f"{download_path.stem}_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(download_path, "r") as archive:
            archive.extractall(extract_dir)
            members = [Path(name) for name in archive.namelist()]

        def _find_member(predicate):
            for member in members:
                if predicate(member):
                    return member
            return None

        shapefile_member = _find_member(lambda m: m.suffix.lower() == ".shp")
        if shapefile_member:
            return extract_dir / shapefile_member, "shapefile"

        ascii_member = _find_member(lambda m: m.suffix.lower() == ".asc")
        if ascii_member:
            return extract_dir / ascii_member, "ascii_grid"

        txt_member = _find_member(lambda m: m.suffix.lower() == ".txt")
        if txt_member:
            return extract_dir / txt_member, "txt"

        csv_member = _find_member(lambda m: m.suffix.lower() == ".csv")
        if csv_member:
            return extract_dir / csv_member, "csv"

        json_member = _find_member(lambda m: m.suffix.lower() in {".json", ".geojson"})
        if json_member:
            return extract_dir / json_member, "json"

        hdr_member = _find_member(lambda m: m.name.lower().endswith("hdr.adf"))
        if hdr_member:
            return extract_dir / hdr_member, "esri_grid"

        raise ValueError("Unsupported ZIP archive contents for climate ingestion")

    return download_path, canonical_format


def _build_embedding(generator: EmbeddingGenerator, text: str) -> List[float]:
    embedding_array = generator.generate_embeddings([text])
    return embedding_array[0].tolist()


def _numeric_stats(values: np.ndarray) -> Optional[Dict[str, float]]:
    if values.size == 0:
        return None
    stats = {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "count": int(values.size),
    }
    if any(math.isnan(v) for v in stats.values()):
        return None
    return stats


def infer_temporal_coverage(frame: pd.DataFrame) -> Optional[Dict[str, str]]:
    candidate_cols = [
        col
        for col in frame.columns
        if isinstance(col, str)
        and col.strip()
        and any(keyword in col.lower() for keyword in ("year", "date", "time"))
    ]

    for col in candidate_cols:
        if "year" in col.lower():
            years = pd.to_numeric(frame[col], errors="coerce").dropna()
            if not years.empty:
                start_year = int(years.min())
                end_year = int(years.max())
                return {
                    "start": f"{start_year:04d}-01-01T00:00:00Z",
                    "end": f"{end_year:04d}-12-31T23:59:59Z",
                    "field": col,
                }

        parsed_dates = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna()
        if not parsed_dates.empty:
            return {
                "start": parsed_dates.min().isoformat().replace("+00:00", "Z"),
                "end": parsed_dates.max().isoformat().replace("+00:00", "Z"),
                "field": col,
            }

    return None


def _dataframe_embeddings(
    generator: EmbeddingGenerator,
    df: pd.DataFrame,
    source_id: str,
    timestamp: str,
    processed_path: Path,
    logger,
) -> FormatProcessResult:
    df.to_parquet(processed_path)

    coverage = infer_temporal_coverage(df)
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

    embeddings: List[Dict[str, object]] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats = {
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "count": int(series.count()),
        }
        text = (
            f"Column '{col}' statistics: n={stats['count']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f}."
        )
        metadata: Dict[str, object] = {
            "source_id": source_id,
            "variable": col,
            "text": text,
            "timestamp": timestamp,
        }
        if coverage:
            metadata.update(
                {
                    "start_timestamp": coverage["start"],
                    "end_timestamp": coverage["end"],
                    "temporal_coverage_field": coverage["field"],
                }
            )
        embeddings.append(
            {
                "id": f"{source_id}_{col}_{timestamp}",
                "embedding": _build_embedding(generator, text),
                "metadata": metadata,
            }
        )

    if not embeddings:
        logger.warning("No numeric columns found for tabular dataset")

    artifacts = {
        "format": "tabular",
        "columns": numeric_cols,
    }
    if coverage:
        artifacts["temporal_coverage"] = coverage

    return FormatProcessResult(embeddings=embeddings, processed_file=processed_path, artifacts=artifacts)


def _dataset_embeddings(
    generator: EmbeddingGenerator,
    ds: "xr.Dataset",
    source_id: str,
    timestamp: str,
    processed_path: Path,
    logger,
) -> FormatProcessResult:
    ds.to_netcdf(processed_path)

    embeddings: List[Dict[str, object]] = []
    for var in ds.data_vars:
        var_data = ds[var]
        data_np = var_data.values
        flat = data_np.reshape(-1)
        flat = flat[~np.isnan(flat)]
        if flat.size == 0:
            continue
        stats = _numeric_stats(flat)
        if not stats:
            continue
        units = var_data.attrs.get("units", "unknown")
        text = (
            f"Variable '{var}' summary: mean={stats['mean']:.3f} {units}, std={stats['std']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f}."
        )
        if var_data.attrs.get("long_name"):
            text += f" Description: {var_data.attrs['long_name']}."
        metadata = {
            "source_id": source_id,
            "variable": var,
            "units": units,
            "timestamp": timestamp,
            "text": text,
        }
        embeddings.append(
            {
                "id": f"{source_id}_{var}_{timestamp}",
                "embedding": _build_embedding(generator, text),
                "metadata": metadata,
            }
        )

    if not embeddings:
        logger.warning("No numeric variables found in dataset")

    return FormatProcessResult(
        embeddings=embeddings,
        processed_file=processed_path,
        artifacts={"variables": list(ds.data_vars)},
    )


def _raster_embeddings(
    generator: EmbeddingGenerator,
    raster_path: Path,
    source_id: str,
    timestamp: str,
    processed_path: Path,
    logger,
) -> FormatProcessResult:
    if not rasterio:
        raise RuntimeError("rasterio is not installed, cannot process raster data")

    shutil.copy(raster_path, processed_path)
    embeddings: List[Dict[str, object]] = []

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        meta_common = {
            "source_id": source_id,
            "timestamp": timestamp,
            "crs": str(src.crs) if src.crs else None,
            "bounds": {
                "west": bounds.left,
                "south": bounds.bottom,
                "east": bounds.right,
                "north": bounds.top,
            },
        }
        for band_idx in range(1, src.count + 1):
            data = src.read(band_idx, masked=True)
            values = data.compressed() if np.ma.isMaskedArray(data) else data.flatten()
            values = values[np.isfinite(values)]
            stats = _numeric_stats(values)
            if not stats:
                continue
            text = (
                f"Raster band {band_idx}: n={stats['count']}, mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                f"min={stats['min']:.3f}, max={stats['max']:.3f}."
            )
            metadata = {
                **meta_common,
                "band": band_idx,
                "text": text,
            }
            embeddings.append(
                {
                    "id": f"{source_id}_band{band_idx}_{timestamp}",
                    "embedding": _build_embedding(generator, text),
                    "metadata": metadata,
                }
            )

    if not embeddings:
        logger.warning("No valid raster samples found")

    return FormatProcessResult(
        embeddings=embeddings,
        processed_file=processed_path,
        artifacts={"type": "raster"},
    )


def _json_to_dataframe(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "features" in payload and gpd is not None:
        gdf = gpd.GeoDataFrame.from_features(payload["features"])
        if "geometry" in gdf:
            gdf = gdf.drop(columns=["geometry"], errors="ignore")
        return pd.DataFrame(gdf)

    return pd.json_normalize(payload)


def _detect_delimited_header_offset(file_path: Path, delimiter: str) -> int:
    """Skip metadata lines until the first line that looks like a header."""

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            for idx, line in enumerate(handle):
                if delimiter in line:
                    return idx
    except OSError:
        return 0
    return 0


def _promote_numeric_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Best-effort conversion of object columns that mostly contain numbers."""

    promoted = frame.copy()
    if promoted.empty:
        return promoted

    min_required = max(3, int(0.2 * len(promoted)))
    for col in promoted.columns:
        if pd.api.types.is_numeric_dtype(promoted[col]):
            continue
        converted = pd.to_numeric(promoted[col], errors="coerce")
        if converted.notna().sum() >= min_required:
            promoted[col] = converted
    return promoted


def _open_dataset_with_fallback(file_path: Path, logger):
    """Open NetCDF file but fall back to cfgrib engine when needed."""

    if not xr:
        raise RuntimeError("xarray is not installed, cannot process NetCDF data")

    try:
        ds = xr.open_dataset(file_path)
        return ds, "netcdf"
    except Exception as exc:
        logger.warning(
            "Default NetCDF open failed for %s (%s); trying cfgrib fallback",
            file_path.name,
            exc,
        )
        try:
            ds = xr.open_dataset(file_path, engine="cfgrib")
            logger.info("Opened %s via cfgrib fallback", file_path.name)
            return ds, "grib"
        except Exception as cf_exc:
            logger.error(
                "cfgrib fallback failed for %s: %s",
                file_path.name,
                cf_exc,
            )
            raise RuntimeError(
                f"Could not open dataset {file_path} as NetCDF or GRIB"
            ) from cf_exc


def generate_embeddings_for_file(
    generator: EmbeddingGenerator,
    canonical_format: str,
    file_path: Path,
    source_id: str,
    timestamp: str,
    processed_dir: Path,
    logger,
) -> FormatProcessResult:
    processed_dir.mkdir(parents=True, exist_ok=True)

    format_lower = canonical_format.lower()

    if format_lower == "netcdf":
        ds, opened_as = _open_dataset_with_fallback(file_path, logger)
        suffix = "grib.nc" if opened_as == "grib" else "nc"
        processed_path = processed_dir / f"{source_id}_processed.{suffix}"
        try:
            return _dataset_embeddings(generator, ds, source_id, timestamp, processed_path, logger)
        finally:
            ds.close()

    if format_lower == "grib":
        if not xr:
            raise RuntimeError("xarray is not installed, cannot process GRIB data")
        ds = xr.open_dataset(file_path, engine="cfgrib")
        processed_path = processed_dir / f"{source_id}_processed.grib.nc"
        try:
            return _dataset_embeddings(generator, ds, source_id, timestamp, processed_path, logger)
        finally:
            ds.close()

    if format_lower == "hdf5":
        if not xr:
            raise RuntimeError("xarray is not installed, cannot process HDF5 data")
        ds = xr.open_dataset(file_path, engine="h5netcdf")
        processed_path = processed_dir / f"{source_id}_processed.h5"
        try:
            return _dataset_embeddings(generator, ds, source_id, timestamp, processed_path, logger)
        finally:
            ds.close()

    if format_lower in {"geotiff", "esri_grid", "ascii_grid"}:
        suffix = {
            "geotiff": "tif",
            "esri_grid": "grid",
            "ascii_grid": "asc",
        }[format_lower]
        processed_path = processed_dir / f"{source_id}_processed.{suffix}"
        return _raster_embeddings(generator, file_path, source_id, timestamp, processed_path, logger)

    if format_lower in {"csv", "tsv", "txt"}:
        if format_lower == "txt":
            df = pd.read_csv(file_path, sep=r"\s+", engine="python")
        else:
            sep = "," if format_lower == "csv" else "\t"
            skiprows = _detect_delimited_header_offset(file_path, sep)
            df = pd.read_csv(
                file_path,
                sep=sep,
                engine="python",
                comment="#",
                skip_blank_lines=True,
                skiprows=skiprows,
                na_values=list(CSV_MISSING_TOKENS),
            )
        df = df.replace(list(CSV_MISSING_TOKENS), np.nan)
        df = df.dropna(how="all")
        df = _promote_numeric_columns(df)
        processed_path = processed_dir / f"{source_id}_processed.parquet"
        return _dataframe_embeddings(generator, df, source_id, timestamp, processed_path, logger)

    if format_lower == "json":
        df = _json_to_dataframe(file_path)
        processed_path = processed_dir / f"{source_id}_processed.parquet"
        return _dataframe_embeddings(generator, df, source_id, timestamp, processed_path, logger)

    if format_lower == "shapefile":
        if gpd is None:
            raise RuntimeError("geopandas is not installed, cannot process shapefiles")
        gdf = gpd.read_file(file_path)
        df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
        processed_path = processed_dir / f"{source_id}_processed.parquet"
        return _dataframe_embeddings(generator, df, source_id, timestamp, processed_path, logger)

    raise ValueError(f"Unsupported canonical format '{canonical_format}' for embedding generation")
