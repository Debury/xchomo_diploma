"""
Memory-safe raster → embeddings pipeline for climate datasets.

Features
--------
* Auto-detects NetCDF, GRIB, HDF5, GeoTIFF/COG, ASCII grids, CSV/TXT station data, and Zarr stores.
* Streams rasters with xarray+dask chunking or rasterio windows (never loads entire files into RAM).
* Applies time slicing, spatial cropping, variable selection, unit harmonization, normalization, and pooling.
* Emits deterministic embeddings per chunk/tile with metadata suitable for vector DB ingestion.
* Can persist embeddings to .jsonl/.npy or push them to an arbitrary vector database client.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import dask.array as da
from dask.array.core import slices_from_chunks
import rasterio
from rasterio.windows import Window

# Optional imports for specific engines
try:  # pragma: no cover
    import cfgrib  # noqa: F401
except ImportError:  # pragma: no cover
    cfgrib = None

try:  # pragma: no cover
    import h5netcdf  # noqa: F401
except ImportError:  # pragma: no cover
    h5netcdf = None

SUPPORTED_EXTENSIONS: Dict[str, str] = {
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
    ".asc": "ascii",
    ".txt": "csv",
    ".csv": "csv",
    ".zarr": "zarr",
}

TEMPORAL_POOLING_RULES = {
    "monthly": "1M",
    "seasonal": "QS-DEC",
    "annual": "1Y",
}

UNIT_CONVERSIONS = {
    ("kelvin", "celsius"): lambda data: data - 273.15,
    ("k", "celsius"): lambda data: data - 273.15,
    ("pa", "hpa"): lambda data: data / 100.0,
    ("m/s", "km/h"): lambda data: data * 3.6,
}


@dataclass
class RasterLoadResult:
    """Holds either an xarray dataset or a streaming chunk iterator plus metadata."""

    dataset: Optional[xr.Dataset]
    chunk_iterator: Optional[Iterator[Tuple[np.ndarray, Dict[str, Any]]]]
    source_path: Path
    metadata: Dict[str, Any]


class UnsupportedRasterFormat(RuntimeError):
    """Raised when the loader cannot infer a supported format."""


def _guess_format(path: Path) -> str:
    for suffix in path.suffixes[::-1]:
        fmt = SUPPORTED_EXTENSIONS.get(suffix.lower())
        if fmt:
            return fmt
    raise UnsupportedRasterFormat(f"Unsupported file extension(s) for {path}")


def _estimate_chunk_pixels(path: Path, target_mb: int = 64) -> int:
    size_mb = max(path.stat().st_size / 1024 / 1024, 1)
    ratio = min(target_mb / size_mb, 1.0)
    baseline = 1024  # ~32MB window for float32 multi-band rasters
    chunk = max(int(baseline * math.sqrt(ratio)), 128)
    return chunk


def _coerce_chunk_dict(
    chunks: Union[str, Dict[str, int]],
    dim_names: Iterable[str],
    path: Path,
) -> Dict[str, int]:
    if chunks == "auto":
        size = _estimate_chunk_pixels(path)
        return {dim: size for dim in dim_names}
    if isinstance(chunks, dict):
        return chunks
    return {}


def _apply_filters(
    ds: xr.Dataset,
    variables: Optional[List[str]],
    time_slice: Optional[Tuple[str, str]],
    bbox: Optional[Tuple[float, float, float, float]],
) -> xr.Dataset:
    if variables:
        missing = set(variables) - set(ds.data_vars)
        if missing:
            raise KeyError(f"Variables not found in dataset: {missing}")
        ds = ds[variables]

    if time_slice and "time" in ds.coords:
        ds = ds.sel(time=slice(time_slice[0], time_slice[1]))

    if bbox:
        lon_candidates = [name for name in ("lon", "longitude", "x") if name in ds.coords]
        lat_candidates = [name for name in ("lat", "latitude", "y") if name in ds.coords]
        if lon_candidates and lat_candidates:
            lon_name, lat_name = lon_candidates[0], lat_candidates[0]
            lon_min, lat_min, lon_max, lat_max = bbox
            ds = ds.sel({lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})
    return ds


def _convert_units(data_array: xr.DataArray, target_unit: Optional[str]) -> xr.DataArray:
    current_unit = (data_array.attrs.get("units") or data_array.attrs.get("Units") or "").lower()
    if not current_unit or not target_unit:
        return data_array
    target_unit = target_unit.lower()
    if current_unit == target_unit:
        return data_array
    conversion = UNIT_CONVERSIONS.get((current_unit, target_unit))
    if conversion:
        converted = conversion(data_array)
        converted.attrs["units"] = target_unit
        return converted
    return data_array


def _ensure_monotonic_coord(data_array: xr.DataArray, coord_name: str) -> xr.DataArray:
    if coord_name not in data_array.coords:
        return data_array
    coord = data_array.coords[coord_name]
    if coord.ndim == 1 and not coord.to_index().is_monotonic_increasing:
        sorted_idx = np.argsort(coord.values)
        data_array = data_array.isel({coord_name: sorted_idx})
    return data_array


def _prepare_data_array(
    data_array: xr.DataArray,
    target_unit: Optional[str],
    fill_method: str,
) -> xr.DataArray:
    da_prepared = _convert_units(data_array, target_unit)
    for coord_name in ("lat", "latitude", "y", "lon", "longitude", "x"):
        da_prepared = _ensure_monotonic_coord(da_prepared, coord_name)
    if fill_method == "interpolate" and "time" in da_prepared.coords:
        da_prepared = da_prepared.interpolate_na(dim="time")
    da_prepared = da_prepared.where(np.isfinite(da_prepared))
    return da_prepared


def _dimension_chunk_sizes(array: da.Array, dims: Tuple[str, ...]) -> Dict[str, int]:
    return {dim: max(size_list) for dim, size_list in zip(dims, array.chunks)}


def _shrink_chunks(array: da.Array, dims: Tuple[str, ...]) -> da.Array:
    current = _dimension_chunk_sizes(array, dims)
    smaller = {dim: max(size // 2, 32) for dim, size in current.items()}
    return array.rechunk(smaller)


def load_raster_auto(
    path: Union[str, Path],
    *,
    chunks: Union[str, Dict[str, int]] = "auto",
    convert_to_zarr: bool = False,
    time_slice: Optional[Tuple[str, str]] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    variables: Optional[List[str]] = None,
    station_chunksize: Optional[int] = None,
) -> RasterLoadResult:
    """Detect dataset format, open with optimal settings, and stream chunk references."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    fmt = _guess_format(path)
    metadata: Dict[str, Any] = {"format": fmt, "path": str(path)}

    if fmt in {"netcdf", "hdf5", "grib"}:
        engine_map = {"netcdf": None, "hdf5": "h5netcdf", "grib": "cfgrib"}
        chunk_hint = _coerce_chunk_dict(chunks, ["y", "x"], path)
        try:
            ds = xr.open_dataset(
                path,
                engine=engine_map[fmt],
                chunks=chunk_hint or "auto",
                mask_and_scale=True,
                use_cftime=True,
            )
        except MemoryError:
            smaller = {dim: max(size // 2, 64) for dim, size in chunk_hint.items() or {"y": 512, "x": 512}.items()}
            ds = xr.open_dataset(
                path,
                engine=engine_map[fmt],
                chunks=smaller,
                mask_and_scale=True,
                use_cftime=True,
            )
        ds = _apply_filters(ds, variables, time_slice, bbox)
        if convert_to_zarr:
            zarr_path = path.with_suffix(path.suffix + ".zarr")
            if not zarr_path.exists():
                ds.chunk(chunk_hint or {}).to_zarr(zarr_path, mode="w")
            metadata["zarr_store"] = str(zarr_path)
        return RasterLoadResult(dataset=ds, chunk_iterator=None, source_path=path, metadata=metadata)

    if fmt == "zarr":
        ds = xr.open_zarr(path, chunks=chunks if chunks != "auto" else None)
        ds = _apply_filters(ds, variables, time_slice, bbox)
        return RasterLoadResult(dataset=ds, chunk_iterator=None, source_path=path, metadata=metadata)

    if fmt == "ascii":
        ds = xr.open_dataset(path, engine="rasterio", chunks=chunks if chunks != "auto" else None)
        ds = _apply_filters(ds, variables, time_slice, bbox)
        return RasterLoadResult(dataset=ds, chunk_iterator=None, source_path=path, metadata=metadata)

    if fmt == "geotiff":
        window_size = _estimate_chunk_pixels(path, target_mb=32)

        def _geo_generator() -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
            with rasterio.Env():
                with rasterio.open(path) as src:
                    nrows, ncols = src.height, src.width
                    nodata = src.nodata
                    transform = src.transform
                    for row in range(0, nrows, window_size):
                        for col in range(0, ncols, window_size):
                            height = min(window_size, nrows - row)
                            width = min(window_size, ncols - col)
                            window = Window(col, row, width, height)
                        data = src.read(window=window, out_dtype="float32")
                        if nodata is not None:
                            data = np.where(data == nodata, np.nan, data)
                        bounds = rasterio.windows.bounds(window, transform)
                        # bounds is (left, bottom, right, top) tuple
                        meta = {
                            "lon_min": bounds[0],
                            "lon_max": bounds[2],
                            "lat_min": bounds[1],
                            "lat_max": bounds[3],
                            "row": row,
                            "col": col,
                            "height": height,
                            "width": width,
                        }
                        yield data, meta        return RasterLoadResult(dataset=None, chunk_iterator=_geo_generator(), source_path=path, metadata=metadata)

    if fmt == "csv":
        chunk_rows = station_chunksize or (50_000 if path.stat().st_size > 5 * 1024 * 1024 else None)

        def _csv_generator() -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
            reader = pd.read_csv(path, chunksize=chunk_rows) if chunk_rows else [pd.read_csv(path)]
            for frame in reader:
                numeric_cols = frame.select_dtypes(include="number")
                for column in numeric_cols.columns:
                    series = numeric_cols[column].dropna().astype("float32")
                    meta = {
                        "column": column,
                        "rows": len(series),
                    }
                    yield series.to_numpy(), meta

        return RasterLoadResult(dataset=None, chunk_iterator=_csv_generator(), source_path=path, metadata=metadata)

    raise UnsupportedRasterFormat(f"Format '{fmt}' is not supported yet")


def _normalize_chunk(data: np.ndarray, method: str) -> np.ndarray:
    buffer = data.astype("float32", copy=True)
    mask = np.isfinite(buffer)
    if not np.any(mask):
        return buffer
    valid = buffer[mask]
    if method == "zscore":
        mean = valid.mean()
        std = valid.std() or 1.0
        buffer[mask] = (valid - mean) / std
    elif method == "minmax":
        mn = valid.min()
        mx = valid.max()
        span = mx - mn if mx > mn else 1.0
        buffer[mask] = (valid - mn) / span
    return buffer


def _stats_vector(chunk: np.ndarray, pooling: bool) -> np.ndarray:
    data = chunk
    if pooling and data.ndim > 2:
        data = np.nanmean(data, axis=0, keepdims=True)
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return np.zeros(7, dtype="float32")
    percentiles = np.percentile(valid, [10, 50, 90]).astype("float32")
    return np.array(
        [
            valid.mean(),
            valid.std(),
            valid.min(),
            valid.max(),
            percentiles[0],
            percentiles[1],
            percentiles[2],
        ],
        dtype="float32",
    )


def _chunk_metadata(data_array: xr.DataArray, slices: Tuple[slice, ...]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for dim, s in zip(data_array.dims, slices):
        coord = data_array.coords.get(dim)
        if coord is None:
            continue
        lower = (s.start or 0)
        upper = s.stop
        window_values = coord.isel({dim: slice(lower, upper)})
        if dim.lower().startswith(("lat", "y")):
            meta["lat_min"] = float(window_values.min())
            meta["lat_max"] = float(window_values.max())
        elif dim.lower().startswith(("lon", "x")):
            meta["lon_min"] = float(window_values.min())
            meta["lon_max"] = float(window_values.max())
        elif dim.lower().startswith("time"):
            meta["time_start"] = str(window_values.min().values)
            meta["time_end"] = str(window_values.max().values)
    return meta


def raster_to_embeddings(
    source: RasterLoadResult,
    *,
    normalization: str = "zscore",
    spatial_pooling: bool = True,
    temporal_pooling: Optional[str] = None,
    fill_method: str = "interpolate",
    target_units: Optional[Dict[str, str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Convert either xarray chunks or streamed tiles into embedding dicts."""

    np.random.seed(seed)
    embeddings: List[Dict[str, Any]] = []

    def _collect(vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        embeddings.append({"vector": vector.tolist(), "metadata": metadata})

    if source.dataset is not None:
        ds = source.dataset
        for variable, data_array in ds.data_vars.items():
            prepared = _prepare_data_array(
                data_array,
                target_units.get(variable) if target_units else None,
                fill_method,
            )
            if temporal_pooling and "time" in prepared.dims:
                rule = TEMPORAL_POOLING_RULES.get(temporal_pooling.lower())
                if rule:
                    prepared = prepared.resample(time=rule).mean()
            da_data: da.Array = prepared.data
            if not hasattr(da_data, "chunks"):
                da_data = da.from_array(da_data, chunks="auto")
            for slices in slices_from_chunks(da_data.chunks):
                try:
                    chunk = da_data[slices].compute()
                except MemoryError:
                    da_data = _shrink_chunks(da_data, prepared.dims)
                    chunk = da_data[slices].compute()
                chunk = _normalize_chunk(np.array(chunk, dtype="float32"), normalization)
                vector = _stats_vector(chunk, spatial_pooling)
                metadata = {
                    "variable": variable,
                    "source_path": str(source.source_path),
                    "format": source.metadata.get("format"),
                }
                metadata.update(_chunk_metadata(prepared, slices))
                _collect(vector, metadata)
    elif source.chunk_iterator is not None:
        for tile_idx, (chunk, meta) in enumerate(source.chunk_iterator):
            chunk = _normalize_chunk(chunk, normalization)
            vector = _stats_vector(chunk, spatial_pooling)
            metadata = {
                "tile_index": tile_idx,
                "source_path": str(source.source_path),
                "format": source.metadata.get("format"),
            }
            metadata.update(meta)
            _collect(vector, metadata)
    else:
        raise RuntimeError("RasterLoadResult contains neither dataset nor chunk iterator")

    return embeddings


def save_embeddings(
    embeddings: List[Dict[str, Any]],
    destination: Union[str, Path],
    *,
    fmt: str = "jsonl",
    vector_client: Optional[Any] = None,
) -> None:
    """Persist embeddings to disk or send them to a vector database interface."""

    destination = Path(destination)
    fmt = fmt.lower()
    if fmt == "jsonl":
        with destination.open("w", encoding="utf-8") as handle:
            for record in embeddings:
                handle.write(json.dumps(record) + "\n")
    elif fmt == "npy":
        vectors = np.array([record["vector"] for record in embeddings], dtype="float32")
        np.save(destination, vectors)
    elif fmt == "vector":
        if vector_client is None:
            raise ValueError("vector_client is required when fmt='vector'")
        for record in embeddings:
            vector_client.upsert_embedding(vector=record["vector"], metadata=record["metadata"])
    else:
        raise ValueError(f"Unsupported save format '{fmt}'")


def _example_air_temperature() -> None:
    """Run the pipeline against the xarray tutorial air temperature dataset."""

    sample_path = Path("air_temperature.nc")
    if not sample_path.exists():
        ds = xr.tutorial.open_dataset("air_temperature")
        ds.to_netcdf(sample_path)

    result = load_raster_auto(
        sample_path,
        chunks="auto",
        time_slice=("2014-12-01", "2015-01-01"),
        bbox=(-140, 20, -60, 60),
        variables=["air"],
    )

    embeddings = raster_to_embeddings(
        result,
        normalization="zscore",
        spatial_pooling=True,
        temporal_pooling="monthly",
        target_units={"air": "celsius"},
    )

    output_path = Path("air_embeddings.jsonl")
    save_embeddings(embeddings, output_path, fmt="jsonl")
    print(f"Generated {len(embeddings)} embeddings -> {output_path}")


if __name__ == "__main__":  # pragma: no cover
    _example_air_temperature()
