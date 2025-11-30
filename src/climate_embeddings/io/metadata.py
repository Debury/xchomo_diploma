"""Metadata extraction utilities for climate datasets."""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr


class MetadataExtractor:
    """Extract structured summaries from xarray datasets."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.sample_count = int(self.config.get("sample_count", 5))

    def extract_from_dataset(
        self,
        data: xr.Dataset,
        file_path: str,
        dataset_id: str,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        timestamp = dt.datetime.utcnow().isoformat()

        for var_name, data_array in data.data_vars.items():
            record = self._build_record(
                dataset_id=dataset_id,
                file_path=file_path,
                timestamp=timestamp,
                variable_name=var_name,
                data_array=data_array,
            )
            records.append(record)
        return records

    # ------------------------------------------------------------------
    def _build_record(
        self,
        dataset_id: str,
        file_path: str,
        timestamp: str,
        variable_name: str,
        data_array: xr.DataArray,
    ) -> Dict[str, Any]:
        values = data_array.values
        finite_values = self._flatten(values)
        stats = self._stats(finite_values)

        record: Dict[str, Any] = {
            "id": f"{dataset_id}-{variable_name}",
            "dataset_id": dataset_id,
            "file_path": file_path,
            "variable": variable_name,
            "long_name": data_array.attrs.get("long_name", variable_name),
            "unit": data_array.attrs.get("units", data_array.attrs.get("unit", "")),
            "dimensions": list(data_array.dims),
            "shape": [int(dim) for dim in data_array.shape],
            "stat_min": stats["min"],
            "stat_max": stats["max"],
            "stat_mean": stats["mean"],
            "stat_std": stats["std"],
            "timestamp": timestamp,
            "spatial_extent": self._spatial_extent(data_array),
            "temporal_extent": self._temporal_extent(data_array),
            "sample_values": finite_values[: self.sample_count],
        }
        return record

    def _flatten(self, values: Any) -> List[float]:
        arr = np.asarray(values).astype(float)
        finite = arr[np.isfinite(arr)]
        return finite.tolist()

    def _stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
        arr = np.asarray(values, dtype=np.float32)
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    def _spatial_extent(self, data_array: xr.DataArray) -> Dict[str, Any]:
        extent: Dict[str, Any] = {}
        lat = self._coord_if_exists(data_array, {"lat", "latitude"})
        lon = self._coord_if_exists(data_array, {"lon", "longitude"})
        if lat is not None:
            extent["lat"] = {
                "min": float(np.nanmin(lat.values)),
                "max": float(np.nanmax(lat.values)),
            }
        if lon is not None:
            extent["lon"] = {
                "min": float(np.nanmin(lon.values)),
                "max": float(np.nanmax(lon.values)),
            }
        return extent

    def _temporal_extent(self, data_array: xr.DataArray) -> Dict[str, Any]:
        time_coord = self._coord_if_exists(data_array, {"time", "date"})
        if time_coord is None:
            return {}
        values = time_coord.values
        earliest = values[0]
        latest = values[-1]
        return {
            "start": self._to_iso(earliest),
            "end": self._to_iso(latest),
        }

    def _coord_if_exists(self, data_array: xr.DataArray, names: set[str]) -> Optional[xr.DataArray]:
        for name in names:
            coord = data_array.coords.get(name)
            if coord is not None:
                return coord
        return None

    def _to_iso(self, value: Any) -> str:
        if isinstance(value, np.datetime64):
            return str(np.datetime_as_string(value, unit="s"))
        if isinstance(value, dt.datetime):
            return value.isoformat()
        return str(value)
