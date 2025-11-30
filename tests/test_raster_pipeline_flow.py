"""End-to-end tests for auto-detecting formats and generating embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.embeddings.raster_pipeline import load_raster_auto, raster_to_embeddings


@dataclass
class SampleCase:
    name: str
    expected_format: str
    builder: Callable[[Path], Path]

    def __call__(self, tmp_path: Path) -> Path:  # pragma: no cover - helper
        return self.builder(tmp_path)


def _build_netcdf(tmp_path: Path) -> Path:
    times = pd.date_range("2020-01-01", periods=2)
    y = np.linspace(48.0, 49.0, 3)
    x = np.linspace(17.0, 18.0, 4)
    data = xr.DataArray(
        np.random.rand(len(times), len(y), len(x)).astype("float32"),
        coords={"time": times, "y": y, "x": x},
        dims=("time", "y", "x"),
        name="air",
    )
    ds = xr.Dataset({"air": data})
    path = tmp_path / "sample.nc"
    ds.to_netcdf(path)
    return path


def _build_geotiff(tmp_path: Path) -> Path:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = (np.arange(1, 17, dtype="float32").reshape(1, 4, 4))
    transform = from_origin(17.0, 49.0, 0.25, 0.25)
    path = tmp_path / "sample.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


def _build_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        {
            "station_id": [1, 1, 2, 2],
            "value": [10.5, 11.0, 9.8, 10.2],
            "timestamp": pd.date_range("2020-01-01", periods=4, freq="H"),
        }
    )
    path = tmp_path / "stations.csv"
    df.to_csv(path, index=False)
    return path


SAMPLE_CASES = [
    SampleCase("netcdf", "netcdf", _build_netcdf),
    SampleCase("geotiff", "geotiff", _build_geotiff),
    SampleCase("csv", "csv", _build_csv),
]


@pytest.mark.parametrize("case", SAMPLE_CASES, ids=lambda case: case.name)
def test_raster_pipeline_auto_detects_and_embeds(case: SampleCase, tmp_path: Path):
    data_path = case(tmp_path)
    result = load_raster_auto(data_path)

    assert result.metadata["format"] == case.expected_format

    embeddings = raster_to_embeddings(result)
    assert embeddings, f"Expected embeddings for {case.name}"

    for record in embeddings:
        vector = record.get("vector")
        metadata = record.get("metadata", {})
        assert isinstance(vector, list) and vector, "Vector should be non-empty list"
        assert metadata.get("format") == case.expected_format
        assert metadata.get("source_path", "").endswith(data_path.name)
