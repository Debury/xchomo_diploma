import gzip
import json
import logging
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.utils.format_processing as format_processing
from src.utils.format_processing import (
    generate_embeddings_for_file,
    prepare_file_for_processing,
)


class DummyGenerator:
    """Deterministic embedding generator used for fast unit tests."""

    def generate_embeddings(self, texts):
        if not texts:
            raise ValueError("No texts provided")
        vectors = []
        for idx, text in enumerate(texts):
            base = float(len(text) + idx + 1)
            vectors.append(np.array([base, base + 1.0, base + 2.0], dtype=np.float32))
        return np.vstack(vectors)


@pytest.fixture(scope="module")
def dummy_generator():
    return DummyGenerator()


@pytest.fixture(scope="module")
def test_logger():
    return logging.getLogger("format-processing-tests")


@pytest.fixture
def processed_dir(tmp_path):
    dir_path = tmp_path / "processed"
    dir_path.mkdir()
    return dir_path


def test_generate_embeddings_for_netcdf(tmp_path, processed_dir, dummy_generator, test_logger):
    xr = pytest.importorskip("xarray")

    data = xr.Dataset(
        {
            "temperature": (("time", "lat"), np.array([[278.0, 279.2], [280.5, 281.7]])),
        },
        coords={
            "time": pd.date_range("2020-01-01", periods=2),
            "lat": [50.0, 51.0],
        },
    )
    data.temperature.attrs["units"] = "K"
    data.temperature.attrs["long_name"] = "Near surface air temperature"

    file_path = tmp_path / "sample.nc"
    data.to_netcdf(file_path)

    result = generate_embeddings_for_file(
        dummy_generator,
        "netcdf",
        file_path,
        "source_nc",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.processed_file.exists()


def test_generate_embeddings_for_hdf5(tmp_path, processed_dir, dummy_generator, test_logger):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")

    dataset = xr.Dataset(
        {
            "humidity": (("time",), np.array([0.45, 0.5, 0.55])),
        },
        coords={"time": pd.date_range("2020-01-01", periods=3)},
    )

    file_path = tmp_path / "sample.h5"
    dataset.to_netcdf(file_path, engine="h5netcdf")

    result = generate_embeddings_for_file(
        dummy_generator,
        "hdf5",
        file_path,
        "source_hdf5",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.processed_file.exists()


def test_generate_embeddings_for_grib(monkeypatch, tmp_path, processed_dir, dummy_generator, test_logger):
    xr = pytest.importorskip("xarray")

    dataset = xr.Dataset(
        {
            "pressure": (("level",), np.array([1012.0, 1010.5, 1008.3])),
        },
        coords={"level": [1000, 900, 800]},
    )

    called = {"count": 0}

    def fake_open_dataset(path, engine=None):
        called["count"] += 1
        assert engine == "cfgrib"
        return dataset

    monkeypatch.setattr(format_processing.xr, "open_dataset", fake_open_dataset)

    file_path = tmp_path / "sample.grib"
    file_path.write_bytes(b"")

    result = generate_embeddings_for_file(
        dummy_generator,
        "grib",
        file_path,
        "source_grib",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert called["count"] == 1
    assert result.embeddings
    assert result.processed_file.exists()


def _write_geotiff(path: Path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    transform = from_origin(0, 2, 1, 1)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_generate_embeddings_for_geotiff(tmp_path, processed_dir, dummy_generator, test_logger):
    file_path = tmp_path / "sample.tif"
    _write_geotiff(file_path)

    result = generate_embeddings_for_file(
        dummy_generator,
        "geotiff",
        file_path,
        "source_tif",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.processed_file.exists()


def _write_ascii_grid(path: Path):
    contents = """ncols         2\nnrows         2\nxllcorner     0\nyllcorner     0\ncellsize      1\nNODATA_value -9999\n1 2\n3 4\n"""
    path.write_text(contents)


def test_generate_embeddings_for_ascii_grid(tmp_path, processed_dir, dummy_generator, test_logger):
    file_path = tmp_path / "grid.asc"
    _write_ascii_grid(file_path)

    result = generate_embeddings_for_file(
        dummy_generator,
        "ascii_grid",
        file_path,
        "source_asc",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.processed_file.exists()


@pytest.mark.parametrize(
    "fmt, filename, contents",
    [
        ("csv", "sample.csv", "year,value\n2020,1\n2021,2\n"),
        ("tsv", "sample.tsv", "year\tvalue\n2020\t3\n2021\t4\n"),
        ("txt", "sample.txt", "year value\n2020 5\n2021 6\n"),
    ],
)
def test_generate_embeddings_for_tabular_formats(tmp_path, processed_dir, dummy_generator, test_logger, fmt, filename, contents):
    file_path = tmp_path / filename
    file_path.write_text(contents)

    result = generate_embeddings_for_file(
        dummy_generator,
        fmt,
        file_path,
        f"source_{fmt}",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.artifacts["format"] == "tabular"


def test_generate_embeddings_for_json(tmp_path, processed_dir, dummy_generator, test_logger):
    file_path = tmp_path / "sample.json"
    payload = [
        {"year": 2020, "value": 1.23},
        {"year": 2021, "value": 1.45},
    ]
    file_path.write_text(json.dumps(payload))

    result = generate_embeddings_for_file(
        dummy_generator,
        "json",
        file_path,
        "source_json",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.artifacts["format"] == "tabular"


def _write_shapefile(folder: Path) -> Path:
    gpd = pytest.importorskip("geopandas")
    from shapely.geometry import Point

    folder.mkdir(parents=True, exist_ok=True)

    frame = gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )

    shp_path = folder / "sample.shp"
    frame.to_file(shp_path)
    return shp_path


def test_generate_embeddings_for_shapefile(tmp_path, processed_dir, dummy_generator, test_logger):
    shp_path = _write_shapefile(tmp_path)

    result = generate_embeddings_for_file(
        dummy_generator,
        "shapefile",
        shp_path,
        "source_shp",
        "20240101",
        processed_dir,
        test_logger,
    )

    assert result.embeddings
    assert result.processed_file.exists()


def test_prepare_file_for_processing_zip_shapefile(tmp_path, test_logger):
    shp_path = _write_shapefile(tmp_path / "shp_bundle")
    archive_path = tmp_path / "bundle.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        for file in shp_path.parent.glob("sample.*"):
            archive.write(file, arcname=file.relative_to(shp_path.parent))

    prepared, fmt = prepare_file_for_processing(archive_path, "zip", test_logger)

    assert fmt == "shapefile"
    assert prepared.suffix == ".shp"
    assert prepared.exists()


def test_prepare_file_for_processing_zip_ascii(tmp_path, test_logger):
    asc_folder = tmp_path / "ascii_archive"
    asc_folder.mkdir()
    asc_path = asc_folder / "grid.asc"
    _write_ascii_grid(asc_path)

    archive_path = tmp_path / "ascii.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(asc_path, arcname="grid.asc")

    prepared, fmt = prepare_file_for_processing(archive_path, "zip", test_logger)

    assert fmt == "ascii_grid"
    assert prepared.suffix == ".asc"
    assert prepared.exists()


def test_prepare_file_for_processing_gzip_geotiff(tmp_path, test_logger):
    tif_path = tmp_path / "sample.tif"
    _write_geotiff(tif_path)

    gz_path = tmp_path / "sample.tif.gz"
    with open(tif_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    prepared, fmt = prepare_file_for_processing(gz_path, "netcdf", test_logger)

    assert fmt == "geotiff"
    assert prepared.suffix == ".tif"
    assert prepared.exists()
