"""
Tests for the catalog module: Excel reader, phase classifier,
metadata pipeline, location enricher, and batch orchestrator.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# ---------------------------------------------------------------------------
# CatalogEntry + Excel reader
# ---------------------------------------------------------------------------

from src.catalog.excel_reader import CatalogEntry, read_catalog, _clean_value


class TestCatalogEntry:
    def test_source_id_generation(self):
        entry = CatalogEntry(row_index=0, dataset_name="ERA5")
        assert entry.source_id == "catalog_ERA5_0"

    def test_source_id_with_spaces(self):
        entry = CatalogEntry(row_index=5, dataset_name="ERA5 Land")
        assert entry.source_id == "catalog_ERA5_Land_5"

    def test_source_id_with_special_chars(self):
        entry = CatalogEntry(row_index=10, dataset_name="WorldClim - Future climate data")
        sid = entry.source_id
        assert "(" not in sid and ")" not in sid

    def test_to_dict(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="ERA5",
            hazard="Temperature",
            access="Open",
        )
        d = entry.to_dict()
        assert d["dataset_name"] == "ERA5"
        assert d["source_id"] == "catalog_ERA5_0"
        assert "row_index" in d

    def test_to_dict_excludes_none(self):
        entry = CatalogEntry(row_index=0, dataset_name="ERA5")
        d = entry.to_dict()
        assert "link" not in d
        assert "notes" not in d


class TestCleanValue:
    def test_none_for_nan(self):
        import math
        assert _clean_value(float("nan")) is None

    def test_strip_whitespace(self):
        assert _clean_value("  hello  ") == "hello"

    def test_empty_string_to_none(self):
        assert _clean_value("  ") is None

    def test_normal_value(self):
        assert _clean_value("ERA5") == "ERA5"


class TestReadCatalog:
    def test_reads_real_excel(self):
        """Test reading the actual D1.1.xlsx file."""
        excel_path = Path("Kopie souboru D1.1.xlsx")
        if not excel_path.exists():
            excel_path = Path(__file__).parent.parent / "Kopie souboru D1.1.xlsx"
        if not excel_path.exists():
            pytest.skip("Excel file not found")

        entries = read_catalog(str(excel_path))

        # Should have entries (the file has 234 rows)
        assert len(entries) > 200
        assert len(entries) <= 250

        # All entries should have dataset_name
        for entry in entries:
            assert entry.dataset_name is not None

        # Check that hazard forward-fill worked
        hazards = [e.hazard for e in entries if e.hazard]
        assert len(hazards) > 100  # Most rows should have hazard after ffill

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_catalog("/nonexistent/file.xlsx")


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------

from src.catalog.phase_classifier import classify_source, classify_all


class TestPhaseClassifier:
    def test_no_link_is_phase4(self):
        entry = CatalogEntry(row_index=0, dataset_name="test", link=None)
        assert classify_source(entry) == 4

    def test_contact_required_is_phase4(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="test",
            access="Get in contact (specify contact in notes)",
            link="https://example.com",
        )
        assert classify_source(entry) == 4

    def test_cds_portal_is_phase3(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="ERA5",
            access="Open (upon registration)",
            link="https://cds.climate.copernicus.eu/datasets/reanalysis-era5",
        )
        assert classify_source(entry) == 3

    def test_esgf_portal_is_phase3(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="CMIP6",
            access="Open",
            link="https://pcmdi.llnl.gov/CMIP6/",
        )
        assert classify_source(entry) == 3

    def test_direct_nc_open_is_phase1(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="test",
            access="Open",
            link="https://example.com/data/temperature.nc",
        )
        assert classify_source(entry) == 1

    def test_direct_csv_registration_is_phase2(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="test",
            access="Open (upon registration)",
            link="https://example.com/data.csv",
        )
        assert classify_source(entry) == 2

    def test_open_no_extension_is_phase1(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="test",
            access="Open",
            link="https://example.com/data-browser",
        )
        assert classify_source(entry) == 1

    def test_classify_all_includes_phase0(self):
        entries = [
            CatalogEntry(row_index=0, dataset_name="A", access="Open", link="https://x.com/a.nc"),
            CatalogEntry(row_index=1, dataset_name="B", access="Open", link=None),
        ]
        grouped = classify_all(entries)
        # Phase 0 should contain ALL entries
        assert len(grouped[0]) == 2
        # Phase 1 for direct download
        assert len(grouped[1]) == 1
        # Phase 4 for no-link
        assert len(grouped[4]) == 1


# ---------------------------------------------------------------------------
# Location enricher
# ---------------------------------------------------------------------------

from src.catalog.location_enricher import enrich_location, describe_bbox


class TestLocationEnricher:
    def test_region_country_takes_priority(self):
        entry = CatalogEntry(row_index=0, dataset_name="ERA5", region_country="Europe")
        assert enrich_location(entry) == "Europe"

    def test_spatial_coverage_global(self):
        entry = CatalogEntry(row_index=0, dataset_name="GISTEMP", spatial_coverage="Global")
        assert enrich_location(entry) == "Global"

    def test_infer_from_dataset_name(self):
        entry = CatalogEntry(row_index=0, dataset_name="EURO-CORDEX", spatial_coverage="Regional")
        assert "Europe" in enrich_location(entry)

    def test_unknown_fallback(self):
        entry = CatalogEntry(row_index=0, dataset_name="unknown_dataset")
        assert enrich_location(entry) == "Unknown"

    def test_describe_bbox_global(self):
        assert describe_bbox(-90, 90, -180, 180) == "Global"

    def test_describe_bbox_europe(self):
        result = describe_bbox(40, 60, -10, 30)
        assert result == "Europe"


# ---------------------------------------------------------------------------
# Metadata pipeline
# ---------------------------------------------------------------------------

from src.catalog.metadata_pipeline import _build_metadata_text, _build_payload


class TestMetadataPipeline:
    def test_build_text_basic(self):
        entry = CatalogEntry(
            row_index=0,
            dataset_name="ERA5",
            hazard="Mean surface temperature",
            data_type="Reanalysis data",
            spatial_coverage="Global",
            temporal_coverage="1940-Present",
            temporal_resolution="Hourly",
            region_country="Global",
        )
        text = _build_metadata_text(entry)
        assert "ERA5" in text
        assert "temperature" in text.lower()
        assert "Reanalysis" in text
        assert "Global" in text
        assert "1940" in text

    def test_build_payload(self):
        entry = CatalogEntry(
            row_index=5,
            dataset_name="CMIP6",
            hazard="Temperature",
            access="Open (upon registration)",
            link="https://example.com",
            impact_sector="Health, Energy",
            region_country="Global",
        )
        payload = _build_payload(entry)
        assert payload["is_metadata_only"] is True
        assert payload["catalog_source"] == "D1.1.xlsx"
        assert payload["dataset_name"] == "CMIP6"
        assert payload["hazard_type"] == "Temperature"
        assert payload["impact_sector"] == "Health, Energy"
        assert payload["location_name"] == "Global"

    def test_process_metadata_only(self):
        """Test process_metadata_only with mocked embedder and db."""
        from src.catalog.metadata_pipeline import process_metadata_only

        entry = CatalogEntry(
            row_index=0,
            dataset_name="ERA5",
            hazard="Temperature",
            region_country="Global",
        )

        # Mock embedder
        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [np.random.rand(1024).astype(np.float32)]

        # Mock db
        mock_db = MagicMock()

        result = process_metadata_only(entry, mock_embedder, mock_db)
        assert result is True
        mock_embedder.embed_documents.assert_called_once()
        mock_db.add_embeddings.assert_called_once()

    def test_process_metadata_batch(self):
        """Test batch processing with mocks."""
        from src.catalog.metadata_pipeline import process_metadata_batch

        entries = [
            CatalogEntry(row_index=i, dataset_name=f"Dataset_{i}", hazard="Temperature")
            for i in range(5)
        ]

        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [np.random.rand(1024).astype(np.float32) for _ in range(5)]

        mock_db = MagicMock()

        result = process_metadata_batch(entries, mock_embedder, mock_db, batch_size=3)
        assert result["processed"] == 5
        assert result["failed"] == 0
        assert result["total"] == 5
        assert len(result["succeeded_ids"]) == 5
        assert result["failed_entries"] == []


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------

from src.catalog.batch_orchestrator import BatchProgress


@pytest.fixture(autouse=False)
def _setup_test_db(monkeypatch):
    """Create an in-memory SQLite DB for BatchProgress tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.database.models import Base
    from src.database import connection as conn_mod

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    monkeypatch.setattr(conn_mod, "_engine", engine)
    monkeypatch.setattr(conn_mod, "_SessionLocal", factory)
    return engine


class TestBatchProgress:
    @pytest.fixture(autouse=True)
    def db(self, _setup_test_db):
        """Ensure each test gets a clean in-memory DB."""
        pass

    def test_mark_started_and_completed(self):
        progress = BatchProgress(total=10)
        progress.mark_started("src_1", "ERA5", 0)
        # After mark_started the row is "processing" in DB
        assert not progress.is_completed("src_1", phase=0)

        progress.mark_completed("src_1", phase=0)
        assert progress.is_completed("src_1", phase=0)

    def test_mark_failed(self):
        progress = BatchProgress(total=10)
        progress.mark_started("src_1", "ERA5", 0)
        progress.mark_failed("src_1", "Download timeout", phase=0)
        assert not progress.is_completed("src_1", phase=0)
        assert progress.get_overall_status("src_1", 0) == "failed"

    def test_is_completed(self):
        progress = BatchProgress()
        progress.mark_started("src_1", "ERA5", 0)
        progress.mark_completed("src_1", phase=0)
        assert progress.is_completed("src_1", phase=0) is True
        assert progress.is_completed("src_2", phase=0) is False

    def test_get_summary(self):
        progress = BatchProgress(total=10)
        # Simulate 5 completed phase-1 sources and 2 failed phase-1 sources
        for i in range(5):
            progress.mark_started(f"src_{i}", f"Dataset_{i}", 1)
            progress.mark_completed(f"src_{i}", phase=1)
        for i in range(5, 7):
            progress.mark_started(f"src_{i}", f"Dataset_{i}", 1)
            progress.mark_failed(f"src_{i}", "error", phase=1)
        progress.skipped = 1

        summary = progress.get_summary()
        assert summary["total"] == 10
        assert summary["processed"] == 5
        assert summary["failed"] == 2
        assert summary["pending"] == 2

    def test_mark_interrupted(self):
        progress = BatchProgress(total=5)
        progress.mark_started("src_1", "ERA5", 0)
        # src_1 is now "processing"
        count = progress.mark_interrupted()
        assert count == 1
        assert progress.get_overall_status("src_1", 0) == "pending"

    def test_get_overall_status_metadata_only(self):
        progress = BatchProgress(total=5)
        progress.mark_started("src_1", "ERA5", 0)
        progress.mark_completed("src_1", phase=0)
        # Target phase is 1 but only phase 0 is done
        assert progress.get_overall_status("src_1", 1) == "metadata_only"

    def test_get_completed_set(self):
        progress = BatchProgress(total=5)
        for i in range(3):
            progress.mark_started(f"src_{i}", f"DS_{i}", 0)
            progress.mark_completed(f"src_{i}", phase=0)
        # src_3 is started but not completed
        progress.mark_started("src_3", "DS_3", 0)

        completed = progress.get_completed_set(phase=0)
        assert completed == {"src_0", "src_1", "src_2"}
        assert "src_3" not in completed

    def test_get_completed_set_empty(self):
        progress = BatchProgress()
        assert progress.get_completed_set(phase=0) == set()

    def test_mark_started_bulk(self):
        progress = BatchProgress(total=5)
        entries = [
            CatalogEntry(row_index=i, dataset_name=f"DS_{i}")
            for i in range(3)
        ]
        progress.mark_started_bulk(entries, phase=0)
        # All should be in processing state (not completed)
        for entry in entries:
            assert not progress.is_completed(entry.source_id, phase=0)
            assert progress.get_overall_status(entry.source_id, 0) == "processing"

    def test_mark_started_bulk_idempotent(self):
        """Calling mark_started_bulk twice should not fail (upsert behavior)."""
        progress = BatchProgress(total=3)
        entries = [CatalogEntry(row_index=0, dataset_name="DS_0")]
        progress.mark_started_bulk(entries, phase=0)
        progress.mark_started_bulk(entries, phase=0)  # should not raise
        assert progress.get_overall_status(entries[0].source_id, 0) == "processing"

    def test_mark_completed_bulk(self):
        progress = BatchProgress(total=5)
        entries = [
            CatalogEntry(row_index=i, dataset_name=f"DS_{i}")
            for i in range(3)
        ]
        progress.mark_started_bulk(entries, phase=0)

        ids = [e.source_id for e in entries]
        progress.mark_completed_bulk(ids, phase=0)

        completed = progress.get_completed_set(phase=0)
        assert completed == set(ids)

    def test_mark_failed_bulk(self):
        progress = BatchProgress(total=5)
        entries = [
            CatalogEntry(row_index=i, dataset_name=f"DS_{i}")
            for i in range(2)
        ]
        progress.mark_started_bulk(entries, phase=0)

        pairs = [(entries[0].source_id, "timeout"), (entries[1].source_id, "404")]
        progress.mark_failed_bulk(pairs, phase=0)

        for entry in entries:
            assert not progress.is_completed(entry.source_id, phase=0)
            assert progress.get_overall_status(entry.source_id, 0) == "failed"

    def test_mark_completed_bulk_empty(self):
        """Passing empty list should be a no-op."""
        progress = BatchProgress()
        progress.mark_completed_bulk([], phase=0)  # should not raise

    def test_mark_failed_bulk_empty(self):
        progress = BatchProgress()
        progress.mark_failed_bulk([], phase=0)  # should not raise


# ---------------------------------------------------------------------------
# Schema extension
# ---------------------------------------------------------------------------
# Import schema module directly to avoid heavy __init__.py transitive imports
# (sentence-transformers, torch, etc. may not be installed locally)

import importlib.util
import sys

def _import_schema_directly():
    """Load schema.py without triggering climate_embeddings.__init__."""
    spec = importlib.util.spec_from_file_location(
        "src.climate_embeddings.schema",
        str(Path(__file__).parent.parent / "src" / "climate_embeddings" / "schema.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

_schema_mod = _import_schema_directly()
ClimateChunkMetadata = _schema_mod.ClimateChunkMetadata
generate_human_readable_text = _schema_mod.generate_human_readable_text


class TestSchemaExtension:
    def test_new_fields_default_none(self):
        meta = ClimateChunkMetadata(
            dataset_name="ERA5",
            source_id="test",
            variable="2m_temperature",
        )
        assert meta.hazard_type is None
        assert meta.data_type is None
        assert meta.catalog_source is None
        assert meta.location_name is None

    def test_new_fields_set(self):
        meta = ClimateChunkMetadata(
            dataset_name="ERA5",
            source_id="test",
            variable="2m_temperature",
            hazard_type="Temperature",
            data_type="Reanalysis",
            catalog_source="D1.1.xlsx",
            location_name="Europe",
        )
        assert meta.hazard_type == "Temperature"
        assert meta.catalog_source == "D1.1.xlsx"

    def test_to_dict_includes_catalog_fields(self):
        meta = ClimateChunkMetadata(
            dataset_name="ERA5",
            source_id="test",
            variable="temperature",
            hazard_type="Temperature",
            impact_sector="Health, Energy",
        )
        d = meta.to_dict()
        assert d["hazard_type"] == "Temperature"
        assert d["impact_sector"] == "Health, Energy"

    def test_generate_text_includes_catalog_fields(self):
        meta_dict = {
            "variable": "2m_temperature",
            "dataset_name": "ERA5",
            "hazard_type": "Mean surface temperature",
            "data_type": "Reanalysis data",
            "region_country": "Europe",
            "impact_sector": "Health, Energy",
        }
        text = generate_human_readable_text(meta_dict)
        assert "Hazard: Mean surface temperature" in text
        assert "Data type: Reanalysis" in text
        assert "Region: Europe" in text
        assert "Impact sectors: Health, Energy" in text

    def test_backward_compatibility(self):
        """Old metadata without catalog fields still works."""
        meta_dict = {
            "variable": "TMAX",
            "dataset_name": "NOAA_GSOM",
            "long_name": "Maximum Temperature",
            "unit": "°C",
            "stats_mean": 25.5,
            "stats_min": 10.0,
            "stats_max": 40.0,
        }
        text = generate_human_readable_text(meta_dict)
        assert "TMAX" in text
        assert "NOAA_GSOM" in text
        assert "25.50" in text
