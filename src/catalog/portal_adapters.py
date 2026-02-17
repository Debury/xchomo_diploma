"""
Portal adapters for Phase 3: API-based data portals.

Each adapter downloads a small sample (1 month, 1 variable, small bbox)
from the respective portal and processes it through the embedding pipeline.
"""

import logging
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from src.catalog.excel_reader import CatalogEntry

logger = logging.getLogger(__name__)


class PortalAdapter(ABC):
    """Base class for portal adapters."""

    @abstractmethod
    def download_and_process(self, entry: CatalogEntry) -> bool:
        """Download a sample and process it. Returns True on success."""
        ...


class CDSAdapter(PortalAdapter):
    """
    Adapter for Copernicus Climate Data Store (ERA5, CERRA, etc.).
    Uses the cdsapi Python package.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        try:
            import cdsapi

            client = cdsapi.Client()

            # Determine CDS dataset name from link
            dataset = self._extract_dataset_id(entry.link or "")
            if not dataset:
                logger.warning(f"Cannot determine CDS dataset ID from {entry.link}")
                return False

            # Download a small sample: 1 month, single variable, small bbox
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name

            request = {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": "2023",
                "month": "01",
                "day": ["01"],
                "time": ["12:00"],
                "area": [50, 10, 45, 20],  # Small European bbox
                "format": "netcdf",
            }

            logger.info(f"CDS: downloading sample from {dataset}")
            client.retrieve(dataset, request, tmp_path)

            # Process through standard pipeline
            ok = self._process_file(tmp_path, entry)

            Path(tmp_path).unlink(missing_ok=True)
            return ok

        except ImportError:
            logger.error("cdsapi not installed — cannot use CDS adapter")
            return False
        except Exception as e:
            logger.error(f"CDS adapter failed for {entry.dataset_name}: {e}")
            return False

    def _extract_dataset_id(self, link: str) -> Optional[str]:
        """Extract CDS dataset ID from URL."""
        # e.g. https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=overview
        parsed = urlparse(link)
        parts = parsed.path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part == "datasets" and i + 1 < len(parts):
                return parts[i + 1].split("?")[0]
        return None

    def _process_file(self, file_path: str, entry: CatalogEntry) -> bool:
        """Process a downloaded file through the embedding pipeline."""
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto, raster_to_embeddings
        from src.climate_embeddings.schema import ClimateChunkMetadata, generate_human_readable_text
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader

        config = ConfigLoader("config/pipeline_config.yaml").load()
        embedder = TextEmbedder()
        db = VectorDatabase(config=config)

        result = load_raster_auto(file_path)
        embeddings_data = raster_to_embeddings(result)

        for j, emb_data in enumerate(embeddings_data):
            meta = ClimateChunkMetadata.from_chunk_metadata(
                raw_metadata=emb_data["metadata"],
                stats_vector=emb_data["vector"],
                source_id=entry.source_id,
                dataset_name=entry.dataset_name,
            )
            meta_dict = meta.to_dict()
            meta_dict["catalog_source"] = "D1.1.xlsx"

            text = generate_human_readable_text(meta_dict)
            text_embedding = embedder.embed_documents([text])[0]

            point_id = f"{entry.source_id}_portal_chunk_{j}"
            db.add_embeddings(
                ids=[point_id],
                embeddings=[text_embedding.tolist()],
                metadatas=[meta_dict],
            )

        logger.info(f"CDS adapter: stored {len(embeddings_data)} chunks for {entry.dataset_name}")
        return True


class ESGFAdapter(PortalAdapter):
    """Adapter for ESGF (CMIP6, CORDEX, etc.) — downloads sample via OpenDAP/HTTP."""

    def download_and_process(self, entry: CatalogEntry) -> bool:
        # ESGF requires complex authentication; for now, just log a placeholder
        logger.warning(f"ESGF adapter not yet fully implemented for {entry.dataset_name}")
        return False


class NOAAAdapter(PortalAdapter):
    """Adapter for NOAA data portals."""

    def download_and_process(self, entry: CatalogEntry) -> bool:
        logger.warning(f"NOAA adapter not yet fully implemented for {entry.dataset_name}")
        return False


# Registry of adapters by portal name
_ADAPTERS = {
    "CDS": CDSAdapter,
    "ESGF": ESGFAdapter,
    "NOAA": NOAAAdapter,
}


def get_adapter(entry: CatalogEntry) -> Optional[PortalAdapter]:
    """Get the appropriate portal adapter for a catalog entry."""
    from src.catalog.phase_classifier import _get_portal

    portal = _get_portal(entry.link or "")
    if portal and portal in _ADAPTERS:
        return _ADAPTERS[portal]()
    return None
