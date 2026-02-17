"""
Portal adapters for Phase 3: API-based data portals.

Each adapter downloads a small sample (1 month, 1 variable, small bbox)
from the respective portal and processes it through the embedding pipeline.
"""

import logging
import os
import tempfile
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

from src.catalog.excel_reader import CatalogEntry

logger = logging.getLogger(__name__)


def _process_file(file_path: str, entry: CatalogEntry, adapter_name: str = "portal") -> bool:
    """Shared: process a downloaded file through the embedding pipeline."""
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

    if not embeddings_data:
        logger.warning(f"{adapter_name}: loader produced 0 chunks for {entry.dataset_name}")
        return False

    for j, emb_data in enumerate(embeddings_data):
        meta = ClimateChunkMetadata.from_chunk_metadata(
            raw_metadata=emb_data["metadata"],
            stats_vector=emb_data["vector"],
            source_id=entry.source_id,
            dataset_name=entry.dataset_name,
        )
        meta_dict = meta.to_dict()
        meta_dict["catalog_source"] = "D1.1.xlsx"

        if entry.hazard:
            meta_dict["hazard_type"] = entry.hazard
        if entry.impact_sector:
            meta_dict["impact_sector"] = entry.impact_sector
        if entry.region_country:
            meta_dict["location_name"] = entry.region_country

        text = generate_human_readable_text(meta_dict)
        text_embedding = embedder.embed_documents([text])[0]

        point_id = f"{entry.source_id}_portal_chunk_{j}"
        db.add_embeddings(
            ids=[point_id],
            embeddings=[text_embedding.tolist()],
            metadatas=[meta_dict],
        )

    logger.info(f"{adapter_name}: stored {len(embeddings_data)} chunks for {entry.dataset_name}")
    return True


class PortalAdapter(ABC):
    """Base class for portal adapters."""

    @abstractmethod
    def download_and_process(self, entry: CatalogEntry) -> bool:
        """Download a sample and process it. Returns True on success."""
        ...


# ---------------------------------------------------------------------------
# CDS Adapter — Copernicus Climate Data Store
# ---------------------------------------------------------------------------

# Dataset-specific configurations for CDS API requests.
# Each maps a CDS dataset_id to a minimal sample request.
_CDS_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "reanalysis-era5-single-levels": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2023"],
        "month": ["01"],
        "day": ["15"],
        "time": ["12:00"],
        "area": [50, 10, 45, 20],
        "data_format": "netcdf",
    },
    "reanalysis-era5-land": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2023"],
        "month": ["01"],
        "day": ["15"],
        "time": ["12:00"],
        "area": [50, 10, 45, 20],
        "data_format": "netcdf",
    },
    "reanalysis-cerra-single-levels": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "level_type": ["surface_or_atmosphere"],
        "year": ["2020"],
        "month": ["01"],
        "day": ["15"],
        "time": ["12:00"],
        "data_format": "netcdf",
    },
    "cams-global-reanalysis-eac4": {
        "variable": ["total_column_ozone"],
        "year": ["2020"],
        "month": ["01"],
        "day": ["15"],
        "time": ["12:00"],
        "area": [50, 10, 45, 20],
        "data_format": "netcdf",
    },
    "satellite-fire-radiative-power": {
        "product_type": ["gridded"],
        "sensor": ["modis"],
        "horizontal_aggregation": ["0_25_degree_x_0_25_degree"],
        "year": ["2020"],
        "month": ["01"],
        "day": ["15"],
        "version": ["1_0"],
        "variable": ["fire_radiative_power"],
    },
    "derived-utci-historical": {
        "variable": ["universal_thermal_climate_index"],
        "product_type": ["consolidated_dataset"],
        "year": ["2020"],
        "month": ["07"],
        "day": ["15"],
        "area": [50, 10, 45, 20],
        "version": ["1_1"],
    },
    "sis-european-risk-flood-indicators": {
        "variable": ["flood_recurrence"],
        "product_type": ["summary"],
        "return_period": ["10-years"],
    },
}

# Map dataset names (from Excel) to CDS dataset IDs.
_DATASET_NAME_TO_CDS_ID: Dict[str, str] = {
    "ERA5": "reanalysis-era5-single-levels",
    "ERA5 Land": "reanalysis-era5-land",
    "CERRA": "reanalysis-cerra-single-levels",
    "CAMS": "cams-global-reanalysis-eac4",
    "Fire radiative power (Copernicus)": "satellite-fire-radiative-power",
    "ERA5-HEAT": "derived-utci-historical",
}


class CDSAdapter(PortalAdapter):
    """
    Adapter for Copernicus Climate Data Store (ERA5, CERRA, etc.).
    Uses the cdsapi Python package (v2 API — cds.climate.copernicus.eu).

    Requires CDS_API_KEY environment variable or ~/.cdsapirc file.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            import cdsapi

            # Build client — prefer env vars, fall back to .cdsapirc
            api_url = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")
            api_key = os.getenv("CDS_API_KEY", "")
            if api_key:
                client = cdsapi.Client(url=api_url, key=api_key)
            else:
                client = cdsapi.Client()

            # Determine CDS dataset ID
            dataset_id = self._resolve_dataset_id(entry)
            if not dataset_id:
                logger.warning(f"CDS: cannot determine dataset ID for {entry.dataset_name} ({entry.link})")
                return False

            # Get dataset-specific request params, or use a sensible default
            request = _CDS_DATASET_CONFIGS.get(dataset_id, self._default_request())

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name

            logger.info(f"CDS: downloading {dataset_id} for {entry.dataset_name}")
            client.retrieve(dataset_id, request, tmp_path)

            ok = _process_file(tmp_path, entry, adapter_name="CDS")
            return ok

        except ImportError:
            logger.error("cdsapi not installed — cannot use CDS adapter")
            return False
        except Exception as e:
            logger.error(f"CDS adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _resolve_dataset_id(self, entry: CatalogEntry) -> Optional[str]:
        """Resolve CDS dataset ID from dataset name or URL."""
        # Try name mapping first
        name = entry.dataset_name or ""
        if name in _DATASET_NAME_TO_CDS_ID:
            return _DATASET_NAME_TO_CDS_ID[name]

        # Try extracting from URL
        return self._extract_dataset_id(entry.link or "")

    def _extract_dataset_id(self, link: str) -> Optional[str]:
        """Extract CDS dataset ID from URL."""
        # https://cds.climate.copernicus.eu/datasets/reanalysis-cerra-single-levels?tab=overview
        parsed = urlparse(link)
        parts = parsed.path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part == "datasets" and i + 1 < len(parts):
                return parts[i + 1].split("?")[0]

        # ADS (Atmosphere Data Store) URL pattern
        # https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-reanalysis-eac4
        if "cdsapp" in parsed.path:
            for part in parts:
                if part.startswith("dataset"):
                    # Could be "dataset/cams-global-reanalysis-eac4"
                    pass
            # Try fragment
            fragment = parsed.fragment
            if "dataset/" in fragment:
                return fragment.split("dataset/")[-1].split("?")[0]

        return None

    def _default_request(self) -> Dict[str, Any]:
        """Default CDS request for unknown datasets — ERA5-like."""
        return {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2023"],
            "month": ["01"],
            "day": ["15"],
            "time": ["12:00"],
            "area": [50, 10, 45, 20],
            "data_format": "netcdf",
        }


# ---------------------------------------------------------------------------
# ESGF Adapter — CMIP6, CORDEX via OpenDAP
# ---------------------------------------------------------------------------

# Known OpenDAP sample endpoints for common ESGF datasets.
# These are small files accessible without authentication.
_ESGF_OPENDAP_SAMPLES: Dict[str, str] = {
    "CMIP6": (
        # CEDA DAP — verified anonymous HTTP access (no auth)
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
        "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/latest/"
        "tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
    ),
    "EURO-CORDEX": (
        "https://esgf-data.dkrz.de/thredds/dodsC/cordex/output/"
        "EUR-11/MPI-CSC/MPI-M-MPI-ESM-LR/historical/r1i1p1/"
        "MPI-CSC-REMO2009/v1/mon/tas/"
        "tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_MPI-CSC-REMO2009_v1_mon_200601-200612.nc"
    ),
    "MED-CORDEX": (
        "https://esgf-data.dkrz.de/thredds/dodsC/cordex/output/"
        "EUR-11/MPI-CSC/MPI-M-MPI-ESM-LR/historical/r1i1p1/"
        "MPI-CSC-REMO2009/v1/mon/tas/"
        "tas_EUR-11_MPI-M-MPI-ESM-LR_historical_r1i1p1_MPI-CSC-REMO2009_v1_mon_200601-200612.nc"
    ),
    "CMIP6-BCCAQ": (
        # CEDA DAP — same CMIP6 endpoint, BCCAQ-specific data requires CEDA auth
        "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
        "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/latest/"
        "tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
    ),
}


class ESGFAdapter(PortalAdapter):
    """
    Adapter for ESGF (CMIP6, CORDEX, etc.).
    Downloads sample data via OpenDAP (no authentication required for public nodes).
    Falls back to HTTP download of a small NetCDF file.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            opendap_url = self._resolve_opendap_url(entry)
            if not opendap_url:
                logger.warning(f"ESGF: no OpenDAP URL for {entry.dataset_name}")
                return False

            # Try OpenDAP via xarray first (streams only needed slices)
            tmp_path = self._download_via_opendap(opendap_url, entry)
            if tmp_path is None:
                # Fall back to direct HTTP download of small file
                tmp_path = self._download_via_http(opendap_url, entry)

            if tmp_path is None:
                logger.warning(f"ESGF: could not download sample for {entry.dataset_name}")
                return False

            ok = _process_file(tmp_path, entry, adapter_name="ESGF")
            return ok

        except Exception as e:
            logger.error(f"ESGF adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _resolve_opendap_url(self, entry: CatalogEntry) -> Optional[str]:
        """Find an OpenDAP URL for this dataset."""
        name = entry.dataset_name or ""
        # Check known samples
        for key, url in _ESGF_OPENDAP_SAMPLES.items():
            if key in name or name in key:
                return url

        # Try to convert ESGF catalog URL to OpenDAP
        link = entry.link or ""
        if "thredds/dodsC" in link:
            return link  # Already an OpenDAP URL
        if "thredds/catalog" in link:
            return link.replace("thredds/catalog", "thredds/dodsC").replace("catalog.html", "")

        return None

    def _download_via_opendap(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Download a small spatial/temporal slice via OpenDAP."""
        try:
            import xarray as xr

            logger.info(f"ESGF: opening OpenDAP {url} for {entry.dataset_name}")
            ds = xr.open_dataset(url, engine="netcdf4")

            # Select only first few time steps to keep download small
            time_dim = None
            for dim in ["time", "Time", "T"]:
                if dim in ds.dims:
                    time_dim = dim
                    break

            if time_dim and ds.sizes[time_dim] > 12:
                ds = ds.isel({time_dim: slice(0, 12)})

            # Write to temp NetCDF
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name
            ds.to_netcdf(tmp_path)
            ds.close()

            logger.info(f"ESGF: downloaded OpenDAP slice to {tmp_path}")
            return tmp_path

        except Exception as e:
            logger.warning(f"ESGF OpenDAP failed for {entry.dataset_name}: {e}")
            return None

    def _download_via_http(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Fall back to direct HTTP download (replace dodsC with fileServer)."""
        try:
            import requests as http_requests

            # Convert OpenDAP URL to HTTP file server URL
            http_url = url.replace("/thredds/dodsC/", "/thredds/fileServer/")

            logger.info(f"ESGF: HTTP download from {http_url}")
            resp = http_requests.get(http_url, timeout=300, stream=True,
                                     headers={"User-Agent": "ClimateRAG/1.0"})
            resp.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                max_bytes = 500 * 1024 * 1024  # 500 MB limit
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"ESGF file too large ({written / 1e6:.0f} MB)")
                    tmp.write(chunk)
                return tmp.name

        except Exception as e:
            logger.warning(f"ESGF HTTP download failed for {entry.dataset_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# NOAA Adapter — PSL, NCEI datasets
# ---------------------------------------------------------------------------

# Direct download URLs for NOAA PSL datasets (NetCDF files).
_NOAA_DIRECT_URLS: Dict[str, str] = {
    "CPC": "https://downloads.psl.noaa.gov/Datasets/cpc_global_precip/precip.day.2020.nc",
    "GPCC": "https://downloads.psl.noaa.gov/Datasets/gpcc/full_v2022/precip.mon.total.1x1.v2022.nc",
    "GPCP": "https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc",
    "NOAA-NCEP/NCAR Reanalysis 1": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/air.sig995.2020.nc",
    "NCEP-NCAR2": "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis2/gaussian_grid/air.2m.gauss.2020.nc",
}


class NOAAAdapter(PortalAdapter):
    """
    Adapter for NOAA data portals (PSL, NCEI).
    Uses direct HTTPS downloads from downloads.psl.noaa.gov.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            import requests as http_requests

            url = self._resolve_url(entry)
            if not url:
                logger.warning(f"NOAA: no direct URL for {entry.dataset_name}")
                return False

            logger.info(f"NOAA: downloading {entry.dataset_name} from {url}")
            resp = http_requests.get(url, timeout=300, stream=True,
                                     headers={"User-Agent": "ClimateRAG/1.0"})
            resp.raise_for_status()

            # Check we got actual data, not HTML
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning(f"NOAA: got HTML instead of data for {entry.dataset_name}")
                return False

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                max_bytes = 500 * 1024 * 1024
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"NOAA file too large ({written / 1e6:.0f} MB)")
                    tmp.write(chunk)
                tmp_path = tmp.name

            ok = _process_file(tmp_path, entry, adapter_name="NOAA")
            return ok

        except Exception as e:
            logger.error(f"NOAA adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _resolve_url(self, entry: CatalogEntry) -> Optional[str]:
        """Find direct download URL for NOAA dataset."""
        name = entry.dataset_name or ""
        if name in _NOAA_DIRECT_URLS:
            return _NOAA_DIRECT_URLS[name]

        # Check if the link already points to a downloadable file
        link = entry.link or ""
        if link.endswith((".nc", ".csv", ".gz")):
            return link

        return None


# ---------------------------------------------------------------------------
# NASA Adapter — Earthdata, GESDISC, PO.DAAC
# ---------------------------------------------------------------------------

# Direct download URLs for NASA datasets that allow anonymous or token-based HTTPS.
_NASA_DIRECT_URLS: Dict[str, str] = {
    "IMERG": (
        "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGM.07/"
        "2020/3B-MO.MS.MRG.3IMERG.20200101-S000000-E235959.01.V07B.HDF5.nc4"
    ),
    "MERRA-2": (
        "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/"
        "2020/01/MERRA2_400.tavg1_2d_slv_Nx.20200115.nc4"
    ),
    "MERRA2": (
        "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/"
        "2020/01/MERRA2_400.tavg1_2d_slv_Nx.20200115.nc4"
    ),
    "CERES-EBAF": (
        "https://asdc.larc.nasa.gov/data/CERES/EBAF/Edition4.2/"
        "CERES_EBAF_Edition4.2_200003-202407.nc"
    ),
    "JPL GRACE": (
        "https://podaac-tools.jpl.nasa.gov/drive/files/allData/tellus/L3/mascon/RL06.1/JPL/"
        "CRI/netcdf/GRCTellus.JPL.200204_202312.GLO.RL06.1M.MSCNv03CRI.nc"
    ),
}


class NASAAdapter(PortalAdapter):
    """
    Adapter for NASA Earthdata datasets.
    Many NASA datasets require an Earthdata login token via bearer auth.
    Set NASA_EARTHDATA_TOKEN env var for authenticated access.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            import requests as http_requests

            url = self._resolve_url(entry)
            if not url:
                logger.warning(f"NASA: no direct URL for {entry.dataset_name}")
                return False

            token = os.getenv("NASA_EARTHDATA_TOKEN", "")
            if not token:
                logger.warning("NASA: NASA_EARTHDATA_TOKEN not set — most datasets require auth")
                return False

            # NASA Earthdata uses OAuth redirects — need a session with cookies
            session = http_requests.Session()
            session.headers.update({"User-Agent": "ClimateRAG/1.0"})
            session.headers.update({"Authorization": f"Bearer {token}"})

            logger.info(f"NASA: downloading {entry.dataset_name} from {url}")
            resp = session.get(url, timeout=300, stream=True, allow_redirects=True)
            resp.raise_for_status()

            # Check we got data, not a login page
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("Got HTML response — token may be expired or invalid")

            ext = ".nc4" if ".nc4" in url else ".nc"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                max_bytes = 500 * 1024 * 1024
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"NASA file too large ({written / 1e6:.0f} MB)")
                    tmp.write(chunk)
                tmp_path = tmp.name

            ok = _process_file(tmp_path, entry, adapter_name="NASA")
            return ok

        except Exception as e:
            logger.error(f"NASA adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _resolve_url(self, entry: CatalogEntry) -> Optional[str]:
        name = entry.dataset_name or ""
        for key, url in _NASA_DIRECT_URLS.items():
            if key in name or name in key:
                return url
        return None


# ---------------------------------------------------------------------------
# Marine Copernicus Adapter
# ---------------------------------------------------------------------------

class MarineCopernicusAdapter(PortalAdapter):
    """
    Adapter for Copernicus Marine Service (CMEMS) datasets.
    Uses the `copernicusmarine` Python toolbox for authenticated access.
    Requires CMEMS_USERNAME and CMEMS_PASSWORD env vars.
    """

    # Map product IDs to their dataset IDs for the copernicusmarine API
    _PRODUCT_DATASETS: Dict[str, str] = {
        "SST_MED_SST_L4_REP_OBSERVATIONS_010_021": "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021",
        "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004": "cmems_SST_MED_SST_L4_NRT_OBSERVATIONS_010_004",
    }

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            product_id = self._extract_product_id(entry.link or "")
            if not product_id:
                logger.warning(f"Marine: cannot extract product ID from {entry.link}")
                return False

            # Try copernicusmarine toolbox first
            tmp_path = self._download_via_toolbox(product_id, entry)

            if tmp_path is None:
                logger.warning(f"Marine: could not download {entry.dataset_name}")
                return False

            ok = _process_file(tmp_path, entry, adapter_name="Marine")
            return ok

        except Exception as e:
            logger.error(f"Marine adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _extract_product_id(self, link: str) -> Optional[str]:
        """Extract CMEMS product ID from URL."""
        parsed = urlparse(link)
        parts = parsed.path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part == "product" and i + 1 < len(parts):
                return parts[i + 1].split("?")[0]
        return None

    def _download_via_toolbox(self, product_id: str, entry: CatalogEntry) -> Optional[str]:
        """Download a small sample using the copernicusmarine Python toolbox."""
        try:
            import copernicusmarine

            username = os.getenv("CMEMS_USERNAME", "")
            password = os.getenv("CMEMS_PASSWORD", "")
            if not username or not password:
                logger.warning("Marine: CMEMS_USERNAME/CMEMS_PASSWORD not set")
                return None

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name

            # Download a small subset (1 month, Mediterranean bbox)
            logger.info(f"Marine: downloading {product_id} via copernicusmarine toolbox")
            copernicusmarine.subset(
                dataset_id=product_id,
                variables=["analysed_sst"],
                minimum_longitude=0,
                maximum_longitude=20,
                minimum_latitude=35,
                maximum_latitude=45,
                start_datetime="2023-01-01",
                end_datetime="2023-01-31",
                output_filename=Path(tmp_path).name,
                output_directory=str(Path(tmp_path).parent),
                username=username,
                password=password,
                force_download=True,
            )

            if Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0:
                return tmp_path
            return None

        except ImportError:
            logger.warning("copernicusmarine not installed — pip install copernicusmarine")
            return None
        except Exception as e:
            logger.warning(f"Marine toolbox failed for {entry.dataset_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS: Dict[str, type] = {
    "CDS": CDSAdapter,
    "ESGF": ESGFAdapter,
    "NOAA": NOAAAdapter,
    "NASA": NASAAdapter,
    "MARINE": MarineCopernicusAdapter,
}


def get_adapter(entry: CatalogEntry) -> Optional[PortalAdapter]:
    """Get the appropriate portal adapter for a catalog entry."""
    from src.catalog.phase_classifier import _get_portal

    link = entry.link or ""
    portal = _get_portal(link)

    # Special case: marine Copernicus
    if "marine.copernicus.eu" in link or "cmems" in link.lower():
        return MarineCopernicusAdapter()

    if portal and portal in _ADAPTERS:
        return _ADAPTERS[portal]()
    return None
