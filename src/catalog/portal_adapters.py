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

from src.utils.logger import setup_logger

logger = setup_logger("catalog_pipeline", "logs/catalog_pipeline.log", "INFO")


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

    all_ids = []
    all_embeddings = []
    all_metadatas = []

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

        all_ids.append(f"{entry.source_id}_portal_chunk_{j}")
        all_embeddings.append(text_embedding.tolist())
        all_metadatas.append(meta_dict)

    # Batched upsert (database.py handles chunking + retry)
    db.add_embeddings(
        ids=all_ids,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    # Upsert dataset summary chunk
    try:
        from src.climate_embeddings.schema import build_dataset_summary
        entry_meta = {}
        if entry.hazard:
            entry_meta["hazard_type"] = entry.hazard
        if entry.impact_sector:
            entry_meta["impact_sector"] = entry.impact_sector
        if entry.region_country:
            entry_meta["location_name"] = entry.region_country
        entry_meta["catalog_source"] = "D1.1.xlsx"

        summary_meta = build_dataset_summary(all_metadatas, entry_meta)
        if summary_meta:
            summary_text = generate_human_readable_text(summary_meta)
            summary_emb = embedder.embed_documents([summary_text])[0]
            db.add_embeddings(
                ids=[f"{entry.source_id}_summary"],
                embeddings=[summary_emb.tolist()],
                metadatas=[summary_meta],
            )
    except Exception as sum_err:
        logger.warning(f"{adapter_name}: summary chunk failed for {entry.dataset_name}: {sum_err}")

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

# Map hazard types (from Excel) to CDS variable names per dataset.
# When a dataset appears for multiple hazards, we download the relevant variable.
_HAZARD_TO_CDS_VARIABLE: Dict[str, Dict[str, str]] = {
    "reanalysis-era5-single-levels": {
        "Mean surface temperature": "2m_temperature",
        "Extreme heat": "maximum_2m_temperature_since_previous_post_processing",
        "Cold spell": "minimum_2m_temperature_since_previous_post_processing",
        "Frost": "minimum_2m_temperature_since_previous_post_processing",
        "Mean precipitation": "total_precipitation",
        "Heavy precipitation and pluvial floods": "total_precipitation",
        "Mean wind speed": "10m_u_component_of_wind",
        "Severe wind storm": "10m_wind_gust_since_previous_post_processing",
        "Tropical cyclone": "mean_sea_level_pressure",
        "Permafrost": "soil_temperature_level_1",
        "Relative sea level": "mean_sea_level_pressure",
        "Radiation at surface": "surface_solar_radiation_downwards",
    },
    "reanalysis-era5-land": {
        "Mean surface temperature": "2m_temperature",
        "Extreme heat": "2m_temperature",
        "Mean precipitation": "total_precipitation",
        "River flood": "total_precipitation",
        "Heavy precipitation and pluvial floods": "total_precipitation",
    },
    "reanalysis-cerra-single-levels": {
        "Mean surface temperature": "2m_temperature",
        "Extreme heat": "2m_temperature",
        "Cold spell": "2m_temperature",
        "Frost": "2m_temperature",
        "Mean precipitation": "total_precipitation",
        "Heavy precipitation and pluvial floods": "total_precipitation",
        "Mean wind speed": "10m_wind_speed",
        "Severe wind storm": "10m_wind_speed",
        "Radiation at surface": "2m_temperature",  # radiation vars need forecast mode which CDS rejects; use temp as proxy
    },
    "cams-global-reanalysis-eac4": {
        "Mean wind speed": "10m_u_component_of_wind",
        "Severe wind storm": "10m_v_component_of_wind",
        "Tropical cyclone": "mean_sea_level_pressure",
        "Sand and dust storm": "dust_aerosol_optical_depth_550nm",
        "Air pollution weather": "particulate_matter_10um",
        "Atmospheric CO₂ at surface": "total_aerosol_optical_depth_550nm",  # EAC4 has no CO2; use AOD as representative atmospheric var
        "Radiation at surface": "total_aerosol_optical_depth_550nm",  # EAC4 has no radiation; use AOD as representative var
    },
}

# Dataset-specific configurations for CDS API requests.
# Each maps a CDS dataset_id to a sample request covering 4 seasonal months
# (Jan/Apr/Jul/Oct) to capture seasonal variation — critical for meaningful RAG.
_CDS_DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "reanalysis-era5-single-levels": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2023"],
        "month": ["01", "04", "07", "10"],
        "day": ["15"],
        "time": ["12:00"],
        "data_format": "netcdf",
    },
    "reanalysis-era5-land": {
        "product_type": ["reanalysis"],
        "variable": ["2m_temperature"],
        "year": ["2023"],
        "month": ["01", "04", "07", "10"],
        "day": ["15"],
        "time": ["12:00"],
        "data_format": "netcdf",
    },
    "reanalysis-cerra-single-levels": {
        "product_type": ["analysis"],
        "data_type": ["reanalysis"],
        "level_type": ["surface_or_atmosphere"],
        "variable": ["2m_temperature"],
        "year": ["2020", "2023"],
        "month": ["01", "07"],
        "day": ["15"],
        "time": ["12:00"],
        "data_format": "grib",
    },
    "satellite-fire-radiative-power": {
        "product_type": "gridded",
        "satellite": "sentinel_3a",
        "horizontal_aggregation": "0_25_degree_x_0_25_degree",
        "time_aggregation": "month",
        "observation_time": "night",
        "year": "2023",
        "month": "07",
        "day": "15",
        "version": "1_2",
        "variable": "all",
    },
    "derived-utci-historical": {
        "variable": "universal_thermal_climate_index",
        "product_type": "consolidated_dataset",
        "year": "2020",
        "month": ["01", "07"],
        "day": "15",
        "version": "1_1",
    },
    "sis-european-risk-flood-indicators": {
        "variable": ["flood_recurrence"],
        "product_type": ["summary"],
        "return_period": ["10-years"],
    },
    # CAMS EAC4 on ADS — uses date/time params (not year/month/day)
    # Download 4 seasonal days to capture dust/aerosol variation
    "cams-global-reanalysis-eac4": {
        "variable": ["total_column_ozone"],
        "date": ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"],
        "time": ["12:00"],
        "data_format": "netcdf",
    },
    # GFAS fire emissions on ADS — multiple dates to capture fire season
    "cams-global-fire-emissions-gfas": {
        "variable": ["wildfire_radiative_power"],
        "date": ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"],
        "data_format": "grib",
    },
}

# Map dataset names (from Excel) to CDS dataset IDs.
_DATASET_NAME_TO_CDS_ID: Dict[str, str] = {
    "ERA5": "reanalysis-era5-single-levels",
    "ERA5 Land": "reanalysis-era5-land",
    "CERRA": "reanalysis-cerra-single-levels",
    "Fire radiative power (Copernicus)": "cams-global-fire-emissions-gfas",
    "ERA5-HEAT": "derived-utci-historical",
    # CAMS datasets — hosted on ADS (ads.atmosphere.copernicus.eu), not CDS
    "CAMS": "cams-global-reanalysis-eac4",
}

# CDS datasets that are known to be removed/dead — skip them entirely
_CDS_SKIP_DATASETS: set = set()

# CERRA variables that are only available as forecasts (not analysis).
# When one of these is requested, product_type must be switched to "forecast"
# and a leadtime_hour must be provided.
_CERRA_FORECAST_ONLY_VARIABLES: set = {
    "total_precipitation",
    "10m_wind_gust_since_previous_post_processing",
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "surface_net_short_wave_radiation_flux",
    "surface_net_long_wave_radiation_flux",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
}

# Datasets hosted on ADS (Atmosphere Data Store) instead of CDS.
# Same cdsapi library, different base URL: ads.atmosphere.copernicus.eu/api
_ADS_DATASETS: set = {
    "cams-global-reanalysis-eac4",
    "cams-global-reanalysis-eac4-monthly",
    "cams-global-reanalysis-eac4-complete",
    "cams-global-fire-emissions-gfas",
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

            api_key = os.getenv("CDS_API_KEY", "")

            # Determine CDS dataset ID
            dataset_id = self._resolve_dataset_id(entry)
            if not dataset_id:
                logger.warning(f"CDS: cannot determine dataset ID for {entry.dataset_name} ({entry.link})")
                return False

            if dataset_id in _CDS_SKIP_DATASETS:
                logger.info(f"CDS: skipping {dataset_id} — dataset removed from CDS, metadata-only")
                return True

            # Use ADS URL for atmosphere/CAMS datasets, CDS URL for the rest
            if dataset_id in _ADS_DATASETS:
                api_url = "https://ads.atmosphere.copernicus.eu/api"
                logger.info(f"CDS: using ADS endpoint for {dataset_id}")
            else:
                api_url = os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api")

            # Build client — prefer env vars, fall back to .cdsapirc
            if api_key:
                client = cdsapi.Client(url=api_url, key=api_key)
            else:
                client = cdsapi.Client(url=api_url)

            # Get dataset-specific request params, or use a sensible default
            request = dict(_CDS_DATASET_CONFIGS.get(dataset_id, self._default_request()))

            # Override variable based on hazard type (same dataset, different variable per hazard)
            hazard = (entry.hazard or "").strip()
            if hazard and dataset_id in _HAZARD_TO_CDS_VARIABLE:
                hazard_var = _HAZARD_TO_CDS_VARIABLE[dataset_id].get(hazard)
                if hazard_var:
                    request["variable"] = [hazard_var]
                    logger.info(f"CDS: using variable '{hazard_var}' for hazard '{hazard}'")

            # CERRA forecast-only variables need product_type=forecast + leadtime_hour
            var_name = request.get("variable", [""])[0] if isinstance(request.get("variable"), list) else request.get("variable", "")
            if dataset_id == "reanalysis-cerra-single-levels" and var_name in _CERRA_FORECAST_ONLY_VARIABLES:
                request["product_type"] = ["forecast"]
                request.setdefault("leadtime_hour", ["1"])
                logger.info(f"CDS: switched to forecast mode for CERRA variable '{var_name}'")

            # Use correct file extension based on requested format
            fmt = request.get("data_format", "netcdf")
            if fmt == "grib":
                suffix = ".grib"
            elif fmt == "zip":
                suffix = ".zip"
            else:
                suffix = ".nc"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp_path = tmp.name

            logger.info(f"CDS: downloading {dataset_id} var={request.get('variable')} for {entry.dataset_name} (format={fmt})")

            # cdsapi.retrieve() blocks with no timeout — wrap in thread
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(client.retrieve, dataset_id, request, tmp_path)
                try:
                    future.result(timeout=600)  # 10 min max for CDS
                except FuturesTimeoutError:
                    future.cancel()
                    raise TimeoutError(f"CDS retrieve exceeded 600s for {dataset_id}")

            # CDS sometimes wraps NetCDF in a ZIP — detect and unzip
            import zipfile
            if zipfile.is_zipfile(tmp_path):
                logger.info(f"CDS: downloaded file is ZIP, extracting...")
                with zipfile.ZipFile(tmp_path) as zf:
                    names = zf.namelist()
                    # Find the data file inside
                    data_file = None
                    for n in names:
                        if n.endswith((".nc", ".grib", ".grib2", ".tif")):
                            data_file = n
                            break
                    if not data_file:
                        data_file = names[0]  # fallback to first file
                    extracted = tempfile.NamedTemporaryFile(
                        suffix=Path(data_file).suffix, delete=False
                    )
                    extracted.write(zf.read(data_file))
                    extracted.close()
                    Path(tmp_path).unlink(missing_ok=True)
                    tmp_path = extracted.name
                    logger.info(f"CDS: extracted {data_file} ({os.path.getsize(tmp_path)} bytes)")

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
        """Default CDS request for unknown datasets — ERA5-like, 4 seasonal days."""
        return {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature"],
            "year": ["2023"],
            "month": ["01", "04", "07", "10"],
            "day": ["15"],
            "time": ["12:00"],
            "data_format": "netcdf",
        }


# ---------------------------------------------------------------------------
# ESGF Adapter — CMIP6, CORDEX via direct HTTP download
# ---------------------------------------------------------------------------

# Direct HTTP download URLs for ESGF datasets.
# Keyed by (dataset_name, hazard_type) for multi-variable support.
# Falls back to dataset_name-only key if no hazard-specific URL.
_ESGF_HAZARD_URLS: Dict[str, Dict[str, str]] = {
    "CMIP6": {
        "_default": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/latest/"
            "tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
        "Mean precipitation": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/pr/gn/latest/"
            "pr_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
        "Heavy precipitation and pluvial floods": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/pr/gn/latest/"
            "pr_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
        "Mean wind speed": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/sfcWind/gn/latest/"
            "sfcWind_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
        "Severe wind storm": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/sfcWind/gn/latest/"
            "sfcWind_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
        "Relative sea level": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/psl/gn/latest/"
            "psl_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
    },
    "EURO-CORDEX": {
        "_default": (
            "https://g-52ba3.fd635.8443.data.globus.org/css03_data/CMIP6/CMIP/"
            "EC-Earth-Consortium/EC-Earth3/historical/r120i1p1f1/Amon/rsdt/gr/"
            "v20200412/rsdt_Amon_EC-Earth3_historical_r120i1p1f1_gr_200501-200512.nc"
        ),
    },
    "MED-CORDEX": {
        # ESGF IPSL data node for MED-CORDEX (MED-11 domain)
        "_default": (
            "https://vesg.ipsl.upmc.fr/thredds/fileServer/cmip6/CMIP/"
            "IPSL/IPSL-CM6A-LR/historical/r1i1p1f1/Amon/tas/gr/v20180803/"
            "tas_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_185001-201412.nc"
        ),
    },
    "CMIP6-BCCAQ": {
        # CEDA DAP — CMIP6 ScenarioMIP (distinct from base CMIP6 entry)
        "_default": (
            "https://dap.ceda.ac.uk/badc/cmip6/data/CMIP6/CMIP/MOHC/"
            "HadGEM3-GC31-LL/historical/r1i1p1f3/Amon/tas/gn/latest/"
            "tas_Amon_HadGEM3-GC31-LL_historical_r1i1p1f3_gn_195001-201412.nc"
        ),
    },
}


class ESGFAdapter(PortalAdapter):
    """
    Adapter for ESGF (CMIP6, CORDEX, etc.).
    Downloads sample data via direct HTTP from ESGF data nodes / Globus endpoints.
    Falls back to OpenDAP if direct URL not available.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            # Try direct HTTP download first (most reliable)
            url = self._resolve_direct_url(entry)
            if url:
                tmp_path = self._download_via_http(url, entry)

            # Fall back to OpenDAP if no direct URL or HTTP failed
            if tmp_path is None:
                opendap_url = self._resolve_opendap_url(entry)
                if opendap_url:
                    tmp_path = self._download_via_opendap(opendap_url, entry)

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

    def _resolve_direct_url(self, entry: CatalogEntry) -> Optional[str]:
        """Find a direct HTTP download URL, using hazard-specific URL if available."""
        name = entry.dataset_name or ""
        hazard = (entry.hazard or "").strip()
        for key, hazard_urls in _ESGF_HAZARD_URLS.items():
            if key in name or name in key:
                # Try hazard-specific URL first, then default
                url = hazard_urls.get(hazard) or hazard_urls.get("_default")
                if url:
                    logger.info(f"ESGF: resolved URL for {name} hazard='{hazard}'")
                    return url
        return None

    def _resolve_opendap_url(self, entry: CatalogEntry) -> Optional[str]:
        """Find an OpenDAP URL for this dataset (fallback)."""
        link = entry.link or ""
        if "thredds/dodsC" in link:
            return link
        if "thredds/catalog" in link:
            return link.replace("thredds/catalog", "thredds/dodsC").replace("catalog.html", "")
        return None

    def _download_via_http(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Download a sample file via direct HTTP."""
        try:
            import requests as http_requests

            logger.info(f"ESGF: downloading {entry.dataset_name} from {url}")
            resp = http_requests.get(url, timeout=300, stream=True,
                                     headers={"User-Agent": "ClimateRAG/1.0"})
            resp.raise_for_status()

            # Check we got data, not HTML error page
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning(f"ESGF: got HTML instead of data for {entry.dataset_name}")
                return None

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    tmp.write(chunk)
                logger.info(f"ESGF: downloaded {written} bytes for {entry.dataset_name}")
                return tmp.name

        except Exception as e:
            logger.warning(f"ESGF HTTP download failed for {entry.dataset_name}: {e}")
            return None

    def _download_via_opendap(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Download a small spatial/temporal slice via OpenDAP (fallback)."""
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

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name
            ds.to_netcdf(tmp_path)
            ds.close()

            logger.info(f"ESGF: downloaded OpenDAP slice to {tmp_path}")
            return tmp_path

        except Exception as e:
            logger.warning(f"ESGF OpenDAP failed for {entry.dataset_name}: {e}")
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
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
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

# Direct HTTPS download URLs for NASA datasets.
# Keyed by (dataset_name, hazard) for multi-variable support.
_NASA_HAZARD_URLS: Dict[str, Dict[str, str]] = {
    "MERRA-2": {
        # M2T1NXSLV = single-level diagnostics (T2M, U10M, V10M, SLP, etc.)
        # Monthly mean files (M2TMNXSLV) — one file covers an entire month
        "_default": (
            "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2_MONTHLY/M2TMNXSLV.5.12.4/"
            "2023/MERRA2_400.tavgM_2d_slv_Nx.202307.nc4"
        ),
        # M2TMNXRAD = monthly radiation diagnostics
        "Radiation at surface": (
            "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2_MONTHLY/M2TMNXRAD.5.12.4/"
            "2023/MERRA2_400.tavgM_2d_rad_Nx.202307.nc4"
        ),
        # M2TMNXAER = monthly aerosol diagnostics (dust, black carbon, etc.)
        "Sand and dust storm": (
            "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2_MONTHLY/M2TMNXAER.5.12.4/"
            "2023/MERRA2_400.tavgM_2d_aer_Nx.202307.nc4"
        ),
    },
    "MERRA2": {
        "_default": (
            "https://data.gesdisc.earthdata.nasa.gov/data/MERRA2_MONTHLY/M2TMNXSLV.5.12.4/"
            "2023/MERRA2_400.tavgM_2d_slv_Nx.202307.nc4"
        ),
    },
}

# Simple direct URLs (no hazard variation needed)
_NASA_DIRECT_URLS: Dict[str, str] = {
    # IMERG monthly precipitation (V07B) — July 2023 for fire season relevance
    "IMERG": (
        "https://data.gesdisc.earthdata.nasa.gov/data/GPM_L3/GPM_3IMERGM.07/"
        "2023/3B-MO.MS.MRG.3IMERG.20230701-S000000-E235959.07.V07B.HDF5"
    ),
    "CERES-EBAF": (
        "https://asdc.larc.nasa.gov/data/CERES/EBAF/Edition4.2/"
        "CERES_EBAF_Edition4.2_200003-202407.nc"
    ),
    # JPL GRACE — multi-year file (2002-2023), already covers full temporal range
    "JPL GRACE": (
        "https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/"
        "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.1_V3/"
        "GRCTellus.JPL.200204_202312.GLO.RL06.1M.MSCNv03CRI.nc"
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

            # NASA Earthdata: OAuth redirect flow with username/password.
            # data.gesdisc → 302 to URS → 302 back with auth code → data
            # requests strips auth on cross-domain redirects, so we follow manually.
            username = os.getenv("NASA_EARTHDATA_USER", "")
            password = os.getenv("NASA_EARTHDATA_PASSWORD", "")
            if not username or not password:
                logger.warning("NASA: NASA_EARTHDATA_USER/PASSWORD not set")
                return False

            session = http_requests.Session()
            session.headers.update({"User-Agent": "ClimateRAG/1.0"})

            # Use Bearer token if available (works for all NASA Earthdata endpoints)
            token = os.getenv("NASA_EARTHDATA_TOKEN", "")

            logger.info(f"NASA: downloading {entry.dataset_name} from {url}")

            resp = None

            # Try Bearer token first (if available)
            if token:
                try:
                    resp = session.get(
                        url, timeout=(60, 600), stream=True,
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    # Some servers redirect even with bearer — follow with token
                    if resp.status_code in (301, 302):
                        resp = session.get(
                            resp.headers["Location"], timeout=(60, 600), stream=True,
                            headers={"Authorization": f"Bearer {token}"},
                        )
                    if resp.status_code == 401:
                        logger.warning(f"NASA: Bearer token rejected (401), falling back to URS")
                        resp = None
                    else:
                        resp.raise_for_status()
                except Exception as token_err:
                    logger.warning(f"NASA: Bearer token failed ({token_err}), falling back to URS")
                    resp = None

            # Fallback: URS redirect flow with username/password
            # NASA Earthdata requires .netrc-style auth. We write a temporary
            # .netrc so that requests sends credentials to urs.earthdata.nasa.gov
            # during the OAuth redirect chain.
            if resp is None:
                import netrc as _netrc_mod
                netrc_path = Path.home() / ".netrc"
                wrote_netrc = False
                try:
                    # Write .netrc if it doesn't exist or doesn't have URS entry
                    need_write = True
                    if netrc_path.exists():
                        try:
                            nrc = _netrc_mod.netrc(str(netrc_path))
                            if nrc.authenticators("urs.earthdata.nasa.gov"):
                                need_write = False
                        except Exception:
                            pass
                    if need_write:
                        with open(netrc_path, "a") as nf:
                            nf.write(f"\nmachine urs.earthdata.nasa.gov login {username} password {password}\n")
                        netrc_path.chmod(0o600)
                        wrote_netrc = True

                    # Use a fresh session so it picks up .netrc
                    import requests as _req
                    s2 = _req.Session()
                    s2.headers.update({"User-Agent": "ClimateRAG/1.0"})
                    resp = s2.get(url, timeout=(60, 600), stream=True)
                    resp.raise_for_status()
                finally:
                    pass  # leave .netrc for future use

            # Check we got data, not a login page
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("Got HTML response — token may be expired or invalid")

            # Detect file extension from URL
            url_lower = url.lower()
            if ".hdf5" in url_lower or ".he5" in url_lower:
                ext = ".h5"
            elif ".nc4" in url_lower:
                ext = ".nc4"
            elif ".hdf" in url_lower:
                ext = ".hdf"
            else:
                ext = ".nc"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    tmp.write(chunk)
                tmp_path = tmp.name

            # HDF5 files with nested groups (e.g. IMERG /Grid): extract to flat NetCDF
            logger.info(f"NASA: downloaded {written} bytes, ext={ext}, path={tmp_path}")
            if ext in (".h5", ".hdf") and tmp_path:
                try:
                    import h5py
                    with h5py.File(tmp_path, "r") as hf:
                        top_groups = list(hf.keys())
                    if top_groups and not any(
                        k in top_groups for k in ("lat", "lon", "time", "latitude", "longitude")
                    ):
                        # Data is in a group — convert to flat NetCDF via xarray
                        import xarray as xr
                        for grp in top_groups:
                            try:
                                ds = xr.open_dataset(tmp_path, engine="h5netcdf", group=grp)
                                if list(ds.data_vars):
                                    nc_tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
                                    nc_tmp.close()
                                    ds.to_netcdf(nc_tmp.name)
                                    ds.close()
                                    Path(tmp_path).unlink(missing_ok=True)
                                    tmp_path = nc_tmp.name
                                    logger.info(f"NASA: converted HDF5 group '{grp}' → NetCDF for {entry.dataset_name}")
                                    break
                                ds.close()
                            except Exception:
                                continue
                except Exception as conv_err:
                    logger.warning(f"NASA: HDF5 conversion failed: {conv_err}")

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
        hazard = (entry.hazard or "").strip()

        # Try hazard-specific URLs first (MERRA-2 has different collections per hazard)
        for key, hazard_urls in _NASA_HAZARD_URLS.items():
            if key in name or name in key:
                url = hazard_urls.get(hazard) or hazard_urls.get("_default")
                if url:
                    logger.info(f"NASA: resolved URL for {name} hazard='{hazard}'")
                    return url

        # Fall back to simple direct URLs
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

    # Map product IDs to their dataset IDs for the copernicusmarine API.
    # The copernicusmarine toolbox expects dataset-level IDs (with cmems_ prefix),
    # not the product-level IDs extracted from marine.copernicus.eu URLs.
    _PRODUCT_DATASETS: Dict[str, str] = {
        "SST_MED_SST_L4_REP_OBSERVATIONS_010_021": "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021",
        "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004": "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2",
    }

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            product_id = self._extract_product_id(entry.link or "")
            if not product_id:
                logger.warning(f"Marine: cannot extract product ID from {entry.link}")
                return False

            # Resolve product ID to dataset ID for the copernicusmarine toolbox
            dataset_id = self._PRODUCT_DATASETS.get(product_id, product_id)

            # Try copernicusmarine toolbox first
            tmp_path = self._download_via_toolbox(dataset_id, entry)

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

            import tempfile as _tmpmod
            tmp_dir = _tmpmod.mkdtemp(prefix="cmems_")

            # Download 4 seasonal months (Jan/Apr/Jul/Oct) for Mediterranean bbox
            logger.info(f"Marine: downloading {product_id} via copernicusmarine toolbox")
            result = copernicusmarine.subset(
                dataset_id=product_id,
                variables=["analysed_sst"],
                minimum_longitude=0,
                maximum_longitude=20,
                minimum_latitude=35,
                maximum_latitude=45,
                start_datetime="2023-01-01",
                end_datetime="2023-10-31",
                output_directory=tmp_dir,
                username=username,
                password=password,
                force_download=True,
            )
            logger.info(f"Marine: subset returned: {result}")

            # Find any .nc file in the output directory
            nc_files = list(Path(tmp_dir).glob("*.nc"))
            if nc_files:
                logger.info(f"Marine: found output file {nc_files[0]} ({nc_files[0].stat().st_size} bytes)")
                return str(nc_files[0])

            # Check if result is a path
            if result and Path(str(result)).exists():
                return str(result)

            logger.warning(f"Marine: no .nc file found in {tmp_dir}, contents: {list(Path(tmp_dir).iterdir())}")
            return None

        except ImportError:
            logger.warning("copernicusmarine not installed — pip install copernicusmarine")
            return None
        except Exception as e:
            logger.warning(f"Marine toolbox failed for {entry.dataset_name}: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            return None


# ---------------------------------------------------------------------------
# EIDC Adapter — Environmental Information Data Centre (Hydro-JULES/CHESS)
# ---------------------------------------------------------------------------

# Direct download URLs for EIDC datasets (require HTTP basic auth).
_EIDC_URLS: Dict[str, str] = {
    # Hydro-JULES SPEI drought index — open access on CEDA, use DAP for subsetting
    "Hydro-JULES": "https://dap.ceda.ac.uk/badc/hydro-jules/data/Global_drought_indices/CHIRPS_hPET/spei01.nc",
    # CMIP6-BCCAQ downscaled climate data — files are 28GB+, must use OpenDAP subsetting
    "CMIP6-BCCAQ": "https://dap.ceda.ac.uk/badc/evoflood/data/Downscaled_CMIP6_Climate_Data/tas/Historical/KACE-1-0-G",
}


class EIDCAdapter(PortalAdapter):
    """
    Adapter for EIDC/CEH/CEDA datasets (e.g. Hydro-JULES).
    Uses OpenDAP subsetting for large files, or Bearer token for direct download.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            url = _EIDC_URLS.get(entry.dataset_name or "")
            if not url:
                logger.warning(f"EIDC: no download URL for {entry.dataset_name}")
                return False

            # Try OpenDAP subsetting first (for CEDA DAP URLs), fall back to HTTP
            if "dap.ceda.ac.uk" in url:
                tmp_path = self._download_via_opendap(url, entry)
                if tmp_path is None:
                    # OpenDAP may require auth — fall back to direct HTTP with CEDA token
                    logger.info(f"EIDC: OpenDAP failed, trying direct HTTP for {entry.dataset_name}")
                    tmp_path = self._download_via_http(url, entry)
            else:
                tmp_path = self._download_via_http(url, entry)

            if tmp_path is None:
                logger.warning(f"EIDC: could not download {entry.dataset_name}")
                return False

            ok = _process_file(tmp_path, entry, adapter_name="EIDC")
            return ok

        except Exception as e:
            logger.error(f"EIDC adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _download_via_opendap(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Download a small subset via CEDA OpenDAP (for large files)."""
        try:
            import xarray as xr

            logger.info(f"EIDC: opening OpenDAP {url} for {entry.dataset_name}")
            ds = xr.open_dataset(url, engine="netcdf4")

            # Take only first 12 time steps to keep it small
            time_dim = None
            for dim in ["time", "Time", "T"]:
                if dim in ds.dims:
                    time_dim = dim
                    break
            if time_dim and ds.sizes[time_dim] > 12:
                ds = ds.isel({time_dim: slice(0, 12)})

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name
            ds.to_netcdf(tmp_path)
            ds.close()

            logger.info(f"EIDC: downloaded OpenDAP subset for {entry.dataset_name}")
            return tmp_path

        except Exception as e:
            logger.warning(f"EIDC OpenDAP failed for {entry.dataset_name}: {e}")
            return None

    def _download_via_http(self, url: str, entry: CatalogEntry) -> Optional[str]:
        """Download via direct HTTP with optional auth."""
        try:
            import requests as http_requests

            token = os.getenv("CEDA_TOKEN", "")
            headers = {"User-Agent": "ClimateRAG/1.0"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            logger.info(f"EIDC: downloading {entry.dataset_name} from {url}")
            resp = http_requests.get(url, timeout=(30, 600), stream=True, headers=headers)
            resp.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    tmp.write(chunk)
                return tmp.name

        except Exception as e:
            logger.warning(f"EIDC HTTP download failed for {entry.dataset_name}: {e}")
            return None


# ---------------------------------------------------------------------------
# NCAR RDA Adapter — JRA-55 reanalysis
# ---------------------------------------------------------------------------

# NCAR RDA dataset URLs for JRA-55 (small sample files).
_NCAR_RDA_URLS: Dict[str, str] = {
    "JRA-55": (
        "https://data.rda.ucar.edu/d628000/anl_surf/2020/"
        "anl_surf.011_tmp.reg_tl319.2020010100_2020013118"
    ),
    "JRA55": (
        "https://data.rda.ucar.edu/d628000/anl_surf/2020/"
        "anl_surf.011_tmp.reg_tl319.2020010100_2020013118"
    ),
}


class NCARAdapter(PortalAdapter):
    """
    Adapter for NCAR RDA datasets (e.g. JRA-55).
    Uses NCAR RDA API token for authenticated downloads.
    Requires NCAR_RDA_TOKEN env var.
    """

    def download_and_process(self, entry: CatalogEntry) -> bool:
        tmp_path = None
        try:
            import requests as http_requests

            token = os.getenv("NCAR_RDA_TOKEN", "")
            if not token:
                logger.warning("NCAR: NCAR_RDA_TOKEN not set")
                return False

            url = self._resolve_url(entry)
            if not url:
                logger.warning(f"NCAR: no download URL for {entry.dataset_name}")
                return False

            logger.info(f"NCAR: downloading {entry.dataset_name} from {url}")
            resp = http_requests.get(
                url, timeout=300, stream=True,
                headers={
                    "User-Agent": "ClimateRAG/1.0",
                    "Authorization": f"Bearer {token}",
                },
            )
            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("Got HTML instead of data — token may be invalid")

            # JRA-55 files are GRIB format
            suffix = ".grib"
            if url.endswith(".nc") or url.endswith(".nc4"):
                suffix = ".nc"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                written = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    written += len(chunk)
                    tmp.write(chunk)
                tmp_path = tmp.name
                logger.info(f"NCAR: downloaded {written} bytes for {entry.dataset_name}")

            ok = _process_file(tmp_path, entry, adapter_name="NCAR")
            return ok

        except Exception as e:
            logger.error(f"NCAR adapter failed for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
            return False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def _resolve_url(self, entry: CatalogEntry) -> Optional[str]:
        name = entry.dataset_name or ""
        for key, url in _NCAR_RDA_URLS.items():
            if key in name or name in key:
                return url
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
    "EIDC": EIDCAdapter,
    "NCAR": NCARAdapter,
}


def get_adapter(entry: CatalogEntry) -> Optional[PortalAdapter]:
    """Get the appropriate portal adapter for a catalog entry."""
    from src.catalog.phase_classifier import _get_portal

    name = entry.dataset_name or ""
    link = entry.link or ""

    # Name-based overrides (DOI links don't match portal domains)
    if name in _DATASET_NAME_TO_CDS_ID:
        return CDSAdapter()
    if name in _EIDC_URLS:
        return EIDCAdapter()
    if name in _NCAR_RDA_URLS:
        return NCARAdapter()
    if name in _ESGF_HAZARD_URLS:
        return ESGFAdapter()

    portal = _get_portal(link)

    # Special case: marine Copernicus
    if "marine.copernicus.eu" in link or "cmems" in link.lower():
        return MarineCopernicusAdapter()

    if portal and portal in _ADAPTERS:
        return _ADAPTERS[portal]()
    return None
