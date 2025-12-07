# 🚀 Implementation Plan - Konkrétne kroky

## Okamžité akcie (Tento týždeň)

### 1. ERA5 Integrácia (Priorita #1)

**Súbory na vytvorenie:**

```python
# src/data_acquisition/era5_client.py
"""
ERA5 Data Download Client
Integrácia s Copernicus CDS API pre automatické stiahnutie ERA5 dát
"""

import cdsapi
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ERA5Client:
    """Client pre stiahnutie ERA5 dát z Copernicus CDS API"""
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("CDS_API_KEY")
        self.api_url = api_url or os.getenv("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
        
        if not self.api_key:
            raise ValueError("CDS_API_KEY must be set")
        
        self.client = cdsapi.Client(
            url=self.api_url,
            key=self.api_key
        )
    
    def download(
        self,
        dataset: str,
        request_params: Dict[str, Any],
        output_path: Path,
        timeout: int = 3600
    ) -> Path:
        """
        Stiahne ERA5 dáta podľa zadaných parametrov
        
        Args:
            dataset: Dataset name (e.g., 'reanalysis-era5-single-levels')
            request_params: CDS API request parameters
            output_path: Where to save the file
            timeout: Request timeout in seconds
            
        Returns:
            Path to downloaded file
        """
        logger.info(f"Downloading ERA5 data: {dataset}")
        logger.info(f"Parameters: {request_params}")
        
        try:
            self.client.retrieve(
                dataset,
                request_params,
                str(output_path)
            )
            logger.info(f"Downloaded to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ERA5 download failed: {e}")
            raise
    
    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extrahuje ERA5-špecifické metadata"""
        import xarray as xr
        
        with xr.open_dataset(file_path) as ds:
            return {
                "source": "ERA5",
                "dataset": ds.attrs.get("source", "ERA5"),
                "variables": list(ds.data_vars),
                "dimensions": dict(ds.dims),
                "spatial_resolution": self._get_resolution(ds),
                "temporal_coverage": self._get_temporal_coverage(ds),
                "institution": ds.attrs.get("institution", "ECMWF"),
                "references": ds.attrs.get("references", "")
            }
    
    def _get_resolution(self, ds) -> Dict[str, float]:
        """Vypočíta priestorové rozlíšenie"""
        # Implementation
        pass
    
    def _get_temporal_coverage(self, ds) -> Dict[str, str]:
        """Vypočíta časové pokrytie"""
        # Implementation
        pass
```

```python
# dagster_project/ops/era5_ops.py
"""
Dagster operations pre ERA5 data acquisition
"""

from dagster import op, In, Out, Output
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@op(
    description="Download ERA5 data from CDS API",
    ins={"request_config": In(dict)},
    out=Out(dict),
    tags={"source": "era5", "type": "acquisition"}
)
def download_era5(context, request_config: Dict[str, Any]) -> Dict[str, Any]:
    """Download ERA5 data"""
    from src.data_acquisition.era5_client import ERA5Client
    from dagster_project.resources import DataPathResource
    
    data_paths = context.resources.data_paths
    output_dir = data_paths.get_raw_path()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    client = ERA5Client()
    
    # Build request
    dataset = request_config.get("dataset", "reanalysis-era5-single-levels")
    request_params = request_config.get("parameters", {})
    
    # Generate output filename
    output_file = output_dir / f"era5_{request_config.get('source_id', 'unknown')}.nc"
    
    # Download
    downloaded_path = client.download(
        dataset=dataset,
        request_params=request_params,
        output_path=output_file
    )
    
    # Extract metadata
    metadata = client.extract_metadata(downloaded_path)
    
    return {
        "source_id": request_config.get("source_id"),
        "status": "success",
        "file_path": str(downloaded_path),
        "format": "netcdf",
        "metadata": metadata
    }
```

### 2. Regridding Implementation

```python
# src/data_transformation/regridding.py
"""
Spatial regridding utilities
Normalizácia priestorového rozlíšenia medzi rôznymi datasety
"""

import xarray as xr
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def regrid_to_target(
    source_data: xr.Dataset,
    target_grid: Dict[str, Any],
    method: str = "bilinear"
) -> xr.Dataset:
    """
    Regrid source data to target grid
    
    Args:
        source_data: Source xarray Dataset
        target_grid: Target grid specification
            {
                "lat": np.array,  # Target latitudes
                "lon": np.array,  # Target longitudes
                "method": "bilinear" | "conservative" | "nearest"
            }
        method: Interpolation method
        
    Returns:
        Regridded Dataset
    """
    logger.info(f"Regridding to target grid: {method}")
    
    # Implementation using xESMF or similar
    # For now, simple interpolation
    target_ds = source_data.interp(
        lat=target_grid["lat"],
        lon=target_grid["lon"],
        method=method
    )
    
    return target_ds

def get_standard_grid(resolution: float = 0.25) -> Dict[str, np.ndarray]:
    """Get standard grid (e.g., 0.25° resolution)"""
    lats = np.arange(-90, 90 + resolution, resolution)
    lons = np.arange(-180, 180 + resolution, resolution)
    return {"lat": lats, "lon": lons}
```

### 3. Rozšírené testy

```python
# tests/test_era5_integration.py
"""
Integration tests pre ERA5
"""

import pytest
from pathlib import Path
from src.data_acquisition.era5_client import ERA5Client

@pytest.mark.skipif(
    not os.getenv("CDS_API_KEY"),
    reason="CDS_API_KEY not set"
)
def test_era5_download_small():
    """Test ERA5 download s malým datasetom"""
    client = ERA5Client()
    
    # Small request (1 day, small area)
    request_params = {
        "product_type": "reanalysis",
        "variable": "2m_temperature",
        "year": "2024",
        "month": "01",
        "day": "01",
        "time": "12:00",
        "area": [50, 13, 48, 19],  # Small area
        "format": "netcdf"
    }
    
    output_path = Path("/tmp/test_era5.nc")
    result = client.download(
        dataset="reanalysis-era5-single-levels",
        request_params=request_params,
        output_path=output_path
    )
    
    assert result.exists()
    assert result.stat().st_size > 0
```

---

## Štruktúra projektu po dokončení

```
xchomo_diploma/
├── src/
│   ├── data_acquisition/          # NOVÉ
│   │   ├── __init__.py
│   │   ├── era5_client.py         # ERA5 integrácia
│   │   ├── cmip6_client.py        # CMIP6 integrácia
│   │   ├── eobs_client.py         # E-OBS integrácia
│   │   ├── cru_client.py          # CRU integrácia
│   │   └── cordex_client.py       # EURO-CORDEX
│   │
│   ├── data_transformation/       # NOVÉ
│   │   ├── __init__.py
│   │   ├── regridding.py          # Spatial regridding
│   │   ├── temporal_ops.py        # Temporal alignment
│   │   ├── metadata_normalizer.py # Metadata normalization
│   │   ├── variable_ops.py         # Variable operations
│   │   └── quality_checks.py      # Data quality
│   │
│   ├── climate_embeddings/        # ✅ Existuje
│   ├── embeddings/                # ✅ Existuje
│   ├── llm/                       # ✅ Existuje
│   ├── sources.py                 # ✅ Existuje
│   └── utils/                     # ✅ Existuje
│
├── dagster_project/
│   ├── ops/
│   │   ├── era5_ops.py            # NOVÉ
│   │   ├── cmip6_ops.py           # NOVÉ
│   │   ├── dynamic_source_ops.py  # ✅ Existuje
│   │   └── embedding_ops.py      # ✅ Existuje
│   ├── jobs.py                    # ✅ Existuje
│   └── repository.py              # ✅ Existuje
│
├── tests/
│   ├── test_era5_integration.py   # NOVÉ
│   ├── test_regridding.py         # NOVÉ
│   ├── test_formats_comprehensive.py  # NOVÉ
│   └── ...                        # ✅ Existuje
│
├── docs/
│   ├── ARCHITECTURE.md            # NOVÉ
│   ├── API.md                     # NOVÉ
│   ├── USER_GUIDE.md              # NOVÉ
│   └── ...                        # ✅ Existuje
│
└── config/
    ├── pipeline_config.yaml       # ✅ Existuje
    └── era5_config.yaml           # NOVÉ (prípadne)
```

---

## Konfigurácia pre nové datasety

```yaml
# config/pipeline_config.yaml (rozšírenie)

data_acquisition:
  era5:
    enabled: true
    api_key: ${CDS_API_KEY}
    default_area: [51, 13, 48, 19]  # Central Europe
    default_variables:
      - "2m_temperature"
      - "total_precipitation"
  
  cmip6:
    enabled: true
    thredds_base_url: "https://esgf-data.dkrz.de/thredds"
    default_models:
      - "MPI-ESM1-2-HR"
    default_scenarios:
      - "ssp585"
  
  eobs:
    enabled: true
    base_url: "https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php"
    default_variables:
      - "tg"  # Mean temperature
      - "rr"  # Precipitation

regridding:
  enabled: true
  target_resolution: 0.25  # degrees
  method: "bilinear"  # bilinear, conservative, nearest
  standard_grid:
    lat_range: [-90, 90]
    lon_range: [-180, 180]
    resolution: 0.25
```

---

## Next Steps (Tento týždeň)

1. **Dnes**: Vytvoriť ERA5 client základ
2. **Zajtra**: Integrácia do Dagster
3. **Pozajtra**: Testy s malým ERA5 datasetom
4. **Koniec týždňa**: Regridding základ

**Priorita**: ERA5 → Regridding → Testy → Dokumentácia

---

**Status**: 🚀 Ready to implement  
**Estimated Time**: 4-6 týždňov pre kompletnú implementáciu

