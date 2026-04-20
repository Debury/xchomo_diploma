# Demo-Ready Source URLs

Curated list of climate data URLs verified to work with the per-source ETL path
(`POST /sources/` → auto-embed → Qdrant → RAG). Sorted by expected runtime, so
pick the top entries for live demo (fast feedback loop) and the larger ones as
"also prepared" backup talking points.

The whole stack uses the Docker compose network, so the commands below assume
you're hitting `http://localhost:8000`. Swap in your auth token after login.

---

## Tier 1 — Instant (< 30 s, tiny CSV/TXT)

These are ideal for the live "Add Source" step in front of an audience — the
user sees the pulse dot start in ETL Monitor and the source flips to
`completed` before they can blink.

| Label | `source_id` | URL | Format | Notes |
|-------|-------------|-----|--------|-------|
| NOAA global temperature anomaly | `demo_noaa_global_temp` | `https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/ytd/12/1880-2023.csv` | csv | Annual anomaly, 1880–2023. Already ingested in this environment (`2` chunks, RAG tested). |
| NASA GISTEMP GLB.Ts+dSST | `demo_gistemp_glb` | `https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv` | csv | Monthly anomalies. `catalog_GISTEMP_7` already completed here. |
| ISIMIP historical CO₂ | `demo_isimip_co2` | `https://files.isimip.org/ISIMIP3b/InputData/climate/atmosphere_composition/co2/historical/co2_historical_annual_1850_2014.txt` | csv | Single-column annual CO₂ concentration. |

## Tier 2 — Short (1–3 min, small NetCDF / GeoTIFF)

Good for a "watch the progress bar move" moment if Tier 1 is too fast to be
visible.

| Label | `source_id` | URL | Format | Notes |
|-------|-------------|-----|--------|-------|
| HadISST sea-surface temperature | `demo_hadisst` | `https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz` | netcdf | `.nc.gz` — exercises the gunzip path. ~200 MB gzipped. |
| CHIRPS 2020 annual precipitation | `demo_chirps_2020` | `https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_annual/tifs/chirps-v2.0.2020.tif` | geotiff | ~40 MB global raster. |
| NCEP/NCAR R1 surface air 2023 | `demo_ncep_ncar_2023` | `https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/surface/air.sig995.2023.nc` | netcdf | ~30 MB, daily reanalysis. |
| GPCP monthly precipitation | `demo_gpcp` | `https://downloads.psl.noaa.gov/Datasets/gpcp/precip.mon.mean.nc` | netcdf | Global monthly precip. |
| GPCC full v2020 precip | `demo_gpcc_2020` | `https://downloads.psl.noaa.gov/Datasets/gpcc/full_v2020/precip.mon.total.0.25x0.25.v2020.nc` | netcdf | High-res monthly totals. |

## Tier 3 — Long (5+ min, hundreds of MB)

Only run these before the demo starts. They are large enough that Dagster's
in-process executor will stay busy for the duration, which can crowd out a
simultaneous live trigger.

| Label | `source_id` | URL | Format | Notes |
|-------|-------------|-----|--------|-------|
| E-OBS tg mean 0.25° v32 | `demo_eobs_tg` | `https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.25deg_reg_ensemble/tg_ens_mean_0.25deg_reg_2011-2025_v32.0e.nc` | netcdf | Daily temperature mean, Europe, 2011–2025. Hundreds of MB. |
| CRU TS 4.09 temperature | `demo_cru_tmp` | `https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.09/cruts.2503051245.v4.09/tmp/cru_ts4.09.2021.2024.tmp.dat.nc.gz` | netcdf | Monthly global temperature, also `.nc.gz`. |
| SLOCLIM Slovenia daily Tmax | `demo_sloclim_tmax` | `https://zenodo.org/api/records/4108543/files/sloclim_tmax_h.nc/content` | netcdf | Slovenian high-resolution. `catalog_SLOCLIM_31` also exists here with 254 571 chunks. |

## Skip these (known-broken or auth-required)

| Source | Why |
|--------|-----|
| ERA5, ERA5-Land, CERRA | CDS API — needs `CDS_API_KEY`. Add via Settings first. |
| MERRA-2, IMERG, CERES-EBAF | NASA Earthdata token required. |
| EURO-CORDEX, MED-CORDEX, CMIP6-BCCAQ | ESGF / CEDA auth. |
| CMEMS / Copernicus Marine | `CMEMS_USERNAME` + `CMEMS_PASSWORD`. |
| Figshare Aridity Index | AWS WAF blocks automated downloads. |
| SPEI-GD (Zenodo 8060268) | Returns 403 on direct download — Zenodo-side auth changed. |
| STEAD tmax_pen.nc | Upstream `IncompleteRead` mid-transfer; unreliable for live demo. |
| Any "portal only" HTML URLs (MSWEP, MSWX-Past, NOA-GR, SAFRAN, RADOME, TUDES, ArCIS, Rete Mareografica, etc.) | No programmatic download. |

## Quick copy-paste: Add Source request body

```json
{
  "source_id": "demo_noaa_global_temp",
  "url": "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/ytd/12/1880-2023.csv",
  "format": "csv",
  "description": "NOAA global land-ocean temperature anomaly 1880-2023",
  "hazard_type": "Mean surface temperature",
  "region_country": "Global",
  "spatial_coverage": "Global",
  "impact_sector": "Climate reference",
  "keywords": ["temperature", "anomaly", "NOAA"],
  "is_active": true
}
```

Or via UI: **Sources → + Add source → fill the wizard → Submit**. The backend
auto-triggers `single_source_etl_job` with `trigger_type=auto_embed`, so
switching to **ETL Monitor** immediately shows the live pulse.

## Suggested chat prompts after ingest

- "What does NOAA show about global temperature trends since 1880?" — targets
  `demo_noaa_global_temp` (`2` chunks) or `catalog_GISTEMP_7` (`3` chunks).
- "What is the range of sea surface temperature anomalies in the HadISST
  record?" — targets `catalog_HadlSST_213`.
- "Summarize precipitation trends in Slovenia from the SLOCLIM dataset" —
  targets `catalog_SLOCLIM_31` (254 571 chunks, the showpiece for scale).

Use the "Filter by source" dropdown in Chat to scope retrieval to one source
at a time — makes the provenance story cleaner during Q&A.
