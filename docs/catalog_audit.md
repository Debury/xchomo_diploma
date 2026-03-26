# Catalog Audit: 233 Entries — Full Status

**Total: 233 entries | 69 unique datasets**
**Multi-variable support: each hazard row downloads its own variable (not just temperature)**

## Current Classification

| Phase | Entries | Status |
|-------|---------|--------|
| Phase 0 (Metadata) | 233 | ALL entries have metadata in Qdrant |
| Phase 1 (Direct download) | 97 | Automated — web pages scraped, direct files downloaded |
| Phase 3 (API portals) | 70 | Automated — CDS, ADS, NASA, ESGF, Marine, NCAR, EIDC, CEDA |
| Phase 4 (Manual/blocked) | 66 | See breakdown below |

## Automated (Phase 1 + 3): 167 entries

These are fully automated with hazard-specific variables:
- **ERA5** (12 entries) — CDS API, different variable per hazard (temp, precip, wind, pressure, soil temp, radiation)
- **CERRA** (9 entries) — CDS API, multi-variable
- **ERA5 Land** (5 entries) — CDS API, multi-variable
- **CAMS** (7 entries) — ADS API, multi-variable (wind, dust, PM10, CO2, radiation)
- **MERRA-2 / MERRA2** (8 entries) — NASA Earthdata, different collections per hazard (SLV, RAD, AER)
- **JRA-55** (6 entries) — NCAR RDA API
- **CMIP6** (15 entries) — CEDA/ESGF, multi-variable URLs (tas, pr, sfcWind, psl)
- **EURO-CORDEX** (9 entries) — ESGF Globus, all hazard rows now Phase 3
- **IMERG** (2 entries) — NASA GES DISC
- **CERES-EBAF** (1 entry) — NASA ASDC (10,456 chunks)
- **JPL GRACE** (2 entries) — PO.DAAC
- **SST_MED** (2 entries) — Marine Copernicus
- **MED-CORDEX** (1 entry) — ESGF IPSL
- **Hydro-JULES** (1 entry) — CEDA DAP
- **CMIP6-BCCAQ** (3 entries) — CEDA DAP with OpenDAP subsetting (you have CEDA token)
- **ERA5-HEAT** (1 entry) — CDS derived-utci
- **+ 83 Phase 1 entries** (E-OBS, NOAA, WorldClim, CHIRPS, Zenodo, etc.)

## Phase 4: Manual Action Needed (66 entries)

### You Can Fix These (manual download, then pipeline processes automatically)

| Dataset | Entries | Download Link | What To Do |
|---------|---------|---------------|------------|
| **Aridity Index** | 1 | [Figshare](https://doi.org/10.6084/m9.figshare.7504448.v5) | Download ZIP manually (AWS WAF blocks bots), put in `data/manual/` |
| **MSWEP** | 3 | [gloh2o.org/mswep](https://www.gloh2o.org/mswep/) | Request Google Drive access, download a sample .nc file to `data/manual/` |
| **MSWX-Past** | 7 | [gloh2o.org/mswx](https://www.gloh2o.org/mswx/) | Same GloH2O — request access, download sample to `data/manual/` |
| **ArCIS** | 2 | [arcis.it](https://www.arcis.it/wp/) | Register (Italian climate archive), download sample station data |
| **Mistral/CINECA** | 5 | [mistralportal.it](https://meteohub.mistralportal.it/) | Register at CINECA, download sample to `data/manual/` |

**Total fixable: 18 entries — put downloaded files in `data/manual/{dataset_name}.nc` and run reprocess**

### Need Registration You Can't Do (19 entries)

| Dataset | Entries | Portal | Blocker |
|---------|---------|--------|---------|
| **SAFRAN** | 7 | [meteofrance.fr](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=230&id_rubrique=39) | Needs EU VAT number |
| **SYNOP** | 5 | [meteofrance.fr](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=93&id_rubrique=32) | Needs EU VAT number |
| **Données Météo-France** | 5 | [meteo.data.gouv.fr](https://portail-api.meteofrance.fr/web/fr/) | Needs EU VAT number |
| **RADOME** | 2 | [meteofrance.fr](https://donneespubliques.meteofrance.fr/?fond=produit&id_produit=93&id_rubrique=32) | Needs EU VAT number |

### No Download API / Web Portal Only (15 entries)

| Dataset | Entries | Portal |
|---------|---------|--------|
| **Rete Mareografica Italiana** | 2 | [mareografico.it](https://www.mareografico.it/en/homepage.html) |
| **RMI-ISPRA** | 4 | [mareografico.it](https://www.mareografico.it/en/homepage.html) |
| **COST-g** | 2 | [cost-g.org](https://cost-g.org/products/) |
| **G3P** | 2 | [g3p.eu](https://www.g3p.eu/) |
| **NOA-GR** | 1 | [meteo.gr](https://meteosearch.meteo.gr/) |
| **HWE-DB** | 1 | [meteo.gr](https://meteo.gr/weather_cases.cfm) |
| **TUDES** | 4 | [tudes.harita.gov.tr](https://tudes.harita.gov.tr/) — Turkish gov, contact required |

### Contact-Only / No URL (14 entries)

| Dataset | Entries | Reason |
|---------|---------|--------|
| **CY-OBS** | 5 | Must email Cyprus Met Service |
| **MOLOCH** | 4 | Must contact CNR-ISAC Italy |
| **APGD** | 2 | No public URL exists (Alpine Precipitation Grid) |
| **LAPrec1871/1901** | 2 | No public URL |
| **HadlSST** | 1 | No URL in catalog (Met Office Hadley Centre) |

## Summary

| Category | Entries | % |
|----------|---------|---|
| Fully automated (Phase 0+1+3) | 167 | 72% |
| Manual download possible (you do it) | 18 | 8% |
| Blocked registration (Météo-France VAT) | 19 | 8% |
| No API / portal only | 15 | 6% |
| Contact-only / no URL | 14 | 6% |
| **Total** | **233** | **100%** |

All 233 entries have Phase 0 metadata in Qdrant regardless of download status.
