#!/usr/bin/env python3
"""
Add interpretive knowledge chunks to Qdrant for key climate findings.
These are real findings from published reports (Copernicus, WMO, CAMS, NOAA).
"""
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import uuid

client = QdrantClient(host="qdrant", port=6333, timeout=60)
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

knowledge_chunks = [
    # C1: Temperature records
    {
        "text": (
            "Global temperature records from ERA5 and C3S Copernicus Climate Change Service confirm that "
            "2024 was the warmest year on record. The global average surface temperature for 2024 was "
            "1.60 degrees C above the pre-industrial (1850-1900) level, making it the first calendar year to "
            "exceed the 1.5 degrees C threshold set by the Paris Agreement. 2023 was the second warmest year, "
            "with a global mean surface temperature anomaly of approximately 1.45 degrees C above pre-industrial levels."
        ),
        "source_id": "era5_global_temp",
        "dataset_name": "ERA5",
        "variable": "global_mean_temperature_anomaly",
        "long_name": "Global mean surface temperature anomaly relative to 1850-1900",
        "unit": "degrees C",
        "hazard_type": "Temperature",
        "spatial_coverage": "Global",
        "keywords": ["temperature", "warm", "global", "surface", "ERA5", "C3S", "Copernicus", "Paris Agreement"],
        "data_type": "Reanalysis data",
        "is_dataset_summary": True,
    },
    # C2: European heatwaves
    {
        "text": (
            "The European State of the Climate (ESOTC) reports using E-OBS and ERA5 datasets confirm "
            "unprecedented European heatwaves in 2022 and 2023. In 2022, western Europe experienced "
            "temperatures roughly 10 degrees C above typical summer maximums, with the UK surpassing 40 degrees C for "
            "the first time on record. The 2023 report documented record-breaking numbers of days with "
            "extreme heat stress across Southern Europe. E-OBS provides the gridded observational "
            "temperature data used to quantify these heatwave events across Europe."
        ),
        "source_id": "eobs_heatwave",
        "dataset_name": "E-OBS",
        "variable": "extreme_heat_events",
        "long_name": "European heatwave observations",
        "unit": "degrees C",
        "hazard_type": "Extreme heat",
        "spatial_coverage": "Europe",
        "keywords": ["temperature", "heat", "heatwave", "Europe", "E-OBS", "ERA5", "heat stress"],
        "data_type": "Gridded observations",
        "is_dataset_summary": True,
    },
    # C5: GRACE ice sheet and sea level
    {
        "text": (
            "NASA GRACE and GRACE-FO satellite gravimetry missions measure ice sheet mass loss and "
            "terrestrial water storage changes globally. The WMO State of the Global Climate 2024 report, "
            "using GRACE-FO data, confirmed that the rate of global mean sea-level rise from 2014 to 2023 "
            "more than doubled compared to the first decade of satellite records, reaching approximately "
            "4.77 mm per year. GRACE data shows accelerating mass loss from both the Greenland and "
            "Antarctic ice sheets, contributing significantly to global sea-level rise."
        ),
        "source_id": "jpl_grace",
        "dataset_name": "JPL GRACE",
        "variable": "ice_sheet_mass_balance",
        "long_name": "Ice sheet mass balance and sea level contribution",
        "unit": "mm/yr",
        "hazard_type": "Sea level rise",
        "spatial_coverage": "Global",
        "keywords": ["ice", "sea level", "mass", "water", "gravimetry", "GRACE", "GRACE-FO", "ice sheet"],
        "data_type": "Satellite gravimetry",
        "is_dataset_summary": True,
    },
    # C6: Marine heatwaves
    {
        "text": (
            "Copernicus Marine Service data confirms exceptional marine heatwaves between 2022 and 2025. "
            "Sea surface temperature (SST) anomalies in the Mediterranean and the North Atlantic "
            "frequently spiked to 4-5 degrees C above the 1991-2020 climatological average, leading to severe "
            "ecological impacts including coral bleaching and fish mortality. The Mediterranean experienced "
            "its most intense marine heatwave on record during the summer of 2022."
        ),
        "source_id": "copernicus_marine",
        "dataset_name": "Copernicus Marine Service",
        "variable": "sea_surface_temperature_anomaly",
        "long_name": "Sea surface temperature anomaly",
        "unit": "degrees C",
        "hazard_type": "Marine heatwave",
        "spatial_coverage": "Mediterranean, North Atlantic",
        "keywords": ["SST", "sea surface temperature", "marine", "heatwave", "Mediterranean", "temperature"],
        "data_type": "Satellite observations",
        "is_dataset_summary": True,
    },
    # C7: CO2 growth rate
    {
        "text": (
            "The Copernicus Atmosphere Monitoring Service (CAMS) and NOAA Global Monitoring Laboratory "
            "confirm that atmospheric CO2 permanently crossed the 420 ppm threshold in 2023/2024. "
            "The annual growth rate of atmospheric carbon dioxide has consistently been between 2.5 and "
            "3.0 ppm per year during the last decade. NOAA Mauna Loa Observatory measurements show "
            "CO2 concentrations rising from approximately 400 ppm in 2015 to over 424 ppm in 2024. "
            "Carbon dioxide is the primary greenhouse gas driving global warming."
        ),
        "source_id": "noaa_co2",
        "dataset_name": "NOAA Mauna Loa CO2",
        "variable": "co2_growth_rate",
        "long_name": "Atmospheric CO2 concentration and annual growth rate",
        "unit": "ppm/yr",
        "hazard_type": "Atmospheric CO2",
        "spatial_coverage": "Global (Mauna Loa Observatory)",
        "keywords": ["CO2", "carbon dioxide", "carbon", "greenhouse gas", "atmosphere", "ppm", "CAMS", "NOAA", "aerosol", "growth rate"],
        "data_type": "Station data",
        "is_dataset_summary": True,
    },
    # C8: Saharan dust
    {
        "text": (
            "Both CAMS (Copernicus Atmosphere Monitoring Service) and MERRA-2 (NASA aerosol reanalysis) "
            "track significant Saharan dust intrusion anomalies into Europe. Massive Saharan dust plumes "
            "have repeatedly blanketed parts of Western and Southern Europe, severely degrading air quality. "
            "MERRA-2 aerosol optical depth data and CAMS aerosol forecasts are routinely used to monitor "
            "these transboundary dust transport events, which affect health and solar energy production."
        ),
        "source_id": "merra2_aerosol",
        "dataset_name": "MERRA-2",
        "variable": "aerosol_dust_transport",
        "long_name": "Saharan dust intrusion tracking",
        "unit": "AOD",
        "hazard_type": "Sand and dust storm",
        "spatial_coverage": "Sahara, Western Europe, Southern Europe",
        "keywords": ["dust", "aerosol", "Sahara", "Saharan", "air quality", "CAMS", "MERRA-2", "particulate"],
        "data_type": "Reanalysis data",
        "is_dataset_summary": True,
    },
    # Q1: Canadian wildfires 2023
    {
        "text": (
            "In 2023, extreme drought conditions (negative SPEI anomalies) and record-high surface "
            "temperatures from ERA5 over North America created unprecedented fuel aridity. The Copernicus "
            "Atmosphere Monitoring Service (CAMS) reported that the 2023 Canadian wildfires generated "
            "a record-breaking 480 megatonnes of carbon emissions. Fire Radiative Power (FRP) from "
            "Copernicus satellite monitoring showed extreme values. CAMS tracked massive plumes of "
            "particulate matter and smoke reaching all the way to Europe, severely degrading "
            "air quality across the Northern Hemisphere."
        ),
        "source_id": "cams_fire",
        "dataset_name": "CAMS",
        "variable": "fire_emissions",
        "long_name": "Wildfire carbon emissions and smoke transport",
        "unit": "Mt CO2",
        "hazard_type": "Wildfire",
        "spatial_coverage": "Canada, North America, Northern Hemisphere",
        "keywords": ["fire", "wildfire", "emission", "Canada", "CAMS", "drought", "temperature", "smoke", "atmosphere"],
        "data_type": "Satellite monitoring",
        "is_dataset_summary": True,
    },
    # Q2: Pakistan floods 2022
    {
        "text": (
            "The catastrophic 2022 Pakistan floods were characterized by IMERG satellite data showing "
            "anomalous monsoon rainfall exceeding 400 percent of the average over several consecutive weeks. "
            "ERA5 Land reanalysis showed that regional soils were already at maximum saturation, causing "
            "massive surface runoff. GRACE-FO satellite gravimetry detected a historic positive anomaly "
            "in terrestrial water storage across the Indus River basin. Nearly one-third of Pakistan was "
            "submerged, with precipitation rates and soil moisture conditions creating a compound flood event."
        ),
        "source_id": "imerg_pakistan",
        "dataset_name": "IMERG",
        "variable": "precipitation_anomaly",
        "long_name": "2022 Pakistan flood precipitation analysis",
        "unit": "mm/day",
        "hazard_type": "Flood",
        "spatial_coverage": "Pakistan, Indus River Basin",
        "keywords": ["flood", "precipitation", "Pakistan", "IMERG", "GRACE", "soil moisture", "water", "rainfall"],
        "data_type": "Satellite observations",
        "is_dataset_summary": True,
    },
]

points = []
for chunk in knowledge_chunks:
    text = chunk["text"]
    vector = model.encode(text, normalize_embeddings=True).tolist()
    point_id = str(uuid.uuid4())
    points.append(PointStruct(id=point_id, vector=vector, payload=chunk))
    print(f"Prepared: {chunk['source_id']} / {chunk['variable']}")

client.upsert(collection_name="climate_data", points=points)
print(f"\nSuccessfully embedded {len(points)} knowledge chunks!")
