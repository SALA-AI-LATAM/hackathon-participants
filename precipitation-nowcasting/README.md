# Precipitation Nowcasting — Dataset Guide

## Starter Notebook

**`precipitation_nowcasting.ipynb`** — walks you through loading and exploring the weather station data, preprocessing (gap handling, feature engineering, extreme event labeling), and training RNN/LSTM/GRU baselines for 3/6/12-hour precipitation nowcasting.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SALA-AI-LATAM/hackathon-participants/blob/main/precipitation-nowcasting/precipitation_nowcasting.ipynb)

## Challenge Guidelines

**[`raincaster_guidelines.pdf`](raincaster_guidelines.pdf)** — the official RainCaster Galapagos Dataset guidelines. Covers the challenge rules, evaluation strategy (walk-forward validation over the final year), 3-class precipitation thresholds for each horizon, required metrics, submission CSV format, and deliverables for final judging. **Read this before starting your model.**

## Overview

San Cristóbal Island in the Galápagos has a network of four weather stations operated by the Galápagos Science Center (GSC) and the Institute of Geography at Universidad San Francisco de Quito (USFQ). These stations record meteorological variables at 15-minute intervals and have been running since June 2015.

The goal is to build an **AI-based Early Warning System** that predicts the probability of heavy precipitation or anomalous temperature at 3, 6, and 12-hour horizons (**nowcasting**), by identifying precursor patterns in the meteorological time series.

Live dashboard: https://weathergsc.usfq.edu.ec/

## Dataset Structure

```
precipitation-nowcasting/
├── weather_stations/          # Point observations, 15-min intervals
│   ├── CER_consolid_f15.csv   # Cerro Alto station
│   ├── JUN_consolid_f15.csv   # El Junco station
│   ├── MERC_consolid_f15.csv  # Merceditas station
│   └── MIRA_consolid_f15.csv  # El Mirador station
├── ldas/                      # Gridded model output, daily
│   ├── SURF_MOD_gal_001_daily_LIS_HIST.nc   # Surface model (33 vars)
│   └── ROUTING_gal_001_daily_LIS_HIST.nc    # Hydrological routing (4 vars)
└── README.md
```

## Weather Station Data

Four stations on San Cristóbal Island, each recording at **15-minute intervals** since **June 2015** through **March 2026** (~376K rows per station, 46 columns).

### Stations

| Station | File | Description |
|---------|------|-------------|
| Cerro Alto | `CER_consolid_f15.csv` | Highland station |
| El Junco | `JUN_consolid_f15.csv` | Near the freshwater lake at the island's summit |
| Merceditas | `MERC_consolid_f15.csv` | Mid-elevation agricultural zone |
| El Mirador | `MIRA_consolid_f15.csv` | Coastal/lowland station |

### Variables

**Timestamp and metadata:**

| Column | Description | Unit |
|--------|-------------|------|
| TIMESTAMP | Date and time of recording | month/day/year hour:minute |
| RECORD | Record number | — |
| BattV_Avg | Average battery voltage | Volts |
| PTemp_C_Avg | Average panel temperature | °C |

**Precipitation (primary target):**

| Column | Description | Unit |
|--------|-------------|------|
| Rain_mm_Tot | Total precipitation in interval | mm |

**Temperature:**

| Column | Description | Unit |
|--------|-------------|------|
| AirTC_Avg | Average air temperature | °C |
| AirTC_Max | Maximum air temperature | °C |
| AirTC_Min | Minimum air temperature | °C |

**Humidity:**

| Column | Description | Unit |
|--------|-------------|------|
| RH_Max | Maximum relative humidity | % |
| RH_Min | Minimum relative humidity | % |
| RH_Avg | Average relative humidity | % |

**Solar and net radiation:**

| Column | Description | Unit |
|--------|-------------|------|
| SlrkW_Avg | Average global solar radiation | kW |
| SlrMJ_Tot | Total solar radiation | MJ |
| SlrW_Max / SlrW_Min | Max/min solar radiation | W |
| NR_Wm2_Avg | Average net radiation | W/m² |
| NR_Wm2_Max / NR_Wm2_Min | Max/min net radiation | W/m² |
| CNR_Wm2_Avg / Max / Min | Corrected net radiation (avg/max/min) | W/m² |

**Wind:**

| Column | Description | Unit |
|--------|-------------|------|
| WS_ms_Avg | Average wind speed | m/s |
| WS_ms_Max / WS_ms_Min | Max/min wind speed | m/s |
| WindDir | Wind direction | Degrees (0–360) |
| WindDir_Max / Min / Avg | Max/min/avg wind direction | Degrees (0–360) |

**Soil moisture and conductivity (3 horizons):**

| Column | Description | Unit |
|--------|-------------|------|
| VW / VW_2 / VW_3 | Volumetric water content (horizons 1–3) | 0–1 |
| VW_Max / VW_Min (per horizon) | Max/min volumetric water content | 0–1 |
| PA_uS / PA_uS_2 / PA_uS_3 | Electrical conductivity (horizons 1–3) | μS |
| PA_uS_Max / PA_uS_Min (per horizon) | Max/min conductivity | μS |

**Leaf wetness:**

| Column | Description | Unit |
|--------|-------------|------|
| LWmV_Avg | Leaf wetness sensor average signal | mV |
| LWMDry_Tot | Minutes of dry leaf surface in interval | minutes |
| LWMCon_Tot | Minutes of condensation on leaf surface | minutes |
| LWMWet_Tot | Minutes of wet leaf surface in interval | minutes |

### Data Quality Notes

- **Significant data gaps** exist across all stations due to sensor failures and maintenance periods. Gap handling is a first-class concern for any modeling approach.
- **NA values** are common, especially in the min/max/avg columns which were only added in March 2025.
- **Timestamp format** is `month/day/year hour:minute` — parse with `pd.to_datetime(df['TIMESTAMP'], format='%m/%d/%Y %H:%M')`.
- Battery voltage (BattV_Avg) and panel temperature (PTemp_C_Avg) are station health indicators, not meteorological variables, but drops in battery voltage can correlate with cloudy/rainy conditions.

## LDAS Gridded Data (Supplementary)

NASA Land Data Assimilation System (LDAS) model output covering the Galápagos region on a **500×600 spatial grid** at **daily resolution** from **January 2, 2015 to July 31, 2021** (2403 time steps). These are physics-based reanalysis products — not direct observations — that fill in spatial context between the point-based weather stations.

Requires the `netCDF4` Python package to read: `pip install netCDF4`.

### SURF_MOD (Surface Model) — 33 variables

| Variable | Description | Unit |
|----------|-------------|------|
| Rainf_tavg | Precipitation rate | kg/m²/s |
| Tair_f_tavg | Air temperature | K |
| Wind_f_tavg | Wind speed | m/s |
| Qair_f_tavg | Specific humidity | kg/kg |
| Psurf_f_tavg | Surface pressure | Pa |
| SWdown_f_tavg | Downward shortwave radiation | W/m² |
| LWdown_f_tavg | Downward longwave radiation | W/m² |
| Swnet_tavg | Net downward shortwave radiation | W/m² |
| Lwnet_tavg | Net downward longwave radiation | W/m² |
| Qle_tavg | Latent heat flux | W/m² |
| Qh_tavg | Sensible heat flux | W/m² |
| Qg_tavg | Soil heat flux | W/m² |
| Evap_tavg | Total evapotranspiration | kg/m²/s |
| PotEvap_tavg | Potential evapotranspiration | kg/m²/s |
| ECanop_tavg | Interception evaporation | kg/m²/s |
| TVeg_tavg | Vegetation transpiration | kg/m²/s |
| ESoil_tavg | Bare soil evaporation | kg/m²/s |
| Qs_tavg | Surface runoff | kg/m²/s |
| Qsb_tavg | Subsurface runoff | kg/m²/s |
| SoilMoist_inst | Soil moisture (4 layers) | m³/m³ |
| SoilTemp_inst | Soil temperature (4 layers) | K |
| SWE_inst | Snow water equivalent | kg/m² |
| Snowf_tavg | Snowfall rate | kg/m²/s |
| Qsm_tavg | Snowmelt | kg/m²/s |
| Albedo_inst | Surface albedo | — |
| WaterTableD_inst | Water table depth | m |
| TWS_inst | Terrestrial water storage | mm |
| GWS_inst | Groundwater storage | mm |
| LAI_inst | Leaf area index | — |
| Greenness_inst | Green vegetation fraction | — |

### ROUTING (Hydrological Routing) — 4 variables

| Variable | Description | Unit |
|----------|-------------|------|
| Streamflow_tavg | Streamflow | m³/s |
| RiverDepth_inst | River depth | m |
| RiverFlowVelocity_inst | River flow velocity | m/s |
| FloodedArea_inst | Flooded area | m² |

### Variable Overlap with Weather Stations

Several LDAS variables correspond directly to weather station measurements, enabling paired analysis:

| Station Variable | LDAS Variable | Notes |
|-----------------|---------------|-------|
| AirTC (°C) | Tair_f_tavg (K) | Convert: °C = K − 273.15 |
| Rain_mm_Tot (mm) | Rainf_tavg (kg/m²/s) | Rate vs accumulation — convert by multiplying rate × seconds |
| WS_ms (m/s) | Wind_f_tavg (m/s) | Direct comparison |
| SlrkW_Avg (kW) | SWdown_f_tavg (W/m²) | Unit conversion needed |
| RH (%) | Qair_f_tavg + Psurf + Tair | Relative humidity derivable from specific humidity |
| VW (soil moisture) | SoilMoist_inst (4 layers) | Direct comparison, matched by horizon |
| NR_Wm2_Avg (W/m²) | Swnet + Lwnet (W/m²) | Sum of shortwave + longwave net |

### Quick Start — Reading LDAS Data

```python
import netCDF4
import numpy as np

ds = netCDF4.Dataset('ldas/SURF_MOD_gal_001_daily_LIS_HIST.nc')

# Time axis
times = netCDF4.num2date(ds['time'][:], ds['time'].units)

# Air temperature at one grid cell (e.g., row=300, col=250) across all days
temp_K = ds['Tair_f_tavg'][:, 300, 250]
temp_C = temp_K - 273.15

# Precipitation rate → daily accumulation (mm/day)
rain_rate = ds['Rainf_tavg'][:, 300, 250]  # kg/m²/s
rain_mm_day = rain_rate * 86400             # 1 kg/m² = 1 mm

ds.close()
```

## Project Ideas

### Core Task — Precipitation Nowcasting

Build a model that predicts precipitation intensity at **+1h, +3h, and +6h** horizons for all 4 weather stations. The official evaluation uses **3-class labels** (No rain / Light rain / Heavy rain) with horizon-dependent thresholds, **walk-forward validation** over the final year, and **Macro/Micro/Weighted F1** metrics — see [`raincaster_guidelines.pdf`](raincaster_guidelines.pdf) for full details.

> **Note:** The starter notebook uses 3h/6h/12h horizons with binary labels as a teaching baseline. To match the official challenge, change `HORIZONS` to `{'1h': 4, '3h': 12, '6h': 24}` and implement 3-class labeling per the guidelines.

Recommended approaches:

- **RNN / LSTM / GRU** — sequence models that learn temporal dependencies in the multivariate time series
- **XGBoost / LightGBM** — tree-based models with engineered lag features (e.g., rainfall in the last 1h, 3h, 6h; humidity trends; wind direction shifts)
- **Hybrid** — use tree models for feature importance analysis, then feed top features into a sequence model

Key modeling decisions:
- How do you define "heavy precipitation"? (threshold on Rain_mm_Tot — e.g., top 5% of nonzero readings)
- How do you handle data gaps? (interpolation, masking, or gap-aware architectures)
- Single-station models vs multi-station models that capture spatial correlations across the island

### Extension Ideas

- **Cross-station spatial modeling** — the four stations span a coastal-to-highland transect. Weather systems often move across the island in predictable patterns. Can information from one station predict events at another?

- **Soil moisture as a leading indicator** — soil moisture (VW) responds to rainfall but also to subsurface water movement. Do soil moisture trends at deeper horizons (VW_2, VW_3) provide early warning of heavy rain?

- **Leaf wetness signals** — the leaf wetness sensor (LWMWet_Tot, LWMDry_Tot) provides a direct proxy for surface moisture. Is it a better predictor than humidity alone?

- **LDAS as spatial context** — extract the LDAS grid cell(s) nearest each weather station and use the gridded variables as additional features. The LDAS provides mesoscale atmospheric context that a single point station cannot capture. This requires finding the station coordinates — they can be obtained from the live dashboard map at https://weathergsc.usfq.edu.ec/ or by contacting the station operators.

- **LDAS pretraining** — pretrain a temporal encoder on the 2403-day LDAS sequences (predict next-day surface conditions), then fine-tune on the higher-frequency station data. The LDAS provides continuous daily coverage without gaps, making it ideal for learning general atmospheric dynamics before specializing to 15-minute nowcasting.

- **Representation learning with paired data** — during the 2015–2021 overlap period, station and LDAS data are available simultaneously. Train a model to align station point observations with gridded spatial context (e.g., contrastive learning), then use the learned representations for downstream nowcasting.

- **Hydrological flood indicators** — the ROUTING file contains streamflow, river depth, and flooded area. These are direct consequences of heavy precipitation and could serve as auxiliary prediction targets or validation signals.

- **Anomalous temperature prediction** — beyond precipitation, predict temperature anomalies (deviations from seasonal norms) at the same 3/6/12h horizons. Temperature on the Galápagos is influenced by ENSO, ocean currents, and altitude — different dynamics than precipitation.
