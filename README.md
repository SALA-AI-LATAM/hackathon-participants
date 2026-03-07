# SALA Hackathon — Participant Materials

AI for weather and marine ecology on San Cristóbal Island, Galápagos.

## Tracks

### 1. Precipitation Nowcasting (`precipitation-nowcasting/`)

Predict heavy precipitation at 3, 6, and 12-hour horizons using meteorological time series from four inland weather stations. Includes a full starter notebook with EDA, preprocessing, and RNN/LSTM/GRU baselines.

| File | Description |
|------|-------------|
| `precipitation_nowcasting.ipynb` | Main notebook — data exploration through model training |
| `train_precipitation.py` | Standalone training script (same pipeline as notebook) |
| `README.md` | Dataset guide — variables, stations, LDAS gridded data |
| `weather_stations/*.csv` | 4 station CSVs (15-min intervals, 2015–present) |
| `ldas/*.nc` | NASA LDAS gridded reanalysis (daily, NetCDF) |
| `checkpoints/` | Pre-trained baseline weights + evaluation plots |

### 2. Marine Acoustic Monitoring (`marine-accoustic-monitoring/`)

Build AI pipelines to analyze unlabeled underwater audio from SoundTrap hydrophones deployed in the Bay of San Cristóbal. Dolphins, sea lions, snapping shrimp, boat noise — no labels, lots of signal. Two download options: **core** (~7.3 GB, 12h of audio — Colab-friendly) or **full** (~57 GB, 97h).

| File | Description |
|------|-------------|
| `acoustic_explorer.ipynb` | Interactive notebook — load audio, plot spectrograms, listen to clips |
| `acoustic_data.py` | Standalone module — same functions, importable |
| `README.md` | Hackathon guide — project ideas across 3 tiers + CETI bonus |
| `examples/*.png` | Pre-generated spectrograms for each hydrophone unit |

## Shared Utilities

| File | Description |
|------|-------------|
| `r2_download.py` | Download datasets from Cloudflare R2 (with resume + checksum verification) |
| `data_download.ipynb` | Interactive notebook for R2 setup, credential config, and data download |

## Getting Started

1. Open `data_download.ipynb` and follow the setup cells to download datasets
2. Pick a track and open its notebook
3. Read the track's `README.md` for project ideas and background

Both tracks are designed to be open-ended — the notebooks and guides give you a starting point, but the best projects will go in unexpected directions.
