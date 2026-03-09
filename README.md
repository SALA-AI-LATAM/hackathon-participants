# SALA Hackathon — Participant Materials

AI for weather and marine ecology on San Cristóbal Island, Galápagos.

## Tracks

### 1. [Precipitation Nowcasting](precipitation-nowcasting/)

Predict heavy precipitation at 3, 6, and 12-hour horizons using meteorological time series from four inland weather stations. Includes a full starter notebook with EDA, preprocessing, and RNN/LSTM/GRU baselines.

| File | Description |
|------|-------------|
| `precipitation_nowcasting.ipynb` | Main notebook — data exploration through model training |
| `train_precipitation.py` | Standalone training script (same pipeline as notebook) |
| `README.md` | Dataset guide — variables, stations, LDAS gridded data |
| `weather_stations/*.csv` | 4 station CSVs (15-min intervals, 2015–present) |
| `ldas/*.nc` | NASA LDAS gridded reanalysis (daily, NetCDF) |
| `checkpoints/` | Pre-trained baseline weights + evaluation plots |

### 2. [Marine Acoustic Monitoring](marine-accoustic-monitoring/)


Explore unlabeled underwater audio from SoundTrap hydrophones deployed in the Bay of San Cristóbal. Build AI pipelines for marine soundscape analysis, or work with domain experts to propose research directions for marine bioacoustics in the Galápagos. Two download options: **core** (~7.3 GB, 12h of audio — Colab-friendly) or **full** (~57 GB, 97h).

| File | Description |
|------|-------------|
| `acoustic_explorer.ipynb` | Interactive notebook — load audio, plot spectrograms, listen to clips |
| `acoustic_data.py` | Standalone module — same functions, importable |
| `README.md` | Hackathon guide — project ideas across 3 tiers + CETI bonus |
| `examples/*.png` | Pre-generated spectrograms for each hydrophone unit |

### 3. [BRUV Fish Counting](bruv-fish-counting/)

Count fish species in underwater video from Baited Remote Underwater Video stations (BRUVs) deployed by MigraMar. Primary target: determine the maximum number of *Caranx caballus* (green jack) in a single frame. This track has a **Kaggle competition** for submissions.

**Kaggle:** https://www.kaggle.com/competitions/marine-conservation-with-migra-mar

| File | Description |
|------|-------------|
| `bruv_explorer.ipynb` | Starter notebook — frame extraction, visualization, motion baseline |
| `README.md` | Pipeline guide — project ideas across 3 tiers |

Label CSVs are on Kaggle. Videos (~65 GB) are on R2.

## Shared Utilities

| File | Description |
|------|-------------|
| `r2_download.py` | Download datasets from Cloudflare R2 (with resume + checksum verification) |
| `data_download.ipynb` | Interactive notebook for R2 setup, credential config, and data download |

## Getting Started

1. Open `data_download.ipynb` and follow the setup cells to download datasets
2. Pick a track and open its notebook
3. Read the track's `README.md` for project ideas and background

All three tracks are designed to be open-ended — the notebooks and guides give you a starting point, but the best projects will go in unexpected directions.

For detailed setup instructions, RunPod usage, budget info, and environment guides, see **[HACKATHON_GUIDE.md](HACKATHON_GUIDE.md)**.
