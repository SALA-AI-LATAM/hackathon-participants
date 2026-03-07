# Marine Acoustic Monitoring — Hackathon Guide

> **This document is a starting point, not a recipe.** It collects ideas, tools, and references to help you orient yourself — but it's V1, may contain errors, and is deliberately open-ended. Treat it as context to riff on, not instructions to follow step by step. The best hackathon projects will go in directions we didn't anticipate.

## Overview

The Bay of San Cristóbal in the Galápagos is a critical habitat for marine mammals (dolphins, sea lions, possibly humpback whales) and also one of the highest maritime traffic areas in the archipelago. The Galápagos Science Center is building a Passive Acoustic Monitoring (PAM) system to continuously record the underwater soundscape.

You have access to underwater audio from SoundTrap ST300 hydrophones deployed in the bay. Unlike the precipitation dataset, **this data has no labels** — nobody has annotated what sounds are in these recordings. Your challenge is to build AI pipelines that help marine biologists analyze this acoustic data.

### Core vs. Full Dataset

The full dataset is ~57 GB (926 WAV files, ~97 hours of audio). For Colab and quick starts, download the **core** subset:

| Subset | Size | Files | Audio | What's included |
|--------|------|-------|-------|-----------------|
| **`marine-acoustic-core`** | ~7.3 GB | 123 WAVs | ~12 hours | 100 Pilot + 20 unit-6478 + 3 unit-5783, temporally diverse |
| **`marine-acoustic-full`** | ~57 GB | 926 WAVs | ~97 hours | Everything — all 3 units, all recordings |

```python
import r2_download as hd

# Core subset (recommended for Colab)
stats = hd.download_dataset(manifest, dataset_name="marine-acoustic-core")

# Full dataset (local/RunPod only — 57 GB)
# stats = hd.download_dataset(manifest, dataset_name="marine-acoustic-full")
```

The core subset has enough data for all project tiers. The full dataset adds more temporal coverage for diel/seasonal analysis.

## Dataset Structure

```
marine-accoustic-monitoring/
├── 5783/                       # SoundTrap unit 5783
│   ├── *.wav                   # 16 recordings, 20 min each, 144 kHz
│   ├── *.log.xml               # 4,166 metadata files (full deployment log)
│   └── *.sud                   # 23 compressed recordings (SoundTrap format)
├── 6478/                       # SoundTrap unit 6478
│   ├── *.wav                   # 189 recordings, 10 min each, 96 kHz
│   ├── *.log.xml               # 3,413 metadata files
│   └── *.sud                   # 200 compressed recordings
└── Music_Soundtrap_Pilot/      # Pilot deployment
    └── *.wav                   # 721 recordings, ~5 min each, 48 kHz
```

### Recording Details

| Unit | Sample Rate | WAV Files | Duration per File | Total Audio | Time Range (UTC) |
|------|------------|-----------|-------------------|-------------|-----------------|
| 5783 | 144 kHz | 16 | 20 min | ~5.3 hours | Deployment logs span ~2 years |
| 6478 | 96 kHz | 189 | 10 min | ~31.5 hours | Oct 2022 – Jul 2023 |
| Pilot | 48 kHz | 721 | ~5 min | ~60 hours | Aug 2019, Oct 2020 |

### File Naming Convention

- **5783/6478**: `{unit_id}.{YYMMDDHHMMSS}.{ext}` — e.g., `6478.230723151251.wav` = unit 6478, 2023-07-23 at 15:12:51
- **Pilot**: `{YYMMDD}_{sequence}.wav` — e.g., `190806_3755.wav` = 2019-08-06, sequence 3755

### XML Metadata

Each `.log.xml` contains deployment metadata including:
- UTC start/stop timestamps
- Sample rate configuration
- Hardware ID, battery voltage, water temperature
- Gain settings

```python
import xml.etree.ElementTree as ET
tree = ET.parse('6478/6478.230723151251.log.xml')
root = tree.getroot()
for pe in root.findall('.//PROC_EVENT'):
    for child in pe:
        if 'SamplingStartTimeUTC' in child.attrib:
            print(child.attrib['SamplingStartTimeUTC'])
```

### What You'll Hear

The recordings contain a mix of:
- **Biological sounds**: dolphin whistles (5–20 kHz), echolocation clicks (broadband, up to 130 kHz), sea lion barks (200 Hz–4 kHz), snapping shrimp (broadband clicks, dominant 2–20 kHz), fish choruses (< 2 kHz)
- **Anthropogenic noise**: boat engines (~100 Hz–1 kHz tonal + broadband), port operations
- **Abiotic sounds**: waves, wind, rain (broadband impulsive)

### Important: Most Data is in .sud Format

Only a fraction of recordings have been extracted to WAV. The `.sud` files are SoundTrap's compressed format and contain additional audio that could be extracted with Ocean Instruments' SoundTrap Host software. The `.log.xml` files provide metadata for ALL recordings, including those still in `.sud` format.

## Quick Start

### Helper Module

`acoustic_data.py` handles file discovery, timestamp parsing, audio loading, filtering, and spectrogram visualization. Import it in your notebook:

```python
import acoustic_data as hd

# See what's available
DATA_DIR = "./data"  # adjust to wherever your data lives
recs = hd.inventory(DATA_DIR)
# Unit       WAVs  Sample Rate Time Range
# 5783         16     144000 Hz  2004-01-22 → 2004-01-23
# 6478        189      96000 Hz  2022-10-19 → 2023-07-27
# pilot       721      48000 Hz  2019-08-06 → 2020-10-10
```

### Loading Audio

```python
# Load 30 seconds from a recording
recs = hd.list_recordings(DATA_DIR, unit="6478")
audio, sr = hd.load_audio(recs[0]["path"], duration_s=30.0)

# Always highpass filter to remove DC offset and self-noise
audio = hd.highpass_filter(audio, sr, cutoff_hz=50)
```

### Spectrograms

```python
# Full-band spectrogram
fig, ax = hd.plot_spectrogram(audio, sr, title="My recording")

# Band-split: LOW (fish/boats), MID (shrimp/dolphins), HIGH (echolocation)
fig, axes = hd.plot_spectrogram_bands(audio, sr)
```

### Listening in Notebooks

```python
# Play a 10-second clip (auto-resamples for browser playback)
hd.listen(recs[0]["path"], duration_s=10)
```

### Example Spectrograms

Pre-generated spectrograms for each unit are in `examples/`:
- `spectrogram_6478_full.png` / `spectrogram_6478_bands.png` — 96 kHz, shows broadband transients and biological activity
- `spectrogram_pilot_full.png` / `spectrogram_pilot_bands.png` — 48 kHz, broadband ambient (shrimp-dominated)
- `spectrogram_5783_full.png` / `spectrogram_5783_bands.png` — 144 kHz, includes echolocation-range HIGH band

### Raw Loading (without helper)

```python
import wave
import numpy as np

with wave.open('6478/6478.230723151251.wav', 'rb') as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())

audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
```

---

## Project Ideas

This dataset is intentionally open-ended. Below are three tiers of project ideas, ranging from "no labels needed" to "build a classifier from scratch." Teams can pick one focus area or combine elements across tiers.

### Tier 1: Soundscape Ecology — Acoustic Indices (No Labels Needed)

**Goal:** Characterize the underwater soundscape of San Cristóbal Bay using standard acoustic ecology metrics. No labels required — this is pure signal processing.

**Key library:** `scikit-maad` (`pip install scikit-maad`) — provides 50+ acoustic indices via two convenience functions and built-in visualization.

**What to compute:**
- **Band-specific SPL** (Sound Pressure Level): The most reliable marine metric. Split into LOW (50–2000 Hz, fish), MID (2–10 kHz, snapping shrimp), HIGH (>10 kHz, abiotic). Use `scipy.signal.welch` for PSD, integrate over bands.
- **Bioacoustic Index (BI)**: Overall biological sound intensity. Use `maad.features.bioacoustics_index()` — but adjust `flim` for marine (default 2–8 kHz is terrestrial).
- **Acoustic Entropy (H)**: Shannon entropy of amplitude envelope (temporal) and power spectrum (spectral). High entropy = diverse soundscape. Use `maad.features.temporal_entropy()` and `frequency_entropy()`.
- **NDSI** (Normalized Difference Soundscape Index): Ratio of biophony to anthrophony: `(biophony - anthrophony) / (biophony + anthrophony)`. **Warning:** the default frequency bands (anthrophony 1–2 kHz, biophony 2–8 kHz) come from terrestrial ecology where birdsong is 2–8 kHz and human noise is below 2 kHz. Underwater this is inverted — fish vocalizations (biophony) are **below 2 kHz** and would be classified as "anthrophony," while boat engine noise (~100 Hz–1 kHz) is also missed. You need to define marine-appropriate bands, e.g. anthrophony 100–1000 Hz (boats), biophony 2000–20000 Hz (shrimp + dolphins). Pass custom `flim` parameters instead of using library defaults.

**Expected diel patterns to look for:**
- Fish chorusing peaks at **dawn and dusk** (< 2 kHz) — can be 10–20 dB above daytime
- Snapping shrimp activity peaks at **night** (2–20 kHz) — 6–10 dB nocturnal increase
- Boat noise strongest during **daytime** (< 1 kHz)

**Visualizations:**
- Standard and zoomed spectrograms (per frequency band)
- Long-Term Spectral Average (LTSA) — compressed time-frequency view over hours/days
- Diel activity heatmaps — index values by hour-of-day
- PSD percentile plots — the marine ecology standard
- False-color spectrograms (3 indices → RGB channels)

**Key insight for students:** Many terrestrial acoustic indices don't transfer directly to marine. The frequency zonation is inverted — in terrestrial soundscapes, biophony is 2–8 kHz (birdsong) and anthrophony < 2 kHz. In marine, fish biophony is < 2 kHz and invertebrate biophony is 2–20 kHz. Blindly applying terrestrial defaults produces ecologically meaningless results.

**References:**
- Ulloa et al. (2021) — scikit-maad paper, Methods in Ecology and Evolution
- Bradfer-Lawrence et al. (2025) — The Acoustic Index User's Guide
- Dimoff et al. (2021) — ACI limitations on coral reefs
- Haver et al. (2023) — Acoustic indices for marine mammals, Frontiers in Marine Science

### Tier 2: Pretrained Models — Zero-Shot and Few-Shot Detection

**Goal:** Apply existing AI models to detect and classify marine sounds without training from scratch. Leverages transfer learning from models trained on birds and general audio.

**Top models (ranked by hackathon feasibility):**

1. **Perch 2.0** (Google DeepMind) — A bioacoustics foundation model trained on 14,500+ species. Despite being trained almost entirely on terrestrial sounds, it achieves AUC 0.86–0.98 on marine tasks (whales, reef sounds) with just 8–16 labeled examples per class. Install via `pip install opensoundscape bioacoustics-model-zoo tensorflow tensorflow-hub`.

2. **Google/NOAA Humpback Whale Model** — The only truly zero-shot option. Binary detector (humpback song: yes/no) with AUC-ROC 0.992. Requires resampling to 10 kHz. Run it on all recordings to check for humpback whale presence — no labels needed at all.

3. **BirdNET embeddings** — Trained on 6,500+ bird species, but its embeddings transfer surprisingly well to marine audio. Use as a feature extractor via `opensoundscape`, not for direct bird predictions. Native 48 kHz sample rate is convenient.

4. **AVES** (Earth Species Project) — Self-supervised transformer for animal vocalizations. MIT license, PyTorch-native. 16 kHz sample rate limits it to low-frequency sounds (whale songs, sea lion barks, boat noise).

**Critical gotcha — sample rate mismatch:**

| Model | Input SR | Nyquist | Misses from Galápagos data |
|-------|----------|---------|---------------------------|
| Perch 2.0 | 32 kHz | 16 kHz | Dolphin whistles >16 kHz, all echolocation |
| BirdNET | 48 kHz | 24 kHz | Echolocation clicks only |
| AVES | 16 kHz | 8 kHz | Most dolphin whistles, echolocation, some shrimp |
| Humpback model | 10 kHz | 5 kHz | Everything except whale songs and boat noise |

**No pretrained model can process frequencies above ~24 kHz.** For dolphin echolocation click detection (up to 130 kHz), you need traditional signal processing (energy detectors, bandpass filters) or custom models trained on high-sample-rate spectrograms.

**Suggested approach:**
1. Run humpback detector on all data (zero-shot, immediate results)
2. Extract Perch embeddings for all 5-second clips
3. Cluster embeddings with UMAP + HDBSCAN to discover natural sound categories
4. Label 8–16 examples per discovered category
5. Train a linear classifier (logistic regression) on the embeddings
6. Evaluate and iterate with active learning

**References:**
- Burns et al. (2025) — Perch 2.0 marine transfer learning
- Google Research Blog — "How AI trained on birds is surfacing underwater mysteries"
- OpenSoundscape documentation — BirdNET/Perch tutorial

### Tier 3: Active Learning — Build a Classifier from Scratch

**Goal:** Bootstrap a marine sound classifier starting from zero labels, using active learning and few-shot techniques.

**The key insight:** With pretrained embeddings (Perch or BirdNET), you need only **8–16 labeled examples per class** to build a useful classifier (AUC > 0.86). For 7 target sound classes, that's ~56–112 total labels — about 30–60 minutes of annotation work.

**Suggested workflow:**

1. **Unsupervised exploration (1 hour):** Compute Perch embeddings for all clips → UMAP + HDBSCAN clustering → listen to examples from each cluster → identify what each cluster "sounds like"

2. **Seed labeling (1 hour):** Guided by cluster structure, label 10–20 examples per class. Target classes: dolphin whistles, echolocation clicks, sea lion barks, whale song, snapping shrimp, boat noise, ambient/silence.

3. **First classifier (30 min):** Logistic regression on embeddings. Evaluate on held-out set. This is your baseline.

4. **Active learning round 1 — diversity (30 min):** Select unlabeled examples the model is most uncertain about. Label them. Retrain.

5. **Active learning round 2 — uncertainty (30 min):** Repeat with uncertainty sampling. Compare improvement.

6. **Advanced approaches (remaining time):**
   - Prototypical networks (average embeddings per class, classify by nearest prototype)
   - Spectrogram CNN fine-tuning (treat mel-spectrograms as images, fine-tune ResNet)
   - Autoencoder → UMAP → HDBSCAN pipeline for unsupervised vocalization clustering
   - Ensemble across embedding models (Perch + BirdNET + AVES)

**Annotation tools:**
- **Whombat** (`pip install whombat`): Browser-based, ML-assisted, bioacoustics-native. Best standalone tool.
- **Notebook-based**: Display spectrogram grids in Jupyter, type labels. Avoids setup overhead.
- **Raven Lite** (free from Cornell): Desktop spectrogram viewer. Good for exploration, not scriptable.

**What NOT to do during the hackathon:** Don't train self-supervised models from scratch (BYOL, SimCLR) — they need 8–24 hours. Use pretrained embeddings instead. Don't spend time on raw audio I/O and format conversion — pre-process data ahead of time.

**References:**
- DCASE 2024 Task 5 — Few-shot bioacoustic event detection challenge
- Ghani et al. (2023) — BirdNET embeddings for bioacoustic classification
- Sethi et al. (2023) — Autoencoder + UMAP clustering for vocalizations
- Lapp et al. (2023) — OpenSoundscape

### Bonus: Project CETI — Generative Models for Marine Vocalizations

**Special resource:** Orr Paradise (co-author of WhAM, Project CETI) will be available at the hackathon to answer questions and brainstorm ideas with teams interested in generative bioacoustic approaches.

**What is WhAM?** The Whale Acoustics Model (NeurIPS 2025, Paradise et al.) fine-tunes [VampNet](https://github.com/Project-CETI/wham) — a music-pretrained masked acoustic token model — on annotated sperm whale codas. It demonstrates that music generation models make excellent priors for biological vocalizations, since both encode temporal, rhythmic, and tonal structure.

**Architecture at a glance:**
1. **DAC (Descript Audio Codec)** — converts audio to discrete tokens via residual vector quantization
2. **VampNet** — bidirectional transformer doing masked acoustic token prediction (like BERT, but for audio tokens)
3. **LoRA fine-tuning** — parameter-efficient adaptation; keeps pretrained weights frozen

**What WhAM can do:**
- **Classify**: Detect codas (91.3%), classify rhythm types (87.4%), social units (70.5%)
- **Generate**: Synthesize novel codas; can be used for data augmentation
- **Translate**: Map vocalizations from one cetacean species to acoustically resemble sperm whale codas (4 of 12 tested species become indistinguishable by automated metrics)

**Note:** WhAM's weights are trained on sperm whale codas (~1–15 kHz). The Galápagos data has dolphins, sea lions, and potentially humpbacks — direct inference won't give species-specific results. The power is in the *methodology* and the pre-trained *backbone*.

**Hackathon ideas (easy → ambitious):**

1. **Embedding exploration (1–2 hours):** Download WhAM, extract intermediate-layer embeddings from Galápagos audio, cluster with UMAP. Does the acoustic representation space separate biological sounds from boat noise without any labels?

2. **LoRA fine-tuning on dolphin whistles (half a day):** The domain-adapted backbone (before sperm whale species fine-tuning) already captures general audio structure. Apply LoRA on 50–100 labeled dolphin whistle clips to build a dolphin-specific detector.

3. **Acoustic augmentation (half a day):** Use WhAM's masking-based infilling to generate synthetic variants of labeled dolphin whistles. Train a classifier on real + synthetic data — does augmentation help?

4. **Cross-species acoustic translation (ambitious):** Attempt acoustic translation from Galápagos dolphin whistles into the sperm whale coda "style," and back. Compare translated audio to the originals — what structure is preserved?

**Setup:**
```bash
pip install git+https://github.com/Project-CETI/wham
# Requires: torch, audiotools, DAC, VampNet — pre-install in hackathon env
# Sample rate: WhAM uses 44.1 kHz — resample your WAVs with librosa.resample
```

**References:**
- Paradise et al. (2025) — WhAM: Whale Acoustics Model. NeurIPS 2025.
- Garcia-Aguilar et al. (2023) — VampNet: Music Generation via Masked Acoustic Token Modeling
- Project CETI: https://www.projectceti.org/ | Code: https://github.com/Project-CETI/wham

---

### Cross-Tier Ideas

- **Acoustic indices + classification**: Compute indices on segments classified by your Tier 2/3 model. Do biological sounds correlate with certain index values? Can indices serve as weak labels?
- **Temporal ecology**: Plot detection rates across time-of-day and across months. Do dolphin detections peak at certain times? Does boat noise increase on weekends?
- **Noise impact assessment**: Correlate anthropogenic noise levels (SPL in boat-noise band) with biological activity indices. Does boat traffic suppress biological sound production?
- **Multi-scale pipeline**: Use acoustic indices for coarse filtering (find "interesting" time periods), then run pretrained models on those periods for fine-grained classification.

## Practical Notes

### Memory Management
A 10-minute WAV at 96 kHz, 16-bit mono ≈ 115 MB. Loading many files simultaneously can exhaust RAM. Process files in chunks:
```python
import soundfile as sf
for block in sf.blocks('long_recording.wav', blocksize=sr * 60):  # 1-minute chunks
    # process each chunk
    pass
```

### Resampling for Pretrained Models
Most models expect lower sample rates. Use `librosa.resample`:
```python
import librosa
audio_32k = librosa.resample(audio, orig_sr=96000, target_sr=32000)
```

### Calibration
These recordings are **uncalibrated** (no hydrophone sensitivity metadata). All SPL values are relative (dB full-scale). Unitless indices (ACI, entropy, NDSI) work fine without calibration. For calibrated SPL, you'd need the hydrophone sensitivity in dB re 1V/µPa.

### Filtering
Always highpass filter before analysis to remove DC offset and low-frequency self-noise:
```python
from scipy.signal import butter, sosfilt
sos = butter(4, 50, btype='highpass', fs=sr, output='sos')
audio_filtered = sosfilt(sos, audio)
```

## Long-Term Project Goals

This hackathon contributes to a larger effort to build a permanent acoustic monitoring system for the Galápagos Marine Reserve. The long-term goals include:
- Multi-year acoustic database of PAM records
- Automated detection, classification, and visualization pipeline
- Catalog of marine mammal vocalizations for the region
- Predictive models of species presence based on acoustic signals
- Assessment of acoustic pollution impacts on marine fauna
- Management recommendations for noise reduction

Your hackathon work could become the foundation for these tools.
