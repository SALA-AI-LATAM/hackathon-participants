# Marine Acoustic Monitoring — Hackathon Guide

## Starter Notebook

**`acoustic_explorer.ipynb`** — load underwater audio, plot spectrograms, and listen to clips in your browser.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SALA-AI-LATAM/hackathon-participants/blob/main/marine-accoustic-monitoring/acoustic_explorer.ipynb)

> **This document is a starting point, not a recipe.** It may contain errors and is deliberately open-ended. The best hackathon projects will go in directions we didn't anticipate.

## Overview

The Bay of San Cristóbal in the Galápagos is home to dolphins, sea lions, and possibly humpback whales — but it's also one of the busiest maritime traffic areas in the archipelago. The Galápagos Science Center has deployed underwater microphones (hydrophones) to continuously record what's happening beneath the surface.

You have access to these underwater audio recordings. **This data has no labels** — nobody has gone through and annotated what sounds are in these files. Your challenge is to build tools that help marine biologists make sense of it.

### Important: biological content is unverified

The domain experts have not yet confirmed the presence of marine animal sounds (dolphins, whales, sea lions) in these specific recordings. The data clearly contains boat noise, ambient ocean sounds, and various transient signals — but animal vocalizations may be sparse or absent.

**What this means for your project:** This track has two viable directions:

1. **Build tools (code-focused):** Create pipelines that detect and classify underwater sounds — even if this particular dataset turns out to lack animal calls, the tools you build are valuable for future deployments where animals are confirmed to be present.

2. **Propose research directions (research-focused):** Work with the domain experts to understand open problems in underwater acoustics for the Galápagos. Do literature review, explore what methods exist, and propose a research roadmap. Projects taking this path will be judged on the quality of their research and proposals, not on code.

Both directions are equally valid. Talk to the domain experts early to figure out which approach fits your team.

### Core vs. Full Dataset

The full dataset is ~57 GB (926 WAV files, ~97 hours of audio). For Colab and quick starts, download the **core** subset:

| Subset | Size | Files | Audio | What's included |
|--------|------|-------|-------|-----------------|
| **`marine-acoustic-core`** | ~7.3 GB | 123 WAVs | ~12 hours | 100 Pilot + 20 unit-6478 + 3 unit-5783, spread across time |
| `marine-acoustic-full` *(not yet uploaded)* | ~57 GB | 926 WAVs | ~97 hours | Everything — all 3 units, all recordings |

```python
import r2_download as hd

# Core subset (recommended for Colab)
stats = hd.download_dataset(manifest, dataset_name="marine-acoustic-core")

# Full dataset (local/RunPod only — 57 GB, not yet uploaded to R2)
# stats = hd.download_dataset(manifest, dataset_name="marine-acoustic-full")
```

The core subset is enough for all project ideas below.

## Dataset Structure

```
marine-accoustic-monitoring/
├── 5783/                       # Hydrophone unit 5783
│   ├── *.wav                   # 16 recordings, 20 min each, 144 kHz
│   ├── *.log.xml               # 4,166 metadata files (deployment log)
│   └── *.sud                   # 23 compressed recordings (SoundTrap format)
├── 6478/                       # Hydrophone unit 6478
│   ├── *.wav                   # 189 recordings, 10 min each, 96 kHz
│   ├── *.log.xml               # 3,413 metadata files
│   └── *.sud                   # 200 compressed recordings
└── Music_Soundtrap_Pilot/      # Pilot deployment
    └── *.wav                   # 721 recordings, ~5 min each, 48 kHz
```

### Recording Details

| Unit | Sample Rate | WAV Files | Duration per File | Total Audio | Time Range |
|------|------------|-----------|-------------------|-------------|-----------|
| 5783 | 144 kHz | 16 | 20 min | ~5.3 hours | Deployment logs span ~2 years |
| 6478 | 96 kHz | 189 | 10 min | ~31.5 hours | Oct 2022 – Jul 2023 |
| Pilot | 48 kHz | 721 | ~5 min | ~60 hours | Aug 2019, Oct 2020 |

**Sample rate** = how many audio samples per second. Higher rates capture higher-pitched sounds. 96 kHz can record sounds up to 48 kHz (half the sample rate).

### File Naming

- **5783/6478**: `{unit_id}.{YYMMDDHHMMSS}.{ext}` — e.g., `6478.230723151251.wav` = unit 6478, recorded 2023-07-23 at 15:12:51
- **Pilot**: `{YYMMDD}_{sequence}.wav` — e.g., `190806_3755.wav` = 2019-08-06, sequence 3755

### XML Metadata

Each `.log.xml` file contains recording metadata: timestamps, sample rate, hardware ID, battery voltage, water temperature, and gain settings.

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

**Note:** animal sounds have not been confirmed in this dataset (see note above).

- **Confirmed sounds:** boat engines (low rumbling, ~100 Hz–1 kHz), port activity, waves, wind, rain
- **Possible animal sounds (unverified):** dolphin whistles (5–20 kHz), echolocation clicks (short, sharp pulses), sea lion barks, snapping shrimp (crackling noise, 2–20 kHz), fish choruses (low hums, < 2 kHz)

### Most Data is in .sud Format

Only some recordings have been converted to WAV. The `.sud` files are a compressed format that can be converted using Ocean Instruments' SoundTrap Host software. The `.log.xml` files cover ALL recordings, including those still in `.sud` format.

## Quick Start

### Helper Module

`acoustic_data.py` handles file discovery, timestamp parsing, audio loading, filtering, and spectrogram plotting:

```python
import acoustic_data as hd

# See what's available
DATA_DIR = "./data"  # adjust to wherever your data lives
recs = hd.inventory(DATA_DIR)
```

### Loading Audio

```python
# Load 30 seconds from a recording
recs = hd.list_recordings(DATA_DIR, unit="6478")
audio, sr = hd.load_audio(recs[0]["path"], duration_s=30.0)

# Filter out low-frequency noise (always do this before analysis)
audio = hd.highpass_filter(audio, sr, cutoff_hz=50)
```

### Spectrograms

A spectrogram shows how the frequency content of audio changes over time — the x-axis is time, y-axis is frequency, and color is intensity.

```python
# Full spectrogram
fig, ax = hd.plot_spectrogram(audio, sr, title="My recording")

# Split into frequency bands: LOW (boats/fish), MID (shrimp/dolphins), HIGH (echolocation)
fig, axes = hd.plot_spectrogram_bands(audio, sr)
```

### Listening in Notebooks

```python
# Play a 10-second clip in your browser
hd.listen(recs[0]["path"], duration_s=10)
```

### Example Spectrograms

Pre-generated spectrograms for each unit are in `examples/`:
- `spectrogram_6478_full.png` / `spectrogram_6478_bands.png` — 96 kHz unit
- `spectrogram_pilot_full.png` / `spectrogram_pilot_bands.png` — 48 kHz pilot
- `spectrogram_5783_full.png` / `spectrogram_5783_bands.png` — 144 kHz unit

### Loading Audio Without the Helper

```python
import wave
import numpy as np

with wave.open('6478/6478.230723151251.wav', 'rb') as wf:
    sr = wf.getframerate()
    raw = wf.readframes(wf.getnframes())

# Convert raw bytes to floating point values between -1 and 1
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
```

---

## Project Ideas

This dataset is open-ended. Below are three tiers of ideas, from "no labels needed" to "build a classifier from scratch." You can pick one or combine elements across tiers.

### Tier 1: Measure the Soundscape (No Labels Needed)

**Goal:** Characterize what the underwater environment sounds like using numerical metrics. No labels required — this is signal processing.

**Key library:** `scikit-maad` (`pip install scikit-maad`) — computes 50+ acoustic indices with built-in plotting.

**What to compute:**

- **Sound levels by frequency band:** Split the audio into LOW (50–2000 Hz, boats and fish), MID (2–10 kHz, shrimp and dolphins), HIGH (>10 kHz). Measure average loudness in each band over time. Use `scipy.signal.welch` to compute power spectra.

- **Acoustic entropy:** How "diverse" does the soundscape sound? Shannon entropy of the audio signal — high entropy means lots of different sound types happening. Use `maad.features.temporal_entropy()`.

- **Biological vs. human noise ratio (NDSI):** A standard index that compares biological sound levels to human-generated noise. **Important caveat:** the default frequency bands in `scikit-maad` are designed for forests (birds at 2–8 kHz, human noise at 1–2 kHz). Underwater, these are wrong — boat noise is at 100–1000 Hz and shrimp/dolphins are at 2–20 kHz. You'll need to set custom frequency ranges.

**Patterns to look for:**
- Does the soundscape change between day and night? (e.g., more shrimp crackling at night, more boats during the day)
- Are there patterns by hour of day?
- Do different recording units (locations) sound different?

**Visualizations:**
- Spectrograms (standard and zoomed by frequency band)
- Heatmaps of acoustic indices by hour of day
- Time series of sound levels over days/weeks

**Key insight:** Most acoustic ecology tools were built for land-based recordings (birds in forests). The frequency ranges are different underwater — don't use library defaults without thinking about what they mean in an ocean context.

**References:**
- Ulloa et al. (2021) — scikit-maad paper, Methods in Ecology and Evolution
- Bradfer-Lawrence et al. (2025) — The Acoustic Index User's Guide

### Tier 2: Use Pretrained Models (Minimal Labels)

**Goal:** Apply existing AI models (trained on other audio datasets) to these recordings. You don't need to train anything from scratch.

**Models to try (easiest first):**

1. **Google/NOAA Humpback Whale Model** — A simple yes/no detector for humpback whale song. No labels needed at all. Resample your audio to 10 kHz and run it. Even a negative result ("no humpbacks detected") is useful information.

2. **Perch 2.0** (Google DeepMind) — A general-purpose audio model trained on 14,500+ species. Despite being trained mostly on land animals, it works surprisingly well on marine audio. Extract "embeddings" (numerical summaries) from your audio clips, then cluster them to discover different sound types. Install: `pip install opensoundscape bioacoustics-model-zoo tensorflow tensorflow-hub`.

3. **BirdNET embeddings** — Trained on 6,500+ bird species. Don't use it to identify birds — use it as a feature extractor. The patterns it learned about animal sounds transfer to marine audio. Install via `opensoundscape`.

4. **AVES** (Earth Species Project) — A PyTorch model for animal sounds. Limited to lower frequencies (up to 8 kHz), so it can't hear dolphin whistles, but works for whale songs, sea lion barks, and boat noise.

**Watch out for sample rate mismatches:** These models expect specific sample rates (10–48 kHz). Our recordings are at 48–144 kHz. You'll need to resample:

```python
import librosa
audio_resampled = librosa.resample(audio, orig_sr=96000, target_sr=32000)
```

**Suggested approach:**
1. Run the humpback detector on all recordings (immediate results, no labels)
2. Extract Perch embeddings for 5-second clips across the dataset
3. Use UMAP (dimensionality reduction) + HDBSCAN (clustering) to group similar-sounding clips
4. Listen to examples from each cluster to understand what the groups represent

**References:**
- Burns et al. (2025) — Perch 2.0 marine transfer learning
- OpenSoundscape documentation — tutorials for BirdNET and Perch

### Tier 3: Build a Classifier with Active Learning

**Goal:** Start from zero labels and iteratively build a useful sound classifier.

**The idea:** With pretrained embeddings (from Tier 2), you only need **8–16 labeled examples per category** to train a simple classifier that works reasonably well. For ~7 sound categories, that's about 60–100 labels — roughly 30–60 minutes of listening and labeling.

**Workflow:**

1. **Explore without labels (1 hour):** Compute embeddings for all clips → cluster them → listen to examples from each cluster → figure out what categories exist in the data.

2. **Label a small seed set (1 hour):** Pick 10–20 examples per category. Possible categories: boat noise, ambient/silence, unknown biological, rain, shrimp crackling, etc. (Adjust based on what you actually hear.)

3. **Train a simple classifier (30 min):** Logistic regression on the embeddings. See how well it separates the categories.

4. **Improve with active learning (30 min):** Find clips where the model is most uncertain → label those → retrain. Repeat. Each round should improve accuracy.

5. **Try more advanced approaches (remaining time):**
   - Fine-tune a CNN on spectrograms (treat them as images)
   - Use prototypical networks (classify by similarity to category averages)
   - Combine embeddings from multiple pretrained models

**Labeling tools:**
- **Whombat** (`pip install whombat`): Browser-based audio annotation tool built for bioacoustics
- **In a notebook:** Display spectrogram images in a grid, type labels manually. Simple but effective.
- **Raven Lite** (free, from Cornell): Desktop app for viewing and annotating spectrograms

**Tip:** Don't try to train models from scratch during the hackathon — use pretrained embeddings instead. Training a new audio model takes hours/days, but training a classifier on top of pretrained embeddings takes minutes.

### Bonus: Project CETI — Generative Audio Models

Orr Paradise (co-author of WhAM, Project CETI) will be at the hackathon to answer questions.

**WhAM** (Whale Acoustics Model, NeurIPS 2025) takes a model originally trained to generate music and adapts it for whale sounds. The key insight: music and animal vocalizations both have rhythm, pitch patterns, and temporal structure, so a music model is a great starting point.

**How it works:**
1. Audio is converted to discrete tokens (like words in a sentence) using an audio codec
2. A transformer model learns to predict masked (hidden) tokens — similar to how GPT learns language
3. The model is adapted to whale sounds using LoRA (a lightweight fine-tuning technique)

**Hackathon ideas:**
1. **Extract embeddings (1–2 hours):** Run WhAM on Galápagos audio and cluster the results. Do different sound types naturally separate?
2. **Fine-tune for detection (half a day):** Adapt the model to detect specific sounds using a small labeled set
3. **Generate synthetic training data (half a day):** Use WhAM to create synthetic versions of rare sounds for data augmentation

**Setup:**
```bash
pip install git+https://github.com/Project-CETI/wham
# WhAM uses 44.1 kHz — resample your audio with librosa.resample
```

**References:**
- Paradise et al. (2025) — WhAM: Whale Acoustics Model. NeurIPS 2025.
- Project CETI: https://www.projectceti.org/ | Code: https://github.com/Project-CETI/wham

---

### Combining Ideas Across Tiers

- **Acoustic indices + classification:** Compute soundscape metrics on clips that your classifier labeled. Do certain metrics predict certain sound types?
- **Temporal patterns:** Plot detection rates by hour of day and across months. When is boat traffic heaviest? Do sound patterns differ between weekdays and weekends?
- **Noise impact:** Compare human noise levels with biological activity. Does more boat traffic correlate with less biological sound?

## Practical Notes

### Memory

A 10-minute WAV at 96 kHz ≈ 115 MB. Loading many files at once can run out of RAM. Process files one at a time or in chunks:
```python
import soundfile as sf
for block in sf.blocks('long_recording.wav', blocksize=sr * 60):  # 1-minute chunks
    # process each chunk
    pass
```

### Resampling

Most pretrained models need lower sample rates than our recordings. Use librosa:
```python
import librosa
audio_32k = librosa.resample(audio, orig_sr=96000, target_sr=32000)
```

### Calibration

These recordings don't have calibration data (we don't know the exact sensitivity of the hydrophone). This means all sound levels are relative — you can compare "louder vs. quieter" within the dataset, but not convert to absolute physical units. Ratio-based metrics (entropy, NDSI) still work fine.

### Filtering

Always filter out very low frequencies before analysis — they're just electrical noise from the sensor:
```python
from scipy.signal import butter, sosfilt
sos = butter(4, 50, btype='highpass', fs=sr, output='sos')
audio_filtered = sosfilt(sos, audio)
```

## Long-Term Goals

This hackathon is part of a larger effort to build a permanent underwater acoustic monitoring system for the Galápagos Marine Reserve. Long-term goals include:
- A multi-year database of underwater recordings
- Automated tools to detect and classify sounds
- A catalog of marine animal sounds for the region
- Understanding how human noise affects marine life
- Practical recommendations for reducing underwater noise pollution

Your hackathon work could become the foundation for these tools.
