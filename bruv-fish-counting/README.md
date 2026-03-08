# Track 3: BRUV Fish Counting — Marine Conservation with MigraMar

> **V1 — This document is a starting point, not a recipe.** It provides context and
> ideas for where to begin. The best projects will go in unexpected directions.
> It may also contain errors — trust your own exploration over these suggestions.

## The Challenge

[Baited Remote Underwater Video stations (BRUVs)](https://en.wikipedia.org/wiki/Baited_remote_underwater_video) let marine biologists monitor reef fish without human interference. The hard part: turning hours of video into species counts.

**Your task:** Count the maximum number of *Caranx caballus* (green jack) that appear in a single frame from the BRUV videos. The Kaggle competition has expert estimates you can try to match or surpass.

**Kaggle competition:** https://www.kaggle.com/competitions/marine-conservation-with-migra-mar

## Data

### On Kaggle (CSV labels)

The Kaggle competition data tab has:

| File | Description |
|------|-------------|
| `CumulativeMaxN.csv` | Frame-level species counts with timestamps — the main label file |
| `TimeFirstSeen.csv` | First appearance time of each species |
| `MaxN.csv` | Maximum count per species per video (ground truth leaderboard) |
| `MaxN_testsubmission.csv` | Submission template |

**Key columns in `CumulativeMaxN.csv`:**
- `Filename` — which sub-video (e.g., `LGH020002.MP4`)
- `Frame` — frame number within that sub-video
- `Time (mins)` — time in the *original* (unsegmented) video
- `Family`, `Genus`, `Species` — taxonomic classification
- `Cumulative MaxN` — count of that species at that frame

### On R2 (video files)

The MP4 files are too large for Kaggle (~65 GB total). Download them from the hackathon's R2 bucket.

**2 video series, 18 sub-videos:**

| Series | Sub-videos | Duration each | Description |
|--------|-----------|---------------|-------------|
| Vid 1 (`LGH__0001.MP4`) | 9 files (LGH01–LGH09) | ~11:47 each | First BRUV deployment |
| Vid 2 (`LGH__0002.MP4`) | 9 files (LGH01–LGH09) | ~11:47 each | Second BRUV deployment |

Each sub-video is ~4 GB except the final segments (LGH09) which are shorter.

**Download videos:**
```python
import r2_download as hd
import os

os.environ["R2_ENDPOINT"] = "https://YOUR_ACCOUNT_ID.r2.cloudflarestorage.com"
os.environ["R2_ACCESS_KEY_ID"] = "YOUR_ACCESS_KEY"
os.environ["R2_SECRET_ACCESS_KEY"] = "YOUR_SECRET_KEY"
os.environ["R2_BUCKET"] = "sala-2026-hackathon-data"

client = hd.get_s3_client()
manifest = hd.load_manifest(bucket=os.environ["R2_BUCKET"], s3_client=client, cache_path="manifest.json")

# Download just one sub-video to start (~4 GB)
stats = hd.download_dataset(manifest, dataset_name="bruv-videos", tags=["vid1-sub02"])

# Or download all of vid1 (~33 GB)
stats = hd.download_dataset(manifest, dataset_name="bruv-videos", tags=["vid1"])
```

### Sub-video timing math

The timestamps in the CSVs refer to the **original unsegmented video**. Each sub-video is ~11.783 minutes (707 seconds). To find the right frame in a sub-video:

```
sub-video index = floor(time_mins / 11.783) + 1
local time = time_mins - (sub_video_index - 1) * 11.783
```

For example, a `Time (mins)` of 14.775 corresponds to:
- Sub-video index: floor(14.775 / 11.783) + 1 = 2 → `LGH020002.MP4`
- Local time: 14.775 - 11.783 = 2.992 minutes into that sub-video

## Species in the Data

| Common Name | Scientific Name | Max Count (vid1) | Max Count (vid2) |
|-------------|----------------|----------------:|----------------:|
| Green jack | *Caranx caballus* | 251 | 52 |
| Almaco jack | *Seriola rivoliana* | 11 | — |
| Rainbow runner | *Elagatis bipinnulata* | 4 | — |
| Pilotfish | *Naucrates ductor* | 3 | — |
| Silky shark | *Carcharhinus falciformis* | 2 | — |
| Unicorn filefish | *Aluterus monoceros* | 2 | — |
| Mahi-mahi | *Coryphaena hippurus* | 1 | — |

The primary target is *Caranx caballus* — by far the most abundant species, appearing in large schools.

## Pipeline Overview

The competition describes a multi-step pipeline. You don't need to complete all steps — contributions at any stage are valued.

```
Raw MP4 video
    │
    ▼
1. Frame extraction / video segmentation
    │
    ▼
2. Object detection (fish vs. BRUV apparatus vs. background)
    │
    ▼
3. Species classification (is this fish Caranx caballus?)
    │
    ▼
4. Counting (how many in this frame?)
    │
    ▼
5. MaxN determination (which frame has the most?)
```

## Project Ideas

### Tier 1: Get started (hours)

- **Frame sampling + manual inspection:** Extract 1 frame/sec from a sub-video, visualize alongside the CSV timestamps, confirm you can locate the labeled frames
- **Background subtraction:** The BRUV is stationary — use frame differencing or MOG2 to isolate moving objects (fish)
- **Pre-trained object detection:** Run YOLOv8 or RT-DETR out of the box on extracted frames. These won't know fish species, but they can detect "fish-shaped" objects vs. the BRUV structure

### Tier 2: Build something (a day)

- **Fine-tune a detector:** Use the labeled frames from `CumulativeMaxN.csv` to create bounding box annotations (semi-automated with SAM or Grounding DINO), then fine-tune YOLOv8 on your annotations
- **Tracking-based counting:** Use ByteTrack or BoT-SORT on detected fish to count unique individuals across frames — avoids double-counting the same fish
- **Temporal aggregation:** Instead of single-frame counting, aggregate detections across a sliding window of frames to get more robust MaxN estimates

### Tier 3: Push boundaries (ambitious)

- **Zero-shot with VLMs:** Use a vision-language model (GPT-4V, Gemini, LLaVA) to count fish in frames. Prompt: "How many green jack fish are in this underwater image?"
- **Dense crowd counting:** 251 fish in a frame is a dense counting problem. Adapt crowd counting methods (CSRNet, CAN) to fish schools
- **Active frame selection:** Instead of processing every frame, train a lightweight model to predict which frames are likely to contain peak fish counts, then only run the expensive detector on those

### Practical tips

- **Start with one sub-video.** Don't download all 65 GB on day one. `LGH020002.MP4` has the most action (Caranx caballus first appears there).
- **OpenCV is your friend.** `cv2.VideoCapture` for frame extraction, `cv2.createBackgroundSubtractorMOG2()` for motion detection.
- **Sample rate matters.** At 30fps, an 11-minute video has ~20,000 frames. You probably don't need all of them — start with 1-2 fps.
- **The BRUV bait arm is always visible.** Any detector needs to learn to ignore it. Consider masking it out as a preprocessing step.
