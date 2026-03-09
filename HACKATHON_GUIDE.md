# SALA Hackathon 2026 Instructions
**Participant Guide**

---

## 1. Overview

During this hackathon you'll work in teams to build machine learning solutions for real-world problems from the Galápagos Islands. There are **three tracks** to choose from:

| Track | Dataset | Size | What you'll do |
|---|---|---|---|
| 1. Precipitation Nowcasting | Weather station time series | ~2.4 GB | Predict heavy rain at 3/6/12h horizons |
| 2. Marine Acoustic Monitoring | Underwater audio (SoundTrap hydrophones) | ~7.3 GB | Build AI pipelines for unlabeled ocean sounds |
| 3. BRUV Fish Counting | Underwater video (BRUVs) | ~4 GB per sub-video | Count fish species — [Kaggle competition](https://www.kaggle.com/competitions/marine-conservation-with-migra-mar) |

You have three computing environments available to you:

| Environment | GPU | Cost | Best for |
|---|---|---|---|
| Your laptop | None (usually) | Free | Data exploration, writing code, light preprocessing |
| Google Colab | None (free tier) | Free | Prototyping, quick experiments, running starter notebooks |
| RunPod | RTX 4090 (or assigned) | Team budget | Full training runs, large-scale experiments |

The key idea: develop and iterate on your laptop and Colab (free), and use RunPod only when you need serious GPU power. This stretches your team's GPU budget as far as possible.

**What you'll use:**

- **hackathon-participants repo** ([GitHub](https://github.com/SALA-AI-LATAM/hackathon-participants)) — starter notebooks, READMEs, and the `r2_download.py` helper for downloading data
- **Git / GitHub** — your team's code should live in a GitHub repo, which keeps everything portable across environments
- **Cloudflare R2** — where all datasets are stored. The starter notebooks handle downloads for you via `r2_download.py`
- **WhatsApp** — for team coordination and communication with organizers and domain experts

**Hackathon Space:** The hackathon will take place on the second floor of the *(location TBD)*.

---

## 2. Team Setup

### 2.1 Designate your Team Lead (GPU liaison)

Each team designates one person as the GPU liaison ("Dev"). This person is the only one who needs a RunPod account. Their responsibilities:

- Start and stop the team's GPU pod
- Share the JupyterLab URL with the rest of the team
- Communicate with organizers about GPU-related requests: Your team may choose to switch to a pod with different GPUs or may need to migrate pods when re-starting them.

Everyone else on the team does **NOT** need a RunPod account. You'll access the GPU pod through a JupyterLab link that your Dev shares — it opens in any web browser.

### 2.2 Dev: accept your RunPod invite

After your team designates a Dev, let the organizers know. We'll send you an invite link via WhatsApp.

1. Click the invite link and select **Join Team**
2. Once joined, switch to the team account using the dropdown in the top-right corner of the RunPod console (this is easy to miss — by default you'll be on your personal dashboard)
3. Navigate to **Pods** in the left sidebar — you'll see your team's pod (e.g., `teamA-pod`)

### 2.3 Create your team's GitHub repo

One team member should create a GitHub repository for your team's code. Everyone clones it:

```bash
git clone https://github.com/your-team/repo.git
```

Add a `.gitignore` from the start that excludes `hackathon_data/`, `checkpoints/`, `.env`, `*.pt`, `*.wav`, `*.MP4`, and other large or sensitive files.

### 2.4 Join your WhatsApp groups

You'll be added to two groups:

- **Team Leads + Organizers chat** — your team lead + hackathon organizers. Use for: requesting GPU changes, reporting pod issues, budget questions. We encourage you to make your own separate chats to coordinate with your team about your project.
- **Track chat** — all teams working on the same track + domain experts. Use for: questions about the data (schema, meaning of columns, edge cases), problem formulation, and evaluation criteria. This chat is shared across all teams on a track, so everyone benefits from every question and answer.

Domain expert availability will be shared per track. Please calibrate your expectations around their schedules — they're volunteering their expertise.

---

## 3. Getting Started with Code

### 3.1 Open a starter notebook

The fastest way to get started is to open a track's starter notebook directly in Google Colab from the [hackathon-participants repo](https://github.com/SALA-AI-LATAM/hackathon-participants):

| Track | Starter Notebook |
|---|---|
| Precipitation Nowcasting | `precipitation-nowcasting/precipitation_nowcasting.ipynb` |
| Marine Acoustic Monitoring | `marine-accoustic-monitoring/acoustic_explorer.ipynb` |
| BRUV Fish Counting | `bruv-fish-counting/bruv_explorer.ipynb` |

Each notebook includes setup cells at the top that handle installing dependencies, downloading `r2_download.py`, and pulling the dataset. You can also use `data_download.ipynb` for a standalone download flow.

### 3.2 Configure R2 credentials

Organizers will provide a `participant-download.env` file with R2 credentials. Each starter notebook has a credentials cell that supports two options:

**Option A (recommended):** Upload `participant-download.env` to your Colab file panel (drag and drop), then run the credentials cell. It auto-detects the file and loads the values.

**Option B:** Edit the placeholder values directly in the credentials cell:

```python
os.environ["R2_ENDPOINT"] = "https://..."
os.environ["R2_ACCESS_KEY_ID"] = "YOUR_ACCESS_KEY"
os.environ["R2_SECRET_ACCESS_KEY"] = "YOUR_SECRET_KEY"
os.environ["R2_BUCKET"] = "sala-2026-hackathon-data"
```

- **On RunPod:** Credentials may be pre-set as environment variables on your pod. Check before filling them in.
- **On your laptop:** Place `participant-download.env` in your working directory, or create a `.env` file (gitignored).
- **On Google Colab:** Upload the `.env` file or paste credentials directly in the cell.

### 3.3 Download your track's data

The setup cells in each starter notebook handle this automatically:

```python
import r2_download as hd

client = hd.get_s3_client()
manifest = hd.load_manifest(
    bucket=os.environ["R2_BUCKET"], s3_client=client, cache_path="manifest.json"
)
hd.summarize_manifest(manifest)  # shows all available datasets with sizes

# Download your track's dataset
stats = hd.download_dataset(manifest, dataset_name="precipitation-nowcasting")
# Or: "marine-acoustic-core", "bruv-videos"

# For BRUV, start with just one sub-video (~4 GB) instead of the full 65 GB:
stats = hd.download_dataset(manifest, dataset_name="bruv-videos", tags=["vid2-sub02"])
```

Downloads support **resume** — if interrupted, re-run the cell and it picks up where it left off.

### 3.4 Explore your track

After downloading, open the rest of the starter notebook. Each track's notebook walks you through:
- Loading and exploring the data
- Key visualizations
- Baseline approaches

Read the track's `README.md` for background information, project ideas at multiple difficulty tiers, and practical tips.

---

## 4. Environment Guides

### 4.1 Laptop / local machine

Your laptop is where you'll do most of your development — editing code, exploring data samples, debugging, and writing your training logic.

```bash
# Clone the hackathon repo to get starter notebooks
git clone https://github.com/SALA-AI-LATAM/hackathon-participants.git
cd hackathon-participants

# Install dependencies
pip install boto3 tqdm

# Open a starter notebook and fill in R2 credentials
jupyter notebook precipitation-nowcasting/precipitation_nowcasting.ipynb
```

> **Tips:** You probably don't need the full dataset locally — the precipitation and acoustic tracks are Colab-friendly sizes. Commit and push your code frequently so it's available in other environments.

### 4.2 Google Colab

Colab gives you a free environment for prototyping and quick experiments. The downside is that your session resets when you disconnect, so you'll need to re-download data each time.

To open a starter notebook in Colab:
1. Go to the [hackathon-participants repo](https://github.com/SALA-AI-LATAM/hackathon-participants) on GitHub
2. Navigate to your track's notebook
3. Click the "Open in Colab" badge, or replace `github.com` with `colab.research.google.com/github` in the URL

The notebook's setup cells handle everything: installing packages, downloading `r2_download.py`, and pulling data.

> **Tips:** Colab sessions are ephemeral — data and packages are gone when you disconnect. The precipitation (~2.4 GB) and acoustic (~7.3 GB) datasets are designed for Colab. For BRUV, download just one sub-video at a time. Push code changes to GitHub before your session ends.

### 4.3 RunPod

RunPod is your team's paid GPU resource. Use it for serious training runs — not for writing code or exploring data (do that on your laptop or Colab for free).

#### For the Dev (GPU liaison)

**Starting your pod:**

1. Log into the RunPod console (runpod.io/console)
2. Switch to the team account (top-left dropdown)
3. Go to **Pods** → find your team's pod → click **Start**
4. Wait for status to show **Running** (1–2 minutes)
5. Click **Connect** → **Connect to Jupyter Lab (Port 8888)**
6. Copy the JupyterLab URL and share it with your team in the team chat

**Stopping your pod (IMPORTANT — saves budget):**

1. Go to **Pods** → your team's pod → click **Stop**
2. Your files in `/workspace` are preserved

> Always stop your pod when your team is done working. A running pod bills by the second. There's an automatic safety timer, but don't rely on it.

**Rules:**
- Only start/stop your team's pod — do not touch other teams' pods
- Do not create new pods — unauthorized pods are automatically detected and terminated
- If you need a different GPU or configuration, request it in the team-lead + organizers WhatsApp chat

#### For everyone (using JupyterLab)

Once your Dev starts the pod and shares the URL:

1. Open the JupyterLab URL in your browser — no account needed
2. You'll see a file browser on the left with `/workspace`
3. Clone the hackathon repo if not already there, open your track's notebook

The `/workspace` directory is persistent — your files survive pod stop/restart. Everything outside `/workspace` (like `/root` or `/tmp`) is wiped on stop.

**Suggested `/workspace` layout:**

```
/workspace/
├── hackathon-participants/        # cloned starter repo
│   ├── precipitation-nowcasting/
│   ├── marine-accoustic-monitoring/
│   └── bruv-fish-counting/
├── hackathon_data/                # downloaded datasets (auto-created by r2_download.py)
│   ├── precipitation-nowcasting/
│   ├── marine-acoustic/
│   └── bruv-videos/
├── team-repo/                     # your team's git repo
│   ├── train.py
│   └── experiments/
└── checkpoints/                   # saved model weights
```

**Working with Git on the pod:**

```bash
# First time: clone your repo
cd /workspace
git clone https://github.com/your-team/repo.git team-repo

# Before each session: pull latest code
cd /workspace/team-repo && git pull

# After making changes: push
git add -A && git commit -m "describe changes" && git push
```

> **Multiple teammates on one pod:** Multiple people can have JupyterLab open simultaneously. To avoid conflicts, use Git (pull before editing, push when done), work on separate files, and communicate about who's editing what.

### 4.4 SSH access (optional)

If anyone on your team prefers a terminal or VS Code Remote over JupyterLab, you can set up SSH. This is entirely self-service — no admin involvement needed.

**One-time setup (from JupyterLab terminal on your pod):**

```bash
# Create persistent key storage
mkdir -p /workspace/.ssh-keys

# Add each team member's public key
echo "ssh-ed25519 AAAA... teammate@laptop" >> /workspace/.ssh-keys/authorized_keys
```

Keys are automatically restored on every pod restart — no repeated setup needed. Your team's keys only exist on your pod, not on any other team's pod.

**To activate immediately (without waiting for pod restart):**

```bash
mkdir -p /root/.ssh
cp /workspace/.ssh-keys/authorized_keys /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
```

SSH connection details are available in the RunPod console under your pod's **Connect** menu.

---

## 5. Your Budget

### 5.1 How it works

Each team has a fixed GPU budget (you'll be told the exact amount at the start). This covers RunPod compute time — the clock runs whenever your pod is running, billed by the second.

| Pod state | GPU billing | Storage billing |
|---|---|---|
| Running | Yes (e.g., ~$0.59/hr for RTX 4090) | Included |
| Stopped | No | Tiny (~$0.20/GB/month, negligible) |

Data downloads from R2 are always free — no egress fees. Download as much as you need.

### 5.2 Budget enforcement

Organizers run a budget watchdog that tracks each team's cumulative spend. If your team exceeds its budget, your pod will be **automatically stopped**. The watchdog also enforces a maximum continuous runtime to catch forgotten pods.

If you think your budget was consumed too quickly or need an exception, message the team WhatsApp chat.

### 5.3 Tips to stretch your budget

- Stop your pod whenever you're not actively running GPU workloads
- Develop on your laptop or Colab — free environments for writing code, debugging, exploring data
- Use the GPU pod only for training — don't use it to browse data or edit code
- Pull only the data you need — start with a subset, scale up when your approach is working
- Use checkpoints — save model state regularly so you can resume if interrupted
- Coordinate with your team — don't leave the pod running while nobody's using it
- If you need a more powerful GPU, you can request a swap — but you'll have fewer total hours

---

## 6. Communication Channels

### 6.1 Team Leads chat (WhatsApp)

- **Members:** your team lead + hackathon organizers
- **Use for:** requesting GPU configuration changes, reporting pod issues, budget questions, and general coordination with organizers.

### 6.2 Track chat (WhatsApp)

- **Members:** Team leads for all teams working on the same track + domain experts + organizers
- **Use for:** questions about the data (schema, column meanings, edge cases, data quality), problem formulation, and evaluation criteria. This chat is shared, so everyone benefits from every question and answer.
- **Not for:** team-internal strategy discussions (use your own team channels for that).

---

## 7. Requesting Changes

If your team needs a change to your GPU setup, message the team lead WhatsApp chat with your request. Examples:

- **GPU swap:** "We'd like to upgrade to an A100 for our final training run." We'll create a new pod with the requested GPU and attach your existing data volume. A more powerful GPU costs more per hour, so your total available hours will decrease.
- **More disk space:** if you're running out of room in `/workspace`.
- **Troubleshooting:** pod won't start, Jupyter isn't loading, SSH not working, etc.

Please do not create new pods yourself. Unauthorized pods are automatically detected and terminated by our monitoring system.

---

## 8. Final Submission

On the final day:

1. Push your final code to your team's GitHub repository
2. Prepare your presentation — slides should cover your approach, results, and any interesting findings
3. Upload your slides by re-submitting your team registration form. 
4. Present! Each team will give a short presentation to the judges and other teams

Make sure your GitHub repo is clean and includes a README explaining how to reproduce your results.

---

## 9. Quick Reference

### Key Python API (`r2_download.py`)

```python
import r2_download as hd

# Browse available datasets
client = hd.get_s3_client()
manifest = hd.load_manifest(bucket=os.environ["R2_BUCKET"], s3_client=client, cache_path="manifest.json")
hd.summarize_manifest(manifest)

# Download a full dataset
hd.download_dataset(manifest, dataset_name="precipitation-nowcasting")

# Download a filtered subset (e.g., one BRUV sub-video)
hd.download_dataset(manifest, dataset_name="bruv-videos", tags=["vid2-sub02"])

# List shards with metadata
shards = hd.list_shards(manifest, dataset="marine-acoustic-core")
```

### Environment cheat sheet

| | Laptop | Colab | RunPod |
|---|---|---|---|
| Get `r2_download.py` | Already in cloned repo | Auto-downloaded by notebook | Already in cloned repo |
| Data directory | `./hackathon_data/` | `/content/hackathon_data/` | `/workspace/hackathon_data/` |
| R2 credentials | `.env` file or set in notebook | Set in notebook cell | Pre-set (env vars) or set in notebook |
| Data persists? | Yes | No (ephemeral) | Yes (in `/workspace`) |
| GPU | None (usually) | None (free tier) | Paid (from budget) |
| Git | Normal | `!git clone` in cell | `git clone` in terminal |

### Who to contact

| Question | Where to ask |
|---|---|
| Pod issues, GPU requests, budget | Team WhatsApp chat |
| Data questions, problem formulation | Track WhatsApp chat |
| Account access issues | Message Rudy directly |
| Emergency | Message Rudy directly |
