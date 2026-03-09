# RunPod: Notebook Setup & Running Guide

Step-by-step instructions for running hackathon starter notebooks on your team's RunPod pod.

---

## 1. Open JupyterLab

Your team lead will share a JupyterLab URL with you. Open it in your browser — no account or login needed. You'll see a file browser on the left showing `/workspace`.

---

## 2. First-Time Setup (Once Per Pod)

Open a **Terminal** in JupyterLab (File → New → Terminal) and run:

```bash
cd /workspace
git clone https://github.com/SALA-AI-LATAM/hackathon-participants.git
```

Everything in `/workspace` persists across pod stop/restart. You only need to clone once.

---

## 3. Upload Credentials

Organizers will give you a file called `participant-download.env` with R2 storage credentials.

1. In JupyterLab, use the **file browser** on the left to navigate to `/workspace/hackathon-participants/`
2. **Drag and drop** the `participant-download.env` file into that folder

The starter notebooks automatically search for this file in the repo root and parent directories, so placing it here works for all three tracks.

> **Alternative:** If credentials are pre-set as environment variables on your pod (check with `echo $R2_ENDPOINT` in a terminal), you can skip this step — the notebooks detect pre-set variables automatically.

---

## 4. Open & Run a Starter Notebook

In the JupyterLab file browser, navigate to your track's notebook:

| Track | Path |
|---|---|
| Precipitation Nowcasting | `hackathon-participants/precipitation-nowcasting/precipitation_nowcasting.ipynb` |
| Marine Acoustic Monitoring | `hackathon-participants/marine-acoustic-monitoring/acoustic_explorer.ipynb` |
| BRUV Fish Counting | `hackathon-participants/bruv-fish-counting/bruv_explorer.ipynb` |

Double-click the notebook to open it, then run cells from the top:

1. **Install cell** — installs Python packages (runs once per pod start)
2. **Credentials cell** — auto-detects your `participant-download.env` file
3. **Download cell** — downloads your track's dataset to `/workspace/hackathon_data/`

Downloads support **resume** — if interrupted, re-run the cell and it picks up where it left off. Data persists in `/workspace`, so you only download once.

---

## 5. After a Pod Restart

When you stop and restart your pod:

- Your **data** (`/workspace/`) and **code** are preserved (backed by a persistent network volume)
- You **do** need to re-run the pip install cell (installed packages don't persist)
- You do **not** need to re-download data or re-upload credentials

Just open your notebook and run from the top — the download cell will skip files that are already present.

---

## 6. If You See a Migration Prompt

RunPod may occasionally need to migrate your pod to a different server (e.g., when restarting a stopped pod whose GPU was reassigned). If you see a migration prompt:

1. Choose **"Automatically migrate Pod data"**
2. Wait for the migration to complete — this may take a few minutes
3. Your data and credentials in `/workspace` are preserved (they live on a network volume)
4. **Immediately notify organizers via WhatsApp with your new pod ID** — the budget watchdog tracks pods by ID, and an unregistered pod will be auto-stopped
5. Share the new JupyterLab URL with your team (it changes after migration)
6. Re-run the pip install cell in your notebook (installed packages don't persist across migrations)

---

## 7. Pulling Updates

If the hackathon repo is updated during the event:

```bash
cd /workspace/hackathon-participants
git pull
```

**Important:** If you have a notebook open in JupyterLab, it won't refresh automatically after `git pull`. Close the notebook tab and reopen it from the file browser to see changes.

---

## Tips

- **Stop your pod** when your team is done working — it bills by the second
- **Use Git** for your team's code — push before stopping the pod, pull when restarting
- **BRUV track:** Start with one sub-video (~4 GB) rather than the full 65 GB dataset
- **Multiple teammates:** Everyone can have JupyterLab open at the same time via the shared URL. Coordinate who's editing what to avoid conflicts
