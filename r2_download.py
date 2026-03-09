"""
r2_download.py — Download, verify, and manage hackathon datasets from Cloudflare R2.

Single-file helper module for participants. Works across Colab, Kaggle, RunPod, and local.
Deps: boto3, tqdm (hashlib is stdlib).
"""

import hashlib
import json
import os
from pathlib import Path

import boto3

from tqdm import tqdm


# ============================================================================
# Environment detection
# ============================================================================

def _detect_environment():
    """Detect runtime environment: 'colab', 'kaggle', 'runpod', or 'local'."""
    if "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if "RUNPOD_POD_ID" in os.environ:
        return "runpod"
    return "local"


def _default_data_dir():
    """Return a sensible default download directory for the current environment."""
    env = _detect_environment()
    defaults = {
        "colab": "/content/hackathon_data",
        "kaggle": "/kaggle/working/hackathon_data",
        "runpod": "/workspace/hackathon_data",
        "local": "./hackathon_data",
    }
    return defaults[env]


# ============================================================================
# S3 client
# ============================================================================

def get_s3_client(endpoint=None, access_key=None, secret_key=None):
    """Create a boto3 S3 client for Cloudflare R2.

    Args fall back to environment variables if not provided:
        R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY
    """
    endpoint = endpoint or os.environ.get("R2_ENDPOINT")
    access_key = access_key or os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = secret_key or os.environ.get("R2_SECRET_ACCESS_KEY")

    if not all([endpoint, access_key, secret_key]):
        missing = []
        if not endpoint:
            missing.append("R2_ENDPOINT")
        if not access_key:
            missing.append("R2_ACCESS_KEY_ID")
        if not secret_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        raise ValueError(
            f"Missing credentials: {', '.join(missing)}. "
            "Pass them as arguments or set the environment variables."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


# ============================================================================
# Manifest operations
# ============================================================================

def load_manifest(bucket, s3_client, cache_path=None):
    """Download and parse manifest.json from the bucket root.

    If cache_path is given and the file exists locally, loads from cache
    instead of re-downloading. Pass cache_path=None to always fetch fresh.
    """
    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    # === Download manifest from R2 ===
    resp = s3_client.get_object(Bucket=bucket, Key="manifest.json")
    manifest = json.loads(resp["Body"].read().decode("utf-8"))

    # === Cache locally if requested ===
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return manifest


def list_shards(manifest, dataset=None, tags=None):
    """Filter shards from the manifest by dataset name and/or tags.

    Args:
        manifest: Parsed manifest dict.
        dataset: Dataset name to filter by (e.g., 'dataset-a'). None = all datasets.
        tags: List of tags to filter by. A shard matches if it has ALL specified tags.

    Returns:
        List of shard dicts, each with an added 'dataset' field for provenance.
    """
    results = []
    datasets = manifest.get("datasets", {})

    # === Filter by dataset name ===
    if dataset:
        if dataset not in datasets:
            available = list(datasets.keys())
            raise KeyError(
                f"Dataset '{dataset}' not found. Available: {available}"
            )
        selected = {dataset: datasets[dataset]}
    else:
        selected = datasets

    # === Iterate shards, filter by tags ===
    for ds_name, ds_info in selected.items():
        for shard in ds_info.get("shards", []):
            if tags:
                shard_tags = set(shard.get("tags", []))
                if not set(tags).issubset(shard_tags):
                    continue
            # Add dataset provenance to shard dict
            entry = {**shard, "dataset": ds_name}
            results.append(entry)

    return results


# ============================================================================
# Checksum verification
# ============================================================================

def _sha256_file(filepath, chunk_size=8 * 1024 * 1024):
    """Compute SHA-256 hash of a file, reading in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Download
# ============================================================================

def download_shards(shards, dest_dir, s3_client, bucket,
                    resume=True, verify=True, quiet=False):
    """Download a list of shards with progress bars, resume, and checksum verification.

    Args:
        shards: List of shard dicts (from list_shards). Each must have 'key' and 'size_bytes'.
        dest_dir: Local directory to download into. Preserves key structure as subdirs.
        s3_client: boto3 S3 client (from get_s3_client).
        bucket: Bucket name.
        resume: If True, skip files that already exist with correct size (or checksum if verify=True).
        verify: If True, verify SHA-256 checksum after download.
        quiet: If True, suppress progress bars.

    Returns:
        dict with 'downloaded', 'skipped', 'failed' counts and 'errors' list.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    # === Outer progress bar: shards ===
    shard_iter = tqdm(shards, desc="Shards", unit="shard", disable=quiet)
    for shard in shard_iter:
        key = shard["key"]
        size = shard.get("size_bytes", 0)
        checksum = shard.get("checksum_sha256")
        local_path = dest_dir / key

        shard_iter.set_postfix(file=Path(key).name, size_mb=f"{size / 1e6:.0f}")

        # === Resume: skip if already downloaded and valid ===
        if resume and local_path.exists():
            local_size = local_path.stat().st_size
            if local_size == size:
                # Size matches — optionally verify checksum
                if verify and checksum:
                    if _sha256_file(local_path) == checksum:
                        stats["skipped"] += 1
                        continue
                else:
                    stats["skipped"] += 1
                    continue

        # === Download ===
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Get object metadata for progress bar
            head = s3_client.head_object(Bucket=bucket, Key=key)
            total_bytes = head["ContentLength"]

            # Inner progress bar: bytes within this shard
            with tqdm(total=total_bytes, unit="B", unit_scale=True,
                      desc=Path(key).name, leave=False, disable=quiet) as pbar:

                def _progress(bytes_transferred):
                    pbar.update(bytes_transferred)

                s3_client.download_file(
                    bucket, key, str(local_path),
                    Callback=_progress,
                )

        except Exception as e:
            stats["failed"] += 1
            stats["errors"].append({"key": key, "error": str(e)})
            continue

        # === Verify checksum ===
        if verify and checksum:
            if not quiet:
                size_mb = local_path.stat().st_size / 1e6
                print(f"  Verifying checksum ({size_mb:.0f} MB)...", end=" ", flush=True)
            actual = _sha256_file(local_path)
            if not quiet:
                print("ok" if actual == checksum else "FAILED")
            if actual != checksum:
                stats["failed"] += 1
                stats["errors"].append({
                    "key": key,
                    "error": f"Checksum mismatch: expected {checksum[:16]}..., got {actual[:16]}...",
                })
                local_path.unlink()  # remove corrupted file
                continue

        stats["downloaded"] += 1

    return stats


def download_dataset(manifest, dataset_name, dest_dir=None, s3_client=None,
                     bucket=None, tags=None, resume=True, verify=True, quiet=False):
    """Convenience wrapper: download all shards for a dataset (or filtered subset).

    Args:
        manifest: Parsed manifest dict.
        dataset_name: Which dataset to download.
        dest_dir: Download directory. Defaults to environment-appropriate path.
        s3_client: boto3 client. If None, creates one from env vars.
        bucket: Bucket name. If None, reads from manifest or R2_BUCKET env var.
        tags: Optional list of tags to filter shards.
        resume, verify, quiet: Passed through to download_shards.

    Returns:
        dict with download stats (see download_shards).
    """
    dest_dir = dest_dir or _default_data_dir()
    s3_client = s3_client or get_s3_client()
    bucket = bucket or manifest.get("bucket") or os.environ.get("R2_BUCKET")

    if not bucket:
        raise ValueError("No bucket specified. Pass bucket= or set R2_BUCKET env var.")

    shards = list_shards(manifest, dataset=dataset_name, tags=tags)
    if not shards:
        print(f"No shards found for dataset='{dataset_name}', tags={tags}")
        return {"downloaded": 0, "skipped": 0, "failed": 0, "errors": []}

    total_size = sum(s.get("size_bytes", 0) for s in shards)
    print(f"Downloading {len(shards)} shards ({total_size / 1e9:.2f} GB) "
          f"for '{dataset_name}'...")

    return download_shards(
        shards, dest_dir, s3_client, bucket,
        resume=resume, verify=verify, quiet=quiet,
    )


# ============================================================================
# Summary / display helpers (for notebook use)
# ============================================================================

def summarize_manifest(manifest):
    """Print a summary table of all datasets in the manifest."""
    datasets = manifest.get("datasets", {})
    print(f"{'Dataset':<20} {'Shards':>7} {'Size (GB)':>10} {'Format':<10} Description")
    print("-" * 80)
    for name, info in datasets.items():
        n_shards = len(info.get("shards", []))
        total = info.get("total_size_bytes", 0)
        fmt = info.get("format", "?")
        desc = info.get("description", "")[:30]
        print(f"{name:<20} {n_shards:>7} {total / 1e9:>10.2f} {fmt:<10} {desc}")
