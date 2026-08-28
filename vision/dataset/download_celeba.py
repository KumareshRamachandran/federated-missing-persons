"""
vision/dataset/download_celeba.py

Downloads and prepares the CelebA dataset (primary FL training dataset).
CelebA: 202,599 images / 10,177 identities — MMLAB, CUHK

For this project, a subset of 500–1,000 identities is used,
partitioned across federated edge nodes (Police, Hospital, NGO).

Usage:
    python -m vision.dataset.download_celeba --output_dir data/raw/celeba --n_identities 1000
"""

from __future__ import annotations

import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


# ── Download helpers ──────────────────────────────────────────────────────────

def _download_via_torchvision(output_dir: str) -> Path:
    """
    Download CelebA via torchvision (requires Google Drive access).
    Returns path to the downloaded img_align_celeba directory.
    """
    import torchvision

    print("[CelebA] Downloading via torchvision (may be slow — Google Drive)...")
    torchvision.datasets.CelebA(
        root=output_dir,
        split="all",
        target_type="identity",
        download=True,
    )
    return Path(output_dir) / "celeba" / "img_align_celeba"


def _download_via_kaggle(output_dir: str) -> Path:
    """
    Download CelebA via Kaggle CLI (recommended — faster).
    Requires KAGGLE_USERNAME and KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json.
    Returns path to the extracted img_align_celeba directory.
    """
    import subprocess

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    print("[CelebA] Downloading via Kaggle CLI...")
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", "jessicali9530/celeba-dataset",
            "-p", str(dest),
            "--unzip",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n"
            "Ensure kaggle is installed (pip install kaggle) and credentials are set."
        )
    img_dir = dest / "img_align_celeba" / "img_align_celeba"
    if not img_dir.exists():
        img_dir = dest / "img_align_celeba"
    return img_dir


# ── Identity parsing ──────────────────────────────────────────────────────────

def _parse_identity_file(celeba_root: Path) -> dict[str, list[str]]:
    """
    Parse Anno/identity_CelebA.txt to build {identity_id: [image_name, ...]}.

    File format (space-separated):
        000001.jpg 2880
        000002.jpg 2937
        ...
    """
    identity_file = celeba_root / "Anno" / "identity_CelebA.txt"
    if not identity_file.exists():
        # Try alternate locations produced by torchvision download
        for candidate in celeba_root.rglob("identity_CelebA.txt"):
            identity_file = candidate
            break

    if not identity_file.exists():
        raise FileNotFoundError(
            f"identity_CelebA.txt not found under {celeba_root}. "
            "Check your download."
        )

    identity_map: dict[str, list[str]] = defaultdict(list)
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, identity_id = parts
                identity_map[identity_id].append(img_name)

    return dict(identity_map)


# ── Organisation into identity folders ───────────────────────────────────────

def _organize_by_identity(
    img_dir: Path,
    identity_map: dict[str, list[str]],
    output_dir: Path,
    n_identities: int,
) -> None:
    """
    Copy images into output_dir/<identity_id>/<image>.jpg structure.

    Args:
        img_dir:      Directory containing all raw CelebA .jpg files.
        identity_map: {identity_id: [filename, ...]} from identity file.
        output_dir:   Destination root directory.
        n_identities: How many identities to keep (sorted by most images).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort identities by number of images (descending) for richest subset
    sorted_ids = sorted(identity_map.keys(), key=lambda k: len(identity_map[k]), reverse=True)
    selected_ids = sorted_ids[:n_identities]

    print(f"[CelebA] Organising {n_identities} identities into {output_dir} ...")
    for identity_id in tqdm(selected_ids, desc="Organising identities"):
        id_dir = output_dir / identity_id
        id_dir.mkdir(exist_ok=True)

        for img_name in identity_map[identity_id]:
            src = img_dir / img_name
            dst = id_dir / img_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    print(f"[CelebA] Done — {n_identities} identities saved to {output_dir}")


# ── Public entry point ────────────────────────────────────────────────────────

def download_celeba(
    output_dir: str,
    n_identities: int = 1000,
    method: str = "torchvision",
) -> None:
    """
    Download and prepare CelebA dataset.

    Args:
        output_dir:    Root directory to save dataset.
        n_identities:  Number of identities to use (subset of 10,177).
        method:        'torchvision' or 'kaggle'.
    """
    out_path = Path(output_dir)
    organised_dir = out_path / "by_identity"

    if organised_dir.exists() and any(organised_dir.iterdir()):
        print(f"[CelebA] Dataset already organised at {organised_dir}. Skipping download.")
        return

    # Step 1: Download
    if method == "kaggle":
        img_dir = _download_via_kaggle(str(out_path))
        celeba_root = out_path
    else:
        img_dir = _download_via_torchvision(str(out_path))
        celeba_root = out_path / "celeba"

    # Step 2: Parse identity mapping
    identity_map = _parse_identity_file(celeba_root)
    print(f"[CelebA] Found {len(identity_map)} total identities.")

    # Step 3: Organise into identity sub-folders
    _organize_by_identity(img_dir, identity_map, organised_dir, n_identities)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare CelebA dataset")
    parser.add_argument("--output_dir",    type=str, default="data/raw/celeba",
                        help="Root directory to save dataset")
    parser.add_argument("--n_identities",  type=int, default=1000,
                        help="Number of identities to use (max 10177)")
    parser.add_argument("--method",        type=str, default="torchvision",
                        choices=["torchvision", "kaggle"],
                        help="Download method: torchvision (Google Drive) or kaggle CLI")
    args = parser.parse_args()
    download_celeba(args.output_dir, args.n_identities, args.method)
