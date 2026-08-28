"""
vision/dataset/download_lfw.py

Downloads and prepares the Labeled Faces in the Wild (LFW) dataset.
LFW: 13,233 images / 5,749 identities — UMass Amherst

Used for:
  - Federated node partitioning (40/30/30 Non-IID split)
  - Evaluation and face verification testing
  - Query set (held-out identity images for search simulation)

Usage:
    python -m vision.dataset.download_lfw --output_dir data/raw/lfw
"""

from __future__ import annotations

import os
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
LFW_MIN_IMAGES_PER_IDENTITY = 2  # Need at least gallery + query image


# ── Download helpers ──────────────────────────────────────────────────────────

class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_tarball(url: str, dest_path: Path) -> None:
    """Download a tarball from url to dest_path with progress bar."""
    print(f"[LFW] Downloading from {url} ...")
    with _DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc="lfw.tgz"
    ) as t:
        urllib.request.urlretrieve(url, filename=str(dest_path), reporthook=t.update_to)


def _extract_tarball(tar_path: Path, extract_to: Path) -> None:
    """Extract a .tgz tarball."""
    print(f"[LFW] Extracting {tar_path.name} ...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(extract_to))
    print(f"[LFW] Extracted to {extract_to}")


# ── sklearn fallback ──────────────────────────────────────────────────────────

def _download_via_sklearn(output_dir: Path) -> Path:
    """
    Download a subset of LFW using sklearn (min_faces_per_person=2).
    Returns path to organised output directory.
    """
    from sklearn.datasets import fetch_lfw_people
    from PIL import Image as PILImage
    import numpy as np

    print("[LFW] Downloading via sklearn (min 2 images per identity)...")
    lfw_data = fetch_lfw_people(
        min_faces_per_person=LFW_MIN_IMAGES_PER_IDENTITY,
        resize=None,
        color=True,
    )

    sklearn_dir = output_dir / "sklearn_lfw"
    sklearn_dir.mkdir(parents=True, exist_ok=True)

    target_names = lfw_data.target_names
    images = lfw_data.images           # (N, H, W, 3) float64 in [0, 1]
    targets = lfw_data.target          # (N,)

    # Organise into identity sub-folders
    for idx, (img_array, target_id) in enumerate(
        tqdm(zip(images, targets), total=len(targets), desc="Saving LFW images")
    ):
        name = target_names[target_id].replace(" ", "_")
        id_dir = sklearn_dir / name
        id_dir.mkdir(exist_ok=True)
        img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_uint8)
        pil_img.save(str(id_dir / f"{idx:05d}.jpg"))

    return sklearn_dir


# ── Public entry point ────────────────────────────────────────────────────────

def download_lfw(
    output_dir: str,
    method: str = "direct",
    min_images: int = LFW_MIN_IMAGES_PER_IDENTITY,
) -> None:
    """
    Download and prepare the LFW dataset.

    Args:
        output_dir:  Root directory to save dataset.
        method:      'direct' (tarball from UMass) or 'sklearn'.
        min_images:  Minimum images per identity to include (default 2).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lfw_dir = out_path / "lfw"

    if lfw_dir.exists() and any(lfw_dir.iterdir()):
        n_ids = sum(1 for p in lfw_dir.iterdir() if p.is_dir())
        print(f"[LFW] Already downloaded — {n_ids} identities at {lfw_dir}")
        return

    if method == "sklearn":
        lfw_dir = _download_via_sklearn(out_path)
    else:
        # Direct download
        tar_path = out_path / "lfw.tgz"
        if not tar_path.exists():
            _download_tarball(LFW_URL, tar_path)
        _extract_tarball(tar_path, out_path)
        # Cleanup tarball to save space
        tar_path.unlink(missing_ok=True)

    # Filter to identities with >= min_images images
    _filter_identities(lfw_dir, min_images)

    n_ids = sum(1 for p in lfw_dir.iterdir() if p.is_dir())
    n_imgs = sum(1 for p in lfw_dir.rglob("*.jpg"))
    print(f"[LFW] Ready — {n_ids} identities, {n_imgs} images at {lfw_dir}")


def _filter_identities(lfw_dir: Path, min_images: int) -> None:
    """Remove identity directories with fewer than min_images images."""
    removed = 0
    for id_dir in list(lfw_dir.iterdir()):
        if not id_dir.is_dir():
            continue
        images = list(id_dir.glob("*.jpg")) + list(id_dir.glob("*.png"))
        if len(images) < min_images:
            shutil.rmtree(id_dir)
            removed += 1

    if removed:
        print(f"[LFW] Removed {removed} identities with < {min_images} images.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare LFW dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw/lfw",
                        help="Root directory to save dataset")
    parser.add_argument("--method",     type=str, default="direct",
                        choices=["direct", "sklearn"],
                        help="Download method: direct tarball or sklearn API")
    parser.add_argument("--min_images", type=int, default=2,
                        help="Minimum images per identity (for gallery + query split)")
    args = parser.parse_args()
    download_lfw(args.output_dir, args.method, args.min_images)
