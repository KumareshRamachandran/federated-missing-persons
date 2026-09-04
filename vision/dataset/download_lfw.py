"""
vision/dataset/download_lfw.py

Downloads and prepares the Labeled Faces in the Wild (LFW) dataset.
LFW: 13,233 images / 5,749 identities — UMass Amherst

Usage:
    python vision/dataset/download_lfw.py --output_dir data/raw/lfw
"""

from __future__ import annotations

import os
import sys
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


LFW_URL = "https://vis-www.cs.umass.edu/lfw/lfw.tgz"
LFW_MIN_IMAGES_PER_IDENTITY = 2


class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_tarball(url: str, dest_path: Path) -> None:
    """Download a tarball with User-Agent header and progress bar."""
    print(f"[LFW] Downloading from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_length = response.getheader("Content-Length")
        total_size = int(total_length) if total_length is not None else None
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="lfw.tgz") as pbar:
            block_size = 8192
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                out_file.write(buffer)
                pbar.update(len(buffer))


def _extract_tarball(tar_path: Path, extract_to: Path) -> None:
    """Extract a .tgz tarball."""
    print(f"[LFW] Extracting {tar_path.name} ...")
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(extract_to))
    print(f"[LFW] Extracted to {extract_to}")


def _download_via_sklearn(output_dir: Path) -> Path:
    """
    Download LFW using sklearn (min_faces_per_person=2).
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

    lfw_dir = output_dir / "lfw"
    lfw_dir.mkdir(parents=True, exist_ok=True)

    target_names = lfw_data.target_names
    images = lfw_data.images
    targets = lfw_data.target

    for idx, (img_array, target_id) in enumerate(
        tqdm(zip(images, targets), total=len(targets), desc="Saving LFW images")
    ):
        name = target_names[target_id].replace(" ", "_")
        id_dir = lfw_dir / name
        id_dir.mkdir(exist_ok=True)
        img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_uint8)
        pil_img.save(str(id_dir / f"{idx:05d}.jpg"))

    return lfw_dir


def download_lfw(
    output_dir: str = "data/raw/lfw",
    method: str = "direct",
    min_images: int = LFW_MIN_IMAGES_PER_IDENTITY,
) -> None:
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
        # Direct download with automatic sklearn fallback
        tar_path = out_path / "lfw.tgz"
        try:
            if not tar_path.exists():
                _download_tarball(LFW_URL, tar_path)
            _extract_tarball(tar_path, out_path)
            tar_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[LFW] Direct download note: {e}")
            print("[LFW] Falling back to sklearn dataset downloader...")
            lfw_dir = _download_via_sklearn(out_path)

    _filter_identities(lfw_dir, min_images)

    n_ids = sum(1 for p in lfw_dir.iterdir() if p.is_dir())
    n_imgs = sum(1 for p in lfw_dir.rglob("*.jpg"))
    print(f"[LFW] Ready — {n_ids} identities, {n_imgs} images at {lfw_dir}")


def _filter_identities(lfw_dir: Path, min_images: int) -> None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare LFW dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw/lfw", help="Output directory")
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "sklearn"], help="Download method")
    parser.add_argument("--min_images", type=int, default=2, help="Minimum images per identity")
    args = parser.parse_args()

    download_lfw(args.output_dir, args.method, args.min_images)

