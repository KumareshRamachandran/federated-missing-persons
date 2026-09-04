"""
vision/dataset/download_celeba.py

Downloads and prepares the CelebA dataset using torchvision.datasets.
CelebA: 202,599 images / 10,177 identities — MMLAB, CUHK

Usage:
    python vision/dataset/download_celeba.py --output_dir data/raw/celeba --n_identities 1000
"""

from __future__ import annotations

import os
import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def download_celeba_torchvision(output_dir: str = "data/raw/celeba") -> Path:
    """
    Downloads CelebA dataset using torchvision.datasets.CelebA.

    Args:
        output_dir: Output directory path to save dataset.

    Returns:
        Path to img_align_celeba directory.
    """
    import torchvision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[CelebA] Downloading CelebA dataset via torchvision.datasets to {output_dir}...")
    try:
        celeba_dataset = torchvision.datasets.CelebA(
            root=str(out_path),
            split="all",
            target_type="identity",
            download=True,
        )
        print(f"[CelebA] Successfully downloaded CelebA dataset ({len(celeba_dataset)} samples).")
    except Exception as e:
        print(f"[CelebA] Torchvision download note: {e}")
        print("         If Google Drive download quota is exceeded, please ensure raw images are present.")

    # Locate img_align_celeba
    candidate = out_path / "celeba" / "img_align_celeba"
    if not candidate.exists():
        for found in out_path.rglob("img_align_celeba"):
            if found.is_dir():
                candidate = found
                break

    return candidate


def _parse_identity_file(celeba_root: Path) -> dict[str, list[str]]:
    """
    Parse identity_CelebA.txt to map identity_id -> list of image filenames.
    """
    identity_file = celeba_root / "Anno" / "identity_CelebA.txt"
    if not identity_file.exists():
        for candidate in celeba_root.rglob("identity_CelebA.txt"):
            identity_file = candidate
            break

    if not identity_file.exists():
        return {}

    identity_map: dict[str, list[str]] = defaultdict(list)
    with open(identity_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_name, identity_id = parts
                identity_map[identity_id].append(img_name)

    return dict(identity_map)


def organize_celeba_by_identity(
    img_dir: Path,
    identity_map: dict[str, list[str]],
    output_dir: Path,
    n_identities: int = 1000,
) -> None:
    """
    Organizes raw CelebA images into identity subfolders:
    output_dir/<identity_id>/<img_name>.jpg
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not identity_map:
        print("[CelebA] Warning: Empty identity map. Skipping identity organization.")
        return

    sorted_ids = sorted(identity_map.keys(), key=lambda k: len(identity_map[k]), reverse=True)
    selected_ids = sorted_ids[:n_identities]

    print(f"[CelebA] Organizing {len(selected_ids)} identities into {output_dir} ...")
    for identity_id in tqdm(selected_ids, desc="Organizing identities"):
        id_dir = output_dir / identity_id
        id_dir.mkdir(exist_ok=True)

        for img_name in identity_map[identity_id]:
            src = img_dir / img_name
            dst = id_dir / img_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

    print(f"[CelebA] Done — organized into {output_dir}")


def download_and_prepare_celeba(
    output_dir: str = "data/raw/celeba",
    n_identities: int = 1000,
) -> None:
    """
    Main download and preparation entry point.
    """
    out_path = Path(output_dir)
    organized_dir = out_path / "by_identity"
    if organized_dir.exists() and any(organized_dir.iterdir()):
        print(f"[CelebA] Dataset already present at {organized_dir}.")
        return

    img_dir = download_celeba_torchvision(output_dir)
    celeba_root = out_path / "celeba" if (out_path / "celeba").exists() else out_path
    identity_map = _parse_identity_file(celeba_root)
    if img_dir.exists() and identity_map:
        organize_celeba_by_identity(img_dir, identity_map, organized_dir, n_identities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CelebA dataset using torchvision")
    parser.add_argument("--output_dir", type=str, default="data/raw/celeba", help="Output directory")
    parser.add_argument("--n_identities", type=int, default=1000, help="Number of identities to process")
    args = parser.parse_args()

    download_and_prepare_celeba(args.output_dir, args.n_identities)

