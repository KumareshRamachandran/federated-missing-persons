"""
vision/dataset/partition_data.py

Splits a face dataset (LFW or CelebA) into federated nodes (Non-IID simulation).

Federation Strategy:
  - Node 1 (Police):   40% of identities (exclusive by default)
  - Node 2 (Hospital): 30% of identities (exclusive by default)
  - Node 3 (NGO):      30% of identities (exclusive by default)
  - Small overlap (~5%) optionally enabled for multi-node match testing

Per identity:
  - 1 image withheld as the "query" set (simulates the missing person's photo)
  - Remaining images form the local "gallery" (the org's face database)

Output structure:
    data/nodes/
        node_police/gallery/<identity>/<image>.jpg
        node_hospital/gallery/<identity>/<image>.jpg
        node_ngo/gallery/<identity>/<image>.jpg
        query_set/<identity>/<image>.jpg    ← held-out query images

Usage:
    python -m vision.dataset.partition_data \\
        --dataset_dir data/raw/lfw/lfw \\
        --output_dir  data/nodes \\
        --overlap
"""

from __future__ import annotations

import os
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# ── Node configuration ────────────────────────────────────────────────────────

NODE_SPLITS = {
    "node_police":   0.40,
    "node_hospital": 0.30,
    "node_ngo":      0.30,
}

OVERLAP_FRACTION = 0.05   # 5% of identities shared between two nodes


# ── Partitioning logic ────────────────────────────────────────────────────────

def partition_dataset(
    dataset_dir: str,
    output_dir: str,
    seed: int = 42,
    overlap: bool = False,
    min_images: int = 2,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> dict:
    """
    Partition a face dataset into federated node directories.

    Args:
        dataset_dir: Source directory with <identity>/<image>.* structure.
        output_dir:  Root output directory (data/nodes/).
        seed:        Random seed for reproducible splits.
        overlap:     If True, 5% of identities appear in two nodes.
        min_images:  Minimum images per identity (need gallery + query).
        extensions:  Image file extensions to consider.

    Returns:
        dict: Summary statistics {node_name: {identities, images}}.
    """
    random.seed(seed)

    src = Path(dataset_dir)
    out = Path(output_dir)

    # ── Step 1: Collect and filter identities ────────────────────────────────
    all_identities = _collect_identities(src, min_images, extensions)
    random.shuffle(all_identities)

    total = len(all_identities)
    if total == 0:
        raise ValueError(
            f"No identities with >= {min_images} images found in {dataset_dir}. "
            "Check your dataset path."
        )

    print(f"\n[Partition] Found {total} eligible identities in {dataset_dir}")
    print(f"[Partition] Splitting {total} identities across 3 nodes "
          f"(40/30/30 Non-IID){' + 5% overlap' if overlap else ''}\n")

    # ── Step 2: Compute split boundaries ────────────────────────────────────
    node_names = list(NODE_SPLITS.keys())
    splits = _compute_splits(total, NODE_SPLITS)
    assignment = _assign_identities(all_identities, splits, overlap)

    # ── Step 3: Copy files ───────────────────────────────────────────────────
    stats = {}
    query_dir = out / "query_set"
    query_dir.mkdir(parents=True, exist_ok=True)

    for node_name, identity_list in assignment.items():
        node_gallery = out / node_name / "gallery"
        n_images = 0

        for identity in tqdm(identity_list, desc=f"  {node_name}", leave=False):
            images = _get_images(src / identity, extensions)
            if len(images) < min_images:
                continue

            random.shuffle(images)
            query_img = images[0]       # 1 image → query set
            gallery_imgs = images[1:]   # rest   → gallery

            # Copy query image
            q_dest_dir = query_dir / identity
            q_dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(query_img, q_dest_dir / query_img.name)

            # Copy gallery images
            g_dest_dir = node_gallery / identity
            g_dest_dir.mkdir(parents=True, exist_ok=True)
            for img in gallery_imgs:
                shutil.copy2(img, g_dest_dir / img.name)
                n_images += 1

        n_ids = len(identity_list)
        stats[node_name] = {"identities": n_ids, "gallery_images": n_images}
        print(f"  ✓ {node_name:<20} → {n_ids:>4} identities, {n_images:>5} gallery images")

    # Query set stats
    q_ids = sum(1 for p in query_dir.iterdir() if p.is_dir())
    q_imgs = sum(1 for p in query_dir.rglob("*") if p.is_file())
    stats["query_set"] = {"identities": q_ids, "query_images": q_imgs}
    print(f"  ✓ {'query_set':<20} → {q_ids:>4} identities, {q_imgs:>5} query images")
    print(f"\n[Partition] Complete — output at {output_dir}\n")

    return stats


# ── Private helpers ───────────────────────────────────────────────────────────

def _collect_identities(
    src: Path, min_images: int, extensions: tuple
) -> list[str]:
    """Return list of identity folder names that have >= min_images images."""
    identities = []
    for id_dir in sorted(src.iterdir()):
        if not id_dir.is_dir():
            continue
        imgs = _get_images(id_dir, extensions)
        if len(imgs) >= min_images:
            identities.append(id_dir.name)
    return identities


def _get_images(id_dir: Path, extensions: tuple) -> list[Path]:
    """Return all image files in an identity directory."""
    return [
        p for p in sorted(id_dir.iterdir())
        if p.is_file() and p.suffix.lower() in extensions
    ]


def _compute_splits(total: int, fractions: dict) -> list[int]:
    """
    Compute integer split sizes that sum to total.
    Last split absorbs any rounding remainder.
    """
    sizes = []
    remainder = total
    frac_values = list(fractions.values())
    for i, frac in enumerate(frac_values):
        if i == len(frac_values) - 1:
            sizes.append(remainder)
        else:
            n = round(total * frac)
            sizes.append(n)
            remainder -= n
    return sizes


def _assign_identities(
    identities: list[str],
    splits: list[int],
    overlap: bool,
) -> dict[str, list[str]]:
    """
    Assign identities to nodes according to split sizes.

    If overlap=True, a 5% subset from node_police is also added to
    node_hospital (simulating a shared inter-agency database).
    """
    node_names = list(NODE_SPLITS.keys())
    assignment: dict[str, list[str]] = {}
    cursor = 0

    for node_name, size in zip(node_names, splits):
        assignment[node_name] = identities[cursor: cursor + size]
        cursor += size

    if overlap:
        n_overlap = max(1, round(len(assignment["node_police"]) * OVERLAP_FRACTION))
        overlap_ids = random.sample(assignment["node_police"], n_overlap)
        assignment["node_hospital"] = assignment["node_hospital"] + overlap_ids
        print(f"  [overlap] Added {n_overlap} shared identities to node_hospital")

    return assignment


# ── Convenience wrapper (backward compatible) ─────────────────────────────────

def partition_lfw(
    lfw_dir: str,
    output_dir: str,
    seed: int = 42,
    overlap: bool = False,
) -> None:
    """
    Backwards-compatible wrapper — partition LFW into federated nodes.
    Called by the project README quick-start command.
    """
    # LFW stores images as data/raw/lfw/lfw/<name>/<image>.jpg
    lfw_path = Path(lfw_dir)
    inner = lfw_path / "lfw"
    src = inner if inner.exists() else lfw_path
    partition_dataset(str(src), output_dir, seed=seed, overlap=overlap)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition a face dataset into federated nodes (40/30/30 Non-IID)"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="data/raw/lfw/lfw",
        help="Source dataset directory with <identity>/<image>.* layout",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/nodes",
        help="Root output directory for federated node galleries",
    )
    parser.add_argument("--seed",    type=int,  default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--overlap", action="store_true",
                        help="Enable 5%% identity overlap between nodes")
    args = parser.parse_args()

    partition_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        overlap=args.overlap,
    )
