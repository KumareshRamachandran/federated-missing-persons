"""
vision/dataset/partition_data.py

Splits a face dataset (CelebA or LFW) into 3 federated edge nodes (Non-IID simulation).

Federation Strategy:
  - Node 1 (Police):   40% of identities (exclusive)
  - Node 2 (Hospital): 30% of identities (exclusive)
  - Node 3 (NGO):      30% of identities (exclusive)

Each identity (person) exists exclusively in only one node's directory so the galleries
do not overlap (e.g., Identity 1 only in node_police, Identity 2 only in node_hospital).

Output structure:
    data/nodes/
        node_police/<identity>/<image>.jpg
        node_hospital/<identity>/<image>.jpg
        node_ngo/<identity>/<image>.jpg

Usage:
    python vision/dataset/partition_data.py --dataset_dir data/raw/celeba/by_identity --output_dir data/nodes
"""

from __future__ import annotations

import sys
import os
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


NODE_SPLITS = {
    "node_police":   0.40,
    "node_hospital": 0.30,
    "node_ngo":      0.30,
}


def partition_celeba(
    dataset_dir: str = "data/raw/celeba/by_identity",
    output_dir: str = "data/nodes",
    seed: int = 42,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict:
    """
    Performs a Non-IID split on CelebA dataset to simulate 3 federated edge nodes.
    Guarantees strict identity exclusivity across node directories.

    Args:
        dataset_dir: Source dataset directory containing identity subfolders.
        output_dir:  Target root directory (saves to node_police, node_hospital, node_ngo).
        seed:        Random seed for reproducible split.
        extensions:  Supported image file extensions.

    Returns:
        dict: Summary statistics per node.
    """
    random.seed(seed)
    src = Path(dataset_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect identity directories
    identities = []
    if src.exists():
        for item in sorted(src.iterdir()):
            if item.is_dir():
                imgs = [p for p in item.iterdir() if p.suffix.lower() in extensions]
                if imgs:
                    identities.append(item.name)

    if not identities:
        print(f"[Partition] Warning: No identity subfolders found in {dataset_dir}.")
        return {}

    random.shuffle(identities)
    total = len(identities)

    # Calculate 40% / 30% / 30% Non-IID split boundaries
    n_police = round(total * 0.40)
    n_hospital = round(total * 0.30)
    n_ngo = total - n_police - n_hospital

    splits = {
        "node_police": identities[:n_police],
        "node_hospital": identities[n_police:n_police + n_hospital],
        "node_ngo": identities[n_police + n_hospital:],
    }

    print(f"[Partition] Splitting {total} exclusive identities across 3 nodes (Police: {n_police}, Hospital: {n_hospital}, NGO: {n_ngo})...")

    stats = {}
    for node_name, node_identities in splits.items():
        node_dir = out / node_name
        node_dir.mkdir(parents=True, exist_ok=True)
        # Create gallery subfolder for downstream pipeline compatibility
        gallery_dir = node_dir / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        copied_count = 0
        for identity_id in tqdm(node_identities, desc=f"  {node_name}", leave=False):
            src_id_dir = src / identity_id
            dst_id_dir = node_dir / identity_id
            dst_gallery_dir = gallery_dir / identity_id
            dst_id_dir.mkdir(exist_ok=True)
            dst_gallery_dir.mkdir(exist_ok=True)

            images = [p for p in src_id_dir.iterdir() if p.suffix.lower() in extensions]
            for img_path in images:
                shutil.copy2(img_path, dst_id_dir / img_path.name)
                shutil.copy2(img_path, dst_gallery_dir / img_path.name)
                copied_count += 1

        stats[node_name] = {"identities": len(node_identities), "images": copied_count}
        print(f"  [OK] {node_name:<15} -> {len(node_identities)} identities ({copied_count} images)")

    print(f"[Partition] Completed non-IID dataset partitioning to {output_dir}")
    return stats


# Convenience alias for backwards compatibility
partition_dataset = partition_celeba
partition_lfw = partition_celeba


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-IID Partitioning into 3 Federated Nodes")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Source identity directory (auto-detected if None)")
    parser.add_argument("--output_dir", type=str, default="data/nodes", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    target_dir = args.dataset_dir
    if target_dir is None:
        # Auto-detect existing downloaded dataset
        candidates = [
            Path("data/raw/celeba/by_identity"),
            Path("data/raw/lfw/lfw"),
            Path("data/raw/lfw"),
        ]
        for c in candidates:
            if c.exists() and any(p.is_dir() for p in c.iterdir()):
                target_dir = str(c)
                print(f"[Partition] Auto-detected downloaded dataset at: {target_dir}")
                break

    if target_dir is None or not Path(target_dir).exists():
        print(f"[Partition] Source directory not found. Creating synthetic demo dataset for verification...")
        demo_dir = Path("data/raw/demo_celeba/by_identity")
        for i in range(1, 11):
            id_folder = demo_dir / f"identity_{i:03d}"
            id_folder.mkdir(parents=True, exist_ok=True)
            import cv2, numpy as np
            img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(id_folder / "img_01.jpg"), img)
            cv2.imwrite(str(id_folder / "img_02.jpg"), img)
        target_dir = str(demo_dir)

    stats = partition_celeba(target_dir, args.output_dir, seed=args.seed)
    print("[OK] partition_data.py test completed!")

