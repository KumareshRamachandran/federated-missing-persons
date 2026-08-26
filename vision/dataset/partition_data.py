"""
data_scripts/partition_data.py

Splits the LFW dataset into federated nodes (Non-IID simulation).

Strategy:
  - Node 1 (Police):   40% of identities (exclusive)
  - Node 2 (Hospital): 30% of identities (exclusive)
  - Node 3 (NGO):      30% of identities (exclusive)
  - Small overlap (~5%) optionally enabled for multi-node match testing

For each identity:
  - 1 image withheld as the "query" (search) set
  - Remaining images form the local "gallery" (database)

Usage:
    python data_scripts/partition_data.py --lfw_dir data/raw/lfw --output_dir data/nodes
"""

import os
import argparse
import shutil
import random


def partition_lfw(lfw_dir: str, output_dir: str, seed: int = 42, overlap: bool = False):
    """
    Partition LFW into federated node directories.

    Output structure:
        data/nodes/
            node_police/gallery/<identity>/<image>.jpg
            node_hospital/gallery/<identity>/<image>.jpg
            node_ngo/gallery/<identity>/<image>.jpg
            query_set/<identity>/<image>.jpg   ← held-out query images
    """
    random.seed(seed)
    # TODO: List all identity directories in lfw_dir
    # TODO: Filter identities with >= 2 images (need at least 1 gallery + 1 query)
    # TODO: Shuffle identities
    # TODO: Split into 40/30/30 proportions for 3 nodes
    # TODO: For each identity in each node, hold out 1 image → query_set
    # TODO: Copy remaining images to node's gallery dir
    # TODO: If overlap=True, add 5% of identities to two nodes
    print(f"Partitioning LFW from {lfw_dir} into federated nodes at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition LFW into federated nodes")
    parser.add_argument("--lfw_dir", type=str, default="data/raw/lfw")
    parser.add_argument("--output_dir", type=str, default="data/nodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap", action="store_true")
    args = parser.parse_args()
    partition_lfw(args.lfw_dir, args.output_dir, args.seed, args.overlap)
