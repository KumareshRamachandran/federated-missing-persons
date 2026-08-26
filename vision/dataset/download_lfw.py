"""
data_scripts/download_lfw.py

Script to download and prepare the LFW (Labeled Faces in the Wild) dataset.
LFW is the primary dataset for federated simulation in this project.

Usage:
    python data_scripts/download_lfw.py --output_dir data/raw/lfw
"""

import os
import argparse


def download_lfw(output_dir: str):
    """
    Download LFW dataset.
    Uses sklearn's fetch_lfw_people or direct URL download.

    Args:
        output_dir: Directory to save the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    # TODO: from sklearn.datasets import fetch_lfw_people
    # TODO: OR download tar from http://vis-www.cs.umass.edu/lfw/lfw.tgz
    # TODO: Extract and organize into data/raw/lfw/person_name/image.jpg format
    print(f"LFW dataset will be saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download LFW dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw/lfw")
    args = parser.parse_args()
    download_lfw(args.output_dir)
