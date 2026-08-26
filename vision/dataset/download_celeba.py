"""
data_scripts/download_celeba.py

Script to download and prepare the CelebA dataset (primary training dataset).
CelebA: 202,599 images / 10,177 identities — MMLAB, CUHK

For this project, we use a subset of 500–1,000 identities
partitioned across federated edge nodes (simulating Police, Hospital, NGO).

Usage:
    python data_scripts/download_celeba.py --output_dir data/raw/celeba --n_identities 1000
"""

import os
import argparse


def download_celeba(output_dir: str, n_identities: int = 1000):
    """
    Download CelebA dataset via torchvision or Kaggle CLI.

    Args:
        output_dir: Directory to save dataset.
        n_identities: Number of identities to use (subset of 10,177 total).

    Notes:
        Option A (torchvision): torchvision.datasets.CelebA — requires Google Drive access.
        Option B (Kaggle): kaggle datasets download jessicali9530/celeba-dataset
    """
    os.makedirs(output_dir, exist_ok=True)

    # TODO: Option A — torchvision
    # import torchvision
    # dataset = torchvision.datasets.CelebA(root=output_dir, split='all', download=True)

    # TODO: Option B — Kaggle CLI (recommended)
    # os.system(f"kaggle datasets download -d jessicali9530/celeba-dataset -p {output_dir} --unzip")

    # TODO: Filter to n_identities, organize as:
    # data/raw/celeba/<identity_id>/<image>.jpg
    print(f"CelebA dataset ({n_identities} identities) will be saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CelebA dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw/celeba")
    parser.add_argument("--n_identities", type=int, default=1000,
                        help="Number of identities to use (max 10177)")
    args = parser.parse_args()
    download_celeba(args.output_dir, args.n_identities)
