"""
federated/client/local_trainer.py

Local training of the ArcFace embedding model on the org's private dataset.
Uses Opacus (PyTorch DP-SGD) for differentially private gradient updates.

DP-SGD guarantees that model updates sent to the coordinator cannot be
used to reconstruct raw training images (gradient inversion attack defence).

Reference: Abadi et al. (2016) — Deep Learning with Differential Privacy.

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Dataset — loads org node's gallery images
# ──────────────────────────────────────────────────────────────────

class GalleryDataset(Dataset):
    """
    Loads face images from an org node's gallery directory.

    Expected structure:
        data_dir/gallery/<person_id>/<image>.jpg

    Labels are integer-encoded person IDs (for ArcFace softmax loss).
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, gallery_dir: str, img_size: int = 112):
        self.samples = []      # [(img_path, label_int)]
        self.label_map = {}    # {person_id_str: int}

        gallery_path = Path(gallery_dir)
        if not gallery_path.exists():
            raise FileNotFoundError(f"Gallery not found: {gallery_dir}")

        for label_idx, person_dir in enumerate(sorted(gallery_path.iterdir())):
            if not person_dir.is_dir():
                continue
            person_id = person_dir.name
            self.label_map[person_id] = label_idx
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((str(img_path), label_idx))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        logger.info(
            "GalleryDataset: %d images | %d identities | dir=%s",
            len(self.samples), len(self.label_map), gallery_dir,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label

    @property
    def num_classes(self) -> int:
        return len(self.label_map)


# ──────────────────────────────────────────────────────────────────
# LocalTrainer
# ──────────────────────────────────────────────────────────────────

class LocalTrainer:
    """
    Trains the ArcFace model locally on the org's private gallery data.

    Opacus PrivacyEngine wraps the optimizer + model to ensure each
    gradient update satisfies (ε, δ)-DP before being shared.
    """

    def __init__(
        self,
        model: nn.Module,
        data_dir: str,
        epochs: int = 1,
        lr: float = 1e-4,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        """
        Args:
            model:      ArcFace model instance (from vision.arcface_model).
            data_dir:   Org node data directory (contains gallery/ subdirectory).
            epochs:     Local training epochs per federation round.
            lr:         SGD / Adam learning rate.
            batch_size: DataLoader batch size. Keep small for Opacus compatibility.
            device:     "cuda", "cpu", or None (auto-detect).
        """
        self.model = model
        self.data_dir = data_dir
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info("LocalTrainer on device=%s | epochs=%d | lr=%s", self.device, epochs, lr)

    def train(
        self,
        use_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
    ) -> Tuple[dict, float, Optional[float]]:
        """
        Run local training for self.epochs with optional DP-SGD.

        Args:
            use_dp:            Enable Opacus Differential Privacy.
            noise_multiplier:  DP noise scale σ (higher = more privacy, less accuracy).
            max_grad_norm:     Per-sample gradient clipping norm C.
            delta:             DP delta (target failure probability).

        Returns:
            Tuple of:
              - state_dict: Updated model weights (to be sent as FL update)
              - avg_loss:   Average training loss across all epochs
              - epsilon:    Privacy budget consumed (None if DP disabled)
        """
        gallery_dir = os.path.join(self.data_dir, "gallery")
        try:
            dataset = GalleryDataset(gallery_dir)
        except FileNotFoundError:
            logger.error("Gallery not found at %s — skipping local training.", gallery_dir)
            return self.model.state_dict(), float("inf"), None

        if len(dataset) == 0:
            logger.warning("Empty gallery at %s — nothing to train on.", gallery_dir)
            return self.model.state_dict(), float("inf"), None

        # Opacus requires batch_size <= dataset size
        effective_batch = min(self.batch_size, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=effective_batch,
            shuffle=True,
            drop_last=True,          # Required by Opacus (uniform batches)
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        epsilon = None

        if use_dp:
            try:
                from opacus import PrivacyEngine
                privacy_engine = PrivacyEngine()
                self.model, optimizer, loader = privacy_engine.make_private(
                    module=self.model,
                    optimizer=optimizer,
                    data_loader=loader,
                    noise_multiplier=noise_multiplier,
                    max_grad_norm=max_grad_norm,
                )
                logger.info(
                    "Opacus DP-SGD enabled | σ=%.3f | C=%.3f | δ=%.1e",
                    noise_multiplier, max_grad_norm, delta,
                )
            except ImportError:
                logger.warning("Opacus not installed — running without DP. Install: pip install opacus")
                use_dp = False

        # ── Training loop ─────────────────────────────────────────
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                embeddings = self.model(images)         # [B, 512]

                # Use embeddings directly with cross-entropy for classification
                # (In full ArcFace, this would use ArcFace loss head — simplification for FL)
                loss = criterion(embeddings, labels) if embeddings.shape[-1] == dataset.num_classes \
                    else self._embedding_loss(embeddings, labels, criterion)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_batches += 1

            avg_epoch_loss = epoch_loss / max(len(loader), 1)
            logger.info(
                "Epoch %d/%d | loss=%.4f | samples=%d",
                epoch + 1, self.epochs, avg_epoch_loss, len(dataset),
            )
            total_loss += epoch_loss

        avg_loss = total_loss / max(total_batches, 1)

        # Compute epsilon spent
        if use_dp:
            try:
                epsilon = privacy_engine.get_epsilon(delta=delta)
                logger.info("DP budget spent: ε=%.4f, δ=%.1e", epsilon, delta)
            except Exception:
                epsilon = None

        # Unwrap Opacus model to get clean state_dict
        state_dict = self._get_clean_state_dict()

        logger.info(
            "Local training complete | avg_loss=%.4f | ε=%s",
            avg_loss, f"{epsilon:.4f}" if epsilon is not None else "N/A (DP off)",
        )
        return state_dict, avg_loss, epsilon

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _embedding_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """
        Compute contrastive loss on embeddings when model output ≠ num_classes.
        Uses a simple linear projection layer in-place.
        This is a fallback for when ArcFace head is not attached.
        """
        # In-place projection: compute squared distances to class centroids
        # This is a simplified approach suitable for the FL training stage
        unique_labels = labels.unique()
        centroids = torch.stack([
            embeddings[labels == lbl].mean(0) for lbl in unique_labels
        ])
        dists = torch.cdist(embeddings, centroids)
        pseudo_logits = -dists  # closer = higher logit
        # Map original labels to [0, n_unique)
        label_map = {lbl.item(): i for i, lbl in enumerate(unique_labels)}
        mapped = torch.tensor([label_map[l.item()] for l in labels], device=self.device)
        return criterion(pseudo_logits, mapped)

    def _get_clean_state_dict(self) -> dict:
        """
        Return the model's state_dict, unwrapping Opacus's GradSampleModule if needed.
        """
        model = self.model
        # Opacus wraps model in GradSampleModule — unwrap if needed
        if hasattr(model, "_module"):
            model = model._module
        return {k: v.cpu().detach().numpy() for k, v in model.state_dict().items()}
