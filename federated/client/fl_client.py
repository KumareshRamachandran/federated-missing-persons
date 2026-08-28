"""
federated/client/fl_client.py

Flower NumPy Client — runs inside each organization node.

Responsibilities:
  1. Receive global model weights from coordinator (get_parameters / fit)
  2. Train locally with DP-SGD via LocalTrainer (Opacus)
  3. Send weight updates back — never raw data
  4. Evaluate local model on held-out validation images
  5. Expose LocalMatcher for privacy-preserving query inference

Usage (each org node runs this in a separate terminal):
    python -m federated.client.fl_client --node node_police --data data/nodes/node_police

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import NDArrays, Scalar

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from federated.client.local_trainer import LocalTrainer, GalleryDataset
from federated.client.local_matcher import LocalMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fl_client")


# ──────────────────────────────────────────────────────────────────
# Lightweight embedding model for FL (works without GPU)
# Uses a simple linear projector when full ArcFace isn't available
# ──────────────────────────────────────────────────────────────────

class EmbeddingModel(nn.Module):
    """
    Lightweight face embedding model for federated training.

    In production: replace backbone with InsightFace ArcFace iResNet50.
    This class wraps or falls back gracefully depending on what's installed.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._backend = self._load_backend()

    def _load_backend(self) -> nn.Module:
        """Try to load ArcFace; fall back to MobileNetV2 if unavailable."""
        try:
            from vision.arcface_model import ArcFaceModel
            model = ArcFaceModel()
            logger.info("Using ArcFace iResNet50 backbone.")
            return model
        except Exception as e:
            logger.warning("ArcFace unavailable (%s). Using MobileNetV2 fallback.", e)
            import torchvision.models as models
            backbone = models.mobilenet_v2(weights=None)
            backbone.classifier = nn.Linear(backbone.last_channel, self.embedding_dim)
            return backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self._backend(x)
        # L2 normalise
        return nn.functional.normalize(emb, p=2, dim=1)

    def state_dict(self, **kwargs):
        return self._backend.state_dict(**kwargs)

    def load_state_dict(self, state_dict, **kwargs):
        return self._backend.load_state_dict(state_dict, **kwargs)

    def parameters(self, **kwargs):
        return self._backend.parameters(**kwargs)

    def train(self, mode=True):
        return self._backend.train(mode)

    def eval(self):
        return self._backend.eval()

    def to(self, device):
        self._backend = self._backend.to(device)
        return self


# ──────────────────────────────────────────────────────────────────
# Flower Client
# ──────────────────────────────────────────────────────────────────

class OrgFLClient(fl.client.NumPyClient):
    """
    Flower NumPy client for a single organization node (Police / Hospital / NGO).

    The FL training loop:
      Round N:
        1. Coordinator → global model weights (parameters_to_ndarrays)
        2. Load weights into local model
        3. Train locally with Opacus DP-SGD
        4. Return updated weights + num_samples + metrics
        5. Coordinator aggregates all node updates (FedAvgWithDP)
        6. Repeat
    """

    def __init__(
        self,
        node_id: str,
        data_dir: str,
        embedding_dim: int = 512,
        local_epochs: int = 1,
        use_dp: bool = True,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
    ):
        self.node_id = node_id
        self.data_dir = data_dir
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        # Core model
        self.model = EmbeddingModel(embedding_dim=embedding_dim)

        # Local training engine
        self.trainer = LocalTrainer(
            model=self.model,
            data_dir=data_dir,
            epochs=local_epochs,
        )

        # Privacy-preserving matcher (for query inference)
        self.matcher = LocalMatcher(data_dir=data_dir)

        logger.info(
            "OrgFLClient initialised | node=%s | data=%s | DP=%s",
            node_id, data_dir, use_dp,
        )

    # ── Flower required methods ────────────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        """
        Return current local model weights as a list of NumPy arrays.
        Called by coordinator at the start to get initial parameters.
        """
        state_dict = self.model.state_dict()
        return [v.cpu().numpy() for v in state_dict.values()]

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Receive global weights → load → local DP training → return updated weights.

        Args:
            parameters: Global model weights from coordinator.
            config:     Training config sent by on_fit_config_fn (rounds, lr, DP settings).

        Returns:
            (updated_weights, num_training_samples, metrics)
        """
        server_round = int(config.get("server_round", 0))
        logger.info("[%s] Round %d: fit() called", self.node_id, server_round)

        # ── 1. Load global weights into local model ────────────────
        self._set_parameters(parameters)

        # ── 2. Override DP config from coordinator if provided ──────
        use_dp = bool(config.get("use_dp", self.use_dp))
        noise_multiplier = float(config.get("noise_multiplier", self.noise_multiplier))
        max_grad_norm = float(config.get("max_grad_norm", self.max_grad_norm))
        local_epochs = int(config.get("local_epochs", self.trainer.epochs))
        self.trainer.epochs = local_epochs
        self.trainer.lr = float(config.get("learning_rate", self.trainer.lr))

        # ── 3. Local DP training ────────────────────────────────────
        state_dict, avg_loss, epsilon = self.trainer.train(
            use_dp=use_dp,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
        )

        # ── 4. Convert state_dict to list of NDArrays ───────────────
        updated_weights = list(state_dict.values())

        # ── 5. Count training samples ───────────────────────────────
        gallery_dir = f"{self.data_dir}/gallery"
        try:
            dataset = GalleryDataset(gallery_dir)
            num_samples = len(dataset)
        except Exception:
            num_samples = 1  # avoid 0 which breaks FedAvg weighting

        metrics = {
            "loss": avg_loss,
            "node_id": self.node_id,
        }
        if epsilon is not None:
            metrics["epsilon"] = epsilon

        logger.info(
            "[%s] Round %d fit done | samples=%d | loss=%.4f | ε=%s",
            self.node_id, server_round, num_samples, avg_loss,
            f"{epsilon:.4f}" if epsilon else "off",
        )

        return updated_weights, num_samples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Load global weights and evaluate on local validation set.

        Returns:
            (loss, num_eval_samples, metrics)
        """
        self._set_parameters(parameters)
        self.model.eval()

        gallery_dir = f"{self.data_dir}/gallery"
        try:
            dataset = GalleryDataset(gallery_dir)
        except Exception:
            return 0.0, 0, {"accuracy": 0.0}

        if len(dataset) == 0:
            return 0.0, 0, {"accuracy": 0.0}

        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        device = next(iter(self.model._backend.parameters())).device

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                embeddings = self.model(images)
                # Simple nearest-centroid accuracy as proxy metric
                if embeddings.shape[-1] != dataset.num_classes:
                    # Embedding space accuracy: measure top-1 cosine NN accuracy
                    acc = self._embedding_accuracy(embeddings, labels)
                    correct += int(acc * len(labels))
                else:
                    loss = criterion(embeddings, labels)
                    total_loss += loss.item() * len(labels)
                    preds = embeddings.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                total += len(labels)

        accuracy = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)

        logger.info(
            "[%s] Evaluate | accuracy=%.4f | loss=%.4f | samples=%d",
            self.node_id, accuracy, avg_loss, total,
        )
        return avg_loss, total, {"accuracy": accuracy}

    # ── Inference (query routing) ──────────────────────────────────

    def query(self, embedding: np.ndarray) -> Dict:
        """
        Privacy-preserving local search query.
        Called by the query router when an investigator uploads a photo.

        Returns only {"match": bool, "confidence": float}.
        """
        return self.matcher.match(embedding)

    def build_gallery(self):
        """Rebuild the matcher's gallery cache (call after data is ready)."""
        self.matcher.build_gallery()

    # ── Internal helpers ───────────────────────────────────────────

    def _set_parameters(self, parameters: NDArrays):
        """Load coordinator's weight arrays into the local model."""
        state_dict = self.model.state_dict()
        keys = list(state_dict.keys())
        if len(keys) != len(parameters):
            logger.warning(
                "Parameter count mismatch: model=%d, received=%d. "
                "Check that all nodes use the same architecture.",
                len(keys), len(parameters),
            )
            return
        new_state = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(new_state, strict=False)

    def _embedding_accuracy(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        Compute top-1 nearest-centroid accuracy in embedding space.
        Used when model outputs raw embeddings (not class logits).
        """
        unique_labels = labels.unique()
        centroids = torch.stack([
            embeddings[labels == lbl].mean(0) for lbl in unique_labels
        ])
        dists = torch.cdist(embeddings, centroids)
        pred_indices = dists.argmin(dim=1)
        pred_labels = unique_labels[pred_indices]
        return (pred_labels == labels).float().mean().item()


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

def start_client(
    node_id: str,
    data_dir: str,
    server_address: str = "localhost:8080",
    use_dp: bool = True,
    noise_multiplier: float = 1.1,
):
    """Start this org node as a Flower client and connect to coordinator."""
    client = OrgFLClient(
        node_id=node_id,
        data_dir=data_dir,
        use_dp=use_dp,
        noise_multiplier=noise_multiplier,
    )

    logger.info("Connecting to coordinator at %s as node '%s'", server_address, node_id)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Org Node FL Client")
    parser.add_argument("--node",       type=str,   required=True,          help="Node ID (e.g. node_police)")
    parser.add_argument("--data",       type=str,   required=True,          help="Path to node's data directory")
    parser.add_argument("--server",     type=str,   default="localhost:8080",help="Coordinator gRPC address")
    parser.add_argument("--no-dp",      action="store_true",                 help="Disable Differential Privacy")
    parser.add_argument("--dp-noise",   type=float, default=1.1,             help="Opacus DP noise multiplier")
    args = parser.parse_args()

    start_client(
        node_id=args.node,
        data_dir=args.data,
        server_address=args.server,
        use_dp=not args.no_dp,
        noise_multiplier=args.dp_noise,
    )
