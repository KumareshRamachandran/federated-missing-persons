"""
vision/arcface_model.py

ArcFace model wrapper for federated learning.

Uses facenet_pytorch's InceptionResnetV1 (pre-trained on VGGFace2) as the
backbone. This is a proper PyTorch nn.Module, making it directly compatible
with Flower's weight extraction/aggregation (FedAvg).

Output: 512-dimensional L2-normalized face embeddings.

Design note:
    InsightFace's original ArcFace ships as ONNX (not a raw nn.Module), which
    makes FL weight aggregation difficult. InceptionResnetV1 from facenet_pytorch
    is pre-trained, produces 512-d embeddings, and is fully FL-compatible.
    It is used for the FL training phase; InsightFace is reserved for the
    paper's centralized baseline comparison.

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class ArcFaceModel(nn.Module):
    """
    Wrapper around InceptionResnetV1 pre-trained on VGGFace2.

    Input:  Batch of aligned face tensors [B, 3, 112, 112] in [-1, 1]
    Output: L2-normalized 512-d face embeddings [B, 512]
    """

    EMBEDDING_DIM = 512

    def __init__(
        self,
        pretrained: str | None = "vggface2",
        weights_path: str | None = None,
        freeze_base: bool = False,
    ):
        """
        Args:
            pretrained:   'vggface2' (recommended) or 'casia-webface'.
                          Set to None to initialise with random weights.
            weights_path: Path to a local .pt checkpoint to load instead of
                          downloading pretrained weights.
            freeze_base:  If True, freeze all backbone layers except the final
                          embedding layer (useful for fine-tuning).
        """
        super().__init__()

        # Load InceptionResnetV1 backbone
        if weights_path is not None:
            # Custom checkpoint — load architecture without pretrained weights,
            # then overwrite with the provided state dict.
            self.backbone = InceptionResnetV1(pretrained=None, classify=False)
            state = torch.load(weights_path, map_location="cpu")
            # Support both raw state dicts and checkpoint dicts
            state_dict = state.get("model_state_dict", state)
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            self.backbone = InceptionResnetV1(
                pretrained=pretrained,
                classify=False,   # embedding mode — no softmax head
            )

        if freeze_base:
            self._freeze_base_layers()

        # Default to evaluation mode; caller switches to train() for FL rounds
        self.eval()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of aligned face tensors [B, 3, 112, 112] in [-1, 1].

        Returns:
            embeddings: [B, 512] L2-normalized face embeddings.
        """
        embeddings = self.backbone(x)        # [B, 512]
        embeddings = F.normalize(embeddings, p=2, dim=1)  # L2 normalise
        return embeddings

    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: generate embedding for a single face tensor.

        Args:
            face_tensor: [3, 112, 112] or [1, 3, 112, 112].

        Returns:
            embedding: [512] tensor (CPU, detached).
        """
        if face_tensor.dim() == 3:
            face_tensor = face_tensor.unsqueeze(0)  # add batch dim → [1, 3, H, W]

        device = next(self.parameters()).device
        face_tensor = face_tensor.to(device)

        with torch.no_grad():
            embedding = self.forward(face_tensor)   # [1, 512]

        return embedding.squeeze(0).cpu().detach()  # [512]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_parameters(self) -> list:
        """
        Return model parameters as a list of numpy arrays.
        Called by Flower's NumPy client during FL rounds.
        """
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, parameters: list) -> None:
        """
        Load parameters from a list of numpy arrays.
        Called by Flower's NumPy client after receiving global model weights.
        """
        import numpy as np
        for p, new_val in zip(self.parameters(), parameters):
            p.data = torch.tensor(new_val, dtype=p.dtype).to(p.device)

    def save_checkpoint(self, path: str) -> None:
        """Save model weights to disk."""
        torch.save({"model_state_dict": self.state_dict()}, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> "ArcFaceModel":
        """Load a model from a saved checkpoint."""
        model = cls(pretrained=None, weights_path=path)
        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze_base_layers(self) -> None:
        """Freeze all layers except the last block (last linear + BN)."""
        for name, param in self.backbone.named_parameters():
            # Keep the final linear layer trainable
            if "last_linear" not in name and "last_bn" not in name:
                param.requires_grad = False
