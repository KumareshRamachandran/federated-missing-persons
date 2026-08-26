"""
face_engine/model.py

ArcFace model wrapper.
Loads a pre-trained ArcFace backbone (iResNet50 from InsightFace)
for face embedding generation.
"""

import torch
import torch.nn as nn


class ArcFaceModel(nn.Module):
    """
    Wrapper around a pre-trained ArcFace backbone.
    Outputs 512-dimensional face embeddings.
    """

    def __init__(self, weights_path: str = None):
        super().__init__()
        # TODO: Load iResNet50 backbone from insightface or torchvision
        # TODO: If weights_path provided, load pretrained weights
        # TODO: Set model to eval mode by default
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of aligned face tensors [B, 3, 112, 112]
        Returns:
            embeddings: [B, 512] L2-normalized face embeddings
        """
        # TODO: Pass through backbone
        # TODO: L2-normalize output
        pass

    def get_embedding(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """Convenience method for single-face embedding."""
        # TODO: Add batch dim, run forward(), return squeezed result
        pass
