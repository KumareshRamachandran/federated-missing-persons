"""
client/local_trainer.py

Handles local training of the ArcFace model on the org's private dataset.
Uses Opacus (PyTorch DP) for differentially private gradient updates.
"""

import torch
from torch.utils.data import DataLoader


class LocalTrainer:
    """Trains the model locally using the org's private image gallery."""

    def __init__(self, model, data_dir: str, epochs: int = 1, lr: float = 1e-4):
        self.model = model
        self.data_dir = data_dir
        self.epochs = epochs
        self.lr = lr

    def train(self, use_dp: bool = True, noise_multiplier: float = 1.0, max_grad_norm: float = 1.0):
        """
        Run local training for self.epochs.

        Args:
            use_dp: Whether to apply Differential Privacy via Opacus.
            noise_multiplier: DP noise scale.
            max_grad_norm: DP gradient clipping norm.

        Returns:
            Updated model state_dict and training loss.
        """
        # TODO: Load local dataset from self.data_dir
        # TODO: Create DataLoader
        # TODO: If use_dp, wrap optimizer + model with Opacus PrivacyEngine
        # TODO: Training loop (forward pass, loss, backward, optimizer step)
        # TODO: Return model.state_dict(), avg_loss
        pass
