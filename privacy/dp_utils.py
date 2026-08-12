"""
privacy/dp_utils.py

Differential Privacy utilities.
Provides functions to:
  - Add Gaussian noise to model gradients/weights (for DP-SGD)
  - Clip gradients to a maximum norm
  - Compute the privacy budget (epsilon) spent per round
"""

import numpy as np
import torch


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0):
    """Clip model gradients to max_norm (L2 norm)."""
    # TODO: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    pass


def add_gaussian_noise(weights: list, noise_multiplier: float, sensitivity: float = 1.0) -> list:
    """
    Add calibrated Gaussian noise to a list of weight arrays for DP.

    Args:
        weights: List of np.ndarray model weight arrays.
        noise_multiplier: Controls noise scale (higher = more privacy, less accuracy).
        sensitivity: Global sensitivity (usually == max_grad_norm).

    Returns:
        Noisy weight arrays.
    """
    # TODO: For each weight array, add np.random.normal(0, noise_multiplier * sensitivity, shape)
    pass


def compute_epsilon(
    noise_multiplier: float,
    num_samples: int,
    batch_size: int,
    num_rounds: int,
    delta: float = 1e-5
) -> float:
    """
    Estimate the DP privacy budget (epsilon) consumed.
    Uses the moments accountant (via Google DP library or Opacus).

    Returns:
        epsilon: Privacy budget consumed.
    """
    # TODO: Use Opacus or autodp to compute epsilon
    pass
