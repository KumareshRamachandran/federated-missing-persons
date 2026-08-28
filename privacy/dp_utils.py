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


def clip_gradients(model: torch.nn.Module, max_norm: float = 1.0) -> float:
    """
    Clip model gradients to max_norm (L2 norm).

    Args:
        model: PyTorch module whose gradients should be clipped.
        max_norm: Maximum L2 norm for gradients.

    Returns:
        total_norm: Total L2 norm of model parameters before clipping.
    """
    if model is None:
        return 0.0
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)


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
    if noise_multiplier <= 0.0:
        return [np.copy(w) for w in weights]

    std = noise_multiplier * sensitivity
    noisy_weights = []
    for w in weights:
        w_arr = np.asarray(w, dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=std, size=w_arr.shape).astype(w_arr.dtype)
        noisy_weights.append(w_arr + noise)
    return noisy_weights


def compute_epsilon(
    noise_multiplier: float,
    num_samples: int,
    batch_size: int,
    num_rounds: int,
    delta: float = 1e-5
) -> float:
    """
    Estimate the DP privacy budget (epsilon) consumed over multiple training rounds.
    Uses Opacus RdpAccountant if available; otherwise falls back to analytical RDP composition.

    Args:
        noise_multiplier: Noise multiplier sigma used during training.
        num_samples: Total number of training samples across dataset.
        batch_size: Batch size per step.
        num_rounds: Total number of federated rounds / training steps.
        delta: Target privacy parameter delta (default 1e-5).

    Returns:
        epsilon: Privacy budget consumed.
    """
    if noise_multiplier <= 0.0:
        return float('inf')

    try:
        from opacus.accountants import RdpAccountant
        accountant = RdpAccountant()
        sample_rate = batch_size / max(num_samples, 1)
        for _ in range(num_rounds):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        epsilon = accountant.get_epsilon(delta=delta)
        return float(epsilon)
    except Exception:
        # Analytical Rényi Differential Privacy (RDP) calculation fallback
        sample_rate = batch_size / max(num_samples, 1)
        if sample_rate >= 1.0:
            # Standard Gaussian mechanism composition without subsampling
            eps = (num_rounds / (2.0 * (noise_multiplier ** 2))) + (
                np.sqrt(2.0 * num_rounds * np.log(1.25 / delta)) / noise_multiplier
            )
            return float(eps)

        # Optimize RDP order alpha in range [1.1, 128.0]
        alphas = np.linspace(1.1, 128.0, 200)
        best_eps = float('inf')
        for alpha in alphas:
            # Subsampled RDP upper bound approximation: alpha * q^2 / sigma^2
            rdp = alpha * (sample_rate ** 2) * num_rounds / (noise_multiplier ** 2)
            eps = rdp + np.log(1.0 / delta) / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps
        return float(best_eps)

