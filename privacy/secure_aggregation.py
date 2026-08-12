"""
privacy/secure_aggregation.py

Simulated Secure Aggregation.
In production, this would use cryptographic pairwise masking (as in Bonawitz et al., 2017).
For the capstone, we simulate the protocol to demonstrate the concept:
  - Each client masks its updates before sending
  - Masks cancel out at the coordinator during aggregation
  - Coordinator sees only the sum, never individual updates
"""

import numpy as np


def generate_mask_pair(shape: tuple, seed: int) -> tuple:
    """
    Generate a pair of cancelling masks for two clients.
    mask_a + mask_b = 0 (they cancel in aggregation).

    Returns:
        (mask_a, mask_b): Two np.ndarray masks of given shape.
    """
    # TODO: np.random.seed(seed); mask = np.random.randn(*shape)
    # TODO: return mask, -mask
    pass


def apply_mask(weights: list, masks: list) -> list:
    """Add masks to weight arrays before sending to coordinator."""
    # TODO: return [w + m for w, m in zip(weights, masks)]
    pass


def simulate_secure_aggregation(client_updates: list) -> list:
    """
    Simulate secure aggregation: sum all (masked) client updates.
    Since masks cancel, the result equals the unmasked sum.

    Args:
        client_updates: List of masked weight update lists from each client.

    Returns:
        Aggregated weights (sum of all updates).
    """
    # TODO: Element-wise sum across all client_updates
    pass
