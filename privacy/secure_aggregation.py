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

    Args:
        shape: Dimensions of mask array.
        seed: Random seed for deterministic mask generation.

    Returns:
        (mask_a, mask_b): Two np.ndarray masks of given shape such that mask_a + mask_b = 0.
    """
    rng = np.random.RandomState(seed)
    mask_a = rng.randn(*shape).astype(np.float32)
    mask_b = -mask_a
    return mask_a, mask_b


def apply_mask(weights: list, masks: list) -> list:
    """
    Add masks to weight arrays before sending to coordinator.

    Args:
        weights: List of weight np.ndarrays.
        masks: List of mask np.ndarrays corresponding to weights.

    Returns:
        List of masked weight arrays.
    """
    masked_weights = []
    for w, m in zip(weights, masks):
        w_arr = np.asarray(w, dtype=np.float32)
        m_arr = np.asarray(m, dtype=np.float32)
        masked_weights.append(w_arr + m_arr)
    return masked_weights


def simulate_secure_aggregation(client_updates: list) -> list:
    """
    Simulate secure aggregation: sum all (masked) client updates.
    Since masks cancel, the result equals the unmasked sum.

    Args:
        client_updates: List of masked weight update lists from each client.

    Returns:
        Aggregated weights (sum of all updates).
    """
    if not client_updates:
        return []

    num_layers = len(client_updates[0])
    aggregated = []
    for layer_idx in range(num_layers):
        layer_sum = np.zeros_like(np.asarray(client_updates[0][layer_idx], dtype=np.float32))
        for client_weights in client_updates:
            layer_sum += np.asarray(client_weights[layer_idx], dtype=np.float32)
        aggregated.append(layer_sum)

    return aggregated

