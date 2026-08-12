"""
evaluation/metrics.py

Evaluation metrics for the federated face recognition system.

Metrics:
  - Rank-1 Identification Rate: % of queries where top match is correct
  - False Match Rate (FMR): % of non-matching pairs incorrectly matched
  - False Non-Match Rate (FNMR): % of matching pairs incorrectly rejected
  - ROC Curve data for threshold selection
  - Federated vs. Centralized accuracy comparison
"""

import numpy as np


def rank1_identification_rate(query_embeddings: dict, gallery_embeddings: dict, threshold: float = 0.6) -> float:
    """
    Compute Rank-1 Identification Rate across all federated nodes.

    Args:
        query_embeddings: {person_id: query_embedding}
        gallery_embeddings: {person_id: gallery_embedding}
        threshold: Cosine similarity threshold for match.

    Returns:
        rank1_rate: float between 0 and 1
    """
    # TODO: For each query, find gallery embedding with highest cosine similarity
    # TODO: If best match person_id == query person_id AND similarity > threshold → correct
    # TODO: Return correct / total
    pass


def false_match_rate(impostor_pairs: list, threshold: float) -> float:
    """Compute False Match Rate from a list of (emb_a, emb_b) non-matching pairs."""
    # TODO: Count pairs where cosine_similarity > threshold
    pass


def false_non_match_rate(genuine_pairs: list, threshold: float) -> float:
    """Compute False Non-Match Rate from a list of (emb_a, emb_b) matching pairs."""
    # TODO: Count pairs where cosine_similarity <= threshold
    pass


def compute_roc(genuine_pairs: list, impostor_pairs: list, thresholds: np.ndarray = None) -> dict:
    """
    Compute ROC curve data (FMR vs FNMR) for threshold selection.

    Returns:
        {"thresholds": [...], "fmr": [...], "fnmr": [...]}
    """
    # TODO: Iterate over thresholds, compute FMR and FNMR at each
    pass
