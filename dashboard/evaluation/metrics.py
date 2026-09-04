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


def rank1_identification_rate(
    query_embeddings: dict, gallery_embeddings: dict, threshold: float = 0.45
) -> float:
    """
    Compute Rank-1 Identification Rate across gallery embeddings.

    Args:
        query_embeddings: {person_id: query_embedding}
        gallery_embeddings: {person_id: gallery_embedding}
        threshold: Cosine similarity threshold for match.

    Returns:
        rank1_rate: float between 0 and 1
    """
    if not query_embeddings or not gallery_embeddings:
        return 0.0

    correct = 0
    total = len(query_embeddings)

    for q_id, q_emb in query_embeddings.items():
        best_score = -1.0
        best_id = None
        for g_id, g_emb in gallery_embeddings.items():
            sim = float(np.dot(q_emb, g_emb))
            if sim > best_score:
                best_score = sim
                best_id = g_id

        if best_id == q_id and best_score >= threshold:
            correct += 1

    return float(correct / total) if total > 0 else 0.0


def false_match_rate(impostor_scores: list, threshold: float) -> float:
    """Compute False Match Rate (FMR) at a given threshold."""
    if not impostor_scores:
        return 0.0
    arr = np.asarray(impostor_scores)
    return float(np.mean(arr >= threshold))


def false_non_match_rate(genuine_scores: list, threshold: float) -> float:
    """Compute False Non-Match Rate (FNMR) at a given threshold."""
    if not genuine_scores:
        return 0.0
    arr = np.asarray(genuine_scores)
    return float(np.mean(arr < threshold))


def compute_roc(
    genuine_scores: list, impostor_scores: list, thresholds: np.ndarray = None
) -> dict:
    """
    Compute ROC curve data (FMR vs FNMR) across thresholds.

    Returns:
        {
            "thresholds": list[float],
            "fmr": list[float],
            "fnmr": list[float],
            "eer": float,
            "eer_threshold": float,
        }
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    gen = np.asarray(genuine_scores) if genuine_scores else np.array([0.8])
    imp = np.asarray(impostor_scores) if impostor_scores else np.array([0.2])

    fmr_list = []
    fnmr_list = []
    min_eer_diff = 1.0
    eer_val = 0.0
    eer_thresh = 0.45

    for t in thresholds:
        fmr = float(np.mean(imp >= t))
        fnmr = float(np.mean(gen < t))
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)

        diff = abs(fmr - fnmr)
        if diff < min_eer_diff:
            min_eer_diff = diff
            eer_val = (fmr + fnmr) / 2.0
            eer_thresh = float(t)

    return {
        "thresholds": [float(t) for t in thresholds],
        "fmr": fmr_list,
        "fnmr": fnmr_list,
        "eer": round(eer_val, 4),
        "eer_threshold": round(eer_thresh, 4),
    }
