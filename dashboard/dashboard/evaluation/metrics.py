"""
dashboard/evaluation/metrics.py

Evaluation metrics for the federated face recognition and missing person identification system.

Metrics implemented:
  - Rank-1 Identification Rate (Closed-set & Open-set matching)
  - Cumulative Match Characteristic (CMC) curve for Rank-k accuracy
  - False Match Rate (FMR / False Acceptance Rate FAR)
  - False Non-Match Rate (FNMR / False Rejection Rate FRR)
  - Receiver Operating Characteristic (ROC) curve analysis
  - Equal Error Rate (EER) and optimal threshold calculation
  - Area Under the Curve (AUC)

Member: Aswin Maheswaran (23BCE8540) — UI Dashboard & Integration Module
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Safe against zero-norm vectors.
    """
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def rank1_identification_rate(
    query_embeddings: Union[Dict[str, np.ndarray], List[Tuple[str, np.ndarray]]],
    gallery_embeddings: Union[Dict[str, np.ndarray], List[Tuple[str, np.ndarray]]],
    threshold: float = 0.45,
) -> float:
    """
    Compute Rank-1 Identification Rate across gallery embeddings.

    For each query embedding:
      1. Find gallery subject with the maximum cosine similarity.
      2. If best match subject == query subject and similarity >= threshold, it is a hit.

    Args:
        query_embeddings: Dict mapping person_id -> embedding (512,) or list of (person_id, embedding).
        gallery_embeddings: Dict mapping person_id -> embedding (512,) or list of (person_id, embedding).
        threshold: Minimum cosine similarity required to declare a match.

    Returns:
        rank1_rate: Float between 0.0 and 1.0 (0.0 if no queries provided).
    """
    queries = (
        list(query_embeddings.items())
        if isinstance(query_embeddings, dict)
        else query_embeddings
    )
    galleries = (
        list(gallery_embeddings.items())
        if isinstance(gallery_embeddings, dict)
        else gallery_embeddings
    )

    if not queries or not galleries:
        return 0.0

    # Pre-normalize gallery embeddings
    gallery_ids = [gid for gid, _ in galleries]
    gallery_matrix = np.array([emb.flatten() for _, emb in galleries], dtype=np.float32)
    norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    gallery_matrix = gallery_matrix / norms

    correct_matches = 0

    for q_id, q_emb in queries:
        q_vec = np.asarray(q_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            continue
        q_vec = q_vec / q_norm

        # Vectorized cosine similarities against all gallery entries
        sims = np.dot(gallery_matrix, q_vec)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_id = gallery_ids[best_idx]

        if best_id == q_id and best_score >= threshold:
            correct_matches += 1

    return float(correct_matches / len(queries))


def cumulative_match_characteristic(
    query_embeddings: Union[Dict[str, np.ndarray], List[Tuple[str, np.ndarray]]],
    gallery_embeddings: Union[Dict[str, np.ndarray], List[Tuple[str, np.ndarray]]],
    ranks: Optional[List[int]] = None,
    threshold: float = 0.0,
) -> Dict[int, float]:
    """
    Compute Cumulative Match Characteristic (CMC) for ranks 1..k.

    Args:
        query_embeddings: Queries {person_id: embedding} or list of tuples.
        gallery_embeddings: Gallery {person_id: embedding} or list of tuples.
        ranks: List of ranks to calculate (default [1, 5, 10, 20]).
        threshold: Minimum similarity score filter.

    Returns:
        Dict mapping rank -> identification rate (e.g. {1: 0.95, 5: 0.98, ...}).
    """
    if ranks is None:
        ranks = [1, 5, 10, 20]

    queries = (
        list(query_embeddings.items())
        if isinstance(query_embeddings, dict)
        else query_embeddings
    )
    galleries = (
        list(gallery_embeddings.items())
        if isinstance(gallery_embeddings, dict)
        else gallery_embeddings
    )

    if not queries or not galleries:
        return {r: 0.0 for r in ranks}

    gallery_ids = [gid for gid, _ in galleries]
    gallery_matrix = np.array([emb.flatten() for _, emb in galleries], dtype=np.float32)
    norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    gallery_matrix = gallery_matrix / norms

    rank_hits = {r: 0 for r in ranks}

    for q_id, q_emb in queries:
        q_vec = np.asarray(q_emb, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            continue
        q_vec = q_vec / q_norm

        sims = np.dot(gallery_matrix, q_vec)
        sorted_indices = np.argsort(-sims)

        for r in ranks:
            top_r_indices = sorted_indices[: min(r, len(sorted_indices))]
            matched = False
            for idx in top_r_indices:
                if gallery_ids[idx] == q_id and sims[idx] >= threshold:
                    matched = True
                    break
            if matched:
                rank_hits[r] += 1

    total = len(queries)
    return {r: float(rank_hits[r] / total) for r in ranks}


def _extract_pair_similarities(pairs: list) -> np.ndarray:
    """Helper to convert pairs of embeddings or raw similarity floats into a 1D numpy array."""
    if not pairs:
        return np.array([], dtype=np.float32)

    first = pairs[0]
    if isinstance(first, (float, int, np.floating, np.integer)):
        return np.asarray(pairs, dtype=np.float32)

    scores = []
    for item in pairs:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            scores.append(_cosine_similarity(item[0], item[1]))
        elif isinstance(item, (float, int, np.floating, np.integer)):
            scores.append(float(item))
    return np.asarray(scores, dtype=np.float32)


def false_match_rate(impostor_pairs: list, threshold: float) -> float:
    """
    Compute False Match Rate (FMR / FAR) from impostor (non-matching) pairs.

    FMR(τ) = (Number of impostor pairs with similarity >= threshold) / Total impostor pairs

    Args:
        impostor_pairs: List of (emb_a, emb_b) pairs from different people,
                        or list of pre-calculated impostor similarity scores.
        threshold: Cosine similarity threshold.

    Returns:
        fmr: Float between 0.0 and 1.0.
    """
    scores = _extract_pair_similarities(impostor_pairs)
    if len(scores) == 0:
        return 0.0
    false_matches = int(np.sum(scores >= threshold))
    return float(false_matches / len(scores))


def false_non_match_rate(genuine_pairs: list, threshold: float) -> float:
    """
    Compute False Non-Match Rate (FNMR / FRR) from genuine (same identity) pairs.

    FNMR(τ) = (Number of genuine pairs with similarity < threshold) / Total genuine pairs

    Args:
        genuine_pairs: List of (emb_a, emb_b) pairs from the same person,
                       or list of pre-calculated genuine similarity scores.
        threshold: Cosine similarity threshold.

    Returns:
        fnmr: Float between 0.0 and 1.0.
    """
    scores = _extract_pair_similarities(genuine_pairs)
    if len(scores) == 0:
        return 0.0
    false_non_matches = int(np.sum(scores < threshold))
    return float(false_non_matches / len(scores))


def compute_roc(
    genuine_pairs: list,
    impostor_pairs: list,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Union[List[float], float]]:
    """
    Compute ROC curve data across a sweep of cosine similarity thresholds.

    Args:
        genuine_pairs: List of genuine embedding pairs or similarity scores.
        impostor_pairs: List of impostor embedding pairs or similarity scores.
        thresholds: 1D array of thresholds to evaluate. Defaults to np.linspace(0.0, 1.0, 101).

    Returns:
        {
            "thresholds": List[float],
            "fmr": List[float],         # False Match Rate (FPR)
            "fnmr": List[float],        # False Non-Match Rate (1 - TPR)
            "tpr": List[float],         # True Positive Rate (1 - FNMR)
            "eer": float,               # Equal Error Rate
            "eer_threshold": float,     # Operating threshold at EER
            "auc": float,               # Area Under the ROC Curve
        }
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    gen_scores = _extract_pair_similarities(genuine_pairs)
    imp_scores = _extract_pair_similarities(impostor_pairs)

    fmr_list: List[float] = []
    fnmr_list: List[float] = []
    tpr_list: List[float] = []

    for tau in thresholds:
        fmr = float(np.mean(imp_scores >= tau)) if len(imp_scores) > 0 else 0.0
        fnmr = float(np.mean(gen_scores < tau)) if len(gen_scores) > 0 else 0.0
        tpr = 1.0 - fnmr

        fmr_list.append(round(fmr, 6))
        fnmr_list.append(round(fnmr, 6))
        tpr_list.append(round(tpr, 6))

    # Compute Equal Error Rate (where |FMR - FNMR| is minimized)
    diffs = np.abs(np.array(fmr_list) - np.array(fnmr_list))
    eer_idx = int(np.argmin(diffs)) if len(diffs) > 0 else 0
    eer = float((fmr_list[eer_idx] + fnmr_list[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Compute AUC using trapezoidal rule (sorted by FPR/FMR)
    sorted_indices = np.argsort(fmr_list)
    sorted_fpr = np.array(fmr_list)[sorted_indices]
    sorted_tpr = np.array(tpr_list)[sorted_indices]
    auc = float(np.trapz(sorted_tpr, sorted_fpr))
    auc = float(np.clip(auc, 0.0, 1.0))

    return {
        "thresholds": [round(float(t), 4) for t in thresholds],
        "fmr": fmr_list,
        "fnmr": fnmr_list,
        "tpr": tpr_list,
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "auc": round(auc, 4),
    }


def compute_eer(
    genuine_pairs: list,
    impostor_pairs: list,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the optimal threshold.

    Returns:
        (eer, eer_threshold)
    """
    roc = compute_roc(genuine_pairs, impostor_pairs, thresholds)
    return float(roc["eer"]), float(roc["eer_threshold"])
