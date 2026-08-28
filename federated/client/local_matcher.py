"""
federated/client/local_matcher.py

Privacy-Preserving Local Inference Engine.

KEY DESIGN PRINCIPLE:
  The query face embedding arrives from the coordinator.
  This module matches it against the org's private gallery LOCALLY.
  ONLY a binary Match/No-Match + confidence score is ever returned.
  No gallery embeddings, person IDs, or images leave this node.

  This is the core novelty contribution vs. all surveyed papers
  ([2],[4],[6],[7],[10]) which still centralize inference.

Author: R Kumaresh (23BCE9585) — Federated Learning Module
"""

import os
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class LocalMatcher:
    """
    Performs privacy-preserving face matching inside an org node's boundary.

    Gallery embeddings are computed once and cached on disk (.pkl).
    During a query, only cosine similarity scores are computed — the
    gallery content is never serialised or transmitted.
    """

    CACHE_FILE = "gallery_cache.pkl"

    def __init__(
        self,
        data_dir: str,
        threshold: float = 0.45,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            data_dir:  Path to this org node's gallery directory.
                       Structure: data_dir/<person_id>/<image>.jpg
            threshold: Cosine similarity threshold for a positive match.
                       ArcFace embeddings: τ ≈ 0.45 is a good starting point
                       (from [1]: YOLO–ArcFace used τ analysis).
            cache_dir: Where to store the gallery embedding cache.
                       Defaults to data_dir/.cache/
        """
        self.data_dir = data_dir
        self.threshold = threshold
        self.cache_dir = cache_dir or os.path.join(data_dir, ".cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # {person_id: np.ndarray (512,)} — loaded from cache or rebuilt
        self.gallery: Dict[str, np.ndarray] = {}
        self._load_cache()

    # ──────────────────────────────────────────────────────────────
    # Gallery Management
    # ──────────────────────────────────────────────────────────────

    def build_gallery(self, embedder_fn=None, force_rebuild: bool = False) -> int:
        """
        Pre-compute and cache embeddings for all persons in the local gallery.

        Args:
            embedder_fn:   Callable (image_path -> np.ndarray embedding).
                           Defaults to importing from vision.embedder if None.
            force_rebuild: If True, recomputes even if cache exists.

        Returns:
            Number of identities indexed in the gallery.
        """
        cache_path = os.path.join(self.cache_dir, self.CACHE_FILE)
        if os.path.exists(cache_path) and not force_rebuild:
            logger.info("Gallery cache already exists. Use force_rebuild=True to refresh.")
            return len(self.gallery)

        if embedder_fn is None:
            # Import lazily to avoid hard dependency at module load time
            from vision.embedder import generate_embedding
            embedder_fn = generate_embedding

        gallery_path = Path(self.data_dir) / "gallery"
        if not gallery_path.exists():
            logger.warning("Gallery directory not found: %s", gallery_path)
            return 0

        new_gallery: Dict[str, np.ndarray] = {}
        person_dirs = [p for p in gallery_path.iterdir() if p.is_dir()]

        logger.info("Building gallery for %d identities in %s ...", len(person_dirs), gallery_path)

        for person_dir in person_dirs:
            person_id = person_dir.name
            embeddings = []

            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                emb = embedder_fn(str(img_path))
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                # Average multiple images of the same person → more robust
                new_gallery[person_id] = np.mean(embeddings, axis=0)
                new_gallery[person_id] /= np.linalg.norm(new_gallery[person_id])  # re-normalise

        self.gallery = new_gallery
        self._save_cache()
        logger.info("Gallery built: %d identities indexed.", len(self.gallery))
        return len(self.gallery)

    def update_gallery_entry(self, person_id: str, embedding: np.ndarray):
        """
        Add or update a single identity in the gallery (used after a confirmed match).
        Triggers a cache save.
        """
        normed = embedding / (np.linalg.norm(embedding) + 1e-10)
        self.gallery[person_id] = normed
        self._save_cache()
        logger.info("Gallery updated for person_id=%s", person_id)

    # ──────────────────────────────────────────────────────────────
    # Matching (Privacy-Preserving Inference)
    # ──────────────────────────────────────────────────────────────

    def match(self, query_embedding: np.ndarray) -> Dict:
        """
        Match a query face embedding against this org's private gallery.

        PRIVACY CONTRACT:
          Input:  512-d embedding vector (mathematical representation only)
          Output: {"match": bool, "confidence": float}
          ⚠️  No person IDs, images, or gallery data are ever returned.

        Args:
            query_embedding: L2-normalized 512-d ArcFace embedding.

        Returns:
            {"match": bool, "confidence": float}
            confidence ∈ [0.0, 1.0] — cosine similarity of best match.
        """
        if not self.gallery:
            logger.warning("Gallery is empty. Run build_gallery() first.")
            return {"match": False, "confidence": 0.0}

        # Normalise query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)

        # Compute cosine similarities — all gallery members
        best_score = -1.0
        for _, gallery_emb in self.gallery.items():
            score = self._cosine_similarity(query_norm, gallery_emb)
            if score > best_score:
                best_score = score

        confidence = float(np.clip(best_score, 0.0, 1.0))
        matched = confidence >= self.threshold

        logger.debug(
            "Query matched=%s | confidence=%.4f | threshold=%.2f | gallery_size=%d",
            matched, confidence, self.threshold, len(self.gallery)
        )

        # Return ONLY binary result — no gallery data
        return {"match": matched, "confidence": round(confidence, 4)}

    def match_batch(self, query_embeddings: list) -> list:
        """
        Match multiple query embeddings (e.g., video frame sequence).

        Returns:
            List of {"match": bool, "confidence": float} dicts.
        """
        return [self.match(qe) for qe in query_embeddings]

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two L2-normalized vectors.
        Since both are pre-normalised, this reduces to a dot product.
        """
        return float(np.dot(a, b))

    def _save_cache(self):
        """Persist gallery embeddings to disk."""
        path = os.path.join(self.cache_dir, self.CACHE_FILE)
        with open(path, "wb") as f:
            pickle.dump(self.gallery, f)
        logger.debug("Gallery cache saved to %s (%d entries)", path, len(self.gallery))

    def _load_cache(self):
        """Load gallery embeddings from disk cache if it exists."""
        path = os.path.join(self.cache_dir, self.CACHE_FILE)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.gallery = pickle.load(f)
            logger.info("Loaded gallery cache: %d identities from %s", len(self.gallery), path)
        else:
            self.gallery = {}
            logger.info("No gallery cache found at %s. Run build_gallery().", path)

    def gallery_size(self) -> int:
        """Return number of identities currently in the gallery."""
        return len(self.gallery)

    def set_threshold(self, threshold: float):
        """Update the match threshold (useful for threshold tuning experiments)."""
        self.threshold = threshold
        logger.info("Match threshold updated to %.4f", threshold)
