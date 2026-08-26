"""
client/local_matcher.py

Privacy-preserving local inference.
Receives a query face embedding from the coordinator.
Performs cosine similarity matching against the local gallery — locally.
Returns ONLY Match/No-Match + confidence score. No gallery data is ever sent out.
"""

import numpy as np
from face_engine.embedder import generate_embedding


class LocalMatcher:
    """Performs local face matching against the org's private gallery."""

    def __init__(self, model, data_dir: str, threshold: float = 0.6):
        self.model = model
        self.data_dir = data_dir
        self.threshold = threshold
        self.gallery_embeddings = {}  # {person_id: embedding_vector}

    def build_gallery(self):
        """Pre-compute and cache embeddings for all persons in the local gallery."""
        # TODO: Walk through data_dir images
        # TODO: Generate embedding for each image via generate_embedding()
        # TODO: Store in self.gallery_embeddings
        pass

    def match(self, query_embedding: np.ndarray) -> dict:
        """
        Match a query embedding against local gallery.

        Args:
            query_embedding: 512-d face embedding from coordinator.

        Returns:
            {"match": bool, "confidence": float}
            NOTE: No gallery content, person IDs, or images are returned.
        """
        # TODO: Compute cosine similarity between query and all gallery embeddings
        # TODO: If max similarity > self.threshold → match = True
        # TODO: Return {"match": match, "confidence": max_similarity}
        pass

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # TODO: return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        pass
