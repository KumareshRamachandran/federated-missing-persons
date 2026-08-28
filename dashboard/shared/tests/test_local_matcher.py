"""
shared/tests/test_local_matcher.py

Unit tests for the privacy-preserving local matcher.
Ensures ONLY binary results are returned (no gallery data leakage).
"""

import pytest
import numpy as np
import tempfile
import os

from federated.client.local_matcher import LocalMatcher


class TestLocalMatcher:
    @pytest.fixture
    def mock_matcher(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            matcher = LocalMatcher(data_dir=tmpdir, threshold=0.45)
            # Add known mock embeddings to gallery
            np.random.seed(42)
            for i in range(5):
                pid = f"person_{i:03d}"
                emb = np.random.randn(512).astype(np.float32)
                emb /= np.linalg.norm(emb)
                matcher.update_gallery_entry(pid, emb)
            yield matcher

    def test_match_returns_binary_only(self, mock_matcher):
        """Match result must only contain 'match' and 'confidence' — no gallery data."""
        query_emb = np.random.randn(512).astype(np.float32)
        query_emb /= np.linalg.norm(query_emb)
        result = mock_matcher.match(query_emb)

        assert "match" in result
        assert "confidence" in result
        assert isinstance(result["match"], (bool, np.bool_))
        assert isinstance(result["confidence"], (float, np.floating))
        # Ensure no gallery identities, names, or raw vectors are leaked
        assert "gallery" not in result
        assert "person_id" not in result
        assert "identity" not in result

    def test_known_match_detected(self, mock_matcher):
        """A query of a person IN the gallery should return match=True."""
        known_emb = mock_matcher.gallery["person_001"]
        # Add slight realistic noise
        noisy_query = known_emb + np.random.normal(0, 0.02, 512).astype(np.float32)
        noisy_query /= np.linalg.norm(noisy_query)

        result = mock_matcher.match(noisy_query)
        assert result["match"] is True
        assert result["confidence"] >= 0.85

    def test_unknown_person_not_matched(self, mock_matcher):
        """A query of a person NOT in the gallery should return match=False."""
        # Orthogonal / random embedding
        np.random.seed(999)
        unknown_emb = np.random.randn(512).astype(np.float32)
        unknown_emb /= np.linalg.norm(unknown_emb)

        result = mock_matcher.match(unknown_emb)
        assert result["match"] is False
        assert result["confidence"] < mock_matcher.threshold

    def test_cosine_similarity_range(self, mock_matcher):
        """Confidence score must be between 0 and 1."""
        query_emb = np.random.randn(512).astype(np.float32)
        result = mock_matcher.match(query_emb)
        assert 0.0 <= result["confidence"] <= 1.0

