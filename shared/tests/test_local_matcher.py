"""
tests/test_local_matcher.py

Unit tests for the privacy-preserving local matcher.
Ensures ONLY binary results are returned (no gallery data leakage).
"""

import pytest
import numpy as np


class TestLocalMatcher:
    def test_match_returns_binary_only(self):
        """Match result must only contain 'match' and 'confidence' — no gallery data."""
        # TODO: matcher = LocalMatcher(model, data_dir="tests/assets/gallery")
        # TODO: matcher.build_gallery()
        # TODO: result = matcher.match(np.random.randn(512))
        # TODO: assert set(result.keys()) == {"match", "confidence"}
        pass

    def test_known_match_detected(self):
        """A query of a person IN the gallery should return match=True."""
        pass

    def test_unknown_person_not_matched(self):
        """A query of a person NOT in the gallery should return match=False."""
        pass

    def test_cosine_similarity_range(self):
        """Confidence score must be between 0 and 1."""
        pass
