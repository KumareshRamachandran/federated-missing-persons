"""
tests/test_face_engine.py

Unit tests for the face detection and embedding pipeline.
"""

import pytest
import numpy as np


class TestFaceDetector:
    def test_detect_face_returns_array(self):
        """detect_face() should return a (112, 112, 3) numpy array."""
        # TODO: from face_engine.detector import detect_face
        # TODO: result = detect_face("tests/assets/sample_face.jpg")
        # TODO: assert result is not None
        # TODO: assert result.shape == (112, 112, 3)
        pass

    def test_detect_face_no_face(self):
        """detect_face() should return None when no face is in the image."""
        # TODO: result = detect_face("tests/assets/no_face.jpg")
        # TODO: assert result is None
        pass


class TestEmbedder:
    def test_embedding_shape(self):
        """generate_embedding() should return a (512,) vector."""
        # TODO: from face_engine.embedder import generate_embedding
        # TODO: emb = generate_embedding("tests/assets/sample_face.jpg")
        # TODO: assert emb.shape == (512,)
        pass

    def test_embedding_normalized(self):
        """Embedding should be L2-normalized (norm ≈ 1.0)."""
        # TODO: emb = generate_embedding("tests/assets/sample_face.jpg")
        # TODO: assert abs(np.linalg.norm(emb) - 1.0) < 1e-5
        pass

    def test_same_person_high_similarity(self):
        """Two images of the same person should have cosine similarity > 0.6."""
        pass

    def test_different_person_low_similarity(self):
        """Images of different persons should have cosine similarity < 0.4."""
        pass
