"""
face_engine/embedder.py

Generates 512-d ArcFace face embeddings from aligned face images.
Used by both the coordinator (query embedding) and the org nodes (gallery building).
"""

import numpy as np
import torch
from face_engine.model import ArcFaceModel
from face_engine.detector import detect_face


_model = None  # Singleton model instance


def _load_model(weights_path: str = None) -> ArcFaceModel:
    """Load ArcFace model once and cache it."""
    global _model
    if _model is None:
        _model = ArcFaceModel(weights_path=weights_path)
        _model.eval()
    return _model


def generate_embedding(image_path: str, weights_path: str = None) -> np.ndarray:
    """
    Full pipeline: detect face → align → generate embedding.

    Args:
        image_path: Path to the input image.
        weights_path: Optional path to ArcFace weights.

    Returns:
        embedding: np.ndarray of shape (512,), or None if no face detected.
    """
    # TODO: detect_face(image_path) → aligned_face
    # TODO: Convert to tensor, normalize
    # TODO: model.get_embedding(tensor) → embedding
    # TODO: Return as numpy array
    pass


def generate_embeddings_batch(image_paths: list, weights_path: str = None) -> dict:
    """
    Generate embeddings for a list of images.

    Returns:
        dict: {image_path: embedding_array}
    """
    # TODO: Loop generate_embedding() over image_paths
    pass
