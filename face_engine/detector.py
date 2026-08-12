"""
face_engine/detector.py

Face detection and alignment using MTCNN (via facenet-pytorch).
Detects faces from input images and returns aligned 112x112 crops
ready for ArcFace embedding generation.
"""

import numpy as np
from PIL import Image


def detect_face(image_path: str) -> np.ndarray:
    """
    Detect and align the primary face from an image.

    Args:
        image_path: Path to the input image.

    Returns:
        aligned_face: np.ndarray of shape (112, 112, 3), or None if no face detected.
    """
    # TODO: Load image via PIL
    # TODO: Initialize MTCNN detector
    # TODO: Detect face, get bounding box and landmarks
    # TODO: Align and crop face to 112x112
    # TODO: Return aligned_face numpy array
    pass


def detect_faces_batch(image_paths: list) -> list:
    """Detect and align faces from a batch of image paths."""
    # TODO: Loop detect_face() over image_paths
    # TODO: Return list of aligned faces (skip None results)
    pass
