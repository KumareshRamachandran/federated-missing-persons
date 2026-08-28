"""
vision/face_detector.py

Face detection and alignment using MTCNN (via facenet-pytorch).
Detects faces from input images and returns aligned 112x112 crops
ready for ArcFace embedding generation.

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

# ── Module-level singleton ────────────────────────────────────────────────────
# Instantiate MTCNN once to avoid re-loading on every call.
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_mtcnn: MTCNN | None = None


def _get_mtcnn() -> MTCNN:
    """Return the shared MTCNN instance, creating it on first call."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=112,   # ArcFace expects 112×112 input
            margin=20,        # Extra margin around detected face
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net
            factor=0.709,
            post_process=True,            # Returns tensor in [-1, 1] range
            keep_all=False,               # Return only the highest-confidence face
            device=_device,
        )
    return _mtcnn


# ── Public API ────────────────────────────────────────────────────────────────

def detect_face(image_path: str) -> np.ndarray | None:
    """
    Detect and align the primary face from an image file.

    Args:
        image_path: Path to the input image (any PIL-readable format).

    Returns:
        aligned_face: np.ndarray of shape (112, 112, 3), dtype uint8,
                      or None if no face is detected.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError):
        return None

    mtcnn = _get_mtcnn()

    # detect_and_save returns a tensor [3, 112, 112] in [-1, 1] if a face
    # is found, or None otherwise.
    face_tensor = mtcnn(img)

    if face_tensor is None:
        return None

    # Convert [-1, 1] float tensor → [0, 255] uint8 numpy (H, W, C)
    face_np = _tensor_to_uint8(face_tensor)
    return face_np


def detect_face_from_array(image: np.ndarray) -> np.ndarray | None:
    """
    Detect and align the primary face from a BGR numpy array (e.g., cv2 crop).

    Args:
        image: BGR numpy array (H, W, 3).

    Returns:
        aligned_face: np.ndarray (112, 112, 3) uint8, or None.
    """
    from PIL import Image as _PIL
    import cv2

    if image is None or image.size == 0:
        return None

    # OpenCV uses BGR; MTCNN expects RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = _PIL.fromarray(img_rgb)

    mtcnn = _get_mtcnn()
    face_tensor = mtcnn(pil_img)

    if face_tensor is None:
        return None

    return _tensor_to_uint8(face_tensor)


def detect_faces_batch(image_paths: list) -> list:
    """
    Detect and align faces from a batch of image paths.

    Args:
        image_paths: List of image file paths.

    Returns:
        List of (image_path, aligned_face) tuples where aligned_face is
        np.ndarray (112, 112, 3). Paths where no face was detected are skipped.
    """
    results = []
    for path in image_paths:
        face = detect_face(path)
        if face is not None:
            results.append((path, face))
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tensor_to_uint8(face_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert MTCNN output tensor [3, 112, 112] in [-1, 1] to
    uint8 numpy array [112, 112, 3] in [0, 255].
    """
    # [3, H, W] → [H, W, 3]
    face_np = face_tensor.permute(1, 2, 0).numpy()
    # [-1, 1] → [0, 255]
    face_np = ((face_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return face_np


def _uint8_to_tensor(face_np: np.ndarray) -> torch.Tensor:
    """
    Convert uint8 numpy array [112, 112, 3] in [0, 255] to
    float tensor [3, 112, 112] in [-1, 1].
    Used by embedder.py for model input.
    """
    face_float = face_np.astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(face_float).permute(2, 0, 1)  # [3, H, W]
    return tensor
