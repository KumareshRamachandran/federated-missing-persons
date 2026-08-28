"""
vision/embedder.py

Generates 512-d ArcFace face embeddings from raw images.
Used by both the coordinator (query embedding) and the org nodes (gallery building).

Pipeline:
    image path / numpy array
        → MTCNN detect & align  (112×112 RGB face crop)
        → ArcFaceModel forward  (512-d L2-normalised embedding)
        → numpy array

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from vision.arcface_model import ArcFaceModel
from vision.face_detector import detect_face, detect_face_from_array, _uint8_to_tensor


# ── Singleton model ───────────────────────────────────────────────────────────

_model: ArcFaceModel | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(weights_path: str | None = None) -> ArcFaceModel:
    """Load ArcFace model once and cache in module-level singleton."""
    global _model
    if _model is None:
        _model = ArcFaceModel(
            pretrained="vggface2" if weights_path is None else None,
            weights_path=weights_path,
        )
        _model.to(_device)
        _model.eval()
    return _model


def reset_model() -> None:
    """Force reload of the model on next call (useful after FL weight update)."""
    global _model
    _model = None


# ── Public API ────────────────────────────────────────────────────────────────

def generate_embedding(
    image_path: str,
    weights_path: str | None = None,
) -> np.ndarray | None:
    """
    Full pipeline: detect face → align → generate 512-d embedding.

    Args:
        image_path:   Path to the input image.
        weights_path: Optional path to ArcFace weights (.pt checkpoint).

    Returns:
        embedding: np.ndarray of shape (512,), L2-normalised, or None if
                   no face is detected in the image.
    """
    # Stage 1: MTCNN face detection & alignment → (112, 112, 3) uint8
    aligned_face = detect_face(str(image_path))
    if aligned_face is None:
        return None

    return _embed_aligned_face(aligned_face, weights_path)


def generate_embedding_from_crop(
    crop: np.ndarray,
    weights_path: str | None = None,
) -> np.ndarray | None:
    """
    Generate embedding from a BGR numpy crop (e.g., YOLO person crop).

    Args:
        crop:         BGR numpy array of a person region.
        weights_path: Optional path to ArcFace weights.

    Returns:
        embedding: np.ndarray (512,) or None if no face detected in crop.
    """
    aligned_face = detect_face_from_array(crop)
    if aligned_face is None:
        return None

    return _embed_aligned_face(aligned_face, weights_path)


def generate_embeddings_batch(
    image_paths: list,
    weights_path: str | None = None,
    show_progress: bool = True,
) -> dict:
    """
    Generate embeddings for a list of image paths.

    Args:
        image_paths:   List of image file paths (str or Path).
        weights_path:  Optional ArcFace weights path.
        show_progress: Show tqdm progress bar.

    Returns:
        dict: {image_path_str: embedding_np_array}
              Paths where no face was detected are excluded.
    """
    results: dict = {}
    iterator = tqdm(image_paths, desc="Generating embeddings") if show_progress else image_paths

    for path in iterator:
        embedding = generate_embedding(str(path), weights_path)
        if embedding is not None:
            results[str(path)] = embedding

    return results


def build_gallery(
    gallery_dir: str,
    weights_path: str | None = None,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict:
    """
    Build an embedding gallery from a directory of identity sub-folders.

    Expected directory layout:
        gallery_dir/
            <identity_id>/
                image1.jpg
                image2.jpg
                ...

    Returns:
        dict: {identity_id: np.ndarray (512,)}
              One embedding per identity (mean-pooled over all their images).
    """
    gallery_path = Path(gallery_dir)
    gallery: dict = {}

    for identity_dir in sorted(gallery_path.iterdir()):
        if not identity_dir.is_dir():
            continue

        identity_id = identity_dir.name
        image_paths = [
            p for p in identity_dir.iterdir()
            if p.suffix.lower() in extensions
        ]

        if not image_paths:
            continue

        embeddings = generate_embeddings_batch(
            image_paths, weights_path, show_progress=False
        )

        if not embeddings:
            continue

        # Mean-pool all embeddings for this identity, then re-normalise
        stacked = np.stack(list(embeddings.values()), axis=0)  # [N, 512]
        mean_emb = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        gallery[identity_id] = mean_emb / (norm + 1e-8)

    return gallery


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalised embeddings."""
    return float(np.dot(a, b))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _embed_aligned_face(
    aligned_face: np.ndarray,
    weights_path: str | None,
) -> np.ndarray:
    """
    Run ArcFace forward pass on a pre-aligned (112×112) face numpy array.

    Args:
        aligned_face: uint8 numpy array (112, 112, 3).
        weights_path: Optional ArcFace weights path.

    Returns:
        embedding: np.ndarray (512,).
    """
    model = _load_model(weights_path)

    # Convert uint8 [0,255] → float [-1,1] tensor [3, 112, 112]
    face_tensor = _uint8_to_tensor(aligned_face)  # [3, 112, 112]

    embedding = model.get_embedding(face_tensor)   # [512] cpu tensor
    return embedding.numpy()                        # (512,) float32
