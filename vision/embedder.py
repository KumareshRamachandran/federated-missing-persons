"""
vision/embedder.py

Generates 512-d ArcFace face embeddings from raw images and aligned face crops.
Wraps the ArcFace (iResNet50 / InceptionResnetV1) PyTorch model in the FaceEmbedder class.

Pipeline:
    aligned face (112x112 RGB image / tensor)
        -> ArcFace forward pass
        -> 512-d L2-normalised embedding vector

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
from tqdm import tqdm

from vision.arcface_model import ArcFaceModel
from vision.face_detector import detect_face, detect_face_from_array, _uint8_to_tensor

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


class FaceEmbedder:
    """
    Wraps the ArcFace PyTorch model.
    Expects a 112x112 RGB normalized image (NumPy array or Tensor) as input,
    extracts 512-dimensional L2-normalized face embedding vectors,
    and computes cosine similarity with threshold matching logic.
    """

    def __init__(self, weights_path: str | None = None, device: torch.device | str | None = None):
        if device is None:
            self.device = _device
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model = ArcFaceModel(
            pretrained="vggface2" if weights_path is None else None,
            weights_path=weights_path,
        )
        self.model.to(self.device)
        self.model.eval()

    def extract_embedding(self, aligned_face_image: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Passes a 112x112 RGB aligned face image through the ArcFace network
        to extract a 512-dimensional L2-normalized embedding vector.

        Args:
            aligned_face_image: (112, 112, 3) RGB uint8 NumPy array OR float Tensor [3, 112, 112] / [1, 3, 112, 112].

        Returns:
            512-dimensional L2-normalized embedding vector as float32 NumPy array (512,).
        """
        if isinstance(aligned_face_image, np.ndarray):
            # Ensure shape is 112x112
            if aligned_face_image.shape[:2] != (112, 112):
                aligned_face_image = torch.nn.functional.interpolate(
                    torch.from_numpy(aligned_face_image).permute(2, 0, 1).unsqueeze(0).float(),
                    size=(112, 112),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)

            # Convert [112, 112, 3] uint8 RGB [0, 255] -> [-1, 1] float tensor [1, 3, 112, 112]
            face_float = aligned_face_image.astype(np.float32) / 255.0 * 2.0 - 1.0
            tensor = torch.from_numpy(face_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        elif isinstance(aligned_face_image, torch.Tensor):
            tensor = aligned_face_image.float()
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)
        else:
            raise TypeError("aligned_face_image must be a NumPy array or PyTorch Tensor.")

        with torch.no_grad():
            embedding_tensor = self.model(tensor)  # [1, 512] L2-normalized

        embedding_np = embedding_tensor.squeeze(0).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        return embedding_np

    def compute_cosine_similarity(
        self, emb1: np.ndarray, emb2: np.ndarray, threshold: float = 0.35
    ) -> bool:
        """
        Calculates the cosine similarity between two embedding vectors and checks against threshold.

        Args:
            emb1: 512-d NumPy embedding vector.
            emb2: 512-d NumPy embedding vector.
            threshold: Cosine similarity threshold for a match (default 0.35).

        Returns:
            True if similarity >= threshold (Match), False otherwise (No-Match).
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return False

        similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
        return similarity >= threshold

    @staticmethod
    def calculate_similarity_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Helper returning raw scalar cosine similarity score in [-1.0, 1.0]."""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))


# ── Public API (Backward Compatibility) ───────────────────────────────────────

def generate_embedding(
    image_path: str,
    weights_path: str | None = None,
) -> np.ndarray | None:
    aligned_face = detect_face(str(image_path))
    if aligned_face is None:
        return None
    return _embed_aligned_face(aligned_face, weights_path)


def generate_embedding_from_crop(
    crop: np.ndarray,
    weights_path: str | None = None,
) -> np.ndarray | None:
    aligned_face = detect_face_from_array(crop)
    if aligned_face is None:
        return None
    return _embed_aligned_face(aligned_face, weights_path)


def generate_embeddings_batch(
    image_paths: list,
    weights_path: str | None = None,
    show_progress: bool = True,
) -> dict:
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

        stacked = np.stack(list(embeddings.values()), axis=0)
        mean_emb = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        gallery[identity_id] = mean_emb / (norm + 1e-8)

    return gallery


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _embed_aligned_face(
    aligned_face: np.ndarray,
    weights_path: str | None,
) -> np.ndarray:
    embedder = FaceEmbedder(weights_path=weights_path)
    return embedder.extract_embedding(aligned_face)


if __name__ == "__main__":
    print("Testing FaceEmbedder...")
    embedder = FaceEmbedder()

    # Create dummy aligned face 112x112 RGB image
    dummy_face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    emb1 = embedder.extract_embedding(dummy_face)
    emb2 = embedder.extract_embedding(dummy_face)

    print(f"Embedding shape: {emb1.shape}, L2 norm: {np.linalg.norm(emb1):.4f}")
    assert emb1.shape == (512,)
    assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-3)

    is_match = embedder.compute_cosine_similarity(emb1, emb2, threshold=0.35)
    sim_score = embedder.calculate_similarity_score(emb1, emb2)
    print(f"Self-similarity score: {sim_score:.4f}, Match boolean: {is_match}")
    assert is_match is True
    print("[OK] FaceEmbedder test passed!")

