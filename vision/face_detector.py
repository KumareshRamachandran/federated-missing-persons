"""
vision/face_detector.py

Face detection and alignment using MTCNN (via facenet-pytorch) and OpenCV.
Detects faces, extracts 5 facial landmarks (eyes, nose, mouth), and applies an
affine transformation to align the eyes horizontally, outputting a 112x112 RGB crop.

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torch

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_mtcnn: MTCNN | None = None


def _get_mtcnn() -> MTCNN:
    """Return the shared MTCNN instance, creating it on first call."""
    global _mtcnn
    if _mtcnn is None:
        _mtcnn = MTCNN(
            image_size=112,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=False,
            device=_device,
        )
    return _mtcnn


class FaceAligner:
    """
    MTCNN + OpenCV affine face aligner.
    Detects the primary face within a cropped human figure, extracts 5 facial landmarks,
    and applies an affine transformation to align the eyes horizontally into a 112x112 RGB crop.
    """

    def __init__(self, device: torch.device | str | None = None):
        if device is None:
            device = _device
        elif isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.mtcnn = MTCNN(
            keep_all=False,
            select_largest=True,
            post_process=False,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            device=self.device,
        )

    def align_face(self, person_crop: np.ndarray) -> np.ndarray | None:
        """
        Detects primary face, extracts 5 facial landmarks (eyes, nose, mouth),
        and applies an affine transformation to align eyes horizontally.

        Args:
            person_crop: Cropped human figure as NumPy array (BGR or RGB).

        Returns:
            Tightly cropped and resized 112x112 RGB NumPy array (uint8),
            or None if no face is detected.
        """
        if person_crop is None or person_crop.size == 0:
            return None

        # Standardize to RGB for MTCNN detection
        if person_crop.ndim == 2:
            img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_GRAY2RGB)
        else:
            # Assume OpenCV BGR input by default
            img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img_rgb)
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

        if boxes is None or landmarks is None or len(landmarks) == 0:
            return None

        prob = probs[0] if probs is not None else 0.0
        if prob is None or prob < 0.5:
            return None

        # Primary face 5 landmarks: [left_eye, right_eye, nose, mouth_left, mouth_right]
        face_landmarks = landmarks[0]
        left_eye = face_landmarks[0]
        right_eye = face_landmarks[1]

        # Calculate angle between eyes
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = float(np.degrees(np.arctan2(dY, dX)))

        # Eyes center point
        eyes_center = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0),
        )

        # Scale factor relative to standard 112x112 ArcFace target eye distance (~33.6px)
        target_eye_dist = 112.0 * 0.30  # 33.6 px
        dist = float(np.sqrt(dX**2 + dY**2))
        scale = target_eye_dist / (dist + 1e-8) if dist > 0 else 1.0

        # Compute rotation and scale affine matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Shift target eye center to canonical position (0.5 * 112, 0.4 * 112) = (56.0, 44.8)
        tX = 112.0 * 0.5
        tY = 112.0 * 0.4
        M[0, 2] += tX - eyes_center[0]
        M[1, 2] += tY - eyes_center[1]

        # Apply affine transformation
        aligned_rgb = cv2.warpAffine(
            img_rgb, M, (112, 112), flags=cv2.INTER_CUBIC
        )

        return aligned_rgb


# ── Public API (Backward Compatibility) ───────────────────────────────────────

def detect_face(image_path: str) -> np.ndarray | None:
    """
    Detect and align the primary face from an image file.

    Returns:
        aligned_face: np.ndarray (112, 112, 3) uint8 RGB, or None.
    """
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return None
    except Exception:
        return None

    aligner = FaceAligner()
    return aligner.align_face(img_bgr)


def detect_face_from_array(image: np.ndarray) -> np.ndarray | None:
    """
    Detect and align the primary face from a BGR numpy array (e.g. YOLO crop).

    Returns:
        aligned_face: np.ndarray (112, 112, 3) uint8 RGB, or None.
    """
    if image is None or image.size == 0:
        return None

    aligner = FaceAligner()
    return aligner.align_face(image)


def detect_faces_batch(image_paths: list) -> list:
    results = []
    for path in image_paths:
        face = detect_face(path)
        if face is not None:
            results.append((path, face))
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tensor_to_uint8(face_tensor: torch.Tensor) -> np.ndarray:
    face_np = face_tensor.permute(1, 2, 0).numpy()
    face_np = ((face_np + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    return face_np


def _uint8_to_tensor(face_np: np.ndarray) -> torch.Tensor:
    face_float = face_np.astype(np.float32) / 255.0 * 2.0 - 1.0
    tensor = torch.from_numpy(face_float).permute(2, 0, 1)
    return tensor


if __name__ == "__main__":
    print("Testing FaceAligner...")
    aligner = FaceAligner()

    sample_bgr = cv2.imread("vision/photos/missing.png")
    if sample_bgr is not None:
        aligned = aligner.align_face(sample_bgr)
        if aligned is not None:
            print(f"Face aligned successfully. Output shape: {aligned.shape}, dtype: {aligned.dtype}")
            assert aligned.shape == (112, 112, 3)
            print("[OK] FaceAligner test passed!")
        else:
            print("No face detected in missing.png.")
    else:
        print("Sample image not found. Creating synthetic face crop for testing.")
        synthetic_bgr = np.full((200, 200, 3), 220, dtype=np.uint8)
        aligned = aligner.align_face(synthetic_bgr)
        print(f"Synthetic face alignment output: {aligned}")

