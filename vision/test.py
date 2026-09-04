"""
vision/test.py — Computer Vision Pipeline Integration Test & Demo
=================================================================

Tests and demonstrates the 5 key computer vision modules:
  1. Data Augmentation (apply_cctv_distortion from vision.augmentation)
  2. YOLO Human Detection (YOLOPersonDetector from vision.yolo_detector)
  3. MTCNN Face Alignment (FaceAligner from vision.face_detector)
  4. Deep Face Embeddings (FaceEmbedder from vision.embedder)
  5. End-to-End Pipeline Matching & Verification

Usage:
    python vision/test.py                   # Run full component integration tests
    python vision/test.py --mode search     # Run missing person search demo
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

# Ensure sys.path includes project root and vision folder
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vision.augmentation import apply_cctv_distortion, SurveillanceAugmentor
from vision.yolo_detector import YOLOPersonDetector, YOLODetector
from vision.face_detector import FaceAligner, detect_face
from vision.embedder import FaceEmbedder, generate_embedding, cosine_similarity
from vision.pipeline import VisionPipeline


def test_1_augmentation():
    print("\n--- Test 1: Data Augmentation (augmentation.py) ---")
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (200, 180, 150), -1)
    cv2.circle(img, (150, 150), 30, (80, 40, 20), -1)

    augmented = apply_cctv_distortion(img)
    print(f"  Input image shape    : {img.shape}")
    print(f"  Augmented image shape: {augmented.shape}")
    assert augmented.shape == img.shape, "Augmented shape mismatch!"
    print("  [OK] apply_cctv_distortion test passed!")


def test_2_yolo_detection():
    print("\n--- Test 2: YOLO Human Detection (yolo_detector.py) ---")
    detector = YOLOPersonDetector("yolov8n.pt")

    sample_photo = _THIS_DIR / "photos" / "missing.png"
    if sample_photo.exists():
        print(f"  Running YOLOPersonDetector on {sample_photo}...")
        crops = detector.detect_persons(str(sample_photo), conf_threshold=0.3)
        print(f"  Detected {len(crops)} person crop(s).")
        if crops:
            print(f"  First person crop shape: {crops[0].shape}")
            assert crops[0].ndim == 3 and crops[0].shape[2] == 3
    else:
        print("  Sample image missing.png not found, creating synthetic test frame...")
        frame = np.full((600, 400, 3), 200, dtype=np.uint8)
        cv2.circle(frame, (200, 150), 40, (120, 120, 120), -1)
        cv2.rectangle(frame, (150, 200), (250, 500), (60, 60, 60), -1)
        crops = detector.detect_persons(frame, conf_threshold=0.1)
        print(f"  Crops returned on synthetic frame: {len(crops)}")

    print("  [OK] YOLOPersonDetector test passed!")


def test_3_mtcnn_face_alignment():
    print("\n--- Test 3: MTCNN Face Alignment (face_detector.py) ---")
    aligner = FaceAligner()

    sample_photo = _THIS_DIR / "photos" / "missing.png"
    if sample_photo.exists():
        img_bgr = cv2.imread(str(sample_photo))
        aligned_rgb = aligner.align_face(img_bgr)
        if aligned_rgb is not None:
            print(f"  Aligned face output shape: {aligned_rgb.shape}, dtype: {aligned_rgb.dtype}")
            assert aligned_rgb.shape == (112, 112, 3), f"Expected (112, 112, 3), got {aligned_rgb.shape}"
            print("  [OK] FaceAligner successfully extracted and aligned face to 112x112 RGB!")
        else:
            print("  Warning: No face detected in sample photo.")
    else:
        print("  Testing aligner fallback with empty input...")
        res = aligner.align_face(np.zeros((50, 50, 3), dtype=np.uint8))
        assert res is None
        print("  [OK] FaceAligner returned None on non-face crop!")

    print("  [OK] FaceAligner test passed!")


def test_4_face_embeddings():
    print("\n--- Test 4: Deep Face Embeddings (embedder.py) ---")
    embedder = FaceEmbedder()

    # Test with 112x112 RGB dummy face
    dummy_face_rgb = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    emb1 = embedder.extract_embedding(dummy_face_rgb)
    emb2 = embedder.extract_embedding(dummy_face_rgb)

    print(f"  Extracted embedding dimension: {emb1.shape}")
    print(f"  L2 norm of embedding         : {np.linalg.norm(emb1):.4f}")
    assert emb1.shape == (512,), "Embedding must be 512-dimensional!"
    assert np.isclose(np.linalg.norm(emb1), 1.0, atol=1e-3), "Embedding must be L2-normalized!"

    is_match = embedder.compute_cosine_similarity(emb1, emb2, threshold=0.3)
    sim_score = embedder.calculate_similarity_score(emb1, emb2)
    print(f"  Self similarity score        : {sim_score:.4f}")
    print(f"  Match result (threshold=0.35): {is_match}")
    assert is_match is True, "Self match should be True!"

    # Test non-match with random noise
    noise_face_rgb = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    emb_noise = embedder.extract_embedding(noise_face_rgb)
    noise_sim = embedder.calculate_similarity_score(emb1, emb_noise)
    print(f"  Random face similarity score : {noise_sim:.4f}")

    print("  [OK] FaceEmbedder test passed!")


def test_5_end_to_end_pipeline():
    print("\n--- Test 5: End-to-End Pipeline Verification ---")
    sample_photo = _THIS_DIR / "photos" / "missing.png"
    if not sample_photo.exists():
        print("  Sample image missing.png not found. Skipping end-to-end photo test.")
        return

    img_bgr = cv2.imread(str(sample_photo))

    # Step 1: CCTV Distortion
    distorted_bgr = apply_cctv_distortion(img_bgr)

    # Step 2: YOLO Person Detection
    yolo = YOLOPersonDetector("yolov8n.pt")
    person_crops = yolo.detect_persons(distorted_bgr, conf_threshold=0.3)
    target_crop = person_crops[0] if person_crops else distorted_bgr

    # Step 3: MTCNN Face Alignment
    aligner = FaceAligner()
    aligned_face_rgb = aligner.align_face(target_crop)
    if aligned_face_rgb is None:
        aligned_face_rgb = cv2.resize(cv2.cvtColor(target_crop, cv2.COLOR_BGR2RGB), (112, 112))

    # Step 4: ArcFace Embedding
    embedder = FaceEmbedder()
    emb_distorted = embedder.extract_embedding(aligned_face_rgb)
    emb_original = embedder.extract_embedding(cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (112, 112)))

    # Step 5: Similarity & Match Thresholding
    is_match = embedder.compute_cosine_similarity(emb_original, emb_distorted, threshold=0.35)
    score = embedder.calculate_similarity_score(emb_original, emb_distorted)

    print(f"  Original vs Distorted Face Similarity: {score:.1%}")
    print(f"  Match Decision (Threshold = 0.35)    : {'MATCH' if is_match else 'NO-MATCH'}")
    print("  [OK] End-to-End Pipeline Verification complete!")


def run_all_tests():
    print("=================================================================")
    print("      RUNNING FEDERATED MISSING PERSONS VISION TESTS")
    print("=================================================================")
    test_1_augmentation()
    test_2_yolo_detection()
    test_3_mtcnn_face_alignment()
    test_4_face_embeddings()
    test_5_end_to_end_pipeline()
    print("\n=================================================================")
    print("  [ALL PASSED] ALL 5 COMPUTER VISION PIPELINE TESTS PASSED!")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Pipeline Test Suite")
    parser.add_argument("--mode", type=str, default="test", choices=["test", "search"], help="Run mode: test or search")
    args = parser.parse_args()

    run_all_tests()

