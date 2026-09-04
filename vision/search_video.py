"""
vision/search_video.py

Searches for a missing person in a surveillance video file using ArcFace face embeddings.

Pipeline:
  1. Extract 512-d ArcFace embedding from missing person photograph.
  2. Stream frames from target video file.
  3. Detect person regions (YOLOv8) & align faces (MTCNN).
  4. Extract ArcFace embeddings for detected faces.
  5. Compute cosine similarity against missing person embedding.
  6. Return matches with timestamps, confidence scores, and optional annotated output video.

Usage:
  python vision/search_video.py --person vision/photos/missing.png --video vision/photos/search.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vision.embedder import (
    FaceEmbedder,
    cosine_similarity,
    generate_embedding,
    generate_embedding_from_crop,
)
from vision.pipeline import VisionPipeline


def search_person_in_video(
    person_image_path: str,
    video_path: str,
    threshold: float = 0.40,
    frame_skip: int = 5,
    output_video_path: Optional[str] = None,
    progress_callback=None,
) -> Dict:
    """
    Search for a missing person photograph in a video file using ArcFace embeddings.

    Args:
        person_image_path: Path to missing person query photo.
        video_path: Path to surveillance video file.
        threshold: Cosine similarity threshold for a positive face match.
        frame_skip: Sample every N-th frame for speed.
        output_video_path: Optional output path to save annotated MP4 video with bounding boxes.
        progress_callback: Optional callable(current_frame, total_frames, matches_count) for UI progress updates.

    Returns:
        Dict containing total_frames, sampled_frames, match_found (bool), best_similarity,
        and list of match details (frame_idx, timestamp_sec, similarity, bbox, frame_rgb).
    """
    print("=" * 65, flush=True)
    print("      ARCFACE VIDEO PERSON SEARCH ENGINE", flush=True)
    print("=" * 65, flush=True)
    print(f"Target Person Photo: {person_image_path}", flush=True)
    print(f"Surveillance Video : {video_path}", flush=True)
    print(f"Match Threshold    : {threshold:.2f}", flush=True)
    print(f"Sampling Frame Skip: Every {frame_skip} frames", flush=True)
    print("-" * 65, flush=True)

    if not Path(person_image_path).exists():
        raise FileNotFoundError(f"Person image not found: {person_image_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Step 1: Extract ArcFace embedding for missing person photo
    print("[1/3] Extracting 512-d ArcFace embedding from query photo...", flush=True)
    query_emb = generate_embedding(person_image_path)
    if query_emb is None:
        # Direct resize fallback if MTCNN alignment failed
        from PIL import Image
        img = Image.open(person_image_path).convert("RGB").resize((112, 112))
        arr = np.asarray(img, dtype=np.uint8)
        embedder = FaceEmbedder()
        query_emb = embedder.extract_embedding(arr)

    print(f"  [OK] Query embedding extracted successfully. Norm: {np.linalg.norm(query_emb):.4f}", flush=True)

    # Step 2: Open Video File
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[2/3] Video Info: {total_frames} frames | {fps:.1f} FPS | Duration: {duration_sec:.2f}s | Res: {width}x{height}", flush=True)

    # Initialize VideoWriter if requested
    out_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        print(f"  [INFO] Saving annotated video to: {output_video_path}", flush=True)

    # Initialize Vision Pipeline with YOLO
    pipeline = VisionPipeline(use_yolo=True)
    embedder = FaceEmbedder()

    matches = []
    best_similarity = -1.0
    sampled_count = 0
    frame_idx = 0
    start_time = time.time()

    print("[3/3] Scanning video frames...", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                sampled_count += 1
                timestamp = frame_idx / fps

                if progress_callback is not None and total_frames > 0:
                    try:
                        progress_callback(frame_idx, total_frames, len(matches))
                    except Exception:
                        pass

                # Detect person crops in frame
                if hasattr(pipeline.detector, "detect_persons_detailed"):
                    detections = pipeline.detector.detect_persons_detailed(frame)
                else:
                    detections = pipeline.detector.detect_persons(frame)

                frame_has_match = False
                for det in detections:
                    if isinstance(det, dict):
                        crop = det.get("crop")
                        bbox = det.get("bbox", [0, 0, width, height])
                    elif isinstance(det, np.ndarray):
                        crop = det
                        bbox = [0, 0, width, height]
                    else:
                        continue

                    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                        continue

                    # Extract face embedding from crop
                    crop_emb = generate_embedding_from_crop(crop)
                    if crop_emb is None:
                        try:
                            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            crop_resized = cv2.resize(crop_rgb, (112, 112))
                            crop_emb = embedder.extract_embedding(crop_resized)
                        except Exception:
                            continue

                    similarity = cosine_similarity(query_emb, crop_emb)
                    if similarity > best_similarity:
                        best_similarity = similarity

                    if similarity >= threshold:
                        frame_has_match = True

                        # Draw bounding box and text on annotated frame
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"TARGET MATCH ({similarity*100:.1f}%)"
                        cv2.putText(
                            frame,
                            label,
                            (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )

                        annotated_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        matches.append(
                            {
                                "frame_idx": frame_idx,
                                "timestamp_sec": round(timestamp, 2),
                                "timestamp_formatted": f"{int(timestamp//60):02d}:{timestamp%60:05.2f}",
                                "similarity": round(similarity, 4),
                                "bbox": bbox,
                                "frame_rgb": annotated_rgb,
                            }
                        )

                if frame_has_match:
                    print(
                        f"  [MATCH DETECTED] Frame {frame_idx:05d} ({matches[-1]['timestamp_formatted']}) | "
                        f"Similarity: {matches[-1]['similarity']*100:.2f}% | BBox: {matches[-1]['bbox']}",
                        flush=True,
                    )

            if out_writer is not None:
                out_writer.write(frame)

            frame_idx += 1

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()

    elapsed = time.time() - start_time
    processing_fps = sampled_count / (elapsed + 1e-6)

    print("=" * 65, flush=True)
    print("      SEARCH SUMMARY & VERIFICATION RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"Total Frames Processed : {total_frames}", flush=True)
    print(f"Sampled Frames Checked : {sampled_count}", flush=True)
    print(f"Processing Time        : {elapsed:.2f}s ({processing_fps:.1f} FPS)", flush=True)
    print(f"Peak Cosine Similarity : {best_similarity*100:.2f}%", flush=True)
    print(f"Total Matches Found    : {len(matches)}", flush=True)
    print(f"Match Status           : {'[POSITIVE MATCH DETECTED]' if len(matches) > 0 else '[NO MATCH FOUND]'}", flush=True)
    print("=" * 65, flush=True)

    return {
        "person_image": person_image_path,
        "video_path": video_path,
        "total_frames": total_frames,
        "sampled_frames": sampled_count,
        "processing_time_sec": round(elapsed, 2),
        "best_similarity": round(best_similarity, 4),
        "match_found": len(matches) > 0,
        "matches_count": len(matches),
        "matches": matches,
        "annotated_video_path": output_video_path if output_video_path else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for a missing person photograph in a surveillance video using ArcFace embeddings."
    )
    parser.add_argument(
        "--person",
        type=str,
        default=str(_PROJECT_ROOT / "vision" / "photos" / "missing.png"),
        help="Path to target missing person photograph.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(_PROJECT_ROOT / "vision" / "photos" / "search.mp4"),
        help="Path to surveillance video file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Cosine similarity threshold for a positive face match (default: 0.40).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Sample every N-th frame for speed (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated output video with bounding boxes.",
    )

    args = parser.parse_args()

    search_person_in_video(
        person_image_path=args.person,
        video_path=args.video,
        threshold=args.threshold,
        frame_skip=args.frame_skip,
        output_video_path=args.output,
    )
