"""
vision/demo.py — Missing Person Search Demo
===========================================

HOW TO USE (no coding knowledge needed):
-----------------------------------------

MODE 1 — Search in PHOTOS:
  Step 1. Put the missing person's photo → photos/missing.png
  Step 2. Put surveillance photos        → photos/search/
  Step 3. Run from root or vision/:
             python vision/demo.py
          or:
             cd vision && python demo.py

MODE 2 — Search in a VIDEO:
  Step 1. Put the missing person's photo → photos/missing.png
  Step 2. Put the video file             → photos/search.mp4  (or .avi, .mkv)
  Step 3. Run:
             python vision/demo.py --video photos/search.mp4
          or:
             python vision/demo.py --video vision/photos/search.mp4
"""

import os
import sys
import argparse
from pathlib import Path

# Fix Unicode output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure BOTH project root and vision folder are on sys.path
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent

for p in [str(_PROJECT_ROOT), str(_THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Helper to resolve paths whether run from root or inside vision/ ──────────

def _resolve_path(rel_path: str) -> Path:
    """Find file in current working directory or relative to vision folder."""
    p_cwd = Path(rel_path)
    if p_cwd.exists():
        return p_cwd
    p_vision = _THIS_DIR / rel_path
    if p_vision.exists():
        return p_vision
    p_root = _PROJECT_ROOT / rel_path
    if p_root.exists():
        return p_root
    # Default fallback
    return p_cwd


# ── User Configuration (change these if not using command-line flags) ─────────

MISSING_PERSON_PHOTO = "photos/missing.png"     # ← path to the missing person's photo
SEARCH_FOLDER        = "photos/search"          # ← folder of photos to search through (MODE 1)
MATCH_THRESHOLD      = 0.45                     # ← how strict the match is (0–1)
FRAME_SKIP           = 10                       # ← video: check every Nth frame
SAVE_SNAPSHOTS       = True                     # ← video: save a photo of every matched frame
SNAPSHOT_FOLDER      = "photos/matches"         # ← where matched frame snapshots are saved

# ─────────────────────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_pipeline(use_yolo: bool):
    try:
        from vision.pipeline import VisionPipeline
    except ImportError:
        from pipeline import VisionPipeline
    return VisionPipeline(use_yolo=use_yolo)


def _get_cosine_similarity():
    try:
        from vision.embedder import cosine_similarity
    except ImportError:
        from embedder import cosine_similarity
    return cosine_similarity


def _get_query_embedding(pipeline, photo_path: str):
    """Embed the missing person's reference photo."""
    embedding = pipeline.process_image(str(photo_path))
    if embedding is None:
        print("\n  ✗ ERROR: No face detected in the missing person's photo.")
        print("    → Use a clear, front-facing photo without sunglasses or masks.\n")
        sys.exit(1)
    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — Search in photos
# ─────────────────────────────────────────────────────────────────────────────

def check_photo_setup(photo_path: Path, search_folder: Path):
    errors = []
    if not photo_path.exists():
        errors.append(
            f"\n  ✗ Missing person photo not found: '{photo_path}'"
            f"\n    → Put your reference photo in photos/missing.png."
        )
    if not search_folder.exists():
        errors.append(
            f"\n  ✗ Search folder not found: '{search_folder}'"
            f"\n    → Create a 'photos/search/' folder and put surveillance photos there."
        )
    else:
        images = [
            p for p in search_folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if not images:
            errors.append(
                f"\n  ✗ No images found in '{search_folder}'"
                f"\n    → Add .jpg / .jpeg / .png photos to that folder."
            )
    _abort_on_errors(errors)


def search_in_photos(photo_path: Path, search_folder: Path):
    print("\n")
    print("━" * 58)
    print("  MISSING PERSON SEARCH — Photo Mode")
    print("━" * 58)

    print("\n[1/4] Loading AI models (first run may take ~30 seconds)...")
    pipeline = _load_pipeline(use_yolo=False)
    print("      ✓ Models loaded.")

    print(f"\n[2/4] Reading missing person's photo: {photo_path}")
    query_emb = _get_query_embedding(pipeline, str(photo_path))
    print("      ✓ Face detected and processed.")

    search_images = [
        p for p in search_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print(f"\n[3/4] Searching {len(search_images)} photos in '{search_folder}'...")
    print()

    cosine_sim = _get_cosine_similarity()
    matches, no_face = [], []

    for img_path in search_images:
        emb = pipeline.process_image(str(img_path))
        if emb is None:
            no_face.append(img_path.name)
            print(f"    {img_path.name:<38} ⊘  no face")
        else:
            score = cosine_sim(query_emb, emb)
            if score >= MATCH_THRESHOLD:
                matches.append((img_path.name, score))
                print(f"    {img_path.name:<38} ✓  MATCH  ({score:.1%})")
            else:
                print(f"    {img_path.name:<38} ✗  {score:.1%}")

    _print_photo_summary(matches, no_face, len(search_images))


def _print_photo_summary(matches, no_face, total):
    print()
    print("━" * 58)
    print("  RESULTS SUMMARY")
    print("━" * 58)
    if matches:
        matches.sort(key=lambda x: x[1], reverse=True)
        print(f"\n  🟢 FOUND IN {len(matches)} PHOTO(S):\n")
        for name, score in matches:
            print(f"      • {name}  (confidence: {score:.1%})")
    else:
        print("\n  🔴 NOT FOUND in any photo.")
        print("     Tips: lower MATCH_THRESHOLD or add more photos.")
    print()
    if no_face:
        print(f"  ⊘  {len(no_face)} photo(s) had no detectable face — skipped")
    print(f"\n  Threshold: {MATCH_THRESHOLD:.0%}  |  Searched: {total} photos")
    print("━" * 58)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — Search in video
# ─────────────────────────────────────────────────────────────────────────────

def check_video_setup(photo_path: Path, video_path: Path):
    errors = []
    if not photo_path.exists():
        errors.append(
            f"\n  ✗ Missing person photo not found: '{photo_path}'"
            f"\n    → Put the reference photo in photos/missing.png."
        )
    if not video_path.exists():
        errors.append(
            f"\n  ✗ Video file not found: '{video_path}'"
            f"\n    → Check the video path you specified."
        )
    _abort_on_errors(errors)


def search_in_video(photo_path: Path, video_path: Path):
    import cv2

    print("\n")
    print("━" * 58)
    print("  MISSING PERSON SEARCH — Video Mode")
    print("━" * 58)
    print(f"\n  Video  : {video_path}")
    print(f"  Photo  : {photo_path}")
    print(f"  Checking every {FRAME_SKIP} frames  |  Threshold: {MATCH_THRESHOLD:.0%}")

    # ── Load models ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading AI models (first run may take ~30 seconds)...")
    pipeline = _load_pipeline(use_yolo=True)
    print("      ✓ Models loaded.")

    # ── Embed query photo ──────────────────────────────────────────────────────
    print(f"\n[2/4] Reading missing person's photo: {photo_path}")
    query_emb = _get_query_embedding(pipeline, str(photo_path))
    print("      ✓ Face detected and processed.")

    # ── Get video info ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"\n  ✗ Could not open video: {video_path}\n")
        sys.exit(1)

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps
    frames_to_check = total_frames // FRAME_SKIP
    cap.release()

    print(f"\n[3/4] Scanning video...")
    print(f"      Duration  : {_fmt_time(duration_s)}")
    print(f"      FPS       : {fps:.1f}")
    print(f"      Total frames: {total_frames:,}  →  checking {frames_to_check:,} frames\n")

    # ── Snapshot folder ────────────────────────────────────────────────────────
    snap_dir = None
    if SAVE_SNAPSHOTS:
        snap_dir = _resolve_path(SNAPSHOT_FOLDER)
        snap_dir.mkdir(parents=True, exist_ok=True)

    # ── Scan video frame by frame ──────────────────────────────────────────────
    cosine_sim = _get_cosine_similarity()
    from tqdm import tqdm

    match_events  = []   # list of {frame, timestamp, score, snapshot_path}
    frames_checked = 0
    best_score_seen = 0.0

    MERGE_GAP_S = 3.0

    with tqdm(total=frames_to_check, desc="  Scanning", unit="frame") as pbar:
        for frame_idx, frame, detections in pipeline.process_video_frame_generator(
            str(video_path), frame_skip=FRAME_SKIP
        ):
            frames_checked += 1
            pbar.update(1)

            timestamp_s = frame_idx / fps

            for det in detections:
                emb = det.get("embedding")
                if emb is None:
                    continue

                score = cosine_sim(query_emb, emb)
                if score > best_score_seen:
                    best_score_seen = score

                if score >= MATCH_THRESHOLD:
                    snap_path = None
                    if SAVE_SNAPSHOTS and snap_dir is not None:
                        snap_name = f"match_{_fmt_time(timestamp_s).replace(':', '-')}_score{score:.0%}.jpg"
                        snap_path = str(snap_dir / snap_name)
                        annotated = frame.copy()
                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"MATCH {score:.0%}"
                        cv2.putText(annotated, label, (x1, max(y1 - 10, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.imwrite(snap_path, annotated)

                    match_events.append({
                        "frame":     frame_idx,
                        "timestamp": timestamp_s,
                        "score":     score,
                        "snapshot":  snap_path,
                    })

    sightings = _merge_sightings(match_events, gap_s=MERGE_GAP_S)
    _print_video_summary(sightings, frames_checked, duration_s, best_score_seen, str(snap_dir))


def _merge_sightings(events: list, gap_s: float) -> list:
    if not events:
        return []

    events = sorted(events, key=lambda e: e["timestamp"])
    sightings = []
    grp_start = events[0]["timestamp"]
    grp_end   = events[0]["timestamp"]
    grp_best  = events[0]["score"]
    grp_snap  = events[0]["snapshot"]

    for evt in events[1:]:
        if evt["timestamp"] - grp_end <= gap_s:
            grp_end = evt["timestamp"]
            if evt["score"] > grp_best:
                grp_best = evt["score"]
                grp_snap = evt["snapshot"]
        else:
            sightings.append({
                "start": grp_start, "end": grp_end,
                "best_score": grp_best, "snapshot": grp_snap,
            })
            grp_start = grp_end = evt["timestamp"]
            grp_best  = evt["score"]
            grp_snap  = evt["snapshot"]

    sightings.append({
        "start": grp_start, "end": grp_end,
        "best_score": grp_best, "snapshot": grp_snap,
    })
    return sightings


def _print_video_summary(sightings, frames_checked, duration_s, best_score, snap_dir_str):
    print()
    print("━" * 58)
    print("  RESULTS SUMMARY — Video Search")
    print("━" * 58)

    if sightings:
        print(f"\n  🟢 PERSON SPOTTED {len(sightings)} TIME(S):\n")
        for i, s in enumerate(sightings, 1):
            start = _fmt_time(s["start"])
            end   = _fmt_time(s["end"])
            time_str = start if s["start"] == s["end"] else f"{start} → {end}"
            snap_note = f"\n         Snapshot: {s['snapshot']}" if s["snapshot"] else ""
            print(f"      {i}. {time_str}   (confidence: {s['best_score']:.1%}){snap_note}")
        if SAVE_SNAPSHOTS:
            print(f"\n  📁 Snapshot images saved to: {snap_dir_str}/")
    else:
        print(f"\n  🔴 PERSON NOT FOUND in the video.")
        print(f"     Best similarity seen: {best_score:.1%}  (threshold: {MATCH_THRESHOLD:.0%})")
        print("     Tips:")
        print("     • Try lowering MATCH_THRESHOLD (e.g. 0.30)")
        print(f"     • Try scanning more frames (lower FRAME_SKIP, currently {FRAME_SKIP})")
        print("     • Use a clearer reference photo")

    print()
    print(f"  Video duration : {_fmt_time(duration_s)}")
    print(f"  Frames checked : {frames_checked:,}")
    print(f"  Frame skip     : every {FRAME_SKIP} frames")
    print(f"  Threshold      : {MATCH_THRESHOLD:.0%}")
    print("━" * 58)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _abort_on_errors(errors: list):
    if errors:
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("  SETUP PROBLEM — Please fix the following:")
        for e in errors:
            print(e)
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Search for a missing person in photos or a video."
    )
    parser.add_argument(
        "--video", type=str, default=None,
        metavar="VIDEO_PATH",
        help="Path to a video file to search (e.g. photos/search.mp4). "
             "If not given, searches the photos/ folder instead.",
    )
    parser.add_argument(
        "--photo", type=str, default=None,
        metavar="PHOTO_PATH",
        help="Override the missing person's photo path.",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        metavar="0.45",
        help="Match threshold 0–1 (default 0.45).",
    )
    parser.add_argument(
        "--frame-skip", type=int, default=None,
        metavar="N",
        help="Video: check every Nth frame (default 10). Lower = slower but more thorough.",
    )
    args = parser.parse_args()

    # Apply CLI overrides
    photo_target = _resolve_path(args.photo if args.photo else MISSING_PERSON_PHOTO)
    search_target = _resolve_path(SEARCH_FOLDER)

    if args.threshold is not None:
        MATCH_THRESHOLD = args.threshold
    if args.frame_skip is not None:
        FRAME_SKIP = args.frame_skip

    if args.video:
        video_target = _resolve_path(args.video)
        check_video_setup(photo_target, video_target)
        search_in_video(photo_target, video_target)
    else:
        check_photo_setup(photo_target, search_target)
        search_in_photos(photo_target, search_target)
