"""
vision/pipeline.py

Unified VisionPipeline entry point.

This is the single public API used by:
  - federated/coordinator/query_router.py  (query embedding)
  - federated/client/local_matcher.py      (gallery embedding)
  - dashboard/app.py                       (live frame processing)
  - demo.py                                (demonstration / testing)

Pipeline:
    Raw surveillance image / frame
        -> YOLO (person detection)     -> person crop(s)
        -> MTCNN (face detection)      -> 112x112 aligned face
        -> ArcFace (embedding)         -> 512-d L2-normalised vector
        -> Output

Member: G N Lokesh (23BCE9603) -- Computer Vision module
"""

import numpy as np
from pathlib import Path

from vision.yolo_detector import YOLODetector
from vision.embedder import (
    generate_embedding,
    generate_embedding_from_crop,
    build_gallery,
    cosine_similarity,
    reset_model,
)


class VisionPipeline:
    """
    End-to-end vision pipeline: image/frame -> 512-d face embedding(s).

    Usage (coordinator -- single query image):
        pipeline = VisionPipeline()
        emb = pipeline.process_image("query.jpg")   # np.ndarray (512,) or None

    Usage (org node -- surveillance frame):
        pipeline = VisionPipeline()
        results = pipeline.process_frame(frame_bgr)
        # [{"bbox": [...], "confidence": 0.87, "embedding": np.ndarray(512,)}]

    Usage (gallery builder):
        gallery = pipeline.build_node_gallery("data/nodes/node_police/gallery")
    """

    def __init__(
        self,
        yolo_model="yolov8n.pt",
        yolo_conf=0.4,
        arcface_weights=None,
        use_yolo=True,
    ):
        """
        Args:
            yolo_model:      Path/name of YOLOv8 weights.
            yolo_conf:       YOLO person detection confidence threshold.
            arcface_weights: Optional path to fine-tuned ArcFace checkpoint.
            use_yolo:        If False, skip YOLO and pass the full image
                             directly to MTCNN (for clean face images like
                             query photos that don't need person detection).
        """
        self.arcface_weights = arcface_weights
        self.use_yolo = use_yolo

        if use_yolo:
            self.detector = YOLODetector(
                model_path=yolo_model,
                confidence=yolo_conf,
            )

    # ------------------------------------------------------------------
    # Primary interfaces
    # ------------------------------------------------------------------

    def process_image(self, image_path):
        """
        Generate a single 512-d embedding from a query image.

        Skips YOLO -- assumes the image is a clean face photo.

        Args:
            image_path: Path to query image.

        Returns:
            np.ndarray of shape (512,) or None if no face detected.
        """
        return generate_embedding(str(image_path), self.arcface_weights)

    def process_frame(self, frame):
        """
        Detect all persons in a surveillance frame and embed each face.

        Args:
            frame: BGR numpy array (from CCTV / cv2.VideoCapture).

        Returns:
            List of dicts per detected person:
            {
                "bbox":       [x1, y1, x2, y2],
                "confidence": float,
                "embedding":  np.ndarray (512,) or None
            }
        """
        if not self.use_yolo:
            raise RuntimeError(
                "process_frame() requires use_yolo=True. "
                "Use process_image() for clean face photos."
            )

        person_detections = self.detector.detect_persons(frame)
        results = []

        for det in person_detections:
            embedding = generate_embedding_from_crop(
                det["crop"], self.arcface_weights
            )
            results.append(
                {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "embedding": embedding,
                }
            )

        return results

    def process_video_frame_generator(self, video_path, frame_skip=5):
        """
        Process a video file frame-by-frame.

        Yields:
            (frame_index, frame, results) where results is the same
            structure as process_frame().
        """
        for frame_idx, frame, person_dets in self.detector.detect_from_video(
            video_path, frame_skip
        ):
            frame_results = []
            for det in person_dets:
                embedding = generate_embedding_from_crop(
                    det["crop"], self.arcface_weights
                )
                frame_results.append(
                    {
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "embedding": embedding,
                    }
                )
            yield frame_idx, frame, frame_results

    # ------------------------------------------------------------------
    # Gallery operations (used by org nodes)
    # ------------------------------------------------------------------

    def build_node_gallery(self, gallery_dir):
        """
        Build an embedding gallery from a node's gallery directory.

        Expected layout:
            gallery_dir/<identity_id>/<image>.jpg

        Returns:
            dict: {identity_id: np.ndarray (512,)}
        """
        return build_gallery(gallery_dir, self.arcface_weights)

    def match(self, query_embedding, gallery, threshold=0.45):
        """
        Privacy-preserving local match: return only Match / No-Match.

        Args:
            query_embedding: (512,) query face embedding.
            gallery:         {identity_id: embedding} dict.
            threshold:       Cosine similarity threshold for a positive match.

        Returns:
            {
                "match":    bool,
                "score":    float,
                "identity": str or None
            }
        """
        best_score = 0.0
        best_id = None

        for identity_id, gallery_embedding in gallery.items():
            score = cosine_similarity(query_embedding, gallery_embedding)
            if score > best_score:
                best_score = score
                best_id = identity_id

        matched = best_score >= threshold

        return {
            "match": matched,
            "score": best_score,
            "identity": best_id if matched else None,
        }

    # ------------------------------------------------------------------
    # Model update (called after FL round)
    # ------------------------------------------------------------------

    def reload_model(self, new_weights_path=None):
        """
        Reset the cached ArcFace model so the next embedding call loads
        the updated global model weights (after an FL aggregation round).
        """
        self.arcface_weights = new_weights_path
        reset_model()

