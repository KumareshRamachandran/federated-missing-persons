"""
vision/yolo_detector.py

Human detection using YOLOv8 (Ultralytics).
Responsible for:
  - Detecting people in surveillance/CCTV frames
  - Cropping detected person bounding boxes for downstream face processing
  - Handling surveillance conditions: low lighting, occlusion, crowd scenes

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class YOLODetector:
    """
    YOLOv8-based human detector for surveillance frames.
    Outputs cropped person bounding boxes for face embedding pipeline.
    """

    PERSON_CLASS_ID = 0  # COCO class index for 'person'

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_path: Path to YOLOv8 weights. Downloads pretrained if not found locally.
            confidence: Minimum detection confidence threshold.
        """
        # Resolve model_path if found in parent directories
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            for parent in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
                candidate = parent / model_path
                if candidate.exists():
                    resolved_path = candidate
                    break

        self.model = YOLO(str(resolved_path))
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def detect_persons(self, image: np.ndarray) -> list:
        """
        Detect all persons in a surveillance frame.

        Args:
            image: BGR image as numpy array (from cv2.imread or CCTV frame).

        Returns:
            List of dicts:
                {
                    "bbox":       [x1, y1, x2, y2],   # absolute pixel coords
                    "confidence": float,
                    "crop":       np.ndarray            # BGR crop of the person region
                }
        """
        if image is None or image.size == 0:
            return []

        # Ensure colour image (convert grayscale if needed)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        results = self.model(
            image,
            classes=[self.PERSON_CLASS_ID],
            conf=self.confidence,
            verbose=False,
        )

        detections = []
        h, w = image.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Add small padding (5 %) around the bounding box
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1_p = max(0, x1 - pad_x)
                y1_p = max(0, y1 - pad_y)
                x2_p = min(w, x2 + pad_x)
                y2_p = min(h, y2 + pad_y)

                crop = image[y1_p:y2_p, x1_p:x2_p].copy()

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "crop": crop,
                    }
                )

        return detections

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def detect_from_path(self, image_path: str) -> list:
        """
        Detect persons from an image file path.

        Args:
            image_path: Absolute or relative path to image file.

        Returns:
            Same structure as detect_persons().
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_persons(image)

    def detect_from_video(self, video_path: str, frame_skip: int = 5):
        """
        Generator: yield detections from each sampled video frame.

        Args:
            video_path:  Path to surveillance video file.
            frame_skip:  Process every Nth frame (reduces compute load).

        Yields:
            Tuple (frame_index, frame, detections) where detections follows
            the same structure as detect_persons().
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    detections = self.detect_persons(frame)
                    yield frame_idx, frame, detections

                frame_idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on image (for debugging).

        Returns:
            Annotated copy of the input image.
        """
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {conf:.2f}"
            cv2.putText(
                annotated, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return annotated
