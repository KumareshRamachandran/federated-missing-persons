"""
vision/yolo_detector.py

Human detection using YOLOv8 (Ultralytics).
Responsible for:
  - Detecting people in surveillance/CCTV frames
  - Cropping detected person bounding boxes for downstream face processing
  - Handling surveillance conditions: low lighting, occlusion, crowd scenes

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class YOLOPersonDetector:
    """
    YOLOv8-based human detector for surveillance frames.
    Detects persons in images/frames and returns cropped person regions.
    """

    PERSON_CLASS_ID = 0  # COCO class index for 'person'

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Loads the pre-trained YOLOv8 model (default yolov8n.pt).
        """
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            for parent in [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]:
                candidate = parent / model_path
                if candidate.exists():
                    resolved_path = candidate
                    break

        self.model = YOLO(str(resolved_path))
        self.confidence = confidence

    def detect_persons(self, image_path_or_frame: str | Path | np.ndarray, conf_threshold: float = 0.5) -> list[np.ndarray]:
        """
        Runs inference, filters results strictly for COCO class 0 (person),
        and returns a list of cropped NumPy array images of detected persons.

        Args:
            image_path_or_frame: File path (str/Path) or BGR image NumPy array.
            conf_threshold: Minimum detection confidence threshold (default 0.5).

        Returns:
            List of cropped person regions as BGR NumPy arrays.
        """
        if isinstance(image_path_or_frame, (str, Path)):
            image = cv2.imread(str(image_path_or_frame))
            if image is None:
                raise FileNotFoundError(f"Could not read image from path: {image_path_or_frame}")
        else:
            image = image_path_or_frame

        if image is None or image.size == 0:
            return []

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        results = self.model(
            image,
            classes=[self.PERSON_CLASS_ID],
            conf=conf_threshold,
            verbose=False,
        )

        person_crops = []
        h, w = image.shape[:2]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1_c = max(0, x1)
                y1_c = max(0, y1)
                x2_c = min(w, x2)
                y2_c = min(h, y2)

                crop = image[y1_c:y2_c, x1_c:x2_c].copy()
                if crop.size > 0:
                    person_crops.append(crop)

        return person_crops

    def detect_persons_detailed(self, image: np.ndarray, conf_threshold: float = 0.5) -> list[dict]:
        """
        Detailed detection method returning bounding box metadata alongside crops.
        Used by VisionPipeline and video frame processors.
        """
        if image is None or image.size == 0:
            return []

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        results = self.model(
            image,
            classes=[self.PERSON_CLASS_ID],
            conf=conf_threshold,
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


class YOLODetector(YOLOPersonDetector):
    """
    Backward-compatible wrapper class for VisionPipeline.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        super().__init__(model_path=model_path)
        self.confidence = confidence

    def detect_persons(self, image: np.ndarray) -> list[dict]:
        return self.detect_persons_detailed(image, conf_threshold=self.confidence)

    def detect_from_path(self, image_path: str) -> list[dict]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.detect_persons(image)

    def detect_from_video(self, video_path: str, frame_skip: int = 5):
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

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
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


if __name__ == "__main__":
    print("Testing YOLOPersonDetector...")
    detector = YOLOPersonDetector("yolov8n.pt")

    sample_path = Path(__file__).resolve().parent / "photos" / "missing.png"
    if sample_path.exists():
        print(f"Running detection on sample image: {sample_path}")
        crops = detector.detect_persons(str(sample_path), conf_threshold=0.3)
        print(f"Detected {len(crops)} person crop(s).")
    else:
        print("Sample image missing.png not found, testing with synthetic image.")
        synthetic_img = np.full((600, 400, 3), 200, dtype=np.uint8)
        cv2.circle(synthetic_img, (200, 150), 40, (100, 100, 100), -1)
        cv2.rectangle(synthetic_img, (150, 200), (250, 500), (50, 50, 50), -1)
        crops = detector.detect_persons(synthetic_img, conf_threshold=0.1)
        print(f"Ran detection on synthetic image. Crops returned: {len(crops)}")

    print("[OK] YOLOPersonDetector test passed!")

