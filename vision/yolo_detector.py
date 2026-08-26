"""
face_engine/yolo_detector.py

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

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        """
        Args:
            model_path: Path to YOLOv8 weights. Downloads pretrained if not found.
            confidence: Minimum detection confidence threshold.
        """
        # TODO: self.model = YOLO(model_path)
        self.confidence = confidence

    def detect_persons(self, image: np.ndarray) -> list:
        """
        Detect all persons in a surveillance frame.

        Args:
            image: BGR image as numpy array (from cv2.imread or CCTV frame).

        Returns:
            List of dicts: [{"bbox": [x1,y1,x2,y2], "confidence": float, "crop": np.ndarray}]
        """
        # TODO: results = self.model(image, classes=[0], conf=self.confidence)
        # TODO: For each detection, extract bbox and crop person region
        # TODO: Return list of detection dicts
        pass

    def detect_from_path(self, image_path: str) -> list:
        """Detect persons from image file path."""
        # TODO: image = cv2.imread(image_path)
        # TODO: return self.detect_persons(image)
        pass

    def detect_from_video(self, video_path: str, frame_skip: int = 5):
        """
        Generator: yield detections from each sampled video frame.

        Args:
            video_path: Path to surveillance video file.
            frame_skip: Process every Nth frame (reduces compute load).
        """
        # TODO: cap = cv2.VideoCapture(video_path)
        # TODO: Yield detections every frame_skip frames
        pass
