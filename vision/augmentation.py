"""
vision/augmentation.py

Dataset augmentation to simulate CCTV/surveillance conditions.
Applied to CelebA and the Custom Surveillance Dataset before training.

Simulates:
  - Low illumination (brightness/contrast reduction)
  - Motion blur (kernel convolution)
  - Occlusion (random rectangular patches)
  - JPEG compression artifacts (quality degradation)
  - Noise (Gaussian)

Member: G N Lokesh (23BCE9603) — Computer Vision
"""

import cv2
import numpy as np
import random
from pathlib import Path


class SurveillanceAugmentor:
    """
    Applies surveillance-realistic augmentations to face images.
    Ensures the model is robust to real-world CCTV conditions.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def low_illumination(self, image: np.ndarray, factor: float = None) -> np.ndarray:
        """Reduce brightness to simulate poor lighting conditions."""
        # TODO: factor = random.uniform(0.3, 0.7) if factor is None else factor
        # TODO: return np.clip(image * factor, 0, 255).astype(np.uint8)
        pass

    def motion_blur(self, image: np.ndarray, kernel_size: int = None) -> np.ndarray:
        """Apply horizontal motion blur to simulate camera movement."""
        # TODO: kernel_size = random.choice([5, 7, 9]) if kernel_size is None else kernel_size
        # TODO: kernel = np.zeros((kernel_size, kernel_size))
        # TODO: kernel[kernel_size//2, :] = 1.0 / kernel_size
        # TODO: return cv2.filter2D(image, -1, kernel)
        pass

    def add_occlusion(self, image: np.ndarray, num_patches: int = 2) -> np.ndarray:
        """Add random black rectangles to simulate occlusion (hair, glasses, mask)."""
        # TODO: For each patch, pick random x,y,w,h and fill with black
        pass

    def add_gaussian_noise(self, image: np.ndarray, std: float = 15.0) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise in low-light cameras."""
        # TODO: noise = np.random.normal(0, std, image.shape)
        # TODO: return np.clip(image + noise, 0, 255).astype(np.uint8)
        pass

    def jpeg_compression(self, image: np.ndarray, quality: int = None) -> np.ndarray:
        """Simulate JPEG compression artifacts common in CCTV footage."""
        # TODO: quality = random.randint(30, 60) if quality is None else quality
        # TODO: encode → decode to simulate compression
        pass

    def augment(self, image: np.ndarray, mode: str = "random") -> np.ndarray:
        """
        Apply a combination of augmentations.

        Args:
            mode: "random" applies a random subset, "all" applies all augmentations.
        """
        # TODO: Select augmentations based on mode and apply sequentially
        pass

    def augment_dataset(self, input_dir: str, output_dir: str, augments_per_image: int = 3):
        """
        Augment all images in a directory and save to output_dir.

        Args:
            input_dir: Source image directory.
            output_dir: Destination for augmented images.
            augments_per_image: Number of augmented variants per original image.
        """
        # TODO: Walk input_dir, apply augment() N times per image, save to output_dir
        pass
