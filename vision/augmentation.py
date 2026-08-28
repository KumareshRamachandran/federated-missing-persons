"""
vision/augmentation.py

Dataset augmentation to simulate CCTV/surveillance conditions.
Applied to CelebA and the Custom Surveillance Dataset before training.

Simulates:
  - Low illumination (brightness/contrast reduction)
  - Motion blur (horizontal kernel convolution)
  - Occlusion (random rectangular patches)
  - JPEG compression artifacts (quality degradation)
  - Gaussian sensor noise

Member: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm


class SurveillanceAugmentor:
    """
    Applies surveillance-realistic augmentations to face / person images.
    Ensures the vision model is robust to real-world CCTV conditions.

    All transform methods accept and return BGR uint8 numpy arrays.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Individual augmentation transforms
    # ------------------------------------------------------------------

    def low_illumination(
        self, image: np.ndarray, factor: float | None = None
    ) -> np.ndarray:
        """
        Reduce brightness to simulate poor or low-light conditions.

        Args:
            image:  BGR uint8 numpy array.
            factor: Brightness multiplier in [0.3, 0.7]. Randomly chosen if None.

        Returns:
            Darkened BGR image (uint8).
        """
        if factor is None:
            factor = random.uniform(0.3, 0.7)
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def motion_blur(
        self, image: np.ndarray, kernel_size: int | None = None
    ) -> np.ndarray:
        """
        Apply horizontal motion blur to simulate camera movement or fast motion.

        Args:
            image:       BGR uint8 numpy array.
            kernel_size: Blur kernel size (odd int). Randomly chosen from [5, 7, 9] if None.

        Returns:
            Motion-blurred BGR image (uint8).
        """
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9])

        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def add_occlusion(
        self, image: np.ndarray, num_patches: int = 2
    ) -> np.ndarray:
        """
        Add random black rectangles to simulate occlusion (hair, glasses, masks).

        Patches are constrained to ≤ 25 % of image dimension per side.

        Args:
            image:       BGR uint8 numpy array.
            num_patches: Number of black rectangles to draw.

        Returns:
            Occluded BGR image (uint8).
        """
        img = image.copy()
        h, w = img.shape[:2]

        for _ in range(num_patches):
            # Patch width/height: 10–25 % of image dimension
            pw = random.randint(int(w * 0.10), int(w * 0.25))
            ph = random.randint(int(h * 0.10), int(h * 0.25))
            # Random top-left corner
            px = random.randint(0, max(0, w - pw))
            py = random.randint(0, max(0, h - ph))
            img[py : py + ph, px : px + pw] = 0

        return img

    def add_gaussian_noise(
        self, image: np.ndarray, std: float = 15.0
    ) -> np.ndarray:
        """
        Add Gaussian noise to simulate sensor noise in low-light cameras.

        Args:
            image: BGR uint8 numpy array.
            std:   Standard deviation of the Gaussian noise.

        Returns:
            Noisy BGR image (uint8).
        """
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def jpeg_compression(
        self, image: np.ndarray, quality: int | None = None
    ) -> np.ndarray:
        """
        Simulate JPEG compression artifacts common in low-bitrate CCTV footage.

        Args:
            image:   BGR uint8 numpy array.
            quality: JPEG quality (1–100). Randomly chosen from [20, 60] if None.

        Returns:
            Re-decoded compressed BGR image (uint8).
        """
        if quality is None:
            quality = random.randint(20, 60)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    # ------------------------------------------------------------------
    # Composite augmentation
    # ------------------------------------------------------------------

    def augment(self, image: np.ndarray, mode: str = "random") -> np.ndarray:
        """
        Apply a combination of augmentations.

        Args:
            image: BGR uint8 numpy array.
            mode:  "random" — apply 1–3 randomly chosen augmentations.
                   "all"    — apply every augmentation in a fixed order.
                   "light"  — low_illumination + noise only (mild degradation).
                   "heavy"  — all augmentations (maximum degradation).

        Returns:
            Augmented BGR image (uint8).
        """
        transforms = [
            self.low_illumination,
            self.motion_blur,
            self.add_occlusion,
            self.add_gaussian_noise,
            self.jpeg_compression,
        ]

        if mode == "all" or mode == "heavy":
            selected = transforms
        elif mode == "light":
            selected = [self.low_illumination, self.add_gaussian_noise]
        else:  # "random"
            k = random.randint(1, 3)
            selected = random.sample(transforms, k)

        result = image.copy()
        for transform in selected:
            result = transform(result)

        return result

    # ------------------------------------------------------------------
    # Dataset-level augmentation
    # ------------------------------------------------------------------

    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augments_per_image: int = 3,
        mode: str = "random",
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        """
        Augment all images in a directory tree and save to output_dir.

        Preserves the identity sub-directory structure:
            input_dir/<identity>/<image>.jpg
            → output_dir/<identity>/<image>.jpg          (original copy)
            → output_dir/<identity>/<image>_aug0.jpg     (augmented variants)
            → output_dir/<identity>/<image>_aug1.jpg

        Args:
            input_dir:          Source directory (identity sub-folders or flat).
            output_dir:         Destination for augmented images.
            augments_per_image: Number of augmented variants per original.
            mode:               Augmentation mode passed to augment().
            extensions:         Image file extensions to process.
        """
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Collect all image files
        all_images = [
            p for p in in_path.rglob("*")
            if p.suffix.lower() in extensions
        ]

        if not all_images:
            print(f"[augmentation] No images found in {input_dir}")
            return

        print(f"[augmentation] Processing {len(all_images)} images "
              f"× {augments_per_image} augmentations from {input_dir}")

        for img_path in tqdm(all_images, desc="Augmenting"):
            # Preserve relative directory structure
            rel_path = img_path.relative_to(in_path)
            dest_dir = out_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Copy original
            cv2.imwrite(str(dest_dir / img_path.name), image)

            # Write augmented variants
            stem = img_path.stem
            ext = img_path.suffix
            for i in range(augments_per_image):
                aug_image = self.augment(image, mode=mode)
                aug_name = f"{stem}_aug{i}{ext}"
                cv2.imwrite(str(dest_dir / aug_name), aug_image)

        print(f"[augmentation] Done — saved to {output_dir}")
