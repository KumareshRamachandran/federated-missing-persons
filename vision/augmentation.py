"""
vision/augmentation.py

Dataset augmentation using Albumentations and OpenCV to simulate CCTV/surveillance conditions.
Includes apply_cctv_distortion() function as requested, plus SurveillanceAugmentor class.

Simulates:
  1) Downscaling and upscaling (compression artifacts)
  2) Gaussian Blur and Motion Blur (moving subjects)
  3) CLAHE (visibility in low illumination)
  4) Random occlusions / black rectangles (obstacles)

Member: G N Lokesh (23BCE9603) — Computer Vision module
"""

from __future__ import annotations

import cv2
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import albumentations as A


def apply_cctv_distortion(image: np.ndarray) -> np.ndarray:
    """
    Applies a randomized Albumentations + OpenCV pipeline simulating CCTV/surveillance distortions.

    Distortion pipeline:
      1) Downscaling and upscaling to simulate low resolution / compression artifacts.
      2) Gaussian Blur and Motion Blur to simulate moving subjects.
      3) CLAHE (Contrast Limited Adaptive Histogram Equalization) for low illumination.
      4) Random occlusions (black rectangles) simulating physical obstacles.

    Args:
        image: BGR uint8 NumPy array.

    Returns:
        Augmented BGR uint8 NumPy array.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]
    max_h = max(8, int(h * 0.25))
    max_w = max(8, int(w * 0.25))

    # Support both Albumentations 2.x and 1.x syntax
    try:
        downscale_transform = A.Downscale(scale_range=(0.25, 0.5), p=0.5)
    except Exception:
        downscale_transform = A.Downscale(scale_min=0.25, scale_max=0.5, p=0.5)

    try:
        dropout_transform = A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(8, max_h),
            hole_width_range=(8, max_w),
            fill=0,
            p=0.5,
        )
    except Exception:
        dropout_transform = A.CoarseDropout(
            max_holes=4,
            max_height=max_h,
            max_width=max_w,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.5,
        )

    pipeline = A.Compose([
        # 1) Downscaling and upscaling to simulate compression artifacts
        downscale_transform,
        # 2) Gaussian Blur and Motion Blur to simulate moving subjects
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.5),
        # 3) CLAHE to improve visibility in low illumination
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        # 4) Random occlusions (black rectangles) simulating obstacles
        dropout_transform,
    ])

    augmented = pipeline(image=image)["image"]
    return augmented


class SurveillanceAugmentor:
    """
    Applies surveillance-realistic augmentations to face / person images.
    Ensures the vision model is robust to real-world CCTV conditions.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def low_illumination(self, image: np.ndarray, factor: float | None = None) -> np.ndarray:
        if factor is None:
            factor = random.uniform(0.3, 0.7)
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def motion_blur(self, image: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
        if kernel_size is None:
            kernel_size = random.choice([5, 7, 9])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        return cv2.filter2D(image, -1, kernel)

    def add_occlusion(self, image: np.ndarray, num_patches: int = 2) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]
        for _ in range(num_patches):
            pw = random.randint(int(w * 0.10), int(w * 0.25))
            ph = random.randint(int(h * 0.10), int(h * 0.25))
            px = random.randint(0, max(0, w - pw))
            py = random.randint(0, max(0, h - ph))
            img[py : py + ph, px : px + pw] = 0
        return img

    def add_gaussian_noise(self, image: np.ndarray, std: float = 15.0) -> np.ndarray:
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def jpeg_compression(self, image: np.ndarray, quality: int | None = None) -> np.ndarray:
        if quality is None:
            quality = random.randint(20, 60)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded

    def augment(self, image: np.ndarray, mode: str = "random") -> np.ndarray:
        if mode == "cctv":
            return apply_cctv_distortion(image)

        transforms = [
            self.low_illumination,
            self.motion_blur,
            self.add_occlusion,
            self.add_gaussian_noise,
            self.jpeg_compression,
        ]

        if mode in ("all", "heavy"):
            selected = transforms
        elif mode == "light":
            selected = [self.low_illumination, self.add_gaussian_noise]
        else:
            k = random.randint(1, 3)
            selected = random.sample(transforms, k)

        result = image.copy()
        for transform in selected:
            result = transform(result)
        return result

    def augment_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augments_per_image: int = 3,
        mode: str = "random",
        extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_images = [p for p in in_path.rglob("*") if p.suffix.lower() in extensions]
        if not all_images:
            print(f"[augmentation] No images found in {input_dir}")
            return

        print(f"[augmentation] Processing {len(all_images)} images × {augments_per_image} augmentations from {input_dir}")

        for img_path in tqdm(all_images, desc="Augmenting"):
            rel_path = img_path.relative_to(in_path)
            dest_dir = out_path / rel_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            cv2.imwrite(str(dest_dir / img_path.name), image)
            stem = img_path.stem
            ext = img_path.suffix
            for i in range(augments_per_image):
                aug_image = self.augment(image, mode=mode)
                aug_name = f"{stem}_aug{i}{ext}"
                cv2.imwrite(str(dest_dir / aug_name), aug_image)

        print(f"[augmentation] Done — saved to {output_dir}")


if __name__ == "__main__":
    print("Testing apply_cctv_distortion()...")
    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (250, 250), (250, 200, 150), -1)
    cv2.circle(test_img, (150, 150), 40, (100, 50, 0), -1)

    augmented_img = apply_cctv_distortion(test_img)
    print(f"Original shape: {test_img.shape}, Augmented shape: {augmented_img.shape}")
    assert augmented_img.shape == test_img.shape
    print("[OK] apply_cctv_distortion test passed!")

