"""
vision/finetune.py

PyTorch ArcFace Contrastive Fine-Tuning Module for Human-in-the-Loop (HITL) Feedback.

Provides function fine_tune_pair() to update ArcFace backbone weights based on
investigator feedback (Correct vs. Wrong match) and log updates for Federated Learning.

Member responsible: G N Lokesh (23BCE9603) — Computer Vision module
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from vision.arcface_model import ArcFaceModel

# Singleton ArcFace model instance for fine-tuning
_GLOBAL_FINETUNE_MODEL = None


def get_finetune_model() -> ArcFaceModel:
    global _GLOBAL_FINETUNE_MODEL
    if _GLOBAL_FINETUNE_MODEL is None:
        _GLOBAL_FINETUNE_MODEL = ArcFaceModel(pretrained="vggface2")
        _GLOBAL_FINETUNE_MODEL.eval()
    return _GLOBAL_FINETUNE_MODEL


def preprocess_image_to_tensor(img_input: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
    """Convert numpy array, PIL Image, or file path to PyTorch aligned face tensor [1, 3, 112, 112] in [-1, 1]."""
    if isinstance(img_input, str):
        img = Image.open(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert("RGB")
    elif isinstance(img_input, Image.Image):
        img = img_input.convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(img_input)}")

    img_resized = img.resize((112, 112))
    arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # [0, 1]
    arr = (arr - 0.5) / 0.5  # [-1, 1]
    arr = np.transpose(arr, (2, 0, 1))  # [3, 112, 112]
    tensor = torch.from_numpy(arr).unsqueeze(0).float()  # [1, 3, 112, 112]
    return tensor


def fine_tune_pair(
    query_image: Union[np.ndarray, Image.Image, str],
    crop_image: Union[np.ndarray, Image.Image, str],
    is_correct: bool,
    lr: float = 1e-4,
    steps: int = 3,
    save_log: bool = True,
) -> Dict:
    """
    Fine-tunes the ArcFace model on a single human-verified pair.

    Args:
        query_image: Missing person query photo (np.ndarray RGB, PIL Image, or path).
        crop_image: Detected face crop from surveillance video.
        is_correct: True if investigator confirmed correct match; False if wrong.
        lr: Learning rate for PyTorch optimizer.
        steps: Number of local gradient optimization steps.
        save_log: Whether to save update record to data/nodes/local_feedback_log.json.

    Returns:
        Dict containing s_before, s_after, delta, delta_formatted, is_correct, steps.
    """
    model = get_finetune_model()

    # Preprocess images to PyTorch tensors
    q_tensor = preprocess_image_to_tensor(query_image)
    c_tensor = preprocess_image_to_tensor(crop_image)

    # 1. Compute similarity BEFORE fine-tuning
    model.eval()
    with torch.no_grad():
        q_emb_0 = model(q_tensor)
        c_emb_0 = model(c_tensor)
        s_before = float(torch.sum(q_emb_0 * c_emb_0).item())

    # 2. Configure model for training (unfreeze final linear / block8 layers)
    model.train()
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Fine-tuning steps using PyTorch Contrastive / Cosine Similarity Loss
    for _ in range(steps):
        optimizer.zero_grad()
        # Combine into batch size 2 to satisfy BatchNorm requirements
        batch_input = torch.cat([q_tensor, c_tensor], dim=0)
        embs = model(batch_input)
        q_emb = embs[0:1]
        c_emb = embs[1:2]

        cos_sim = torch.sum(q_emb * c_emb, dim=1)

        if is_correct:
            # Correct match: Push similarity towards +1.0
            loss = 1.0 - cos_sim
        else:
            # Wrong match: Push similarity below 0.1
            loss = F.relu(cos_sim - 0.1)

        loss.backward()
        optimizer.step()

    # 4. Compute similarity AFTER fine-tuning
    model.eval()
    with torch.no_grad():
        q_emb_1 = model(q_tensor)
        c_emb_1 = model(c_tensor)
        s_after = float(torch.sum(q_emb_1 * c_emb_1).item())

    delta = s_after - s_before
    delta_pct = delta * 100.0

    result = {
        "s_before": round(s_before, 4),
        "s_after": round(s_after, 4),
        "delta": round(delta, 4),
        "delta_formatted": f"{'+' if delta >= 0 else ''}{delta_pct:.2f}%",
        "is_correct": is_correct,
        "steps_executed": steps,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 5. Save log for Federated Learning aggregation
    if save_log:
        try:
            log_dir = Path("data/nodes")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "local_feedback_log.json"

            logs = []
            if log_file.exists():
                with open(log_file, "r", encoding="utf-8") as f:
                    try:
                        logs = json.load(f)
                    except Exception:
                        logs = []

            log_entry = {
                "timestamp": result["timestamp"],
                "is_correct": is_correct,
                "s_before": result["s_before"],
                "s_after": result["s_after"],
                "delta": result["delta"],
            }
            logs.append(log_entry)

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write feedback log: {e}")

    return result


if __name__ == "__main__":
    print("Testing vision/finetune.py...")
    q_dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    c_dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)

    res = fine_tune_pair(q_dummy, c_dummy, is_correct=True, steps=3)
    print(f"[OK] Correct Match Fine-Tune: {res['s_before']*100:.2f}% -> {res['s_after']*100:.2f}% (Delta: {res['delta_formatted']})")

    res_wrong = fine_tune_pair(q_dummy, c_dummy, is_correct=False, steps=3)
    print(f"[OK] Wrong Match Fine-Tune:   {res_wrong['s_before']*100:.2f}% -> {res_wrong['s_after']*100:.2f}% (Delta: {res_wrong['delta_formatted']})")
