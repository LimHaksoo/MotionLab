from __future__ import annotations
from typing import Tuple, Optional
import numpy as np


def seed_everything(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def safe_norm(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(float(np.sum(x * x))) + eps)


def normalize_2d_root_relative(
    kps: np.ndarray,  # (J, 2) in [0,1] image coords
    pelvis_idx: int = 0,
    scale_idx: int = 1,
    eps: float = 1e-6,
) -> np.ndarray:
    """Root-relative, scale-normalized 2D keypoints.

    - Subtract pelvis (root) position.
    - Normalize by distance between pelvis and torso_top (or other scale_idx joint).

    Returns: (J, 2) float32
    """
    kps = kps.astype(np.float32)
    root = kps[pelvis_idx]
    rel = kps - root[None, :]
    scale = np.linalg.norm(kps[scale_idx] - kps[pelvis_idx]) + eps
    rel = rel / scale
    return rel


def stack_ref_window(
    ref_seq: np.ndarray,  # (T, J, 2)
    t: int,
    window: int,
) -> np.ndarray:
    """Stack reference frames: [t-window+1 ... t] (clamped)."""
    T = ref_seq.shape[0]
    frames = []
    for i in range(window):
        idx = max(0, min(T - 1, t - (window - 1 - i)))
        frames.append(ref_seq[idx])
    return np.concatenate(frames, axis=0)  # (window*J, 2)
