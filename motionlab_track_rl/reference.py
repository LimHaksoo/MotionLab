from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from .utils import normalize_2d_root_relative, stack_ref_window


@dataclass
class ReferenceSequence:
    keypoints2d: np.ndarray  # (T, J, 2) normalized [0,1]
    vis: np.ndarray          # (T, J) in [0,1]
    fps: float
    width: int
    height: int
    name: str = "reference"

    @property
    def T(self) -> int:
        return int(self.keypoints2d.shape[0])

    @property
    def J(self) -> int:
        return int(self.keypoints2d.shape[1])


def load_reference_npz(path: str) -> ReferenceSequence:
    d = np.load(path, allow_pickle=True)
    kps = d["keypoints2d"].astype(np.float32)
    vis = d["vis"].astype(np.float32)
    fps = float(d["fps"])
    width = int(d.get("width", 0))
    height = int(d.get("height", 0))
    name = str(d.get("name", "reference"))
    return ReferenceSequence(keypoints2d=kps, vis=vis, fps=fps, width=width, height=height, name=name)


def ref_features_at(
    ref: ReferenceSequence,
    t_idx: int,
    window: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (feat, vis) at a reference time index.

    feat is a flattened vector containing stacked root-relative normalized 2D keypoints.
    vis is stacked similarly (window*J,)
    """
    kps_stack = stack_ref_window(ref.keypoints2d, t_idx, window=window)  # (window*J, 2)
    vis_stack = stack_ref_window(ref.vis[..., None], t_idx, window=window)[..., 0]  # (window*J,)

    # Root-relative normalization per frame in the stack
    WJ = kps_stack.shape[0]
    J = ref.J
    window_eff = WJ // J
    out = []
    for i in range(window_eff):
        k = kps_stack[i * J : (i + 1) * J]
        out.append(normalize_2d_root_relative(k, pelvis_idx=0, scale_idx=1))
    out = np.concatenate(out, axis=0)  # (window*J,2)

    return out.reshape(-1).astype(np.float32), vis_stack.astype(np.float32)


def ref_features_vel_at(
    ref: ReferenceSequence,
    t_idx: int,
    window: int = 1,
    dt: float = 1/120,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference velocity-like features using finite differences in 2D.

    Returns flattened vector (window*J*2,) for current and previous frame difference, and stacked visibility.
    """
    f_now, v_now = ref_features_at(ref, t_idx, window=window)
    f_prev, v_prev = ref_features_at(ref, max(0, t_idx - 1), window=window)
    vel = (f_now - f_prev) / max(dt, 1e-6)
    vis = np.minimum(v_now, v_prev)
    return vel.astype(np.float32), vis.astype(np.float32)
