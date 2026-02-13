from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .config import Camera2DConfig
from .physics3d import Kinematics3D


def _project_points(points: np.ndarray, cfg: Camera2DConfig) -> np.ndarray:
    """Project 3D points to 2D normalized coordinates [0,1] using a simple view model.

    This is NOT a calibrated camera model; it's intentionally simple and intended for
    root-relative tracking (translation/scale largely cancels out).
    """
    # points: (N, 3) in world coords (x forward, y left, z up)
    pts = points.astype(np.float32)
    if cfg.view == "side":
        u = pts[:, 0]
        v = pts[:, 2]
    elif cfg.view == "front":
        u = pts[:, 1]
        v = pts[:, 2]
    elif cfg.view == "45":
        u = (pts[:, 0] + pts[:, 1]) / np.sqrt(2.0)
        v = pts[:, 2]
    else:
        raise ValueError(f"Unknown camera view: {cfg.view}")

    # Normalize to roughly [0,1] around origin using cfg.scale
    x = 0.5 + u * cfg.scale
    y = 0.5 - v * cfg.scale  # image y down
    return np.stack([x, y], axis=-1)


def project_kinematics_to_2d(kin: Kinematics3D, cfg: Camera2DConfig) -> Dict[str, np.ndarray]:
    """Return a small set of 2D keypoints matching the reference schema.

    Schema (J=8):
      0 pelvis
      1 torso_top (head in sim; shoulders in ref)
      2 knee_L
      3 knee_R
      4 ankle_L
      5 ankle_R
      6 toe_L
      7 toe_R
    """
    pts3d = np.stack(
        [
            kin.pelvis,
            kin.head,
            kin.knee_L,
            kin.knee_R,
            kin.ankle_L,
            kin.ankle_R,
            kin.toe_L,
            kin.toe_R,
        ],
        axis=0,
    )
    pts2d = _project_points(pts3d, cfg)
    names = ["pelvis", "torso_top", "knee_L", "knee_R", "ankle_L", "ankle_R", "toe_L", "toe_R"]
    return {n: pts2d[i] for i, n in enumerate(names)}


def project_kinematics_to_2d_array(kin: Kinematics3D, cfg: Camera2DConfig) -> np.ndarray:
    pts3d = np.stack(
        [
            kin.pelvis,
            kin.head,
            kin.knee_L,
            kin.knee_R,
            kin.ankle_L,
            kin.ankle_R,
            kin.toe_L,
            kin.toe_R,
        ],
        axis=0,
    )
    return _project_points(pts3d, cfg).astype(np.float32)  # (8,2)
