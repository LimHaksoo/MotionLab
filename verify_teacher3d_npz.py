from __future__ import annotations

"""Debug viewer for teacher3d.py outputs.

The old scaffold's `teacher3d.py` writes an .npz with:
  - obs: (T,47)
  - act: (T,14)
  - meta: dict(...)

This script lets you quickly verify whether the extracted sequence "looks" like the source run,
by rendering a simple 3D skeleton animation and printing sanity stats.

Usage:
  python verify_teacher3d_npz.py --npz teacher.npz --out preview.mp4
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from motionlab_track_rl.config import BodyParams3D
from motionlab_track_rl.physics3d import compute_kinematics_3d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize teacher3d npz (obs/act) as a skeleton MP4.")
    p.add_argument("--npz", required=True, help="teacher3d output .npz")
    p.add_argument("--out", default="teacher3d_preview.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--azim", type=float, default=-90)
    p.add_argument("--elev", type=float, default=15)
    p.add_argument("--max-seconds", type=float, default=None, help="optional: clip the preview")
    return p.parse_args()


def _draw(ax, kin, title: str = ""):
    def seg(a, b, lw=3):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=lw)

    seg(kin.pelvis, kin.head, lw=4)
    seg(kin.hip_L, kin.knee_L)
    seg(kin.knee_L, kin.ankle_L)
    seg(kin.ankle_L, kin.toe_L)
    seg(kin.hip_R, kin.knee_R)
    seg(kin.knee_R, kin.ankle_R)
    seg(kin.ankle_R, kin.toe_R)

    # ground
    ax.plot([-1.5, 1.5], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [-1.5, 1.5], [0, 0], linewidth=1)

    c = kin.pelvis
    ax.set_xlim(c[0] - 1.2, c[0] + 1.2)
    ax.set_ylim(c[1] - 1.2, c[1] + 1.2)
    ax.set_zlim(0.0, c[2] + 1.2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(False)


def main() -> None:
    args = parse_args()
    d = np.load(args.npz, allow_pickle=True)
    obs = d["obs"].astype(np.float32)
    act = d["act"].astype(np.float32)
    meta = d.get("meta", {})
    if isinstance(meta, np.ndarray):
        # np.savez stores dict as 0-d object array sometimes
        meta = meta.item()

    # Basic sanity prints
    print("files:", list(d.files))
    print("obs:", obs.shape, "act:", act.shape)
    print("meta:", meta)

    # Teacher3d obs layout (from scaffold):
    #   pelvis_z(1), pelvis_v(3), torso(2), torso_rate(2), q(14), qd(14), stance_phase(2), stats(4), body(5)
    q = obs[:, 8:22]
    pelvis_z = obs[:, 0]
    pelvis_v = obs[:, 1:4]

    # Build body params from meta if available (fallback to defaults)
    b = BodyParams3D(
        mass=float(meta.get("mass", 70.0)) if isinstance(meta, dict) else 70.0,
        thigh=float(meta.get("thigh", 0.45)) if isinstance(meta, dict) else 0.45,
        shank=float(meta.get("shank", 0.45)) if isinstance(meta, dict) else 0.45,
        foot=float(meta.get("foot", 0.25)) if isinstance(meta, dict) else 0.25,
        torso=float(meta.get("torso", 0.55)) if isinstance(meta, dict) else 0.55,
        hip_width=0.18,
    )

    # dt derived from meta fps if present, else from output fps
    src_fps = float(meta.get("fps", args.fps)) if isinstance(meta, dict) else float(args.fps)
    dt = 1.0 / max(1e-6, src_fps)

    # quick action saturation check
    sat = float(np.mean(np.abs(act) > 0.98))
    print({"action_saturation_frac": sat})

    # integrate pelvis x/y for visualization
    pelvis_xy = np.zeros((obs.shape[0], 2), dtype=np.float32)
    for t in range(1, obs.shape[0]):
        pelvis_xy[t, 0] = pelvis_xy[t - 1, 0] + pelvis_v[t - 1, 0] * dt
        pelvis_xy[t, 1] = pelvis_xy[t - 1, 1] + pelvis_v[t - 1, 1] * dt

    # render
    stride = max(1, int(round(src_fps / max(1, args.fps))))
    max_frames = obs.shape[0]
    if args.max_seconds is not None:
        max_frames = min(max_frames, int(float(args.max_seconds) * src_fps))

    frames = []
    for t in range(0, max_frames, stride):
        pelvis = np.array([pelvis_xy[t, 0], pelvis_xy[t, 1], pelvis_z[t]], dtype=np.float32)
        kin = compute_kinematics_3d(pelvis=pelvis, q=q[t], body=b, pelvis_yaw=0.0)

        fig = plt.figure(figsize=(6, 4), dpi=140)
        ax = fig.add_subplot(111, projection="3d")
        _draw(ax, kin, title=f"t={t*dt:.2f}s")
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(args.out, frames, fps=int(args.fps), codec="libx264", quality=8)
    print(f"Saved preview: {args.out}  (frames={len(frames)})")


if __name__ == "__main__":
    main()
