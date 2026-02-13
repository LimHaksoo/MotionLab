from __future__ import annotations
from typing import Callable, Dict, Optional, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .physics3d import Kinematics3D
from .camera import project_kinematics_to_2d_array
from .config import Camera2DConfig


def _draw_skeleton_3d(ax, kin: Kinematics3D, title: str = ""):
    # segments
    def seg(a, b, lw=3):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], linewidth=lw)

    seg(kin.pelvis, kin.head, lw=4)

    seg(kin.hip_L, kin.knee_L)
    seg(kin.knee_L, kin.ankle_L)
    seg(kin.ankle_L, kin.toe_L)
    seg(kin.ankle_L, kin.heel_L)

    seg(kin.hip_R, kin.knee_R)
    seg(kin.knee_R, kin.ankle_R)
    seg(kin.ankle_R, kin.toe_R)
    seg(kin.ankle_R, kin.heel_R)

    # ground plane reference
    gx = np.array([-1.5, 1.5])
    gy = np.array([-1.5, 1.5])
    ax.plot(gx, [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], gy, [0, 0], linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # view and limits centered at pelvis
    c = kin.pelvis
    ax.set_xlim(c[0]-1.2, c[0]+1.2)
    ax.set_ylim(c[1]-1.2, c[1]+1.2)
    ax.set_zlim(0.0, c[2]+1.2)
    ax.view_init(elev=15, azim=-90)
    ax.grid(False)


def _draw_2d_overlay(ax, ref2d: Optional[np.ndarray], sim2d: Optional[np.ndarray], title: str = ""):
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y-down
    ax.axis("off")
    if ref2d is not None:
        ax.scatter(ref2d[:, 0], ref2d[:, 1], s=12, marker="o")
    if sim2d is not None:
        ax.scatter(sim2d[:, 0], sim2d[:, 1], s=12, marker="x")


def rollout_compare_to_video(
    env_track,
    policy_track: Callable[[np.ndarray], np.ndarray],
    env_ideal,
    policy_ideal: Callable[[np.ndarray], np.ndarray],
    out_path: str,
    seconds: float = 10.0,
    fps: int = 30,
    deterministic: bool = True,
    fixed_duration: bool = True,
    cam_cfg: Optional[Camera2DConfig] = None,
    title_prefix: str = "MotionLab",
    body_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run tracking and ideal rollouts, render side-by-side 3D skeletons.

    If fixed_duration=True, pads the video to 'seconds' even if an episode ends early.
    """
    cam_cfg = cam_cfg or Camera2DConfig()

    # Reset both envs
    if body_override is not None:
        obs_t, info_t = env_track.reset(randomize_body=False, body_override=body_override)
    else:
        obs_t, info_t = env_track.reset()
    obs_i, info_i = env_ideal.reset(randomize_body=False, body_override=env_track.body)  # match body for comparison

    dt = env_track.cfg.dt
    max_steps = int(seconds / dt)
    stride = max(1, int(round((1.0 / fps) / dt)))
    target_frames = int(round(seconds * fps))

    frames: List[np.ndarray] = []
    last_frame = None

    for step in range(max_steps):
        a_t = policy_track(obs_t)
        a_i = policy_ideal(obs_i)

        obs_t, r_t, done_t, info_t = env_track.step(a_t)
        obs_i, r_i, done_i, info_i = env_ideal.step(a_i)

        if step % stride == 0:
            kin_t = env_track.get_kinematics()
            kin_i = env_ideal.get_kinematics()

            # 2D overlay for tracking (ref vs sim)
            ref2d = None
            if getattr(env_track, "ref", None) is not None:
                ref2d = env_track.ref.keypoints2d[env_track.ref_idx]
            sim2d = project_kinematics_to_2d_array(kin_t, cam_cfg)

            fig = plt.figure(figsize=(10, 4), dpi=140)
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            ax3 = fig.add_subplot(1, 3, 3)

            _draw_skeleton_3d(ax1, kin_t, title=f"Closest (track)\nerr={info_t.get('pose_err',0):.3f}")
            _draw_skeleton_3d(ax2, kin_i, title="Ideal")
            _draw_2d_overlay(ax3, ref2d, sim2d, title="2D overlay (ref=o, sim=x)")

            fig.suptitle(f"{title_prefix} | t={info_t.get('t',0):.2f}s")

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = rgba[:, :, :3].copy()
            frames.append(frame)
            last_frame = frame
            plt.close(fig)

        if (done_t or done_i) and not fixed_duration:
            break

        if done_t and fixed_duration:
            # stop stepping track; keep last state stable by reusing last obs/actions
            pass

    if fixed_duration and last_frame is not None:
        while len(frames) < target_frames:
            frames.append(last_frame)

    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    return {
        "frames": len(frames),
        "seconds": len(frames) / float(fps),
        "final_t": float(info_t.get("t", 0.0)),
        "final_pose_err": float(info_t.get("pose_err", 0.0)),
    }