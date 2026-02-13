from __future__ import annotations
import argparse
import os
import tempfile

import numpy as np
import torch

from motionlab_track_rl.config import EnvConfig, Camera2DConfig
from motionlab_track_rl.envs import RunningBipedEnv3D, TrackingBipedEnv3D
from motionlab_track_rl.reference import load_reference_npz
from motionlab_track_rl.model import ActorCritic
from motionlab_track_rl.render_compare import rollout_compare_to_video
from motionlab_track_rl.teacher_video import extract_reference_from_video


def load_ac_from_ckpt(ckpt_path: str, obs_dim: int, act_dim: int, device: str = "cpu") -> ActorCritic:
    payload = torch.load(ckpt_path, map_location=device)
    model_cfg = payload.get("model_cfg", {})
    arch = str(model_cfg.get("arch", "mog"))
    mog_components = int(model_cfg.get("mog_components", 4))
    hidden_sizes = tuple(int(x) for x in model_cfg.get("hidden_sizes", (1024, 1024, 512)))
    activation = str(model_cfg.get("activation", "silu"))
    layernorm = bool(model_cfg.get("layernorm", True))
    residual_blocks = int(model_cfg.get("residual_blocks", 2))

    # Recreate the *same* architecture that was trained.
    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        arch=arch,
        hidden_sizes=hidden_sizes,
        activation=activation,
        layernorm=layernorm,
        residual_blocks=residual_blocks,
        mog_components=mog_components,
    )
    ac.load_state_dict(payload["state_dict"], strict=False)
    ac.eval()
    return ac


def parse_args():
    p = argparse.ArgumentParser(description="MotionLab inference: closest(track) vs ideal, rendered to MP4.")
    p.add_argument("--ckpt-track", required=True, help="checkpoint for tracking policy (task=track)")
    p.add_argument("--ckpt-ideal", required=True, help="checkpoint for ideal policy (task=ideal)")
    p.add_argument("--out", default="closest_vs_ideal.mp4")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")

    # Input: either ref npz or a video to extract reference from
    p.add_argument("--ref", type=str, default=None)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--every", type=int, default=1, help="when extracting from video, sample every N frames")
    p.add_argument("--min-vis", type=float, default=0.2)

    # projection config for tracking overlay
    p.add_argument("--cam-view", type=str, default="side", choices=["side", "front", "45"])
    p.add_argument("--cam-scale", type=float, default=0.35)

    # body override for personalization (optional)
    p.add_argument("--mass", type=float, default=None)
    p.add_argument("--thigh", type=float, default=None)
    p.add_argument("--shank", type=float, default=None)
    p.add_argument("--foot", type=float, default=None)
    p.add_argument("--torso", type=float, default=None)
    p.add_argument("--hip-width", type=float, default=None)

    # Optional body fitting (no gradient updates / no RL retraining).
    # We simply try a few candidate bodies and select the one with lowest tracking pose error.
    p.add_argument("--body-search", type=int, default=0, help="If >0, sample N body candidates and pick the best.")
    p.add_argument("--body-search-seconds", type=float, default=2.0, help="Horizon (seconds) used for body search.")
    p.add_argument("--body-search-refstart", type=int, default=0, help="Reference start index used for body search.")

    return p.parse_args()


def main():
    args = parse_args()

    # Prepare reference npz
    ref_path = args.ref
    tmp = None
    if ref_path is None:
        if args.video is None:
            raise SystemExit("Provide either --ref or --video")
        tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        tmp.close()
        ref_path = tmp.name
        extract_reference_from_video(
            video_path=args.video,
            out_npz=ref_path,
            every=int(args.every),
            min_vis=float(args.min_vis),
            name=os.path.basename(args.video),
        )

    ref = load_reference_npz(ref_path)
    env_cfg = EnvConfig()
    cam_cfg = Camera2DConfig(view=args.cam_view, scale=float(args.cam_scale))

    env_track = TrackingBipedEnv3D(cfg=env_cfg, ref=ref, cam_cfg=cam_cfg)
    env_ideal = RunningBipedEnv3D(cfg=env_cfg)

    # Body handling
    # 1) If the user provides explicit body parameters, we use them.
    # 2) Otherwise, we optionally *fit* body parameters by a quick random search
    #    (this is not RL training; it just selects among sampled candidates).
    body_override = None
    if any(v is not None for v in [args.mass, args.thigh, args.shank, args.foot, args.torso, args.hip_width]):
        from motionlab_track_rl.config import BodyParams3D
        body_override = BodyParams3D(
            mass=float(args.mass or 70.0),
            thigh=float(args.thigh or 0.45),
            shank=float(args.shank or 0.45),
            foot=float(args.foot or 0.25),
            torso=float(args.torso or 0.55),
            hip_width=float(args.hip_width or 0.18),
        )

    # Build policies before optional body search

    obs_dim_track = env_track.OBS_DIM
    obs_dim_ideal = env_ideal.OBS_BASE_DIM
    act_dim = env_track.ADIM

    ac_track = load_ac_from_ckpt(args.ckpt_track, obs_dim_track, act_dim, device=args.device).to(args.device)
    ac_ideal = load_ac_from_ckpt(args.ckpt_ideal, obs_dim_ideal, act_dim, device=args.device).to(args.device)

    def act_track(obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        a, _, _ = ac_track.pi(o, deterministic=True)
        return a.squeeze(0).detach().cpu().numpy()

    def act_ideal(obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=args.device).unsqueeze(0)
        a, _, _ = ac_ideal.pi(o, deterministic=True)
        return a.squeeze(0).detach().cpu().numpy()

    # Optional body search
    if body_override is None and int(args.body_search) > 0:
        from motionlab_track_rl.config import BodyParams3D

        def eval_body(b: BodyParams3D) -> float:
            """Lower is better (mean pose_err)."""
            obs, _ = env_track.reset(randomize_body=False, body_override=b, ref_start=int(args.body_search_refstart))
            dt = env_track.cfg.dt
            steps = max(1, int(float(args.body_search_seconds) / dt))
            acc = 0.0
            n = 0
            for _ in range(steps):
                a = act_track(obs)
                obs, _, done, info = env_track.step(a)
                acc += float(info.get("pose_err", 0.0))
                n += 1
                if done:
                    break
            return acc / max(1, n)

        best_b = None
        best_err = 1e9
        for _ in range(int(args.body_search)):
            cand = env_track.sample_body()
            err = eval_body(cand)
            if err < best_err:
                best_err = err
                best_b = cand

        body_override = best_b
        print({"body_search": int(args.body_search), "best_pose_err": float(best_err), "best_body": body_override})

    stats = rollout_compare_to_video(
        env_track=env_track,
        policy_track=act_track,
        env_ideal=env_ideal,
        policy_ideal=act_ideal,
        out_path=args.out,
        seconds=float(args.seconds),
        fps=int(args.fps),
        deterministic=True,
        fixed_duration=True,
        cam_cfg=cam_cfg,
        title_prefix="MotionLab: Closest vs Ideal",
        body_override=body_override,
    )

    print(f"Saved: {args.out}")
    print(stats)

    if tmp is not None:
        try:
            os.unlink(ref_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
