from __future__ import annotations
import argparse
import os
import json
from dataclasses import asdict

import torch

from motionlab_track_rl.config import EnvConfig, PPOConfig, ModelConfig, Camera2DConfig
from motionlab_track_rl.envs import RunningBipedEnv3D, TrackingBipedEnv3D
from motionlab_track_rl.reference import load_reference_npz
from motionlab_track_rl.dataset import ReferenceDataset
from motionlab_track_rl.model import ActorCritic
from motionlab_track_rl.ppo import PPOTrainer
from motionlab_track_rl.train_loop import train
from motionlab_track_rl.utils import seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="MotionLab 3D RL training (ideal or video-tracking).")
    p.add_argument("--out", type=str, default="runs/track_vnext")
    p.add_argument("--task", type=str, choices=["ideal", "track"], default="track")

    # Tracking references
    # IMPORTANT: For the *pretrained* workflow, training should NOT depend on the end-user's video.
    # Instead, we train a generic tracking policy on an athlete dataset (teacher videos -> ref npz).
    p.add_argument("--dataset", type=str, default=None, help="Directory of reference .npz files for task=track.")
    p.add_argument("--dataset-pattern", type=str, default="*.npz")
    p.add_argument("--ref", type=str, default=None, help="(Debug) single reference npz. If set, overrides --dataset.")
    p.add_argument("--cam-view", type=str, default="side", choices=["side", "front", "45"])
    p.add_argument("--cam-scale", type=float, default=0.35)
    p.add_argument("--ref-window", type=int, default=1, help="stack N ref frames into observation (temporal context).")

    # Training
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--rollout", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")

    # PPO
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--pi-iters", type=int, default=80)
    p.add_argument("--vf-iters", type=int, default=80)
    p.add_argument("--minibatch", type=int, default=1024)
    p.add_argument("--target-kl", type=float, default=0.015)

    # Model capacity
    p.add_argument("--arch", type=str, default="mog", choices=["gauss", "mog"])
    p.add_argument("--hidden", type=str, default="1024,1024,512")
    p.add_argument("--mog-components", type=int, default=4)
    p.add_argument("--layernorm", action="store_true")
    p.add_argument("--no-layernorm", action="store_true")
    p.add_argument("--residual-blocks", type=int, default=2)
    p.add_argument("--activation", type=str, default="silu", choices=["silu", "relu", "tanh", "gelu"])
    
    p.add_argument("--roll-coef", type=float, default=0.0)

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    env_cfg = EnvConfig()
    env_cfg.ref_window = int(args.ref_window)

    if args.task == "track":
        cam_cfg = Camera2DConfig(view=args.cam_view, scale=float(args.cam_scale))

        if args.ref is not None:
            # single-reference training (useful for debugging)
            ref = load_reference_npz(args.ref)
            env = TrackingBipedEnv3D(cfg=env_cfg, ref=ref, cam_cfg=cam_cfg)
        else:
            if args.dataset is None:
                raise SystemExit("For task=track provide either --dataset (recommended) or --ref (debug).")
            ds = ReferenceDataset.from_dir(args.dataset, pattern=args.dataset_pattern, cache=True)
            env = TrackingBipedEnv3D(cfg=env_cfg, ref_dataset=ds, cam_cfg=cam_cfg)
        obs_dim = env.OBS_DIM
        act_dim = env.ADIM
    else:
        env = RunningBipedEnv3D(cfg=env_cfg)
        obs_dim = env.OBS_BASE_DIM
        act_dim = env.ADIM

    hidden_sizes = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    layernorm = True if args.layernorm else False
    if args.no_layernorm:
        layernorm = False

    ac = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        arch=args.arch,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        layernorm=layernorm,
        residual_blocks=int(args.residual_blocks),
        mog_components=int(args.mog_components),
    )

    ppo_cfg = PPOConfig(
        gamma=float(args.gamma),
        lam=float(args.lam),
        clip_ratio=float(args.clip),
        pi_lr=float(args.pi_lr),
        vf_lr=float(args.vf_lr),
        train_pi_iters=int(args.pi_iters),
        train_v_iters=int(args.vf_iters),
        minibatch_size=int(args.minibatch),
        target_kl=float(args.target_kl),
        roll_coef=float(args.roll_coef),
    )
    trainer = PPOTrainer(ac, cfg=ppo_cfg, device=args.device)

    # write config
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "task": args.task,
            "env_cfg": asdict(env_cfg),
            "ppo_cfg": asdict(ppo_cfg),
            "model": {
                "arch": args.arch,
                "hidden_sizes": hidden_sizes,
                "layernorm": layernorm,
                "residual_blocks": int(args.residual_blocks),
                "mog_components": int(args.mog_components),
                "activation": args.activation,
            },
            "dataset": args.dataset,
            "dataset_pattern": args.dataset_pattern,
            "ref": args.ref,
            "cam": {"view": args.cam_view, "scale": float(args.cam_scale)},
        }, f, ensure_ascii=False, indent=2)

    train(
        env=env,
        ac=ac,
        trainer=trainer,
        total_steps=int(args.steps),
        rollout_steps=int(args.rollout),
        out_dir=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
