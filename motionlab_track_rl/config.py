from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional


@dataclass
class EnvConfig:
    dt: float = 1.0 / 120.0
    max_episode_seconds: float = 4.0

    # Termination (safety)
    min_pelvis_height: float = 0.55
    max_torso_angle: float = 0.9  # rad

    # Ideal task targets (surrogate)
    target_speed: float = 3.0
    target_cadence_spm: float = 170.0

    # Reward weights (ideal)
    w_speed: float = 1.2
    w_upright: float = 0.15
    w_energy: float = 0.08
    w_joint_limit: float = 0.5
    w_impact: float = 0.25
    w_cadence: float = 0.2
    w_symmetry: float = 0.15
    w_lateral: float = 0.05
    alive_bonus: float = 0.05

    # Tracking task weights
    w_track_pose: float = 2.0
    w_track_vel: float = 0.5
    w_foot_penetration: float = 0.3
    w_smooth: float = 0.05

    # Reference handling
    ref_window: int = 1  # how many frames stacked (1 = current frame only)


@dataclass
class BodyParams3D:
    # NOTE: lengths are in meters.
    mass: float = 70.0
    thigh: float = 0.45
    shank: float = 0.45
    foot: float = 0.25
    torso: float = 0.55
    hip_width: float = 0.18

    # Simple joint dynamics parameters
    joint_inertia: float = 0.08
    joint_damping: float = 0.9

    # Torque limits (base @ 70kg); scaled by mass/70 inside env
    max_tau_torso: float = 60.0
    max_tau_hip: float = 110.0
    max_tau_knee: float = 95.0
    max_tau_ankle: float = 70.0

    # Surrogate contact parameters (ideal env)
    leg_spring_k: float = 4200.0
    leg_spring_c: float = 260.0

    # Surrogate friction
    lateral_damp: float = 2.0


@dataclass
class Camera2DConfig:
    """A minimal projection model for 3D->2D tracking reward.

    We intentionally keep this simple. In production you would estimate camera intrinsics/extrinsics
    or use multiple views. Here we project to normalized [0,1] coordinates.
    """
    view: str = "side"   # "side" (x-z), "front"(y-z), "45"(x+y - z)
    scale: float = 0.35  # affects only absolute coordinates; tracking uses root-relative normalization


@dataclass
class PPOConfig:
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    pi_lr: float = 3e-4
    vf_lr: float = 1e-3
    train_pi_iters: int = 80
    train_v_iters: int = 80
    target_kl: float = 0.015
    entropy_coef: float = 0.0
    vf_coef: float = 0.5

    # Minibatch SGD over one rollout buffer
    minibatch_size: int = 1024
    
    # rolling auxiliary loss
    roll_coef: float = 0.0          # 0이면 비활성
    roll_warmup_updates: int = 200
    roll_amp: float = 0.18          # rad, 목표 roll 진폭
    roll_smooth_w: float = 0.05     # roll-rate 억제 가중

    # obs indexing (환경 obs layout에 맞춰 설정)
    obs_q_offset: int = 0
    obs_qd_offset: int = 0
    obs_stance_index: int = 0
    obs_phase_index: int = 0

    # ankle roll indices inside q/qd (q 기준 14DoF ordering 가정)
    ankle_roll_L_q: int = 6
    ankle_roll_R_q: int = 12
    ankle_roll_L_qd: int = 6
    ankle_roll_R_qd: int = 12


@dataclass
class ModelConfig:
    arch: str = "mog"  # "gauss" or "mog"
    hidden_sizes: Tuple[int, ...] = (1024, 1024, 512)
    layernorm: bool = True
    residual_blocks: int = 2
    mog_components: int = 4
    activation: str = "silu"
