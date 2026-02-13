from __future__ import annotations
from dataclasses import asdict
from typing import Optional, Dict, Tuple, Any
import numpy as np

from .config import EnvConfig, BodyParams3D, Camera2DConfig
from .physics3d import compute_kinematics_3d
from .camera import project_kinematics_to_2d_array
from .reference import ReferenceSequence, ref_features_at, ref_features_vel_at
from .dataset import ReferenceDataset
from .utils import clamp, safe_norm


class RunningBipedEnv3D:
    """A lightweight 3D biped environment (surrogate dynamics).

    This is NOT a full physics simulator. It is designed to:
    - Enforce joint limits and simple stability constraints.
    - Provide reasonable signals for RL development and visualization.
    """

    QDIM = 14
    ADIM = 14
    # base observation = 47 (similar to previous scaffold), without reference
    OBS_BASE_DIM = 47

    def __init__(self, cfg: Optional[EnvConfig] = None, body: Optional[BodyParams3D] = None):
        self.cfg = cfg or EnvConfig()
        self.body = body or BodyParams3D()
        self.rng = np.random.RandomState(0)

        # state
        self.t = 0.0
        self.step_count = 0
        self.pelvis = np.zeros(3, dtype=np.float32)
        self.pelvis_v = np.zeros(3, dtype=np.float32)
        self.pelvis_yaw = 0.0

        self.q = np.zeros(self.QDIM, dtype=np.float32)
        self.qd = np.zeros(self.QDIM, dtype=np.float32)

        # gait bookkeeping (very simplified)
        self.stance = 0  # 0 left, 1 right
        self.phase_t = 0.0
        self.last_impact = 0.0
        self.last_step_L = 0.35
        self.last_step_R = 0.35
        self._last_touch_t_L = -1.0
        self._last_touch_t_R = -1.0

        self._max_steps = int(self.cfg.max_episode_seconds / self.cfg.dt)

    # ----------------- body sampling -----------------
    def sample_body(self) -> BodyParams3D:
        # domain randomization range (tunable)
        mass = float(self.rng.uniform(50.0, 95.0))
        thigh = float(self.rng.uniform(0.36, 0.50))
        shank = float(self.rng.uniform(0.36, 0.50))
        foot = float(self.rng.uniform(0.20, 0.28))
        torso = float(self.rng.uniform(0.48, 0.62))
        hip_width = float(self.rng.uniform(0.15, 0.22))

        b = BodyParams3D(
            mass=mass,
            thigh=thigh,
            shank=shank,
            foot=foot,
            torso=torso,
            hip_width=hip_width,
        )

        # Adjust contact stiffness with mass (roughly)
        scale = mass / 70.0
        b.leg_spring_k = float(4200.0 * scale)
        b.leg_spring_c = float(260.0 * scale)
        b.joint_inertia = float(0.08 * scale)
        b.joint_damping = float(0.9)
        return b

    # ----------------- limits -----------------
    def _joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        lo = np.array([
            -0.6, -0.6,    # torso roll/pitch
            -1.2, -0.7, -1.2,  0.0,  -0.8, -0.8,   # left
            -1.2, -0.7, -1.2,  0.0,  -0.8, -0.8,   # right
        ], dtype=np.float32)
        hi = np.array([
            +0.6, +0.6,
            +1.2, +0.7, +1.2,  2.6,  +0.8, +0.8,
            +1.2, +0.7, +1.2,  2.6,  +0.8, +0.8,
        ], dtype=np.float32)
        return lo, hi

    # ----------------- reset/obs -----------------
    def reset(self, randomize_body: bool = True, body_override: Optional[BodyParams3D] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if body_override is not None:
            self.body = body_override
        elif randomize_body:
            self.body = self.sample_body()

        self.t = 0.0
        self.step_count = 0

        # pelvis start
        self.pelvis = np.array([0.0, 0.0, float(max(self.cfg.min_pelvis_height + 0.1, self.body.thigh + self.body.shank - 0.1))], dtype=np.float32)
        self.pelvis_v = np.zeros(3, dtype=np.float32)
        self.pelvis_yaw = 0.0

        # neutral-ish pose
        self.q[:] = 0.0
        self.q[1] = 0.12  # slight forward pitch
        self.q[5] = 0.9
        self.q[11] = 0.9
        self.qd[:] = 0.0

        self.stance = int(self.rng.randint(0, 2))
        self.phase_t = 0.0
        self.last_impact = 0.0
        self.last_step_L = 0.35
        self.last_step_R = 0.35
        self._last_touch_t_L = -1.0
        self._last_touch_t_R = -1.0

        obs = self._get_obs_base()
        info = {"t": self.t, "body": asdict(self.body)}
        return obs, info

    def _get_obs_base(self) -> np.ndarray:
        # Base observation (47)
        pelvis_z = self.pelvis[2]
        vx, vy, vz = self.pelvis_v
        torso_roll, torso_pitch = float(self.q[0]), float(self.q[1])
        torso_w_roll, torso_w_pitch = float(self.qd[0]), float(self.qd[1])

        stats = np.array([self.last_impact, self.last_step_L, self.last_step_R, self.cfg.target_speed], dtype=np.float32)
        body_norm = np.array([self.body.mass / 80.0, self.body.thigh / 0.45, self.body.shank / 0.45, self.body.foot / 0.25, self.body.torso / 0.55], dtype=np.float32)
        stance_phase = np.array([float(self.stance), float(self.phase_t)], dtype=np.float32)

        obs = np.concatenate([
            np.array([pelvis_z], dtype=np.float32),
            np.array([vx, vy, vz], dtype=np.float32),
            np.array([torso_roll, torso_pitch], dtype=np.float32),
            np.array([torso_w_roll, torso_w_pitch], dtype=np.float32),
            self.q.astype(np.float32),
            self.qd.astype(np.float32),
            stance_phase,
            stats,
            body_norm,
        ], axis=0)

        assert obs.shape[0] == self.OBS_BASE_DIM, f"obs dim mismatch: {obs.shape}"
        return obs

    # ----------------- dynamics -----------------
    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        dt = self.cfg.dt
        self.t += dt
        self.step_count += 1
        self.phase_t += dt

        # torque scaling with mass
        scale = float(self.body.mass / 70.0)
        max_tau = np.array(
            [self.body.max_tau_torso, self.body.max_tau_torso] +
            [self.body.max_tau_hip, self.body.max_tau_hip, self.body.max_tau_hip, self.body.max_tau_knee, self.body.max_tau_ankle, self.body.max_tau_ankle] +
            [self.body.max_tau_hip, self.body.max_tau_hip, self.body.max_tau_hip, self.body.max_tau_knee, self.body.max_tau_ankle, self.body.max_tau_ankle],
            dtype=np.float32
        ) * scale

        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, -1.0, 1.0)
        tau = a * max_tau

        # simple joint dynamics: qdd = (tau - d*qd)/I
        qdd = (tau - self.body.joint_damping * self.qd) / max(self.body.joint_inertia, 1e-6)
        self.qd = self.qd + qdd * dt
        self.q = self.q + self.qd * dt

        # joint limits
        lo, hi = self._joint_limits()
        joint_violation = 0.0
        for i in range(self.QDIM):
            if self.q[i] < lo[i]:
                joint_violation += float((lo[i] - self.q[i]) ** 2)
                self.q[i] = lo[i]
                self.qd[i] *= -0.2
            elif self.q[i] > hi[i]:
                joint_violation += float((self.q[i] - hi[i]) ** 2)
                self.q[i] = hi[i]
                self.qd[i] *= -0.2

        # surrogate pelvis motion:
        # forward accel from hip pitch "drive" difference and torso pitch
        drive = float(np.tanh(self.q[4]) + np.tanh(self.q[10])) * 0.8
        self.pelvis_v[0] += (drive - 0.15 * self.pelvis_v[0]) * dt  # vx
        # lateral damping
        self.pelvis_v[1] += (-self.body.lateral_damp * self.pelvis_v[1]) * dt
        # vertical spring to keep pelvis height
        rest_h = float(max(self.cfg.min_pelvis_height + 0.05, self.body.thigh + self.body.shank - 0.05))
        z_err = rest_h - float(self.pelvis[2])
        self.pelvis_v[2] += (z_err * 30.0 - 6.0 * self.pelvis_v[2]) * dt

        self.pelvis = self.pelvis + self.pelvis_v * dt

        # compute kinematics for impact and contact heuristics
        kin = compute_kinematics_3d(self.pelvis, self.q, self.body, pelvis_yaw=self.pelvis_yaw)

        # touchdown heuristic: if swing toe touches ground (z<=0) after min phase -> switch stance
        min_phase = 0.18
        impact = 0.0
        if self.phase_t >= min_phase:
            if self.stance == 0:
                swing_toe_z = float(kin.toe_R[2])
                if swing_toe_z <= 0.005:
                    # impact ~ downward pelvis velocity
                    impact = max(0.0, -float(self.pelvis_v[2]))
                    self.last_impact = impact
                    self.stance = 1
                    if self._last_touch_t_R > 0:
                        self.last_step_R = float(self.t - self._last_touch_t_R)
                    self._last_touch_t_R = float(self.t)
                    self.phase_t = 0.0
            else:
                swing_toe_z = float(kin.toe_L[2])
                if swing_toe_z <= 0.005:
                    impact = max(0.0, -float(self.pelvis_v[2]))
                    self.last_impact = impact
                    self.stance = 0
                    if self._last_touch_t_L > 0:
                        self.last_step_L = float(self.t - self._last_touch_t_L)
                    self._last_touch_t_L = float(self.t)
                    self.phase_t = 0.0

        # reward (ideal)
        vx, vy, vz = [float(x) for x in self.pelvis_v]
        speed_term = float(np.exp(-((vx - self.cfg.target_speed) ** 2) / (2 * (0.75 ** 2))))
        torso_roll, torso_pitch = float(self.q[0]), float(self.q[1])
        upright_term = torso_roll * torso_roll + (torso_pitch - 0.12) ** 2

        energy = float(np.mean((tau / (max_tau + 1e-6)) ** 2))
        impact_term = float(impact ** 2)

        desired_step_t = float(60.0 / (self.cfg.target_cadence_spm / 2.0))
        cadence_err = 0.5 * (abs(self.last_step_L - desired_step_t) + abs(self.last_step_R - desired_step_t))
        symmetry_err = abs(self.last_step_L - self.last_step_R)
        lateral_term = vy * vy + float(self.pelvis[1] ** 2)

        r = 0.0
        r += self.cfg.w_speed * speed_term
        r -= self.cfg.w_upright * upright_term
        r -= self.cfg.w_energy * energy
        r -= self.cfg.w_joint_limit * joint_violation
        r -= self.cfg.w_impact * impact_term
        r -= self.cfg.w_cadence * cadence_err
        r -= self.cfg.w_symmetry * symmetry_err
        r -= self.cfg.w_lateral * lateral_term
        r += self.cfg.alive_bonus

        # done conditions
        done = False
        if self.step_count >= self._max_steps:
            done = True
        if float(self.pelvis[2]) < self.cfg.min_pelvis_height:
            done = True
        if abs(torso_roll) > self.cfg.max_torso_angle or abs(torso_pitch) > self.cfg.max_torso_angle:
            done = True
        if not np.isfinite(self.pelvis).all() or not np.isfinite(self.q).all():
            done = True

        obs = self._get_obs_base()
        info = {
            "t": float(self.t),
            "vx": float(vx),
            "vy": float(vy),
            "vz": float(vz),
            "impact": float(impact),
            "joint_violation": float(joint_violation),
            "energy": float(energy),
            "hip_z": float(self.pelvis[2]),
            "torso_roll": float(torso_roll),
            "torso_pitch": float(torso_pitch),
        }
        return obs, float(r), bool(done), info

    def get_kinematics(self):
        return compute_kinematics_3d(self.pelvis, self.q, self.body, pelvis_yaw=self.pelvis_yaw)


class TrackingBipedEnv3D(RunningBipedEnv3D):
    """Video-conditioned tracking environment.

    Adds a reference sequence and a reprojection-based tracking reward.
    The observation is augmented with reference features.

    Reference schema: J=8 joints in 2D normalized coords.
    """

    def __init__(
        self,
        cfg: Optional[EnvConfig] = None,
        body: Optional[BodyParams3D] = None,
        ref: Optional[ReferenceSequence] = None,
        ref_dataset: Optional[ReferenceDataset] = None,
        cam_cfg: Optional[Camera2DConfig] = None,
    ):
        super().__init__(cfg=cfg, body=body)
        self.ref = ref
        self.ref_dataset = ref_dataset
        self.cam_cfg = cam_cfg or Camera2DConfig()
        self.ref_idx0 = 0
        self.ref_idx = 0
        self._ref_time = 0.0
        self._sim_rr_prev = None  # for velocity tracking

        # observation dims
        self.ref_J = 8
        self.ref_window = int(self.cfg.ref_window)
        self.OBS_DIM = self.OBS_BASE_DIM + (self.ref_J * 2 * self.ref_window) + (self.ref_J * 2 * self.ref_window)

    def reset(
        self,
        randomize_body: bool = True,
        body_override: Optional[BodyParams3D] = None,
        ref_start: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(randomize_body=randomize_body, body_override=body_override)

        # If we were constructed with a dataset, sample a new reference per episode.
        if self.ref is None:
            if self.ref_dataset is None:
                raise ValueError("TrackingBipedEnv3D requires either (ref) or (ref_dataset).")
            self.ref = self.ref_dataset.sample(self.rng)
            # Sanity: keep expected joint count
            if int(self.ref.J) != int(self.ref_J):
                raise ValueError(f"Expected J={self.ref_J} joints but got {self.ref.J} in {self.ref.name}")

        # pick a random start index so we learn across segments
        if ref_start is None:
            max0 = max(0, self.ref.T - int(self.cfg.max_episode_seconds * self.ref.fps) - 2)
            self.ref_idx0 = int(self.rng.randint(0, max0 + 1)) if max0 > 0 else 0
        else:
            self.ref_idx0 = int(max(0, min(self.ref.T - 2, ref_start)))

        self.ref_idx = self.ref_idx0
        self._ref_time = 0.0
        self._sim_rr_prev = None

        obs_aug = self._augment_obs(obs)
        info.update({"ref_name": self.ref.name, "ref_idx0": int(self.ref_idx0)})
        return obs_aug, info

    def _augment_obs(self, obs_base: np.ndarray) -> np.ndarray:
        # reference pose + velocity features (both root-relative normalized)
        f_pose, vis_pose = ref_features_at(self.ref, self.ref_idx, window=self.ref_window)
        f_vel, vis_vel = ref_features_vel_at(self.ref, self.ref_idx, window=self.ref_window, dt=self.cfg.dt)
        # We include visibility as a mask implicitly in reward (not in obs) to keep obs compact.
        obs_aug = np.concatenate([obs_base.astype(np.float32), f_pose, f_vel], axis=0)
        assert obs_aug.shape[0] == self.OBS_DIM, (obs_aug.shape, self.OBS_DIM)
        return obs_aug

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # run base dynamics + ideal reward (small; mostly stability regularizer)
        obs_base, r_ideal, done, info = super().step(a)

        # update reference frame index
        self._ref_time += self.cfg.dt
        self.ref_idx = self.ref_idx0 + int(round(self._ref_time * self.ref.fps))
        done_ref = False
        if self.ref_idx >= self.ref.T:
            self.ref_idx = self.ref.T - 1
            done_ref = True

        # tracking reward
        kin = self.get_kinematics()
        sim2d = project_kinematics_to_2d_array(kin, self.cam_cfg)  # (8,2) in [0,1] approx
        ref2d = self.ref.keypoints2d[self.ref_idx]                 # (8,2)
        vis = self.ref.vis[self.ref_idx]                           # (8,)

        # root-relative normalization (pelvis=0, torso_top=1)
        def rr(x):
            root = x[0]
            scale = np.linalg.norm(x[1] - x[0]) + 1e-6
            return (x - root[None, :]) / scale

        sim_rr = rr(sim2d)
        ref_rr = rr(ref2d)

        # pose error
        diff = sim_rr - ref_rr
        w = vis[:, None]
        pose_err = float(np.sum(w * diff * diff) / (np.sum(w) + 1e-6))

        # velocity matching (finite difference in ref; sim uses qd proxy via projected keypoints delta)
        # approx sim vel: use pelvis_v and joint velocities are not used; keep it light.
        ref_prev = self.ref.keypoints2d[max(0, self.ref_idx - 1)]
        ref_prev_rr = rr(ref_prev)
        ref_vel = (ref_rr - ref_prev_rr) / max(self.cfg.dt, 1e-6)

        # sim prev from one-step ago is not stored; approximate by zero (regularizer only)
        sim_vel = np.zeros_like(ref_vel)
        vel_err = float(np.sum(w * (sim_vel - ref_vel) ** 2) / (np.sum(w) + 1e-6))

        # foot penetration penalty (encourage toes not below ground)
        toe_pen = 0.0
        toe_pen += float(max(0.0, -kin.toe_L[2]) ** 2)
        toe_pen += float(max(0.0, -kin.toe_R[2]) ** 2)

        # smoothness penalty (jerk proxy): action magnitude
        smooth = float(np.mean(np.asarray(a, dtype=np.float32) ** 2))

        r_track = 0.0
        r_track -= self.cfg.w_track_pose * pose_err
        r_track -= self.cfg.w_track_vel * vel_err
        r_track -= self.cfg.w_foot_penetration * toe_pen
        r_track -= self.cfg.w_smooth * smooth

        # combine: keep a bit of r_ideal to maintain physical regularization
        r = 0.2 * float(r_ideal) + float(r_track)

        # termination if reference ended
        done = bool(done or done_ref)

        obs_aug = self._augment_obs(obs_base)

        info.update({
            "ref_idx": int(self.ref_idx),
            "pose_err": float(pose_err),
            "vel_err": float(vel_err),
            "toe_pen": float(toe_pen),
        })
        return obs_aug, float(r), bool(done), info