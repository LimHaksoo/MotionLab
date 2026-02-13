from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .config import BodyParams3D


def rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)


def rot_z(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)


@dataclass
class Kinematics3D:
    pelvis: np.ndarray
    head: np.ndarray
    hip_L: np.ndarray
    knee_L: np.ndarray
    ankle_L: np.ndarray
    toe_L: np.ndarray
    heel_L: np.ndarray
    hip_R: np.ndarray
    knee_R: np.ndarray
    ankle_R: np.ndarray
    toe_R: np.ndarray
    heel_R: np.ndarray


def compute_kinematics_3d(
    pelvis_pos: np.ndarray,     # (3,)
    q: np.ndarray,              # (14,)
    body: BodyParams3D,
    pelvis_yaw: float = 0.0,
) -> Kinematics3D:
    """Forward kinematics for a minimal 3D biped skeleton.

    Joint DOF ordering:
      0 torso_roll
      1 torso_pitch
      left: 2 hip_yaw,3 hip_roll,4 hip_pitch,5 knee_pitch,6 ankle_roll,7 ankle_pitch
      right: 8 hip_yaw,9 hip_roll,10 hip_pitch,11 knee_pitch,12 ankle_roll,13 ankle_pitch
    """
    pelvis = pelvis_pos.astype(np.float32)
    torso_roll, torso_pitch = float(q[0]), float(q[1])

    R_pelvis = rot_z(pelvis_yaw)  # yaw about z
    # torso points in +z with roll/pitch applied around pelvis frame
    R_torso = R_pelvis @ rot_x(torso_roll) @ rot_y(torso_pitch)
    head = pelvis + (R_torso @ np.array([0, 0, body.torso], dtype=np.float32))

    # Hip anchors
    hip_L = pelvis + (R_pelvis @ np.array([0, +body.hip_width / 2.0, 0], dtype=np.float32))
    hip_R = pelvis + (R_pelvis @ np.array([0, -body.hip_width / 2.0, 0], dtype=np.float32))

    def leg_chain(hip: np.ndarray, idx0: int) -> Dict[str, np.ndarray]:
        hy, hr, hp = float(q[idx0 + 0]), float(q[idx0 + 1]), float(q[idx0 + 2])
        kp = float(q[idx0 + 3])
        ar, ap = float(q[idx0 + 4]), float(q[idx0 + 5])

        R_hip = R_pelvis @ rot_z(hy) @ rot_x(hr) @ rot_y(hp)
        knee = hip + (R_hip @ np.array([0, 0, -body.thigh], dtype=np.float32))

        R_knee = R_hip @ rot_y(kp)
        ankle = knee + (R_knee @ np.array([0, 0, -body.shank], dtype=np.float32))

        R_ankle = R_knee @ rot_x(ar) @ rot_y(ap)
        toe = ankle + (R_ankle @ np.array([body.foot, 0, 0], dtype=np.float32))

        # crude heel approximation: go backwards along foot axis
        heel = ankle - (R_ankle @ np.array([body.foot * 0.45, 0, 0], dtype=np.float32))
        return {"knee": knee, "ankle": ankle, "toe": toe, "heel": heel}

    L = leg_chain(hip_L, 2)
    R = leg_chain(hip_R, 8)

    return Kinematics3D(
        pelvis=pelvis,
        head=head,
        hip_L=hip_L,
        knee_L=L["knee"],
        ankle_L=L["ankle"],
        toe_L=L["toe"],
        heel_L=L["heel"],
        hip_R=hip_R,
        knee_R=R["knee"],
        ankle_R=R["ankle"],
        toe_R=R["toe"],
        heel_R=R["heel"],
    )
