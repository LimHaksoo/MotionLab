"""MotionLab Tracking RL (3D) scaffold.

This package provides:
- A lightweight 3D kinematic biped model with joint limits.
- Two RL tasks:
  (1) 'ideal' locomotion-ish task (surrogate dynamics)
  (2) 'track' video-conditioned tracking task using 2D keypoints reprojection loss.
- PPO trainer with minibatches.
- Optional Mixture-of-Squashed-Gaussians policy for multimodal control.

Coordinate convention:
  x: forward
  y: left
  z: up
"""
