# MotionLab RL 3D Pretrained Tracking (vNext)

This repository is a **scaffold** for the workflow you described:

1) (Offline) Prepare an **athlete teacher dataset**: athlete videos -> `dataset/*.npz` references.
2) (Offline) Train a **pretrained tracking policy** on the athlete dataset.
3) (Offline) Train an **ideal policy** that optimizes stability/efficiency under physics constraints.
4) (Online / Inference) User uploads a running video.
5) We extract 2D pose keypoints -> a temporary `reference.npz`.
6) The pretrained **tracking policy** outputs the closest physically-plausible 3D motion.
7) The **ideal policy** outputs the ideal motion for the *same body parameters*.
8) `inference.py` renders a side-by-side MP4: **Closest vs Ideal**, plus a 2D overlay.

> Important: This is not MuJoCo/OpenSim-level physics.
> It's designed to be self-contained, fast to iterate, and to demonstrate the *structure*
> of tracking-RL + physics constraints.

## Install
```bash
pip install -r requirements.txt
# if you want video->reference extraction:
pip install opencv-python mediapipe
```

## 1) Prepare athlete dataset (teacher videos -> references)

```bash
python -m motionlab_track_rl.teacher_video extract-dir \
  --videos-dir athlete_videos/ --out-dir dataset/ --pattern "*.mp4"
```

Optional sanity check (overlay keypoints on the original video):

```bash
python -m motionlab_track_rl.teacher_video preview \
  --video athlete_videos/sample.mp4 --ref dataset/sample.npz --out sample_preview.mp4
```

## 2) Train pretrained tracking policy (closest motion)

```bash
python train.py --task track --dataset dataset --out runs/track_pretrained \
  --arch mog --hidden 1024,1024,512 --mog-components 4 --layernorm --residual-blocks 2
```

## 3) Train ideal policy
```bash
python train.py --task ideal --out runs/ideal \
  --arch mog --hidden 1024,1024,512 --mog-components 4 --layernorm --residual-blocks 2
```

## 4) Inference: user video -> "Closest vs Ideal"
```bash
python inference.py \
  --ckpt-track runs/track_pretrained/checkpoints/ckpt_latest.pt \
  --ckpt-ideal runs/ideal/checkpoints/ckpt_latest.pt \
  --video user.mp4 --out closest_vs_ideal.mp4 --seconds 10
```

Optional: if you want the system to *guess* body parameters (mass/segment lengths) from the video,
enable a quick random search (still no RL training):

```bash
python inference.py \
  --ckpt-track runs/track_pretrained/checkpoints/ckpt_latest.pt \
  --ckpt-ideal runs/ideal/checkpoints/ckpt_latest.pt \
  --video user.mp4 --out closest_vs_ideal.mp4 --seconds 10 \
  --body-search 24 --body-search-seconds 2.0
```

## Teacher3D output debugging

If you're using the earlier `teacher3d.py` scaffold (which outputs `obs/act`), you can preview the
extracted motion as a skeleton:

```bash
python verify_teacher3d_npz.py --npz teacher3d_output.npz --out teacher3d_preview.mp4
```

## Notes / Next steps (recommended for production)

- Replace surrogate env with a real physics engine (MuJoCo/IsaacGym/OpenSim).
- Add camera estimation (intrinsics/extrinsics) and multi-view support.
- Add temporal encoder (GRU/Transformer) instead of simple frame stacking.
- Use **tracking reward** on end-effectors + joint angles + velocities, and optionally contact events.

