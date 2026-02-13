from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Optional imports
try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None


JOINT_NAMES = ["pelvis", "torso_top", "knee_L", "knee_R", "ankle_L", "ankle_R", "toe_L", "toe_R"]
J = len(JOINT_NAMES)

COLOR_CENTER = (0, 255, 0)    # green
COLOR_LEFT   = (255, 0, 0)    # blue
COLOR_RIGHT  = (0, 0, 255)    # red

LEFT_IDXS  = {2, 4, 6}        # knee_L, ankle_L, toe_L
RIGHT_IDXS = {3, 5, 7}        # knee_R, ankle_R, toe_R
CENTER_IDXS = {0, 1}          # pelvis, torso_top


def _require(name: str, pkg) -> None:
    if pkg is None:
        raise RuntimeError(
            f"{name} is required for teacher extraction but is not installed.\n"
            f"Install with: pip install {name}\n"
        )


def extract_reference_from_video(
    video_path: str,
    out_npz: str,
    every: int = 1,
    max_frames: Optional[int] = None,
    min_vis: float = 0.2,
    name: str = "user_video",
) -> str:
    """Extract 2D keypoints from a video into ReferenceSequence npz.

    Uses MediaPipe Pose 2D landmarks (normalized to [0,1] in image coords).

    Output:
      keypoints2d: (T,8,2)
      vis: (T,8)
      fps, width, height, name
    """
    _require("mediapipe", mp)
    _require("opencv-python", cv2)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    kps_list = []
    vis_list = []
    frame_idx = 0
    kept = 0

    def get_lm(lms, idx):
        lm = lms[idx]
        return float(lm.x), float(lm.y), float(lm.visibility)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if (frame_idx - 1) % every != 0:
            continue
        if max_frames is not None and kept >= max_frames:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            # fill with NaN and zero vis
            kps_list.append(np.full((J, 2), 0.5, dtype=np.float32))
            vis_list.append(np.zeros((J,), dtype=np.float32))
            kept += 1
            continue

        lms = res.pose_landmarks.landmark

        # MediaPipe indices:
        # shoulders 11,12; hips 23,24; knees 25,26; ankles 27,28; foot_index 31,32
        xLh, yLh, vLh = get_lm(lms, 23)
        xRh, yRh, vRh = get_lm(lms, 24)
        pelvis = ((xLh + xRh) / 2.0, (yLh + yRh) / 2.0)
        pelvis_v = (vLh + vRh) / 2.0

        xLs, yLs, vLs = get_lm(lms, 11)
        xRs, yRs, vRs = get_lm(lms, 12)
        torso = ((xLs + xRs) / 2.0, (yLs + yRs) / 2.0)
        torso_v = (vLs + vRs) / 2.0

        xLk, yLk, vLk = get_lm(lms, 25)
        xRk, yRk, vRk = get_lm(lms, 26)
        xLa, yLa, vLa = get_lm(lms, 27)
        xRa, yRa, vRa = get_lm(lms, 28)
        xLt, yLt, vLt = get_lm(lms, 31)
        xRt, yRt, vRt = get_lm(lms, 32)

        kps = np.array([
            [pelvis[0], pelvis[1]],
            [torso[0], torso[1]],
            [xLk, yLk],
            [xRk, yRk],
            [xLa, yLa],
            [xRa, yRa],
            [xLt, yLt],
            [xRt, yRt],
        ], dtype=np.float32)

        vis = np.array([pelvis_v, torso_v, vLk, vRk, vLa, vRa, vLt, vRt], dtype=np.float32)
        vis = np.where(vis >= min_vis, vis, 0.0)

        kps_list.append(kps)
        vis_list.append(vis)
        kept += 1

    cap.release()
    pose.close()

    keypoints2d = np.stack(kps_list, axis=0) if len(kps_list) > 0 else np.zeros((0, J, 2), dtype=np.float32)
    vis = np.stack(vis_list, axis=0) if len(vis_list) > 0 else np.zeros((0, J), dtype=np.float32)

    np.savez_compressed(
        out_npz,
        keypoints2d=keypoints2d,
        vis=vis,
        fps=float(fps if fps and fps > 1e-3 else 30.0),
        width=int(width),
        height=int(height),
        name=str(name),
        joint_names=np.array(JOINT_NAMES),
    )
    return out_npz


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract MotionLab tracking reference (2D keypoints) from a video.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract")
    ex.add_argument("--video", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--every", type=int, default=1, help="sample every N frames")
    ex.add_argument("--max-frames", type=int, default=None)
    ex.add_argument("--min-vis", type=float, default=0.2)
    ex.add_argument("--name", type=str, default="user_video")

    exd = sub.add_parser("extract-dir", help="Extract references for all videos in a directory (athlete dataset prep).")
    exd.add_argument("--videos-dir", required=True)
    exd.add_argument("--out-dir", required=True)
    exd.add_argument("--pattern", type=str, default="*.mp4")
    exd.add_argument("--every", type=int, default=1)
    exd.add_argument("--max-frames", type=int, default=None)
    exd.add_argument("--min-vis", type=float, default=0.2)

    prev = sub.add_parser("preview", help="Create a debug MP4 overlay of extracted 2D keypoints on the original video.")
    prev.add_argument("--video", required=True)
    prev.add_argument("--ref", required=True, help="reference .npz generated by extract")
    prev.add_argument("--out", required=True, help="output mp4 path")
    prev.add_argument("--every", type=int, default=1, help="render every N frames")
    prev.add_argument("--radius", type=int, default=9)
    return p


def main():
    args = _build_cli().parse_args()
    if args.cmd == "extract":
        extract_reference_from_video(
            video_path=args.video,
            out_npz=args.out,
            every=args.every,
            max_frames=args.max_frames,
            min_vis=args.min_vis,
            name=args.name,
        )
        print(f"Saved reference npz to: {args.out}")

    elif args.cmd == "extract-dir":
        from pathlib import Path

        vids = sorted(Path(args.videos_dir).rglob(args.pattern))
        if len(vids) == 0:
            raise SystemExit(f"No videos found under {args.videos_dir} pattern={args.pattern}")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for vp in vids:
            out_npz = out_dir / (vp.stem + ".npz")
            extract_reference_from_video(
                video_path=str(vp),
                out_npz=str(out_npz),
                every=int(args.every),
                max_frames=args.max_frames,
                min_vis=float(args.min_vis),
                name=str(vp.name),
            )
            print(f"{vp.name} -> {out_npz.name}")

        print(f"Dataset prepared: {len(vids)} refs saved to {out_dir}")

    elif args.cmd == "preview":
        _require("opencv-python", cv2)

        d = np.load(args.ref, allow_pickle=True)
        kps = d["keypoints2d"].astype(np.float32)
        vis = d["vis"].astype(np.float32)

        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = cv2.VideoWriter(args.out, fourcc, fps if fps and fps > 1e-3 else 30.0, (width, height))

        t = 0
        fidx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx % int(args.every) != 0:
                fidx += 1
                continue
            if t >= kps.shape[0]:
                break

            pts = kps[t]
            vv = vis[t]
            for j in range(pts.shape[0]):
                if vv[j] <= 0:
                    continue
                x = int(pts[j, 0] * width)
                y = int(pts[j, 1] * height)

                if j in LEFT_IDXS:
                    color = COLOR_LEFT
                elif j in RIGHT_IDXS:
                    color = COLOR_RIGHT
                else:
                    color = COLOR_CENTER

                # 점 크게 + 테두리까지(가시성↑)
                r = int(args.radius)
                cv2.circle(frame, (x, y), r, color, -1)                    # filled
                cv2.circle(frame, (x, y), max(1, r // 2), (255,255,255), 1) # thin outline
            w.write(frame)
            t += 1
            fidx += 1

        w.release()
        cap.release()
        print(f"Saved preview video: {args.out}")


if __name__ == "__main__":
    main()
