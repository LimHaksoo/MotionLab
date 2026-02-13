from __future__ import annotations
import os
import time
import json
from dataclasses import asdict
from typing import Dict, Any, Optional

import numpy as np
import torch

from .buffers import RolloutBuffer
from .ppo import PPOTrainer


def collect_rollout(env, ac, rollout_steps: int, deterministic: bool = False, device: str = "cpu") -> RolloutBuffer:
    obs_list = []
    act_list = []
    rew_list = []
    done_list = []
    val_list = []
    logp_list = []

    obs, info = env.reset()
    for t in range(rollout_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a_t, logp_t, v_t = ac.step(obs_t, deterministic=deterministic)
        a = a_t.squeeze(0).cpu().numpy()
        logp = float(logp_t.squeeze(0).cpu().numpy())
        v = float(v_t.squeeze(0).cpu().numpy())

        next_obs, r, done, step_info = env.step(a)

        obs_list.append(obs.astype(np.float32))
        act_list.append(a.astype(np.float32))
        rew_list.append(float(r))
        done_list.append(float(done))
        val_list.append(float(v))
        logp_list.append(float(logp))

        obs = next_obs
        if done:
            obs, info = env.reset()

    buf = RolloutBuffer(
        obs=np.asarray(obs_list, dtype=np.float32),
        act=np.asarray(act_list, dtype=np.float32),
        rew=np.asarray(rew_list, dtype=np.float32),
        done=np.asarray(done_list, dtype=np.float32),
        val=np.asarray(val_list, dtype=np.float32),
        logp=np.asarray(logp_list, dtype=np.float32),
    )
    return buf


def save_checkpoint(path: str, ac, trainer_cfg: Dict[str, Any], model_cfg: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": ac.state_dict(),
        "trainer_cfg": trainer_cfg,
        "model_cfg": model_cfg,
    }
    torch.save(payload, path)


def train(
    env,
    ac,
    trainer: PPOTrainer,
    total_steps: int,
    rollout_steps: int,
    out_dir: str,
    log_every: int = 1,
    deterministic_rollout: bool = False,
    device: str = "cpu",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    steps_done = 0
    update = 0
    t0 = time.time()

    while steps_done < total_steps:
        buf = collect_rollout(env, ac, rollout_steps=rollout_steps, deterministic=deterministic_rollout, device=device)
        steps_done += int(buf.obs.shape[0])
        update += 1

        stats = trainer.update(buf)

        if update % log_every == 0:
            elapsed = time.time() - t0
            log = {
                "update": update,
                "steps_done": steps_done,
                "elapsed_sec": elapsed,
                **stats,
            }
            print(json.dumps(log, ensure_ascii=False))

        # checkpoint
        if update % 10 == 0 or steps_done >= total_steps:
            ckpt_path = os.path.join(ckpt_dir, "ckpt_latest.pt")
            save_checkpoint(ckpt_path, ac, trainer_cfg=asdict(trainer.cfg), model_cfg=ac.export_config())

