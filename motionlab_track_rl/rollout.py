from __future__ import annotations
from typing import Callable, Dict, Any, Optional
import numpy as np


def run_episode(env, policy_act: Callable[[np.ndarray], np.ndarray], max_steps: int, deterministic: bool = True):
    obs, info = env.reset()
    ep_rew = 0.0
    last_info = info
    for t in range(max_steps):
        a = policy_act(obs)
        obs, r, done, step_info = env.step(a)
        ep_rew += float(r)
        last_info = step_info
        if done:
            break
    return ep_rew, t + 1, last_info
