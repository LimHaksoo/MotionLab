from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import torch


@dataclass
class RolloutBuffer:
    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    done: np.ndarray
    val: np.ndarray
    logp: np.ndarray

    adv: Optional[np.ndarray] = None
    ret: Optional[np.ndarray] = None


def compute_gae(
    rew: np.ndarray,
    val: np.ndarray,
    done: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """GAE-Lambda advantage and returns."""
    T = rew.shape[0]
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(done[t])
        nextval = float(val[t + 1]) if t + 1 < T else 0.0
        delta = float(rew[t]) + gamma * nextval * nonterminal - float(val[t])
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + val
    return adv.astype(np.float32), ret.astype(np.float32)


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)
