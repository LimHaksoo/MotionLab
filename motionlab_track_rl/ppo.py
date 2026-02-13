from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Any, Tuple, Optional, Iterable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

from .buffers import compute_gae, normalize, RolloutBuffer
from .config import PPOConfig

def rolling_aux_loss_from_obs(
    obs: torch.Tensor,
    q_offset: int = 8,
    qd_offset: int = 22,
    stance_index: int = 36,
    phase_index: int = 37,
    ankle_roll_L: int = 6,
    ankle_roll_R: int = 12,
    amp: float = 0.18,
    smooth_w: float = 0.05,
) -> torch.Tensor:
    """
    stance 발 ankle_roll은 stance phase 동안 +amp -> -amp로 '롤링'하도록 유도
    swing 발 ankle_roll은 0 근처로 두어 과도 roll 방지
    phase_t는 env에서 seconds로 누적되므로 0~1 정규화가 필요
    """
    stance = obs[:, stance_index]      # 0 left, 1 right (envs.py 그대로) :contentReference[oaicite:5]{index=5}
    phase_t = obs[:, phase_index]      # seconds 누적(phase_t += dt) :contentReference[oaicite:6]{index=6}

    # phase 정규화: env에서 min_phase=0.18이고 step_time 기본 0.35 근처
    # 대략 0.45s를 stance 한 사이클로 보고 0~1로 클램프
    phase = torch.clamp(phase_t / 0.45, 0.0, 1.0)

    # 목표 롤링 궤적: +amp -> -amp (선형)
    target = amp * (1.0 - 2.0 * phase)

    qL = obs[:, q_offset + ankle_roll_L]
    qR = obs[:, q_offset + ankle_roll_R]
    qdL = obs[:, qd_offset + ankle_roll_L]
    qdR = obs[:, qd_offset + ankle_roll_R]

    is_left = (stance < 0.5).float()
    is_right = 1.0 - is_left

    loss_stance = is_left * (qL - target).pow(2) + is_right * (qR - target).pow(2)
    loss_swing  = is_left * (qR - 0.0).pow(2)   + is_right * (qL - 0.0).pow(2)

    loss = loss_stance.mean() + 0.25 * loss_swing.mean()
    loss = loss + smooth_w * (qdL.pow(2) + qdR.pow(2)).mean()
    return loss

class PPOTrainer:
    def __init__(self, ac: nn.Module, cfg: PPOConfig, device: str = "cpu"):
        self.ac = ac.to(device)
        self.cfg = cfg
        self.device = device
        self.update_count = 0

        # Separate optimizers (important for stability)
        pi_params = [p for n, p in self.ac.named_parameters() if not n.startswith("v_head") and "backbone_v" not in n and n.startswith("backbone_pi") or n.startswith("mu") or n.startswith("logits") or "dist" in n]
        # To be safe, just use all params for pi and v separated by module references if present
        try:
            pi_params = list(self.ac.backbone_pi.parameters()) + list(getattr(self.ac, "mu").parameters())
            if hasattr(self.ac, "logits"):
                pi_params += list(getattr(self.ac, "logits").parameters())
            pi_params += list(getattr(self.ac, "dist").parameters())
        except Exception:
            pi_params = list(self.ac.parameters())

        vf_params = list(self.ac.backbone_v.parameters()) + list(self.ac.v_head.parameters())

        self.pi_opt = optim.Adam(pi_params, lr=cfg.pi_lr)
        self.vf_opt = optim.Adam(vf_params, lr=cfg.vf_lr)

    def update(self, buf: RolloutBuffer) -> Dict[str, float]:
        # compute advantage/returns
        adv, ret = compute_gae(buf.rew, buf.val, buf.done, self.cfg.gamma, self.cfg.lam)
        adv = normalize(adv)

        obs = torch.as_tensor(buf.obs, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(buf.act, dtype=torch.float32, device=self.device)
        logp_old = torch.as_tensor(buf.logp, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(ret, dtype=torch.float32, device=self.device)

        # minibatch indices
        N = obs.shape[0]
        mb = min(self.cfg.minibatch_size, N)
        idx = np.arange(N)

        pi_l_old = 0.0
        v_l_old = 0.0

        # policy updates
        for i in range(self.cfg.train_pi_iters):
            np.random.shuffle(idx)
            approx_kl = 0.0
            ent = 0.0
            pi_loss_acc = 0.0
            n_batches = 0
            for start in range(0, N, mb):
                j = idx[start:start+mb]
                n_batches += 1
                logp = self.ac.log_prob(obs[j], act[j])
                ratio = torch.exp(logp - logp_old[j])
                clip_adv = torch.clamp(ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * adv_t[j]
                loss_pi = -(torch.min(ratio * adv_t[j], clip_adv)).mean()

                # entropy regularization (optional)
                # We approximate entropy by sampling from policy at current obs.
                with torch.no_grad():
                    _, _, entropy = self.ac.pi(obs[j], deterministic=False)
                loss_ent = -entropy.mean()

                self.pi_opt.zero_grad()
                loss = loss_pi + self.cfg.entropy_coef * loss_ent
                
                if getattr(self.cfg, "roll_coef", 0.0) > 0.0:
                    warm = int(getattr(self.cfg, "roll_warmup_updates", 0))
                    if self.update_count < warm:
                        roll_loss = rolling_aux_loss_from_obs(
                            obs=obs[j],                 # <- 너 ppo.py 미니배치 텐서 이름 그대로
                            amp=float(self.cfg.roll_amp),
                            smooth_w=float(self.cfg.roll_smooth_w),
                        )
                        loss = loss + float(self.cfg.roll_coef) * roll_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.pi_opt.step()

                pi_loss_acc += float(loss_pi.detach().cpu().item())
                ent += float(entropy.detach().cpu().mean().item())
                approx_kl += float((logp_old[j] - logp).detach().cpu().mean().item())

            approx_kl /= max(1, n_batches)
            if approx_kl > 1.5 * self.cfg.target_kl:
                break

            pi_l_old = pi_loss_acc / max(1, n_batches)

        # value updates
        for i in range(self.cfg.train_v_iters):
            np.random.shuffle(idx)
            v_loss_acc = 0.0
            n_batches = 0
            for start in range(0, N, mb):
                j = idx[start:start+mb]
                n_batches += 1
                v = self.ac.v(obs[j])
                loss_v = ((v - ret_t[j]) ** 2).mean()

                self.vf_opt.zero_grad()
                loss = self.cfg.vf_coef * loss_v
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.vf_opt.step()

                v_loss_acc += float(loss_v.detach().cpu().item())

            v_l_old = v_loss_acc / max(1, n_batches)
        self.update_count += 1
        return {
            "pi_loss": float(pi_l_old),
            "v_loss": float(v_l_old),
            "entropy": float(ent / max(1, n_batches)),
            "N": float(N),
        }
