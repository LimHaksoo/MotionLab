from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def atanh(x: torch.Tensor) -> torch.Tensor:
    # stable inverse tanh
    eps = 1e-6
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, activation: str = "silu", layernorm: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc1 = nn.Linear(dim, dim)
        self.act = get_activation(activation)
        self.ln2 = nn.LayerNorm(dim) if layernorm else nn.Identity()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.ln2(h)
        h = self.fc2(h)
        return x + h


class MLPBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_sizes: Tuple[int, ...] = (1024, 1024, 512),
        activation: str = "silu",
        layernorm: bool = True,
        residual_blocks: int = 2,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        self.activation = activation
        self.layernorm = layernorm
        self.residual_blocks = residual_blocks

        dims = (in_dim,) + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            if layernorm:
                self.layers.append(nn.LayerNorm(dims[i + 1]))
            self.layers.append(get_activation(activation))
            for _ in range(residual_blocks):
                self.resblocks.append(ResidualMLPBlock(dims[i + 1], activation=activation, layernorm=layernorm))

        self.out_dim = hidden_sizes[-1] if len(hidden_sizes) > 0 else in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        res_i = 0
        for m in self.layers:
            h = m(h)
            if isinstance(m, nn.modules.activation.SiLU) or isinstance(m, nn.ReLU) or isinstance(m, nn.Tanh) or isinstance(m, nn.GELU):
                # after activation, apply residual blocks for this stage
                for _ in range(self.residual_blocks):
                    h = self.resblocks[res_i](h)
                    res_i += 1
        return h


class SquashedDiagGaussian(nn.Module):
    def __init__(self, act_dim: int):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, mu: torch.Tensor, deterministic: bool = False):
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()
        a = torch.tanh(u)
        # log prob with tanh correction
        logp_u = dist.log_prob(u).sum(dim=-1)
        log_det = torch.log(1 - a * a + 1e-6).sum(dim=-1)
        logp = logp_u - log_det
        entropy = dist.entropy().sum(dim=-1)
        return a, logp, entropy

    def log_prob(self, mu: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)
        dist = torch.distributions.Normal(mu, std)
        u = atanh(a)
        logp_u = dist.log_prob(u).sum(dim=-1)
        log_det = torch.log(1 - a * a + 1e-6).sum(dim=-1)
        return logp_u - log_det


class MixtureSquashedDiagGaussian(nn.Module):
    """Mixture of squashed diagonal Gaussians (MoG) for multimodal continuous control."""

    def __init__(self, act_dim: int, K: int = 4):
        super().__init__()
        self.act_dim = act_dim
        self.K = K
        # per-component log_std
        self.log_std = nn.Parameter(torch.zeros(K, act_dim))

    def forward(self, logits: torch.Tensor, mu: torch.Tensor, deterministic: bool = False):
        # logits: (B,K), mu: (B,K,act_dim)
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)  # (K,act_dim)
        probs = torch.softmax(logits, dim=-1)           # (B,K)

        if deterministic:
            k = torch.argmax(probs, dim=-1)  # (B,)
        else:
            cat = torch.distributions.Categorical(probs=probs)
            k = cat.sample()

        # gather component params
        B = mu.shape[0]
        idx = k.view(B, 1, 1).expand(B, 1, self.act_dim)
        mu_k = torch.gather(mu, 1, idx).squeeze(1)  # (B,act_dim)
        std_k = std[k]                              # (B,act_dim)

        dist_k = torch.distributions.Normal(mu_k, std_k)
        u = mu_k if deterministic else dist_k.rsample()
        a = torch.tanh(u)

        # mixture log prob (marginal) for sampled action
        logp = self.log_prob(logits, mu, a)

        # entropy approximation (unsquashed): H(cat)+E_k[H(N_k)]
        cat_ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        comp_ent = 0.0
        for i in range(self.K):
            comp_ent = comp_ent + probs[:, i] * torch.distributions.Normal(mu[:, i], std[i]).entropy().sum(dim=-1)
        entropy = cat_ent + comp_ent
        return a, logp, entropy

    def log_prob(self, logits: torch.Tensor, mu: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # a: (B, act_dim) in [-1,1]
        std = torch.exp(self.log_std).clamp(1e-4, 2.0)  # (K,act_dim)
        probs = torch.softmax(logits, dim=-1)           # (B,K)
        log_probs = torch.log(probs + 1e-8)             # (B,K)

        u = atanh(a)  # (B,act_dim)
        # compute per-component log prob in u-space
        # (B,K,act_dim)
        u_exp = u[:, None, :].expand(mu.shape[0], self.K, self.act_dim)
        std_exp = std[None, :, :].expand(mu.shape[0], self.K, self.act_dim)
        dist = torch.distributions.Normal(mu, std_exp)
        logp_u = dist.log_prob(u_exp).sum(dim=-1)  # (B,K)

        mix_logp_u = torch.logsumexp(log_probs + logp_u, dim=-1)  # (B,)
        log_det = torch.log(1 - a * a + 1e-6).sum(dim=-1)
        return mix_logp_u - log_det


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        arch: str = "mog",
        hidden_sizes: Tuple[int, ...] = (1024, 1024, 512),
        activation: str = "silu",
        layernorm: bool = True,
        residual_blocks: int = 2,
        mog_components: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.arch = arch
        self.mog_components = mog_components
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.layernorm = layernorm
        self.residual_blocks = residual_blocks
        self.backbone_pi = MLPBackbone(
            in_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layernorm=layernorm,
            residual_blocks=residual_blocks,
        )
        self.backbone_v = MLPBackbone(
            in_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            layernorm=layernorm,
            residual_blocks=residual_blocks,
        )

        feat_dim = self.backbone_pi.out_dim

        if arch == "gauss":
            self.mu = nn.Linear(feat_dim, act_dim)
            self.dist = SquashedDiagGaussian(act_dim)
        elif arch == "mog":
            K = mog_components
            self.logits = nn.Linear(feat_dim, K)
            self.mu = nn.Linear(feat_dim, K * act_dim)
            self.dist = MixtureSquashedDiagGaussian(act_dim=act_dim, K=K)
        else:
            raise ValueError(f"Unknown arch: {arch}")

        self.v_head = nn.Linear(self.backbone_v.out_dim, 1)

    def step(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            a, logp, _ = self.pi(obs, deterministic=deterministic)
            v = self.v(obs)
        return a, logp, v

    def pi(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.backbone_pi(obs)
        if self.arch == "gauss":
            mu = torch.tanh(self.mu(h))
            return self.dist(mu, deterministic=deterministic)
        else:
            logits = self.logits(h)
            mu = self.mu(h).view(-1, self.mog_components, self.act_dim)
            mu = torch.tanh(mu)
            return self.dist(logits, mu, deterministic=deterministic)

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        h = self.backbone_pi(obs)
        if self.arch == "gauss":
            mu = torch.tanh(self.mu(h))
            return self.dist.log_prob(mu, act)
        else:
            logits = self.logits(h)
            mu = self.mu(h).view(-1, self.mog_components, self.act_dim)
            mu = torch.tanh(mu)
            return self.dist.log_prob(logits, mu, act)

    def v(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.backbone_v(obs)
        return self.v_head(h).squeeze(-1)

    def export_config(self) -> Dict[str, Any]:
            return {
                "obs_dim": int(self.obs_dim),
                "act_dim": int(self.act_dim),
                "arch": str(self.arch),
                "mog_components": int(self.mog_components),
                "hidden_sizes": tuple(int(x) for x in self.hidden_sizes),
                "activation": str(self.activation),
                "layernorm": bool(self.layernorm),
                "residual_blocks": int(self.residual_blocks),
            }