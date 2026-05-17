"""Conditional GAN v2 — single joint 310-class softmax over (day_idx, year_digit).

v1 used two independent softmaxes (31 + 10), which provably can't capture the
day-of-week constraint because the valid (day, year_digit) set for a given
(DOW, MON, LEAP, DEC) tuple is non-Cartesian. v2 emits one 310-class softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config

NOISE_DIM: int = 32
HIDDEN: int = 256


def gumbel_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-Softmax over the 310-class joint."""
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


class Generator(nn.Module):
    """noise (32) ⊕ cond (62) → MLP[256, 256, 256] → 310 logits (joint softmax)."""

    def __init__(self, noise_dim: int = NOISE_DIM, cond_dim: int = config.COND_DIM,
                 hidden: int = HIDDEN, out_dim: int = config.JOINT_DIM) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([noise, cond], dim=-1))

    def sample_onehot(self, cond: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Differentiable 310-dim one-hot via Gumbel-Softmax."""
        z = torch.randn(cond.shape[0], self.noise_dim, device=cond.device)
        return gumbel_onehot(self.forward(z, cond), tau=tau)

    @torch.no_grad()
    def sample_indices(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Multinomial draw, then split into (day_idx, year_digit)."""
        z = torch.randn(cond.shape[0], self.noise_dim, device=cond.device)
        probs = F.softmax(self.forward(z, cond), dim=-1)
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        day_idx = idx // config.YEAR_DIGIT_DIM
        yr_idx = idx % config.YEAR_DIGIT_DIM
        return day_idx, yr_idx


class Discriminator(nn.Module):
    """(310 ⊕ 62) → MLP[256, 256] → scalar logit."""

    def __init__(self, in_dim: int = config.JOINT_DIM, cond_dim: int = config.COND_DIM,
                 hidden: int = HIDDEN) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, sample: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([sample, cond], dim=-1)).squeeze(-1)


def d_loss(d_real: torch.Tensor, d_fake: torch.Tensor, smooth: float = 0.9) -> torch.Tensor:
    """One-sided label smoothing: real → 0.9, fake → 0."""
    return F.binary_cross_entropy_with_logits(d_real, torch.full_like(d_real, smooth)) \
        + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))


def g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: maximise log D(G(z))."""
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
