"""Conditional GAN over a 31|10 split of (day_idx, year_last_digit)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config

NOISE_DIM: int = 32
HIDDEN: int = 128


def _split_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a (B, 41) tensor into (day, year_digit) chunks."""
    return logits[:, : config.DAY_DIM], logits[:, config.DAY_DIM:]


def gumbel_onehot(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Straight-through Gumbel-Softmax on a single categorical block."""
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


class Generator(nn.Module):
    """noise (32) ⊕ cond (62) → MLP[128,128] → 41 logits, sampled via Gumbel-softmax."""

    def __init__(self, noise_dim: int = NOISE_DIM, cond_dim: int = config.COND_DIM,
                 hidden: int = HIDDEN, out_dim: int = config.GAN_OUT_DIM) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, noise: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, cond], dim=-1)
        return self.net(x)

    def sample_onehot(self, cond: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Return a (B, 41) hard one-hot draw via Gumbel-softmax (differentiable)."""
        b = cond.shape[0]
        z = torch.randn(b, self.noise_dim, device=cond.device)
        logits = self.forward(z, cond)
        day_l, yr_l = _split_logits(logits)
        return torch.cat([gumbel_onehot(day_l, tau=tau), gumbel_onehot(yr_l, tau=tau)], dim=-1)

    @torch.no_grad()
    def sample_indices(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (day_idx, year_digit) by sampling categoricals (no Gumbel needed)."""
        b = cond.shape[0]
        z = torch.randn(b, self.noise_dim, device=cond.device)
        day_l, yr_l = _split_logits(self.forward(z, cond))
        day_probs = F.softmax(day_l, dim=-1)
        yr_probs = F.softmax(yr_l, dim=-1)
        day_idx = torch.multinomial(day_probs, num_samples=1).squeeze(-1)
        yr_idx = torch.multinomial(yr_probs, num_samples=1).squeeze(-1)
        return day_idx, yr_idx


class Discriminator(nn.Module):
    """(41 ⊕ 62) → MLP[128,128] → scalar logit."""

    def __init__(self, in_dim: int = config.GAN_OUT_DIM, cond_dim: int = config.COND_DIM,
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
        x = torch.cat([sample, cond], dim=-1)
        return self.net(x).squeeze(-1)


def d_loss(d_real: torch.Tensor, d_fake: torch.Tensor, smooth: float = 0.9) -> torch.Tensor:
    """One-sided label smoothing: real → 0.9, fake → 0."""
    real_target = torch.full_like(d_real, smooth)
    fake_target = torch.zeros_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_real, real_target) + \
        F.binary_cross_entropy_with_logits(d_fake, fake_target)


def g_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating generator loss: maximise log D(G(z))."""
    target = torch.ones_like(d_fake)
    return F.binary_cross_entropy_with_logits(d_fake, target)
