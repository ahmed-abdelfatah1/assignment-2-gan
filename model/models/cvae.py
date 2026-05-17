"""Conditional VAE v2 — single joint 310-class softmax over (day_idx, year_digit).

Mirrors the v2 cGAN: independent (day, year_digit) softmaxes cannot recover the
day-of-week constraint, so we switch to a single 310-class joint head and
single CE term over the joint target index.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config

Z_DIM: int = 32
HIDDEN: int = 256


class CVAE(nn.Module):
    """Encoder(target_310 ⊕ cond) → (μ, logσ²); Decoder(z ⊕ cond) → 310 logits."""

    def __init__(self, in_dim: int = config.JOINT_DIM, cond_dim: int = config.COND_DIM,
                 hidden: int = HIDDEN, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.z_dim = z_dim

        self.enc_body = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, target: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_body(torch.cat([target, cond], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, cond], dim=-1))

    def forward(self, target: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(target, cond)
        z = self.reparam(mu, logvar)
        return self.decode(z, cond), mu, logvar

    @torch.no_grad()
    def sample_indices(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn(cond.shape[0], self.z_dim, device=cond.device)
        probs = F.softmax(self.decode(z, cond), dim=-1)
        idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        day_idx = idx // config.YEAR_DIGIT_DIM
        yr_idx = idx % config.YEAR_DIGIT_DIM
        return day_idx, yr_idx


def vae_loss(
    logits: torch.Tensor,
    day_idx: torch.Tensor,
    year_digit: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE over the 310-class joint + β · KL."""
    target_idx = day_idx * config.YEAR_DIGIT_DIM + year_digit
    ce = F.cross_entropy(logits, target_idx)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    total = ce + beta * kl
    return total, {"ce": float(ce.item()), "kl": float(kl.item())}
