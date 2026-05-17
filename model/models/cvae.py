"""Conditional VAE over a 31|10 split of (day_idx, year_last_digit)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config

Z_DIM: int = 16
HIDDEN: int = 128


class CVAE(nn.Module):
    """Encoder(target ⊕ cond) → (μ, logσ²); Decoder(z ⊕ cond) → 41 logits."""

    def __init__(self, in_dim: int = config.GAN_OUT_DIM, cond_dim: int = config.COND_DIM,
                 hidden: int = HIDDEN, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.z_dim = z_dim

        self.enc_body = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.enc_mu = nn.Linear(hidden, z_dim)
        self.enc_logvar = nn.Linear(hidden, z_dim)

        self.dec = nn.Sequential(
            nn.Linear(z_dim + cond_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, target: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc_body(torch.cat([target, cond], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, cond], dim=-1))

    def forward(self, target: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(target, cond)
        z = self.reparam(mu, logvar)
        logits = self.decode(z, cond)
        return logits, mu, logvar

    @torch.no_grad()
    def sample_indices(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = cond.shape[0]
        z = torch.randn(b, self.z_dim, device=cond.device)
        logits = self.decode(z, cond)
        day_l = logits[:, : config.DAY_DIM]
        yr_l = logits[:, config.DAY_DIM:]
        day_probs = F.softmax(day_l, dim=-1)
        yr_probs = F.softmax(yr_l, dim=-1)
        day_idx = torch.multinomial(day_probs, num_samples=1).squeeze(-1)
        yr_idx = torch.multinomial(yr_probs, num_samples=1).squeeze(-1)
        return day_idx, yr_idx


def vae_loss(
    logits: torch.Tensor,
    day_idx: torch.Tensor,
    year_digit: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE(day) + CE(year) + β · KL. Returns scalar + components for logging."""
    day_l = logits[:, : config.DAY_DIM]
    yr_l = logits[:, config.DAY_DIM:]
    ce_day = F.cross_entropy(day_l, day_idx)
    ce_yr = F.cross_entropy(yr_l, year_digit)
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    total = ce_day + ce_yr + beta * kl
    return total, {
        "ce_day": float(ce_day.item()),
        "ce_yr": float(ce_yr.item()),
        "kl": float(kl.item()),
    }
