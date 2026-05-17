"""Hand-written decoder-only Transformer.

Architecture: 2 decoder blocks, d_model=64, 4 heads, FFN=128.
The 62-dim condition is projected to d_model and prepended as token 0; the
remaining positions are character embeddings of the date string. A causal mask
is applied over the whole sequence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config
from model.tokenizer import DateTokenizer

D_MODEL: int = 128
N_HEADS: int = 4
D_FF: int = 256
N_LAYERS: int = 4
N_DOW: int = len(config.DOW_TOKENS)  # 7


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product MHA; supports a Boolean causal mask."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    """Pre-norm decoder block: residual MHA + residual FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ff(self.ln2(x))
        return x


def sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CTransformer(nn.Module):
    """Decoder-only Transformer. Position 0 = condition; positions 1..T = chars."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE, cond_dim: int = config.COND_DIM,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS, d_ff: int = D_FF,
                 n_layers: int = N_LAYERS, max_len: int = config.MAX_SEQ_LEN + 1) -> None:
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=config.PAD_ID)
        self.cond_proj = nn.Linear(cond_dim, d_model)
        self.register_buffer("pe", sinusoidal_pe(max_len, d_model), persistent=False)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len)),
                             persistent=False)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.aux_head = nn.Linear(d_model, N_DOW)

    def _backbone(self, cond: torch.Tensor, seq_in: torch.Tensor) -> torch.Tensor:
        b, t = seq_in.shape
        cond_tok = self.cond_proj(cond).unsqueeze(1)        # (B, 1, d_model)
        char_emb = self.emb(seq_in)                          # (B, T, d_model)
        x = torch.cat([cond_tok, char_emb], dim=1)           # (B, 1+T, d_model)
        x = x + self.pe[: x.shape[1]].unsqueeze(0)
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.ln_f(x)

    def forward(self, cond: torch.Tensor, seq_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training forward. Returns:
            - char_logits over the T char positions (cond-token logits discarded)
            - aux_dow_logits from the cond-token's final hidden (position 0)
        """
        h = self._backbone(cond, seq_in)
        char_logits = self.head(h[:, 1:, :])              # (B, T, V)
        aux_dow_logits = self.aux_head(h[:, 0, :])        # (B, 7)
        return char_logits, aux_dow_logits

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, temperature: float = 1.0,
               max_len: int = config.MAX_SEQ_LEN) -> list[str]:
        device = cond.device
        b = cond.shape[0]
        toks = torch.full((b, 1), config.BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        out_ids: list[list[int]] = [[] for _ in range(b)]
        for _ in range(max_len - 1):
            h = self._backbone(cond, toks)
            logits = self.head(h[:, -1, :]) / max(1e-6, temperature)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            for i in range(b):
                if not done[i].item():
                    if int(next_tok[i].item()) == config.EOS_ID:
                        done[i] = True
                    else:
                        out_ids[i].append(int(next_tok[i].item()))
            toks = torch.cat([toks, next_tok.unsqueeze(-1)], dim=1)
            if bool(done.all().item()):
                break
        return [DateTokenizer.decode(torch.tensor(ids, dtype=torch.long)) for ids in out_ids]


def transformer_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=config.PAD_ID,
    )
