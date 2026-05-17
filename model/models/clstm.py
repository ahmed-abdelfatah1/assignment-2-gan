"""Conditional LSTM decoder v3.1.

Changes vs v2/v3:
- cond is injected at every input position (not just h0/c0) so the DOW signal
  doesn't wash out during autoregressive generation.
- Aux DOW classifier head reads the final LSTM hidden state and predicts the
  conditioned day-of-week; the CE gradient flows back through every step,
  forcing the recurrent state to encode DOW.

`forward` returns (char_logits, aux_dow_logits). `sample` is unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config
from model.tokenizer import DateTokenizer

EMB_DIM: int = 64
HIDDEN: int = 256
N_DOW: int = len(config.DOW_TOKENS)  # 7


class CLSTM(nn.Module):
    """Single-layer LSTM hidden=256 with per-step cond injection + DOW aux head."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE, cond_dim: int = config.COND_DIM,
                 emb_dim: int = EMB_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.cond_proj = nn.Linear(cond_dim, 2 * hidden)              # → (h0, c0)
        self.cond_step_proj = nn.Linear(cond_dim, emb_dim)            # added at every step
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)
        self.aux_head = nn.Linear(hidden, N_DOW)

    def init_state(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0c0 = self.cond_proj(cond)
        h0, c0 = torch.chunk(h0c0, 2, dim=-1)
        return h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous()

    def _step_input(self, cond: torch.Tensor, seq_in: torch.Tensor) -> torch.Tensor:
        emb = self.emb(seq_in)                                         # (B, T, E)
        cond_step = self.cond_step_proj(cond).unsqueeze(1)             # (B, 1, E)
        return emb + cond_step                                         # broadcast over T

    def forward(self, cond: torch.Tensor, seq_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (char_logits over T input positions, aux DOW logits from final hidden)."""
        h, c = self.init_state(cond)
        x = self._step_input(cond, seq_in)
        out, (h_final, _) = self.lstm(x, (h, c))
        char_logits = self.head(out)                                   # (B, T, V)
        aux_dow_logits = self.aux_head(h_final.squeeze(0))             # (B, 7)
        return char_logits, aux_dow_logits

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, temperature: float = 1.0,
               max_len: int = config.MAX_SEQ_LEN) -> list[str]:
        device = cond.device
        b = cond.shape[0]
        h, c = self.init_state(cond)
        cond_step = self.cond_step_proj(cond).unsqueeze(1)             # (B, 1, E)
        tok = torch.full((b, 1), config.BOS_ID, dtype=torch.long, device=device)
        out_ids: list[list[int]] = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            emb = self.emb(tok) + cond_step                            # (B, 1, E)
            o, (h, c) = self.lstm(emb, (h, c))
            logits = self.head(o[:, -1, :]) / max(1e-6, temperature)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).squeeze(-1)
            for i in range(b):
                if not done[i].item():
                    if int(next_tok[i].item()) == config.EOS_ID:
                        done[i] = True
                    else:
                        out_ids[i].append(int(next_tok[i].item()))
            tok = next_tok.unsqueeze(-1)
            if bool(done.all().item()):
                break
        return [DateTokenizer.decode(torch.tensor(ids, dtype=torch.long)) for ids in out_ids]


def lstm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CE per token, ignoring PAD."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=config.PAD_ID,
    )
