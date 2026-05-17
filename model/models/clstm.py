"""Conditional LSTM decoder: condition seeds (h0, c0); chars autoregressed from BOS."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import config
from model.tokenizer import DateTokenizer

EMB_DIM: int = 32
HIDDEN: int = 128


class CLSTM(nn.Module):
    """Single-layer LSTM with hidden=128; condition projects to (h0, c0)."""

    def __init__(self, vocab_size: int = config.VOCAB_SIZE, cond_dim: int = config.COND_DIM,
                 emb_dim: int = EMB_DIM, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=config.PAD_ID)
        self.cond_proj = nn.Linear(cond_dim, 2 * hidden)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def init_state(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h0c0 = self.cond_proj(cond)                          # (B, 2H)
        h0, c0 = torch.chunk(h0c0, 2, dim=-1)                # each (B, H)
        return h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous()

    def forward(self, cond: torch.Tensor, seq_in: torch.Tensor) -> torch.Tensor:
        """seq_in: (B, T) token ids — feed everything except the final position; the
        model predicts the next token at each step.
        """
        h, c = self.init_state(cond)
        emb = self.emb(seq_in)
        out, _ = self.lstm(emb, (h, c))
        return self.head(out)  # (B, T, V)

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, temperature: float = 1.0,
               max_len: int = config.MAX_SEQ_LEN) -> list[str]:
        """Autoregressive sample. Returns one date string per batch row."""
        device = cond.device
        b = cond.shape[0]
        h, c = self.init_state(cond)
        tok = torch.full((b, 1), config.BOS_ID, dtype=torch.long, device=device)
        out_ids: list[list[int]] = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            emb = self.emb(tok)
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
