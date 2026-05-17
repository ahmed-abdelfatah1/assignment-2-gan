"""Dataset + 80/10/10 deterministic split."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from model import config
from model.constraints import parse_conds
from model.tokenizer import ConditionEncoder, DateTokenizer, encode_gan_target

Mode = Literal["vec", "seq"]


def load_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f if ln.strip()]


class DateDataset(Dataset):
    """Wraps the parsed data.txt. Two modes:

    - 'vec': returns (cond[62], day_idx, year_digit) for cGAN / cVAE.
    - 'seq': returns (cond[62], char_ids[MAX_SEQ_LEN]) for cLSTM / cTransformer.
    """

    def __init__(
        self,
        lines: list[str] | None = None,
        path: str | Path | None = None,
        mode: Mode = "vec",
        indices: list[int] | None = None,
    ) -> None:
        if lines is None:
            if path is None:
                path = config.DATA_PATH
            lines = load_lines(path)
        self._lines = lines
        self._mode = mode
        self._indices = indices if indices is not None else list(range(len(lines)))
        self._parsed: list[dict[str, str]] = [parse_conds(self._lines[i]) for i in self._indices]

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def parsed(self) -> list[dict[str, str]]:
        return self._parsed

    def __getitem__(self, i: int):
        c = self._parsed[i]
        cond = ConditionEncoder.encode(c)
        date = c["date"]
        if self._mode == "vec":
            day_idx, year_digit = encode_gan_target(date)
            return cond, torch.tensor(day_idx, dtype=torch.long), torch.tensor(
                year_digit, dtype=torch.long
            )
        ids = DateTokenizer.encode(date)
        return cond, ids


def build_splits(
    n: int, seed: int = config.SEED, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> dict[str, list[int]]:
    """Deterministic 80/10/10 (or supplied) split over the integer range [0, n)."""
    assert abs(sum(ratios) - 1.0) < 1e-6, ratios
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def build_datasets(
    mode: Mode, path: str | Path | None = None, seed: int = config.SEED
) -> tuple[DateDataset, DateDataset, DateDataset]:
    """Convenience: load all data + build all three split datasets in one call."""
    if path is None:
        path = config.DATA_PATH
    lines = load_lines(path)
    splits = build_splits(len(lines), seed=seed)
    return (
        DateDataset(lines=lines, mode=mode, indices=splits["train"]),
        DateDataset(lines=lines, mode=mode, indices=splits["val"]),
        DateDataset(lines=lines, mode=mode, indices=splits["test"]),
    )
