"""Global configuration: seed, vocabularies, paths, helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

SEED: int = 42

DOW_TOKENS: list[str] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_TOKENS: list[str] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]
LEAP_TOKENS: list[str] = ["False", "True"]
DECADE_TOKENS: list[str] = [str(d) for d in range(180, 221)]  # 41 tokens

DOW_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(DOW_TOKENS)}
MONTH_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(MONTH_TOKENS)}
LEAP_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(LEAP_TOKENS)}
DECADE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(DECADE_TOKENS)}

COND_DIM: int = len(DOW_TOKENS) + len(MONTH_TOKENS) + len(LEAP_TOKENS) + len(DECADE_TOKENS)
# 7 + 12 + 2 + 41 = 62
assert COND_DIM == 62

DAY_DIM: int = 31           # day-of-month index 0..30
YEAR_DIGIT_DIM: int = 10    # last digit 0..9
GAN_OUT_DIM: int = DAY_DIM + YEAR_DIGIT_DIM   # 41

CHAR_VOCAB: list[str] = [
    "<pad>", "<bos>", "<eos>", "-",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]
CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CHAR_VOCAB)}
PAD_ID: int = CHAR_TO_IDX["<pad>"]
BOS_ID: int = CHAR_TO_IDX["<bos>"]
EOS_ID: int = CHAR_TO_IDX["<eos>"]
VOCAB_SIZE: int = len(CHAR_VOCAB)  # 14

MAX_SEQ_LEN: int = 12   # BOS + up to 10 chars (e.g. "31-12-2200") + EOS

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
DATA_PATH: Path = DATA_DIR / "data.txt"
EXAMPLE_INPUT_PATH: Path = DATA_DIR / "example_input.txt"
WEIGHTS_DIR: Path = REPO_ROOT / "model" / "weights"
REPORT_DIR: Path = REPO_ROOT / "report"

MODEL_NAMES: tuple[str, ...] = ("cgan", "cvae", "clstm", "ctransformer")


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def device(prefer: str = "auto") -> torch.device:
    """Resolve the compute device. prefer ∈ {auto, cpu, cuda}."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
