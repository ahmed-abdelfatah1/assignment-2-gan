"""Tokenizers + decoders. Pure functions / lightweight classes (no NN state)."""

from __future__ import annotations

import torch

from model import config


class ConditionEncoder:
    """One-hot encode the 4-condition tuple into a 62-dim vector."""

    @staticmethod
    def encode(conds: dict[str, str]) -> torch.Tensor:
        vec = torch.zeros(config.COND_DIM, dtype=torch.float32)
        off = 0
        vec[off + config.DOW_TO_IDX[conds["dow"]]] = 1.0
        off += len(config.DOW_TOKENS)
        vec[off + config.MONTH_TO_IDX[conds["mon"]]] = 1.0
        off += len(config.MONTH_TOKENS)
        vec[off + config.LEAP_TO_IDX[conds["leap"]]] = 1.0
        off += len(config.LEAP_TOKENS)
        vec[off + config.DECADE_TO_IDX[conds["dec"]]] = 1.0
        return vec

    @staticmethod
    def decode(vec: torch.Tensor) -> dict[str, str]:
        v = vec.detach().cpu()
        off = 0
        dow = config.DOW_TOKENS[int(v[off:off + len(config.DOW_TOKENS)].argmax())]
        off += len(config.DOW_TOKENS)
        mon = config.MONTH_TOKENS[int(v[off:off + len(config.MONTH_TOKENS)].argmax())]
        off += len(config.MONTH_TOKENS)
        leap = config.LEAP_TOKENS[int(v[off:off + len(config.LEAP_TOKENS)].argmax())]
        off += len(config.LEAP_TOKENS)
        dec = config.DECADE_TOKENS[int(v[off:off + len(config.DECADE_TOKENS)].argmax())]
        return {"dow": dow, "mon": mon, "leap": leap, "dec": dec}


class DateTokenizer:
    """Char-level tokenizer for the 'd-m-yyyy' string."""

    @staticmethod
    def encode(date_str: str) -> torch.Tensor:
        ids = [config.BOS_ID]
        for ch in date_str:
            if ch not in config.CHAR_TO_IDX:
                raise ValueError(f"char {ch!r} not in vocab")
            ids.append(config.CHAR_TO_IDX[ch])
        ids.append(config.EOS_ID)
        while len(ids) < config.MAX_SEQ_LEN:
            ids.append(config.PAD_ID)
        if len(ids) > config.MAX_SEQ_LEN:
            raise ValueError(f"date {date_str!r} exceeds MAX_SEQ_LEN={config.MAX_SEQ_LEN}")
        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def decode(ids: torch.Tensor) -> str:
        out: list[str] = []
        for tok in ids.tolist():
            if tok in (config.BOS_ID, config.PAD_ID):
                continue
            if tok == config.EOS_ID:
                break
            out.append(config.CHAR_VOCAB[tok])
        return "".join(out)


def decode_gan_output(day_idx: int, year_digit: int, mon: str, dec: str) -> str:
    """Build a 'd-m-yyyy' string from a 31|10 split + the given conditions."""
    day = day_idx + 1
    month = config.MONTH_TO_IDX[mon] + 1
    year = int(dec) * 10 + year_digit
    return f"{day}-{month}-{year}"


def encode_gan_target(date_str: str) -> tuple[int, int]:
    """Return (day_idx, year_last_digit) for cGAN/cVAE training."""
    d, _m, y = (int(p) for p in date_str.split("-"))
    return d - 1, y % 10


def gan_target_onehot(day_idx: int, year_digit: int) -> torch.Tensor:
    """Build the 41-dim one-hot target used by the legacy v1 cGAN discriminator."""
    t = torch.zeros(config.GAN_OUT_DIM, dtype=torch.float32)
    t[day_idx] = 1.0
    t[config.DAY_DIM + year_digit] = 1.0
    return t


def joint_idx(day_idx: int, year_digit: int) -> int:
    """Pack (day_idx, year_digit) into a single index in [0, 310)."""
    return day_idx * config.YEAR_DIGIT_DIM + year_digit


def joint_split(idx: int) -> tuple[int, int]:
    """Inverse of joint_idx."""
    return divmod(idx, config.YEAR_DIGIT_DIM)


def joint_onehot(idx: int) -> torch.Tensor:
    """310-dim one-hot for the v2 GAN discriminator."""
    t = torch.zeros(config.JOINT_DIM, dtype=torch.float32)
    t[idx] = 1.0
    return t
