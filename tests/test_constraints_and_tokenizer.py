"""Unit tests for the verification primitive, the tokenizers, and model smoke tests."""

from __future__ import annotations

import random

import pytest
import torch

from model import config
from model.constraints import parse_conds, verify_date
from model.dataset import load_lines
from model.models.cgan import Discriminator, Generator
from model.models.clstm import CLSTM
from model.models.ctransformer import CTransformer
from model.models.cvae import CVAE
from model.tokenizer import (
    ConditionEncoder,
    DateTokenizer,
    decode_gan_output,
    encode_gan_target,
)


@pytest.fixture(scope="module")
def lines() -> list[str]:
    return load_lines(config.DATA_PATH)


def test_verify_date_true_for_50_real_lines(lines: list[str]) -> None:
    random.seed(0)
    for ln in random.sample(lines, 50):
        c = parse_conds(ln)
        assert verify_date(c["date"], c), ln


def test_verify_date_false_for_corrupted_lines(lines: list[str]) -> None:
    random.seed(1)
    samples = random.sample(lines, 10)
    rot_dow = {a: b for a, b in zip(config.DOW_TOKENS,
                                    config.DOW_TOKENS[1:] + config.DOW_TOKENS[:1])}
    for ln in samples:
        c = parse_conds(ln)
        bad = dict(c); bad["dow"] = rot_dow[c["dow"]]
        assert not verify_date(c["date"], bad)
        bad2 = dict(c); bad2["mon"] = "JAN" if c["mon"] != "JAN" else "FEB"
        assert not verify_date(c["date"], bad2)
        bad3 = dict(c); bad3["leap"] = "True" if c["leap"] == "False" else "False"
        assert not verify_date(c["date"], bad3)
        bad4 = dict(c); bad4["dec"] = "999"
        assert not verify_date(c["date"], bad4)


def test_condition_encoder_roundtrip(lines: list[str]) -> None:
    random.seed(2)
    for ln in random.sample(lines, 100):
        c = parse_conds(ln)
        v = ConditionEncoder.encode(c)
        d = ConditionEncoder.decode(v)
        assert d == {k: c[k] for k in ("dow", "mon", "leap", "dec")}


def test_date_tokenizer_roundtrip(lines: list[str]) -> None:
    random.seed(3)
    for ln in random.sample(lines, 100):
        c = parse_conds(ln)
        ids = DateTokenizer.encode(c["date"])
        back = DateTokenizer.decode(ids)
        assert back == c["date"]


def test_gan_target_roundtrip(lines: list[str]) -> None:
    random.seed(4)
    for ln in random.sample(lines, 100):
        c = parse_conds(ln)
        day_idx, year_digit = encode_gan_target(c["date"])
        rebuilt = decode_gan_output(day_idx, year_digit, c["mon"], c["dec"])
        assert rebuilt == c["date"]


def test_models_forward_smoke() -> None:
    B = 4
    cond = torch.zeros(B, config.COND_DIM)
    cond[:, 0] = 1; cond[:, 7] = 1; cond[:, 19] = 1; cond[:, 21] = 1

    g = Generator(); d = Discriminator()
    fake = g.sample_onehot(cond)
    assert fake.shape == (B, config.GAN_OUT_DIM)
    assert d(fake, cond).shape == (B,)

    v = CVAE()
    target = torch.zeros(B, config.GAN_OUT_DIM); target[:, 0] = 1; target[:, 31] = 1
    logits, mu, lv = v(target, cond)
    assert logits.shape == (B, config.GAN_OUT_DIM)
    assert mu.shape == (B, 16) and lv.shape == (B, 16)

    seq = torch.zeros(B, config.MAX_SEQ_LEN, dtype=torch.long)
    seq[:, 0] = config.BOS_ID
    lm = CLSTM()
    assert lm(cond, seq[:, :-1]).shape == (B, config.MAX_SEQ_LEN - 1, config.VOCAB_SIZE)

    tm = CTransformer()
    assert tm(cond, seq[:, :-1]).shape == (B, config.MAX_SEQ_LEN - 1, config.VOCAB_SIZE)
