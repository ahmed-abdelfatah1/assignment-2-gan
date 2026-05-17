"""Inference CLI: python predict.py -i IN -o OUT [--model {cgan,cvae,clstm,ctransformer,all}].

For --model all, writes 4 sibling files (<stem>_<model>.txt) plus the user's -o path,
the latter mirroring the transformer's predictions (so the assignment grader's exact
spec invocation produces a spec-format file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _root in (_HERE.parent, _HERE.parent.parent):
    if (_root / "model" / "__init__.py").exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import torch

from model import config
from model.constraints import parse_conds, verify_date
from model.models.cgan import Generator
from model.models.clstm import CLSTM
from model.models.ctransformer import CTransformer
from model.models.cvae import CVAE
from model.tokenizer import ConditionEncoder, decode_gan_output

K_RETRIES: int = 20


def _format_line(conds: dict[str, str], date_str: str) -> str:
    return f"[{conds['dow']}] [{conds['mon']}] [{conds['leap']}] [{conds['dec']}] {date_str}\n"


def _load_model(name: str, device: torch.device):
    path = config.WEIGHTS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"weights not found: {path}. Train the model first.")
    sd = torch.load(path, map_location=device)
    if name == "cgan":
        m = Generator().to(device)
        m.load_state_dict(sd["G"] if isinstance(sd, dict) and "G" in sd else sd)
    elif name == "cvae":
        m = CVAE().to(device); m.load_state_dict(sd)
    elif name == "clstm":
        m = CLSTM().to(device); m.load_state_dict(sd)
    elif name == "ctransformer":
        m = CTransformer().to(device); m.load_state_dict(sd)
    else:
        raise ValueError(name)
    m.eval()
    return m


def _sample_one(model_name: str, model, cond: torch.Tensor, conds: dict[str, str]) -> str:
    cb = cond.unsqueeze(0)
    if model_name in ("cgan", "cvae"):
        day_idx, yr_idx = model.sample_indices(cb)
        return decode_gan_output(int(day_idx[0]), int(yr_idx[0]), conds["mon"], conds["dec"])
    return model.sample(cb, temperature=1.0)[0]


def _predict_with(model_name: str, model, conds_list: list[dict[str, str]],
                  device: torch.device) -> tuple[list[str], int]:
    """Return (preds, n_failures). One date per input line, up to K_RETRIES per line."""
    preds: list[str] = []
    n_fail = 0
    for c in conds_list:
        cond = ConditionEncoder.encode(c).to(device)
        last = ""
        accepted = False
        for _ in range(K_RETRIES):
            cand = _sample_one(model_name, model, cond, c)
            last = cand
            if verify_date(cand, c):
                preds.append(cand)
                accepted = True
                break
        if not accepted:
            preds.append(last)
            n_fail += 1
    return preds, n_fail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", choices=list(config.MODEL_NAMES) + ["all"], default="all")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    config.set_seed(config.SEED)
    dev = config.device(args.device)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        raw_lines = [ln for ln in (l.rstrip("\n") for l in f) if ln.strip()]
    conds_list = [parse_conds(ln) for ln in raw_lines]
    print(f"loaded {len(conds_list)} input rows from {in_path}")

    targets = list(config.MODEL_NAMES) if args.model == "all" else [args.model]
    per_model_preds: dict[str, list[str]] = {}
    per_model_fail: dict[str, int] = {}
    for name in targets:
        print(f"loading {name} ...")
        m = _load_model(name, dev)
        preds, n_fail = _predict_with(name, m, conds_list, dev)
        per_model_preds[name] = preds
        per_model_fail[name] = n_fail
        print(f"  {name}: {n_fail}/{len(preds)} lines emitted with no valid sample after K={K_RETRIES}")

    if args.model == "all":
        stem = out_path.with_suffix("").name
        parent = out_path.parent
        for name in targets:
            sib = parent / f"{stem}_{name}.txt"
            with open(sib, "w", encoding="utf-8") as f:
                for c, d in zip(conds_list, per_model_preds[name]):
                    f.write(_format_line(c, d))
            print(f"wrote {sib}")
        canonical = per_model_preds["ctransformer"]
    else:
        canonical = per_model_preds[args.model]

    with open(out_path, "w", encoding="utf-8") as f:
        for c, d in zip(conds_list, canonical):
            f.write(_format_line(c, d))
    print(f"wrote {out_path}")
    print("\nfailure summary (samples that exhausted K retries):")
    for name in targets:
        print(f"  {name:14s}  {per_model_fail[name]}/{len(conds_list)}")


if __name__ == "__main__":
    main()
