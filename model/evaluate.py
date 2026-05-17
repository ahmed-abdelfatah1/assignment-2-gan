"""Evaluate the four trained models on the test split + emit report artifacts."""

from __future__ import annotations

import json
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import config
from model.constraints import verify_date, which_condition_failed
from model.dataset import build_datasets
from model.metrics import (
    condition_satisfaction_rate,
    diversity,
    per_condition_breakdown,
)
from model.models.cgan import Generator
from model.models.clstm import CLSTM
from model.models.ctransformer import CTransformer
from model.models.cvae import CVAE
from model.tokenizer import ConditionEncoder, decode_gan_output

N_DIVERSITY_SAMPLES: int = 10
DIVERSITY_GROUPS: int = 200
SUCCESS_EXAMPLES: int = 10
FAILURE_EXAMPLES: int = 5


def _load(name: str, device: torch.device):
    path = config.WEIGHTS_DIR / f"{name}.pt"
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


def _sample_batch(name: str, model, cond: torch.Tensor,
                  conds_list: list[dict[str, str]]) -> list[str]:
    if name in ("cgan", "cvae"):
        day_idx, yr_idx = model.sample_indices(cond)
        return [
            decode_gan_output(int(day_idx[j]), int(yr_idx[j]),
                              conds_list[j]["mon"], conds_list[j]["dec"])
            for j in range(cond.shape[0])
        ]
    return model.sample(cond, temperature=1.0)


def _evaluate_model(name: str, model, conds_list: list[dict[str, str]],
                    device: torch.device, batch_size: int = 512) -> dict[str, object]:
    cond_tensor = torch.stack([ConditionEncoder.encode(c) for c in conds_list]).to(device)
    preds: list[str] = []
    for i in range(0, len(conds_list), batch_size):
        cb = cond_tensor[i:i + batch_size]
        preds.extend(_sample_batch(name, model, cb, conds_list[i:i + batch_size]))

    csr = condition_satisfaction_rate(preds, conds_list)
    per_cond = per_condition_breakdown(preds, conds_list)

    distinct = list({(c["dow"], c["mon"], c["leap"], c["dec"]): c for c in conds_list}.values())
    random.seed(config.SEED)
    random.shuffle(distinct)
    groups = distinct[:DIVERSITY_GROUPS]
    cond_div = torch.stack([ConditionEncoder.encode(c) for c in groups]).to(device)
    samples_per_cond: list[list[str]] = [[] for _ in groups]
    for _ in range(N_DIVERSITY_SAMPLES):
        out = _sample_batch(name, model, cond_div, groups)
        for j, s in enumerate(out):
            samples_per_cond[j].append(s)
    div = diversity(samples_per_cond)

    successes: list[tuple[dict[str, str], str]] = []
    failures: list[tuple[dict[str, str], str, str]] = []
    for c, p in zip(conds_list, preds):
        if verify_date(p, c):
            if len(successes) < SUCCESS_EXAMPLES:
                successes.append((c, p))
        else:
            if len(failures) < FAILURE_EXAMPLES:
                failures.append((c, p, which_condition_failed(p, c)))
        if len(successes) >= SUCCESS_EXAMPLES and len(failures) >= FAILURE_EXAMPLES:
            break

    return {
        "csr": csr,
        "per_condition": per_cond,
        "diversity": div,
        "successes": successes,
        "failures": failures,
    }


def _plot_curves(histories: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, name in zip(axes.flat, config.MODEL_NAMES):
        h = histories.get(name)
        if not h:
            ax.set_title(f"{name} (no history)")
            ax.axis("off")
            continue
        epochs = [r["epoch"] for r in h]
        ax2 = ax.twinx()
        if "train_loss" in h[0]:
            ax.plot(epochs, [r["train_loss"] for r in h], label="train loss", color="C0")
        if "train_g" in h[0]:
            ax.plot(epochs, [r["train_g"] for r in h], label="G", color="C0")
            ax.plot(epochs, [r["train_d"] for r in h], label="D", color="C3")
        ax2.plot(epochs, [r["val_csr"] for r in h], label="val CSR", color="C2", linestyle="--")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("val CSR")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(name)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    fig.tight_layout()
    out = config.REPORT_DIR / "loss_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    config.set_seed(config.SEED)
    device = config.device("auto")
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, test_ds = build_datasets("vec")  # mode irrelevant; we just need .parsed
    conds_list = test_ds.parsed
    print(f"test split size: {len(conds_list)}")

    results: dict[str, dict] = {}
    histories: dict[str, list[dict]] = {}
    sample_lines: list[str] = []
    for name in config.MODEL_NAMES:
        wpath = config.WEIGHTS_DIR / f"{name}.pt"
        if not wpath.exists():
            print(f"[skip] {name}: no weights at {wpath}")
            continue
        print(f"evaluating {name} ...")
        m = _load(name, device)
        out = _evaluate_model(name, m, conds_list, device)
        results[name] = {
            "csr": out["csr"],
            "per_condition": out["per_condition"],
            "diversity": out["diversity"],
        }
        h_path = config.WEIGHTS_DIR / f"{name}_history.json"
        if h_path.exists():
            with open(h_path, "r", encoding="utf-8") as f:
                histories[name] = json.load(f)

        sample_lines.append(f"### {name}\n")
        sample_lines.append(
            f"CSR={out['csr']:.4f}  per_cond={out['per_condition']}  diversity={out['diversity']:.4f}\n\n")
        sample_lines.append("-- successes --\n")
        for c, p in out["successes"]:
            sample_lines.append(
                f"[{c['dow']}] [{c['mon']}] [{c['leap']}] [{c['dec']}]  -> {p}  OK\n")
        sample_lines.append("\n-- failures --\n")
        for c, p, why in out["failures"]:
            sample_lines.append(
                f"[{c['dow']}] [{c['mon']}] [{c['leap']}] [{c['dec']}]  -> {p}  FAIL({why})\n")
        sample_lines.append("\n")

    with open(config.REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {config.REPORT_DIR / 'metrics.json'}")

    with open(config.REPORT_DIR / "sample_outputs.txt", "w", encoding="utf-8") as f:
        f.writelines(sample_lines)
    print(f"wrote {config.REPORT_DIR / 'sample_outputs.txt'}")

    _plot_curves(histories)

    print("\n===== test summary =====")
    for n, r in results.items():
        print(f"  {n:14s}  CSR={r['csr']:.4f}  div={r['diversity']:.4f}  per_cond={r['per_condition']}")


if __name__ == "__main__":
    main()
