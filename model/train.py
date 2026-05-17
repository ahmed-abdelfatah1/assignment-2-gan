"""Training entry point: hand-written loops for cGAN / cVAE / cLSTM / cTransformer."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as `python model/train.py ...` or `cd model && python train.py ...`
_HERE = Path(__file__).resolve().parent
for _root in (_HERE.parent, _HERE.parent.parent):
    if (_root / "model" / "__init__.py").exists() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import config
from model.dataset import build_datasets, DateDataset
from model.metrics import condition_satisfaction_rate
from model.models.cgan import Discriminator, Generator, d_loss, g_loss
from model.models.clstm import CLSTM, lstm_loss
from model.models.ctransformer import CTransformer, transformer_loss
from model.models.cvae import CVAE, vae_loss
from model.tokenizer import ConditionEncoder, decode_gan_output, joint_onehot


def _subset(val_ds: DateDataset, n: int) -> DateDataset:
    """Cheap clone of val_ds restricted to its first n items."""
    out = DateDataset.__new__(DateDataset)
    out._lines = val_ds._lines
    out._mode = val_ds._mode
    out._indices = val_ds._indices[:n]
    out._parsed = val_ds._parsed[:n]
    return out


def _make_loader(ds: DateDataset, batch_size: int, device: torch.device) -> DataLoader:
    """DataLoader with CUDA-only num_workers+pin_memory speedups (Windows + num_workers > 0 in this venv triggers fork issues, so we keep workers=0 on CPU)."""
    use_cuda = device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )


def _val_csr_vec(sample_fn, val_ds: DateDataset, device: torch.device,
                 batch_size: int = 1024) -> float:
    conds_list = val_ds.parsed
    cond_tensor = torch.stack([ConditionEncoder.encode(c) for c in conds_list]).to(device)
    preds: list[str] = []
    for i in range(0, len(conds_list), batch_size):
        cb = cond_tensor[i:i + batch_size]
        day_idx, yr_idx = sample_fn(cb)
        for j in range(cb.shape[0]):
            c = conds_list[i + j]
            preds.append(decode_gan_output(int(day_idx[j]), int(yr_idx[j]), c["mon"], c["dec"]))
    return condition_satisfaction_rate(preds, conds_list)


def _val_csr_seq(sample_fn, val_ds: DateDataset, device: torch.device,
                 batch_size: int = 256) -> float:
    conds_list = val_ds.parsed
    cond_tensor = torch.stack([ConditionEncoder.encode(c) for c in conds_list]).to(device)
    preds: list[str] = []
    for i in range(0, len(conds_list), batch_size):
        cb = cond_tensor[i:i + batch_size]
        preds.extend(sample_fn(cb))
    return condition_satisfaction_rate(preds, conds_list)


def _save_history(name: str, history: list[dict[str, float]]) -> None:
    path = config.WEIGHTS_DIR / f"{name}_history.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _save_weights(name: str, state_dict: dict) -> None:
    path = config.WEIGHTS_DIR / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


# ---------------------------------------------------------------------------
# cGAN
# ---------------------------------------------------------------------------

def train_cgan(epochs: int, batch_size: int, lr: float, device: torch.device,
               val_subset: int = 4096) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("vec")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        G.train(); D.train()
        sum_g = sum_d = 0.0; nb = 0
        for cond, day_idx, yr_idx in tqdm(loader, desc=f"cgan e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); day_idx = day_idx.to(device); yr_idx = yr_idx.to(device)
            real = torch.stack([joint_onehot(int(d) * config.YEAR_DIGIT_DIM + int(y))
                                for d, y in zip(day_idx, yr_idx)]).to(device)
            fake = G.sample_onehot(cond).detach()
            opt_d.zero_grad()
            loss_d = d_loss(D(real, cond), D(fake, cond))
            loss_d.backward(); opt_d.step()
            fake_g = G.sample_onehot(cond)
            opt_g.zero_grad()
            loss_g = g_loss(D(fake_g, cond))
            loss_g.backward(); opt_g.step()
            sum_g += float(loss_g.item()); sum_d += float(loss_d.item()); nb += 1
        G.eval()
        val_csr = _val_csr_vec(G.sample_indices, val_small, device)
        history.append({"epoch": ep + 1, "train_g": sum_g / max(1, nb),
                        "train_d": sum_d / max(1, nb), "val_csr": val_csr})
        print(f"[cgan] epoch {ep+1}/{epochs}  G={sum_g/max(1,nb):.4f}  D={sum_d/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights("cgan", {"G": G.state_dict(), "D": D.state_dict()})
    if best_csr == 0.0:  # save final state even if no improvement seen
        _save_weights("cgan", {"G": G.state_dict(), "D": D.state_dict()})
    _save_history("cgan", history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# cVAE
# ---------------------------------------------------------------------------

def train_cvae(epochs: int, batch_size: int, lr: float, device: torch.device,
               val_subset: int = 4096) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("vec")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    M = CVAE().to(device)
    opt = torch.optim.Adam(M.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        beta = min(1.0, (ep + 1) / 10.0)
        M.train()
        sum_loss = 0.0; nb = 0
        for cond, day_idx, yr_idx in tqdm(loader, desc=f"cvae e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); day_idx = day_idx.to(device); yr_idx = yr_idx.to(device)
            target = torch.stack([joint_onehot(int(d) * config.YEAR_DIGIT_DIM + int(y))
                                  for d, y in zip(day_idx, yr_idx)]).to(device)
            opt.zero_grad()
            logits, mu, logvar = M(target, cond)
            loss, _parts = vae_loss(logits, day_idx, yr_idx, mu, logvar, beta=beta)
            loss.backward(); opt.step()
            sum_loss += float(loss.item()); nb += 1
        M.eval()
        val_csr = _val_csr_vec(M.sample_indices, val_small, device)
        history.append({"epoch": ep + 1, "beta": beta, "train_loss": sum_loss / max(1, nb),
                        "val_csr": val_csr})
        print(f"[cvae] epoch {ep+1}/{epochs}  beta={beta:.2f}  loss={sum_loss/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights("cvae", M.state_dict())
    if best_csr == 0.0:
        _save_weights("cvae", M.state_dict())
    _save_history("cvae", history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# cLSTM / cTransformer share a loop
# ---------------------------------------------------------------------------

def train_seq_model(name: str, build_model, loss_fn, epochs: int, batch_size: int,
                    lr: float, device: torch.device, val_subset: int = 2048) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("seq")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    M = build_model().to(device)
    opt = torch.optim.Adam(M.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        M.train()
        sum_loss = 0.0; nb = 0
        for cond, seq in tqdm(loader, desc=f"{name} e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); seq = seq.to(device)
            opt.zero_grad()
            logits = M(cond, seq[:, :-1])
            loss = loss_fn(logits, seq[:, 1:])
            loss.backward(); opt.step()
            sum_loss += float(loss.item()); nb += 1
        M.eval()
        val_csr = _val_csr_seq(lambda c: M.sample(c, temperature=1.0), val_small, device)
        history.append({"epoch": ep + 1, "train_loss": sum_loss / max(1, nb), "val_csr": val_csr})
        print(f"[{name}] epoch {ep+1}/{epochs}  loss={sum_loss/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights(name, M.state_dict())
    if best_csr == 0.0:
        _save_weights(name, M.state_dict())
    _save_history(name, history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(config.MODEL_NAMES) + ["all"], default="all")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr-gan", type=float, default=2e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    config.set_seed(config.SEED)
    dev = config.device(args.device)
    print(f"using device: {dev}")
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(config.MODEL_NAMES) if args.model == "all" else [args.model]
    summary: dict[str, dict[str, float]] = {}
    for name in targets:
        print(f"\n===== training {name} =====")
        if name == "cgan":
            best, dt = train_cgan(args.epochs, args.batch_size, args.lr_gan, dev)
        elif name == "cvae":
            best, dt = train_cvae(args.epochs, args.batch_size, args.lr, dev)
        elif name == "clstm":
            best, dt = train_seq_model("clstm", CLSTM, lstm_loss,
                                       args.epochs, args.batch_size, args.lr, dev)
        elif name == "ctransformer":
            best, dt = train_seq_model("ctransformer", CTransformer, transformer_loss,
                                       args.epochs, args.batch_size, args.lr, dev)
        else:
            raise ValueError(name)
        summary[name] = {"best_val_csr": best, "wall_seconds": dt}
        print(f"---- {name}: best val CSR = {best:.4f}, time = {dt:.1f}s")

    print("\n===== training summary =====")
    for n, v in summary.items():
        print(f"  {n:14s}  best_val_csr={v['best_val_csr']:.4f}  wall={v['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
