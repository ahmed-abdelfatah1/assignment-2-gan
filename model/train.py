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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import config
from model.constraints import INVALID_DOW, dow_lookup_tensor
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


class _EarlyStopper:
    """Stops training when val_csr hasn't improved by min_delta for `patience` epochs.
    The first `min_epochs` epochs are exempt (warmup) — important for cGAN, whose
    aux-DOW loss often plateaus at chance for ~17 epochs before breaking through.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-3, min_epochs: int = 20) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best = -1.0
        self.epochs_since_improvement = 0
        self.best_epoch = 0

    def step(self, epoch: int, val_metric: float) -> bool:
        """Update with the latest val metric. Returns True iff training should stop."""
        if val_metric > self.best + self.min_delta:
            self.best = val_metric
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        if epoch + 1 < self.min_epochs:
            return False
        return self.epochs_since_improvement >= self.patience


def _cond_indices(cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (dow_idx, mon_idx, dec_idx) from a (B, 62) one-hot cond batch."""
    n_dow = len(config.DOW_TOKENS)
    n_mon = len(config.MONTH_TOKENS)
    n_leap = len(config.LEAP_TOKENS)
    dow_idx = cond[:, :n_dow].argmax(dim=-1)
    mon_idx = cond[:, n_dow:n_dow + n_mon].argmax(dim=-1)
    dec_off = n_dow + n_mon + n_leap
    dec_idx = cond[:, dec_off:dec_off + len(config.DECADE_TOKENS)].argmax(dim=-1)
    return dow_idx, mon_idx, dec_idx


def aux_dow_loss_from_probs(probs: torch.Tensor, cond: torch.Tensor,
                            dow_lut: torch.Tensor,
                            invalid_weight: float = 1.0) -> torch.Tensor:
    """Same machinery as aux_dow_loss but consumes joint probabilities directly
    (used by seq_aux_dow_loss where probs come from combining char positions)."""
    dow_idx, mon_idx, dec_idx = _cond_indices(cond)
    dow_map = dow_lut[mon_idx, dec_idx]                                 # (B, 310)
    mask = F.one_hot(dow_map, num_classes=INVALID_DOW + 1).float()      # (B, 310, 8)
    dow_probs = torch.einsum("bj,bjk->bk", probs, mask)                 # (B, 8)
    eps = 1e-8
    target_prob = dow_probs.gather(1, dow_idx.unsqueeze(1)).squeeze(1)  # (B,)
    ce = -(target_prob + eps).log().mean()
    invalid_mass = dow_probs[:, INVALID_DOW].mean()
    return ce + invalid_weight * invalid_mass


def aux_dow_loss(logits: torch.Tensor, cond: torch.Tensor,
                 dow_lut: torch.Tensor, invalid_weight: float = 1.0) -> torch.Tensor:
    """AC-GAN-style auxiliary loss for cGAN/cVAE: takes 310-class joint logits."""
    return aux_dow_loss_from_probs(F.softmax(logits, dim=-1), cond, dow_lut, invalid_weight)


def _digit_token_ids(device: torch.device) -> torch.Tensor:
    return torch.tensor([config.CHAR_TO_IDX[str(d)] for d in range(10)],
                        dtype=torch.long, device=device)


def seq_aux_dow_loss(char_logits: torch.Tensor, cond: torch.Tensor,
                     dow_lut: torch.Tensor) -> torch.Tensor:
    """Direct DOW supervision for fixed-width `dd-mm-yyyy` seq models.

    char_logits shape (B, T, V). Under fixed-width tokenisation the input seq
    is `[BOS, d1, d2, '-', m1, m2, '-', y1, y2, y3, y4]` (length 11) and
    char_logits[:, k, :] predicts the (k+1)-th char of the full sequence.
    So:
      - char_logits[:, 0, :] predicts day-tens (d1, ∈ {0..3})
      - char_logits[:, 1, :] predicts day-ones (d2, ∈ {0..9})
      - char_logits[:, 9, :] predicts year-last-digit (y4, ∈ {0..9})

    Restrict each to digit-token logits, softmax, outer-product into a soft
    joint over (day_idx ∈ {0..30}, year_digit ∈ {0..9}), apply aux DOW loss.
    """
    digit_ids = _digit_token_ids(char_logits.device)                    # (10,)
    p_d_tens = F.softmax(char_logits[:, 0, digit_ids[:4]], dim=-1)       # (B, 4) — tens ∈ {0..3}
    p_d_ones = F.softmax(char_logits[:, 1, digit_ids], dim=-1)           # (B, 10)
    p_y9     = F.softmax(char_logits[:, 9, digit_ids], dim=-1)           # (B, 10)

    # Joint over day-number ∈ {0..39}; we only keep {1..31}, i.e. day_idx {0..30}.
    joint_40 = (p_d_tens.unsqueeze(-1) * p_d_ones.unsqueeze(-2)).reshape(-1, 40)
    p_day31 = joint_40[:, 1:32]                                          # (B, 31)
    p_day31 = p_day31 / p_day31.sum(-1, keepdim=True).clamp_min(1e-8)

    p_joint = (p_day31.unsqueeze(-1) * p_y9.unsqueeze(-2)).reshape(-1, config.JOINT_DIM)
    return aux_dow_loss_from_probs(p_joint, cond, dow_lut)


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
               aux_weight: float = 2.0, val_subset: int = 4096,
               patience: int = 10, min_delta: float = 1e-3,
               min_epochs: int = 20) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("vec")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    dow_lut = dow_lookup_tensor().to(device)
    stopper = _EarlyStopper(patience=patience, min_delta=min_delta, min_epochs=min_epochs)

    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        G.train(); D.train()
        sum_g = sum_d = sum_aux = 0.0; nb = 0
        for cond, day_idx, yr_idx in tqdm(loader, desc=f"cgan e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); day_idx = day_idx.to(device); yr_idx = yr_idx.to(device)
            real = torch.stack([joint_onehot(int(d) * config.YEAR_DIGIT_DIM + int(y))
                                for d, y in zip(day_idx, yr_idx)]).to(device)
            fake = G.sample_onehot(cond).detach()
            opt_d.zero_grad()
            loss_d = d_loss(D(real, cond), D(fake, cond))
            loss_d.backward(); opt_d.step()
            # Generator step: adversarial + auxiliary DOW supervision (AC-GAN style)
            z = torch.randn(cond.shape[0], G.noise_dim, device=device)
            logits = G(z, cond)
            fake_g = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
            opt_g.zero_grad()
            adv = g_loss(D(fake_g, cond))
            aux = aux_dow_loss(logits, cond, dow_lut)
            loss_g = adv + aux_weight * aux
            loss_g.backward(); opt_g.step()
            sum_g += float(adv.item()); sum_d += float(loss_d.item()); sum_aux += float(aux.item()); nb += 1
        G.eval()
        val_csr = _val_csr_vec(G.sample_indices, val_small, device)
        history.append({"epoch": ep + 1, "train_g": sum_g / max(1, nb),
                        "train_d": sum_d / max(1, nb),
                        "train_aux": sum_aux / max(1, nb), "val_csr": val_csr})
        print(f"[cgan] epoch {ep+1}/{epochs}  G={sum_g/max(1,nb):.4f}  D={sum_d/max(1,nb):.4f}  aux={sum_aux/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights("cgan", {"G": G.state_dict(), "D": D.state_dict()})
        if stopper.step(ep, val_csr):
            print(f"[cgan] early stop at epoch {ep+1} (best {stopper.best:.3f} at epoch {stopper.best_epoch+1})")
            break
    if best_csr == 0.0:  # save final state even if no improvement seen
        _save_weights("cgan", {"G": G.state_dict(), "D": D.state_dict()})
    _save_history("cgan", history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# cVAE
# ---------------------------------------------------------------------------

def train_cvae(epochs: int, batch_size: int, lr: float, device: torch.device,
               aux_weight: float = 2.0, val_subset: int = 4096,
               patience: int = 10, min_delta: float = 1e-3,
               min_epochs: int = 20) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("vec")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    M = CVAE().to(device)
    opt = torch.optim.Adam(M.parameters(), lr=lr)
    dow_lut = dow_lookup_tensor().to(device)
    stopper = _EarlyStopper(patience=patience, min_delta=min_delta, min_epochs=min_epochs)
    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        beta = min(1.0, (ep + 1) / 10.0)
        M.train()
        sum_loss = sum_aux = 0.0; nb = 0
        for cond, day_idx, yr_idx in tqdm(loader, desc=f"cvae e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); day_idx = day_idx.to(device); yr_idx = yr_idx.to(device)
            target = torch.stack([joint_onehot(int(d) * config.YEAR_DIGIT_DIM + int(y))
                                  for d, y in zip(day_idx, yr_idx)]).to(device)
            opt.zero_grad()
            # Posterior pass: encoder + decoder(z_post) for reconstruction CE + KL.
            mu, logvar = M.encode(target, cond)
            z_post = M.reparam(mu, logvar)
            logits_post = M.decode(z_post, cond)
            loss_main, _parts = vae_loss(logits_post, day_idx, yr_idx, mu, logvar, beta=beta)
            # Prior pass: decoder(z~N(0,I)) for DOW aux. Decoder gets no target info
            # via z, so it must use cond's DOW — matches inference distribution.
            z_prior = torch.randn(cond.shape[0], M.z_dim, device=device)
            logits_prior = M.decode(z_prior, cond)
            aux = aux_dow_loss(logits_prior, cond, dow_lut)
            loss = loss_main + aux_weight * aux
            loss.backward(); opt.step()
            sum_loss += float(loss_main.item()); sum_aux += float(aux.item()); nb += 1
        M.eval()
        val_csr = _val_csr_vec(M.sample_indices, val_small, device)
        history.append({"epoch": ep + 1, "beta": beta, "train_loss": sum_loss / max(1, nb),
                        "train_aux": sum_aux / max(1, nb), "val_csr": val_csr})
        print(f"[cvae] epoch {ep+1}/{epochs}  beta={beta:.2f}  loss={sum_loss/max(1,nb):.4f}  aux={sum_aux/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights("cvae", M.state_dict())
        if stopper.step(ep, val_csr):
            print(f"[cvae] early stop at epoch {ep+1} (best {stopper.best:.3f} at epoch {stopper.best_epoch+1})")
            break
    if best_csr == 0.0:
        _save_weights("cvae", M.state_dict())
    _save_history("cvae", history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# cLSTM / cTransformer share a loop
# ---------------------------------------------------------------------------

def train_seq_model(name: str, build_model, loss_fn, epochs: int, batch_size: int,
                    lr: float, device: torch.device, val_subset: int = 2048,
                    aux_charpos_weight: float = 1.0,
                    patience: int = 10, min_delta: float = 1e-3,
                    min_epochs: int = 20) -> tuple[float, float]:
    train_ds, val_ds, _ = build_datasets("seq")
    val_small = _subset(val_ds, val_subset)
    loader = _make_loader(train_ds, batch_size, device)

    M = build_model().to(device)
    opt = torch.optim.Adam(M.parameters(), lr=lr)
    dow_lut = dow_lookup_tensor().to(device)
    stopper = _EarlyStopper(patience=patience, min_delta=min_delta, min_epochs=min_epochs)
    history: list[dict[str, float]] = []
    best_csr = 0.0
    t0 = time.time()
    for ep in range(epochs):
        M.train()
        sum_loss = sum_aux = 0.0; nb = 0
        for cond, seq in tqdm(loader, desc=f"{name} e{ep+1}/{epochs}", leave=False):
            cond = cond.to(device); seq = seq.to(device)
            opt.zero_grad()
            char_logits = M(cond, seq[:, :-1])
            loss_main = loss_fn(char_logits, seq[:, 1:])
            loss_aux = seq_aux_dow_loss(char_logits, cond, dow_lut)
            loss = loss_main + aux_charpos_weight * loss_aux
            loss.backward(); opt.step()
            sum_loss += float(loss_main.item()); sum_aux += float(loss_aux.item()); nb += 1
        M.eval()
        val_csr = _val_csr_seq(lambda c: M.sample(c, temperature=1.0), val_small, device)
        history.append({"epoch": ep + 1, "train_loss": sum_loss / max(1, nb),
                        "train_aux": sum_aux / max(1, nb), "val_csr": val_csr})
        print(f"[{name}] epoch {ep+1}/{epochs}  loss={sum_loss/max(1,nb):.4f}  aux={sum_aux/max(1,nb):.4f}  val_csr={val_csr:.3f}")
        if val_csr > best_csr:
            best_csr = val_csr
            _save_weights(name, M.state_dict())
        if stopper.step(ep, val_csr):
            print(f"[{name}] early stop at epoch {ep+1} (best {stopper.best:.3f} at epoch {stopper.best_epoch+1})")
            break
    if best_csr == 0.0:
        _save_weights(name, M.state_dict())
    _save_history(name, history)
    return best_csr, time.time() - t0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="all",
                   help="Comma-separated subset of {cgan,cvae,clstm,ctransformer} or 'all'. "
                        "Examples: --model clstm | --model clstm,ctransformer | --model all")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr-gan", type=float, default=2e-4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux-dow-weight", type=float, default=2.0,
                   help="cGAN/cVAE auxiliary DOW loss weight (0 disables; v3 default 2.0)")
    p.add_argument("--aux-charpos-weight", type=float, default=3.0,
                   help="cLSTM/cTransformer direct char-position DOW loss weight "
                        "(v3.3 default 3.0; v3.2 used 1.0 but cTransformer needed more "
                        "pull to escape the marginal-only minimum)")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience on val_csr (epochs of no improvement)")
    p.add_argument("--min-delta", type=float, default=1e-3,
                   help="Minimum val_csr improvement to reset patience")
    p.add_argument("--min-epochs", type=int, default=20,
                   help="Warmup before early stopping is allowed to trigger")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()

    config.set_seed(config.SEED)
    dev = config.device(args.device)
    print(f"using device: {dev}")
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        targets = list(config.MODEL_NAMES)
    else:
        targets = [t.strip() for t in args.model.split(",") if t.strip()]
        unknown = [t for t in targets if t not in config.MODEL_NAMES]
        if unknown:
            raise ValueError(f"unknown model(s): {unknown}; valid: {list(config.MODEL_NAMES)}")
    summary: dict[str, dict[str, float]] = {}
    for name in targets:
        print(f"\n===== training {name} =====")
        es_kw = dict(patience=args.patience, min_delta=args.min_delta, min_epochs=args.min_epochs)
        if name == "cgan":
            best, dt = train_cgan(args.epochs, args.batch_size, args.lr_gan, dev,
                                  aux_weight=args.aux_dow_weight, **es_kw)
        elif name == "cvae":
            best, dt = train_cvae(args.epochs, args.batch_size, args.lr, dev,
                                  aux_weight=args.aux_dow_weight, **es_kw)
        elif name == "clstm":
            best, dt = train_seq_model("clstm", CLSTM, lstm_loss,
                                       args.epochs, args.batch_size, args.lr, dev,
                                       aux_charpos_weight=args.aux_charpos_weight, **es_kw)
        elif name == "ctransformer":
            best, dt = train_seq_model("ctransformer", CTransformer, transformer_loss,
                                       args.epochs, args.batch_size, args.lr, dev,
                                       aux_charpos_weight=args.aux_charpos_weight, **es_kw)
        else:
            raise ValueError(name)
        summary[name] = {"best_val_csr": best, "wall_seconds": dt}
        print(f"---- {name}: best val CSR = {best:.4f}, time = {dt:.1f}s")

    print("\n===== training summary =====")
    for n, v in summary.items():
        print(f"  {n:14s}  best_val_csr={v['best_val_csr']:.4f}  wall={v['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
