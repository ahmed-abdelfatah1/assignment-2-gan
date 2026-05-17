# DSAI 490 Assignment 2 — Conditional Date Generator
**Methodology, design decisions, and analysis** — Ahmed Abdelfatah

> Read this as a 1-page methodology summary of what was built, **why** each design
> decision was made, and what we learned. Full per-epoch curves are in
> `report/loss_curves.png`; per-model metrics are in `report/metrics.json`; sample
> outputs (10 successes + 5 failures per model) are in `report/sample_outputs.txt`.

---

## 1. Problem formulation — the insight that makes this tractable

We generate a date `d-m-yyyy` in `[1-1-1800, 31-12-2200]` that satisfies four
conditions: `[day_of_week] [month] [leap] [decade]`. A naïve framing treats this
as free-form string generation. **A closer look at the conditions** reveals they
leak almost the entire answer:

- `[MON]` directly fixes the month.
- `[DEC]` (three-digit decade) fixes 3 of 4 year digits.
- `[LEAP]` further filters acceptable years inside the decade.

So the model only has two free degrees of freedom — `day ∈ {1..31}` and
`year_last_digit ∈ {0..9}` — and the **hard** constraint is making
`(day, month, year)` land on the requested weekday. We exploit this directly:
output dimensionality, loss design, and architecture all follow from it.

## 2. Architecture decisions

We deliver **four PyTorch models**, all trained with hand-written loops (no
`.fit`, no Lightning). The hand-written `MultiHeadAttention` + `DecoderBlock` in
the transformer satisfies the "no `nn.Transformer` shortcut" constraint and
keeps the code readable.

| Model | Output | Why |
|---|---|---|
| **cGAN** (mandatory) | One 310-class softmax over `(day, year_digit)` pairs | A single joint softmax is the only representation that can express the non-Cartesian set of valid (day, year_digit) pairs for a given DOW (see §4). Generator: MLP `[256, 256, 256]`, Gumbel-softmax sampling, non-saturating G loss + one-sided label smoothing. Discriminator: MLP `[256, 256]`. |
| **cVAE** (from-the-course) | One 310-class softmax | Same representational reason as cGAN. Encoder/decoder MLPs hidden=256, z=32. β annealed 0→1 over 10 epochs. |
| **cLSTM** (outside-course) | Char sequence `dd-mm-yyyy` | Different paradigm — tests whether autoregressive char models can recover the same constraint without an explicit joint head. Hidden=256; cond projects to `(h0, c0)` **and** added to every input embedding (per-step injection). |
| **cTransformer** (outside-course) | Char sequence `dd-mm-yyyy` | Hand-written 4-layer decoder, d_model=128, 4 heads, FFN=256. Condition becomes the position-0 prepended token (attendable via causal mask) **and** is added to every char embedding (`cond_step_proj`) — the prepended-token-only variant failed because attention from the day positions back to position 0 wasn't strong enough early in training (see §4 v3.3). |

Conditions encode as one 62-dim one-hot concat: `DOW(7) + MON(12) + LEAP(2) + DEC(41)`.

## 3. Training & evaluation

- **Seed = 42** governs Python `random`, NumPy, and PyTorch (CPU + CUDA). The 80/10/10 train/val/test split is deterministic.
- **Optimiser:** Adam, lr=2e-4 for GAN (β₁=0.5, β₂=0.999), lr=1e-3 for the rest.
- **Early stopping** on val CSR with patience=10, min_delta=1e-3, min_epochs=20 (warmup — cGAN's aux loss plateaus at chance for ~17 epochs before breaking through, so a shorter warmup would kill it prematurely).
- **Inference: K=20 rejection sampling.** Per input row, we sample up to 20 dates and emit the first that satisfies the four conditions; if all 20 fail we emit the last sample. The `predict.py` CLI matches the assignment spec exactly: `python predict.py -i IN -o OUT`. With `--model all` (default) it writes four sibling files plus the spec-format `-o` (mirroring the transformer's predictions).

### Metrics — what we measure and why

We track four numbers (all implemented in `model/metrics.py`); each one captures something the others miss.

**(1) Condition Satisfaction Rate (CSR)** — *primary, headline metric.*
```python
CSR = mean(verify_date(generated_date, conds) for each (date, conds) pair)
```
`verify_date` is True iff the produced `d-m-yyyy` is a valid calendar date in `[1800, 2200]` **and** matches all four input conditions. **We deliberately do not use accuracy:** this is generation, not classification — for `(WED, JAN, False, 200)` there are ~30 valid Wednesdays in Jan 2001–2009, and accuracy would only credit the one training example's date, punishing the other 29 valid outputs and making a working model look broken. CSR is the right family of metric for tasks with multiple valid outputs (BLEU, pass@k, etc., are all variants).

**Strict K=1 reporting.** The CSR numbers in §5 are computed with **one sample per condition** — *not* the K=20 rejection-sampled number that `predict.py` emits. With raw CSR ≈ 0.98, `1 − 0.02^20 ≈ 100%` of input lines get a valid output via retries, which would inflate the apparent capability. The strict K=1 measures the model itself, not the rejection-sampling wrapper.

**(2) Per-condition breakdown** — *the diagnostic metric.*
```python
{"dow": hit_rate, "mon": hit_rate, "leap": hit_rate, "dec": hit_rate}
```
For each of the four conditions, the fraction of generated dates that match *that condition specifically*, ignoring the others. A date that doesn't parse counts as failing all four. **This is what made the v1 failure debuggable in 60 seconds:** CSR was 0.14 across all models, but the breakdown showed MON/LEAP/DEC at 99% and DOW alone at 14.3% = 1/7. That number immediately pointed at the DOW-ignoring failure mode rather than "the model is generally broken" — and drove the v3 AC-GAN aux loss design. Without the breakdown we'd have been guessing.

**(3) Diversity** — guards against the failure CSR alone misses.
```python
diversity = mean(unique_outputs / total_samples) over K=10 samples per cond tuple
```
A trivially "perfect" model could memorise one valid Wednesday in Jan 2001–2009 and emit it for *every* (WED, JAN, False, 200) input — that hits CSR=1.0 but is useless. Diversity ∈ [0, 1] checks that the model produces distinct valid dates across multiple samples for the same cond. cGAN/cVAE land at ≥ 0.95, which rules out mode collapse despite the discrete 310-class output and the strong AC-GAN aux pressure.

**(4) Per-epoch train_loss / train_aux / val_csr** — *training-time signal.*
Logged into `{model}_history.json`, plotted into `report/loss_curves.png`. We monitor losses *and* val CSR per epoch because **they can move independently**. The v1 cGAN/cVAE both had perfectly healthy decreasing train loss while val CSR sat at chance — train-loss optimisation ≠ task solved. Conversely, the v3 cGAN's loss kept slowly improving past epoch 30 while val CSR had plateaued at 0.98. This is exactly why **early stopping is on val_csr, not val_loss** (patience=10, min_epochs=20 warmup; the AC-GAN breakthrough doesn't fire until epoch 17–18, so a shorter warmup would terminate the run before it has a chance to work).

## 4. The day-of-week plateau and how we solved it

A clean negative result drove most of our design iterations.

### v1: independent (day, year_digit) softmaxes — **all four models at 0.14 ≈ 1/7**

Two independent softmaxes can only learn marginals; the day-of-week constraint
defines a *non-Cartesian* subset of `{1..31} × {0..9}` for any given (DOW, MON, DEC),
which independent marginals cannot represent. The 1/7 ≈ 14.3% number is exactly random chance over 7 weekdays.

### v2: joint 310-class softmax — cGAN/cVAE still at 0.14

Representational capacity is now sufficient, but the model **ignores the DOW dimension of the condition** — the textbook "mode-collapse-to-marginal" failure for conditional GANs (AC-GAN, Odena 2016) and "posterior-collapse-of-the-condition" failure for conditional VAEs (Lu et al. 2020). The three "leaky" conditions (MON/LEAP/DEC) are nailed at 99%+; DOW stays at chance.

### v3: AC-GAN-style auxiliary DOW loss

Precompute a `(12, 41, 310)` lookup mapping `(mon, dec, joint_idx) → DOW ∈ {0..6, INVALID=7}`. Per batch, compute `P(DOW) = softmax(predicted_joint) @ mask`, cross-entropy against the requested DOW, plus an invalid-mass penalty. This is **fully differentiable in soft-probability space** and provides a direct gradient pushing the model to honour DOW.

- **cGAN: 0.144 → 0.981.** Breakthrough at epoch 17–18 (aux loss drops from 1.95 to 0.84, val CSR jumps to 0.65). The clean AC-GAN signature.
- **cVAE: stayed at 0.156.** Despite getting the same aux loss.

### v3.1: why cVAE needed a different fix — and what worked

During cVAE training the encoder packs the target into `z`. The decoder receives **both** `z` (target info) and `cond` — the easiest path to minimise reconstruction CE is to read off `z` and ignore `cond`. The aux DOW loss can't break this because the decoder already has a target-shortcut. At inference time `z ~ N(0, I)` is uninformative, so the decoder collapses to the marginal over `cond` it never properly learned. **Two-pass training fixes this:** keep the posterior pass for reconstruction + KL, but compute the aux DOW loss on a **second decoder forward with `z ~ N(0, I)`** — denying the shortcut. **cVAE: 0.156 → 0.988.**

### v3.1 attempt for seq models — an instructive failure

A symmetric idea — add a `Linear(hidden, 7)` DOW classifier head on the LSTM's final hidden / Transformer's cond-token hidden — **did not work**. cLSTM stayed at 0.164, cTransformer at 0.153. The aux head learned to predict DOW from the hidden state (its CE went down), but the char output comes from a *separate* projection trained only by next-char CE. The DOW gradient never reached the char distribution sampled at inference. **It's the same failure pattern as v1 cVAE: a side-channel that satisfies the auxiliary objective without constraining the inference path.**

### v3.2: direct char-position aux DOW

Switch the seq tokenizer to **fixed-width** `dd-mm-yyyy` (zero-padded internally, leading zeros stripped on decode to match `data.txt`). With fixed positions:

- `char_logits[:, 0, :]` always predicts the day-tens digit ∈ {0..3}
- `char_logits[:, 1, :]` always predicts the day-ones digit ∈ {0..9}
- `char_logits[:, 9, :]` always predicts the year-last-digit ∈ {0..9}

We restrict each to digit-token logits, softmax, outer-product into a soft (day, year_digit) joint of dimension 310, and feed that to the **same** `aux_dow_loss_from_probs` machinery as cGAN/cVAE. Now the DOW gradient flows through the same logits sampled at inference — closing the side-channel gap.

**Results split**:

- **cLSTM: 0.140 → 0.697.** Clean breakthrough at epoch 5 (aux loss 1.95 → 0.23, val CSR 0.13 → 0.58), then a slow climb plateauing around 0.69. The aux signal locked in; the residual gap is from the soft-joint independence approximation (under teacher forcing, `P(d_tens)` and `P(d_ones)` are not actually independent, so the soft joint we feed into aux loss is a biased estimator).
- **cTransformer: 0.143 → 0.160 — no movement.** Aux loss never decreased (~1.94 throughout); char loss converged in 2 epochs to a position-conditioned marginal that ignores cond. Early stopping killed it at epoch 20.

### v3.3: cTransformer per-step cond injection (final seq fix)

cLSTM works in v3.2 because of a v3.1 design carry-over: cond is injected at *every* input position (`emb + cond_step_proj(cond)`), not just at `(h0, c0)`. cTransformer relied purely on a prepended cond token at position 0 + causal self-attention. With 4 layers and a large model, attention from the day positions back to position 0 evidently isn't strong enough early in training to carry DOW — so the model finds the easy attractor (position-conditioned marginal char distribution) and never escapes.

**Fix:** mirror cLSTM's pattern. Add `cond_step_proj: Linear(62, d_model)` to cTransformer and add `cond_step_proj(cond).unsqueeze(1)` to every char embedding before the cond-token concatenation. cond information is now *unavoidable* — it's baked into every input position, not just one that attention has to learn to use. Also bump `--aux-charpos-weight` default from 1.0 → 3.0 to give the aux loss more leverage against the marginal-only attractor.

*(Final cTransformer number to be filled in from `report/metrics.json` after the v3.3 retrain.)*

## 5. Results summary

| Model | v1 CSR | Final CSR | Wall time | Notes |
|---|---|---|---|---|
| cGAN          | 0.144 | **0.981** | ~640s | v3 AC-GAN aux DOW; breakthrough at epoch 17–18 |
| cVAE          | 0.139 | **0.988** | ~615s | v3.1 two-pass training (posterior recon + prior aux) |
| cLSTM         | 0.140 | **0.697** | ~338s | v3.2 fixed-width tokenizer + char-position aux DOW; per-step cond injection from v3.1 |
| cTransformer  | 0.143 | _pending v3.3 retrain (per-step cond injection + λ=3)_ | ~170s on v3.2 (early-stopped at chance) | |

Per-condition breakdown (cGAN/cVAE — from `report/metrics.json`): MON, LEAP, DEC all ≥ 99%; DOW now ≥ 98%. cLSTM's gap is the soft-joint independence approximation; it still nails the leaky three at ≥ 99%.

Diversity (mean unique outputs per cond tuple, K=10) is **≥ 0.95** for cGAN/cVAE — no mode collapse despite the discrete output.

## 6. Limitations & honest caveats

- **Decade 220 has only one valid year (2200) in the entire dataset** — a heavy long-tail. Every model struggles most on this tuple; failures concentrate there.
- **K=20 rejection sampling at inference inflates apparent reliability.** With raw CSR ≈ 0.98, `1 − 0.02^20 ≈ 100%` of input lines get a valid sample. The CSR numbers in §5 are the **strict K=1** model capability, which is what we report.
- **GAN training dynamics required a warmup** — for the first ~17 epochs aux loss stays at ~1.96 and val CSR walks around 0.10–0.15 noisily; the breakthrough is sharp once D and G reach a useful equilibrium. `--min-epochs 20` exists specifically for this.
- **We chose direct DOW supervision** over alternatives (FiLM at every layer, fixed-width-only without aux, distillation from cVAE) because it surgically addresses the diagnosed failure mode with minimal disruption to the standard training loops the assignment prescribes.
- **cLSTM plateau at ~0.70 is a soft-joint approximation ceiling, not a capacity ceiling.** Our `seq_aux_dow_loss` treats `P(day_tens)` and `P(day_ones)` as independent (outer product) when computing the soft joint over (day, year_digit). Under teacher forcing they are *not* independent — `P(day_ones)` is conditioned on the ground-truth `day_tens`. A true joint would need Gumbel-softmax-sampled `day_tens` fed back into `day_ones`' input. We accepted the biased gradient because the chance-level → ~70% jump confirms the diagnosis and a clean ~98% would require non-trivial architectural changes to the autoregressive sampling path.

## 7. Reproducing

```bash
conda env create -f environment.yml && conda activate dsai490-a2
python -m pytest tests/                                                      # all green
python model/train.py --model all                                            # ~30–40 min on T4 GPU
python model/evaluate.py                                                     # writes report/* artifacts
python model/predict.py -i data/example_input.txt -o predictions.txt         # spec invocation
```

A Colab notebook (`notebooks/train_colab.ipynb`) runs the full pipeline top-to-bottom on a fresh T4 runtime.

## 8. References

- Odena, Olah, Shlens. *Conditional Image Synthesis with Auxiliary Classifier GANs.* 2016 — AC-GAN, the v3 cGAN fix.
- Lu et al. *Mitigating Posterior Collapse in Strongly Conditioned VAEs.* OpenReview 2020 — diagnoses the v3.1 cVAE issue.
- Nguyen et al. *Beyond Vanilla VAEs.* arXiv 2306.05023, 2023 — posterior collapse in conditional / hierarchical settings.
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018 — considered as a stronger alternative; deferred.
