# Channel Estimation under Distribution Shift — Experiment 1 Design

**Project:** Deep Learning course project (CentraleSupélec) — *Can deep learning estimate wireless channels when propagation conditions differ from training?*
**Scope of this document:** project understanding, dataset facts, and the design of **Experiment 1** (MLP vs. time-frequency CNN, no ResNet yet).
**Status:** design only. No code written yet. Items marked **[VERIFY]** must be checked when the data is first loaded. Items marked **[DECISION]** are choices I made that you may want to change.

---

## 1. Project in one paragraph

A receiver needs the channel `H` to equalise the received signal (`Y = H·X + N`). It only gets a noisy, sparse estimate from pilots. We train neural networks to map an **imperfect channel estimate** (`X_input`) to the **true channel** (`Y_label`) and ask two questions from the proposal: *which model estimates best*, and *which stays robust when propagation conditions change*. "Change" is made concrete by **splitting the data by radio condition** (e.g. Doppler) rather than randomly, so the out-of-distribution (OOD) test set contains conditions never seen in training.

The proposal plans MLP, CNN and ResNet, a limited hyperparameter search (not a full grid), six evaluation metrics, and a 4-week plan. **Experiment 1 covers MLP and CNN only**, with a small, hand-picked set of hyperparameter combinations. Its results decide what Experiment 2 investigates (and when ResNet is added).

---

## 2. The ChanEst dataset (what the sources say)

**Sources:** Mendeley Data record (DOI 10.17632/8w3zjj8hzc.1, published 12 Feb 2026) and the dataset paper (Okoyeigbo et al., *Telecom* 7(3):65, 2026, DOI 10.3390/telecom7030065). Generation code is on GitHub (`obydelion/ChanEst-Dataset-Generation`, MATLAB).

| Item | Value |
|---|---|
| Nature | **Synthetic**, 3GPP-compliant, SISO downlink OFDM, 7 GHz carrier (FR3) |
| Grid | **612 subcarriers × 14 OFDM symbols** per sample (60 kHz SCS, 51 RBs) |
| Pilots | DM-RS Type 2, 408 pilot REs of 8568 (≈4.76 % density) |
| Channel model | 3GPP TDL-A … TDL-E, balanced across samples |
| `X_input` | LS estimate at pilots, then 2-D **linear interpolation** to the full grid (noisy, imperfect receiver view) |
| `Y_label` | Perfect channel response on the same grid |
| Tensor format | Real-valued, stored as `[K × L × C × N]` = `[612 × 14 × 2 × N]`; `C = 2` (real, imaginary) for SISO |
| Normalisation | **None applied** in the release (raw physical scale) |
| Sampling | Stratified random over SNR, delay spread, Doppler, TDL profile; independent permutations so the three are not correlated |
| SNR range | −10 to 30 dB |
| Delay spread range | 10 to 2000 ns |
| Doppler range | 5 to 5000 Hz (≈ 0.2–214 m/s at 7 GHz) |
| Metadata per sample | SNR, delay spread, Doppler, equivalent speed, TDL profile, input–label correlation / NMSE, input/label power |
| Formats | MATLAB v7.3 `.mat` and HDF5 `.h5` |
| Paper baseline | Lightweight fully-convolutional CNN, random 80/10/10 split; **≈ 2.25 dB average NMSE gain over plain LS-interpolation** |

The example rows on slide 4 of the proposal show metadata columns (`Corr, MSE, NMSE, SNR, DS, fd, v, Pin, Pgt, …`). These are used **only to split and stratify**, never as model input (as the proposal states).

**Facts not found in the sources, to be checked on download [VERIFY]:**
- Number of samples `N` in the released file (the paper's generation example uses 10k, but that is an example, not necessarily the release).
- Exact HDF5 key names for tensors and metadata, and the metadata units.
- Axis order after loading. MATLAB v7.3 files reverse axis order in HDF5, so `h5py` will probably give `(N, C, L, K) = (N, 2, 14, 612)`. That is already PyTorch's `N×C×H×W` layout with `H` = time and `W` = frequency.
- Scale of the tensors (the slide shows `Pin`/`Pgt` near 0 dB, i.e. roughly unit power, which would make normalisation mild).

---

## 3. Research questions and expectations for Experiment 1

Questions Experiment 1 must answer:

1. **Q1 — Architecture:** at comparable tuning effort, how do MLP and CNN compare on every metric, in-distribution (ID) and OOD?
2. **Q2 — Hyperparameter sensitivity:** which hyperparameters move the metrics most, and do they affect MLP and CNN differently?
3. **Q3 — Robustness:** how large is the ID → OOD degradation per model, and does any hyperparameter (e.g. dropout, capacity) reduce it?
4. **Q4 — Where to look next:** which regions of the hyperparameter space and which conditions (SNR, delay spread, TDL profile) deserve a follow-up?

Prior expectations (hypotheses, **not** results; the experiment can contradict them):
- The CNN should beat the MLP: the channel has strong local structure in time–frequency, and the CNN shares weights across the grid.
- The MLP has a structural handicap: with widths of 16–64 and a flattened 17,136-value input/output, it must squeeze the whole channel through a very narrow layer. Whether that bottleneck is harmful is an empirical question (the channel may be low-dimensional).
- Dropout may not help much: the task is dense regression and overfitting is not obvious with this many output values per sample.
- OOD error is expected to be worse than ID for everyone, partly because the *input itself* is worse at high Doppler (interpolation error grows). Hence the "gain over the unprocessed input" metric in §7.

---

## 4. Data protocol

### 4.1 Splits (Doppler shift)
Following slide 3, splits are by **radio condition**, not random. Experiment 1 uses a **single shift axis: Doppler**, to keep the story clean. SNR and delay-spread shifts are candidates for Experiment 2.

| Split | Rule | Purpose |
|---|---|---|
| Train | Doppler 5–2500 Hz, ~80 % of that pool | Fit weights |
| Validation | Doppler 5–2500 Hz, ~10 % of that pool | Early stopping, **all hyperparameter selection** |
| Test ID | Doppler 5–2500 Hz, ~10 % of that pool | Same conditions, unseen samples |
| *Buffer* | Doppler 2500–3000 Hz | **Discarded** (avoids a hard boundary / near-duplicate conditions) |
| Test OOD | Doppler 3000–5000 Hz | Never used for training, tuning or model selection |

Notes:
- With uniform Doppler sampling, roughly half the data is in the train pool, 10 % in the buffer and 40 % in OOD **[VERIFY after load]**. The slide says ranges are "illustrative and will be adapted to the real data, with no leakage": the cut points above are parameters in a config file.
- Each sample is an independent channel realisation, so a random ID split inside the Doppler range does not leak across samples. Fix the split seed and save sample indices to disk.
- Metadata columns are used for splitting and per-condition reporting only.

### 4.2 Input/output and normalisation
- Input and target: `2 × 14 × 612` tensors (real/imag channels, time, frequency).
- **[DECISION] Per-sample scaling:** divide the input and the label by the **input's** RMS (never the label's, which would leak ground truth at test time). Metrics are computed after undoing the scaling so RMSE/MAE remain in physical units. If the load check shows tensors already have roughly unit power, this step can be dropped.
- Statistics of any kind are computed on the training split only.

### 4.3 Reference baselines (no training)
- **Identity / LS-interp baseline:** `Ĥ = X_input`. Every learned model must be compared with this on every split. It is the same "LS + linear interpolation" the paper compares against.
- Sanity check: a well-behaved CNN should beat this baseline by a margin of the same order as the paper's ≈ 2.25 dB average on ID data. If it does not, suspect a bug before suspecting the method.

---

## 5. Models

Both models take the same input and return a `2 × 14 × 612` estimate `Ĥ`, trained with MSE loss on the (scaled) tensors.

### 5.1 MLP (flattened full grid)
- Flatten input to 17,136 values → `L` hidden layers of width `W`, ReLU, dropout `p` after each hidden activation → linear output of 17,136 → reshape.
- `L` = number of hidden layers, `W` = neurons per hidden layer (the proposal's "filters / neurons").
- Parameter count is dominated by the first and last layers (≈ 2.2 M at `W = 64`, ≈ 0.6 M at `W = 16`).
- **[DECISION]** I kept the proposal's literal definition (full-grid MLP, widths 16/32/64). The bottleneck is then part of what is being tested. If you would rather have a "fair fight" MLP, the alternative is a **patch-wise MLP** (shared weights applied to non-overlapping time–frequency patches). That is a one-line change in the model file and could be added as a third model in Experiment 2.

### 5.2 CNN (time–frequency, 2-D)
- Input treated as a 2-channel image with `H = 14` (time), `W = 612` (frequency).
- `L` convolutional layers, 3×3 kernels, `F` filters, "same" padding, stride 1, **no pooling** (the grid is small and fine detail matters), ReLU, element-wise dropout `p` after each activation. Final 3×3 conv with 2 output channels, no activation.
- Parameter count is small (≈ 20 k at `L = 3`, `F = 32`), so it is orders of magnitude smaller than the MLP. Parameter counts are reported in the results.
- Receptive field is small (5×5 up to 9×9 pixels at `L` = 2–4). That is deliberate: it matches the proposal's depth range and keeps the CNN a lightweight baseline like the paper's.

### 5.3 Shared design choice
- **[DECISION] Residual output:** `Ĥ = X_input + f(X_input)` for both models, so the network learns a correction to the LS-interpolated estimate rather than the whole channel. This is still "predict Ĥ" as in the proposal, makes the identity baseline the starting point, and tends to make training easier. It applies identically to both architectures, so the comparison stays fair. Easy to switch off.

---

## 6. Hyperparameters and run matrix

### 6.1 Proposal grid (slide 5)
| Hyperparameter | Values |
|---|---|
| Learning rate | 0.01, 0.001, 0.0001 |
| Batch size | 32, 64, 128 |
| Layers `L` | 2, 3, 4 |
| Filters / neurons `F`, `W` | 16, 32, 64 |
| Dropout | 0, 0.2, 0.5 |

Per architecture that is 3⁵ = 243 combinations (486 total): too many, as the proposal acknowledges.

### 6.2 Selected design: centre point + one-factor-at-a-time + a few interactions
Both architectures use **identical hyperparameter values** in every run, so differences come from the architecture and not from the tuning budget.

**Centre point (C0):** lr = 0.001, batch = 64, `L` = 3, `F`/`W` = 32, dropout = 0.2 (the middle value of every range).

| Run | Change vs. centre | Why it is in the design |
|---|---|---|
| R00 | none (centre) — **3 seeds** | Baseline and an estimate of seed-to-seed noise |
| R01 | lr = 0.01 | Learning-rate sensitivity (high) |
| R02 | lr = 0.0001 | Learning-rate sensitivity (low) |
| R03 | batch = 32 | Batch-size sensitivity |
| R04 | batch = 128 | Batch-size sensitivity |
| R05 | `L` = 2 | Depth |
| R06 | `L` = 4 | Depth |
| R07 | `F`/`W` = 16 | Capacity |
| R08 | `F`/`W` = 64 | Capacity |
| R09 | dropout = 0 | Regularisation (none) |
| R10 | dropout = 0.5 | Regularisation (strong) |
| R11 | "Lean": `L` = 2, `F`/`W` = 16, dropout = 0 | Smallest model, no regularisation |
| R12 | "Big, no reg": `L` = 4, `F`/`W` = 64, dropout = 0 | Max capacity — overfitting / OOD risk |
| R13 | "Big, regularised": `L` = 4, `F`/`W` = 64, dropout = 0.5 | Does regularisation rescue large models, especially OOD? |
| R14 | "Fast": lr = 0.01, batch = 128 | High-step, large-batch corner |
| R15 | "Slow": lr = 0.0001, batch = 32 | Low-step, small-batch corner (unconstrained by the epoch cap, see §8) |

R01–R10 give a clean main effect for each hyperparameter around a shared reference. R11–R15 are the few "interesting" interactions (capacity × regularisation, learning rate × batch size).

**Totals:** 16 configurations × 2 architectures = 32 configurations; centre re-run with 2 extra seeds per architecture → **36 training runs**. A single 1-epoch timing run should be done first to confirm this fits the compute budget (not yet measured).

### 6.3 Fixed settings (same for all runs)
- Optimiser Adam (default betas), no weight decay, no LR scheduler (the learning rate is itself a studied hyperparameter).
- Loss: MSE on scaled tensors. Gradient-norm clipping at 1.0 (so the lr = 0.01 runs fail gracefully rather than diverging).
- Max 40 epochs, early stopping on **validation NMSE** with patience 5, best checkpoint restored.
- Seed 0 for all runs; seeds 0, 1, 2 for R00.
- fp32 (keeps the inference-time measurement meaningful).

---

## 7. Metrics

All six proposal metrics, computed on **validation, test ID and test OOD**, after undoing per-sample scaling.

| Metric | Definition used |
|---|---|
| **RMSE** | √ mean over samples and all real/imag elements of `(Ĥ − H)²` |
| **NMSE** | Per sample `‖Ĥ − H‖² / ‖H‖²`, averaged over samples in linear scale, reported in dB (matches the dataset's per-sample NMSE column). Median NMSE (dB) is also logged because the mean is dominated by bad samples |
| **MAE** | Mean of `|Ĥ − H|` over all elements |
| **R²** | `1 − Σ(Ĥ − H)² / Σ(H − H̄)²` over all test elements |
| **Inference time** | Median ms per estimate at batch size 1 (20 warm-up, 200 timed passes, device synchronised), plus ms/sample at batch 128. Same hardware for every run |
| **Worst-condition NMSE** | **[DECISION]** NMSE (dB) in the hardest bin of a 4 (SNR) × 3 (delay spread) grid, ignoring bins with fewer than 30 samples; computed separately for ID and OOD. Alternative: NMSE on the worst 10 % of samples ranked by the unprocessed input's NMSE |

Additional logged quantities (not among the six, but needed to interpret them):
- **Gain over identity baseline (dB):** `NMSE_identity − NMSE_model` on the same split. This separates "the model is good" from "the OOD data is simply harder".
- **ID → OOD gap (dB):** `NMSE_OOD − NMSE_ID`.
- Parameter count, training time, epochs to convergence.
- NMSE broken down by SNR bin, delay-spread bin, Doppler bin and TDL profile.

---

## 8. Analysis plan

**Selection rule.** Hyperparameters and "best configuration" are chosen on **validation NMSE only**. OOD results are reported but never used for selection (otherwise OOD would no longer be unseen).

**Noise floor.** Use the three R00 seeds to estimate the standard deviation of NMSE (dB) per architecture. Differences between runs smaller than about 2 σ are reported as "not distinguishable".

**Outputs and plots** (mapped to slide 6):
1. Results table: one row per run, all metrics on ID and OOD, plus parameter count and timing.
2. NMSE vs. SNR for the best MLP, best CNN and the identity baseline — ID and OOD side by side.
3. Main-effect plot per hyperparameter (R01–R10 vs. R00), one line per architecture, effect expressed as ΔNMSE in dB.
4. Architecture comparison: bar chart of all six metrics for the best config of each architecture.
5. Inference time vs. NMSE (accuracy/cost trade-off), with parameter count as marker size.
6. Interaction runs (R11–R15) summarised against the main-effect predictions.
7. Training curves for the best runs and any run that diverged or stopped at the epoch cap.

**Known caveats to keep in mind when reading results**
- R15 (lr 0.0001, batch 32) may hit the 40-epoch cap before converging; if so it is flagged as "under-trained", not as a bad hyperparameter.
- Batch size changes the number of optimiser steps per epoch, so it is partly confounded with training budget.
- One-factor-at-a-time cannot reveal interactions beyond R11–R15. This is deliberate for a first pass.
- Only Doppler shift is tested; conclusions about "robustness" apply to that axis.
- A single synthetic simulator (3GPP TDL at 7 GHz): results say nothing about real measured channels.

**Decision rules for Experiment 2** (to be filled in from the results): take the top few validation configurations per architecture, run them on 3 seeds, and add ResNet; consider SNR / delay-spread shifts and a patch-wise MLP if the full-grid MLP clearly suffers from its bottleneck.

---

## 9. Planned code structure (next step, not written yet)

```
chanest/
  config.yaml        # split cut points, fixed settings, run matrix
  data.py            # load .h5, build splits (save indices), scaling, Dataset
  models.py          # MLP, CNN (+ residual wrapper)
  train.py           # one run: train, early stop, save checkpoint + log
  evaluate.py        # six metrics + per-condition breakdown + timing
  run_all.py         # loops over the run matrix, writes results.csv
  analyze.ipynb      # tables and plots from results.csv
```

Course constraint: minimal coding effort for you. The intent is that you run `run_all.py` once and open the analysis notebook.

---

## 10. Open points for you to confirm

1. **MLP form:** full-grid flatten (default, faithful to the proposal) or patch-wise?
2. **Residual output** (`Ĥ = X + f(X)`) for both models: keep or drop?
3. **Doppler cut points:** train ≤ 2500 Hz, buffer 2500–3000, OOD ≥ 3000 — fine to adopt as is?
4. **Worst-condition NMSE definition:** the SNR × delay-spread bin version above, or the "worst 10 % of samples" version?
5. **Compute:** what will the runs execute on (GPU model, or Colab)? This sets whether 36 runs is comfortable or needs trimming.
6. **Dataset size:** use the full release or a capped subset (decide after seeing `N`)?

---

## Sources

- Project proposal slides (Nov 2026): problem statement, splits, dataset, hyperparameters, metrics, plan.
- Mendeley Data: *ChanEst Dataset for Deep Learning-Based 6G Channel Estimation*, DOI 10.17632/8w3zjj8hzc.1.
- Okoyeigbo, Deng, Sheriff, Jeremiah, Shobayo, *ChanEst Dataset: A Reconfigurable Framework and Benchmark for Deep Learning–Based 6G Channel Estimation*, Telecom 7(3):65, 2026, DOI 10.3390/telecom7030065.
