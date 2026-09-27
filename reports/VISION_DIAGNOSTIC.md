# WAVE-3b vision diagnostic — why the JevX choice-head stack trails the MLP probe on Imagenette

APPENDED 2026-09-26 (WAVE-3b). Every number below was produced by executed
code and is copied from
`benchmarks/results/vision_diagnostic_features.json`,
`benchmarks/results/vision_diagnostic_grid.json` and
`benchmarks/results/vision_diagnostic_summary.md`, written by
`benchmarks/run_vision_diagnostic.py` on this host. Nothing is estimated,
interpolated, extrapolated or hand-typed. Language: *measured* / *observed* only.

This is a **benchmark-only** pass: nothing under `jevx/`, `docs/`,
`reports/MODALITY_BENCHMARKS.md`, `reports/MULTI_QUESTION.md`,
`reports/HEAD_ABLATION_MATRIX.md` or `benchmarks/run_head_ablation.py` was
modified. The V1–V6 variants are assembled in the benchmark script from the
shipped public surface only.

---

## 0. What was executed

| | |
|---|---|
| feature cache | `benchmarks/cache/vision_diagnostic/deit_tiny_cls_features.npz`, 1,151,599 B, sha256 `f3861d4b0124481a584183977608c665e4c99ffa58992d1f1101adfd72466e94` |
| featurization | 1,600 images, frozen backbone, `pooling_method="cls"`, batch 32, **267.59 s** forward; total phase **362.34 s** |
| grid | 13 cells (V0–V6 + V0L/V2m/V2I + the 2 reference probes) × 5 seeds (0,1,2,3,4) = 65 cell-runs |
| failures | **0** (`"failures": {}`) |
| sum of per-cell wall time | **2,375.4 s** (V0 966.5 s, V0L 593.7 s, V1 119.9 s, V2 111.9 s, V3 106.5 s, V4 129.4 s, V2m 78.6 s, V5a 75.4 s, V5b 64.9 s, V6 58.2 s, V2I 62.5 s, R1 3.5 s, R2 4.4 s) |
| eval invariants | 780 batches checked, `max|Σp − 1| = 4.17e-7`, `p ∈ [1.97e-23, 1.0]`, `K = 10` on every batch |
| wrapper equivalence | `VisionHead(conditioning=True)` vs the shipped `benchmarks.bench_common.FeaturizedJevX`, canonical model: **max_abs_diff = 0.0** |
| environment | Windows 11 10.0.26100, Python 3.14.3, torch 2.14.0+cpu (`torch_threads=2`, `cuda_available=False`), transformers 5.16.1, datasets 4.4.1, numpy 2.3.5, fp32 |
| host note | other agents shared the 4 logical CPUs throughout; the per-cell wall times are contended and are **not** a clean latency measurement (only their ratios are used below, and they are labelled as such) |

### Ids and shas (verified this run, not copied from the earlier report)

| role | id | revision | evidence |
|---|---|---|---|
| backbone | `facebook/deit-tiny-patch16-224` | `b3428f18dcc7b543470d07f14b4a4157815d1880` | API sha == `backbone_config["_commit_hash"]` after load; 5,561,472 frozen params measured; load report again shows `pooler.dense.{weight,bias}` **MISSING** (so `pooling_method="cls"` remains the correct choice, §4 of MODALITY_BENCHMARKS.md) |
| dataset | `frgfm/imagenette` | `a8a496123b40fa739da2acf9b0dae339d30c7bae`, config `imagenette2-160` | archive `imagenette2-160.tgz`, 99,003,388 B, sha256 `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5` (cache hit, re-hashed); `train.txt` sha256 `971de5a4…967`, `val.txt` sha256 `fb0f7c7d…0221`; 9,469 train / 3,925 val members indexed, 0 unmatched |
| question | "Which category does this image belong to?" | — | 10 options = the loader script's `ClassLabel` names in WNID order |
| entry points | `python benchmarks/run_vision_diagnostic.py --phase featurize` / `--phase grid` / `--phase summarize` | | |

---

## 1. Featurization protocol and split verification (TASK 0)

The deit features were **not** cached anywhere in the repo before this pass
(`benchmarks/cache/head_quality/` held only `ag_news`, `ravdess` and `sst2`).
They were produced here with the published EXP-V path, reused from
`benchmarks/run_modality.py` without modification:

* `pooling_method="cls"` (the checkpoint ships no pooler — the load report in
  §0 marks `pooler.dense.*` MISSING, so the "model-native" `pooler_output`
  would be a random projection, different on every reload);
* frozen backbone, `torch.no_grad()`, featurize batch 32, one pass over the
  1,600 selected images;
* splits built with `benchmarks.bench_common.make_splits` /
  `stratified_indices` at `split_seed=0`: train 800 + calibration 300 carved
  from the official train split, test = 500 stratified from the **official
  validation** split, never fitted on;
* the three splits are asserted disjoint at construction
  (`len(set(paths)) == len(paths)`, asserted in the script).

**Split reproduction check against the published protocol** (per-class counts):

| split | n | this run | MODALITY_BENCHMARKS.md §3 | match |
|---|---|---|---|---|
| train | 800 | 81, 81, 84, 72, 80, 81, 81, 79, 80, 81 | 72–84 per class; identical vector in `vision_jevx_quality.json` | **exact** |
| calibration | 300 | 31, 30, 32, 27, 30, 30, 30, 30, 30, 30 | 27–32 per class; identical vector | **exact** |
| test | 500 | 49, 50, 46, 49, 52, 50, 50, 53, 51, 50 | 46–53 per class; identical vector | **exact** |

The split was not changed. Feature matrix: 1,600 × 192, float32, all finite,
`|x|` mean 1.0796.

### Class-imbalance quantification (for V6)

| population | n | min | max | max/min | max dev. from uniform | entropy deficit (nats) | χ² vs uniform (df 9) |
|---|---|---|---|---|---|---|---|
| official train pool | 9,469 | 858 | 993 | 1.157 | 0.094 | 0.000632 | 11.734 |
| train split | 800 | 72 | 84 | 1.167 | 0.100 | 0.000687 | 1.075 |
| calibration split | 300 | 27 | 32 | 1.185 | 0.100 | 0.000790 | 0.467 |
| **official val pool** | 3,925 | 357 | 419 | **1.174** | 0.090 | 0.000778 | 6.055 |
| **test split** | 500 | 46 | 53 | **1.152** | 0.080 | 0.000644 | **0.640** |

The Imagenette test subsample is imbalanced **by construction** (the official
validation split itself is: 357 "cassette player" vs 419 "parachute", ratio
1.174, and a 500-row stratified subsample cannot undo a 1.17:1 population
ratio). But the resulting test imbalance is **not statistically
distinguishable from uniform**: χ² = 0.64 on 9 df, against a 5 % critical
value of 16.92. The normalized entropy deficit is 2.8e-4 — three orders of
magnitude below the capacity-sweep effects measured in §3. The inverse-
frequency weights V6 actually applies are therefore almost flat:
`w = 0.9877, 0.9877, 0.9524, 1.1111, 1.0000, 0.9877, 0.9877, 1.0127, 1.0000,
0.9877` (max/min = 1.167).

---

## 2. The V0–V6 grid (TASK 1)

Protocol held **identical** across every cell: same frozen 192-d feature
matrix, same 800/300/500 split (seed 0), same 5 training seeds, same
optimizer (`HeadTrainer` AdamW, `lr=1e-3`, `grad_clip=1.0`, `weight_decay=0`),
same 4 epochs, same head batch 64, same `choice_loss`, heads fit only on
train, temperature fitted only on calibration, test only ever scored. V4 is
the single deliberate exception (20 epochs) and is labelled as such. **No
hyperparameter was selected on any split** except where a cell is explicitly a
sensitivity probe.

`params w/ grad` counts the parameter **tensors** that receive a gradient in
the training graph (one measured forward+backward, not an assertion) — it is
what separates the cells that route the question/schema encoders through the
forward from the cells that do not.

### 2.1 Headline

| cell | definition | acc | macro-F1 | NLL | ECE(15) | Brier(15) | ECE(15) +temp | params | params w/ grad | train loss 1st→last | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **V0** | reference: shipped stack, interaction ON, `state_pathway="mlp"` | 0.9412 ± 0.0078 | 0.9415 | 0.3568 | 0.0421 | 0.1011 | 0.0242 | 2,179,073 | 29 | 1.2424 → 0.0482 | 966.5 |
| **V0L** | shipped stack with the *current* config default `state_pathway="linear"` (supplementary; closest to the MODALITY §4.1 stack) | 0.9436 ± 0.0144 | 0.9434 | 0.3021 | 0.0425 | 0.0942 | 0.0206 | 2,067,905 | 27 | 1.0580 → 0.0325 | 593.7 |
| **V1** | `conditioning_mode="off"` state-only head (bypass-equivalent) | 0.9452 ± 0.0076 | 0.9452 | 0.2120 | 0.0345 | 0.0852 | 0.0216 | 2,179,073 | 19 | 1.9918 → 0.0232 | 119.9 |
| **V2** | state path ONLY as an exact `nn.Linear(D,K)` probe (P1) | 0.9688 ± 0.0018 | 0.9691 | 0.1145 | 0.0241 | 0.0510 | 0.0183 | 2,067,905 | 6 | 1.7144 → 0.0331 | 111.9 |
| **V3** | V2 + probe-style init of the state path | 0.9660 ± 0.0024 | 0.9662 | 0.1113 | 0.0168 | 0.0524 | 0.0184 | 2,067,905 | 6 | 0.7073 → 0.0096 | 106.5 |
| **V4** | V2 + 20 epochs | **0.9728 ± 0.0011** | 0.9731 | 0.1222 | 0.0202 | 0.0494 | 0.0174 | 2,067,905 | 6 | 1.7144 → 0.0017 | 129.4 |
| **V2m** | V2 with the mlp state path, H = D = 192 (V5 centre) | 0.9656 ± 0.0038 | 0.9658 | 0.1318 | 0.0233 | 0.0583 | 0.0203 | 2,179,073 | 8 | 1.9183 → 0.0114 | 78.6 |
| **V5a** | V2m, `state_hidden_size=96` (H halved) | 0.9648 ± 0.0030 | 0.9650 | 0.1224 | 0.0172 | 0.0556 | 0.0177 | 2,098,817 | 8 | 2.0712 → 0.0345 | 75.4 |
| **V5b** | V2m, `state_hidden_size=384` (H doubled) | 0.9608 ± 0.0027 | 0.9610 | 0.1619 | 0.0276 | 0.0647 | 0.0197 | 2,339,585 | 8 | 1.6742 → 0.0084 | 64.9 |
| **V6** | V2 + class-balanced (inverse-frequency) CE | 0.9692 ± 0.0011 | 0.9695 | 0.1147 | 0.0245 | 0.0510 | 0.0188 | 2,067,905 | 6 | 1.7169 → 0.0333 | 58.2 |
| **V2I** | V2 through **shipped config fields only**: `fusion_type="identity"` + `conditioning_mode="off"` + `state_pathway="linear"` | 0.9616 ± 0.0030 | 0.9619 | 0.1453 | 0.0267 | 0.0620 | 0.0189 | 2,031,041 | 8 | 1.9058 → 0.0105 | 62.5 |
| R1 | B1 linear probe `Linear(192,10)` (== P1) | 0.9492 ± 0.0087 | 0.9494 | 0.2710 | 0.1219 | 0.0993 | 0.0209 | 1,930 | 2 | 2.0790 → 0.3249 | 3.5 |
| R2 | B2 MLP probe `Linear(192,192)-ReLU-Linear(192,10)` (== P2) | 0.9660 ± 0.0037 | 0.9663 | 0.1054 | 0.0255 | 0.0474 | 0.0159 | 38,986 | 4 | 1.5272 → 0.0653 | 4.4 |

### 2.2 Per-seed accuracy (paired — same features, same split, same seeds)

| cell | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean | std | min | max |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 0.9320 | 0.9520 | 0.9460 | 0.9380 | 0.9380 | 0.9412 | 0.0078 | 0.9320 | 0.9520 |
| V0L | 0.9380 | 0.9600 | 0.9500 | 0.9220 | 0.9480 | 0.9436 | 0.0144 | 0.9220 | 0.9600 |
| V1 | 0.9500 | 0.9500 | 0.9520 | 0.9360 | 0.9380 | 0.9452 | 0.0076 | 0.9360 | 0.9520 |
| V2 | 0.9700 | 0.9680 | 0.9660 | 0.9700 | 0.9700 | 0.9688 | 0.0018 | 0.9660 | 0.9700 |
| V3 | 0.9660 | 0.9680 | 0.9620 | 0.9660 | 0.9680 | 0.9660 | 0.0024 | 0.9620 | 0.9680 |
| V4 | 0.9720 | 0.9740 | 0.9720 | 0.9740 | 0.9720 | 0.9728 | 0.0011 | 0.9720 | 0.9740 |
| V2m | 0.9680 | 0.9700 | 0.9640 | 0.9600 | 0.9660 | 0.9656 | 0.0038 | 0.9600 | 0.9700 |
| V5a | 0.9680 | 0.9680 | 0.9620 | 0.9640 | 0.9620 | 0.9648 | 0.0030 | 0.9620 | 0.9680 |
| V5b | 0.9560 | 0.9620 | 0.9620 | 0.9620 | 0.9620 | 0.9608 | 0.0027 | 0.9560 | 0.9620 |
| V6 | 0.9700 | 0.9680 | 0.9680 | 0.9700 | 0.9700 | 0.9692 | 0.0011 | 0.9680 | 0.9700 |
| V2I | 0.9620 | 0.9620 | 0.9580 | 0.9600 | 0.9660 | 0.9616 | 0.0030 | 0.9580 | 0.9660 |
| R1 | 0.9520 | 0.9360 | 0.9500 | 0.9480 | 0.9600 | 0.9492 | 0.0087 | 0.9360 | 0.9600 |
| R2 | 0.9620 | 0.9700 | 0.9640 | 0.9700 | 0.9640 | 0.9660 | 0.0037 | 0.9620 | 0.9700 |

### 2.3 Paired per-seed deltas (accuracy, percentage points, positive = better)

vs **V0**:

| cell | s0 | s1 | s2 | s3 | s4 | mean | std | seeds better | ranges overlap |
|---|---|---|---|---|---|---|---|---|---|
| V0L | +0.60 | +0.80 | +0.40 | −1.60 | +1.00 | +0.24 | 1.05 | 4/5 | YES |
| V1 | +1.80 | −0.20 | +0.60 | −0.20 | +0.00 | **+0.40** | 0.85 | **2/5** | **YES** |
| V2 | +3.80 | +1.60 | +2.00 | +3.20 | +3.20 | **+2.76** | 0.92 | 5/5 | NO |
| V3 | +3.40 | +1.60 | +1.60 | +2.80 | +3.00 | +2.48 | 0.83 | 5/5 | NO |
| V4 | +4.00 | +2.20 | +2.60 | +3.60 | +3.40 | +3.16 | 0.74 | 5/5 | NO |
| V2I | +3.00 | +1.00 | +1.20 | +2.20 | +2.80 | +2.04 | 0.91 | 5/5 | NO |
| V6 | +3.80 | +1.60 | +2.20 | +3.20 | +3.20 | +2.80 | 0.88 | 5/5 | NO |
| V2m | +3.60 | +1.80 | +1.80 | +2.20 | +2.80 | +2.44 | 0.77 | 5/5 | NO |
| V5a | +3.60 | +1.60 | +1.60 | +2.60 | +2.40 | +2.36 | 0.83 | 5/5 | NO |
| V5b | +2.40 | +1.00 | +1.60 | +2.40 | +2.40 | +1.96 | 0.64 | 5/5 | NO |
| R1 | +2.00 | −1.60 | +0.40 | +1.00 | +2.20 | +0.80 | 1.53 | 4/5 | YES |
| R2 | +3.00 | +1.80 | +1.80 | +3.20 | +2.60 | +2.48 | 0.66 | 5/5 | NO |

vs **V2**:

| cell | s0 | s1 | s2 | s3 | s4 | mean | std | seeds better | ranges overlap |
|---|---|---|---|---|---|---|---|---|---|
| V0 | −3.80 | −1.60 | −2.00 | −3.20 | −3.20 | −2.76 | 0.92 | 0/5 | NO |
| V0L | −3.20 | −0.80 | −1.60 | −4.80 | −2.20 | −2.52 | 1.55 | 0/5 | NO |
| V1 | −2.00 | −1.80 | −1.40 | −3.40 | −3.20 | **−2.36** | 0.89 | 0/5 | NO |
| V3 | −0.40 | +0.00 | −0.40 | −0.40 | −0.20 | **−0.28** | 0.18 | 0/5 | YES |
| V4 | +0.20 | +0.60 | +0.60 | +0.40 | +0.20 | **+0.40** | 0.20 | **5/5** | **NO** |
| V6 | +0.00 | +0.00 | +0.20 | +0.00 | +0.00 | +0.04 | 0.09 | 1/5 | YES |
| V2m | −0.20 | +0.20 | −0.20 | −1.00 | −0.40 | −0.32 | 0.44 | 1/5 | YES |
| V5a | −0.20 | +0.00 | −0.40 | −0.60 | −0.80 | −0.40 | 0.32 | 0/5 | YES |
| V5b | −1.40 | −0.60 | −0.40 | −0.80 | −0.80 | −0.80 | 0.37 | 0/5 | NO |
| V2I | −0.80 | −0.60 | −0.80 | −1.00 | −0.40 | −0.72 | 0.23 | 0/5 | YES |
| R1 | −1.80 | −3.20 | −1.60 | −2.20 | −1.00 | −1.96 | 0.82 | 0/5 | NO |
| R2 | −0.80 | +0.20 | −0.20 | +0.00 | −0.60 | −0.28 | 0.41 | 1/5 | YES |

### 2.4 Wall-clock cost of the conditioning pathway (V0 vs V2, 5 seeds each)

Contended host, so read these as ratios, not as a latency benchmark:

| cell | per-seed wall (s) | mean | ratio to V2 |
|---|---|---|---|
| V0 | 17.7, 234.0, 201.9, 233.8, 279.1 | 193.3 | **8.6×** |
| V1 | 20.8, 13.5, 21.0, 38.7, 25.8 | 23.9 | 1.07× |
| V2 | 18.4, 30.2, 18.6, 30.4, 14.2 | 22.4 | 1.00× |
| V2I | 14.4, 16.0, 11.0, 11.2, 9.8 | 12.5 | 0.56× |
| V4 | 38.8, 36.0, 25.9, 14.4, 14.3 | 25.9 | 1.16× |

The 8.6× is dominated by the conditioned forward: with `conditioning=True`
the question encoder and the 10-option schema encoder re-run on **every**
training step and **every** eval batch; V1/V2/V2I do not. V0's seed 0 ran at
17.7 s before the host was loaded and 279.1 s after, so the absolute numbers
are contention-dominated; the V0↔V2 gap persisted across the whole run.

---

## 3. The decision tree (TASK 2) — branch by branch, with the evidence

**BRANCH 3 FIRED: `V2 ≫ V0` with `V1 ≈ V0` → the CONDITIONING pathway is the
cost.** The other three branches did not fire.

| branch | condition | measured | fired? |
|---|---|---|---|
| init | `V2 ≈ V1` **and** `V3 ≫ V2` | `V2 − V1 = +2.36 pp`, 0/5 seeds in V1's favour, ranges **do not** overlap — so `V2 ≈ V1` is **false**. And `V3 − V2 = −0.28 pp`, 0/5 seeds better for V3 — so `V3 ≫ V2` is **false** (V3 is slightly *worse*). | **no** |
| budget | `V2 ≈ V1` **and** `V4 ≫ V2` | the first clause is **false** (same as above). The second clause does hold in the direction sense: `V4 − V2 = +0.40 pp`, 5/5 seeds, non-overlapping ranges. | **no** (its own precondition fails) |
| **conditioning** | **`V2 ≫ V0`** and **`V1 ≈ V0`** | `V2 − V0 = +2.76 pp`, 5/5 seeds, ranges **do not** overlap ([0.9660, 0.9700] vs [0.9320, 0.9520]) → first clause true. `V1 − V0 = +0.40 pp`, std 0.85, only **2/5** seeds better, ranges **overlap** ([0.9360, 0.9520] vs [0.9320, 0.9520]), and the delta is well inside one seed std (0.0078 / 0.0076) → `≈` holds. | **YES** |
| capacity | `V2 <` MLP probe even with the best cell | V2 = 0.9688 **>** R2 = 0.9660, and V4 = 0.9728 > R2 with non-overlapping ranges. The premise is **false**. | **no** |

### 3.1 What the fired branch actually identifies, stated precisely

The brief's branch names "the gate / `conditioning_mode` default". The
measurement is more specific than that, and the difference matters:

* Turning the option-interaction **off** (`conditioning_mode="off"`, cell V1)
  recovers only **+0.40 pp** paired, 2/5 seeds, overlapping ranges — a null.
* Removing the **question/fusion half** of the conditioning pathway as well
  (cell V2) recovers **+2.76 pp**, 5/5 seeds, non-overlapping ranges.

So the option residual is **not** the cost. Roughly 2.36 of the 2.76 pp sits
in the question-row → `ConcatFusion` path, and only ~0.40 pp in the
schema/interaction path. The measured "params w/ grad" column is the
mechanical counterpart: V0 routes 29 parameter tensors through the training
graph, V1 19, V2 **6**.

This is the same structural point the class docstring already makes
(`jevx/decisions/choice.py:44-51`): with one fixed question and one fixed
schema, every option row `e_k` is identical on every batch row, so the
bilinear residual collapses to `z_k = sᵀ(W[k,:] + A e_k) + b[k]` — still
**linear** in `s`. The interaction is a reparameterization, not extra
capacity. The measurement adds that on this task the *question* mixing
(`ConcatFusion.mix`, a `Linear(384→192)` re-mixing the state with a constant
vector) is what actually costs accuracy, and it costs it twice: once by
destroying the state's geometry and once by putting ~2.1 M trainable
parameters in the graph that must be fit from 800 examples.

### 3.2 The shipped-change recommendation

Two candidate shipped changes were named by the brief; only one is supported
by the numbers, and both were measured here.

**(a) `conditioning_mode` default → `"off"` — NOT supported.** That is cell
V1: **+0.40 pp** paired vs V0, 2/5 seeds, overlapping seed ranges. It is a
null on accuracy (it does help probability quality: NLL 0.3568 → 0.2120,
ECE(15) 0.0421 → 0.0345). Changing the shipped default on this evidence would
trade the conditioning pathway's whole reason to exist for 0.4 pp inside the
noise.

**(b) `fusion_type` → `"identity"` — supported, and measured.** This is cell
V2I, expressed **only through shipped config fields**, with no code change:

```python
JevXConfig(
    base_model_name_or_path="facebook/deit-tiny-patch16-224",
    freeze_base_model=True,
    pooling_method="cls",
    fusion_type="identity",      # jevx/configuration_jevx.py:283 (default "concat")
    conditioning_mode="off",     # jevx/configuration_jevx.py:319 (default is ALREADY "off")
    state_pathway="linear",      # jevx/configuration_jevx.py:296 (default "linear")
)
```

Measured: **0.9616 ± 0.0030**, i.e. **+2.04 pp paired vs V0, 5/5 seeds,
non-overlapping ranges**, and **0.56× the wall time of V2** — 62.5 s vs V0's 966.5 s, i.e. **0.065× of V0** (the §2.4 table is ratio-to-V2, not ratio-to-V0; read the two bases apart). It is not quite
V2 (0.9688) because `IdentityFusion` is a *trainable* `Linear(192,192)`
initialized to the identity (`jevx/representations/fusion.py:430-457`), not a
hard bypass — that extra 36,864-parameter map is the measured 0.72 pp
residue.

**Recommendation to the next agent (benchmark-only pass, nothing landed):**
do **not** flip either default globally on this evidence. Instead, land
`fusion_type="identity"` as a **per-task opt-in** (it is already a config
field, so this needs no library change) and, if a default must be chosen,
change `conditioning_mode`'s default only together with the task-level
`jevx.training.conditioning_gate` machinery that already exists to decide
this per task. The honest framing of the finding is: *on a 10-way frozen-CLS
task with 800 training examples, the conditioning pathway is a net cost of
2.04–2.76 points of accuracy and ~8.6× the wall time; on the tasks in
reports/HEAD_ABLATION_MATRIX.md the same pathway paid off, so the change is
task-conditional, not a library-wide default flip.*

**Also measured, and a real but small effect:** V4 (`epochs=20` on the V2
config) is **+0.40 pp over V2, 5/5 seeds, non-overlapping ranges**, at
1.16× V2's wall time (25.9 s vs 22.4 s per seed, contended). Its train loss
reaches 0.0017 versus V2's 0.0331, i.e. V2 at 4 epochs is still
under-trained. This is a budget effect, not a conditioning effect, and it is
the only cell that beats the MLP probe on every seed.

### 3.3 The two effects that were measured and are NULL

* **Initialization (V3).** Replacing the shipped zero-init state path with
  `nn.Linear`'s own kaiming-uniform/uniform scheme is **−0.28 pp** vs V2,
  0/5 seeds better, ranges overlapping. It does visibly change the
  optimization: the first-epoch train loss drops from 1.71 to 0.70 and the
  final loss from 0.0331 to 0.0096, and raw ECE(15) improves 0.0241 → 0.0168 —
  but accuracy does not move. The zero-init residual convention is **not**
  costing accuracy here.
* **State capacity (V2m/V5a/V5b).** The capacity sweep is monotone
  *downward*: H = 96 → 0.9648, H = 192 → 0.9656, H = 384 → 0.9608, and all
  three are ≤ V2 (0.9688). Widening the state pathway never helped; 384
  costs 0.80 pp paired vs V2 (0/5 seeds). `state_hidden_size` is not a lever
  on this task.
* **Class re-weighting (V6).** `V6 − V2 = +0.04 pp`, 1/5 seeds, ranges
  overlapping — a null, exactly as §1 predicts: the test imbalance is
  χ² = 0.64 on 9 df and the weights span only 0.9524–1.1111.

---

## 4. Closing the MLP-gap question (TASK 3)

The brief quotes the published 3-seed numbers (MLP probe 0.9673, linear probe
0.9467). To make the comparison *paired*, both probes were re-run here on the
**identical** cached features, identical 800/300/500 split and the same 5
seeds as every JevX cell:

| reference | this run (5 seeds 0–4) | published (3 seeds 0–2) |
|---|---|---|
| B2 MLP probe (R2) | **0.9660 ± 0.0037** | 0.9673 ± 0.0042 |
| B1 linear probe (R1) | **0.9492 ± 0.0087** | 0.9467 ± 0.0046 |
| shipped stack (V0L, `state_pathway="linear"`) | 0.9436 ± 0.0144 | 0.9087 ± 0.0257 (JevX, `state_pathway` not yet a field) |

The probe numbers reproduce to within 0.2 pp of the published ones, so the
probes are stable across the two runs. **The JevX reference does not**: on the
published 3 seeds (0,1,2) this run's V0L scores 0.9493 and V0 scores 0.9433,
against the published 0.9087 — a 3.5–4.1 pp difference, with the trainable
parameter count also moved (published 2,018,497; current HEAD 2,067,905 for
`linear` and 2,179,073 for `mlp`). The `state_pathway` / `state_hidden_size`
flags are listed under `[Unreleased]` in CHANGELOG.md, i.e. they landed **after**
the 2026-09-25 vision run. The G5 reference number is therefore stale w.r.t.
HEAD and is not reproduced here; that is reported, not reconciled away.

### 4.1 Best cell vs the probes, paired

Best JevX-family cell by mean accuracy: **V4 (V2 config at 20 epochs),
0.9728 ± 0.0011**.

| comparison | paired Δ (pp) | paired std | seeds better | JevX range | probe range | overlap |
|---|---|---|---|---|---|---|
| **V4 − R2 (MLP)** | **+0.68** | 0.27 | **5/5** | [0.9720, 0.9740] | [0.9620, 0.9700] | **NO** |
| V2 − R2 (MLP) | +0.28 | 0.41 | 3/5 | [0.9660, 0.9700] | [0.9620, 0.9700] | YES |
| V3 − R2 | +0.00 | 0.37 | 2/5 | [0.9620, 0.9680] | [0.9620, 0.9700] | YES |
| V6 − R2 | +0.32 | 0.41 | 3/5 | [0.9680, 0.9700] | [0.9620, 0.9700] | YES |
| V2I − R2 | −0.44 | 0.52 | 1/5 | [0.9580, 0.9660] | [0.9620, 0.9700] | YES |
| V1 − R2 | −2.08 | 0.94 | 0/5 | [0.9360, 0.9520] | [0.9620, 0.9700] | NO |
| V0 − R2 | −2.48 | 0.66 | 0/5 | [0.9320, 0.9520] | [0.9620, 0.9700] | NO |
| **V4 − R1 (linear)** | **+2.36** | 0.95 | **5/5** | [0.9720, 0.9740] | [0.9360, 0.9600] | **NO** |
| **V2 − R1 (linear)** | **+1.96** | 0.82 | **5/5** | [0.9660, 0.9700] | [0.9360, 0.9600] | **NO** |
| V2I − R1 | +1.24 | 0.79 | 5/5 | [0.9580, 0.9660] | [0.9360, 0.9600] | YES |
| V1 − R1 | −0.40 | 1.37 | 2/5 | [0.9360, 0.9520] | [0.9360, 0.9600] | YES |
| V0 − R1 | −0.80 | 1.53 | 1/5 | [0.9320, 0.9520] | [0.9360, 0.9600] | YES |

Against the **published** numbers: V4 − 0.9673 = **+0.55 pp**; V2 − 0.9673 =
+0.15 pp; V4 − 0.9467 = **+2.61 pp**; V2 − 0.9467 = **+2.21 pp**.

### 4.2 One-sentence honest conclusion

**On these frozen DeiT-tiny CLS features the JevX stack is not behind the MLP
probe at all once the conditioning pathway is removed — V2 (0.9688) already
matches the MLP probe within noise and V4 (0.9728, 20 epochs) beats it on
5/5 seeds by +0.68 pp with non-overlapping ranges, and every state-only cell
(V2/V3/V4/V6) beats the linear probe on 5/5 seeds by +1.7 to +2.4 pp, so the
MLP gap is a property of the conditioning pathway and not of the state
capacity — but this is one frozen backbone, one 10-way near-domain task, 800
training examples and one host, and the V0 reference in
MODALITY_BENCHMARKS.md §4.1 does not reproduce at HEAD (§4 above), so the
published 5.9-point gap should be restated as "the conditioning pathway costs
2.0–2.8 points on this task" rather than carried forward as-is.**

### 4.3 Does any configuration MATCH the linear probe?

Yes, and several **exceed** it. Measured on the identical features/split/seeds:
**V2 = 0.9688 ± 0.0018 vs linear probe 0.9492 ± 0.0087 — +1.96 pp paired,
5/5 seeds, non-overlapping ranges**; V4 = 0.9728 (+2.36 pp, 5/5, non-overlapping);
V6 = 0.9692 (+2.00 pp, 5/5, non-overlapping); V3 = 0.9660 (+1.68 pp, 5/5,
non-overlapping). The shipped stack itself already *matches* the linear probe
statistically (V0 − R1 = −0.80 pp, 1/5 seeds, overlapping ranges) and V1 is
within −0.40 pp. So the only probe JevX does **not** match is the MLP probe
(R2), and only in its conditioned arms.

---

## 5. What this does NOT establish

1. **It does not establish a general result about the conditioning pathway.**
   One backbone (deit-tiny, 192-d CLS, frozen), one near-domain 10-way task,
   800 train / 300 cal / 500 test, one host, 5 seeds. The same pathway paid
   off on the datasets in `reports/HEAD_ABLATION_MATRIX.md`; nothing here
   overrides those cells.
2. **It does not re-establish the published G5 reference.** The 0.9087 ±
   0.0257 in MODALITY_BENCHMARKS.md §4.1 is **not reproduced** at HEAD
   (0.9433–0.9493 on the same 3 seeds; trainable params 2,018,497 →
   2,067,905/2,179,073). The published table is left untouched (append-only,
   owned by the audio agent) and the discrepancy is reported here, not fixed.
3. **It does not establish that a backbone/depth change is needed.** The
   "state capacity is the limit" branch was **not** entered because its
   premise was false: the best state-only cell already beats the MLP probe.
   No claim is made about what a larger backbone or deeper stack would do,
   and none was measured.
4. **It does not generalize past 4 (or 20) epochs / lr 1e-3 / batch 64.** No
   hyperparameter search was run on any split. V4's 20 epochs is a labelled
   sensitivity probe; if 20 epochs is better than 4 at this point on the
   learning curve, then the a-priori choice of 4 is itself part of the cost
   and this report has NOT located how much of the residual gap that explains.
5. **The wall-clock numbers are not a latency benchmark.** Other agents
   shared the 4 logical CPUs; V0's per-seed wall varied 17.7 s → 279.1 s for
   identical work. Only the coarse ratio (V0 ≈ 8.6× V2) is used, and it
   should be re-measured on an idle host before being quoted as a
   throughput claim.
6. **`params w/ grad` counts parameter TENSORS, not scalars.** It is a
   structural indicator (6 vs 19 vs 29 tensors in the graph), not a
   parameter budget.
7. **The cells are benchmark wrappers, not library configurations.** V2
   bypasses the `Fusion` module entirely; only V2I is expressed purely in
   shipped config fields. Any claim about what `fusion_type="identity"` does
   rests on V2I, which is 0.72 pp below V2 because `IdentityFusion` is
   trainable — that 0.72 pp was not decomposed further.
8. **It does not say the published accuracy gap is closed in general.** It
   says the gap on *this* task is attributable to the conditioning pathway,
   and that a config-only change recovers 2.04 pp of the 2.76 pp.

---

## 5b. CONCURRENT-EDIT record

Another agent was concurrently refactoring `jevx/decisions/choice.py`. Timeline
on this host (all UTC, from file mtimes):

| time | event |
|---|---|
| 14:26:03 | featurization launched |
| 14:40:16 | grid launched (V0 … V6, R1, R2) |
| **15:02:32** | **`jevx/decisions/choice.py` last written by the other agent** (17,112 B) |
| 15:20:08 | grid completed and wrote its result JSON |
| 15:22:28 / 15:24:58 | V2I run, then the V2I re-run (both *after* the edit) |

**No `ImportError`, `SyntaxError` or any other transient failure occurred**: the
grid reported `"failures": {}` and all 65 cell-runs completed. The V0–V6
cells executed against the pre-edit `choice.py` as it was in memory when the
process imported it at 14:40:16; the V2I cell executed against the post-edit
file. V2I was deliberately run twice against the post-edit code and returned
**0.9616 ± 0.0030 both times**, which is the evidence that the concurrent
edit did not perturb this measurement. No re-run of any cell was required and
no number in this report was produced against a half-written file.

---

## 6. Validation

`python -m pytest tests/unit -q`, run verbatim at the end of this pass, on the
same host (nothing under `jevx/` was modified by this agent):

```
.................                                                      [ 84%]
........................................................................ [ 89%]
........................................................................ [ 94%]
........................................................................ [ 99%]
............                                                             [100%]
============================== warnings summary ===============================
tests/unit/test_breadth_direction.py::test_hidden_states_only_output_is_reported_honestly
  C:\Users\ysnrfd\AppData\Roaming\Python\Python314\site-packages\torch\jit\_script.py:1485: FutureWarning: `torch.jit.script` is not supported in Python 3.14+ and may break. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1451 passed, 1 skipped, 1 warning in 170.85s (0:02:50)
```

---

## 7. Reproduce

```
python benchmarks/run_vision_diagnostic.py --phase featurize   # 362.34 s
python benchmarks/run_vision_diagnostic.py --phase grid        # 13 cells x 5 seeds
python benchmarks/run_vision_diagnostic.py --phase summarize   # renders the tables above
```

Artifacts:

| path | contents |
|---|---|
| `benchmarks/run_vision_diagnostic.py` | the runner (new; this pass owns it) |
| `benchmarks/cache/vision_diagnostic/deit_tiny_cls_features.npz` | the 1,600 × 192 feature cache (new) |
| `benchmarks/cache/vision_diagnostic/featurize_provenance.json` | ids, shas, split construction, class balance, wall times |
| `benchmarks/results/vision_diagnostic_features.json` | the featurization record (copy of the above) |
| `benchmarks/results/vision_diagnostic_grid.json` | all 13 cells × 5 seeds, per-seed metrics, failures, invariants |
| `benchmarks/results/vision_diagnostic_summary.md` | the machine-rendered tables |
| `reports/VISION_DIAGNOSTIC.md` | this report |
