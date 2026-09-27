# JevX-Next ablation matrix — does the nonlinear state path + option residual + gate + curriculum pay?

APPENDED 2026-09-26. Every number here is copied from
`benchmarks/results/head_ablation_{synthetic,ag_news,sst2,ravdess}.json`, written
by `benchmarks/run_head_ablation.py` on 2026-09-26. Nothing is estimated,
interpolated or hand-typed. Language: *measured* / *observed* only.

## 0. What was executed

| | |
|---|---|
| cells | 9 (3 references + the 6-cell JevX-Next matrix) |
| datasets | 4 (synthetic control, ag_news, sst2, ravdess) |
| seeds | 5 (0, 1, 2, 3, 4), **every cell on every seed** |
| cell-runs | **9 x 4 x 5 = 180** |
| failures | **0** (`"failures": {}` in all four payloads) |
| wall time | **2062.3 s = 34 min 22 s** (ravdess 703.0 s, ag_news 470.0 s, synthetic 462.9 s, sst2 426.4 s) |
| environment | Windows 11, Python 3.14.3, torch 2.14.0+cpu (4 cores, 2 torch threads, no CUDA), transformers 5.16.1, datasets 4.4.1, numpy 2.3.5, fp32 |

Protocol, identical for every cell: same frozen feature matrix, same splits,
same seeds, same optimizer (AdamW via `HeadTrainer`, `grad_clip=1.0`,
`weight_decay=0.0`), same `lr`, same 4 epochs, same batch size, same
`jevx.training.choice_loss`. The backbone is frozen and featurized once per
split (the ag_news / ravdess caches are reused from the EXP-H runs; sst2 and
the synthetic control had no cache — sst2 was featurized once inside this run,
frozen backbone, Mode A; the whole sst2 pass including featurization took
426.4 s against 470.0 s for the 45 already-cached ag_news cell-runs, so
featurization is not separately timed and is not claimed as a separate figure).

| cell | state path | interaction | gate | curriculum | `stage_a + stage_b` |
|---|---|---|---|---|---|
| A | linear | ON | no | no | - |
| B | mlp | **OFF** | no | no | - |
| C | mlp | ON | no | no | - |
| D | mlp | ON | YES | no | - |
| E | mlp | ON | no | YES | 2 + 2 = 4 |
| F | mlp | ON | YES | YES | 2 + 2 = 4 |

References on the identical features / splits / seeds: **P1** linear probe
`Linear[D->K]`, **P2** MLP probe `Linear[D->D] -> ReLU -> Linear[D->K]`,
**P4** JevX with the question/schema pathway bypassed (a fixed,
state-independent option basis).

`interaction OFF` (cell B) is enforced **explicitly, two ways, both asserted at
run time**: `model.config.conditioning_enabled = False` **and**
`head.set_interaction_scale(0.0)`. No config default is relied on — a fresh
model maps `JevXConfig.conditioning_scale_init == 0.0` onto a head lambda of
`1.0` by design (`JevXForDecision._state_pathway_kwargs`), so the default is
*not* "off". The recorded wiring for every run is in the JSON
(`per_seed[].wiring`).

The λ a conditioned cell runs at is pinned to 1.0 explicitly; the curriculum
cells pin `lambda_start=0.0`, `lambda_end=1.0`, `lambda_schedule="linear"`.

## 1. Headline table (test split, 5 seeds, mean±std of the per-seed value)

### synthetic control (D=128, K=4, 3000/1000/1500)

| cell | acc | F1 | NLL | ECE(15) | Brier(15) | params | train loss 1st→last | val loss 1st→last |
|---|---|---|---|---|---|---|---|---|
| P1 linear | 0.9384±0.0021 | 0.9386 | 0.3290 | 0.1760 | 0.1452 | 516 | 1.2138 → 0.3518 | 0.9230 → 0.3331 |
| **P2 MLP** | **0.9616±0.0022** | 0.9617 | 0.1230 | 0.0294 | 0.0610 | 17 028 | 0.9064 → 0.0910 | 0.4596 → 0.1337 |
| P4 bypass | 0.9523±0.0035 | 0.9523 | 0.1889 | 0.0251 | 0.0774 | 1 280 385 | 0.2770 → 0.0768 | 0.1673 → 0.2271 |
| A linear+int | 0.9445±0.0031 | 0.9444 | 0.1841 | 0.0242 | 0.0854 | 1 280 385 | 0.4007 → 0.0944 | 0.1927 → 0.2135 |
| B mlp, no int | 0.9505±0.0035 | 0.9504 | 0.1860 | 0.0255 | 0.0800 | 1 296 897 | 0.5796 → 0.0783 | 0.1832 → 0.2232 |
| C mlp+int | 0.9453±0.0071 | 0.9453 | 0.1799 | 0.0199 | 0.0855 | 1 296 897 | 0.4844 → 0.1010 | 0.2015 → 0.2137 |
| D +gate | 0.9453±0.0071 | 0.9453 | 0.1799 | 0.0199 | 0.0855 | 1 296 897 | 0.4844 → 0.1010 | 0.2015 → 0.2137 |
| E +curr | 0.9508±0.0023 | 0.9507 | 0.1873 | 0.0240 | 0.0798 | 1 296 897 | 0.5795 → 0.0783 | n/a (see §2) |
| F +both | 0.9497±0.0026 | 0.9497 | 0.1881 | 0.0261 | 0.0806 | 1 296 897 | 0.5795 → 0.0783 | n/a (see §2) |

### ag_news, text, 4-way (D=128, K=4, 3000/1000/1500)

| cell | acc | F1 | NLL | ECE(15) | Brier(15) | params | train loss 1st→last | val loss 1st→last |
|---|---|---|---|---|---|---|---|---|
| P1 linear | 0.8255±0.0033 | 0.8223 | 0.5000 | 0.0531 | 0.2624 | 516 | 1.0667 → 0.4581 | 0.8049 → 0.5095 |
| **P2 MLP** | **0.8361±0.0031** | 0.8327 | 0.4608 | 0.0365 | 0.2424 | 17 028 | 0.7592 → 0.3689 | 0.5062 → 0.4666 |
| P4 bypass | 0.8316±0.0149 | 0.8285 | 0.4794 | 0.0468 | 0.2509 | 1 280 385 | 0.5557 → 0.3827 | 0.5149 → 0.4945 |
| A linear+int | 0.8327±0.0108 | 0.8302 | 0.4913 | 0.0444 | 0.2535 | 1 280 385 | 0.6015 → 0.3798 | 0.5510 → 0.5018 |
| B mlp, no int | 0.8328±0.0082 | 0.8304 | 0.4857 | 0.0506 | 0.2530 | 1 296 897 | 0.7432 → 0.3725 | 0.5156 → 0.4933 |
| C mlp+int | 0.8313±0.0123 | 0.8287 | 0.4975 | 0.0492 | 0.2537 | 1 296 897 | 0.6453 → 0.3808 | 0.5250 → 0.5059 |
| D +gate | 0.8313±0.0123 | 0.8287 | 0.4975 | 0.0492 | 0.2537 | 1 296 897 | 0.6453 → 0.3808 | 0.5250 → 0.5059 |
| E +curr | 0.8321±0.0062 | 0.8294 | 0.4922 | 0.0538 | 0.2546 | 1 296 897 | 0.7432 → 0.3725 | n/a |
| F +both | 0.8332±0.0081 | 0.8306 | 0.4866 | 0.0488 | 0.2533 | 1 296 897 | 0.7432 → 0.3725 | n/a |

### sst2, text, binary (D=128, K=2, 3000/1000/872)

| cell | acc | F1 | NLL | ECE(15) | Brier(15) | params | train loss 1st→last | val loss 1st→last |
|---|---|---|---|---|---|---|---|---|
| **P1 linear** | 0.7000±0.0046 | 0.6998 | 0.5805 | 0.0299 | 0.3953 | 258 | 0.6815 → 0.5401 | 0.6287 → 0.5643 |
| P2 MLP | 0.6966±0.0164 | 0.6940 | 0.5889 | 0.0518 | 0.4018 | 16 770 | 0.6057 → 0.5088 | 0.5713 → 0.5558 |
| P4 bypass | 0.6975±0.0184 | 0.6945 | 0.6005 | 0.0621 | 0.4088 | 1 280 385 | 0.6878 → 0.5354 | 0.6068 → 0.5686 |
| A linear+int | 0.6929±0.0256 | 0.6882 | 0.6018 | 0.0645 | 0.4105 | 1 280 385 | 0.5977 → 0.5195 | 0.5683 → 0.5796 |
| B mlp, no int | 0.7014±0.0143 | 0.6985 | 0.5840 | 0.0521 | 0.3993 | 1 296 897 | 0.6084 → 0.5232 | 0.5726 → 0.5679 |
| C mlp+int | 0.7000±0.0122 | 0.6972 | 0.5865 | 0.0484 | 0.4009 | 1 296 897 | 0.6144 → 0.5234 | 0.5829 → 0.5670 |
| D +gate | 0.7002±0.0127 | 0.6974 | 0.5834 | 0.0445 | 0.3995 | 1 296 897 | 0.6144 → 0.5234 | 0.5829 → 0.5670 |
| E +curr | 0.7007±0.0135 | 0.6978 | 0.5825 | 0.0472 | 0.3985 | 1 296 897 | 0.6084 → 0.5232 | n/a |
| F +both | 0.7007±0.0135 | 0.6978 | 0.5825 | 0.0472 | 0.3985 | 1 296 897 | 0.6084 → 0.5232 | n/a |

### ravdess, audio, 8-way (D=768, K=8, 700/240/500)

| cell | acc | F1 | NLL | ECE(15) | Brier(15) | params | train loss 1st→last | val loss 1st→last |
|---|---|---|---|---|---|---|---|---|
| P1 linear | 0.3472±0.0162 | 0.2870 | 1.9169 | 0.1692 | 0.8330 | 6 152 | 2.0668 → 1.9195 | 2.0234 → 1.9169 |
| **P2 MLP** | **0.3632±0.0217** | 0.3080 | 1.7304 | 0.0907 | 0.7776 | 596 744 | 2.0451 → 1.7238 | 1.9774 → 1.7204 |
| P4 bypass | 0.2720±0.0572 | 0.1971 | 2.0982 | 0.1946 | 0.8686 | 13 579 265 | 7.5494 → 1.9964 | 4.3151 → 2.0977 |
| A linear+int | 0.2104±0.0351 | 0.1234 | 2.0500 | 0.1080 | 0.8633 | 13 579 265 | 6.8857 → 1.9865 | 3.2156 → 2.0311 |
| B mlp, no int | 0.3072±0.0511 | 0.2543 | 1.7640 | 0.0979 | 0.7981 | 14 169 857 | 2.0642 → 1.7076 | 1.9841 → 1.7807 |
| C mlp+int | 0.2376±0.0274 | 0.1436 | 1.8722 | 0.0402 | 0.8294 | 14 169 857 | 3.2286 → 1.9498 | 2.1416 → 1.8693 |
| D +gate | 0.2376±0.0274 | 0.1436 | 1.8722 | 0.0402 | 0.8294 | 14 169 857 | 3.2286 → 1.9498 | 2.1416 → 1.8693 |
| E +curr | 0.2812±0.0579 | 0.2247 | 1.8638 | 0.1496 | 0.8361 | 14 169 857 | 2.0642 → 1.7076 | n/a |
| F +both | 0.3068±0.0511 | 0.2541 | 1.7637 | 0.0972 | 0.7980 | 14 169 857 | 2.0642 → 1.7076 | n/a |

`n/a` for E/F `val loss`: `run_curriculum` records its per-epoch calibration
metrics under the `cal_` prefix in `EpochRecord.metrics` (they are in the JSON
at `per_seed[].convergence_curve[].metrics`), not under `HeadTrainer`'s
`val_` prefix; the cells are compared on their test metrics and on the
train-loss series, both of which are present for all nine cells.

Convergence curves (per-epoch JSON series) are in
`per_seed[].convergence_curve`. First-epoch train loss is the curriculum's
Stage-A epoch-1 figure for E/F.

## 2. Paired per-seed deltas (same features, same seeds, same order)

Paired because the cells share the data, the split and the seed stream, so the
per-seed difference is the unit of comparison; "seeds favouring" counts how many
of the 5 seeds the LEFT cell wins.

| dataset | comparison | mean Δacc (pt) | seeds favouring left | per-seed Δacc (pt) | mean ΔNLL | seeds favouring left on NLL |
|---|---|---|---|---|---|---|
| synthetic | B − P2 | **−1.11** | 0/5 | −1.53 −0.80 −0.93 −0.80 −1.47 | +0.0630 | 0/5 |
| synthetic | C − B | −0.52 | 1/5 | +0.40 −0.47 −0.33 −1.80 −0.40 | −0.0061 | 3/5 |
| synthetic | D − C | 0.00 | 0/5 | +0.00 ×5 | +0.0000 | 0/5 |
| synthetic | E − C | +0.55 | 3/5 | −0.13 +0.47 −0.07 +2.07 +0.40 | +0.0074 | 1/5 |
| synthetic | F − C | +0.44 | 3/5 | −0.40 +0.47 −0.07 +1.80 +0.40 | +0.0082 | 1/5 |
| synthetic | F − E | −0.11 | 0/5 | −0.27 +0.00 +0.00 −0.27 +0.00 | +0.0008 | 1/5 |
| synthetic | A − C | −0.08 | 1/5 | −0.07 −0.27 −0.60 +0.67 −0.13 | +0.0042 | 2/5 |
| ag_news | B − P2 | −0.33 | 1/5 | −1.53 −0.27 −0.33 −0.60 +1.07 | +0.0248 | 1/5 |
| ag_news | C − B | −0.15 | 2/5 | −0.87 −0.53 +0.27 +1.20 −0.80 | +0.0119 | 1/5 |
| ag_news | D − C | 0.00 | 0/5 | +0.00 ×5 | +0.0000 | 0/5 |
| ag_news | E − C | +0.08 | 2/5 | +1.27 +0.00 −0.40 −1.00 +0.53 | −0.0053 | 3/5 |
| ag_news | F − C | +0.19 | 3/5 | +0.87 +0.53 −0.27 −1.00 +0.80 | −0.0109 | 3/5 |
| ag_news | F − E | +0.11 | 3/5 | −0.40 +0.53 +0.13 +0.00 +0.27 | −0.0056 | 4/5 |
| ag_news | A − C | +0.13 | 3/5 | +0.27 +0.40 −0.07 −0.07 +0.13 | −0.0063 | 4/5 |
| sst2 | B − P2 | +0.48 | 2/5 | +2.41 −0.23 +4.13 −1.38 −2.52 | −0.0049 | 2/5 |
| sst2 | C − B | −0.14 | 1/5 | +0.00 +0.00 −1.15 −0.11 +0.57 | +0.0025 | 2/5 |
| sst2 | D − C | +0.02 | 1/5 | +0.11 +0.00 +0.00 +0.00 +0.00 | −0.0030 | 2/5 |
| sst2 | E − C | +0.07 | 1/5 | −0.11 −0.11 +1.03 −0.11 −0.34 | −0.0040 | 5/5 |
| sst2 | F − C | +0.07 | 1/5 | −0.11 −0.11 +1.03 −0.11 −0.34 | −0.0040 | 5/5 |
| sst2 | F − E | 0.00 | 0/5 | +0.00 ×5 | +0.0000 | 0/5 |
| sst2 | A − C | −0.71 | 1/5 | +0.00 −0.11 +1.03 −3.67 −0.80 | +0.0154 | 2/5 |
| **ravdess** | B − P2 | **−5.60** | **0/5** | −6.20 −8.60 −2.80 −0.60 −9.80 | +0.0336 | 2/5 |
| **ravdess** | C − B | **−6.96** | **0/5** | −6.00 −3.80 −7.40 −11.40 −6.20 | +0.1083 | 0/5 |
| ravdess | D − C | 0.00 | 0/5 | +0.00 ×5 | +0.0000 | 0/5 |
| **ravdess** | E − C | **+4.36** | 4/5 | −0.80 +3.40 +9.40 +7.60 +2.20 | −0.0085 | 2/5 |
| **ravdess** | F − C | **+6.92** | **5/5** | +5.80 +3.80 +7.40 +11.40 +6.20 | −0.1086 | **5/5** |
| ravdess | F − E | +2.56 | 4/5 | +6.60 +0.40 −2.00 +3.80 +4.00 | −0.1001 | **5/5** |
| ravdess | A − C | −2.72 | 1/5 | +1.80 −1.40 −5.40 −6.20 −2.40 | +0.1778 | 0/5 |

Per-seed accuracies, for overlap visibility:

| dataset | P1 | P2 | P4 | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|---|---|
| synthetic | .9367 .9387 .9393 .9360 .9413 | .9607 .9600 .9640 .9593 .9640 | .9500 .9560 .9540 .9540 .9473 | .9487 .9447 .9453 .9400 .9440 | .9453 .9520 .9547 .9513 .9493 | .9493 .9473 .9513 .9333 .9453 | = C | .9480 .9520 .9507 .9540 .9493 | .9453 .9520 .9507 .9513 .9493 |
| ag_news | .8273 .8260 .8253 .8287 .8200 | .8360 .8353 .8407 .8367 .8320 | .8053 .8407 .8400 .8353 .8367 | .8147 .8313 .8393 .8420 .8360 | .8207 .8327 .8373 .8307 .8427 | .8120 .8273 .8400 .8427 .8347 | = C | .8247 .8273 .8360 .8327 .8400 | .8207 .8327 .8373 .8327 .8427 |
| sst2 | .7076 .6972 .6961 .6984 .7007 | .6961 .7030 .6686 .7064 .7087 | .7144 .7030 .6938 .6674 .7087 | .7202 .6995 .7087 .6548 .6812 | .7202 .7007 .7099 .6927 .6835 | .7202 .7007 .6984 .6915 .6892 | .7213 .7007 .6984 .6915 .6892 | .7190 .6995 .7087 .6904 .6858 | = E |
| ravdess | .3540 .3260 .3600 .3340 .3620 | .3760 .3260 .3620 .3780 .3740 | .2260 .1960 .3000 .3180 .3200 | .2720 .1880 .2060 .1960 .1900 | .3140 .2400 .3340 .3720 .2760 | .2540 .2020 .2600 .2580 .2140 | = C | .2460 .2360 .3540 .3340 .2360 | .3120 .2400 .3340 .3720 .2760 |

## 3. The three decisive questions

### Q1 — is B (mlp state path, interaction OFF) ≈ P2 (MLP probe)? **No.**

| dataset | B | P2 | paired Δ | seeds with B ≥ P2 |
|---|---|---|---|---|
| synthetic | 0.9505±0.0035 | 0.9616±0.0022 | **−1.11 pt** | 0/5 |
| ag_news | 0.8328±0.0082 | 0.8361±0.0031 | −0.33 pt | 1/5 |
| sst2 | 0.7014±0.0143 | 0.6966±0.0164 | +0.48 pt | 2/5 |
| ravdess | 0.3072±0.0511 | 0.3632±0.0217 | **−5.60 pt** | 0/5 |

The hypothesis said: if B ≈ P2, the missing nonlinear state pathway was the
dominant factor. The data does **not** support that. On synthetic the MLP probe
is ahead on all 5 seeds; on RAVDESS it is ahead on all 5 seeds by 5.6 points;
on ag_news the two are within one seed std of each other. The single dataset
where B is nominally ahead (sst2, +0.48 pt, 2/5) is the dataset where the MLP
probe does not separate from the *linear* probe either (P1 0.7000 vs P2
0.6966) — so it is a null, not a win.

**What else is implicated, with numbers.** Cell B is not short of state
capacity: its state pathway is 49 536 parameters at D=128 (787 456 at D=768),
i.e. *wider* than P2's whole trunk (17 028 / 596 744), and its per-module
gradient table (§5) shows the state pathway receiving a healthy 6.6–16.5 grad
norm, 25–70x what the linear pathway gets in cell A. The binding constraint is
the **rest of the decision stack**: 1 296 897 trainable parameters at D=128 and
14 169 857 at D=768 against P2's 17 028 and 596 744 — **76x and 24x** — of
which 98.8% / 95.8% is the question encoder (574 080 / 4 919 040), the schema
encoder (the same again) and the fusion (65 920 / 2 361 600). All three encode
a quantity that is **constant for a whole dataset**: there is one fixed
question and one fixed schema. Under a 4-epoch budget those 1.2 M–13.5 M
parameters are learned from scratch per run, and the measured result is exactly
that: on RAVDESS cell A's first-epoch train loss is 6.8857 against cell B's
2.0642 and P2's 2.0451 — the stack spends its entire first epoch fitting the
encoders before the state signal is readable at all.

The calibration ordering points the same way. On RAVDESS cell C has the lowest
ECE of any cell (0.0402 vs P2's 0.0907) while sitting 12.6 points below P2 on
accuracy. A model that is better calibrated and much worse at the argmax is
underfit / over-smoothed, not capacity-limited.

### Q2 — is C (mlp + interaction) > B (mlp, interaction OFF)? **No. C ≤ B on all four datasets.**

| dataset | C − B (acc) | seeds favouring C | C − B (NLL) | seeds favouring C on NLL |
|---|---|---|---|---|
| synthetic | −0.52 pt | 1/5 | **−0.0061** | 3/5 |
| ag_news | −0.15 pt | 2/5 | +0.0119 | 1/5 |
| sst2 | −0.14 pt | 1/5 | +0.0025 | 2/5 |
| **ravdess** | **−6.96 pt** | **0/5** | **+0.1083** | **0/5** |

Option conditioning carries **no** measurable information on top of the state
path here, and on the audio features it costs 7.0 accuracy points. This is what
the design review's own math predicts, now measured: with one fixed question and
one fixed schema every option row `e_k` is the same vector on every batch row,
so the bilinear residual `s^T A e_k` is a **constant** vector and
`z_k = s^T (W_k + A e_k) + b_k` is still linear in `s`. The residual therefore
adds no information — only 16 384 (D=128) / 589 824 (D=768) parameters and a
second optimization path — and on the hardest dataset it is a 7-point tax.
The interaction's parameters are also the ones that move least relative to their
initialisation: in cell C they collect a 27.8 grad norm at D=768 but the score
they produce is worth −0.1083 NLL there.

The one place it pays is likelihood on the synthetic control: C's NLL 0.1799
against B's 0.1860 (paired −0.0061, better on 3 of 5 seeds), with ECE 0.0199
against 0.0255. A real but small effect, invisible at the argmax (C is 0.52 pt
*behind* on accuracy). Reported as measured, not claimed as a win.

### Q3 — D > C (gate), and F vs E / C (gate + curriculum)?

**D vs C: a no-op on three of four datasets.**

| dataset | D − C (acc) | gate flips (of 5 seeds) |
|---|---|---|
| synthetic | 0.00 pt, 0/5 | 0/5 → `conditioned` ×5 |
| ag_news | 0.00 pt, 0/5 | 0/5 → `conditioned` ×5 |
| ravdess | 0.00 pt, 0/5 | 0/5 → `conditioned` ×5 |
| sst2 | +0.02 pt, 1/5 | 2/5 → `state_only` ×2 |

Cell D's per-seed accuracies are **bit-identical to C's** on synthetic, ag_news
and ravdess, because the gate kept `conditioned` and `apply_gate` wrote back
the flag the model already had. The gate never fires on a model whose residual
was actually trained — ΔNLL is strongly negative there (−0.10 to −0.25 on
ag_news/synthetic, −0.09 to −0.23 on ravdess), i.e. conditioning genuinely helps
on the calibration split, so `ΔNLL > τ` is false.

**E (curriculum) vs C:**

| dataset | E − C (acc) | seeds favouring E |
|---|---|---|
| synthetic | +0.55 pt | 3/5 |
| ag_news | +0.08 pt | 2/5 |
| sst2 | +0.07 pt | 1/5 |
| **ravdess** | **+4.36 pt** | 4/5 |

**F (gate + curriculum) vs C and vs E:**

| dataset | F − C (acc) | seeds | F − C (NLL) | seeds | F − E (acc) | seeds | F − E (NLL) | seeds |
|---|---|---|---|---|---|---|---|---|
| synthetic | +0.44 pt | 3/5 | +0.0082 | 1/5 | −0.11 pt | 0/5 | +0.0008 | 1/5 |
| ag_news | +0.19 pt | 3/5 | −0.0109 | 3/5 | +0.11 pt | 3/5 | −0.0056 | 4/5 |
| sst2 | +0.07 pt | 1/5 | −0.0040 | 5/5 | 0.00 pt | 0/5 | +0.0000 | 0/5 |
| **ravdess** | **+6.92 pt** | **5/5** | **−0.1086** | **5/5** | **+2.56 pt** | 4/5 | **−0.1001** | **5/5** |

The only place the combination is a clear, seed-consistent result is RAVDESS:
F beats C on 5/5 seeds on accuracy (+6.92 pt) and on NLL (−0.1086), and beats E
on NLL on 5/5 seeds (−0.1001). On the three text/synthetic datasets every
paired delta is inside one seed std and flips sign across seeds, so no winner is
declared from them.

## 4. Gate flip table (calibration split only; evaluation on test)

The gate is measured with `measure_modes` on the **calibration** split under
`ConditioningGateConfig(split_name="calibration")`, which passes the shipped
no-test-leakage guard. τ comes from the one-standard-error protocol on the same
calibration batches (`fit_tau`, `z=1.0`). **The test split is never shown to
the gate**; it is scored after the verdict, through `apply_gate`, and only the
verdict's flag survives into the test forward.

**Flip frequency: D 2/20, F 12/20** (20 = 4 datasets x 5 seeds).

| dataset | cell | seed | ΔNLL (cal) | τ | τ source | verdict | cond NLL | state-only NLL | cal acc cond | cal acc state-only | test acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ravdess | D | 0 | −0.1001 | 0.03197 | one_se(z=1, se=0.0319702, batches=4) | conditioned | 1.9122 | 2.0123 | 0.2125 | 0.2042 | 0.2540 |
| ravdess | D | 1 | −0.0924 | 0.01830 | one_se(z=1, se=0.0183028, batches=4) | conditioned | 1.8872 | 1.9796 | 0.1792 | 0.2333 | 0.2020 |
| ravdess | D | 2 | −0.1098 | 0.01910 | one_se(z=1, se=0.0190968, batches=4) | conditioned | 1.8641 | 1.9739 | 0.2208 | 0.1958 | 0.2600 |
| ravdess | D | 3 | −0.2262 | 0.02611 | one_se(z=1, se=0.026105, batches=4) | conditioned | 1.7960 | 2.0222 | 0.2667 | 0.2125 | 0.2580 |
| ravdess | D | 4 | −0.0965 | 0.02528 | one_se(z=1, se=0.0252824, batches=4) | conditioned | 1.9217 | 2.0182 | 0.2208 | 0.1667 | 0.2140 |
| **ravdess** | **F** | 0 | **+0.0584** | 0.02397 | one_se(z=1, se=0.023969, batches=4) | **state_only** | 1.9234 | 1.8650 | 0.2208 | 0.3167 | 0.3120 |
| **ravdess** | **F** | 1 | **+0.0637** | 0.03119 | one_se(z=1, se=0.0311876, batches=4) | **state_only** | 1.9783 | 1.9147 | 0.2125 | 0.2167 | 0.2400 |
| **ravdess** | **F** | 2 | **+0.1142** | 0.01781 | one_se(z=1, se=0.0178139, batches=4) | **state_only** | 1.7865 | 1.6723 | 0.3542 | 0.3667 | 0.3340 |
| **ravdess** | **F** | 3 | **+0.1307** | 0.03996 | one_se(z=1, se=0.0399564, batches=4) | **state_only** | 1.8078 | 1.6771 | 0.3250 | 0.3417 | 0.3720 |
| **ravdess** | **F** | 4 | **+0.0935** | 0.02130 | one_se(z=1, se=0.0213034, batches=4) | **state_only** | 1.8720 | 1.7785 | 0.2083 | 0.2333 | 0.2760 |
| synthetic | D | 0–4 | −0.1022 … −0.1989 | 0.01294–0.02609 | one_se(z=1, se=…, batches=16) | conditioned ×5 | 0.1486–0.2597 | 0.3247–0.4359 | 0.919–0.952 | 0.917–0.952 | 0.9333–0.9513 |
| synthetic | F | 0 | +0.0033 | 0.00171 | one_se(z=1, se=0.00170602, batches=16) | state_only | 0.2672 | 0.2639 | 0.9290 | 0.9300 | 0.9453 |
| synthetic | F | 1 | +0.0008 | 0.00236 | one_se(z=1, se=0.00236213, batches=16) | conditioned | 0.2041 | 0.2034 | 0.9360 | 0.9350 | 0.9520 |
| synthetic | F | 2 | −0.0002 | 0.00289 | one_se(z=1, se=0.00289034, batches=16) | conditioned | 0.2413 | 0.2416 | 0.9330 | 0.9320 | 0.9507 |
| synthetic | F | 3 | +0.0074 | 0.00409 | one_se(z=1, se=0.00409369, batches=16) | state_only | 0.1953 | 0.1879 | 0.9390 | 0.9420 | 0.9513 |
| synthetic | F | 4 | +0.0089 | 0.00414 | one_se(z=1, se=0.00413576, batches=16) | state_only | 0.2043 | 0.1954 | 0.9320 | 0.9370 | 0.9493 |
| ag_news | D | 0–4 | −0.1620 … −0.2487 | 0.01752–0.02227 | one_se(z=1, se=…, batches=16) | conditioned ×5 | 0.4874–0.5246 | 0.6734–0.7565 | 0.831–0.850 | 0.819–0.841 | 0.8120–0.8427 |
| ag_news | F | 0 | +0.0185 | 0.00720 | one_se(z=1, se=0.00720311, batches=16) | state_only | 0.5313 | 0.5128 | 0.8360 | 0.8340 | 0.8207 |
| ag_news | F | 1 | +0.0178 | 0.00462 | one_se(z=1, se=0.00461598, batches=16) | state_only | 0.5055 | 0.4877 | 0.8440 | 0.8470 | 0.8327 |
| ag_news | F | 2 | +0.0114 | 0.00242 | one_se(z=1, se=0.0024195, batches=16) | state_only | 0.4960 | 0.4846 | 0.8400 | 0.8400 | 0.8373 |
| ag_news | F | 3 | +0.0024 | 0.00690 | one_se(z=1, se=0.00690357, batches=16) | conditioned | 0.5152 | 0.5128 | 0.8230 | 0.8280 | 0.8327 |
| ag_news | F | 4 | +0.0035 | 0.00210 | one_se(z=1, se=0.00210242, batches=16) | state_only | 0.4674 | 0.4639 | 0.8360 | 0.8370 | 0.8427 |
| sst2 | D | 0 | +0.0042 | 0.00240 | one_se(z=1, se=0.00240478, batches=16) | state_only | 0.5507 | 0.5464 | 0.7230 | 0.7230 | 0.7213 |
| sst2 | D | 1 | −0.0050 | 0.00779 | one_se(z=1, se=0.00779095, batches=16) | conditioned | 0.5484 | 0.5535 | 0.7230 | 0.7240 | 0.7007 |
| sst2 | D | 2 | −0.0143 | 0.01113 | one_se(z=1, se=0.0111303, batches=16) | conditioned | 0.5527 | 0.5671 | 0.7210 | 0.7210 | 0.6984 |
| sst2 | D | 3 | +0.0191 | 0.00395 | one_se(z=1, se=0.00394892, batches=16) | state_only | 0.5868 | 0.5677 | 0.7210 | 0.7190 | 0.6915 |
| sst2 | D | 4 | −0.0048 | 0.00727 | one_se(z=1, se=0.00726897, batches=16) | conditioned | 0.5745 | 0.5793 | 0.7020 | 0.7030 | 0.6892 |
| sst2 | F | 0–4 | −0.0011 … −0.0006 | 0.00032–0.00146 | one_se(z=1, se=…, batches=16) | conditioned ×5 | 0.5466–0.5822 | 0.5506–0.5827 | 0.700–0.728 | 0.702–0.729 | 0.6858–0.7190 |

**Did flipping help?** On the 20 instances where the gate fired at all, the
paired test-accuracy change versus the same cell without the gate is:
RAVDESS F +2.56 pt (4/5 seeds, NLL 5/5), ag_news F +0.11 pt (3/5), synthetic F
−0.11 pt (0/5), sst2 D +0.02 pt (1/5). So the flips help on the dataset where
the residual is most harmful and are within seed noise elsewhere. They are also
**never worse than 0.11 pt**, which is the useful property: the gate's downside
is bounded by the magnitude of a single mode change on a task where the two
modes are close.

## 5. Parameter budget and the gradient-norm evidence

### Trainable parameters by module (per cell; identical across the 5 seeds)

| D, K | P1 | P2 | state_pathway (linear → mlp) | option_interaction | schema | fusion | question | state_proj | total (A/P4 → B..F) |
|---|---|---|---|---|---|---|---|---|---|
| 128, 4 | 516 | 17 028 | 33 024 → 49 536 | 16 384 | 574 080 | 65 920 | 574 080 | 16 768 | 1 280 385 → 1 296 897 |
| 128, 2 | 258 | 16 770 | 33 024 → 49 536 | 16 384 | 574 080 | 65 920 | 574 080 | 16 768 | 1 280 385 → 1 296 897 |
| 768, 8 | 6 152 | 596 744 | 196 864 → 787 456 | 589 824 | 4 919 040 | 2 361 600 | 4 919 040 | 592 128 | 13 579 265 → 14 169 857 |

`option_pathway` (the sum only option conditioning can reach: interaction +
schema + fusion) is 656 384 at D=128 and 7 870 464 at D=768 — **50.6% and
55.6%** of the JevX-Next stack. The nonlinear state pathway itself costs
16 512 (D=128) / 590 592 (D=768) more than the linear one, i.e. **1.3% / 4.2%**
of the stack. The head-vs-probe gap therefore cannot be a state-capacity gap:
the MLP state pathway is 1.3% of what the stack already spends.

### Per-module gradient norms (seed 0; L2 over the group, max over the epoch's steps, un-clipped, pre-`optimizer.step`)

ravdess (D=768, K=8) — the dataset where the effect is largest:

| cell | epoch | state_pathway | option_interaction | schema | fusion | question | state_proj | option_pathway | total |
|---|---|---|---|---|---|---|---|---|---|
| P1 | epoch1 | - | - | - | - | - | - | 0.000 | 1.645 |
| P1 | final | - | - | - | - | - | - | 0.000 | 1.955 |
| P2 | epoch1 | - | - | - | - | - | - | 0.000 | 16.288 |
| P2 | final | - | - | - | - | - | - | 0.000 | 17.469 |
| P4 | epoch1 | 0.243 | 27.769 | - | - | - | 31.827 | 27.769 | 42.239 |
| P4 | final | 0.284 | 27.752 | - | - | - | 31.882 | 27.752 | 42.269 |
| **A** | epoch1 | **0.231** | 27.739 | 47.658 | 30.987 | 52.573 | 31.827 | **63.253** | 88.192 |
| **A** | final | **0.318** | 27.712 | 47.757 | 31.381 | 52.748 | 31.986 | **63.510** | 88.538 |
| **B** | epoch1 | **16.144** | - | - | 21.911 | 52.487 | 31.802 | 21.911 | 67.134 |
| **B** | final | **16.543** | - | - | 22.371 | 52.812 | 32.294 | 22.371 | 67.869 |
| C | epoch1 | 16.164 | 27.766 | 47.666 | 31.027 | 52.661 | 31.855 | 63.290 | 89.748 |
| C | final | 16.458 | 27.820 | 47.801 | 31.551 | 53.048 | 32.106 | 63.674 | 90.389 |
| D | epoch1 | 16.164 | 27.766 | 47.666 | 31.027 | 52.661 | 31.855 | 63.290 | 89.748 |
| D | final | 16.458 | 27.820 | 47.801 | 31.551 | 53.048 | 32.106 | 63.674 | 90.389 |
| E | epoch1 | 16.145 | - | - | 21.913 | 52.489 | 31.802 | 21.913 | 67.137 |
| E | final | 16.550 | - | - | 22.381 | 52.831 | 32.306 | 22.381 | 67.894 |
| F | epoch1 | 16.145 | - | - | 21.913 | 52.489 | 31.802 | 21.913 | 67.137 |
| F | final | 16.550 | - | - | 22.381 | 52.831 | 32.306 | 22.381 | 67.894 |

synthetic (D=128, K=4): A 0.278→0.445 state vs 22.738→22.837 option_pathway;
B 7.386→7.564 state vs 5.028→5.340 option; C 6.787→6.854 state vs
22.719→22.789 option. P2 total 7.192→8.519.
ag_news (D=128, K=4): A 0.238→0.367 state vs 22.558→22.555 option; B
7.096→7.179 state vs 4.679→4.892 option; C 6.679→6.682 state vs 22.587→22.569
option. P2 total 7.144→7.640.
sst2 (D=128, K=2): A 0.114→0.161 state vs 22.445→22.434 option; B 6.631→6.628
state vs 3.902→4.028 option; C 6.536→6.555 state vs 22.449→22.415 option. P2
total 6.729→7.035.

**What this measures.** The "joint-optimization inefficiency" half of the
hypothesis is *confirmed as a description of the LINEAR state pathway*: in cell
A the state pathway's gradient norm is 0.114–0.278 at epoch 1 and has still only
grown to 0.161–0.445 at the final epoch, against an option-pathway norm of
22.4–63.5 — a ratio of **1:82 (synthetic), 1:95 (ag_news), 1:196 (sst2),
1:274 (ravdess)**. The linear state pathway is starved by two orders of
magnitude and never wakes up inside the budget. The MLP state pathway is
unstarved: 6.5–16.5, a ratio of roughly 1:3 to 1:4 against the option pathway,
i.e. the state term is no longer the neglected half of the model.

**What this also shows.** Starvation was not the binding constraint either.
Cell B has the healthiest state-pathway gradient in the table and is still
5.60 pt below P2 on RAVDESS and 1.11 pt below it on synthetic. The gradient
budget the model receives and the accuracy it reaches are not the binding pair
on these datasets.

## 6. Latency (batch 1, 20 warmup + 120 timed iterations, seed 0)

Pooled features are precomputed, so the "JevX e2e" column is the decision stack
(`state_proj` → question encoder/proj → schema encoder/proj → fusion → head) and
the frozen backbone is **not** in the measurement. "head-only" runs
`heads["choice"]` on cached `[1, D]` / `[1, K, D]` tensors. For P1/P2 the probe
forward *is* the head, so the probe number is the e2e number.

| cell | synthetic (D=128) e2e / head-only | ag_news (D=128) e2e / head-only | sst2 (D=128) e2e / head-only | ravdess (D=768) e2e / head-only |
|---|---|---|---|---|
| P1 | 0.053±0.019 / - | 0.030±0.002 / - | 0.034±0.014 / - | 0.032±0.008 / - |
| P2 | 0.146±0.229 / - | 0.222±0.395 / - | 0.110±0.160 / - | 0.312±0.257 / - |
| P4 | 1.164±0.894 / - | 0.902±0.648 / - | 10.827±18.874 / - | 2.681±0.794 / - |
| A | 4.999±1.752 / 1.069±0.759 | 4.444±1.736 / 1.006±0.685 | 6.543±2.253 / 1.734±1.198 | 9.774±3.241 / 2.403±1.501 |
| B | 4.594±1.988 / 1.324±0.951 | 4.182±1.831 / 1.105±0.728 | 5.606±2.527 / 2.037±1.782 | 7.550±2.647 / 2.077±1.108 |
| C | 5.136±1.961 / 1.160±0.730 | 4.969±1.962 / 1.123±0.814 | 4.864±1.876 / 1.425±1.403 | 9.500±3.324 / 2.149±0.969 |
| D | 4.931±1.813 / 1.146±0.743 | 4.815±1.860 / 1.118±0.755 | 3.765±1.702 / 1.132±0.852 | 14.823±3.895 / 3.007±1.013 |
| E | 4.823±1.874 / 1.126±0.650 | 4.871±1.868 / 1.128±0.724 | 4.543±2.011 / 1.085±0.719 | 11.685±2.353 / 2.704±0.947 |
| F | 4.124±1.786 / 1.109±0.790 | 4.057±1.882 / 1.264±0.773 | 4.317±1.751 / 1.056±0.792 | 9.350±6.885 / 2.017±0.999 |

The JevX decision stack costs **23x–35x** the MLP probe per example at D=128
(4.06–6.54 ms against 0.110–0.222 ms) and **23x–30x** at D=768 (7.55–14.82 ms
against 0.312 ms). The MLP state pathway itself is cheap: B vs A is 4.59 vs
5.00 ms (synthetic) and 7.55 vs 9.77 ms (ravdess). The cost is in the encoders
and the fusion, which is the same 1.2 M–13.5 M of parameters that the gradient
table shows being fitted before the state signal is readable.

Caveat, stated rather than smoothed: the per-iteration std is comparable to the
mean on this 4-core box under concurrent load, so the JevX numbers should be
read as an order of magnitude, not as a ranking between A and F. The one clear
outlier is sst2's P4 at 10.827±18.874 ms; that cell is a 2.7 M–13.5 M-parameter
JevX stack run through `hq.JevXStack`, and its own variance is the largest
recorded in the matrix. It is reported as measured.

## 7. State-only diagnostic — no state-only defect found

Run on cell C's trained stack over 64 test rows, per dataset:

| quantity | synthetic | ag_news | sst2 | ravdess |
|---|---|---|---|---|
| state-only logit spread, mean | 3.1441 | 1.7735 | 1.3110 | 0.7918 |
| state-only logit spread, min | 1.3129 | 0.5393 | 0.0135 | 0.3180 |
| state-only logit spread, max | 4.5398 | 2.7596 | 3.7560 | 1.3196 |
| **state-only rows with all K logits equal** | **0** | **0** | **0** | **0** |
| max abs diff, `condition_options=False` vs `set_interaction_scale(0.0)` | **0.0** | **0.0** | **0.0** | **0.0** |
| conditioned logit spread, mean | 8.5733 | 6.3479 | 1.4612 | 3.4248 |

The degenerate duplicate-option tie is confirmed fixed on the real stack: with
`condition_options=False` the K state-only logits discriminate on every row of
every dataset, and the two routes to the state-only scoring function (the
per-call flag and the scale buffer) agree bit-for-bit. No new file was added
under `tests/unit/test_state_only_path.py` because no state-only defect was
found; the one test that carried the now-obsolete "real-wrapper Stage A is
degenerate" caveat is repaired in place (see §9).

## 8. A defect found in the curriculum — reported, not worked around

**`run_curriculum`'s Stage B does not re-enable the option pathway.** Measured,
not inferred.

`_CurriculumModelView.condition_options` is assigned exactly once, in its
constructor, from `curriculum.stage_a_epochs < 1`
(`jevx/training/curriculum.py:647`, `:1001`). `run_curriculum`'s per-stage
controller (`_StageController.advance`) pushes λ into
`head.set_interaction_scale` and never touches the view's flag. So for any
`stage_a_epochs >= 1` — every configuration in this matrix — the model is called
with `condition_options=False` on **every** step of **both** stages, and
`DecisionHead._residual_is_active(conditioning_enabled=False)` returns False
regardless of the scale buffer. The λ ramp is recorded and applied to the buffer
but the residual is never placed in the graph.

Three independent measurements agree:

1. **The λ series is correct but inert.** Cell E's recorded ramp is
   `a1: 0.00→0.00, a2: 0.00→0.00, b1: 0.00→0.48, b2: 0.52→1.00`, and
   `head.get_interaction_scale()` reads 1.0 after the run.
2. **The option pathway never receives a gradient.** In E's and F's epoch-1
   *and* final-epoch grad buckets, `option_interaction` and `schema` are absent
   (the post-accumulate hook only fires on an actual accumulation) and
   `option_pathway` equals `fusion` alone. Cell C, at λ=1 from step 0, shows
   `option_interaction` 27.766 and `schema` 47.666 in the same bucket.
3. **The weights do not move.** After a full 2+2 curriculum on the synthetic
   control, `bilinear_weight` is bit-for-bit identical to its initialisation
   (`max|Δ| = 0.0`), `bilinear_weight.grad is None`, and every
   `schema_encoder` parameter has `grad is None`, while `config.conditioning_enabled`
   and `head.conditioning_enabled` both read `True`.

**Consequence for this report.** Cells E and F are "state-only for all four
epochs, plus a λ ramp that is recorded but has no effect on the forward". E is
therefore a near-replicate of B trained through `run_curriculum` (synthetic
0.9508 vs 0.9505; sst2 0.7007 vs 0.7014), and F = E plus a gate that compares
the two modes of a model whose interaction weights never left their random
initialisation. That is why the gate fires so often on F (12/20) and never on D
(2/20): on F the "conditioned" mode is scoring through an untrained bilinear
residual, and on D it is scoring through a trained one. The RAVDESS result
(F − C = +6.92 pt, 5/5 seeds) is real, but part of it is the curriculum
deactivating the residual rather than the curriculum *staging* it. The
hypothesis' claim (b) — that the option pathway needs a staged ramp — is
untested by this matrix, because the staging never happens.

The file is in `jevx/training/**`, which this task does not own; it is reported
here for the owner. The one-line fix shape is to have `_StageController.advance`
set `view.condition_options = (stage != STAGE_A)`, which is what the module
docstring at `curriculum.py:22–28` already describes.

A second, smaller observation, also reported rather than worked around: with the
per-call flag `False` the fusion still receives the option rows and the
duplicate-option tie stays armed, whereas `condition_options=False` on
`JevXForDecision.forward` removes the schema rows before fusion and passes
`duplicate_tie=False`. The scoring is the same here (distinct schema rows, so
the tie never fires), and the measured effect on this matrix is nil.

> **SUPERSEDED 2026-09-26 - this defect is FIXED and E/F have been re-run. See
> §11.** `_CurriculumModelView.condition_options` is no longer a construction-
> time constant: `_StageController` now owns the per-stage flag and writes it
> into the view once per epoch and before every training step
> (`curriculum.py:889`, `:900`, `:916`, `:925`), and `EpochRecord.condition_options`
> records what each epoch actually ran under. The four E/F rows above, the E/F
> rows in §1-§4 and the §9 E/F entries are superseded by §11.2-§11.9. Cells
> A-D and the P1/P2/P4 references are NOT affected and stand as measured.

## 9. Verdict per dataset — does JevX-Next match or beat the MLP probe?

| dataset | best JevX-Next cell | its acc | P2 acc | gap | paired seeds favouring JevX-Next |
|---|---|---|---|---|---|
| synthetic control | E (0.9508) / B (0.9505) / F (0.9497) | 0.9508 | 0.9616 | **−1.08 pt** | 0/5 (B − P2) |
| sst2 | B (0.7014) | 0.7014 | 0.6966 | +0.48 pt | 2/5 — **indistinguishable**, not a win |
| ag_news | F (0.8332) | 0.8332 | 0.8361 | **−0.29 pt** | 3/5 (F − E); B − P2 1/5 |
| ravdess | F (0.3068) | 0.3068 | 0.3632 | **−5.64 pt** | 0/5 (F − C 5/5, but F − P2 0/5) |

**JevX-Next does not match the MLP probe on three of the four datasets, and on
the fourth (sst2) the MLP probe does not separate from the linear probe either
(P1 0.7000 vs P2 0.6966), so there is no probe gap there to close.** The
hypothesis' two named causes are both measured and neither is the dominant one:

* **(a) missing nonlinear state capacity — not supported.** The MLP state
  pathway buys +0.52 to +0.14 pt over the interaction-off arm on the three
  text/synthetic datasets and +6.96 pt on RAVDESS *relative to C* (A vs C, the
  linear-vs-mlp contrast with the interaction on in both, is +0.08 / −0.13 /
  +0.71 / **+2.72** pt: the mlp pathway is ahead only on ravdess, 4/5 seeds), but cell B (nonlinear, interaction
  off) is still 1.11 pt (synthetic, 0/5) and 5.60 pt (ravdess, 0/5) below P2.
  The state pathway is 1.3%–4.2% of the stack's parameters, and its gradient
  norm in cell B is 25–70x what the linear pathway gets, so capacity and
  gradient flow are both available and neither closes the gap.
* **(b) joint-optimization inefficiency — supported as a description of the
  LINEAR state pathway (1:82 to 1:274 gradient ratio, no growth over the budget)
  and not the binding constraint.** The dominant cost is the rest of the stack:
  76x (D=128) and 24x (D=768) P2's parameter budget, 98.8%/95.8% of it in the
  question encoder, schema encoder and fusion — all of which encode a quantity
  that is constant for a whole dataset, and all of which are refit from scratch
  in 4 epochs. The signature is visible in the first-epoch train loss on RAVDESS:
  6.8857 (A) and 3.2286 (C) against 2.0642 (B) and 2.0451 (P2).
* **Option conditioning carries no information on top of the state path on any
  of these four datasets** (C ≤ B everywhere, −6.96 pt and 0/5 on ravdess), which
  is what the design review's own collapse argument predicts for a single fixed
  question and a single fixed schema. The gate's value is that it *detects*
  that and flips: 12/20 flips, +2.56 pt and −0.1001 NLL on ravdess (4/5 and
  5/5 seeds), and never worse than −0.11 pt anywhere.
* **The curriculum cannot be credited from this run** (§8): its Stage B never
  activates the option pathway, so the staged-ramp claim is untested here.

No claim of any kind is made about datasets, backbones or tasks outside the four
executed here, and no ranking is declared between cells whose per-seed ranges
overlap.

## 10. Reproduce

```
python benchmarks/run_head_ablation.py --datasets ravdess ag_news synthetic --seeds 0,1,2,3,4
python benchmarks/run_head_ablation.py --datasets sst2 --seeds 0,1,2,3,4
python -m pytest tests/unit -q
```

Artifacts: `benchmarks/results/head_ablation_{synthetic,ag_news,sst2,ravdess}.json`
(per-seed rows, convergence curves, gradient norms, parameter groups, wiring,
gate measurements, latency), `benchmarks/results/head_ablation_all.json` (merged
index), `benchmarks/cache/head_quality/sst2_features.npz` (created by this run).


---

## 11. Re-run of cells E and F after the Stage-B fix (supersedes the E/F rows in §1-§4)

APPENDED 2026-09-26, after the defect reported in §8 was **fixed** in
`jevx/training/curriculum.py`. Every number in §11.2-§11.8 is copied from
`benchmarks/results/head_ablation_curriculum_fixed.json` and
`benchmarks/results/curriculum_fixed_rerun/head_ablation_<dataset>.json`, written by the
**unmodified** harness `benchmarks/run_head_ablation.py` on 2026-09-26. Nothing here is
estimated or hand-typed.

**What was re-run and what was not.** Cells **E** and **F** only, on all four
datasets x seeds 0-4 (40 cell-runs, 0 failures), under the identical protocol:
same frozen feature matrices, same splits, same seeds, same optimizer, same
`lr`, same 4 epochs, same batch size, same loss, `stage_a_epochs=2` /
`stage_b_epochs=2`, `lambda 0.0 -> 1.0` linear. The re-run wrote to a separate
directory (`benchmarks/results/curriculum_fixed_rerun/`) precisely so the
2026-09-26 per-dataset files keep their A-D and P1/P2/P4 rows byte-for-byte.
Cells A-D and the three references were **not** re-run and **not** edited.

### 11.1 The fix, and what it changes

**The defect, restated.** `_CurriculumModelView.condition_options` was
assigned once in the constructor from `stage_a_epochs < 1` and never updated when the
stage flipped to B, so every Stage-B training step still called the model with
`condition_options=False`. That per-call flag, not the scale buffer, is what gates
the residual (`DecisionHead._residual_is_active(False)` is False whatever λ holds),
so the λ ramp was recorded and applied to a buffer the forward never read.

The fix moves the flag into the object that already owns the scale, so the two
are driven from one place and both are pure functions of the stage:

* `_StageController.condition_options` (`curriculum.py:889`) - a property that is
  `False` in Stage A and `True` in Stage B;
* `_StageController.apply_stage()` (`curriculum.py:900`) pushes it into the view
  through `_CurriculumModelView.set_condition_options()` (`curriculum.py:670`);
* it is called from `begin_epoch()` (`curriculum.py:916`, so an epoch’s
  evaluation inherits its training flag) and from `advance()`
  (`curriculum.py:925`, before every training step);
* `EpochRecord.condition_options` (`curriculum.py:398`, recorded at
  `curriculum.py:1131`) now carries the flag **read back off the view**, so the
  record is evidence rather than a restatement of intent, and it survives the
  save/load round trip.

The public API is unchanged: `run_curriculum`, `CurriculumConfig`
and `CurriculumHistory` keep their signatures; the one new `EpochRecord`
field is additive with a default, and the history format key is unchanged.
Behaviour change only: `run_curriculum` now builds the view without a
construction-time flag (`curriculum.py:1075`), and the first `begin_epoch`
puts the Stage-A flag in force before any batch reaches `forward()`.

**Stage A is untouched by the fix, and the re-run shows it.** E’s first-epoch
train loss is identical to the superseded run on every dataset (0.5795 / 0.7432 /
0.6084 / 2.0642), and so is the whole epoch-1 gradient bucket in §11.6 - the
option-pathway columns are `absent` there in the new run too, because Stage A is
state-only in both.

**Four regression tests** in `tests/unit/test_curriculum.py` (lines 1400, 1513,
1561, 1608) would each have caught the defect on its own; all four were run
against the defect restored in-place and **all four fail** (§11.2-§11.8 above).

### 11.2 Old (defective) vs new (fixed) E/F — accuracy, 5 seeds, mean±std

| dataset | E old | **E new** | F old | **F new** | C (not re-run) | B (not re-run) | P2 |
|---|---|---|---|---|---|---|---|
| synthetic control (D=128, K=4, 3000/1000/1500) | 0.9508±0.0023 | **0.9500±0.0024** | 0.9497±0.0026 | **0.9505±0.0024** | 0.9453±0.0071 | 0.9505±0.0035 | **0.9616±0.0022** |
| ag_news, text, 4-way (D=128, K=4, 3000/1000/1500) | 0.8321±0.0062 | **0.8316±0.0080** | 0.8332±0.0081 | **0.8319±0.0082** | 0.8313±0.0123 | 0.8328±0.0082 | **0.8361±0.0031** |
| sst2, text, binary (D=128, K=2, 3000/1000/872) | 0.7007±0.0135 | **0.7014±0.0138** | 0.7007±0.0135 | **0.7011±0.0142** | 0.7000±0.0122 | 0.7014±0.0143 | **0.6966±0.0164** |
| ravdess, audio, 8-way (D=768, K=8, 700/240/500) | 0.2812±0.0579 | **0.3280±0.0364** | 0.3068±0.0511 | **0.3268±0.0360** | 0.2376±0.0274 | 0.3072±0.0511 | **0.3632±0.0217** |

### 11.3 Old (defective) vs new (fixed) E/F — test NLL, 5 seeds, mean±std

| dataset | E old | **E new** | F old | **F new** | C | B | P2 |
|---|---|---|---|---|---|---|---|
| synthetic control (D=128, K=4, 3000/1000/1500) | 0.1873 | **0.1841** | 0.1881 | **0.1832** | 0.1799 | 0.1860 | - |
| ag_news, text, 4-way (D=128, K=4, 3000/1000/1500) | 0.4922 | **0.4857** | 0.4866 | **0.4849** | 0.4975 | 0.4857 | - |
| sst2, text, binary (D=128, K=2, 3000/1000/872) | 0.5825 | **0.5833** | 0.5825 | **0.5832** | 0.5865 | 0.5840 | - |
| ravdess, audio, 8-way (D=768, K=8, 700/240/500) | 1.8638 | **1.7517** | 1.7637 | **1.7320** | 1.8722 | 1.7640 | - |

### 11.4 Per-seed accuracy, old vs new (seed order 0,1,2,3,4)

| dataset | cell | old acc | **new acc** | old NLL | **new NLL** |
|---|---|---|---|---|---|
| synthetic | E | 0.9480 0.9520 0.9507 0.9540 0.9493 | **0.9480 0.9527 0.9487 0.9527 0.9480** | 0.1895 0.1833 0.1831 0.1912 0.1892 | **0.1842 0.1785 0.1841 0.1824 0.1912** |
| synthetic | F | 0.9453 0.9520 0.9507 0.9513 0.9493 | **0.9480 0.9527 0.9480 0.9527 0.9513** | 0.1889 0.1833 0.1831 0.1939 0.1913 | **0.1842 0.1779 0.1846 0.1815 0.1880** |
| ag_news | E | 0.8247 0.8273 0.8360 0.8327 0.8400 | **0.8193 0.8333 0.8367 0.8287 0.8400** | 0.5280 0.5078 0.4708 0.4927 0.4618 | **0.5166 0.5004 0.4681 0.4855 0.4578** |
| ag_news | F | 0.8207 0.8327 0.8373 0.8327 0.8427 | **0.8193 0.8340 0.8373 0.8287 0.8400** | 0.5190 0.4953 0.4664 0.4927 0.4596 | **0.5166 0.4996 0.4682 0.4821 0.4578** |
| sst2 | E | 0.7190 0.6995 0.7087 0.6904 0.6858 | **0.7179 0.7030 0.7087 0.6961 0.6812** | 0.5677 0.5806 0.5744 0.5955 0.5944 | **0.5677 0.5832 0.5770 0.5895 0.5990** |
| sst2 | F | 0.7190 0.6995 0.7087 0.6904 0.6858 | **0.7179 0.7030 0.7087 0.6961 0.6800** | 0.5677 0.5806 0.5744 0.5955 0.5944 | **0.5677 0.5832 0.5770 0.5895 0.5985** |
| ravdess | E | 0.2460 0.2360 0.3540 0.3340 0.2360 | **0.2960 0.2840 0.3380 0.3660 0.3560** | 1.9013 1.9175 1.8108 1.7990 1.8902 | **1.7128 1.8120 1.8205 1.7081 1.7052** |
| ravdess | F | 0.3120 0.2400 0.3340 0.3720 0.2760 | **0.2960 0.2840 0.3320 0.3660 0.3560** | 1.8466 1.8545 1.6846 1.6675 1.7651 | **1.7128 1.8120 1.7218 1.7081 1.7052** |

### 11.5 The decisive question — does the curriculum now pay? (paired, same seeds)

| dataset | comparison | mean Δacc (pt) | seeds favouring left | per-seed Δacc (pt) | mean ΔNLL | 5-seed acc ranges | ranges overlap? |
|---|---|---|---|---|---|---|---|
| synthetic | E − C (mlp+interaction) | **+0.47** | 3/5 | -0.13 +0.53 -0.27 +1.93 +0.27 | +0.0042 | [0.9480, 0.9527] vs [0.9333, 0.9513] | **yes** |
| synthetic | E − B (mlp, no interaction) | **-0.05** | 3/5 | +0.27 +0.07 -0.60 +0.13 -0.13 | -0.0019 | [0.9480, 0.9527] vs [0.9453, 0.9547] | **yes** |
| synthetic | F − C (mlp+interaction) | **+0.52** | 3/5 | -0.13 +0.53 -0.33 +1.93 +0.60 | +0.0033 | [0.9480, 0.9527] vs [0.9333, 0.9513] | **yes** |
| synthetic | F − B (mlp, no interaction) | **-0.00** | 4/5 | +0.27 +0.07 -0.67 +0.13 +0.20 | -0.0027 | [0.9480, 0.9527] vs [0.9453, 0.9547] | **yes** |
| ag_news | E − C (mlp+interaction) | **+0.03** | 3/5 | +0.73 +0.60 -0.33 -1.40 +0.53 | -0.0118 | [0.8193, 0.8400] vs [0.8120, 0.8427] | **yes** |
| ag_news | E − B (mlp, no interaction) | **-0.12** | 1/5 | -0.13 +0.07 -0.07 -0.20 -0.27 | +0.0000 | [0.8193, 0.8400] vs [0.8207, 0.8427] | **yes** |
| ag_news | F − C (mlp+interaction) | **+0.05** | 3/5 | +0.73 +0.67 -0.27 -1.40 +0.53 | -0.0127 | [0.8193, 0.8400] vs [0.8120, 0.8427] | **yes** |
| ag_news | F − B (mlp, no interaction) | **-0.09** | 1/5 | -0.13 +0.13 +0.00 -0.20 -0.27 | -0.0008 | [0.8193, 0.8400] vs [0.8207, 0.8427] | **yes** |
| sst2 | E − C (mlp+interaction) | **+0.14** | 3/5 | -0.23 +0.23 +1.03 +0.46 -0.80 | -0.0032 | [0.6812, 0.7179] vs [0.6892, 0.7202] | **yes** |
| sst2 | E − B (mlp, no interaction) | **-0.00** | 2/5 | -0.23 +0.23 -0.11 +0.34 -0.23 | -0.0007 | [0.6812, 0.7179] vs [0.6835, 0.7202] | **yes** |
| sst2 | F − C (mlp+interaction) | **+0.11** | 3/5 | -0.23 +0.23 +1.03 +0.46 -0.92 | -0.0033 | [0.6800, 0.7179] vs [0.6892, 0.7202] | **yes** |
| sst2 | F − B (mlp, no interaction) | **-0.02** | 2/5 | -0.23 +0.23 -0.11 +0.34 -0.34 | -0.0008 | [0.6800, 0.7179] vs [0.6835, 0.7202] | **yes** |
| ravdess | E − C (mlp+interaction) | **+9.04** | 5/5 | +4.20 +8.20 +7.80 +10.80 +14.20 | -0.1205 | [0.2840, 0.3660] vs [0.2020, 0.2600] | **NO** |
| ravdess | E − B (mlp, no interaction) | **+2.08** | 3/5 | -1.80 +4.40 +0.40 -0.60 +8.00 | -0.0122 | [0.2840, 0.3660] vs [0.2400, 0.3720] | **yes** |
| ravdess | F − C (mlp+interaction) | **+8.92** | 5/5 | +4.20 +8.20 +7.20 +10.80 +14.20 | -0.1402 | [0.2840, 0.3660] vs [0.2020, 0.2600] | **NO** |
| ravdess | F − B (mlp, no interaction) | **+1.96** | 2/5 | -1.80 +4.40 -0.20 -0.60 +8.00 | -0.0320 | [0.2840, 0.3660] vs [0.2400, 0.3720] | **yes** |

### 11.6 The option pathway is now in the graph (per-module grad norms, seed 0)

| dataset | cell | bucket | state_pathway | option_interaction | schema | fusion | option_pathway | total |
|---|---|---|---|---|---|---|---|---|
| synthetic | E old | epoch1 | 7.389 | **absent** | **absent** | 5.030 | 5.030 | 24.231 |
| synthetic | E **new** | epoch1 | 7.389 | **absent** | **absent** | 5.030 | 5.030 | 24.231 |
| synthetic | E old | final | 7.574 | **absent** | **absent** | 5.346 | 5.346 | 24.410 |
| synthetic | E **new** | final | 7.557 | 11.285 | 18.639 | 6.638 | 22.778 | 32.955 |
| synthetic | F old | epoch1 | 7.389 | **absent** | **absent** | 5.030 | 5.030 | 24.231 |
| synthetic | F **new** | epoch1 | 7.389 | **absent** | **absent** | 5.030 | 5.030 | 24.231 |
| synthetic | F old | final | 7.574 | **absent** | **absent** | 5.346 | 5.346 | 24.410 |
| synthetic | F **new** | final | 7.557 | 11.285 | 18.639 | 6.638 | 22.778 | 32.955 |
| ag_news | E old | epoch1 | 7.098 | **absent** | **absent** | 4.680 | 4.680 | 24.091 |
| ag_news | E **new** | epoch1 | 7.098 | **absent** | **absent** | 4.680 | 4.680 | 24.091 |
| ag_news | E old | final | 7.189 | **absent** | **absent** | 4.898 | 4.898 | 24.221 |
| ag_news | E **new** | final | 7.191 | 11.295 | 18.629 | 6.193 | 22.648 | 32.800 |
| ag_news | F old | epoch1 | 7.098 | **absent** | **absent** | 4.680 | 4.680 | 24.091 |
| ag_news | F **new** | epoch1 | 7.098 | **absent** | **absent** | 4.680 | 4.680 | 24.091 |
| ag_news | F old | final | 7.189 | **absent** | **absent** | 4.898 | 4.898 | 24.221 |
| ag_news | F **new** | final | 7.191 | 11.295 | 18.629 | 6.193 | 22.648 | 32.800 |
| sst2 | E old | epoch1 | 6.633 | **absent** | **absent** | 3.903 | 3.903 | 23.559 |
| sst2 | E **new** | epoch1 | 6.633 | **absent** | **absent** | 3.903 | 3.903 | 23.559 |
| sst2 | E old | final | 6.637 | **absent** | **absent** | 4.033 | 4.033 | 23.528 |
| sst2 | E **new** | final | 6.635 | 11.326 | 18.685 | 5.612 | 22.559 | 32.345 |
| sst2 | F old | epoch1 | 6.633 | **absent** | **absent** | 3.903 | 3.903 | 23.559 |
| sst2 | F **new** | epoch1 | 6.633 | **absent** | **absent** | 3.903 | 3.903 | 23.559 |
| sst2 | F old | final | 6.637 | **absent** | **absent** | 4.033 | 4.033 | 23.528 |
| sst2 | F **new** | final | 6.635 | 11.326 | 18.685 | 5.612 | 22.559 | 32.345 |
| ravdess | E old | epoch1 | 16.145 | **absent** | **absent** | 21.913 | 21.913 | 67.137 |
| ravdess | E **new** | epoch1 | 16.145 | **absent** | **absent** | 21.913 | 21.913 | 67.137 |
| ravdess | E old | final | 16.550 | **absent** | **absent** | 22.381 | 22.381 | 67.894 |
| ravdess | E **new** | final | 16.552 | 27.783 | 47.697 | 31.489 | 63.548 | 90.242 |
| ravdess | F old | epoch1 | 16.145 | **absent** | **absent** | 21.913 | 21.913 | 67.137 |
| ravdess | F **new** | epoch1 | 16.145 | **absent** | **absent** | 21.913 | 21.913 | 67.137 |
| ravdess | F old | final | 16.550 | **absent** | **absent** | 22.381 | 22.381 | 67.894 |
| ravdess | F **new** | final | 16.552 | 27.783 | 47.697 | 31.489 | 63.548 | 90.242 |

### 11.7 What the run itself recorded, per epoch (seed 0, both cells)

| dataset | stage | epoch | λ first → last | `condition_options` |
|---|---|---|---|---|
| synthetic | a | 1 | 0.00 → 0.00 | **False** |
| synthetic | a | 2 | 0.00 → 0.00 | **False** |
| synthetic | b | 3 | 0.00 → 0.49 | **True** |
| synthetic | b | 4 | 0.51 → 1.00 | **True** |
| ag_news | a | 1 | 0.00 → 0.00 | **False** |
| ag_news | a | 2 | 0.00 → 0.00 | **False** |
| ag_news | b | 3 | 0.00 → 0.49 | **True** |
| ag_news | b | 4 | 0.51 → 1.00 | **True** |
| sst2 | a | 1 | 0.00 → 0.00 | **False** |
| sst2 | a | 2 | 0.00 → 0.00 | **False** |
| sst2 | b | 3 | 0.00 → 0.49 | **True** |
| sst2 | b | 4 | 0.51 → 1.00 | **True** |
| ravdess | a | 1 | 0.00 → 0.00 | **False** |
| ravdess | a | 2 | 0.00 → 0.00 | **False** |
| ravdess | b | 3 | 0.00 → 0.48 | **True** |
| ravdess | b | 4 | 0.52 → 1.00 | **True** |

(The superseded run recorded the identical lambda series with no condition_options key at all: the flag did not exist as a recorded quantity. a1/a2 are the two Stage-A epochs, b1/b2 the two Stage-B epochs.)

### 11.8 Gate flips on the fixed run (calibration split only; test scored after the verdict)

| dataset | F old verdict per seed | F **new** verdict per seed | flips old | flips new |
|---|---|---|---|---|
| synthetic | STATE cond cond STATE STATE | **cond STATE STATE STATE STATE** | 3/5 | 4/5 |
| ag_news | STATE STATE STATE cond STATE | **cond STATE STATE STATE cond** | 4/5 | 3/5 |
| sst2 | cond cond cond cond cond | **cond cond cond cond STATE** | 0/5 | 1/5 |
| ravdess | STATE STATE STATE STATE STATE | **cond cond STATE cond cond** | 5/5 | 1/5 |

**Flip frequency F: old 12/20, new 9/20.**

### 11.9 The corrected conclusion about the curriculum

§8 stated that the hypothesis’ claim (b) - that the option pathway needs a
staged ramp - was **untested** by this matrix. That was accurate about the
2026-09-26 run, and it is no longer an open item: the fixed re-run tests it.
The old text also attributed RAVDESS `F - C = +6.92 pt` partly to the curriculum
**deactivating** the residual. That explanation is withdrawn. Stage B now ramps a
residual that is genuinely in the graph (§11.6, §11.7), and the same comparison
measured on the fixed run is **+8.92 pt, 5/5 seeds**.

**What the fixed run shows.**

* **The curriculum pays on RAVDESS and only there.** Against C (mlp + interaction,
  no curriculum): **E +9.04 pt** and **F +8.92 pt**, 5/5 seeds on both, on NLL
  **-0.1205** and **-0.1402**, and the 5-seed accuracy ranges do **not** overlap
  (E/F [0.2840, 0.3660] against C [0.2020, 0.2600]). That is the same comparison
  the superseded run put at +4.36 and +6.92 pt, on a model whose residual was
  never trained.
* **On synthetic, ag_news and sst2 there is no measured effect.** Every paired
  delta is inside one seed std, flips sign across seeds, and the 5-seed ranges
  overlap: E - C is +0.47 / +0.03 / +0.14 pt, F - C is +0.52 / +0.05 / +0.11 pt.
  No winner is declared from those three, and none was declared before.
* **Against B (mlp, interaction off) the curriculum does not win anywhere.** On
  RAVDESS E is +2.08 pt and F +1.96 pt, but with overlapping ranges and only 3/5
  and 2/5 seeds - a mean, not a seed-consistent win. On the other three datasets
  it is a dead heat (|delta| <= 0.12 pt, overlapping ranges). The honest reading
  of the RAVDESS column is that the curriculum **recovers most of what the
  interaction costs** there (C sits 6.96 pt below B; E sits 2.08 pt above B) and
  still does not pass the interaction-free arm.
* **The curriculum does not hurt anywhere that was measured.** The worst paired
  accuracy delta is ag_news E - B at **-0.12 pt** (1/5 seeds) and the worst NLL
  delta is synthetic E - C at **+0.0042** (worse). One number moved the wrong way
  and it is stated: on RAVDESS E’s mean **train** loss rose, 1.7076 -> 1.7756,
  while its test accuracy rose 0.2812 -> 0.3280 and its test NLL fell 1.8638 ->
  1.7517. The residual now costs training fit and pays on held-out data; under
  the defect it cost nothing because it was not there.
* **The gate’s behaviour changed with it, and the old explanation goes with it.**
  F’s flip frequency is **9/20** on the fixed run against 12/20 on the superseded
  one, and on RAVDESS it falls from 5/5 flips to 1/5 - the gate now mostly keeps
  `conditioned`, because ΔNLL on a model whose residual was actually trained is
  small, which is the same reason D only flips 2/20. The old reading ("F flips
  often because it scores through an untrained residual") no longer holds.

**What is unchanged, restated without re-litigation.**

* **Q1 - is B (mlp state path, interaction OFF) ≈ P2 (MLP probe)? No.** synthetic
  -1.11 pt (0/5 seeds), ag_news -0.33 pt (1/5), sst2 +0.48 pt (2/5), RAVDESS
  -5.60 pt (0/5).
* **Q2 - is C (mlp + interaction) > B (mlp, interaction OFF)? No. C <= B on all four
  datasets.** synthetic -0.52 pt (1/5), ag_news -0.15 pt (2/5), sst2 -0.14 pt (1/5),
  RAVDESS -6.96 pt (0/5, +0.1083 NLL).

Both were measured with **no curriculum** in any cell, so neither depends on the
Stage-B fix, and neither is re-measured here. The §9 verdict table’s E and F
entries are superseded by §11; its A, B, C, D, P1, P2 and P4 entries stand as
measured.

### 11.10 Reproducing the re-run

```
python benchmarks/run_head_ablation.py --datasets synthetic ag_news sst2 ravdess \
    --seeds 0,1,2,3,4 --cells E_curriculum F_gate_curriculum \
    --output-dir benchmarks/results/curriculum_fixed_rerun
python -m pytest tests/unit -q
```

Artifacts: `benchmarks/results/head_ablation_curriculum_fixed.json` (old vs new
E/F, per-seed rows, per-epoch records with the flag, gradient buckets, gate
verdicts, and the paired comparisons against the untouched B / C / A / D / P
rows), `benchmarks/results/curriculum_fixed_rerun/` (the four re-run payloads,
unchanged format). The four `benchmarks/results/head_ablation_<dataset>.json`
files are **left as the 2026-09-26 matrix wrote them**, E/F rows included, and are
superseded for E/F only by the two artifacts above. Note that
`curriculum_fixed_rerun/head_ablation_all.json§ is this re-run’s own index and
covers **E and F only** - it is not the full matrix index
(`benchmarks/results/head_ablation_all.json§, 9 cells, also untouched).

