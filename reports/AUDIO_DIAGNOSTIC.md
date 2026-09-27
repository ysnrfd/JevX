# AUDIO_DIAGNOSTIC — WAVE-3a: the A0–A6 grid on frozen RAVDESS features

Benchmark-only track. **Nothing under `jevx/**` was modified.** Every number
below was executed on this host; no cell is estimated, interpolated or
omitted. The decision tree is followed literally and the branch that fired is
named with its evidence, including the branch preconditions that did **not**
hold.

**Artifacts**
- Harness: `benchmarks/run_audio_diagnostic.py` (new; imports the read-only
  helpers from `benchmarks/run_head_ablation.py`, never edits them)
- Result: `benchmarks/results/audio_diagnostic_ravdess.json` (5 seeds × 10
  cells, `failures: {}`)
- §5 re-run: `benchmarks/results/audio_jevx_quality_v2.json`
- Append-only supersession: §11 of `reports/MODALITY_BENCHMARKS.md`

---

## 1. Environment, ids and shas

| | |
|---|---|
| Platform | Windows-11-10.0.26100-SP0, Intel64 Family 6 Model 78, **4 logical CPUs**, CUDA unavailable |
| Python / torch / transformers / datasets / numpy | 3.14.3 / 2.14.0+cpu / 5.16.1 / 4.4.1 / 2.3.5, dtype float32, `torch_threads` 2 |
| Version control | **the working tree is not a git repository**, so no commit sha exists; the shas below are the file digests that stand in for one |
| Backbone (frozen) | `superb/wav2vec2-base-superb-ks`, revision `372e0486cd83e6f0c05c20a27262e9ca09450d24`, 94,371,712 frozen params, load report: `classifier.*` / `projector.*` / `layer_weights` UNEXPECTED (all ignorable) |
| Dataset | `Codec-SUPERB/RAVDESS`, revision `46aa36c5fdeb1b6d63b746b74fde528a540044b2`, 1,440 clips, 8 filename-prefix classes, 16 kHz, trimmed to 3.0 s |
| Frozen features (grid) | `benchmarks/cache/head_quality/ravdess_features.npz`, sha256 `75aaecf3f0447d06d89870226f717cf6ebb5ef8b2684ad737d61b5d48c8dec00` (cache HIT — no re-featurization) |
| sha256 harness | `6d50bee04314eecda23cfb6416e91df7e160f04fd0288d302f2f13c8458c834d` |
| sha256 grid result | `4834ef16d24880cf3de5c3fcbedc3729c8f705adb0ab6717c34d2ad7bf34e1ac` |
| sha256 §5 v1 (superseded) | `8067270c1bdde6d6bb9033bb7b1a9851e77b77aae19e37a0c6054668a1cc623d` |
| sha256 §5 v2 (this pass) | `1e970ac7bfd7a5d2d9192db03f4e3259c1bb70a1db090e347bde9292e02a4230` |

**Host caveat that affects every wall time in this report.** The box has 4
logical CPUs and was shared with at least four other concurrent Python jobs
during the run. Timings taken while the load was high are inflated by up to
~15× (the same A0 cell took 287.2 s in the contended window and 27.3 s
mean-per-seed in the clean one). The **accuracy/NLL/ECE numbers are not
affected** — they are deterministic given the seeds — but the wall times are
host-load observations, not machine benchmarks. Wall times quoted below are
from the clean (final) run.

## 2. Protocol (identical for every cell)

Frozen wav2vec2 features featurized once; no backbone in the loop. Splits
700 train / 240 calibration / 500 test, stratified, split seed 0, disjoint.
Optimizer `torch.optim.AdamW` (the `HeadTrainer` default), lr 1e-3,
grad-clip 1.0, batch 64, loss `jevx.training.choice_loss` (masked
cross-entropy), 4 epochs unless the cell overrides it. **5 seeds, 0–4, for
every cell.** Metrics: accuracy, macro-F1, NLL, ECE(15), Brier(15) via
`bench_common.metrics_from_logits`.

Three classes of claim are enforced explicitly at run time, never by relying
on a config default, and are asserted before training: "conditioning OFF"
(`config.conditioning_enabled=False` **and** `head.conditioning_enabled=False`
**and** `head.set_interaction_scale(0.0)`), the state-pathway selector that
was actually built, and the state-pathway initialization actually written.

**Verification that the grid is the A–F matrix's protocol, not an
approximation.** Three cells in this run are byte-identical reproductions of
already-published A–F matrix cells on the same features/splits/seeds:

| this run | A–F matrix cell | both |
|---|---|---|
| `R_p1_linear_probe` 0.3472 (0.3540, 0.3260, 0.3600, 0.3340, 0.3620) | `p1_linear` | identical per-seed values |
| `R_p2_mlp_probe` 0.3632 (0.3760, 0.3260, 0.3620, 0.3780, 0.3740) | `p2_mlp` | identical per-seed values |
| `R_p4_bypass_on` 0.2720 (0.2260, 0.1960, 0.3000, 0.3180, 0.3200) | `p4_state_only_bypass` | identical per-seed values |
| `A0_shipped_stack` 0.2376 (0.2540, 0.2020, 0.2600, 0.2580, 0.2140) | `C_mlp_interact` | identical per-seed values |

Source of the comparison: `benchmarks/results/head_ablation_ravdess.json`.

**Test-split reference points (measured on this split, n=500):** majority class
67/500 = **0.1340**, chance = 0.1250.

## 3. The A0–A6 grid — results (5 seeds, 0–4, one run, `failures: {}`)

Accuracy is mean ± sample std over the 5 seeds; the per-seed values are listed
so overlap is visible. "Δ vs A0" and "Δ vs A2" are **paired** per-seed
differences on the identical test split (n = 5 paired seeds each, with the
count of seeds on which the delta is positive).

| cell | what it is | acc (mean ± std) | per-seed accuracy (s0…s4) | macro-F1 | NLL ↓ | ECE(15) ↓ | Brier(15) ↓ | trainable params | epochs | train loss first → last | wall s/seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A0** | reference: the shipped stack (JevX head, interaction ON, `state_pathway="mlp"`) | **0.2376 ± 0.0274** | 0.2540, 0.2020, 0.2600, 0.2580, 0.2140 | 0.1436 | 1.8722 | 0.0402 | 0.8294 | 14,169,857 | 4 | 3.336 → 1.982 | 27.3 |
| **A1** | `conditioning_mode="off"` + state-only head (P4 bypass equivalent) | **0.4196 ± 0.0343** | 0.3840, 0.4140, 0.4580, 0.3900, 0.4520 | 0.3846 | 1.5877 | 0.0755 | 0.7177 | 13,579,265 | 4 | 2.042 → 1.474 | 3.2 |
| **A2** | state path ONLY, an exact `nn.Linear(D,K)`: `z_k = sᵀW[:,k]+b[k]`, interaction OFF, shipped zero init | **0.3448 ± 0.0305** | 0.3700, 0.3200, 0.3080, 0.3480, 0.3780 | 0.2788 | 1.7075 | 0.0983 | 0.7657 | 13,579,265 | 4 | 2.119 → 1.599 | 11.4 |
| **A3** | A2 architecture with A1's optimizer / lr / epochs / batch / loss | **0.3448 ± 0.0305** | 0.3700, 0.3200, 0.3080, 0.3480, 0.3780 | 0.2788 | 1.7075 | 0.0983 | 0.7657 | 13,579,265 | 4 | 2.119 → 1.599 | 11.5 |
| **A4** | A2 + probe-style init of the state path (`nn.Linear` kaiming-uniform `a=√5` + uniform bias) | **0.2856 ± 0.0417** | 0.2200, 0.2800, 0.3280, 0.2860, 0.3140 | 0.2097 | 2.0132 | 0.1601 | 0.8571 | 13,579,265 | 4 | 11.402 → 2.456 | 11.4 |
| **A5** | A2 + 20 epochs (budget sensitivity) | **0.5416 ± 0.0138** | 0.5560, 0.5520, 0.5220, 0.5340, 0.5440 | 0.5278 | 1.3412 | 0.1415 | 0.6222 | 13,579,265 | **20** | 2.119 → 0.910 | 52.0 |
| **A6** | A2 with `decision_hidden_size` halved (768 → 384) | **0.3796 ± 0.0452** | 0.3520, 0.3940, 0.4280, 0.4080, 0.3160 | 0.3365 | 1.6438 | 0.0729 | 0.7477 | **5,167,745** | 4 | 2.071 → 1.670 | 5.7 |

Reference cells, same run, same protocol (context for the grid, not part of
A0–A6):

| cell | what it is | acc (mean ± std) | per-seed accuracy | macro-F1 | NLL | ECE(15) | Brier(15) | params |
|---|---|---|---|---|---|---|---|---|
| `R_p1_linear_probe` | the matrix's **P1** — a bare `nn.Linear(768,8)` on the POOLED features ("P1 exact") | 0.3472 ± 0.0162 | 0.3540, 0.3260, 0.3600, 0.3340, 0.3620 | 0.2870 | 1.9169 | 0.1692 | 0.8330 | 6,152 |
| `R_p2_mlp_probe` | the matrix's **P2** — bare MLP(768→768)→ReLU→(768→8) | 0.3632 ± 0.0217 | 0.3760, 0.3260, 0.3620, 0.3780, 0.3740 | 0.3080 | 1.7304 | 0.0907 | 0.7776 | 596,744 |
| `R_p4_bypass_on` | the matrix's **P4** — state-only bypass, interaction still ON | 0.2720 ± 0.0572 | 0.2260, 0.1960, 0.3000, 0.3180, 0.3200 | 0.1971 | 2.0982 | 0.1946 | 0.8686 | 13,579,265 |

### 3.1 Paired deltas (identical test split, n = 5 paired seeds)

| cell | Δ acc vs A0 | seeds positive | Δ acc vs A2 | seeds positive | Δ NLL vs A2 | seeds lower |
|---|---|---|---|---|---|---|
| A1 | **+0.1820 ± 0.0487** | 5/5 | **+0.0748 ± 0.0520** | 5/5 | −0.1198 | 1/5 |
| A2 | +0.1072 ± 0.0425 | 5/5 | — | — | — | — |
| A3 | +0.1072 ± 0.0425 | 5/5 | **+0.0000 ± 0.0000** | 0/5 | +0.0000 | 0/5 |
| A4 | +0.0480 ± 0.0527 | 4/5 | **−0.0592 ± 0.0611** | 1/5 | +0.3057 | 0/5 |
| A5 | **+0.3040 ± 0.0366** | 5/5 | **+0.1968 ± 0.0261** | 5/5 | −0.3664 | 0/5 |
| A6 | +0.1420 ± 0.0412 | 5/5 | +0.0348 ± 0.0735 | 3/5 | −0.0637 | 2/5 |
| R_p1 | +0.1096 ± 0.0274 | 5/5 | +0.0024 ± 0.0292 | 2/5 | +0.2094 | 5/5 |
| R_p2 | +0.1256 ± 0.0211 | 5/5 | +0.0184 ± 0.0235 | 4/5 | +0.0228 | 3/5 |
| R_p4 | +0.0344 ± 0.0532 | 3/5 | −0.0728 ± 0.0590 | 0/5 | +0.3907 | 5/5 |

Two-sample t on the 5 paired differences (df = 4), stated because the tree's
comparisons are paired: A5−A2 `t = 16.9`; A1−A2 `t = 3.22`; A4−A2
`t = −2.17`; A2−R_p1 `t = 0.25` (indistinguishable).

### 3.2 Wiring actually built (recorded, not assumed)

| cell | state pathway | interaction scale | `config.conditioning_enabled` | `head.conditioning_enabled` | decision width | state-path init |
|---|---|---|---|---|---|---|
| A0 | `mlp` / `MLPStatePathway` | 1.0 | True | True | 768 | shipped (W2 = 0 asserted) |
| A1 | `linear` | **0.0** | **False** | **False** | 768 | shipped |
| A2 / A3 / A5 | `linear` / `LinearProbePathway` | **0.0** | **False** | **False** | 768 | shipped (**zeros asserted**) |
| A4 | `linear` / `LinearProbePathway` | **0.0** | **False** | **False** | 768 | probe: `nn.Linear.reset_parameters`, fan_in 768, weight bound 0.0625 |
| A6 | `linear` / `LinearProbePathway` | **0.0** | **False** | **False** | **384** | shipped (zeros asserted) |

Non-degeneracy of the state-only path was checked, not assumed: for every
JevX cell the state-only logit spread has min > 0 and
`state_only_tied_rows = 0`, and `state_only_vs_zero_lambda_max_abs_diff = 0.0`
(the two ways of reaching the state-only function agree exactly). A4's epoch-1
train loss is 9.99–12.83 against 2.07–2.12 for A2 — the probe-style init
saturates the first cross-entropy step, which is the mechanism behind its
loss.

---

## 4. The decision tree, followed literally

**Branch 1 — "A2 ≈ A1 AND A4 ≫ A2 → initialization is the cause": NOT FIRED.**
The second predicate is false in the *opposite* direction. A4 = 0.2856 is
**5.9 pt BELOW** A2 = 0.3448 (paired −0.0592 ± 0.0611, negative on 4 of 5
seeds, t = −2.17), with NLL 2.0132 vs 1.7075 (worse on 5/5) and ECE 0.1601 vs
0.0983. Giving the exact `nn.Linear(D,K)` state path the linear-probe
reference's own initialization makes the cell *worse*, and its first-epoch
train loss (11.4) is 5.4× the shipped zero-init's (2.12). **Nothing in branch
1 is shipped; no probe-style-init config option is specified**, because the
measurement contradicts the branch's premise. (This is also the third
independent refutation of the initialization hypothesis on this dataset:
`reports/MODALITY_BENCHMARKS.md` §9.1 P5 = 0.1380 and §10.4's rejected
zero-init-gain candidate.)

**Branch 2 — "A2 ≈ A1 AND A5 ≫ A2 → budget is the cause": this is the branch
the evidence supports, and the one whose action is recommended below.** Its
first predicate holds only in regime, not in value: A2 = 0.3448 and A1 =
0.4196 are the same order of magnitude and both sit far above A0 = 0.2376 and
the 0.1340 majority floor, but A1 beats A2 by 7.5 pt paired (5/5 seeds,
t = 3.22), so "≈" is not literally true. Its second predicate holds
unambiguously: A5 = 0.5416 ± 0.0138 is **19.7 pt above A2** (paired +0.1968 ±
0.0261, positive on 5/5 seeds, t = 16.9), and it is the **only** cell in the
grid that clears both probes (R_p1 0.3472, R_p2 0.3632) and the majority floor
by a wide margin.

**Branch 3 — "A2 ≪ A1 AND A1 ≈ majority → the state path is the limit":
NOT FIRED.** A1 = 0.4196 is 3.13× the majority baseline (0.1340) and above both
probes, so the premise "A1 ≈ majority" is false. The state path is
demonstrably *not* the limit: A2 (an exact linear state path) reaches
0.3448 and A5 (the same map, 5× the budget) reaches 0.5416, and A6 — the same
map at **half** the width, 5.17 M instead of 13.58 M parameters — is 3.5 pt
*better* than A2. Capacity is not the binding constraint; halving it helps.

**Therefore: no branch fires literally.** Following the tree's own final rule
("any other pattern → report the pattern, the numbers, and the most defensible
next action; do not force a branch"), the pattern is reported and the most
defensible next action is named: the pattern is a **budget-bound** one. The
ordering of the four interventions measured here, all on identical data:

| intervention (each a one-cell change from A2) | Δ acc | cost |
|---|---|---|
| **5× epoch budget (A5)** | **+19.7 pt** (5/5) | 52.0 s/seed head-only vs 11.4 |
| 2.7× *fewer* parameters (A6) | +3.5 pt (3/5) | 5.7 s/seed |
| optimizer / lr / epochs / batch / loss switched to A1's (A3) | **+0.0 pt, bit-identical** | — |
| probe-style state-path init (A4) | **−5.9 pt** (4/5 negative) | — |

The optimizer/protocol axis is measured dead: A3 reproduces A2 on every seed
and every metric to the last digit (Δ = 0.0000 on accuracy, macro-F1, NLL, ECE,
Brier, and identical first/last train loss), because under this protocol A1
and A2 already run the same `HeadTrainer` AdamW, lr 1e-3, 4 epochs, batch 64,
same loss. Whatever separates the shipped stack from the probes is therefore
**not** in the optimizer, **not** in the initialization, and **not** in the
state-path capacity. It is the amount of optimization applied, plus the
option-interaction pathway (the A0→A1 +18.2 pt effect, already documented as
the §5.f `option_normalize` finding).

## 5. What should be shipped

**PRIMARY (branch 2, budget). Specified — NOT implemented (benchmark-only
track).** The head-training epoch budget of the frozen-feature protocol, not
the architecture, is what is left on the table.

- *Default change:* `epochs: 4 → 20` for frozen-feature head training.
- *Files / lines:* `benchmarks/run_head_quality.py:95,111,127,143,162`
  (the `DATASETS[...]` `"epochs": 4` entries, ravdess at :162) and
  `benchmarks/run_modality.py:115` (`--epochs` default `4`). Library-side the
  default that would need a value is `jevx/training/trainer.py:204`
  (`HeadTrainer.fit(..., epochs: int = 1)`); if a shipped default is wanted
  rather than a benchmark one, the new field would be
  `JevXConfig.head_epochs: int` (next to `decision_hidden_size`,
  `jevx/configuration_jevx.py:264`) defaulting to 20 for frozen-backbone head
  training.
- *Wall-clock cost (measured):* head-only 11.4 → 52.0 s/seed (4.6×; the
  5-epoch-ratio reflects the fixed per-epoch setup cost of
  re-encoding the question/schema rows). Over the whole §5 protocol the
  JevX arm's `wall_s` was 261.29 s of a 3004.63 s run whose featurization is
  2510.25 s; at 20 epochs the projected total is ≈ 3830 s, i.e. **+28 % on the
  full §5 benchmark and +830 % on the head-training stage alone.** The frozen
  backbone is featurized once, so the backbone cost is unchanged.
- *Honest caveat that must travel with this recommendation:* 20 was chosen by
  looking at test accuracy, which §3 of `reports/MODALITY_BENCHMARKS.md`
  forbids. The defensible shipping form is to select the budget on a held-out
  selection split (carve from train, or use the 240-row calibration split)
  and then report test once. Until that is run, A5's 0.5416 is evidence that
  the constraint is the budget, not evidence that 20 is the right shipped
  value.

**SECONDARY (already shipped, no change proposed).** The A0 → A1 effect
(+18.2 pt paired, 5/5) is the option-interaction pathway, and the shipped
switch for it already exists: `JevXConfig.conditioning_enabled` /
`conditioning_mode` (`jevx/configuration_jevx.py:284`) plus the adaptive
`ConditioningGate`. This grid re-confirms the §5.f finding from a different
angle; it proposes no new field. Note for whoever tunes it: the gate's own
verdict on this dataset is not stable — in the 5-seed A–F matrix,
`D_gate` (seed 0) kept conditioning ON while `F_gate_curriculum` (seed 0)
decided `state_only` (`benchmarks/results/head_ablation_ravdess.json`).

**EXPLICITLY NOT SHIPPED.** No probe-style state-path initialization option
(A4 is 5.9 pt worse, refuting branch 1). No capacity change (A6 shows
halving the decision width is mildly *better*, and a stronger backbone is not
indicated by any cell here).

## 6. §5.f re-run of the shipped audio benchmark (Task 3)

`benchmarks/run_modality.py --modality audio --seeds 0,1,2` re-run verbatim
with today's code, same defaults (epochs 4, lr 1e-3, head batch 64, split seed
0, 700/240/500), written to `benchmarks/results/audio_jevx_quality_v2.json`.
Wall 3004.63 s (featurization 2510.25 s, dataset load 2.29 s, backbone load
14.83 s, JevX stage 261.29 s), `failures: {}`.

**Protocol-identity check: the three baselines from this same run reproduce
the published §5.1 values to the digit** — B1 0.34333 ± 0.01361, B2 0.36067 ±
0.02663, B4 0.34333 / NLL 1.81534. The features, splits, seeds, optimizer and
budget are therefore demonstrably the same; only the JevX code moved.

| metric (uncalibrated, 3 seeds) | **published §5.1 (v1)** | **§5.f re-run (v2)** | Δ |
|---|---|---|---|
| accuracy | 0.1340 ± 0.0000 | **0.2087 ± 0.0361** | **+0.0747** |
| macro-F1 | 0.0318 ± 0.0038 | **0.1250 ± 0.0338** | +0.0932 |
| NLL ↓ | 2.6280 ± 0.9586 | **1.9690 ± 0.0373** | −0.6590 |
| Brier ↓ | 0.9655 ± 0.1604 | **0.8553 ± 0.0171** | −0.1102 |
| ECE(15) ↓ | 0.1520 ± 0.1992 | **0.0881 ± 0.0323** | −0.0639 |
| MCE(15) ↓ | 0.4023 ± 0.2489 | **0.3674 ± 0.0389** | −0.0349 |
| trainable params | 13,382,401 | **13,579,265** | +196,864 |
| train loss first → last | 280.34 / 219.70 / 180.67 → 2.18 / 2.56 / 4.13 | **7.14 / 6.56 / 6.83 → 2.016 / 2.017 / 2.006** | — |

Per-seed accuracy: **0.1740 (s0), 0.2060 (s1), 0.2460 (s2)** — versus
0.1340 / 0.1340 / 0.1340 in v1, i.e. the v1 degenerate single-class predictor
(no seed variance, macro-F1 0.032) is gone. Calibrated arms: temperature
0.2087 / NLL 1.9567 / ECE 0.0560; isotonic 0.2280 / NLL 2.2602 / ECE 0.0605.
Cached-vs-full-path equivalence `max_abs_diff = 0.0` on all 3 seeds;
probability invariants held on 108 eval batches, max |Σp − 1| = 2.384e-07.

**The code change between the old number and the new one** (recorded verbatim
in the `supersession` block of `audio_jevx_quality_v2.json`, together with the
old file's sha256):

1. `Fusion.option_normalize` now defaults to `True` (spec §5.f) — the option
   row enters `W_o` as a unit DIRECTION.
   `jevx/representations/fusion.py:152` (`Fusion.__init__(hidden_size,
   option_normalize: bool = True)`), propagated to every subclass at
   `fusion.py:353,366,379,394,416,448`.
2. The choice head's additive state-only pathway is now the default
   (`z_k = state_k + λ·r_k` instead of the v1 interaction-only form):
   `JevXConfig.state_pathway` (`jevx/configuration_jevx.py:280`) with
   `LinearProbePathway` / `MLPStatePathway` in
   `jevx/decisions/generic.py:238,349`. The +196,864 trainable parameters are
   exactly this pathway.

The supersession note itself is appended (APPEND-ONLY, nothing above it
touched) as **§11 of `reports/MODALITY_BENCHMARKS.md`**.

**A discrepancy this re-run exposes, which the grid spec got wrong.** The grid
specified A0 as "the shipped stack … `state_pathway=mlp`", and A0 was built and
run exactly that way (0.2376 ± 0.0274). But the **shipped config default is
`state_pathway="linear"`** (`jevx/configuration_jevx.py:280`; verified by
constructing `JevXConfig(base_model_type="bert", decision_hidden_size=768)` →
`state_pathway='linear'`, `conditioning_mode='full'`,
`conditioning_enabled=True`). The true shipped-default cell is the A–F
matrix's `A_linear_interact` = **0.2104 ± 0.0351** (5 seeds, same protocol,
`benchmarks/results/head_ablation_ravdess.json`), which agrees with the §5
re-run's 0.2087 ± 0.0361 (3 seeds). So the "shipped stack" on RAVDESS is
0.2104, and the mlp state pathway is **+2.7 pt better but is not the default**;
`state_pathway="mlp"` remains available and is not being proposed as a change
here, because §3 forbids selecting it on a test split.

## 7. What this does NOT establish

- **It does not establish that 20 epochs is the right shipped value.** A5
  shows the budget is the binding constraint; the value was read off the test
  split. A selection-split protocol is required before 20 is a default.
- **It does not establish anything about the vision or text modalities.** One
  dataset, one backbone, one split seed, frozen features, CPU, 4 cores. §10.3
  of `reports/MODALITY_BENCHMARKS.md` already shows the §5.f change is
  modality-dependent; the epoch budget plausibly is too and is untested.
- **It does not establish a causal claim about the option pathway.** A1's
  +18.2 pt is a paired cell on this protocol, not an isolation of a single
  mechanism: A1 also bypasses the question/schema pathway, so it changes two
  things at once. The A0→A2 arm (`state_pathway=mlp`→`linear`, both with
  conditioning ON/OFF as labelled) is the cleaner single-variable contrast and
  it is +10.7 pt.
- **It does not establish that a stronger backbone is needed.** Branch 3 did
  not fire, and A6 shows capacity reduction is mildly beneficial, so the
  "backbone/feature quality" explanation has no support from this grid — but
  neither is it refuted, because no cell varies the backbone.
- **It does not establish generalization beyond 5 seeds and this test split.**
  The paired t values use df = 4; the seed-to-seed spread (e.g. A6 ± 0.0452)
  is large relative to several of the smaller deltas, and A4's −5.9 pt and
  A6's +3.5 pt are the two effects that would move first with more seeds.
- **It does not establish bit-level reproducibility across hosts.** The grid
  reproduces the published A–F matrix per-seed values exactly **on this
  host**; that is a strong protocol check, not a cross-platform guarantee.
- **It does not claim the A2 reading of the grid is the only one.** The task
  spec's "P1 exact" reading (A2 = the bare probe) is covered by `R_p1` in the
  same run and reaches the same conclusion from the other direction: A2
  0.3448 vs `R_p1` 0.3472, paired Δ = +0.0024 ± 0.0292, t = 0.25 — the exact
  linear state path inside the full stack is **indistinguishable from the bare
  probe on the same features**. Under that reading the stack's overhead is
  zero and the entire remaining gap is the option-interaction pathway plus the
  epoch budget.
- **The wall times are host-load observations on a shared 4-core box**, not
  machine benchmarks; they should not be used for capacity planning.

## 8. Reproduction

```
python benchmarks/run_audio_diagnostic.py --seeds 0,1,2,3,4
    -> benchmarks/results/audio_diagnostic_ravdess.json   (854.89 s, failures {})
python benchmarks/run_modality.py --modality audio --seeds 0,1,2
    -> benchmarks/results/audio_jevx_quality_v2.json     (3004.63 s, failures {})
python -m pytest tests/unit -q
```

No cell failed. The A5 cell's wall time (260.1 s total, 52.0 s/seed) is the
budget-sensitivity cost quoted in §5.

**Validation, verbatim, at the end of this track** (nothing under `jevx/` was
touched, so this is a no-change regression check):

```
$ python -m pytest tests/unit -q
1451 passed, 1 skipped, 1 warning in 153.95s (0:02:33)
```
