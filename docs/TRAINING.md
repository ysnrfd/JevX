# Training a JevX decision head

**Status:** JevX-Next wave, 2026-09-26. Covers the plain optimizer loop
(`HeadTrainer`), the staged two-phase curriculum (`run_curriculum`) and the
task-level conditioning gate (`ConditioningGate`). API surface:
`docs/API.md` §14. Mathematics: `docs/MATHEMATICAL_SPECIFICATION.md` (5.g)–(5.h),
§13, §14. Measured evidence: `reports/HEAD_ABLATION_MATRIX.md`,
`reports/ABLATIONS.md` §9–§10.

Every number on this page is copied from an executed run; nothing is estimated.
Claims are scoped to the four datasets and five seeds actually executed
(synthetic control, ag_news, sst2, ravdess), frozen backbones, CPU, 4 epochs.

---

## 1. Three ways to train, and when to use each

| | `HeadTrainer` | `run_curriculum` (staged) | `run_curriculum` + `ConditioningGate` |
|---|---|---|---|
| what it optimizes | the whole stack from step 0 | Stage A state-only, then Stage B with a λ ramp | as staged, then a per-task mode switch |
| option pathway during training | on from step 0 (`condition_options=True`) | **out** of the graph in Stage A, **in** from Stage B step 0 with λ ramping 0→1 | same as staged |
| inference-time option pathway | on | on | **decided by the gate** from a calibration split |
| extra cost | none | none (same optimizer, same epochs) | one calibration pass × 2 modes |
| use it when | the head is not near saturation and you have budget to spare | the state signal is being drowned by the option pathway inside the training budget | you have a calibration split and want the option pathway used only where it measurably helps |

`HeadTrainer` is the plain loop: AdamW over `requires_grad` parameters, grad
clipping, a finite-loss guard with epoch/step context, and `evaluate`. Nothing
about it changed in this wave. `run_curriculum` **reuses** it — the same
`HeadTrainer` drives the optimizer loop, one epoch at a time, so the per-step λ
ramp can be interleaved; it reuses `make_supervised_step_fn`, `accuracy` and
`macro_f1`, and it re-seeds the global RNG and every loader generator. It forks
no part of the training stack.

The choice is **not** "run the curriculum and stop there". §5 below records the one
dataset where it measurably is, and the three where it is not.

---

## 2. `HeadTrainer` — the plain loop

```python
from jevx.training import HeadTrainer, DecisionDataset, decision_collate_fn

loader = DataLoader(DecisionDataset(records), batch_size=32,
                    collate_fn=decision_collate_fn)
trainer = HeadTrainer(model, lr=1e-3, grad_clip=1.0, seed=42)
trainer.fit(loader, epochs=4, eval_loader=cal_loader)
```

Defaults: `AdamW`, `torch.manual_seed(seed)`, `grad_clip=1.0`. Use this when you
have a converged-enough head, or when you specifically want to measure what joint
optimization from step 0 costs (it is the baseline cell of every ablation below).

## 3. `run_curriculum` — the staged curriculum

```python
from jevx.training import CurriculumConfig, run_curriculum

curriculum = CurriculumConfig(stage_a_epochs=2, stage_b_epochs=2,
                             lambda_start=0.0, lambda_end=1.0,
                             lambda_schedule="linear",
                             lr=1e-3, grad_clip=1.0, seed=0,
                             decision_type="choice")
history = run_curriculum(model, train_loader, {"cal": cal_loader},
                         curriculum=curriculum)
history.save("curriculum.pt")
```

**Stage A** (`stage_a_epochs`) calls the model with `condition_options=False` on
**every** training step. Two things follow, and both are asserted by the tests:
the option encoder, the schema projection and the interaction parameters are
outside the autograd graph and keep `grad is None`; and λ is pinned to `0.0` for
the whole stage, so the stage ends with the state pathway fitted and the residual
still exactly zero. `stage_a_epochs=0` skips it.

**Stage B** (`stage_b_epochs`) re-enables the option pathway and ramps λ **per
optimizer step**:

```
step_index = epoch * steps_per_epoch + step        # both 0-based
total_steps = stage_b_epochs * steps_per_epoch
progress    = 0.0            if total_steps == 1
              step_index / (total_steps - 1)  otherwise
shape       = progress                             # "linear"
              0.5 - 0.5 * cos(pi * progress)        # "cosine"
lambda      = lambda_start + (lambda_end - lambda_start) * shape
```

The ramp depends on no seed, no data and no optimizer state, so the recorded
sequence is reproducible by hand. With `steps_per_epoch=47` and
`stage_b_epochs=2` (the ravdess arm), Stage B runs λ 0.00 → 0.48 over epoch 1 and
0.52 → 1.00 over epoch 2.

**Read the history, do not trust the config.** `EpochRecord.condition_options`
records the flag the epoch actually ran under, read back off the model view, and
`EpochRecord.interaction_scale_first`/`interaction_scale` record the scale at the
epoch's first and last training step. From the fixed ravdess re-run, seed 0:

| stage | epoch | λ first → last | `condition_options` |
|---|---|---|---|
| a | 1 | 0.00 → 0.00 | `False` |
| a | 2 | 0.00 → 0.00 | `False` |
| b | 1 | 0.00 → 0.48 | `True` |
| b | 2 | 0.52 → 1.00 | `True` |

A Stage-B record reading `condition_options=False` means the run is not doing what
it claims; that is the exact defect the E/F re-run was built to catch (§6).

**No-test-leakage.** `run_curriculum` refuses any evaluation split whose name —
the mapping key, or a `split`/`split_name`/`name` attribute on the loader or its
dataset — contains `test`. There is no override. Curriculum evaluation is for
model selection on calibration/validation; the test split is measured once, by the
evaluation harness, after the curriculum is fixed.

**Batch contract.** Loaders are built with `DecisionDataset` +
`decision_collate_fn`, and curriculum training additionally requires (a) exactly
ONE question and ONE label per record, and (b) identical question texts across a
batch. Both are checked with the offending batch reported, not silently coerced.

---

## 4. The gate protocol

Run it **after** the head is trained, on a **calibration** split.

```python
from jevx.training import (ConditioningGate, ConditioningGateConfig,
                           apply_gate)

gate = ConditioningGate(model,
                        ConditioningGateConfig(split_name="calibration", z=1.0))
verdict = gate.decide(cal_loader)        # "state_only" | "conditioned"
print(gate.summary())

# pin it on the model (inference default), or pass it per call:
apply_gate(model, gate)                   # -> writes config.conditioning_enabled
flag = gate.condition_options_for(model)  # -> the same boolean, explicitly
```

**What it measures.** One number, on one split:

```
ΔNLL = NLL_conditioned − NLL_state_only        (per example, example-weighted)
```

The loader is drained **once** and the batch list replayed for
`condition_options=True` and then `False`, in `eval()` under `no_grad`, so both
passes see the same weights, the same batches, the same order and the same
conditions. A repeated `decide` on the same loader and weights is bit-identical.

**The threshold.** τ comes from the calibration split alone, by the
one-standard-error rule: with `d_b` the per-batch ΔNLL over `B` calibration
batches, `SE = std(d_b, ddof=1)/sqrt(B)` and `τ = z·SE` (`z = 1.0` by default).
No sampling, no grid search, no search over any evaluation number — so the gate
is independent of the reported evaluation by construction, not by discipline.

**The rule.** `state_only` **iff** `ΔNLL > τ + tie_tolerance`; otherwise
`conditioned`. A tie keeps the shipped behaviour, and below
`min_samples` (default 32) the gate refuses to switch and records why in
`tau_source`. `tau_source` is the audit line: it is either the protocol string
`one_se(z=1, se=0.00240, batches=16)`, `explicit` (a τ the caller pinned via
`set_tau`, for a threshold justified outside any evaluation split), or
`min_samples_unmet(...)`.

**Task level, never per sample.** One verdict for the whole task. A per-sample
rule fitted on one split is exactly the overfitting failure this construction
exists to avoid.

**No-test-leakage, enforced.** Any split name carrying a **whole-token** test-like
token is refused at config construction, in `measure` and in `decide`.
`RAVDESS_test` and `test` are rejected; `contest` is not. The token set is
overridable per gate.

**Order matters.** A curriculum that trains the option pathway must run with
`conditioning_enabled=True`, because a head trained state-only never learns an
option interaction for the gate to keep. If the curriculum itself toggled
`conditioning_enabled` between stages, call `decide` **after** the final stage, so
the flag training leaves behind and the flag the gate reports cannot disagree.

**What applying it does and does not do.** `apply_gate` writes
`model.config.conditioning_enabled`; an explicit per-call `condition_options=`
still wins, which is how you A/B both modes on one model. Flipping it changes
**only** the option pathway at inference — the state pathway, the projections and
every learned weight are untouched, so no retraining is implied. The flipped mode
was never trained as the active path, which is exactly why the calibration
measurement, not a test-split score, is what decides it.

---

## 5. The honest measured effect

Source: `benchmarks/results/head_ablation_curriculum_fixed.json` and
`benchmarks/results/curriculum_fixed_rerun/`, written by the unmodified
`benchmarks/run_head_ablation.py`. 5 seeds (0–4), frozen features, identical
splits / optimizer / lr / 4 epochs / batch / loss for every cell. Cells:
**C** = MLP state path + interaction, no curriculum; **E** = C + curriculum;
**F** = C + gate + curriculum. Paired deltas, same seeds, same order.

### 5.1 The curriculum against C (no curriculum) — test accuracy

| dataset | E − C (pt) | seeds | F − C (pt) | seeds | 5-seed ranges overlap? |
|---|---|---|---|---|---|
| synthetic | +0.47 | 3/5 | +0.52 | 3/5 | yes |
| ag_news | +0.03 | 3/5 | +0.05 | 3/5 | yes |
| sst2 | +0.14 | 3/5 | +0.11 | 3/5 | yes |
| **ravdess** | **+9.04** | **5/5** | **+8.92** | **5/5** | **NO** |

On ravdess the NLL deltas are `-0.1205` (E) and `-0.1402` (F), 5/5 seeds, and
the 5-seed accuracy ranges do not overlap: E/F `[0.2840, 0.3660]` against C
`[0.2020, 0.2600]`. That is a real effect, on the one dataset where there was a
deficit to recover.

### 5.2 The text/synthetic datasets: within one seed std

Every paired delta on synthetic, ag_news and sst2 is inside one seed std and
flips sign across seeds. On sst2 `|Δ| ≤ 0.52` pt against a per-seed std of
1.2–1.4 pt. **No effect is declared from those three datasets**, and none was
declared before the re-run either.

### 5.3 …but it does not pass the interaction-free arm

| dataset | C − B (pt) | E − B (pt) | seeds | F − B (pt) | seeds | ranges overlap? |
|---|---|---|---|---|---|---|
| synthetic | −0.52 | −0.05 | 3/5 | −0.00 | 4/5 | yes |
| ag_news | −0.15 | −0.12 | 1/5 | −0.09 | 1/5 | yes |
| sst2 | −0.14 | −0.00 | 2/5 | −0.02 | 2/5 | yes |
| **ravdess** | **−6.96 (0/5)** | +2.08 | 3/5 | +1.96 | 2/5 | **yes** |

B is the same MLP state path with the interaction **OFF**. The curriculum
recovers most, not all, of what the interaction costs on ravdess: C sits 6.96 pt
*below* B, E sits 2.08 pt *above* it, with overlapping ranges and only 3/5 and
2/5 seeds. It does not win there seed-consistently, and on the other three
datasets it is a dead heat. It also does not hurt anywhere measured: the worst
paired accuracy delta is `-0.12` pt (ag_news E − B, 1/5 seeds).

### 5.4 The gate's tally

Across 4 datasets × 5 seeds = 20 measurements per arm, calibration split only:

| arm | flips to `state_only` | note |
|---|---|---|
| gate-only (cell D) | **2/20** | only sst2; the other three datasets keep `conditioned` ×5 |
| gate + curriculum (cell F, fixed run) | **9/20** | synthetic 4/5, ag_news 3/5, sst2 1/5, ravdess 1/5 |

The gate fires **rarely on a model whose residual was actually trained** — there
ΔNLL on the calibration split is strongly negative, i.e. conditioning genuinely
helps, so `ΔNLL > τ` is false. That is the rule working, not failing. On ravdess
the F arm's 1/5 flip is the one that mattered: the gate kept `conditioned` on 4 of
5 seeds, where flipping had cost nothing measurable.

### 5.5 The cost side, stated plainly

`option_pathway` (interaction + schema + fusion — the sum only option
conditioning can reach) is **656 384 of 1 296 897 trainable parameters = 50.6 %** at
D=128 (synthetic, ag_news, sst2) and **7 870 464 of 14 169 857 = 55.5 %** at
D=768 (ravdess). The question and schema encoders dominate it (574 080 each at
D=128), and on these benchmarks both encode a quantity that is **constant for a
whole dataset** — one fixed question, one fixed schema.

Latency, batch 1, decision stack only (frozen backbone excluded), dataset-matched
against the MLP probe P2, cells A–F:

| dataset | P2 (ms) | JevX stack A–F (ms) | ratio vs P2 |
|---|---|---|---|
| synthetic | 0.146 | 4.12 – 5.14 | 28.2× – 35.1× |
| ag_news | 0.222 | 4.06 – 4.97 | 18.3× – 22.4× |
| sst2 | 0.110 | 3.77 – 6.54 | 34.1× – 59.3× |
| ravdess (D=768) | 0.312 | 7.55 – 14.82 | 24.2× – 47.5× |

Against the *linear* probe P1 the same cells cost **77× – 465×**. Caveat stated
rather than smoothed: per-iteration std is comparable to the mean on this 4-core
box under concurrent load, so read these as an order of magnitude, not as a
ranking between cells.

### 5.6 The two negative results that did not improve

Both were measured with **no curriculum** in any cell, so neither depends on the
Stage-B fix and neither is re-measured by the re-run.

* **Q1 — is B (MLP state path, interaction OFF) ≈ P2 (MLP probe)? No.**
  synthetic `−1.11` pt (0/5 seeds), ag_news `−0.33` pt (1/5), sst2 `+0.48` pt
  (2/5), ravdess `−5.60` pt (0/5). The single dataset where B is nominally ahead
  is the one where P2 does not separate from P1 either (sst2: P1 0.7000 vs P2
  0.6966), so it is a null, not a win. **The state path alone did not close the
  probe gap.**
* **Q2 — is C (MLP + interaction) > B (MLP, interaction OFF)? No. C ≤ B on all four
  datasets**: synthetic `−0.52` pt (1/5), ag_news `−0.15` pt (2/5), sst2
  `−0.14` pt (1/5), ravdess `−6.96` pt (0/5, and `+0.1083` NLL, 0/5). Option
  conditioning carried no measurable information on top of the state path
  anywhere in this matrix.

---

## 6. The defect this wave found, and the fix

`run_curriculum`'s Stage B did not re-enable the option pathway. The per-call flag
was assigned once in the view's constructor and never updated when the stage
flipped, so every Stage-B step still called the model with
`condition_options=False` and the residual was dropped whatever λ held — `λ`
alone cannot open the pathway. The E/F cells of the original matrix were
therefore "state-only for all four epochs plus an inert ramp".

**Fix.** `_StageController` now owns the per-step flag as well as the per-step
scale. `_StageController.condition_options` is a pure function of the stage
(`False` in A, `True` in B); `apply_stage()` pushes it into the model view once per
epoch — so the epoch's evaluation inherits its training flag — and again before
every training step. `EpochRecord.condition_options` records what the epoch
actually ran under, read back off the view.

Four regression tests in `tests/unit/test_curriculum.py` each fail against the
defect restored in place. Stage A is untouched: its first-epoch train loss and its
epoch-1 gradient bucket are identical before and after the fix, and the
option-pathway columns are still absent there, because Stage A is state-only in
both.

The public API did not change: `run_curriculum`, `CurriculumConfig` and
`CurriculumHistory` keep their signatures, the one new `EpochRecord` field is
additive with a default, and the history format key is unchanged.

---

## 7. Reproduce

```
python -m pytest tests/unit -q
python benchmarks/run_head_ablation.py --datasets synthetic ag_news sst2 ravdess \
    --seeds 0,1,2,3,4 --cells E_curriculum F_gate_curriculum \
    --output-dir benchmarks/results/curriculum_fixed_rerun
```

Artifacts: `benchmarks/results/head_ablation_curriculum_fixed.json` (old vs new
E/F, per-seed rows, per-epoch records with the flag, gradient buckets, gate
verdicts, paired comparisons against the untouched B/C/A/D/P rows),
`benchmarks/results/curriculum_fixed_rerun/` (the four re-run payloads).

Nothing in this document is a claim about datasets, backbones or tasks outside
the four executed here.
