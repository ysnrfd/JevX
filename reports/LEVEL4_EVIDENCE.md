# Level-4 Evidence (JevX, WAVE-4)

Scope: earn a support level 4 (native pooled vector + real-dataset quality) on
a checkpoint `TRANSFORMERS_COMPATIBILITY_MATRIX.md` actually cites, using executed
numbers only. Nothing under `jevx/` was modified. Every number below is cited to a
JSON in `benchmarks/results/`; every gate is the repository's own Phase-92 harness
(`tests/integration/matrix_common.py::execute_matrix_case`).

---

## 1. The level-4 criteria, quoted verbatim

**Level 4 (matrix, "Support levels (defined precisely)" table, row 4):**

> **4 — Native pooled vector** | A *trained/pooled* representation is a named output
> field of the standard base-model class (`pooler_output`, contrastive projection, or
> encoder `encoder_last_hidden_state` used directly). Loading via `Auto*`, input prep
> via one tokenizer/processor call, output contract **probe-verified** in this audit,
> checkpoint ID **HF-API verified**. Zero per-architecture glue in JevX beyond reading
> a field.

**The added quality clause (matrix, "Evidence overlay (Phases 92-94)"):**

> **4 additionally requires a real-dataset quality result for that family's cited
> checkpoint** (Accuracy/F1/NLL/Brier/ECE — owned by the benchmark suite).

**The execution clause that level 4 builds on (same section):**

> a family only holds **3 (Validated)** if load → forward → save/load equality →
> output invariants → **REAL-data input** were *executed on the actual checkpoint*

**The composite statement (matrix, "Cross-cutting verification summary"):**

> 4 = 3 + real-dataset quality for that family's checkpoint

**The matrix's own precedent for a family with no trained pooler (t5 row):**

> Pooling still JevX-side → not level 4 (no trained pooler).

Decomposed, a cell holds 4 only if **all** of these are true:

| # | criterion (from the quotes above) |
|---|---|
| C1 | a **trained**/pooled representation is a **named output field** of the standard base-model class |
| C2 | loading via `Auto*`; input prep via one tokenizer/processor call |
| C3 | output contract **probe-verified** in the audit |
| C4 | checkpoint ID **HF-API verified** |
| C5 | zero per-architecture glue in JevX beyond reading a field |
| C6 | load → forward → save/load equality → output invariants → **REAL-data input** executed on the **actual** checkpoint |
| C7 | a **real-dataset quality result** (Accuracy/F1/NLL/Brier/ECE) for that family's cited checkpoint |

---

## 2. ADOPTED vs own work

| artifact | provenance |
|---|---|
| `benchmarks/run_level4.py` | **ADOPTED** — written by another session before this one started. Not edited, not re-run. |
| `benchmarks/results/level4_roberta_sst2.json` | **ADOPTED** — that session's roberta-base quality result. Read, verified against the criteria, partly rejected (§5.3). Not overwritten. |
| `benchmarks/run_level4_continue.py` | **own** — new file (the other session may still be writing `run_level4.py`). |
| `benchmarks/run_level4_diag.py` | **own** — scratch diagnostic. |
| `benchmarks/results/level4_structural_gates.json` | **own** |
| `benchmarks/results/level4_structural_gates_default_pooling.json` | **own** |
| `benchmarks/results/level4_bert-base-uncased_sst2.json` | **own** |
| `vendor/reports/_jevx_matrix_results.json` | **not touched** — the harness `atexit` hook was unregistered in both gate runs so another run's artifact stays intact. |

---

## 3. Route A — extend the matrix to the three benchmark backbones

Route A premise: the quality evidence exists but sits on checkpoints the matrix does
not cite. What Route A actually produced is **three structural gates and three level
verdicts, none of which is 4**. The reason is C1, and it is a property of the
checkpoints, not of the runs.

### 3.1 bert — `google/bert_uncased_L-2_H-128_A-2`

Published quality (pre-existing, cited): `benchmarks/results/exp1_jevx_quality.json`
(SST-2) and `benchmarks/results/exp7_agnews_jevx.json` (ag_news), both 3 seeds,
uncalibrated mean:

| dataset | acc | macro-F1 | NLL | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| SST-2 | 0.7011 +/- 0.0104 | 0.6998 | 0.5759 | 0.3940 | 0.0320 | 0.0717 |
| ag_news | 0.8216 +/- 0.0128 | 0.8182 | 0.5353 | 0.2696 | 0.0623 | 0.1852 |

What I ran: the Phase-92 gate, on both pooling routes —

* `pooling_method="cls"` (the published protocol's strategy): **PASS**, all five error
  metrics exactly 0.0, `representation_field=pooler_output`, adapter `encoder`,
  `backbone_forward_calls_m4=1`. JSON `benchmarks/results/level4_structural_gates.json`.
* `pooling_method=None` (the adapter default, i.e. the named-field route C1 is about):
  **PASS**, all five error metrics exactly 0.0. JSON
  `benchmarks/results/level4_structural_gates_default_pooling.json`.
* two-load stability probe: 0 of 4,385,920 parameters differ between two independent
  loads; `pooler_output` diff 0.0 — the field is stored in the checkpoint and is
  bit-reproducible.

Verdict: **C1 FAIL, C2-C6 PASS, C7 STALE.** The checkpoint is a 2-layer, H=128
*random-weight fixture*; its `pooler_output` is a stable named field but nothing about
it is trained, and the two published quality numbers predate the shipped default
(§3.5). Stays **3**.

### 3.2 deit / vision — `facebook/deit-tiny-patch16-224`

Published quality (pre-existing, cited): `benchmarks/results/vision_jevx_quality.json`
(Imagenette, 3 seeds): uncalibrated accuracy 0.9087 +/- 0.0257, macro-F1 0.9107,
NLL 0.6326, Brier 0.1541, ECE 0.0736, MCE 0.5567; temperature NLL 0.2991, Brier
0.1307, ECE 0.0207, MCE 0.4803. Probes on identical features:
`benchmarks/results/vision_baselines.json` b1_linear 0.9467 +/- 0.0046, b2_mlp
0.9673 +/- 0.0042.

What I ran: the Phase-92 gate → **FAIL**:

> `AssertionError: facebook/deit-tiny-patch16-224: base_forward diverged from the raw
> checkpoint by 1.206e+00`

Root cause, measured by the two-load probe: `AutoModel.from_pretrained` resolves this
checkpoint to **`ViTModel`**, and `pooler.dense.weight` differs by **0.13297396898269653**
between two independent loads of the *same* checkpoint (the bias is bit-stable because
it initialises to zeros). The checkpoint ships no ViT pooler weights, so the pooled
field is randomly re-initialised on every load; `last_hidden_state` is unaffected.
JSON `benchmarks/results/level4_structural_gates.json`.

Verdict: **C1 FAIL twice over** — the field is untrained *and* `DeiT`/`ViT` route the
quality result through `last_hidden_state[:,0]`, i.e. JevX-side CLS pooling, which the
matrix's own t5 precedent says is not level 4. **C6 FAIL** (gate does not pass on the
actual checkpoint). **C7 STALE** (§3.5). Added to the matrix at the **1** floor with the
failure recorded in place, not at 3.

### 3.3 wav2vec2 / audio — `superb/wav2vec2-base-superb-ks`

Published quality (pre-existing, cited): `benchmarks/results/audio_jevx_quality_v2.json`
(RAVDESS, 8 classes, 3 seeds, **current** — see §3.5), uncalibrated:

| metric | mean | std |
|---|---|---|
| accuracy | 0.2087 | 0.0361 |
| macro-F1 | 0.1250 | 0.0338 |
| NLL | 1.9690 | 0.0370 |
| Brier | 0.8553 | 0.0171 |
| ECE | 0.0881 | 0.0323 |
| MCE | 0.3674 | 0.0389 |

Temperature (fit on the calibration split only): NLL 1.9567, Brier 0.8479, ECE 0.0560,
MCE 0.3756. Probes measured in the same run on the same features
(`audio_jevx_quality_v2.json:baselines_same_run`): b1_linear 0.3433 +/- 0.0136,
b2_mlp 0.3607 +/- 0.0266 — i.e. the shipped JevX stack is **13.5 pp below a linear
probe on its own features** on this task. Recorded as measured.

What I ran: the Phase-92 gate → **PASS**, all five error metrics exactly 0.0,
`representation_field=last_hidden_state`, `output_class=Wav2Vec2BaseModelOutput`,
adapter `audio`, `backbone_forward_calls_m4=1`, sha
`372e0486cd83e6f0c05c20a27262e9ca09450d24`, 94,371,712 backbone parameters. Two-load
probe: 0 parameters differ. JSON `benchmarks/results/level4_structural_gates.json`.

Verdict: **C1 FAIL by construction** — `wav2vec2` has no named pooled field; the
representation is `last_hidden_state` masked-mean, which the row itself already
documents as JevX-mirroring the family's own seq-cls head. That is the t5 precedent
verbatim. **C2-C7 PASS/CURRENT.** Level stays **3**; the checkpoint is added to the row
with the number cited, and the level is not moved.

### 3.4 Route A verdict

| backbone | C1 named trained field | C2 Auto*/1 prep | C3 probe | C4 HF-API | C5 no glue | C6 executed on actual ckpt | C7 quality for cited ckpt | level |
|---|---|---|---|---|---|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` | FAIL (untrained) | PASS | PASS | PASS | PASS | PASS | STALE | 3 |
| `facebook/deit-tiny-patch16-224` | FAIL (untrained + JevX-side CLS) | PASS | PASS | PASS | PASS | **FAIL** | STALE | 1 (floor) |
| `superb/wav2vec2-base-superb-ks` | FAIL (no named field) | PASS | PASS | PASS | PASS | PASS | PASS (current) | 3 |

**Route A produced no level-4 cell.** The blocker is C1, which is a property of the
checkpoint, so no amount of extra benchmark compute can move it.

### 3.5 Staleness of the pre-existing numbers (measured, not assumed)

The shipped default changed after those results were written. File mtimes:

* `jevx/representations/fusion.py` (the `option_normalize` flag) — **2026-09-26 09:11:16**
* `jevx/decisions/generic.py` (the `state_pathway` flag) — **2026-09-26 09:18:54**
* `jevx/modeling_jevx.py` — 2026-09-26 09:50:50
* `benchmarks/results/exp1_jevx_quality.json` — **2026-09-24 12:02:38**  (pre-dates both)
* `benchmarks/results/exp7_agnews_jevx.json` — 2026-09-25 09:33:44  (pre-dates both)
* `benchmarks/results/vision_jevx_quality.json` — 2026-09-25 10:10:29  (pre-dates both)
* `benchmarks/results/audio_jevx_quality.json` — 2026-09-25 11:31:08  (pre-dates both)
* `benchmarks/results/audio_jevx_quality_v2.json` — 2026-09-26 15:15:57  (post-dates both)

`CHANGELOG.md` states the effect on the option side: "`option_normalize` is now a
constructor flag on every `Fusion` subclass (default `True`, the normative §5.f
contract). **This changes the computed option features for any previously trained
checkpoint.**" The state-pathway default is documented as bit-identical to the
pre-flag head, so the option flag is the behaviour change.

`audio_jevx_quality_v2.json` says the same thing in its own `supersession` block and
names the two changes. So:

* SST-2 (tiny-bert), ag_news (tiny-bert) and Imagenette (deit) numbers are **stale**
  with respect to the shipped default and are labelled as such on the matrix rows.
  Re-running Imagenette was not affordable on this machine (thousands of 224x224 CPU
  forwards).
* RAVDESS (wav2vec2) `v2` is **current** and is cited as current.
* The ADOPTED roberta result (2026-09-26 10:10:50) was produced by a process that
  started at 09:11:00 and finished at 10:10:50, i.e. it straddles the
  `jevx/modeling_jevx.py` edit at 09:50:50 and entirely pre-dates
  `jevx/decisions/choice.py` at 15:02:32. It is treated as a measurement, not as a
  current-default claim.

---

## 4. Route B — a level-4 cell on a medium real checkpoint

### 4.1 Backbone choice, and why not roberta-base

The ADOPTED roberta run cannot be a level-4 cell, and §3.4 / §5.3 show why: the
checkpoint's pooled field is not trained, so the structural clause fails on the actual
checkpoint and any quality number taken on the shipped-default `pooler_output` route
measures a random projection. The task named roberta-base as the example medium
checkpoint; the evidence says the honest medium real checkpoint with a **trained named
pooled field** that the matrix already cites is **`google-bert/bert-base-uncased`**
(first entry of the matrix's bert row, HF-API verified). That is what Route B ran.

* backbone id `google-bert/bert-base-uncased`
* revision sha `86b5e0934494bd15c9632b12f734a8a67f723594` (recorded by the harness in both
  gate JSONs; the pre-fetch resolved the same sha)
* backbone parameters **109,482,240** (raw `AutoModel`); 123,061,505 with the JevX head
  stack attached
* architecture `BertModel`, adapter `encoder`, `jevx_conversion=supported`

### 4.2 Sizing before committing to the run

The task required sizing first. The executed basis is the ADOPTED roberta probe
(`level4_roberta_sst2.json:timing_probe`): **0.226435 s/example** at batch 64 /
max_len 128 on this machine, projecting **1103 s (~18.4 min)** for the full 4872
example featurization, measured later at **2028 s** for the real thing. The budget was
7200 s, so the protocol was run at full size and **no subset reduction was made**
(`budget_decision` is absent from the JSON, which is the script's own record that the
reduction branch never fired). `bert-base-uncased` is 0.88x the parameter count of
`roberta-base`, so the projection carried over.

### 4.3 Structural gate, on the same checkpoint

Phase-92, adapter **DEFAULT** route (`pooling_method=None`) — this is the route the
level-4 definition is about:

> `MATRIX RESULT | family=bert | id=google-bert/bert-base-uncased |
> sha=86b5e0934494bd15c9632b12f734a8a67f723594 | params=123061505 | load_s=4.49 |
> adapter=encoder | conversion=supported | rep_field=pooler_output | save_load=0.000e+00
> | base_fwd=0.000e+00 | fwd_calls_m4=1 | extract_err=0.000e+00 | state_repr_err=0.000e+00`

with `extracted_field=pooler_output` — i.e. the detector's preferred field and the
field actually extracted **agree**, which is the alignment guard the harness asserts
only on the default route. All five numerical gates are exactly 0.0, not merely small.
The decision invariants ran on a real sentence batch:
`choice_questions=2`, `batch_B2_decisions=4`, `score_expected_score=1.8937225341796875`,
`noul_p_yes=0.617209255695343`. JSON:
`benchmarks/results/level4_structural_gates_default_pooling.json` (and the `cls`-route
twin in `benchmarks/results/level4_structural_gates.json`, also all 0.0).

Two-load stability probe on this checkpoint: **0 of 109,482,240 parameters differ**
between two independent loads; `pooler_output` diff 0.0. The `BertPooler` weights are
*in the checkpoint*, so the field is trained and reproducible — the exact condition
roberta-base and deit-tiny fail.

### 4.4 The quality run — protocol

The established protocol, delegated function-by-function to
`benchmarks/run_level4.py` so it is identical by construction rather than by
re-implementation: `load_data` / `make_splits` / `stage_features` / `stage_jevx` /
`stage_baselines` from that module, verbatim. SST-2 (`stanfordnlp/sst2` @
`8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`), split seed 0 stratified train 3000 / cal
1000, test = the official validation split (872 rows, no cap applied in practice),
frozen backbone, featurize-once per split, 4 epochs, lr 1e-3, batch 64, max_len 128,
fusion `concat`, ECE/MCE 15 bins, calibrators (temperature, isotonic) fit on the
**calibration split only**, seeds 0/1/2. Two cells are measured: `cls`
(`last_hidden_state[:,0]`, the pooling the published protocol uses) and `native` (the
adapter's own pooled vector, `pooler_output` — the field the matrix row names and the
level-4-defining one). Probe baselines B1 linear / B2 MLP / B1+temperature run on the
**identical cached features, splits, hyper-parameters and seeds**.

Nothing is selected on the test split; the test rows are read once, for the final
metrics.

### 4.5 Results

Wall time: featurization **932.9 s** (train 611.86 / cal 172.56 / test 148.48), head
stacks 210 s (cls) + 196 s (pooler_output) + 12 s (baselines), **total 1445.4 s**
(24.1 min), 0 failures. 13,579,265 trainable head parameters, unchanged across seeds.
No subset reduction: `budget_decision` is absent from the JSON, which is the script's own
record that the reduction branch never fired.

Metric order in every table: **accuracy | macro-F1 | NLL | Brier | ECE | MCE**, mean +/- std
over seeds 0/1/2.

**(a) SHIPPED DEFAULT** — `pooling_method=None`, the adapter's own `pooler_output`. This is
the primary row, because it is what "supported" means:

| calibrator | accuracy | macro-F1 | NLL | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| none | 0.6552 +/- 0.0299 | 0.6180 +/- 0.0400 | 0.6703 +/- 0.0850 | 0.4489 +/- 0.0335 | 0.1251 +/- 0.0555 | 0.1913 +/- 0.0781 |
| temperature | 0.6552 +/- 0.0299 | 0.6180 +/- 0.0400 | 0.6027 +/- 0.0092 | 0.4179 +/- 0.0091 | 0.0689 +/- 0.0164 | 0.1338 +/- 0.0221 |
| isotonic | 0.6613 +/- 0.0288 | 0.6369 +/- 0.0248 | 0.6201 +/- 0.0401 | 0.4166 +/- 0.0042 | 0.0641 +/- 0.0127 | 0.2330 +/- 0.1448 |

Per seed, uncalibrated accuracy: 0.6583 (s0), 0.6835 (s1), 0.6239 (s2).

**(b) `cls`** — `last_hidden_state[:,0]`, the pooling the published protocol uses, so this
is the row that compares against `reports/DATASET_BENCHMARKS.md`:

| calibrator | accuracy | macro-F1 | NLL | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| none | 0.8417 +/- 0.0040 | 0.8415 +/- 0.0038 | 0.3881 +/- 0.0284 | 0.2315 +/- 0.0080 | 0.0555 +/- 0.0310 | 0.1480 +/- 0.0335 |
| temperature | 0.8417 +/- 0.0040 | 0.8415 +/- 0.0038 | 0.3552 +/- 0.0026 | 0.2228 +/- 0.0014 | 0.0218 +/- 0.0063 | 0.0946 +/- 0.0140 |
| isotonic | 0.8414 +/- 0.0043 | 0.8411 +/- 0.0041 | 0.4000 +/- 0.0336 | 0.2248 +/- 0.0034 | 0.0276 +/- 0.0054 | 0.4240 +/- 0.2799 |

Per seed, uncalibrated accuracy: 0.8394 (s0), 0.8394 (s1), 0.8463 (s2).

**Probe baselines on identical cached features, splits, hyper-parameters and seeds**
(`feature_set=cls`, `in_dim=768`):

| probe | accuracy | macro-F1 | NLL | Brier | ECE | MCE |
|---|---|---|---|---|---|---|
| B1 linear | 0.8322 +/- 0.0024 | 0.8320 +/- 0.0025 | 0.3740 +/- 0.0059 | 0.2350 +/- 0.0041 | 0.0455 +/- 0.0067 | 0.0802 +/- 0.0109 |
| B1 + temperature | 0.8322 +/- 0.0024 | 0.8320 +/- 0.0025 | 0.3638 +/- 0.0076 | 0.2313 +/- 0.0047 | 0.0175 +/- 0.0017 | 0.0743 +/- 0.0062 |
| B2 MLP | 0.8394 +/- 0.0159 | 0.8389 +/- 0.0168 | 0.3830 +/- 0.0270 | 0.2364 +/- 0.0163 | 0.0552 +/- 0.0119 | 0.1683 +/- 0.0296 |

**What the numbers say, stated plainly:**

* On the `cls` route the JevX stack reaches 0.8417 +/- 0.0040 accuracy against a linear
  probe at 0.8322 +/- 0.0024 and an MLP probe at 0.8394 +/- 0.0159 on the *same* features:
  +0.95 pp over linear and +0.23 pp over MLP, within roughly one std of both. The stack
  is at probe parity on this task, not above it.
* On the **shipped default** route the stack is at 0.6552 +/- 0.0299, **18.6 pp below** the
  same checkpoint's `cls` route. The `BertPooler` is genuinely trained (weights in the
  checkpoint; 0 of 109,482,240 parameters differ across two loads), so C1 is satisfied —
  but an NSP-trained pooler is a weak sentiment feature. This is the primary row and it is
  **not** adjusted, tuned away, or replaced by the better number.
* Temperature scaling is the useful calibrator on both routes (ECE 0.1251 → 0.0689 on the
  default route, 0.0555 → 0.0218 on `cls`); isotonic buys almost no extra ECE and costs
  MCE stability (0.1480 → 0.4240 +/- 0.2799 on `cls`).
* Seed spread is small (std 0.0040 on `cls`, 0.0299 on the default route), so the 18.6 pp
  route gap is not seed noise.
* The featurize-once path is verified against the full `model.choice()`: max abs feature
  difference 1.5e-05 to 5.0e-05 — float re-association — on every seed of both cells.


### 4.6 roberta-base: the ADOPTED number, and what it is worth

For the record, the ADOPTED `cls` cell on `FacebookAI/roberta-base` is a real executed
number and it is **not** a level-4 result, for three independent reasons: the gate
fails on that checkpoint, the `native` cell is invalid, and the run predates the current
shipped default. Its `cls` cell (uncalibrated accuracy 0.7229 +/- 0.0767 over seeds
0/1/2; per-seed 0.6399 / 0.7374 / 0.7913) is reported here as a measurement of a
frozen roberta-base + JevX head stack, and nothing more. The 7.7 pp seed spread is
itself the notable part.

---
## 5. PASS / FAIL against each criterion, per cell

| cell | C1 named trained field | C2 Auto* + one prep call | C3 probe-verified | C4 HF-API | C5 zero glue | C6 executed on the actual checkpoint | C7 quality for the cited checkpoint | **level** |
|---|---|---|---|---|---|---|---|---|
| **bert** @ `google-bert/bert-base-uncased` (Route B) | **PASS** — `pooler_output`, a named field of `BertModel`, weights present in the checkpoint (0/109,482,240 params differ across two loads) | **PASS** — `AutoModel` + one `AutoTokenizer` call, gate executed | **PASS** — bert is one of the 20/20 probe-verified families | **PASS** — cited in the bert row | **PASS** — `extracted_field == representation_field == pooler_output`; the whole route is reading one field | **PASS** — all five error metrics exactly 0.0 on a real sentence, both pooling routes | **PASS** — SST-2, 3 seeds, shipped default, on that exact checkpoint: acc 0.6552 +/- 0.0299, macro-F1 0.6180, NLL 0.6703, Brier 0.4489, ECE 0.1251, MCE 0.1913 (the `cls` route on the same checkpoint: acc 0.8417 +/- 0.0040) | **4** |
| bert @ `google/bert_uncased_L-2_H-128_A-2` (Route A) | FAIL — stable field, but a random-weight 2-layer fixture: nothing is trained | PASS | PASS | PASS | PASS | PASS — all 0.0, both routes | FAIL — published numbers predate the shipped default | **3** |
| deit @ `facebook/deit-tiny-patch16-224` (Route A) | FAIL — no pooler weights shipped; and the quality route is JevX-side CLS | PASS | PASS | PASS | PASS | **FAIL** — `base_forward` 1.206e+00 | FAIL — stale | **1** (floor) |
| wav2vec2 @ `superb/wav2vec2-base-superb-ks` (Route A) | FAIL — no named pooled field; `last_hidden_state` + JevX masked-mean | PASS | PASS | PASS | PASS | PASS — all 0.0, both routes | PASS — current, cited | **3** |
| roberta @ `FacebookAI/roberta-base` (ADOPTED, rejected for level 4) | **FAIL** — `pooler.dense.weight` differs 1.4e-01 between two loads | PASS | PASS | PASS | n/a | **FAIL** — `base_forward` 8.695e-01 | partial — a real number exists but on a run that predates the shipped default, and the `native` cell is invalid | **3** |

**One level-4 cell was earned: bert @ `google-bert/bert-base-uncased`, via Route B.**
Route A earned no level 4 — for the three benchmark backbones the blocker is C1, a
property of the checkpoint itself.

---

## 6. Matrix edits made (and only these)

1. **bert row — level 3 → 4.** The only level change in this report. Reason: the
   executed number for the cited checkpoint `google-bert/bert-base-uncased` now exists
   (this run) and the Phase-92 gate passes on that same checkpoint on the default
   `pooler_output` route with all five error metrics exactly 0.0. Cited to
   `benchmarks/results/level4_bert-base-uncased_sst2.json`,
   `benchmarks/results/level4_structural_gates_default_pooling.json`.
2. **roberta row — no level change.** Added the measured fact that the family's real
   checkpoint ships no pooler weights and therefore fails the Phase-92 gate, citing
   `benchmarks/results/level4_structural_gates.json` and
   `benchmarks/results/level4_roberta_sst2.json`. This *prevents* a future false level-4
   claim on this family; it grants nothing.
3. **wav2vec2 row — no level change.** Added the cited checkpoint
   `superb/wav2vec2-base-superb-ks` with its current RAVDESS numbers, its gate result,
   and the same-run probe baselines, citing
   `benchmarks/results/audio_jevx_quality_v2.json` and
   `benchmarks/results/level4_structural_gates.json`.
4. **New deit row in the Vision table — at the 1 floor.** Added so the Imagenette quality
   evidence has a cited home, with the gate FAIL and the stale-number warning recorded in
   the row. Not a validated-3 claim.
5. **"Evidence overlay" paragraph and the "Level distribution" bullet** updated to state
   the one new 4 and to name the deit row.

No other row was touched.

---

## 7. Files

**Created (own):**

* `reports/LEVEL4_EVIDENCE.md` (this file)
* `benchmarks/run_level4_continue.py` — new file; `run_level4.py` left alone because
  another session owns it
* `benchmarks/run_level4_diag.py` — scratch diagnostic for the roberta failure
* `benchmarks/results/level4_structural_gates.json`
* `benchmarks/results/level4_structural_gates_default_pooling.json`
* `benchmarks/results/level4_bert-base-uncased_sst2.json`

**Edited:**

* `TRANSFORMERS_COMPATIBILITY_MATRIX.md` — the five edits of §6 (the only
  pre-existing file this session changed)

**Read, not modified:** `benchmarks/run_level4.py`,
`benchmarks/results/level4_roberta_sst2.json`, `vendor/reports/_jevx_matrix_results.json`,
everything under `jevx/` (nothing under `jevx/` was changed).

---

## 8. Deviations, and what this report does not claim

* **Route A produced no level-4 cell.** The three benchmark backbones fail C1. The
  deit-tiny finding is a genuine defect in the *checkpoint*, not in JevX: the gate
  compares two independent loads of the same weights and the pooler is re-drawn in
  between. It is reported as a gate failure, not worked around.
* **No re-run of the stale numbers.** SST-2/ag_news on the tiny fixture and Imagenette on
  deit were produced before the §5.f `option_normalize` default flip. Imagenette was not
  re-run because thousands of 224x224 CPU forwards did not fit the budget. Those numbers
  are labelled stale on the matrix row and are not used for any level claim.
* **The shipped default is the primary row.** No recipe row (`fusion_type="identity"` +
  `conditioning_mode="off"`, 20 epochs) was run: the established-protocol numbers are the
  claim, and a recipe row would have doubled a run that was already at the budget limit.
  Nothing here measures that recipe.
* **Single dataset.** The bert level-4 cell rests on SST-2 only. The matrix asks for "a
  real-dataset quality result"; one dataset satisfies the letter of the clause, and this
  report does not claim more.
* **The 1-level deit row is a floor, not a characterisation.** deit's contract *is*
  probe-verified in the breadth sweep; what failed is execution on the real checkpoint.
  The row's "why" says exactly that so the level column is not over-read.
* **ADOPTED numbers are quoted, not re-derived.** The roberta, tiny-bert, deit and
  RAVDESS figures in §3 are quoted from their JSONs with the staleness verdict attached;
  the only quality numbers produced by this session are the bert ones in §4.

---

## 9. Validation

`python -m pytest tests/unit -q`, run verbatim at the end of this session, with
nothing under `jevx/` changed:

```
1451 passed, 1 skipped, 1 warning in 157.52s (0:02:37)
```

The pre-existing baseline in `_wave3b_pytest.log` is `1451 passed, 1 skipped, 1
warning in 170.85s (0:02:50)` — the same test set, unchanged.
