# JevX Limitations and Honest Scope

**Purpose.** This document exists to prevent overclaiming. It is part of the Phase 2
specification deliverable and is written to be quotable against the rest of the repository:
if another document appears to promise more than this one allows, this one wins.
Date of statement: 2026-09-24. Labels: [FACT] observed here; [DESIGN] scoped choice;
[STATISTICAL] data-dependent; [MATH] boundary of a theorem.

---

## 1. Test-environment constraints (CPU-only) [FACT]

Directly observed (``REPOSITORY_RECONNAISSANCE.md``) and not adjustable in this phase:

| Constraint | Value | Consequence |
|---|---|---|
| Accelerators | none: ``torch 2.14.0+cpu``, ``cuda.is_available() = False``, no MPS, no ``nvidia-smi`` | No CUDA/MPS code path can be exercised here. Untested must be reported as untested — never as passing. |
| CPU | Intel i5-6300U (2015 mobile), 4 logical threads | Throughput/latency numbers are machine-specific; only tiny-to-~1B-class checkpoints are practical for repeated forwards; parallel-batch sizes are RAM-bound. |
| RAM | ~8 GB total | Batch × `K_max` × question-count experiments must stay small; large padded tensors can OOM. |
| OS / Python | Windows 11; Python 3.14.3 (very new) | Optional wheels may lag; only the six core packages were verified importable. |
| `transformers` | vendored shallow clone, detached at tag v5.16.1 (commit `93c8b7b4…`) | All integration claims are v5.16.1-specific; history archaeology needs a deepened clone. |
| HF Hub | reachable, unauthenticated (rate-limited) | Large downloads may fail or throttle; cache contents vary per machine. |

**Reporting rule [DESIGN]:** every number produced in this repo carries machine, versions,
dtype, batch shape, and thread count. Numbers without that tuple are not results.

---

## 2. Universality is scoped — the exact scope statement [DESIGN]

JevX's mechanism-level universality claim is scoped, verbatim, to:

> **"HF Transformers-compatible models with a meaningful differentiable representation."**

Unpacked, that means a model must satisfy *all* of:

1. it loads through the `transformers` auto classes resolved from the audited mapping
   (`vendor/reports/auto_model_mapping.json`, 510 `model_type` entries [FACT — the mapping
   exists, not that the models work]);
2. a forward pass returns finite hidden states plus a usable mask for the modality's
   canonical sample (this is the *representation* condition);
3. those states are differentiable w.r.t. inputs and parameters, so Mode A/B training is
   even possible (the *differentiable* condition);
4. the modality has an InputAdapter (text: implemented by design; vision/audio/multimodal:
   specified, implementation status per component, support level below 4 until tested).

**Explicitly outside scope:** non-Hugging-Face formats (ONNX-only, custom checkpoints without
a transformers class); models that expose only generated sequences with no reachable hidden
states; retrieval/rule/tool systems with no differentiable state; remote-code models with
nonstandard outputs; quantized/GGUF kernel paths (untested here); anything the probe fails on.
**"510 model families in the mapping" is a fact about the mapping, not a support claim** —
support is per-family, per-level (`4`), evidence-based.

---

## 3. Functional independence ≠ statistical independence (parallel questions)

This is the most frequently abused distinction in this design; the specification states both
halves (P11.1 and P11.2 in ``MATHEMATICAL_SPECIFICATION.md``):

* **What is guaranteed [MATH]:** *functional* separation — output $j$ is computed from
  $(S, Q_j, C_j)$ only; no graph edge connects question $k$ to output $j$ (D5.1, A7.3). This
  is what makes batching equivalent to looping within tolerance (T7).
* **What is *not* claimed:** *statistical* independence of $(D_1, \dots, D_J)$. The answers
  share $H_S$ (a deterministic function of the same random state), share trained parameters,
  and inherit any correlation between the questions themselves. Consequently:
  * probabilities across questions must never be multiplied or jointly normalized
    ((11.3) is an invariant, not a style preference);
  * there is **no joint-distribution semantics** for the tuple of parallel answers — no
    family-wise error control, no independence-based threshold transfer between questions;
  * reporting per-question metrics does not license aggregate claims about "all questions
    together" without a dependence analysis that this project has not done.

Functional independence buys engineering properties (equivalence, testability,
parallelism). It buys no probabilistic properties.

---

## 4. Calibration ≠ correctness [STATISTICAL]

* Calibration is an **aggregate frequency property on a stated distribution**: among items
  assigned probability $\approx 0.9$, about 90% are correct — *measured on exchangeable
  data*. It says nothing about any individual decision, and nothing about data that is not
  exchangeable with the calibration set.
* **Calibration does not change decisions under temperature scaling** [MATH, `8` of the
  spec]: argmax is preserved, hence accuracy is *identical* before and after. Reading
  "better calibrated" as "more accurate" is a category error. (Non-order-preserving
  calibrators — vector/isotonic — *can* change decisions; JevX reports such changes rather
  than hiding them.)
* All calibration numbers (ECE, NLL, Brier, reliability diagrams) are **estimators** with
  sampling error and, for ECE, binning dependence (known caveat [R26–R28 of RESEARCH.md]).
  Small calibration sets make isotonic fits coarse (few PAVA blocks — piecewise-linear between breakpoints, spec 8.3) and vector scaling
  overparameterized; both can overfit the calibration split.
* Under distribution shift, a fitted calibrator is a fitted function on the wrong
  distribution; nothing in the pipeline detects or repairs shift.
* The forbidden sentence stands: **"a calibrated model is always correct" is false** and is
  never claimed (see the non-claims list in ``PROOFS.md``).

---

## 5. Confidence ≠ correctness (related, but separate) [STATISTICAL]

Confidence functionals (Def. 9) are deterministic functions of logits. Unvalidated, they can
be arbitrarily uninformative — a model emitting constant confidence has zero resolution, and
a confidently wrong model ranks its errors as its most certain outputs. Energy scores are
logit-scale-dependent and are not probabilities at all until linked through a fitted map.
Thresholds $\kappa$ tuned on one split do not transport for free; risk–coverage and
coverage-at-risk numbers (Def. 10) are sample estimates with sampling error, valid only under
the data assumptions stated with them (loss, $N$, $\kappa$-selection procedure).
**"Confidence guarantees correctness" is false and never claimed.**

---

## 6. Numerical limitations [MATH boundary]

1. **Floating-point non-associativity** defeats bit-exact batch/loop equality; T7 guarantees
   equality in real arithmetic and *measured* tolerance in IEEE-754 (Remark 7.2). Tolerance
   assertions are empirics, not theorems, and must be re-measured per configuration.
2. **Strict inequalities soften at extremes:** sigmoid outputs round to exact 0/1 beyond
   $|z| \gtrsim 104$ (binary32) without clamping (Remark 4.1); softmax terms underflow to
   exact 0 (Remark 1.2). Clamps restore stored ranges at the cost of an $O(\varepsilon)$ bias.
3. **Softmax error grows with $K$** ($\gamma_{cK}$-class bounds): very large option sets
   accumulate more rounding; padded `K_max` multiplies softmax work per question even though
   padding itself is exact (Lemma 7.1).
4. **All-masked option sets** would make softmax/argmax undefined; guarded by the $K \ge 1$
   axiom (D3.1) enforced at validation — the guard is code, hence an implementation invariant
   with a corresponding failure mode if bypassed.
5. **dtype is binary32 by default**; no mixed-precision or fp64 path is validated here.

---

## 7. Support levels 0–4 (definition and honesty rules)

**The normative definition of levels 0–4 is the "Support levels (defined precisely)" table
in ``TRANSFORMERS_COMPATIBILITY_MATRIX.md``** (repo root) — that table wins any conflict;
``ARCHITECTURE.md`` §6 and this section only re-state it. Condensed, quoted once: **0** no
standard extraction path through the public API; **1** source-verified only (not
forward-probed here, input pipeline not exercisable, or checkpoint not HF-API verified);
**2** convention-dependent (lossy pooling) — mechanics standard but pooled semantics need a
pinned convention; **3** native states + established standard pooling, executed
(load → forward → save/load equality → output invariants → real-data input) on the cited
checkpoint, HF-API verified; **4** native pooled vector + HF-API-verified route **plus a
real-dataset quality result** for that family's checkpoint. Rules:

* levels are claimed only with the listed evidence — **never pre-assigned, never
  aspirational**;
* level 4 is machine- and version-scoped (this box, these versions); a green level 4 here is
  not a claim about other hardware;
* demotion on failure is mandatory; monotonicity is by definition (4 implies 3 implies 2
  implies 1);
* **levels are evidence-bound, not aspirational** [FACT — updated 2026-09-24]: as of this
  writing the executed matrix assigns levels 3 and 2 to the audited families (none at 1/0),
  from 18 executed checkpoints — see ``TRANSFORMERS_COMPATIBILITY_MATRIX.md`` and
  ``reports/REAL_MODEL_VALIDATION.md``.

---

## 8. Evidence status of the repository right now [FACT]

Stated plainly so no reader mistakes specification for results:

* **Implementation, tests, benchmarks and conversions now exist** [FACT — updated
  2026-09-24]: the ``jevx/`` package is complete; ``python -m pytest`` measured
  **768 passed, 1 skipped** on this host in the pre-fix-wave run quoted in docs/TESTING.md §1
  (the fix-wave tree collects 828; its verbatim summary is refreshed in
  reports/FINAL_VERIFICATION.md); quality/ablation/latency/memory
  benchmarks ran and their numbers live in ``reports/`` with JSON in ``benchmarks/results/``;
  18 checkpoints converted and validated (reports/REAL_MODEL_VALIDATION.md); a calibrator was
  fitted on the benchmark calibration split and risk–coverage curves measured
  (reports/DATASET_BENCHMARKS.md, reports/ABLATIONS.md EXP-6).
* Therefore [FACT]: [EMPIRICAL] labels attach only to numbers actually measured on this
  machine, as cited; support levels are assigned per-family with evidence (levels 3 and 2
  among audited families — ``TRANSFORMERS_COMPATIBILITY_MATRIX.md``); the status of
  hypotheses H1–H12 is whatever ``RESEARCH.md`` and the ``reports/`` record — nothing is
  assumed open or closed here.
* The proofs in ``PROOFS.md`` are paper proofs under stated assumptions; they transfer to
  code only to the extent the implementation satisfies those assumptions — which is exactly
  what the runtime guards (the ValidationEngine component, `jevx/validation.py`, called from
  ``JevXForDecision.forward`` behind ``JevXConfig.runtime_validation`` — fix-wave contract;
  while it is absent, the by-construction invariants and the test suites are the sole
  enforcement) and the written test suite are for. Spec–code drift is
  a real risk; mitigation is that invariants are written against
  *numbered definitions and theorem ids* (D8.I, T7, (11.3)) so tests can cite them directly.

---

## 9. Task-scope limitations [DESIGN]

* **Closed-world schemas (AS6).** For a Choice question, correctness requires the answer to
  be *in* $C$. Type safety (T2) guarantees the output is *an* element of $C$ — it cannot
  make the right element appear if the schema omitted it. Schema completeness is a caller
  obligation.
* **Single-step decisions only.** No planning, lookahead, tool use, or multi-turn state
  tracking is specified; sequential deployments that feed outputs back as inputs inherit
  none of the batch/loop equivalence results (T7 applies to one state → J questions, not to
  feedback loops).
* **Declared type = declared semantics (AS8).** If a task's true structure is ordinal but it
  is declared as flat Choice, no amount of calibration repairs the decision policy mismatch.
* **No reasoning budget.** Unlike autoregressive generation, there is no test-time compute
  knob that trades latency for deliberation (see Q9.4 of RESEARCH.md — a scoped, honest
  tradeoff, not a defect to be papered over).

---

## 10. Statistical-data limitations [STATISTICAL]

* All generalization statements presuppose exchangeable train/calibration/evaluation data
  (AS3) and disjoint splits (AS4); leakage voids every reported statistic.
* Choosing $\kappa$, fusion variant, or calibrator family on the same sample used for
  reporting biases results optimistically; the reporting protocol must state where selection
  happened.
* Many parallel questions evaluated on few samples shrink per-question confidence intervals;
  nothing here performs multiple-comparison correction automatically.
* No claim in this repository is distribution-free over inputs, tasks, or time.

---

## 11. What is deliberately absent

* **No security/privacy audit** of models processing raw inputs — out of scope, unclaimed.
* **No "Jev" internals:** the proprietary-system question is settled by plain ignorance —
  see `1` of ``RESEARCH.md``; nothing here reverse-engineers or claims to reproduce any
  proprietary system.
* **No superiority claims of any kind:** comparative statements are permitted only as scoped by
  the benchmarks that exist (reports/DATASET_BENCHMARKS.md, reports/ABLATIONS.md,
  reports/LATENCY_BENCHMARKS.md, reports/MEMORY_BENCHMARKS.md) together with their
  protocols and caveats; the banned list in ``PROOFS.md`` (including *revolutionary,
  unbeatable, universally superior, guaranteed better*) applies to every future document in
  this repository as well.

---

## 12. One-paragraph summary

JevX specifies a typed, schema-constrained, calibrated decision layer over frozen-or-tuned
HF-compatible backbones, with proofs that hold *in real arithmetic under explicit
assumptions* (T1–T8), runtime invariants that hold *because code asserts them* (D8.I, (11.3)),
statistical properties that hold *only on exchangeable data* (calibration, risk–coverage),
and empirical properties that hold *only where measured on this CPU-only machine, at the
support level actually earned* (currently: per-family support levels 3/2 with executed
  evidence, and SST-2/latency/memory numbers exactly as reported — with documented gaps
  wherever the environment blocks measurement). Nothing here is a claim of
correctness for individual decisions, statistical independence between parallel answers,
superiority over any alternative, or knowledge of any proprietary system.
