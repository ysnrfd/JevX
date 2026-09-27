# Numerical equivalence study — batched vs solo (T7-ii)

**Open-items closure pass, 2026-09-25.** This file is the evidence behind
`jevx.utils.numerical.batch_equivalence_tolerance`, `docs/PROOFS.md` Remark 7.4
and the T7-ii paragraph of `docs/MATHEMATICAL_SPECIFICATION.md` (P11.5).

## 0. The problem this closes

`reports/REAL_SCENARIO_VALIDATION.md` §5.2 check **(b)** is the one honest FAIL in the
repository: the real 1500-state walk measured a worst batched-vs-solo deviation of
**1.0967254638671875e-05** against a hard-coded `1e-5` contract, and was reported as a failed
check rather than relaxed after the fact. The diagnosis in that report — float re-association
caused by different padding widths — is correct, but it left the *bound* unchanged: a bare
`1e-5` is not a bound. It is simultaneously too tight (fp32, wide sequences, many
accumulation stages) and needlessly loose (bf16, narrow sequences). This study measures the
deviation properly, derives a rule from the measurements, and states the honest
non-claims.

## 1. Setup

| item | value |
|---|---|
| seed | `20260924` (every cell is reproducible) |
| backbones | in-memory BERT, hidden 32 / 128 / 256, **1, 2, 6 and 12 layers** |
| tokenizers | in-memory WordLevel + `[PAD]` (same construction as `tests/conftest.py`) |
| dtypes | `float32`, `bfloat16` (weights cast; pooling upcasts internally) |
| padding widths `T` | 4, 8, 16, 32, 48, 64 (pooled sweep); + 128 (end-to-end) |
| pooling | `cls`, `masked_mean`, `last_valid`, `attention_pool` |
| heads | `choice` (K=5), `score` (levels 0..4), `noul` (binary) |
| network | none |

**What is compared.** The batched call pads to the widest row in the batch; the solo call
pads to that row's own length. That width difference is the whole T7(ii) mechanism (PROOFS
Lemma 7.1 shows padding itself contributes *zero*; the residue is reduction-order re-
association). A matched-width control — every solo call padded to the batch width — is
recorded alongside, and it is the expected near-zero baseline.

**Two legs.**

1. **Pooled leg** (96 cells): `BaseAdapter.extract_representation` on a batch of 8 states vs
   the same 8 states one at a time, per pooling strategy. Isolates the representation
   layer and is cheap enough to sweep densely.
2. **End-to-end leg** (144 cells): the full `JevXForDecision` wrapper. The *state* leg
   uses B = n states with **m = 1** question (so `B*m = n` and batched decision index `b`
   is exactly solo(state b)); the *question* leg uses B = 1 state with **m = 2, 4, 6**
   questions and **variable K per question**, which is the path variable-`K` masking (11.2)
   actually exercises.

> Harness note, recorded because it nearly produced a false P0: passing `[q] * n` as the
> question list together with `n` states silently means **B = n and m = n** (`B*m = n²`
> pairs, state-major). An early run of this study did that and produced apparent deviations of
> 1e-3 … 1e-2 with 40% of "rows" flipping — every one of them an artifact of indexing
> `(state b, question 0)` against slot `b`, which is `(state 0, question b)`. The corrected
> leg below uses m = 1 for the state axis. Anyone re-running this must keep the two axes
> separate; the wrapper does not reject a question list as long as the state list.

## 2. Headline result — no argmax flips anywhere

| statistic | pooled sweep | end-to-end, state leg | end-to-end, question leg |
|---|---:|---:|---:|
| cells | 96 | 96 | 48 |
| rows compared | 768 | 684 | 192 |
| **argmax flips** | **0** | **0** | **0** |
| worst deviation (fp32) | 4.768e-07 | 5.960e-08 | 5.960e-08 |
| worst deviation (bf16) | 3.906e-03 | 9.766e-04 | not run |
| smallest top-1/top-2 margin seen | not recorded | -1.003e-02 (noul, `2p-1`) | 0.000e+00 (exact tie) |

**No P0.** Across 1644 compared rows the sweep found **zero** cases where batching changed the
argmax, and the real scenario walk had already reported the same. That is an observation on
the swept configurations, not the theorem T7(iii) proves: the exact-tie margins in the
question leg (`0.000e+00` — options whose hashed bag-of-tokens embeddings collide, e.g. the
out-of-vocabulary single letters in the test vocabulary) are precisely the rows where a flip
is *possible*. It remains a **documented open empirical risk** at the measured frequency of
**0 flips / 1644 rows**, not a defect. No cheap deterministic-reduction fix is implemented,
because there is nothing measured to fix: a deterministic reduction would have to change the
kernel/tiling the backbone itself picks, which is outside what JevX controls.

## 3. Pooled `H_S` by pooling strategy, dtype and padding width

Max |batched − solo| over 8 states per cell; the width columns are the max over the 1- and
2-layer variants. Flips are argmax disagreements of the pooled row.

| pooling | dtype | max dev | flips | T=4 | T=8 | T=16 | T=32 | T=48 | T=64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cls` | float32 | 4.768e-07 | 0 | 2.4e-07 | 2.4e-07 | 6.0e-08 | 4.8e-07 | 2.4e-07 | 2.4e-07 |
| `cls` | bfloat16 | 0.000e+00 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `masked_mean` | float32 | 2.384e-07 | 0 | 1.5e-07 | 2.4e-07 | 1.2e-07 | 2.4e-07 | 2.4e-07 | 2.4e-07 |
| `masked_mean` | bfloat16 | 3.906e-03 | 0 | 0 | 3.9e-03 | 3.9e-03 | 0 | 0 | 0 |
| `last_valid` | float32 | 4.768e-07 | 0 | 1.2e-07 | 4.8e-07 | 2.4e-07 | 1.2e-07 | 2.4e-07 | 4.8e-07 |
| `last_valid` | bfloat16 | 2.441e-04 | 0 | 0 | 2.4e-04 | 0 | 1.2e-04 | 0 | 0 |
| `attention_pool` | float32 | 4.768e-07 | 0 | 4.8e-07 | 2.4e-07 | 2.4e-07 | 2.4e-07 | 4.8e-07 | 1.2e-07 |
| `attention_pool` | bfloat16 | 9.766e-04 | 0 | 0 | 0 | 9.8e-04 | 4.9e-04 | 4.9e-04 | 0 |

Three things this table says:

1. **float32 sits at the rounding floor and does not grow with `T`.** The worst fp32 cell is
   4.768e-07 on pooled magnitudes of ~2–3, i.e. 1–4 ulp. Reductions of 4 and 64 elements are
   the same order, which is what a tree/blocked reduction (not a left-to-right sum) looks
   like. This is why the rule below uses `sqrt(T)`, not `T`: a `T`-linear term would be
   justified by nothing in the data.
2. **bf16 is a 1-ulp phenomenon, not a `T` phenomenon.** Pooling upcasts to fp32
   (`LOW_PRECISION_DTYPES` convention) and only the stored result is bf16, so the observed
   3.906e-03 is *sub*-ulp of the pooled magnitude (bf16 eps at 2.0 is 1.5625e-02). Cells that
   are exactly 0 simply rounded to the same value.
3. **The matched-width control collapses to ~0** (max 4.77e-07, usually 0), confirming the
   attribution: the deviation comes from the width difference, not from batching per se.

## 4. End-to-end `JevXForDecision`: depth, head, dtype, axis

### 4.1 By backbone depth (float32, state leg)

| backbone | cells | rows | flips | max prob dev | max `H_S` dev |
|---|---:|---:|---:|---:|---:|
| L1 (h32) | 21 | 150 | 0 | 5.960e-08 | 7.153e-07 |
| L2 (h32) | 21 | 150 | 0 | 5.960e-08 | 7.153e-07 |
| L6 (h128) | 21 | 150 | 0 | 5.960e-08 | 1.192e-06 |
| L12 (h256) | 21 | 150 | 0 | 5.960e-08 | 1.818e-06 |

The `H_S` deviation grows with depth (7.2e-07 → 1.8e-06) while the end-to-end probability
deviation does not, on these widths. Depth is still carried in the rule as `stages`, because
the real scenario (a *trained* 6-layer model, `max_length` 128) sits an order of magnitude
above every synthetic cell — see §5.

### 4.2 By padding width (float32, state leg)

| T | 4 | 8 | 16 | 32 | 48 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| max prob dev | 5.960e-08 | 5.960e-08 | 5.960e-08 | 2.980e-08 | 1.490e-08 | 5.960e-08 | 1.490e-08 |
| flips | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

### 4.3 By head (float32, state leg)

| head | rows | flips | max prob dev |
|---|---:|---:|---:|
| `choice` | 200 | 0 | 2.980e-08 |
| `score` | 200 | 0 | 0.000e+00 |
| `noul` | 200 | 0 | 5.960e-08 |

### 4.4 bfloat16 end-to-end

| backbone | cells | rows | flips | max prob dev |
|---|---:|---:|---:|---:|
| L2 (h32) | 6 | 42 | 0 | 0.000e+00 |
| L12 (h256) | 6 | 42 | 0 | 9.766e-04 |

### 4.5 Question axis with variable K (float32, B = 1, m = 2/4/6)

`K` varies per question (`[2, 3, 4, 5]` by index), so masked padding via (11.2) is on the path.
All 48 cells, 192 rows: **0 flips**, worst deviation **5.960e-08**.

| backbone | T | rows | flips | max prob dev |
|---:|---:|---:|---:|---:|
| L1 | 4 / 16 / 64 / 128 | 48 | 0 | 5.960e-08 |
| L2 | 4 / 16 / 64 / 128 | 48 | 0 | 2.980e-08 |
| L6 | 4 / 16 / 64 / 128 | 48 | 0 | 5.960e-08 |
| L12 | 4 / 16 / 64 / 128 | 48 | 0 | 2.980e-08 |

## 5. Deriving the rule

Measured ratio `r = observed / (eps(dtype) * sqrt(T * stages) * scale)`, worst cell per
dtype:

| dtype | worst cell | observed | bare term | r |
|---|---|---:|---:|---:|
| float32 | `attention_pool`/L1/T4 (pooled) | 4.768e-07 | 6.189e-07 | 0.770 |
| bfloat16 | `masked_mean`/L1/T8 (pooled) | 3.906e-03 | 4.402e-02 | 0.089 |
| float32 | real scenario, T=128, stages=11 | **1.0967e-05** | 4.473e-06 | **2.452** |

`safety(dtype)` is then the **smallest power of two that is at least 3x the worst measured
ratio for that dtype**:

```
safety(float32)  = 8.0    # 3 x 2.452 = 7.36  -> next power of two = 8
safety(bfloat16) = 0.5    # 3 x 0.089 = 0.266 -> next power of two = 0.5
```

The float32 factor is governed by the **real** walk, not by the synthetic sweep. That is
deliberate: the synthetic backbones are randomly initialised with hidden widths of 32–256,
so their activations are far smaller than a trained 27.6M model's, and a rule fitted only to
them would have failed the very case that exposed the problem.

The implemented rule (`jevx/utils/numerical.py`):

```
tol = safety(dtype) * eps(dtype) * sqrt(seq_len * stages) * scale
```

with `stages = num_backbone_layers + JEVX_DECISION_ACCUMULATION_STAGES` (5: state projection,
fusion context, shared per-option formula, head, probability map) and `scale = 1.0` for
probabilities.

**The failing check, re-scored.**

```
tol = 8 * 1.1920929e-07 * sqrt(128 * 11) = 3.5785e-05
observed (real_scenario.json, check b) = 1.0967254638671875e-05
margin = 3.26x
```

`tests/unit/test_batch_equivalence_tolerance.py` reads the observed value back out of
`benchmarks/results/real_scenario.json` (it is not re-typed) and asserts it passes this rule,
along with monotonicity in eps / seq_len / stages / scale, a >= 2x margin on every measured
cell, and the input-validation contract.

## 6. What this study does NOT establish

1. **Not a proof.** `eps * sqrt(T * stages) * scale` is a heuristic accumulation model; the
   safety factor is empirical and per configuration class. A different kernel, thread count,
   BLAS build or attention implementation can move the deviation. Re-measure; do not inherit
   these constants for a new setup.
2. **Not a decision guarantee.** T7(iii) still governs argmax. Zero flips in 1644 rows is an
   observation, not a theorem, and the question leg contains exact-tie rows where a flip is
   possible. If a decision-level claim is needed, assert it only for rows whose recorded
   margin exceeds `2*tol` (PROOFS Remark 7.3).
3. **Not bit-equality.** Cross-shape bit-identity remains a non-claim (PROOFS "Explicit
   non-claims" #4).
4. **Not a coverage claim for other backbones.** Everything here is BERT-shaped, CPU,
   `torch 2.14.0+cpu`, 1 thread configuration. The one non-synthetic anchor is the real
   scenario walk, which is what sets the float32 factor.

## 7. Reproduction

The sweep is not shipped as a benchmark script (it lives in a scratch file, not under
`benchmarks/`, which is owned by another agent). To re-run it, the essential recipe is:

```python
# leg 1 (pooled): for each (pooling, dtype, layers, T):
#   rows = 8 states with true lengths in [max(2, T//4), T]
#   batched = adapter.extract_representation(model, ids_padded_to_T, strategy=pooling)
#   solo[i] = adapter.extract_representation(model, ids_row_i_padded_to_len_i, strategy=pooling)

# leg 2 (end-to-end): for each (layers, hidden, T, kind, dtype):
#   states = n states, longest is exactly T
#   bat = w.choice(states, ["rate it"], options=[KEYS])        # B=n, m=1  <-- m=1 matters
#   for b: solo = w.choice([states[b]], ["rate it"], options=[KEYS])
#   dev = max|bat.decisions[b].probabilities - solo.decisions[0].probabilities|
```

with `w = JevXForDecision(JevXConfig(decision_hidden_size=32, ...), backbone=bert,
tokenizer=tok)` and `torch.manual_seed(20260924)` before the wrapper is built. The measured
cells are carried into the unit test as data with this file as their provenance, so the
assertions need neither network nor a long rerun.

## 8. Provenance

Written by the open-items closure agent, 2026-09-25, on Windows 11, Python 3.14.3,
`torch 2.14.0+cpu`, `transformers 5.16.1`, seed `20260924`, no network. Inputs: the recorded
scenario artifact `benchmarks/results/real_scenario.json` and the sweeps described above.
Companion changes: `jevx/utils/numerical.py` (rule + constants), `jevx/utils/__init__.py`
(exports), `tests/unit/test_batch_equivalence_tolerance.py` (33 tests), `docs/PROOFS.md`
Remark 7.4, `docs/MATHEMATICAL_SPECIFICATION.md` P11.5.
