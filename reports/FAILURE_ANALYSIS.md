# Failure-Mode Analysis — JevX Phase 90 (+ rows for §72/§73/§74 coverage)

Environment for every assertion below: Windows 11 (10.0.26100), Intel64 Family 6
Model 78 Stepping 3 (i5-6300U class, 4 logical cores), ~8 GB RAM, Python 3.14.3,
torch 2.14.0+cpu (CPU-only — no CUDA/MPS/XPU device), transformers 5.16.1,
2026-09-24. All rows are executed tests in
`tests/unit/test_failure_modes.py`,
`tests/unit/test_numerical_edge.py` and
`tests/unit/test_padding_determinism.py` (29 passed, 1 skipped on this host);
row 15 is environment-blocked and deliberately has NO test.

Each row asserts a USEFUL error: `pytest.raises` + a message substring that
names the offending argument, or (row 12) the structured fields of
`UnsupportedRepresentationError`.

---

## The 16 failure rows (§90)

| # | Scenario | Result | Asserted contract (message / field substring) |
|---|----------|--------|-----------------------------------------------|
| 1 | Empty questions list | PASS | `ValueError`: "questions: expected at least one question (m >= 1)." |
| 2 | Empty options for choice | PASS | `ValueError`: (a) `options=[]` → "requires a schema … axiom D3.1"; (b) `options=[[]]` → "schema has K=0 options" |
| 3 | Single option K=1 choice | PASS (works) | probabilities exactly `[[1.0]]`, predicted_index 0, confidence 1.0 |
| 4 | Duplicate options | PASS (allowed) | rows sum to 1, probs in [0,1], identical texts → identical probs, index in schema |
| 5 | 5000-char option description | PASS (works) | finite probs, sum 1, no tokenizer crash (hashed E_C path never truncates) |
| 6 | Long state (~2000 tok vs tiny max_positions) | **BUG FOUND → FIXED** | `ValueError`: "state token length 2000 exceeds the backbone's learned position limit 64 (max_position_embeddings) — truncate … JevX never silently truncates a state." Boundary (len == limit) still works. |
| 7 | attention_mask batch != input_ids batch | PASS | `ValueError`: "attention_mask: batch 3 does not match the state batch 2." (names both) |
| 8 | Invalid mask (negative / float) | **BUG FOUND → FIXED** | negative ints → `ValueError` "attention_mask: expected values in {0, 1} … first at index (0, 1) with value -1"; float mask → `ValueError` "…got floating dtype torch.float32 — binarize…". Float `option_mask` at the fusion boundary → pre-existing `TypeError` "expected a torch.bool tensor … convert it with `.bool()`". bool and 0/1 int masks still accepted. |
| 9 | NaN pre-encoded questions tensor | PASS | `ValueError` from `check_finite` naming the tensor: "question_representations H_Q: tensor contains non-finite values: 32 NaN …" |
| 10 | long dtype where float expected (pre-encoded questions) | PASS | `TypeError`: "questions: the pre-encoded tensor must be floating point, got torch.int64." |
| 11 | Wrong device | **SKIP (documented)** | Device-generic test builds a cross-device mismatch only when a second torch device exists. This host: `torch.cuda.is_available()` False, no MPS/XPU → skip reason names the absent devices. Not a fake pass. |
| 12 | Incompatible model (forward returns only `logits`) | PASS | `UnsupportedRepresentationError` with all structured fields populated: `detected_outputs=['logits']`, attempted/failed strategies list every priority step (pooler_output, semantic fields, last_hidden_state, hidden_states[-1]) with reasons, non-empty `suggestion`, message contains "Unsupported representation" + "Suggestion:". |
| 13 | Vision state without processor | PASS | `ValueError`: "prepare_inputs: raw modality payloads ['images'] require processor= (AutoProcessor / AutoFeatureExtractor, audit Q8); already-processed tensors can be passed as plain dict fields instead." |
| 14 | Remote-code config, `trust_remote_code=False` | PASS | Local fixture `config.json` with `auto_map` pointing at a missing module raises (exception message contains `trust_remote_code`); asserts **no silent execution**: module never lands in `sys.modules`, no `.py`/`__pycache__` artifacts created in the fixture dir. |
| 15 | Quantized model (bitsandbytes) | **ENVIRONMENT-BLOCKED** | `import bitsandbytes` → `ModuleNotFoundError`; host is CPU-only (torch +cpu build) and bitsandbytes is absent, so no quantized checkpoint can be constructed. **No fake test written** — this row is testable only on a GPU host with bitsandbytes installed. |
| 16 | Save/load after calibration (fitted isotonic) | PASS | isotonic fit (64×4) → `save_pretrained` writes `jevx_calibration.pt` → `from_pretrained` restores `is_fitted=True`, `x_points`/`y_points` bitwise-equal (`torch.equal`), decisions match pre-save within 1e-6, `calibration_metadata={'method': 'isotonic', 'applied': True}` on both sides. |

---

## Bugs found and fixed (minimal diffs, all in `jevx/**`)

### Bug 1 — cryptic kernel error for over-length states (row 6, §72)

* **File:line**: `jevx/modeling_jevx.py` — `JevXForDecision.__init__` (new
  `_learned_position_table_limit` static method, inserted just above the
  HF embedding accessors) + `forward` (pre-check block after the state-mask
  validation).
* **Cause**: a state longer than the backbone's learned absolute position
  table (`embeddings.position_embeddings` for BERT-family, `wpe` for GPT-2)
  fell through to `torch.embedding`/`expand` and failed with
  `RuntimeError: The expanded size of the tensor (2000) must match the
  existing size (64) …` / `IndexError: index out of range in self` — no
  mention of positions, tokens, or what to do.
* **Fix**: eagerly record the smallest learned position-table size on the
  wrapper (`self._learned_position_limit`; `None` for RoPE/relative
  backbones, which are never pre-checked — verified: tiny_llama → `None`).
  `forward` now raises
  `ValueError: forward: state token length {T} exceeds the backbone's learned
  position limit {L} ({config key}) — truncate the state before calling JevX …;
  JevX never silently truncates a state.` Documented contract: **raise, never
  truncate** (silent truncation would change the semantics of the decision).
  Boundary `T == L` still passes (asserted).

### Bug 2 — invalid attention-mask values accepted silently (row 8)

* **File:line**: `jevx/modeling_jevx.py` — `forward`, the
  `attention_mask` batch-check block (extended in place).
* **Cause**: only the batch dimension was validated. Negative values
  (`[1,-1,1,-1,1]`) flowed into the backbone as-is (anything `!= 0` reads
  as "valid token", so `-1` silently meant valid); float masks
  (`[1.0,0.5,...]`) were also silently coerced. Both are near-certain
  caller bugs with no error raised anywhere in the JevX path.
* **Fix**: at the same boundary, `bool` masks pass; integer masks must be
  exactly {0,1} (`ValueError` naming the count, first offending index and
  value, plus the `binarize with (mask != 0).long()` remedy); floating masks
  are rejected with a dtype-naming `ValueError`. bool + 0/1 int masks
  (every existing test and real-tokenizer path) unchanged.

### Bug 3 — stateless `attention_pool`/`learned_pool` not eval-deterministic (§74)

* **File:line**: `jevx/representations/pooling.py` — `_attention_pool`
  and `_learned_pool` (stateless call paths).
* **Cause**: each call built a fresh `AttentionPooler`/`LearnerPooler`,
  whose parameters are drawn from the ambient RNG — two identical eval calls
  produced different pooled vectors (probe: max diff 1.2e-3 on probabilities),
  and each call advanced the caller's RNG stream as a side effect.
* **Fix**: construction now runs inside `torch.random.fork_rng(devices=[])`
  with `torch.default_generator.manual_seed(0)` — deterministic per-call
  re-init, caller RNG state restored (fork), no change for module-owned
  (trained) poolers. Bitwise-identical repeat calls asserted in
  `test_padding_determinism.py`.

### Bug 4 (caught by the first full-suite run) — GPT-2 position table named `wpe`

* **File:line**: `jevx/modeling_jevx.py` `_learned_position_table_limit`
  matched `*.wpe` but `GPT2Model` registers its table as exactly `wpe`
  (no dot prefix) → the row-6 contract didn't hold for the decoder family
  (probe regression: gpt2 L=200 still raised `IndexError`).
* **Fix**: match `lname == "wpe"` as well; decoder-family row asserted in
  `test_decoder_long_sequence_within_and_exceeding`.

### Bug 5 (caught by the first full-suite run) — fitted sidecar round-trip
  disagreed with the wrap path after a concurrent meta-path change

* **File:line**: `jevx/modeling_jevx.py` — `JevXForDecision._materialize_
  calibrator` (buffer loop, now ~line 1049: `buf_dtype = dtype if
  buffer.is_floating_point() else buffer.dtype`).
* **Cause**: `TemperatureCalibrator` registers `log_temperature` as
  float64; the `__init__` cast block aligns the whole calibrator module to
  the backbone dtype (float32) on the direct-construction path, but the
  meta-device reload path skips that cast (necessary: torch refuses to copy
  out of meta tensors) and `_materialize_calibrator` then re-created the
  buffer at its *constructed* dtype (float64). The sidecar stores float32;
  loading widened it to float64, and `temperature = log.exp()` in float64
  vs float32 disagreed at ~1e-8 relative — above the `rel=1e-12` the
  serialization contract asserts (`test_serialization_jevx.py::
  test_calibration_sidecar_roundtrip` failed with
  454.87787331498913 vs 454.87786865234375).
* **Fix**: materialize *floating-point* calibrator buffers at the reference
  dtype (the wrap-path value), keeping non-float buffers (`_fitted` bool) at
  their own dtype. Serialization suite re-run: **5 passed**.

---

## Long-sequence contract (§72) — summary

| Case | Behavior (asserted) |
|------|---------------------|
| L ∈ {8, 64, 512} with a 512-position tiny BERT | forward runs; probabilities finite, rows sum to 1, no NaN in any representation field |
| L = 512 against a 64-position backbone (tiny_bert fixture) | descriptive `ValueError` naming "512", "limit 64" and the truncate remedy |
| L = 64 against limit 64 (at boundary) | works |
| Decoder family (tiny GPT-2, n_positions=64): L=64 works, L=200 | works / descriptive `ValueError` |
| RoPE backbone (tiny_llama) | `_learned_position_limit is None` → no artificial cap |

## Padding (§73) & determinism (§74) — summary

See `reports/LATENCY_BENCHMARKS.md` for the environment block shared by all
phase reports. Measured `max_abs_error` values (decoder tiny-gpt2 left/right
vs unpadded; encoder tiny-random-bert right-pad vs unpadded; JevX-level
batched `choice` vs single) are asserted `< 1e-5` and printed by the tests;
exact numbers from the final run are quoted in the LATENCY report's sibling
section below.

Cross-process bitwise determinism is a documented **non-goal**: separate OS
processes may pick different oneDNN/OMP thread partitions or kernels, so the
contract is within-process eval determinism only (asserted exactly with
`torch.equal` on fp32 CPU probabilities).

### Measured max_abs_error values (final `-s` run, 6 passed in 10.89 s)

| Comparison | max_abs_error |
|---|---:|
| decoder tiny-gpt2: left-pad vs unpadded (adapter pooled) | **7.153e-07** |
| decoder tiny-gpt2: right-pad vs unpadded (adapter pooled) | **7.153e-07** |
| decoder tiny-gpt2: left vs right | **7.749e-07** |
| decoder full `.choice`: batched left-pad row 0 vs single call | **2.980e-08** |
| decoder full `.choice`: batched row 1 (longest) vs single call | **0.000e+00** |
| encoder tiny-random-bert: right-pad vs unpadded (adapter pooled) | **0.000e+00** |
| pool masked_mean / cls / last_valid / max: right & left vs unpadded | **0.000e+00** (all 8) |

All asserted `< 1e-5`; worst observed value is 7.7e-07 (2.5e6x inside the
tolerance). Determinism: bitwise-equal probabilities across repeated eval
calls, including with `attention_pool`/`learned_pool` after fix 3.

## Environment-blocked rows (no fake tests)

| Row | Blocker | What would unblock |
|-----|---------|--------------------|
| 15 quantized inference | CPU-only torch build; `bitsandbytes` raises `ModuleNotFoundError` | GPU host + `pip install bitsandbytes` |
| 11 cross-device input | no second torch device (CUDA/MPS/XPU all absent) | any second device; the test auto-enables and then asserts a device-naming error |

## Test counts

`tests/unit/test_failure_modes.py` + `test_numerical_edge.py` +
`test_padding_determinism.py`: **29 passed, 1 skipped** (skip = row 11,
documented reason) on the final run — verbatim pytest summaries are in the
final delivery report.
