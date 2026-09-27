# Failure Modes — digest of reports/FAILURE_ANALYSIS.md

Digest of [reports/FAILURE_ANALYSIS.md](../reports/FAILURE_ANALYSIS.md) (the
source of truth — every row below cites it). Environment for every row:
Windows 11 (10.0.26100), Intel64 Family 6 Model 78, 4 logical cores, ~8 GB
RAM, Python 3.14.3, torch 2.14.0+cpu (CPU-only), transformers 5.16.1,
2026-09-24. Executing tests: `tests/unit/test_failure_modes.py` (18),
`tests/unit/test_numerical_edge.py` (6), `tests/unit/test_padding_determinism.py`
(6) — **29 passed, 1 skipped** on the report's final run of these files; the
skip is row 11. Every executed row asserts a *useful* error: `pytest.raises` +
a message substring naming the offending argument, or (row 12) the structured
fields of `UnsupportedRepresentationError`.

## 1. The 16 failure rows (§90)

| # | Scenario | Outcome | Error type / asserted contract (verbatim substring, FAILURE_ANALYSIS §rows) |
|---|---|---|---|
| 1 | empty questions list | PASS | `ValueError` — "questions: expected at least one question (m >= 1)." |
| 2 | empty options for choice | PASS | `ValueError` — options=[…] → "requires a schema … axiom D3.1"; `[[]]` → "schema has K=0 options" |
| 3 | single option K=1 choice | PASS (works) | probabilities exactly `[[1.0]]`, index 0, confidence 1.0 |
| 4 | duplicate options | PASS (allowed) | rows sum to 1, identical texts → identical probs, index in schema |
| 5 | 5000-char option description | PASS (works) | finite probs, sum 1, no tokenizer crash (hashed E_C path never truncates) |
| 6 | long state (~2000 tok vs 64-position table) | **BUG FOUND → FIXED** | `ValueError` — "state token length 2000 exceeds the backbone's learned position limit 64 (max_position_embeddings) — truncate … JevX never silently truncates a state." Boundary (T == L) passes |
| 7 | attention_mask batch ≠ input_ids batch | PASS | `ValueError` — "attention_mask: batch 3 does not match the state batch 2." |
| 8 | invalid mask (negative / float) | **BUG FOUND → FIXED** | negative ints → `ValueError` "expected values in {0, 1} … first at index (0, 1) with value -1"; float → `ValueError` "…got floating dtype torch.float32 — binarize…"; float option_mask at fusion → `TypeError` ".bool()"; bool + 0/1 ints still accepted |
| 9 | NaN pre-encoded questions | PASS | `ValueError` from `check_finite` naming the tensor ("question_representations H_Q: tensor contains non-finite values: 32 NaN …") |
| 10 | long dtype where float expected | PASS | `TypeError` — "questions: the pre-encoded tensor must be floating point, got torch.int64." |
| 11 | wrong-device input | **SKIP (documented)** | device-generic test; this host has no second torch device (CUDA/MPS/XPU absent) — skip reason names the absent devices; not a fake pass |
| 12 | incompatible model (forward returns only `logits`) | PASS | `UnsupportedRepresentationError` with all structured fields: `detected_outputs=['logits']`, attempted/failed strategies listed with reasons, non-empty `suggestion`, message contains "Unsupported representation" + "Suggestion:" |
| 13 | vision state without processor | PASS | `ValueError` — "prepare_inputs: raw modality payloads ['images'] require processor= …" |
| 14 | remote-code config, `trust_remote_code=False` | PASS | exception message contains `trust_remote_code`; asserts **no silent execution**: module never lands in `sys.modules`, no `.py`/`__pycache__` artifacts |
| 15 | quantized model (bitsandbytes) | **ENVIRONMENT-BLOCKED** | `import bitsandbytes` → `ModuleNotFoundError`; CPU-only host — **no test written** (fake tests forbidden) |
| 16 | save/load after fitted isotonic calibration | PASS | sidecar `jevx_calibration.pt` written/restored, `is_fitted=True`, breakpoints `torch.equal`, decisions equal within 1e-6, `calibration_metadata={'method':'isotonic','applied':True}` both sides |

## 2. Environment-blocked cases (never faked)

| Row | Blocker (observed) | What would unblock |
|---|---|---|
| 15 quantized inference | CPU-only torch build; `bitsandbytes` raises `ModuleNotFoundError` | GPU host + `pip install bitsandbytes` |
| 11 cross-device input | no second torch device (CUDA/MPS/XPU all absent) | any second device — the test auto-enables and asserts a device-naming error |

This is why the suite's single documented skip exists (docs/TESTING.md §3) and
why row 15 has **no** test at all (reports/FAILURE_ANALYSIS.md
"Environment-blocked rows"). Quantized/GGUF kernel paths are out of scope of
the universality statement (docs/LIMITATIONS.md §2).

## 3. Bugs found by these rows (all in `jevx/**`, fixed; report is source)

1. **Row 6** — cryptic kernel error for over-length states → eager
   `_learned_position_table_limit` + descriptive pre-check in `forward`
   (`jevx/modeling_jevx.py`; `wpe` naming bug for GPT-2 found by the first
   full-suite run and fixed).
2. **Row 8** — invalid attention-mask values accepted silently → boundary
   validation (bool/0-1 ints only; float/negative rejected with remedies).
3. **§74** — stateless `attention_pool`/`learned_pool` not eval-deterministic →
   per-call re-init inside `torch.random.fork_rng(devices=[])` with a fixed
   seed; caller RNG restored; bitwise-identical repeats asserted.
4. **Serialization** — fitted-sidecar round trip disagreed at ~1e-8 relative
   (float64 materialization vs wrap-path float32) → materialize floating
   buffers at the reference dtype; serialization suite 5 passed.

Full detail with file:line: reports/FAILURE_ANALYSIS.md "Bugs found and fixed".

## 4. Long-sequence / padding / determinism contracts (summaries)

* **§72 long sequences:** L ∈ {8, 64, 512} runs on a 512-position model; L >
  limit → descriptive `ValueError` naming both lengths (never truncates);
  boundary works for encoder and decoder families; RoPE backbones
  (tiny_llama) have `_learned_position_limit is None` → no artificial cap.
* **§73 padding:** decoder left/right vs unpadded max_abs_error
  **7.153e-07** (both) / left vs right **7.749e-07**; full `.choice` batched
  row-0 vs single **2.980e-08**, longest row **0.000e+00**; encoder right-pad
  vs unpadded **0.000e+00**; all 8 masked-pooling pad cases **0.000e+00** —
  all asserted < 1e-5 (FAILURE_ANALYSIS measured table).
* **§74 determinism:** within-process eval determinism asserted bitwise
  (`torch.equal` on fp32 CPU probabilities); **cross-process bitwise identity
  is a documented non-goal** (oneDNN/OMP thread partitioning).

## 5. §77 supported-claim checklist → enforcing test files

§77 (failure transparency / supported-claim discipline, as cited by
[TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md)
"failure transparency §77") requires that any *supported* claim be backed by
the following seven points. Mapped to what enforces each in this repository
(the checklist items below are this repository's operative reading of §77,
labeled [DESIGN] — the §-numbered requirements themselves live outside the
repo):

| # | Point (a claim is only "supported" if…) | Enforced by |
|---|---|---|
| 1 | the contract is documented where the API is documented | [docs/API.md](API.md) §15 exception summary + §1.3 raises table (every raise site listed) |
| 2 | at least one executable test asserts it | the raising test: `tests/unit/test_failure_modes.py` (18 rows), `tests/unit/test_numerical_edge.py` (§72), `tests/unit/test_padding_determinism.py` (§73/§74), plus per-area unit files (docs/TESTING.md §1 table) |
| 3 | failure is a *useful* error: typed exception + message naming the offending argument (or structured fields) | `pytest.raises + message-substring` pattern on every FAILURE_ANALYSIS row; structured fields asserted for row 12 (`UnsupportedRepresentationError`) |
| 4 | environment-blocked cases are declared untested, never counted as passes | row 11 → the suite's one documented skip with reason; row 15 → no test written (FAILURE_ANALYSIS "Environment-blocked rows"; docs/TESTING.md §3) |
| 5 | any quantitative statement carries its measured source and environment tuple | every number in FAILURE_ANALYSIS cites its test run and the shared environment block (LIMITATIONS §1 reporting rule) |
| 6 | support scope is stated per family/level — inventory is never support | [TRANSFORMERS_COMPATIBILITY_MATRIX.md](../TRANSFORMERS_COMPATIBILITY_MATRIX.md) levels + evidence overlay; [docs/SUPPORTED_MODELS.md](SUPPORTED_MODELS.md) §1 (§79) |
| 7 | fallbacks are enumerated ladders with recorded errors (no silent substitution) and failure demotes support | factory/featurizer/config-peek ladders record every attempt (`jevx/modeling_jevx.py::_wrap_base_model`, `_resolve_featurizers`, `_peek_config`) asserted in `tests/unit/test_auto_factory.py` (8 tests incl. both peek-fallback regressions); matrix demotion-on-failure rule (matrix header; REAL_MODEL_VALIDATION Appendix C) |
