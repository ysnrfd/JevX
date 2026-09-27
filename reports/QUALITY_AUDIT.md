
# JevX Deep Quality Audit — final hardening wave

**Auditor:** deep-quality-audit agent (read-only; this file is the only artifact created).
**Date:** 2026-09-24. **Host:** Windows 11, Python 3.14.3, torch 2.14.0+cpu, transformers 5.16.1.
**Method:** (1) re-read docs/MATHEMATICAL_SPECIFICATION.md, PROOFS.md, RESEARCH.md, ARCHITECTURE.md;
(2) re-read reports/TRANSFORMERS_SOURCE_AUDIT.md + TRANSFORMERS_COMPATIBILITY_MATRIX.md and
spot-checked 10 vendor claims at their cited vendor/transformers lines;
(3) read all 49 jevx/ files, the full tests/ tree, benchmarks/, reports/, docs/, README, examples —
then executed probes: unit subset `223 passed in 9.39s` (test_decisions/test_modeling_jevx/
test_confidence/test_calibration + tests/numerical), `pytest --collect-only` → **780 tests collected**
(docs claim 769), example_01 re-run green, plus two in-process probes: JevXConfig
dtype/torch_dtype/decision_type round-trips OK; `return_base_output=True` end-to-end populated and
bit-identical to the raw backbone (max_abs diff 0.0).
**Concurrent work observed:** jevx/conversion/ changed *during* this audit (0-byte __init__ →
converter.py/equivalence.py/report.py/transplant.py + populated __init__); no pyproject.toml yet;
conversion findings describe audit-start state and must be re-verified by the fix wave.

## Summary counts

| Severity | Count | Meaning |
|---|---:|---|
| **P0** | 2 | correctness / data-loss / security-class |
| **P1** | 18 | feature gap, stale/overstated claim, missing test for shipped feature |
| **P2** | 16 | hygiene / docs polish |
| **Total** | **36** | |

---

## P0 findings

### Q-001 (P0) — jevx/conversion/ (§30 weight-conversion taxonomy, §63 grounding) was an EMPTY package
* **Evidence:** at audit start `jevx/conversion/__init__.py` was **0 bytes** — no API, no exports,
  no tests anywhere (tests/ grep for `jevx.conversion|ConversionReport|transplant_` → NONE). Only
  references: docs/ARCHITECTURE.md:53,429 (module reserved for "conversion + grafting semantics;
  section 5"). Concurrently during this audit the conversion agent landed `converter.py /
  equivalence.py / report.py / transplant.py` and a populated `__init__.py` exporting `convert`,
  `ConversionMode/ConversionReport/EquivalenceKind`, `transplant_classifier_head /
  transplant_state_encoder_proj / probe_head_compat`, `check_base_equivalence /
  check_representation_equivalence / check_decision_roundtrip / check_functional_equivalence`.
  Still NOT re-exported from `jevx/__init__.py` and still **zero tests reference the conversion API**.
* **Impact:** §30/§63 deliverable invisible from the package root, untested, unverified; the
  ARCHITECTURE grounding table mapped a module that had no content.
* **Recommended fix:** conversion-agent finishes the subsystem: (a) re-export the intended public
  names from `jevx/__init__.py` (or document why not), (b) unit tests for each ConversionMode/
  EquivalenceKind honesty invariant, transplant shape-mismatch rejection, and the three equivalence
  checkers, (c) an API.md section for jevx.conversion. §30/ARCHITECTURE §5 scope alignment: taxonomy
  of *reuse / transplant / fresh-init / (reserved architectural)*, grafting must check exact shape
  match + re-check semantics, and every claim is "reported as transplant, never as identity".
* **Owner:** conversion-agent.

### Q-002 (P0) — Noul / single-column confidence hard-wired to 1.0: abstention can NEVER fire for noul (contradicts spec §9(e))
* **Evidence:** `jevx/confidence/base.py:424-430` — `if probabilities.shape[-1] == 1: return
  torch.ones(...)` short-circuits **before** method dispatch. Spec
  `docs/MATHEMATICAL_SPECIFICATION.md:803-805` (Def 9(e)): "For Noul/Threshold with a single sigmoid:
  c_max = max(p, 1−p) ∈ [0.5, 1]". The repo disagrees with itself: `jevx/utils/numerical.py:224-227`
  (ECE/NLL confidence, K==1 branch) **does** use `max(p, 1−p)`. The rule is pinned by
  `tests/unit/test_confidence.py:104-113` (probs=[[0.7]] → conf==1.0 exactly).
  **Auditor probe:** wrapper `.noul` → `p=0.4954, confidence=1.0`;
  `confidence_threshold=0.6` → `abstained=[False]`. examples/example_01 prints it live:
  "P(yes)=0.4950 … confidence=1.0000".
* **Impact:** Def 10 abstention (10.1) and the documented `confidence_threshold=` parameter are a
  **silent no-op for every noul decision**; noul confidences fed to risk–coverage utilities are
  meaningless; spec, metrics code and confidence engine contradict each other.
* **Recommended fix:** in `ConfidenceEngine.forward` treat a single column as the binary pair
  (p, 1−p): confidence = `max(p, 1−p)` (genuine K=1 *choice* rows stay 1.0 because p==1 there —
  this also matches the ECE path and spec §9(e)); update `test_k_one_edge_gives_one` expectations
  (0.7 → 0.7), docs/CONFIDENCE.md §1, and add a noul abstention test where the threshold fires.
* **Owner:** fix-agent.

---

## P1 findings

### Q-003 (P1) — The "ValidationEngine" component does not exist, yet PROOFS.md claims it asserts invariants "on every call"
* **Evidence:** `grep ValidationEngine jevx/ → 0 hits`. Named as one of the eleven components in
  docs/ARCHITECTURE.md:26,49,88,291,407 (L5 guardrail: schema-hash equality, type-tag equality,
  per-question Σp=1±tol), spec D1.1/D8.I/(11.3) (MATHEMATICAL_SPECIFICATION.md:83,736,947),
  RESEARCH DC-13 and the FINAL_REPORT L5 table — and **docs/PROOFS.md:796** classifies
  "Schema-hash equality across calibration (D8.I); type-tag equality; masked probabilities = 0;
  |Σp−1| ≤ tol" as *[Implementation invariant] "True because runtime validation asserts it on every
  call"*. No schema-hash or type-tag check exists anywhere; Σp=1 holds by construction
  (masked_softmax) and by tests, not by any runtime assertion.
* **Impact:** overstated claim inside the proofs document; ARCHITECTURE's "eleven components" is
  really 10/11; spec's "The ValidationEngine asserts (11.3)" is false as written.
* **Recommended fix:** either (a) add a small runtime validation step (jevx/validation.py) called
  from `forward` at the calibration boundary doing shape/K/dtype + Σp checks, or (b) rewrite
  PROOFS.md:796 and ARCHITECTURE §2 to "holds by construction, asserted by the test suites" —
  do not leave "on every call" standing. Also reconcile LIMITATIONS:170 (Q-009).
* **Owner:** fix-agent.

### Q-004 (P1) — Every documented suite count is stale: the tree now collects 780 tests (docs say 769/768)
* **Evidence:** auditor run `python -m pytest --collect-only -q` → **780 tests collected**;
  780 = unit 704 + integration **32** + golden 3 + numerical 41. docs/TESTING.md §1 claims 769 and
  lists integration = 21 (3+11+7), omitting `tests/integration/test_qwen35_hybrid.py` (**11 tests**,
  collector confirms individually). `_collect_counts.json` (root) also says 21 — stale.
  README "Honest status" (768 passed … 264.90s), FINAL_REPORT ("768 passed … 313.55s"),
  FINAL_VERIFICATION ("769 collected") and the ARCHITECTURE status note all quote runs that
  **excluded the Qwen3.5 battery** (dot-count of `_final_audit_pytest.log` = 769 outcomes).
* **Impact:** the headline "final suite green" claim is not established for the current tree; the
  TESTING.md per-file table and runtimes (~4 min) cannot hold once the 853M-param battery is
  collected (its own report shows one 20-row forward at 423.7 s; the fix wave already budgets ~30 min).
* **Recommended fix:** run the full suite once with the Qwen battery included, then update
  TESTING.md §1 + §6 runtimes, README status row, FINAL_REPORT/FINAL_VERIFICATION counts (or mark
  them "as of the pre-Qwen run"); delete stale `_collect_counts.*`.
* **Owner:** fix-agent.

### Q-005 (P1) — Qwen3.5 test hardcodes an absolute user path with a bare assert: suite not portable
* **Evidence:** `tests/integration/test_qwen35_hybrid.py:78-81` sets
  `SNAPSHOT = Path(r"C:\Users\ysnrfd\.cache\huggingface\hub\models--Qwen--Qwen3.5-0.8B\snapshots\2fc06364715b967f1860aea9cf38778875588b17")`
  and `line 187: assert SNAPSHOT.is_dir(), f"checkpoint snapshot missing: ..."` — **no
  skip/allow_module_level guard**, unlike every other env failure in the repo (docs/TESTING.md §3
  convention: environment problems become classified `pytest.skip`).
* **Impact:** on any other machine or after cache eviction, `python -m pytest` **fails** at this
  assert instead of skipping — the suite cannot run green off this exact host.
* **Recommended fix:** resolve via `huggingface_hub.snapshot_download("Qwen/Qwen3.5-0.8B",
  revision=...)` with cached-file fallback, or `pytest.skip("Qwen3.5 snapshot not cached: ...",
  allow_module_level=True)` when the dir is missing (keep the pinned revision in the message).
* **Owner:** fix-agent.

### Q-006 (P1) — Spec Def 5.e (cross-attention fusion) not implemented while ARCHITECTURE/RESEARCH claim it
* **Evidence:** spec `MATHEMATICAL_SPECIFICATION.md:385-395` defines fusion variant **(e)
  cross-attention** (option queries over H_S^seq, Eq. 5.e) and §4 retains H_S^seq for exactly that.
  `FUSION_REGISTRY` (`jevx/representations/fusion.py`) has **7 kinds — add, additive, bilinear,
  concat, dot, gated, mlp — none cross-attention**; nothing in jevx/ ever feeds
  `Representation.sequence_output` into fusion. Yet docs/ARCHITECTURE.md:45 says "DecisionFusion —
  One of five variants (Def. 5a–e)" (and :79 diagram "5 variants"), and RESEARCH.md:343 DC-7 says
  "Five fusion variants kept interchangeable (concat/gated/additive/bilinear/**cross-attention**)".
  FINAL_REPORT honestly lists only the 7 implemented kinds.
* **Impact:** missing spec feature + overstated architecture claims in the doc that says it
  "describes the code as built".
* **Recommended fix:** either implement a `cross_attn` fusion (the retained sequence makes it
  feasible: adapters/base.py:341-343) **or** reword ARCHITECTURE.md:45,79 and RESEARCH DC-7 to
  "5.e specified, not implemented; implemented set = the 7 documented kinds". Docs-first is cheaper.
* **Owner:** fix-agent (docs) / modality-agent (code) — owner decision.

### Q-007 (P1) — Two incompatible support-level scales documented; the honesty contract points at the wrong one
* **Evidence:** docs/ARCHITECTURE.md §6 defines levels 0 probe-failed / 1 loads / 2 forward-pass /
  3 head-integration-on-smoke / 4 suites-pass, and **docs/LIMITATIONS.md:132-137** ("wins any
  conflict") says verbatim "The levels are defined in ARCHITECTURE.md 6". The live matrix
  (TRANSFORMERS_COMPATIBILITY_MATRIX.md:9-13) uses a **different scale**: 4 = native pooled +
  HF-API-verified + real-dataset quality result; 3 = native states + established pooling executed;
  2 = convention-dependent lossy pooling (executed); 1 = source-verified; 0 = no route. Every
  headline number (13×L3, 7×L2 — README, FINAL_REPORT, SUPPORTED_MODELS) is computed on the
  **matrix** scale; on the ARCHITECTURE scale the executed decoders (llama/qwen2/gpt2: load+
  forward + heads + save/load green) would be ≥3, not 2.
* **Impact:** per-family level claims are ambiguous; the two "authoritative" definitions disagree
  about what a level means, so "level 2" for llama can be read two ways.
* **Recommended fix:** unify on the matrix scale — rewrite ARCHITECTURE §6 and LIMITATIONS §7 to
  the matrix's evidence definitions (or make the matrix explicitly instantiate and cite one
  definition). One table, one meaning.
* **Owner:** fix-agent.

### Q-008 (P1) — Stale "no benchmark results" notes (known finding #8): four sites; benchmarks exist
* **Evidence:** `docs/PROOFS.md:797` — model-performance claims: "None is made in this repository
  yet — no benchmark has been run"; `reports/REAL_MODEL_VALIDATION.md:68` — "no benchmark results
  committed at time of writing"; `TRANSFORMERS_COMPATIBILITY_MATRIX.md:22` "(no results committed at
  time of writing)" and `:92` "(4 = 3 + real-dataset quality … none committed yet)". Actual state:
  benchmarks/results/ holds exp1…exp6 + run summaries (18 JSON), DATASET_BENCHMARKS/ABLATIONS/
  LATENCY/MEMORY are committed, README/FINAL_REPORT quote their numbers.
* **Impact:** PROOFS.md (the taxonomic authority) tells readers no benchmark exists — wrong today;
  the level-4 gate wording ("quality results for the family") is ambiguous (results exist for the
  SST-2 × bert-L2-H128 backbone only, not for the matrix's cited checkpoints).
* **Recommended fix:** rephrase all four to the precise truth: "quality results exist only for
  google/bert_uncased_L-2_H-128_A-2 on SST-2; no per-family quality result for the matrix's cited
  checkpoints — hence no level 4"; delete "no benchmark has been run".
* **Owner:** fix-agent.

### Q-009 (P1) — LIMITATIONS §8 contradicts itself: "(not yet written) test suites"
* **Evidence:** `docs/LIMITATIONS.md:170` — "what the ValidationEngine and the (not yet written)
  test suites are for" — while the same section (:156-161, "updated 2026-09-24") states tests exist
  and 768 passed. Phase-2-era sentence left behind (the only other skeleton-era survivor is the
  MATHEMATICAL_SPECIFICATION "Phase 2 deliverable" status line, which is a fine historical label).
* **Impact:** the honesty-contract document contains a false present-tense claim, stacked on the
  missing ValidationEngine (Q-003).
* **Recommended fix:** replace with "…what the runtime guards and the test suite are for"; align
  with whatever Q-003 decides.
* **Owner:** fix-agent.

### Q-010 (P1) — §31 surface and the decision_type config pin are shipped but untested
* **Evidence:** (a) `return_base_output` appears **zero times in tests/**; the only base_output
  assertion is `tests/unit/test_outputs_jevx.py:84: assert out.base_output is None` (the default-off
  half). Auditor probe confirms the positive path works end-to-end (populated, max_abs 0.0 vs raw
  backbone) — purely a missing test, so §31's forward surface can rot silently.
  (b) The documented resolution rule "argument → config.decision_type → 'choice'"
  (docs/API.md:153; `modeling_jevx.py:1401-1403`): the **middle branch** —
  `config.decision_type` set by `from_pretrained(decision_type=...)` at :781/:927 — and its
  save/load round-trip have **no test** (tests grep for "config.decision_type" → none). Auditor
  probe: `JevXConfig(decision_type=...)`, `to_dict()`, `from_dict()` all round-trip (works, untested).
* **Impact:** §45 factory kwarg and §31 flag are documented in API.md with zero regression cover.
* **Recommended fix:** three tests: (1) `forward(..., return_base_output=True)` →
  `base_output.last_hidden_state` equals the raw backbone output; (2) pin via
  `from_pretrained(id, decision_type="score")` then argument-less forward resolves "score";
  (3) save → reload → `config.decision_type` survives.
* **Owner:** fix-agent.

### Q-011 (P1) — Public API symbols with NO test reference (untested surface)
* **Evidence:** scanning all `tests/*.py` text against package exports, these documented/exported
  names are never mentioned: `return_base_output` (Q-010); `DecisionDataset`
  (`training/data.py:76`, in training `__all__`, API.md §14.4); `confusion_matrix` and
  `extract_logits` (training `__all__`); `l2_regularization` (training/objective `__all__`);
  `register_calibrator` / `build_calibrator` / `validate_fit_inputs` / `IdentityCalibrator`
  (calibration `__all__`); `is_text_state` / `tokenize_texts` / `observed_output_fields` /
  `NATIVE_REPRESENTATION_FIELDS` (adapters `__all__`); `BUILTIN_ADAPTER_CHAIN`;
  `jevx.__version__` (no test asserts the three version literals agree — Q-018).
  Top-level `jevx/__init__` is otherwise well covered (16/17 `__all__` names referenced).
* **Impact:** shipped, documented (API.md §7/§11/§14) surface can break without a red test.
* **Recommended fix:** add `tests/unit/test_public_surface.py`: import every name from each package
  `__all__`, assert version-triple equality, and small behavioral smokes for `DecisionDataset`,
  `confusion_matrix`, `extract_logits` over a toy batch.
* **Owner:** fix-agent.

### Q-012 (P1) — Calibration runs on UNMASKED logits: isotonic's top-label protocol can read padded columns in variable-K batches (untested interaction)
* **Evidence:** `modeling_jevx.py:1629-1638` — `logits = head(...)` is the **raw unmasked**
  `[g, K_max]` tensor and `calibration.apply(logits, "logits")` runs on it. Isotonic
  (`calibration/isotonic.py:118-127`) does `conf, pred = probs.max(dim=-1)` **over all K_max
  columns**, including padded rows whose option features were zeroed by
  `features.masked_fill(~option_mask.unsqueeze(-1), 0.0)` (fusion.py:122-126) — their logits are
  arbitrary (0 for dot/bilinear, MLP-bias for mlp) and can exceed valid negative logits.
  `head.build_output(calibrated, option_mask)` then re-masks: if the isotonic "top" landed on a
  masked column the affine scale cancels under re-normalization → **calibration silently no-ops
  for that row**; otherwise the correction is misplaced. `option_mask is not None` whenever a
  group mixes K (modeling_jevx.py:1618). Grep: **no test combines a fitted calibrator with
  option_mask / variable-K**.
* **Impact:** silently wrong probabilities in a legal call shape (mixed-K batch + fitted isotonic);
  D8.I intent ("calibrate the question's own distribution") violated.
* **Recommended fix:** mask the logits before calibrating, or calibrate only valid columns
  (per-row slice), and document the choice; add a variable-K + fitted-isotonic regression test
  asserting the calibrated row equals the calibration applied to the valid slice alone.
* **Owner:** fix-agent.

### Q-013 (P1) — Known #3 open: input_ids+pixel_values raises a cryptic UPSTREAM AttributeError (exact sites)
* **Evidence:** forward merges extra tensor kwargs into state inputs at
  `jevx/modeling_jevx.py:1267-1268`; the failure surfaces from the backbone call at
  `jevx/adapters/base.py:282` (`outputs = model(**fwd)`) with **no JevX context wrapper** — observed
  as `AttributeError: 'NoneType' object has no attribute 'device'` (image_grid_thw missing), recorded
  verbatim in `benchmarks/results/qwen35_jevx_validation.json:262,271` and
  `tests/integration/test_qwen35_hybrid.py:1103-1104` (the test **records** the cryptic string as
  expected instead of asserting an actionable error; QWEN35_HYBRID_VALIDATION.md:81 lists it as
  "documented finding, not defect"). Contrast: `BaseAdapter.validate()` DOES wrap failures into
  structured errors (`adapters/base.py:502-514`) — the hot path does not.
* **Impact:** §90-style failure-transparency gap: a plausible caller mistake produces a raw
  upstream traceback with no JevX context.
* **Recommended fix:** wrap the backbone call in `extract_representation` with
  `except Exception → raise UnsupportedRepresentationError from exc` adding model class, prepared
  input keys and a suggestion ("companion vision metadata (e.g. image_grid_thw) missing — pass the
  full processor output"); update the qwen test to assert the JevX-context message.
* **Owner:** fix-agent.

### Q-014 (P1) — EncoderDecoderAdapter._run_full still has the pre-fix explicit-only use_cache gate
* **Evidence:** the Qwen bug class was fixed in `adapters/base.py:258-278` (inject
  `use_cache=False` when the forward declares it **or has **kwargs**), but
  `adapters/encoder_decoder.py:279-287` (`_run_full`) still does
  `if "use_cache" in params and "use_cache" not in fwd` — kwargs-only forwards (the v5
  TransformersKwargs pattern) keep allocating caches in the `decoder_state` /
  `encoder_decoder_fusion` modes and the encoder-state full-forward fallback.
* **Impact:** resource waste + two implementations of one documented contract.
* **Recommended fix:** factor the shared gate (`_inject_use_cache_false(model, fwd)` in
  adapters/base.py) and call it from `_run_full`; add a kwargs-only-signature unit test.
* **Owner:** fix-agent.

### Q-015 (P1) — reports/FINAL_VERIFICATION.md contains four checkable false "machine-checked" statements
* **Evidence:** (1) "docs/MATHEMATICAL_SPECIFICATION.png present" — `existsSync → false` (docs/ has
  only .md + .gitkeep); (2) "docs/ — 15 .md files machine-listed" — actually **16**
  (docs/HYBRID_ATTENTION_ANALYSIS.md omitted); (3) "reports/ — 8 reports" — actually **10 .md**
  (QWEN35_HYBRID_VALIDATION.md and FINAL_VERIFICATION itself omitted); (4) "benchmarks/results/
  (17 JSON files)" — actually **18** (qwen35_jevx_validation.json added later).
* **Impact:** the §100 gate's own exhibits are stale/false — the verification report fails its own
  "machine-checked" standard.
* **Recommended fix:** correct the four counts; drop or generate the PNG claim.
* **Owner:** fix-agent.

### Q-016 (P1) — Golden coverage: 3 of 18 validated models
* **Evidence:** `tests/golden/fixtures/` = golden_tiny_random_bert / golden_tiny_random_t5 /
  golden_tiny_gpt2 only. The other 15 executed checkpoints (roberta, distilbert, albert,
  deberta-v2, all-MiniLM, bart, qwen2, llama, vit, clip, siglip, wav2vec2, ast, whisper,
  llava_next) have **no golden re-derivation** — coverage is matrix gates only (live reference
  equality), which cannot pin 7-dp output drift across versions the way fixtures can.
* **Impact:** regression-breadth gap for shipped level-3 families (see matrix below).
* **Recommended fix:** extend `tests/golden/test_golden_references.py` generate/derive to the
  cheapest 4–6 missing families (roberta, bart, llama, vit, wav2vec2, clip) via the documented
  regeneration procedure; state in TESTING.md why the rest are matrix-only.
* **Owner:** fix-agent.

### Q-017 (P1) — Test-taxonomy categories with NO files (§67-68 walk)
* **Evidence:** present: unit (23 files), integration (4), golden (1), numerical (2);
  `tests/property/` and `tests/regression/` are .gitkeep-only (documented in TESTING.md §1).
  **Absent entirely:** distributed/multi-process (grep DistributedDataParallel/world_size → 0),
  generation (by design — no generation machinery), **sharded checkpoints** (no max_shard_size /
  index.json round-trip test), quantized (§35 env-blocked, row 15), cross-device (row 11 skip only),
  **§72 long-sequence for vision/audio** (test_numerical_edge.py covers bert/gpt2/llama only),
  §73 padding for vision/audio (text encoder+decoder only; Qwen left-pad measured-not-asserted per
  QWEN35 §6c).
* **Impact:** §67-68 taxonomy lists categories the suite never instantiates; §72/§73 are text-only
  in practice (FAILURE_MODES §4 is honest about what it quotes, but the spec scope is broader).
* **Recommended fix:** list absent categories explicitly in TESTING.md §1 with reasons ("not
  applicable / environment-blocked / not implemented", §77 point 4); add a cheap sharded-save
  round-trip test (`save_pretrained(max_shard_size=...)` → reload) and one vision/audio long-input
  case where cheap.
* **Owner:** fix-agent (docs) / wontfix-env (distributed, quantized, cross-device).

### Q-018 (P1) — Enterprise blockers: no LICENSE, no CI, version triple-sourced without a guard
* **Evidence:** root listing has **no LICENSE/COPYING, no CHANGELOG, no .github/CI config** and no
  pyproject (known open — README "Install" documents it; packaging-agent concurrent).
  Version literals: `jevx/__init__.py:16`, `jevx/modeling_jevx.py:120` (self-described "mirrored …
  kept local to avoid a cycle"), `jevx/configuration_jevx.py:110 version: str = "0.1.0"` — three
  independent strings, **no test asserts equality**, stamped into jevx_metadata.json and config.json
  respectively, so they can drift silently.
* **Impact:** not redistributable without a license; no automated regression gate; checkpoint
  metadata could disagree about its own version.
* **Recommended fix:** add a LICENSE (owner chooses type); single-source the version
  (`jevx/__init__` as source, others import/re-export lazily) plus a 3-way equality test;
  packaging-agent adds pyproject (+ optional CI running the unit subset).
* **Owner:** packaging-agent (+ repo owner for LICENSE choice).

### Q-019 (P1) — Known #5: learned confidence method never measured (ABLATIONS EXP-6)
* **Evidence:** reports/ABLATIONS.md EXP-6 and docs/CONFIDENCE.md §2: `learned` "recorded not-run"
  (would add a second training stage); only 4 of 5 methods have numbers. FINAL_VERIFICATION marks
  the exhibit **PARTIAL** honestly. Still open.
* **Impact:** a shipped registered method (`confidence_method="learned"`, config-doc'd) has zero
  empirical characterization; risk-coverage claims exclude it.
* **Recommended fix:** either run EXP-6 learned row (train the MLP on the calibration split,
  same seed/protocol, add one JSON row) or mark the method experimental in config/docs until measured.
* **Owner:** benchmark/fix-agent (or wontfix-env if protocol cost forbids — but then label it).

### Q-020 (P1) — Known #6: quality evidence is single-dataset (SST-2 × one frozen backbone)
* **Evidence:** DATASET_BENCHMARKS protocol (one dataset, one backbone, n=872); FINAL_VERIFICATION
  marks "Quality validation across multiple datasets/backbones" **PARTIAL — "no second dataset run
  exists"**; README scope rows repeat it. Still open and honestly labeled everywhere.
* **Impact:** every quality claim is one-dataset-scoped (H2/H3 remain hypotheses).
* **Recommended fix:** add one second dataset (e.g., glue/cola or mrpc, same backbone/protocol) or
  keep the explicit non-claim; no doc change needed unless a run happens.
* **Owner:** benchmark-agent.

---

## P2 findings

### Q-021 (P2) — ECE default bins: spec says B=10, every implementation/report uses 15
* **Evidence:** spec `MATHEMATICAL_SPECIFICATION.md:710` "(default: B = 10 equal-width bins)";
  code default is 15 in three places — `expected_calibration_error(..., n_bins=15)`
  (`jevx/utils/numerical.py:250`), `BaseCalibrator/CalibrationEngine.evaluate(n_bins=15)`
  (`calibration/base.py:210,347`), benchmarks `n_bins=15` (bench_common.py:516, run_quality.py:676);
  every reported number is labeled ECE(15)/MCE(15).
* **Impact:** code+reports are self-consistent; the spec's stated default is the outlier — a reader
  recomputing with B=10 gets different numbers.
* **Recommended fix:** change the spec sentence to B = 15 (matching the shipped default and all
  quoted results), or add "(JevX uses B=15 throughout; see CALIBRATION.md)".
* **Owner:** fix-agent.

### Q-022 (P2) — Fusion context formulas simplify Def 5.a/5.b/5.c/5.d (schema term s_C dropped) while docstrings cite "spec 5.x"
* **Evidence:** spec 5.b (`MATHEMATICAL_SPECIFICATION.md:348-353`): `g = σ(W_g[h_S‖h_Q‖s_C] + b_g)`,
  `h = g⊙φ_S(h_S) + (1−g)⊙φ_Q(h_Q) + φ_C(s_C)`. Code (`representations/fusion.py:240-246,
  176-179, 216-219, 224-227`): gate input is only `[s;q]` and **no φ_C(s_C) term anywhere**;
  concat is `W[s;q]` (2d, not the spec's 3d with s_C); additive drops LN and the 1/√3 scaling;
  bilinear is the diagonal `s ⊙ W_q q` (documented as "vector-level β_diag"). The option-side
  formula `e_k = g ⊙ W_o o_k + (1−g) ⊙ c` itself **matches** API.md:429-430 and
  `fusion.py:11-14` exactly (spec has no e_k formula — API.md is its home).
* **Impact:** partial: schemas do enter through e_k, so information is not lost, but the class
  docstrings say "spec 5.b/5.a shape/family" without noting the dropped s_C/φ_C summands.
* **Recommended fix:** one sentence in fusion.py module docstring + ARCHITECTURE §2: "the s_C
  schema-summary summands of 5.a–5.d are carried by the per-option formula instead; no φ_C term".
* **Owner:** fix-agent.

### Q-023 (P2) — Isotonic applies a piecewise-LINEAR map; spec 8.3 defines a piecewise-CONSTANT PAVA fit
* **Evidence:** spec `MATHEMATICAL_SPECIFICATION.md:673-675` "(least-squares isotonic regression;
  piecewise constant)"; code fits with PAVA (`calibration/isotonic.py:50-86` — correct weighted PAVA
  with tie grouping) but predicts by **linear interpolation between breakpoints**
  (`_interpolate`, isotonic.py:144-157; docstring: "piecewise-linear interpolation").
* **Impact:** the applied map differs from the spec's f* between fitted points (monotone either
  way; endpoints exact) — a real code-vs-spec divergence, documented only on the code side.
* **Recommended fix:** either state in the spec "piecewise-linear interpolation of the PAVA blocks
  (implementation)" or switch to step interpolation (searchsorted → y0). Pick one; test both maps
  against the PAVA block values.
* **Owner:** fix-agent.

### Q-024 (P2) — JevXConfig docstrings: invalid pooling name and a dead config knob
* **Evidence:** (a) `configuration_jevx.py:54-55`: "pooling_method: Sequence pooling for H_S
  ("mean", "cls", "last", …); None lets the base model decide" — **`"last"` is not a registered
  strategy** (POOLING_STRATEGIES has `last_valid`; passing "last" raises ValueError) and "None"
  means *adapter default* (native pooler if present else adapter default), not "the base model decides".
  (b) `representation_strategy` (configuration_jevx.py:51-53, values native/projected/adapter,
  "owned by the representations/adapters modules") is **dead**: grep shows it is only declared,
  serialized and round-tripped — never read by any computation (modeling_jevx.py:76 admits
  "metadata only"; API.md:245 documents it honestly).
* **Impact:** a user following the config docstring gets an immediate ValueError; a user setting
  representation_strategy believes it changes behavior.
* **Recommended fix:** fix the docstring (`"last_valid"`, "None = adapter default: native pooler or
  the adapter's default pooling") and either deprecate/mark `representation_strategy` as a
  metadata tag or wire it up.
* **Owner:** fix-agent.

### Q-025 (P2) — Two false docstrings about JevXOutput.state_representation / fused H provenance
* **Evidence:** (a) `modeling_outputs.py` JevXOutput docstring: "state_representation: **Fused state
  H** of shape [B, D] (H = F(H_S, H_Q, H_C)). **None unless requested**" — code always populates it
  with the *projected H_S* (`modeling_jevx.py:1532,1720`: `state_repr = self.state_proj(pooled)`),
  never the fused H, and never None. docs/API.md:287 says it correctly ("shared projected H_S").
  (b) `modeling_jevx.py:77-80` module docstring: "the per-row fused H lives inside each decision's
  provenance via capability_metadata" — **false**: the `capability_metadata` dict built at
  :1703-1717 contains no fused vectors (model_type/architecture/fields/pooling/fusion/counts only).
* **Impact:** public-output docstrings mis-describe what users receive; the fused H is in fact not
  exposed at all.
* **Recommended fix:** correct both docstrings (say "projected H_S, always populated; per-row fused
  H is internal and not exposed" — or actually expose it, e.g. in DecisionOutput, if §70/§71 need it).
* **Owner:** fix-agent.

### Q-026 (P2) — docs/SUPPORTED_MODELS.md 18-checkpoint table has a broken markdown separator
* **Evidence:** `docs/SUPPORTED_MODELS.md:36` is `"|---|`---|`---|`---|`---|`---|`---|"` —
  stray backticks break the header/separator row, so the table renders as literal text in most
  viewers (line 35 is the header, 37+ the rows).
* **Impact:** the flagship 18-model table is unreadable as a table.
* **Recommended fix:** replace line 36 with `|---|---|---|---|---|---|---|`.
* **Owner:** fix-agent.

### Q-027 (P2) — Repo hygiene: stray root artifacts, __pycache__ in the tree, vendor pyc pollution vs "read-only" claim, not a git repo
* **Evidence:** root files: `_wtest.log` (contents: "hello"), `_collect_counts.py`/`.json` (stale,
  Q-004), `_docs_full_suite.log`, `_docs_full_suite_agent.log`, `_docs_pytest_run.log`,
  `_docs_pytest_run_46.log` (0 bytes), `_final_audit_pytest.log`, `_final_verification_pytest.log`,
  `.pytest_cache/`; **17 `__pycache__` directories outside vendor/** (incl. under tests/ and jevx/);
  **1071 `__pycache__` directories inside vendor/** (bytecode written into the "read-only clone"
  that docs/TRANSFORMERS_INTEGRATION.md §3 calls unmodified — new files, not edits, but it
  contradicts the spirit of the claim and pollutes a pinned tree); directory is **not a git repo**
  (no .git — reproducibility cites file revisions, not commits). vendor tree itself is **~128 MB**
  (not multi-GB).
* **Impact:** noise for any packaging/CI wave; stale logs invite quoting wrong numbers; read-only
  claim needs a footnote.
* **Recommended fix:** delete the stray logs/pyc (after the fix wave re-runs), add .gitignore
  equivalents if a repo is initialized, and footnote "read-only = no upstream file edited;
  interpreter caches may exist".
* **Owner:** packaging-agent.

### Q-028 (P2) — Reload silently loses fitted calibration when the hub sidecar fetch fails
* **Evidence:** `modeling_jevx.py:1076-1083` — `_restore_calibration_sidecar` wraps the
  `cached_file(...)` hub lookup in `except Exception: calibration_file = None` with no warning;
  a network hiccup on a hub reload therefore leaves the calibrator **unfitted** while
  `calibration_method` still reads e.g. "temperature" — `calibrate` then silently no-ops
  (`is_fitted False`) and decisions change without any signal (local-dir reloads are safe).
* **Impact:** silent behavior change on hub reloads; contradicts "no silent fallback".
* **Recommended fix:** emit a `warnings.warn` (or raise with `allow_missing_calibration=` opt-out)
  when `calibration_method != "none"` but no sidecar was found/loaded.
* **Owner:** fix-agent.

### Q-029 (P2) — VectorCalibrator.fit mutates the global RNG stream
* **Evidence:** `calibration/vector.py:76` calls `torch.manual_seed(seed)` inside fit (LBFGS is
  deterministic — the seed is defensive) — this re-seeds the **caller's** global generator as a
  side effect (same class of issue as the attention_pool RNG bug already fixed in pooling).
* **Impact:** fitting a calibrator changes downstream global RNG draws (data-loader shuffles,
  stochastic layers) — surprising and undocumented.
* **Recommended fix:** drop the `torch.manual_seed` (LBFGS RNG-free, comment says so) or scope it
  with `torch.random.fork_rng`.
* **Owner:** fix-agent.

### Q-030 (P2) — Runtimes quoted across docs are stale three ways
* **Evidence:** README "768 passed … 264.90s (0:04:24)" and "full suite ~4 min"; docs/TESTING.md §6
  "233–265 s across three recorded runs"; FINAL_REPORT/FINAL_VERIFICATION "313.55s (0:05:13)";
  `_final_audit_pytest.log` "316.96s". With the Qwen battery now collected (Q-004) the real full
  suite is far longer (its own report: single 20-row bf16 forward 423.7 s, save 42.8 s).
* **Impact:** "how long do tests take" is wrong everywhere; users may skip the suite.
* **Recommended fix:** after the Q-004 re-run, state one measured number per doc with its date.
* **Owner:** fix-agent.

### Q-031 (P2) — TESTING.md "unit is network-free except test_adapter_real_models" is not exact
* **Evidence:** `tests/unit/test_padding_determinism.py:57` falls back to
  `AutoModel.from_pretrained(GPT2_FALLBACK_ID)` (hub) on load failure — a cold cache makes part of
  "network-free" unit tests hit the network (as does test_benchmark_smoke? no — that one is
  explicitly network-free; the padding file is the exception).
* **Impact:** minor doc inaccuracy; surprise failures offline.
* **Recommended fix:** name the two network-touching unit files in TESTING.md §1/§2.
* **Owner:** fix-agent.

### Q-032 (P2) — API.md §15 "Every documented raise site is asserted by tests" is overbroad
* **Evidence:** `docs/API.md:691` makes the universal claim; counterexamples found:
  the ScoreHead convex-hull `RuntimeError` guard (`decisions/score.py:105`) has no test that forces
  it (grep "convex hull" in tests → none; the guard is defensive-by-construction), and the abstract
  `NotImplementedError` sites (listed in the same table) are programming-error paths.
  (The unfitted-isotonic RuntimeError IS tested — `test_calibration.py:142`.)
* **Impact:** small honesty overreach in the API reference.
* **Recommended fix:** reword to "every documented raise site is asserted except the defensive
  ScoreHead convex-hull guard and abstract-base NotImplementedError (listed)" — or add a forced
  RuntimeError test by monkeypatching probabilities.
* **Owner:** fix-agent.

### Q-033 (P2) — E_Q default (hashed bag-of-tokens) differs from the spec's stated default (shared backbone)
* **Evidence:** spec `MATHEMATICAL_SPECIFICATION.md:212`: "θ_q may equal θ (shared backbone —
  [DESIGN] default)"; implementation default is the hashed projection path —
  `modeling_jevx.py:474` builds `QuestionEncoder(hidden_size=...)` with `method="projection"` and
  `modeling_jevx.py:19-21` documents "the question path uses the hashed bag-of-tokens E_Q by default
  and never invokes the shared backbone". The backbone path exists (QUESTION_METHODS) but is not the
  default and is not reachable from the factory without manual construction.
* **Impact:** spec states one default, code ships another (both documented on their own side, never
  reconciled) — plus EXP-4 measured zero question-conditioning gain on SST-2 partly *because* of
  this design.
* **Recommended fix:** one sentence in the spec (or code): "JevX's implemented default is the
  hashed E_Q; the shared-backbone E_Q is available via QuestionEncoder(method='backbone')".
* **Owner:** fix-agent.

### Q-034 (P2) — Option.key typed `str` but accepts numbers
* **Evidence:** `questions/schema.py:30 key: str` vs `modeling_jevx.py:365-367` coercing
  int/float options via `Option(key=option, label=str(option))` (documented as ordinal levels);
  `project_to_schema` then returns non-str keys.
* **Impact:** type hints on a public dataclass are wrong for a documented code path; type-checkers
  will flag legal usage.
* **Recommended fix:** `key: str | int | float` (or a TypeVar) + docstring note.
* **Owner:** fix-agent.

### Q-035 (P2) — Energy confidence sign convention differs from spec 9.4 (documented, but unreconciled)
* **Evidence:** spec 9.4 defines `c_eng(z) = −E = T_e·logsumexp(z/T_e)` (larger = more confident);
  code returns **raw E = −logsumexp(logits)** before fit (more negative = more confident) and only
  the affine (hi−E)/(hi−lo) after fit — `confidence/base.py:238-292`. Both sides document their
  sign; the spec's c_eng and the code's pre-fit value are negatives of each other.
* **Impact:** a reader comparing spec (9.4) with code output gets opposite orderings.
* **Recommended fix:** note in spec 9.4 "(JevX returns E pre-fit and −E-mapped values post-fit —
  see docs/CONFIDENCE.md)" or flip the pre-fit return; prefer documenting over flipping (tests pin it).
* **Owner:** fix-agent.

### Q-036 (P2) — Minor static-health residues
* **Evidence:** three public decorator factories lack return annotations
  (`register_calibrator` calibration/base.py:54, `register_confidence_method` confidence/base.py:41,
  `register_decision_head` decisions/generic.py:57) — the only public defs without `->` in jevx/;
  `ChoiceHead.build_output` (decisions/choice.py:79-81) is a trivial pass-through override (dead
  weight); repo-wide scans were otherwise clean: **0 TODO/FIXME/XXX/HACK**, **0 bare
  `except: pass`**, no name-based `if bert / elif llama` branching anywhere in jevx/ (spec §65
  respected — every adapter predicate is capability-introspection with a justification comment),
  all 13 test-side excepts are classified/documented, all 47 test names referenced from markdown
  exist (0 dangling references).
* **Impact:** cosmetic.
* **Recommended fix:** add the three annotations, delete the trivial override.
* **Owner:** fix-agent.

---

## Coverage matrices

### 1. Spec-requirement × status

Status key: IMPLEMENTED / PARTIAL / MISSING / ENVIRONMENT-BLOCKED.

| Requirement (§ = original spec; anchors from repo docs) | Status | Evidence |
|---|---|---|
| §22/§23 parallel batch + functional independence (Def 11, P11.1/P11.2, T7) | **IMPLEMENTED** | one-backbone-forward asserted by hook (test_theorem_verification test_t7b), batched≡looped <1e-5 with measured values, schema-perturbation independence <1e-7 (t7a/t7c), variable-K T7_OPTIONS [2,3,4,2] |
| §24 confidence functionals (Def 9) | **PARTIAL** | 5 methods implemented + measured (EXP-6) — but noul/K=1 rule violates spec 9(e) (**Q-002**); energy sign differs (Q-035) |
| §25 calibration (Def 8, D8.I) | **PARTIAL** | 4 methods, D8.I asserted by tests, but no runtime schema/type check (Q-003) and isotonic×variable-K gap (Q-012) |
| §30 weight-conversion taxonomy | **MISSING at audit start → in-flight** | conversion/ 0 bytes → populated during audit; no tests, not exported from jevx root (**Q-001**) |
| §31 preserve_base_model_behavior | **IMPLEMENTED / tests PARTIAL** | `base_forward` bitwise-tested per checkpoint; `return_base_output=True` works (auditor probe, 0.0 diff) but never positively tested (**Q-010**) |
| §33 custom adapters | **IMPLEMENTED** | `register_adapter` decorator + priority tests (test_user_registered_adapter_wins, test_register_adapter_priority_and_restore) |
| §34 remote code / trust_remote_code | **IMPLEMENTED** | default False (test_trust_remote_code_defaults_false), row 14 no-silent-execution test |
| §35 quantized inference | **ENVIRONMENT-BLOCKED** | row 15: bitsandbytes absent, no test written (documented) |
| §43 save/load | **IMPLEMENTED** | test_serialization_jevx (5), row-16 isotonic sidecar, matrix gates save/load 0.0 on 18 checkpoints |
| §45/§63 factory (wrap/reload ladders) | **IMPLEMENTED** | _wrap_base_model/_peek_config/_resolve_featurizers ladders + test_auto_factory (8) + UNIVERSAL_CONVERSION §1/§5 |
| §64 register_adapter | **IMPLEMENTED** | adapters/registry.py + §3 tests above; exposed at `jevx.register_adapter` |
| §67/§68 test taxonomy | **PARTIAL** | unit/integration/golden/numerical present; property/ + regression/ empty (documented); **no files** for distributed, generation, sharded, quantized (**Q-017**) |
| §70/§71 equivalence vs manual reference | **IMPLEMENTED** | matrix_common manual reference math on 18 checkpoints (max_abs_error 0.0), golden re-derivation 3 models (Q-016) |
| §72 long sequences | **PARTIAL (text only)** | L∈{8,64,512} + boundary + decoder + RoPE-null for bert/gpt2/llama; **no vision/audio long-input case** |
| §73 padding equivalence | **PARTIAL (text only)** | decoder left/right/encoder: worst 7.749e-07 < 1e-5; vision/audio absent; Qwen left-pad measured-not-asserted (documented) |
| §74 determinism | **IMPLEMENTED** | within-process bitwise (incl. forked attention/learned pool); cross-process a documented non-goal |
| §77 supported-claim checklist (7 points) | **IMPLEMENTED (1 gap)** | FAILURE_MODES §5 maps all 7 points; point-1 claim slightly overbroad (Q-032); point-4 honored (rows 11/15) |
| §78 support levels honesty | **CONTRADICTORY DOCS** | two incompatible level scales in force (**Q-007**) |
| §87 reproducibility | **IMPLEMENTED** | REPRODUCIBILITY.md + bench_common machine/seeds/revision stamps (§87 blocks); stale runtimes (Q-030) |
| §90 failure rows (16) | **IMPLEMENTED** | full row→test matrix below; row 15 none by design |
| ValidationEngine runtime assertions (D8.I, (11.3)) | **MISSING** | no code; false claim in PROOFS.md:796 (**Q-003**) |
| Def 5.e cross-attention fusion | **MISSING** | no registry entry; claimed by ARCHITECTURE/DC-7 (**Q-006**) |
| Def 5.0 option-wise shared scoring / variable K | **IMPLEMENTED** | masked option features + option_mask; 7 fusion kinds |
| Def 6/7 typed heads for the generic types | **PARTIAL (declared scope)** | choice/score/noul shipped; multi_label + regression experimental; **Ordinal (CORAL 7.5), Ranking (PL 7.7), Threshold, Abstention-as-type not implemented** (abstention exists only as Def-10 overlay, which the spec allows); no doc flags the omitted generic types as out-of-scope — add one sentence to API.md §10 |
| Def 10 abstention | **PARTIAL** | works for choice/score; **no-op for noul (Q-002)**; κ-selection protocol documented, not tested multi-threshold |
| ECE (8.5) binning default | **DIVERGENT** | spec B=10 vs code/reports 15 (**Q-021**) |
| PAVA isotonic (8.3) | **PARTIAL** | PAVA fit correct; linear vs piecewise-constant apply (**Q-023**) |
| D1.1/D3.1/D5.1 axioms | **IMPLEMENTED** | all-masked rows raise, K=0 rejected, fusion purity (no RNG/ in-place) — property tests |

### 2. Failure-row × test matrix (§90, all 16 rows verified present in reports/FAILURE_ANALYSIS.md)

| Row | Scenario | Test | State |
|---:|---|---|---|
| 1 | empty questions | test_failure_modes::test_empty_questions_list_raises_value_error | PASS |
| 2 | empty options (choice) | ::test_empty_options_list_for_choice_raises_value_error | PASS |
| 3 | K=1 choice | ::test_single_option_choice_probs_are_exactly_one | PASS |
| 4 | duplicate options | ::test_duplicate_options_allowed_invariants_hold | PASS |
| 5 | 5000-char description | ::test_very_long_option_description_works | PASS |
| 6 | long state vs position table | ::test_long_state_exceeding_position_limit_raises_descriptive_error (+ test_numerical_edge ×4) | BUG FOUND→FIXED |
| 7 | mask batch mismatch | ::test_mismatched_attention_mask_batch_raises_naming_both | PASS |
| 8 | invalid mask values | ::test_negative_... + ::test_float_attention_mask... + ::test_bool_and_zero_one_int... + ::test_float_option_mask_in_fusion... | BUG FOUND→FIXED |
| 9 | NaN pre-encoded questions | ::test_nan_preencoded_questions_raise_check_finite_with_tensor_name | PASS |
| 10 | int dtype where float | ::test_long_dtype_preencoded_questions_raise_type_error | PASS |
| 11 | wrong device | ::test_cross_device_input_raises_device_error | SKIP (no 2nd device) |
| 12 | logits-only model | ::test_logits_only_model_raises_structured_unsupported_representation | PASS |
| 13 | vision state w/o processor | ::test_vision_state_without_processor_raises_clear_error | PASS |
| 14 | remote code refused | ::test_remote_code_fixture_refused_without_trust_remote_code | PASS |
| 15 | quantized (bitsandbytes) | **none** | ENVIRONMENT-BLOCKED (documented) |
| 16 | fitted isotonic save/load | ::test_fitted_isotonic_sidecar_roundtrip_after_save_load | PASS |
| (extra) | cryptic upstream vision-kwarg error (Q-013) | recorded as *expected* in test_qwen35_hybrid::test_09 | **OPEN — not a JevX error yet** |

### 3. Public-API symbol × has-test

Top-level `jevx/__init__` `__all__` (17 symbols + version):

| Symbol | Direct test ref |
|---|---|
| JevX, JevXForDecision, JevXConfig, JevXOutput, DecisionOutput, Option, CalibrationEngine, ConfidenceEngine, JevXCapabilityDetector, UnsupportedRepresentationError, build_decision_head, build_fusion, pool, register_adapter, register_decision_head, risk_coverage_curve, coverage_at_risk | **YES** (all) |
| `__version__` (and cross-module version equality) | **NO** |

Subpackage exports without any test reference (**Q-011**): `return_base_output` (forward kw),
`DecisionDataset`, `confusion_matrix`, `extract_logits`, `l2_regularization`,
`register_calibrator`, `build_calibrator`, `validate_fit_inputs`, `IdentityCalibrator`,
`is_text_state`, `tokenize_texts`, `observed_output_fields`, `NATIVE_REPRESENTATION_FIELDS`,
`BUILTIN_ADAPTER_CHAIN`, plus the config-pin path `config.decision_type` (Q-010).
(Exercised indirectly but never named: FusionOutput, EnergyConfidence/LearnedConfidence classes,
encoder-decoder `mode="decoder_state"/"encoder_decoder_fusion"` — the last two ARE directly tested
in test_adapters.py:168,178.)

### 4. 18 executed models × golden / matrix coverage

| # | Checkpoint (family) | Golden fixture | Integration matrix | Notes |
|---:|---|---|---|---|
| 1 | tiny-random-bert | **YES** | test_matrix_text | also E2E |
| 2 | tiny-random-t5 | **YES** | test_matrix_text | also E2E |
| 3 | sshleifer/tiny-gpt2 | **YES** | test_matrix_text | also E2E |
| 4 | tiny-random-RobertaModel | no | test_matrix_text | |
| 5 | tiny-random-DistilBertModel | no | test_matrix_text | |
| 6 | tiny-random-AlbertModel | no | test_matrix_text | |
| 7 | tiny-random-DebertaV2Model | no | test_matrix_text | |
| 8 | all-MiniLM-L6-v2 | no | test_matrix_text | only real trained ckpt (also example 04) |
| 9 | tiny-random-BartForConditionalGeneration | no | test_matrix_text | |
| 10 | yujiepan/qwen2-tiny-random | no | test_matrix_text | |
| 11 | tiny-random-LlamaForCausalLM | no | test_matrix_text | |
| 12 | tiny-random-ViTModel | no | test_matrix_vision_audio | (example 05) |
| 13 | tiny-random-CLIPModel | no | test_matrix_vision_audio | (example 07) |
| 14 | tiny-random-SiglipModel | no | test_matrix_vision_audio | |
| 15 | tiny-random-Wav2Vec2Model | no | test_matrix_vision_audio | (example 06) |
| 16 | tiny-random-ASTModel | no | test_matrix_vision_audio | |
| 17 | tiny-random-WhisperForConditionalGeneration | no | test_matrix_vision_audio | |
| 18 | tiny-random-llava-next | no | test_matrix_vision_audio | level 2, experimental |
| — | Qwen3.5-0.8B (hybrid) | no | test_qwen35_hybrid (11 tests) | **hardcoded path (Q-005); excluded from documented counts (Q-004)** |

Golden = 3/18 (**Q-016**); every non-golden model still has live matrix gates (§92 protocol).

---

## Re-readings

### A. Math spec ↔ code (every equation with an implementation counterpart)

| Spec item | Code site | Verdict |
|---|---|---|
| (6.2) stable softmax, max-shift | `utils/numerical.py:51-68, 88-140` (F.softmax internal shift; masked via finfo.min; forced exact 0) | **MATCH** |
| (6.4) stable sigmoid + T4 interior | `decisions/noul.py:57-66` (torch.sigmoid + [eps,1−eps] clamp, sanctioned by PROOFS Rem. 4.1) | **MATCH** (clamp bias ≤ eps, documented) |
| (7.1) choice argmax min-index + y∈C | `decisions/generic.py:411-429, 138-181` (tensor argmax = first max; project_to_schema validates) | **MATCH** (T2: 0 escapes/300) |
| (7.2) score expectation s=Σk·p_k, box [0,K−1] | `decisions/score.py:96-109` (assert-not-clamp, tol 1e-4) | **MATCH** (T3: 0 violations) |
| (7.9) Cal strictly between G and Π | `modeling_jevx.py:1634-1644` applies Cal to **logits before** G | **EQUIVALENT-BY-DESIGN** — documented (module docstring steps 6); spec 8.1 itself gives both logits/probs forms; isotonic returns a log p′ representative so G reproduces calibrate_probs |
| (8.1) temperature, argmax preserved | `calibration/temperature.py` (deterministic 400-pt log-grid + 6 refinements vs spec's golden-section — same argmin) | **MATCH** |
| (8.2) vector scaling a⊙z+b | `calibration/vector.py` (LBFGS; argmax caveat documented) | **MATCH** |
| (8.3) isotonic PAVA, **piecewise constant** | `calibration/isotonic.py:50-86` (PAVA fit) + `:144-157` (linear interpolation apply) | **DIVERGENT** (Q-023) |
| (8.5) ECE formula, equal-width bins, **default B=10** | `utils/numerical.py:234-280` formula matches; default 15 in code | formula **MATCH**, default **DIVERGENT** (Q-021) |
| (9.1) max-prob; K=1 → 1 | `confidence/base.py:162-174 + 424-430` | MATCH for genuine K=1 choice; **DIVERGENT for noul (Q-002)** |
| (9.2) margin top1−top2, K=1≡1 | `confidence/base.py:84-90` | **MATCH** |
| (9.3) normalized entropy 1−H/lnK | `confidence/base.py:93-116` (float64, K_eff per row, clamped) | **MATCH** (K_eff extension documented) |
| (9.4) energy c_eng = −E | `confidence/base.py:238-292` returns E pre-fit, affine post-fit | **SIGN DIVERGENT, documented** (Q-035) |
| 9(e) noul c_max = max(p,1−p) | confidence engine returns 1 | **DIVERGENT (Q-002)** — engine is the outlier; ECE/NLL paths agree with spec |
| (10.1)-(10.4) abstention + risk-coverage | `modeling_jevx.py:216-282, 1654-1660` | **MATCH** (stable sort, (i+1)/n, sup via max-feasible; empty-set handled) — noul leg dead (Q-002) |
| (11.2) masked softmax per question | `utils/numerical.py:88-140` | **MATCH** (exact zeros, per-row normalization) |
| (11.3) never normalize across questions | same (row-wise) + property tests | **MATCH** (by construction; no runtime assert — Q-003) |
| (4a-e) pooling operators | `representations/pooling.py` (masked_mean/max/cls/last_valid/attention/learned/native + extra unmasked "mean") | **MATCH** + one documented extra ("mean" is not in the spec's list) |
| (5.b) gated fusion incl. s_C, φ_C(s_C) | `fusion.py GatedFusion` — gate on [s;q] only, no φ_C term | **DIVERGENT (documented partially) — Q-022** |
| (5.a) concat 3d with s_C | `ConcatFusion` — 2d | **DIVERGENT (Q-022)** |
| (5.c) additive LN + 1/√3 | `AdditiveFusion: s + W_q q` | **DIVERGENT (Q-022)** |
| (5.d) bilinear on (H_C,h_Q)/(H_C,h_S) | diagonal context + head-level bilinear interaction | **DIVERGENT (documented as β_diag, Q-022)** |
| (5.e) cross-attention fusion | — | **MISSING (Q-006)** |
| e_k = g⊙W_o o_k + (1−g)⊙c | `fusion.py:11-14, 113-127` | **MATCH** (home is API.md §9, not the math spec) |
| Def 11 shared H_S once | hook-counted test t7b + capability_metadata.backbone_forward_calls=1 | **MATCH** |
| D8.I calibration preserves shape/dtype/K | calibrators are [N,K]→[N,K]; test_calibration schema invariant | **MATCH by construction + tests; runtime check missing (Q-003)** |
| Spec E_Q "shared backbone — DESIGN default" | hashed projection default | **DIVERGENT default (Q-033)** |
| Def 6/7 generic types (Ordinal/Ranking/Threshold) | not implemented | **MISSING, scope not stated in docs** (see spec matrix row) |

### B. Transformers-convention drift (10 audit claims spot-checked against vendor + JevX)

| Audit claim (v5.16.1) | Vendor check | JevX conformance | Verdict |
|---|---|---|---|
| 1. Config field `dtype` canonical, `torch_dtype` deprecated property + BC kwarg (§1.1/§9) | verified `configuration_utils.py:269, 278-289, 461-474` | JevXConfig redeclares neither (inherits); probe: `JevXConfig(dtype=…)` and `(torch_dtype=…)` both OK; loader merges both spellings (modeling_jevx:853-854) | **NO DRIFT** |
| 2. `_tied_weights_keys` is dict{target:source} (§9 table) | verified `modeling_utils.py:1226-1228` | JevX declares none; relies on post_init aggregation; `test_tied_causal_lm_roundtrip` | **NO DRIFT** |
| 3. `return_dict=False` still honored (Q2) | `utils/generic.py can_return_tuple` | JevX implements the same semantics manually (modeling_jevx:1699-1730); tested (test_v5_tuple_mode_via_can_return_tuple) | **NO DRIFT** (different mechanism, same contract) |
| 4. AutoConfig.register idempotent, process-local (Q3) | `configuration_auto.py:439-455` | try/except ValueError in auto/modeling_auto.py:30-35; split-brain documented; reload tested without registration | **NO DRIFT** |
| 5. Embedding accessors must be overridden (risk #2) | probe-backed in audit | `get/set_input_embeddings` delegate (modeling_jevx:624-628) | **NO DRIFT** |
| 6. `post_init()` last in `__init__` | `modeling_utils.py:1376` | called at modeling_jevx:541 after all submodules; pre-loaded backbone marked `_is_hf_initialized` | **NO DRIFT** |
| 7. pass `use_cache=False` incl. kwargs-only forwards (§5/§8.7) | `masking/cache` docs | fixed in adapters/base.py:258-278 **but not** in encoder_decoder._run_full | **PARTIAL DRIFT (Q-014)** |
| 8. Windows trust_remote_code explicit-only (§2) | `dynamic_module_utils.py:774-779` | default False everywhere; row 14 + dedicated test | **NO DRIFT** |
| 9. output_attentions forces eager (§8.6) | `configuration_utils.py:406-415` | flag passed through; constraint documented in forward docstring | **NO DRIFT** |
| 10. ProcessorMixin unified call / AST has no processor entry (Q8) | `processing_utils.py:601,656`; auto_mappings | _call_processor + _resolve_featurizers ladder; AST row executed green | **NO DRIFT** |

---

## Known findings (previous waves) — verified status

| # | Finding | Status | Verification |
|---:|---|---|---|
| 1 | Capability detector layer_types-blind (no hybrid linear/full awareness) | **OPEN** | grep layer_types in jevx/ → 0 hits; QWEN35 report:65 "scope, not a defect"; test_qwen35_hybrid:374-381 **asserts the absence** (capability_mentions_layer_types false). Recommendation: record `config.layer_types/attention patterns` in CapabilityReport.reasons (informational only) so hybrid wraps are self-describing |
| 2 | Qwen3.5 wrap resolves MultimodalAdapter for text-only use | **OPEN — selection-quality, not correctness** | Battery green (save/load 0.0, decisions stable); text path works because Qwen3VLProcessor carries Qwen2Tokenizer (report:73). Real effects: (a) `default_pooling="masked_mean"` instead of the decoder-pinned `last_valid` (representation semantics differ from the matrix's decoder convention); (b) DecoderAdapter's cumsum position_ids left-pad derivation is **not** active (report 6c: differences stayed at bf16 noise floor *for this checkpoint* — GDN has no absolute positions — but the adapter choice would not be benign for absolute-position vision+text causal wrappers); (c) verdict `jevx_conversion="experimental"` even for text-only use. Fix if desired: chain order or a MultimodalAdapter subclass that keeps decoder pooling when state inputs are text-only |
| 3 | pixel_values-only / input_ids+pixel_values cryptic upstream errors | **OPEN** | Q-013 (pixel_values-only at least raises a real upstream ValueError naming input_ids/inputs_embeds; the input_ids+pixel_values combo is the cryptic one: AttributeError from adapters/base.py:282) |
| 4 | No packaging (pyproject) | **OPEN, concurrent** | no pyproject/setup at root at audit end; README documents the gap; FINAL_VERIFICATION marks NOT PRODUCED — packaging-agent owns |
| 5 | Learned confidence never measured (EXP-6) | **OPEN** | Q-019 (honestly recorded as not-run in ABLATIONS/CONFIDENCE/FINAL_VERIFICATION) |
| 6 | Single-dataset quality (SST-2 only) | **OPEN** | Q-020 (FINAL_VERIFICATION PARTIAL, "no second dataset run exists") |
| 7 | Level 4 never claimed | **CLOSED / correct behavior** | matrix demoted all former 4s → 3 on execution; SUPPORTED_MODELS §3, FINAL_REPORT, FINAL_VERIFICATION all agree; nothing claims 4 today (but see Q-007 for the scale conflict) |
| 8 | REAL_MODEL_VALIDATION / matrix stale "no benchmarks" notes | **OPEN** | Q-008: RMV:68, PROOFS:797, MATRIX:22,92 all still say no results committed |
| 9 | Examples not re-executed in final audit pass | **OPEN (partially mitigated)** | FINAL_VERIFICATION Implementation row: "executed Phases 46/81 — not re-run in this pass". This audit re-ran example_01 → green (all invariants held); the other six remain un-re-run since their last recorded pass |
| 10 | noul eps-clamp vs Theorem T4 | **CLOSED / verified intact** | test_theorem_verification.py:363-399 asserts strict interiority at z∈{0,±1e3,±1e4,±1e6} on both `build_output` and the full forward path; probe: p strictly interior; clamp documented as the sanctioned implementation invariant. **No breakage** |
| 11 | quantized / cross-device rows environment-blocked | **OPEN (correctly declared)** | row 11 = the suite's only skip (device-name guarded); row 15 = no test, declared untested; README/LIMITATIONS/FAILURE_MODES all say "untested, never passing" |

---

## Top-10 fix order

1. **Q-002** — noul confidence → max(p,1−p) + test/docs (restores Def-10 abstention for noul).
2. **Q-005** — de-hardcode the Qwen snapshot path / add module-level skip (suite must run off this host).
3. **Q-004** — re-run the full 780-test suite once; refresh every count/runtime claim (README, TESTING, FINAL_REPORT, FINAL_VERIFICATION, delete _collect_counts.*).
4. **Q-001** — finish conversion subsystem: export from package root, add tests + API.md section (coordinate with conversion-agent already in flight).
5. **Q-010 + Q-011** — add the missing tests (return_base_output, config.decision_type pin round-trip, public-surface sweep incl. version triple).
6. **Q-003 + Q-009** — resolve ValidationEngine: implement minimal runtime checks or rewrite PROOFS:796 / ARCHITECTURE §2 / LIMITATIONS:170 to "by construction + tests".
7. **Q-013 + Q-014** — wrap backbone-forward failures with JevX context; unify the use_cache gate (one helper, both adapters).
8. **Q-012** — calibration × option_mask: mask/slice before calibrating + variable-K isotonic regression test.
9. **Q-007 + Q-008 + Q-006 + Q-015 + Q-016 (docs sweep)** — one pass unifying level scales, deleting stale "no benchmark" lines, fixing 5.e claims, correcting FINAL_VERIFICATION counts/PNG, extending golden fixtures.
10. **Q-018 + Q-027** — enterprise metadata (LICENSE, CI, pyproject via packaging-agent, single-source version) and repo-hygiene sweep (stray logs, pyc, vendor-cache footnote).

*End of audit. No code, tests, or docs were modified; this file is the only artifact created.*

---

## Status Addendum — 2026-09-25 (final verification pass)

Append-only: the 36 findings and the known-findings table above are the
audit's historical record and are left untouched; this section records each
item's status after the hardening wave, verified against the tree on
2026-09-25. Evidence: final full suite `python -m pytest -q` →
**`883 passed, 1 skipped, 2 warnings in 2393.61s (0:39:53)`** (884 collected;
`_final_full_suite.log`), `import jevx` → `0.1.0`,
`importlib.metadata.version('jevx')` → `0.1.0`. Status key: **CLOSED**
(fix landed + evidence cited) / **OPEN** (still true as audited, why given) /
**ENVIRONMENT-BLOCKED** (needs hardware the host does not have).

### P0 (2 findings) — 2 closed, 0 open

| # | Status | Evidence / why |
|---|---|---|
| Q-001 | **CLOSED** | `jevx/conversion/` populated (`converter/equivalence/report/transplant.py`), re-exported from `jevx/__init__` (`ConversionMode`, `ConversionReport`, `EquivalenceKind`, `convert`), 21 tests in `tests/unit/test_conversion.py`, grounding in ARCHITECTURE §5, and an API section added this pass (docs/API.md §16, signatures read from `inspect.signature`). |
| Q-002 | **CLOSED** | `jevx/confidence/base.py:447-453` implements `max(p, 1−p)` for the binary single column (genuine K=1 choice rows still read 1); `tests/unit/test_noul_confidence_fix.py` (4 tests incl. a threshold that fires); `docs/CONFIDENCE.md` §1 documents the rule. |

### P1 (18 findings, Q-003…Q-020) — 18 closed, 0 open

| # | Status | Evidence / why |
|---|---|---|
| Q-003 | **CLOSED** | `jevx/validation.py` `ValidationEngine` called from `JevXForDecision.forward` behind `JevXConfig.runtime_validation` (default `True`); `tests/unit/test_validation_engine.py` (12 tests: on-path violations raise, flag-off bitwise-identical); guarded by `test_public_surface.py::test_validation_engine_contract`. PROOFS.md wording corrected this pass (two stale "absent/until then" clauses removed — taxonomy row now describes the checks actually run: simplex bound, shape/K preservation, noul interiority, confidence range, index range). |
| Q-004 | **CLOSED** | All counts refreshed to this run: `docs/TESTING.md` §1 (884 collected = unit 799 + integration 37 + golden 6 + numerical 42, per-file table rewritten), README status row, `docs/REPRODUCIBILITY.md` §5; stale `_collect_counts.*` confirmed absent. |
| Q-005 | **CLOSED** | `tests/integration/test_qwen35_hybrid.py:93-145` `_resolve_snapshot()` — `snapshot_download(local_files_only=True)` → cache-walk fallback → classified `pytest.skip` when absent (no hardcoded host path). |
| Q-006 | **CLOSED** | `cross_attention` registered in `FUSION_REGISTRY` (`jevx/representations/fusion.py`), spec Def 5.e now marked "implemented — `fusion_type="cross_attention"`", ARCHITECTURE DecisionFusion row and RESEARCH DC-7 updated to "five variants registered"; `tests/unit/test_cross_attention_fusion.py` (12 tests). |
| Q-007 | **CLOSED** | ARCHITECTURE §6 now states the matrix's "Support levels (defined precisely)" table is the *normative scale, one source of truth*; LIMITATIONS §7 re-states it. One scale, one meaning. |
| Q-008 | **CLOSED** | All four sites rewritten: PROOFS taxonomy row ("Scoped claims exist only where the benchmark suite ran… **No per-family quality benchmark exists for the matrix's cited checkpoints — hence no level 4**"), REAL_MODEL_VALIDATION quality-pointer table, matrix evidence overlay + level-distribution line (both extended this pass to name all four datasets — none of them on a matrix-cited checkpoint, so the no-level-4 conclusion is unchanged). |
| Q-009 | **CLOSED** | `docs/LIMITATIONS.md` now says "what the runtime guards (the ValidationEngine component, `jevx/validation.py`, …) and the test suite are for" — no "(not yet written)" remains. |
| Q-010 | **CLOSED** | `tests/unit/test_public_surface.py` tests `return_base_output=True` (populates, matches raw backbone), `config.decision_type` pin resolving a forward, and explicit-argument override of the pin. |
| Q-011 | **CLOSED** | `tests/unit/test_public_surface.py` (15 tests): every `jevx.__all__` name imported + documented, version triple, plus behavioral smokes for `DecisionDataset`, `confusion_matrix`, `extract_logits`, `l2_regularization`, the calibrator registry, and adapter text helpers. |
| Q-012 | **CLOSED** | `CalibrationEngine.apply_valid_columns` calibrates valid columns only and re-pads (`modeling_jevx.py` calibration call site); `tests/unit/test_calibration_masking.py` (8 tests: stuffed masked columns are no-ops, contrast case, simplex, noul unaffected). |
| Q-013 | **CLOSED** | `call_backbone()` in `jevx/adapters/base.py` wraps input-contract failures into an actionable `ValueError` (original kept as `__cause__`); encoder-decoder path routes through it too; `tests/unit/test_error_wrapping.py` (9 tests incl. end-to-end through `extract_representation` and the narrow-rule non-wrap cases). The Qwen battery now *records* the wrapped message for `input_ids+pixel_values` (QWEN35 addendum A3) — the unit test owns the assertion. |
| Q-014 | **CLOSED** | Single shared `inject_no_cache()` helper in `jevx/adapters/base.py`, imported and used by `jevx/adapters/encoder_decoder.py`; `tests/unit/test_error_wrapping.py` Q-014 block (kwargs-only/declared/neither forms + "encoder_decoder imports the same single helper" identity assert). |
| Q-015 | **CLOSED** | The four false statements were corrected at the previous gate (PNG claim withdrawn, docs/reports/JSON counts machine-listed); counts grew again since (reports 11→**12**, results JSON 18→**30**, tests 828→**884**) and are re-counted with the counting commands in the Gate Refresh of reports/FINAL_VERIFICATION.md. |
| Q-016 | **CLOSED** | `tests/golden/fixtures/` holds 6 records (bert, roberta, t5, bart, gpt2, vit) + the two modality assets; golden suite (6 tests) green in the final run. |
| Q-017 | **CLOSED** | Sharded round-trip exists (`tests/integration/test_sharded_roundtrip.py`, 3 tests: multi-file save + index → reload → identical decisions); `docs/TESTING.md` §1 "Not covered and why" table enumerates every absent category with its reason (distributed/env-blocked, generation/by-design, quantized/env-blocked, cross-device/1 skip, vision-audio §72–§73/not covered — recorded, not faked). |
| Q-018 | **CLOSED** | `pyproject.toml` (PEP 621) + `LICENSE` (MIT) + `CHANGELOG.md` + `.gitignore` present; `pip install -e .` verified this pass (`pip show jevx` → 0.1.0, editable finder installed, foreign-cwd `import jevx` → 0.1.0); `tests/unit/test_version_consistency.py` (5 tests) pins the version triple. Residual: no CI config (was listed *optional* in the recommendation) — noted, not claimed. |
| Q-019 | **CLOSED** | EXP-6b measured the `learned` method paired with the EXP-6 baselines (bit-identical reproduction): ECE-of-conf 0.0910 (2nd best), AURC 0.1983 — `benchmarks/results/exp6b_learned_confidence.json`, `reports/ABLATIONS.md` EXP-6b. |
| Q-020 | **CLOSED** | Quality now spans **4 datasets × 3 backbones**: SST-2 + ag_news × `bert_uncased_L-2_H-128_A-2` (DATASET_BENCHMARKS §8), Imagenette × `deit-tiny-patch16-224`, RAVDESS × `wav2vec2-base-superb-ks` (MODALITY_BENCHMARKS) — all 3 seeds, same train/cal/test discipline; docs/BENCHMARKS §§6–7 index them. |

### P2 (16 findings, Q-021…Q-036) — 10 closed, 6 open

| # | Status | Evidence / why |
|---|---|---|
| Q-021 | **CLOSED** | Spec §8.5 now reads "default: B = 15 equal-width bins … the shipped default" — spec matches code and every reported number. |
| Q-022 | **CLOSED** | ARCHITECTURE DecisionFusion row already carried the `s_C`-summand note; the `jevx/representations/fusion.py` module docstring gained the matching sentence this pass ("the `s_C` schema-summary summands of 5.a–5.d are carried by the per-option formula — no separate `φ_C(s_C)` term"), closing the docstring half of the recommendation. |
| Q-023 | **CLOSED** | Spec §8.3 now documents the implementation explicitly: "piecewise-linear interpolation between the PAVA breakpoints (clamped outside the fitted range), not the piecewise-constant step function this equation's minimizer takes". |
| Q-024 | **OPEN** | `jevx/configuration_jevx.py:54-55` still documents `("mean", "cls", "last" …)` — `"last"` is not a registered strategy (registry has `last_valid`) — and still says "None lets the base model decide" (actual: adapter default). The dead `representation_strategy` knob is still described as "owned by the representations/adapters modules" in the same docstring (it is metadata-only; API.md and the module docstring say so, the config docstring does not). |
| Q-025 | **OPEN** | `jevx/modeling_outputs.py:88-90` still says `state_representation` is the "Fused state H … None unless requested" (code always populates it with the projected `H_S`, never None); `modeling_jevx.py`'s module docstring still claims the per-row fused H "lives inside each decision's provenance via `capability_metadata`" — the metadata dict contains no fused vectors. API.md documents both fields correctly. |
| Q-026 | **CLOSED** | `docs/SUPPORTED_MODELS.md` table separator row is now a clean `\|---\|---\|…` — the flagship table renders. |
| Q-027 | **CLOSED** | Hygiene sweep this pass: 8 stale root run logs (`_fixA_*`, `_fixpassb_*`, `_conversion_*`, `_exp6b_*`, `_exp7_*`) + the stray `nul` file + the `_logs/` scratch directory deleted after grep confirmed nothing references them; 18 non-vendor `__pycache__` + `.pytest_cache` swept (vendor `__pycache__` count = 0; `vendor/reports/*.log` kept as evidence); `.gitignore` covers logs/caches/egg-info. The "not a git repo" fact is unchanged and documented. |
| Q-028 | **OPEN** | `_restore_calibration_sidecar` (`jevx/modeling_jevx.py:1087-1103`) still swallows a failed hub sidecar fetch silently (`calibration_file = None`, no warning) — a reload with `calibration_method != "none"` can still end unfitted with no signal. |
| Q-029 | **OPEN** | `jevx/calibration/vector.py:69` still calls `torch.manual_seed(seed)` inside `fit()` — LBFGS is RNG-free but the reseed still mutates the caller's global RNG stream as a side effect. |
| Q-030 | **CLOSED** | Runtimes refreshed this pass with one measured number + date everywhere: README status/tests rows, `docs/TESTING.md` §2+§6, `docs/REPRODUCIBILITY.md` §5 (**883 passed … 2393.61s (0:39:53), 2026-09-25**; historical 768/233.31s runs kept as labeled history), plus FINAL_REPORT and FINAL_VERIFICATION below. |
| Q-031 | **CLOSED** | `docs/TESTING.md` §1 table and §2 fast-subset row now name both network-touching unit files: `test_adapter_real_models.py` (Hub checkpoints) and `test_padding_determinism.py` (GPT2 fallback on load failure, flagged "audit Q-031"). |
| Q-032 | **CLOSED** | `docs/API.md` §15's universal claim was reworded this pass (the recommended fix): every documented raise site is asserted **except three defensive by-construction paths** — the ScoreHead convex-hull guard, the `resolve_adapter` unreachable fallback, and the abstract-base `NotImplementedError`s — each named in place. |
| Q-033 | **CLOSED** | Spec Definition 4 now carries "Implemented default [FACT]: JevX instantiates `E_Q` as the hashed bag-of-tokens projection … (`QuestionEncoder(method="projection")`, the constructor default)" with the T7 rationale and the `method="backbone"` alternative. |
| Q-034 | **OPEN** | `jevx/questions/schema.py:51` still declares `key: str` while `modeling_jevx.py:367` legally builds `Option(key=option, …)` from `int`/`float` ordinal options — the public dataclass type hint remains wrong for a documented path. |
| Q-035 | **CLOSED** | Spec §9.4 now carries the "Implementation sign note [FACT]" — pre-fit raw `E = −logsumexp(z)`, post-fit affine `(hi−E)/(hi−lo)` — reconciling spec and `CONFIDENCE.md`. |
| Q-036 | **OPEN** | The three public decorator factories (`register_calibrator`, `register_confidence_method`, `register_decision_head`) still lack `->` return annotations, and the trivial pass-through `ChoiceHead.build_output` override is still present — cosmetic static-health residues only. |

### Known findings (previous waves) — status after this pass

| # | Status | Evidence / why |
|---:|---|---|
| 1 | **CLOSED** | `CapabilityReport.attention_layout` added (`jevx/capabilities/detector.py`, nested `text_config.layer_types` included); `tests/unit/test_layer_types_detection.py` (9 tests); the Qwen battery's `test_01` now **asserts** `attention_layout == "hybrid:18xlinear_attention+6xfull_attention"` plus the `layer_types` reason line (flipped from recording the detector's silence — QWEN35 addendum A1). Verdicts unchanged (informational field only). |
| 2 | **CLOSED** | `MultimodalAdapter` now derives padding-consistent `position_ids = cumsum(mask) − 1` for text-only inputs (`jevx/adapters/multimodal.py`); battery item 6c left-vs-right re-measured **0.0625** (pre-fix 0.125; backbone-alone 0.0465; bf16 2·ulp floor 0.25) — QWEN35 addendum A2. Residuals that remain by design (documented, not defects): adapter pooling stays `masked_mean` for this wrapper, and the text-only verdict stays `experimental` (conservative). |
| 3 | **CLOSED** | Same fix as Q-013: the `input_ids+pixel_values` combo now raises the JevX-wrapped `ValueError` (recorded in the battery JSON, original `AttributeError` kept as `__cause__`); the `pixel_values`-only case remains the actionable upstream `ValueError` naming `input_ids`/`inputs_embeds`. |
| 4 | **CLOSED** | Packaging delivered: `pyproject.toml`, `LICENSE` (MIT), `CHANGELOG.md`, `pip install -e .` verified this pass (editable finder + foreign-cwd import + `pip show jevx` → 0.1.0), version triple guarded by `test_version_consistency.py`. |
| 5 | **CLOSED** | `learned` confidence measured (EXP-6b, paired with EXP-6): see Q-019. |
| 6 | **CLOSED** | Single-dataset scope superseded by 4 datasets × 3 backbones: see Q-020. |
| 7 | **CLOSED / correct behavior** | Re-checked this pass against the matrix's own level-4 gate (quality result **on that family's cited checkpoint**): the four quality runs used `bert_uncased_L-2_H-128_A-2`, `deit-tiny-patch16-224`, `wav2vec2-base-superb-ks` — **none is a matrix-cited checkpoint** (those are the tiny-random fixtures) — so **no cell qualifies for 4**; the matrix still claims none, correctly. Additionally the RAVDESS/JevX result is below the credible bar (majority baseline), so it could not ground a level even on its own backbone. |
| 8 | **CLOSED** | All four stale sites rewritten (see Q-008); this pass also corrected the two matrix parentheticals that had grown stale again after EXP-7/modality (scope now "4 datasets × 3 backbones … none on the checkpoints cited here"). |
| 9 | **CLOSED** | All **seven examples re-executed this pass** (2026-09-25), each printing its invariant line green: `example_01` (probs/index/ES/noul/confidence), `02` (m=4 single forward), `03` (B*m layout), `04` (config pin + bitwise determinism), `05` (vision), `06` (audio + score bounds), `07` (multimodal, one forward). |
| 10 | **CLOSED / verified intact** | The eps-clamp still holds: T4 strict-interiority tests green in the final suite (`test_theorem_verification.py`); noul rows checked by `test_noul_confidence_fix.py` and the runtime validator. |
| 11 | **ENVIRONMENT-BLOCKED** | Unchanged and correctly declared: quantized inference (no `bitsandbytes`, CPU-only torch — FAILURE_ANALYSIS row 15, no fake test), cross-device inputs (row 11 = the suite's single documented skip), CUDA/GPU latency/memory (no device — `torch.cuda.is_available() = False`). |

### Updated tally

| Severity | Findings | Closed | Open | Environment-blocked |
|---|---:|---:|---:|---:|
| P0 | 2 | 2 | 0 | 0 |
| P1 | 18 | 18 | 0 | 0 |
| P2 | 16 | 10 | 6 | 0 |
| **Total (Q-001…Q-036)** | **36** | **30** | **6** | **0** |
| Known #1–#11 | 11 | 10 | 0 | 1 |

Audit summary counts above (P0 2 / P1 18 / P2 16) are unchanged — no findings
were added or merged. The six open items (Q-024, Q-025, Q-028, Q-029, Q-034,
Q-036) are all P2 documentation/annotation/hygiene residues or conservative-
by-design side effects: none affects measured behavior, and each is recorded
above with the exact file:line where the audited text still stands. The one
environment-blocked item (Known #11) stays environment-blocked — never counted
as passing.

---

## Open-Item Closure — 2026-09-25 (open-items closure pass)

Append-only, like the Status Addendum above: the 36 findings, the known-findings
table and the Status Addendum are the audit's historical record and are untouched.
This section records what happened to the **six items the Status Addendum left OPEN**
(Q-024, Q-025, Q-028, Q-029, Q-034, Q-036). Status key: **CLOSED** (fix landed here,
evidence cited) / **HANDED-OFF** (the file is owned by a concurrent agent, so the exact
one-line change is recorded instead of applied) / **STILL-OPEN** (with the reason).

Ownership rule applied: this pass owns `jevx/utils/**`, `jevx/configuration_jevx.py`,
`jevx/modeling_outputs.py`, `jevx/calibration/**`, `jevx/confidence/**`, `docs/**` and the
reports named in its brief. It does **not** own `jevx/modeling_jevx.py`,
`jevx/questions/**` or `jevx/decisions/**`, so findings whose fix lands in those files
are handed off rather than applied.

| # | Status | What changed, or the exact change an owner must make |
|---|---|---|
| Q-024 | **CLOSED** | `jevx/configuration_jevx.py:51-64` — the `representation_strategy` entry no longer claims the knob is "owned by the representations/adapters modules" (it is a metadata-only tag, never read by a computation); the `pooling_method` entry now names only registered strategies, points at `available_pooling_strategies()`, states the registered name is `last_valid` and **not** `last`, and replaces "None lets the base model decide" with the true meaning (**adapter default**: native pooled field when the backbone exposes one, else the adapter `default_pooling`). Docstring-only; no behavior touched. |
| Q-025 | **PARTIALLY CLOSED — 1 of 2 halves, 1 HANDED-OFF** | **Closed:** `jevx/modeling_outputs.py:88-95` — `state_representation` is documented as the shared *projected* state `H_S [B, D]` (not the fused `H = F(H_S, H_Q, H_C)`), always populated by the wrapper, with the fused state explicitly stated to be internal and unexposed; and the class-level note at `:110-114` no longer implies every non-`decisions` field is config-flag-gated. **Handed off** (owner of `jevx/modeling_jevx.py`): `jevx/modeling_jevx.py:78-81` still says the per-row fused `H` "lives inside each decision's provenance via `capability_metadata`". The dict built at `jevx/modeling_jevx.py:1781-1795` contains no fused vectors (`model_type`/`architecture`/`representation_field`/`adapter`/`pooling_method`/`fusion_type`/`decision_types`/`confidence_method`/`calibration`/`num_states`/`questions_per_state`/`pair_ordering`/`backbone_forward_calls`). **Exact one-line change:** replace those four lines with — `state_representation` holds the shared projected `H_S [B, D]`; the per-row fused `H = F(H_S, H_Q, H_C)` is internal to `forward` and is **not** exposed on the output (`capability_metadata` carries no fused vectors — see docs/API.md). |
| Q-028 | **HANDED-OFF** | No edit: the site is `jevx/modeling_jevx.py`, owned by another agent. **Exact change needed**, in `_restore_calibration_sidecar` (`jevx/modeling_jevx.py:1086-1100`): the bare `except Exception: calibration_file = None` around the `cached_file(...)` hub lookup must emit a signal before the `if calibration_file is None: return` early exit — e.g. `warnings.warn(f"jevx calibration sidecar {self.calibration_engine.method!r} not found under {path!r}; the calibrator stays UNFITTED and calibration is a no-op", RuntimeWarning, stacklevel=2)` guarded on `self.calibration_engine.method != "none"`, keeping the `except Exception` (the sidecar is optional) but no longer returning silently. A local-directory reload is unaffected (the `is_file()` branch already resolves). No test change is required beyond one asserting the warning fires on a missing sidecar with a non-`none` method. |
| Q-029 | **CLOSED** | `jevx/calibration/vector.py:72-79` — `fit()` no longer calls bare `torch.manual_seed(seed)`, which re-seeded the *caller's* global generator. The seed is now applied to a **forked** RNG (`with torch.random.fork_rng(devices=[]): torch.default_generator.manual_seed(seed)`), the same contract `jevx/representations/pooling.py:284-285` already uses for the stateless attention pooler. Fitted parameters are unchanged (LBFGS is RNG-free, so the reseed never influenced them); only the side effect is gone. Docstring at `:29-34` and the `seed` arg doc at `:66-68` state the scoping. Evidence: `tests/unit/test_calibration.py` (29) green. |
| Q-034 | **HANDED-OFF** | No edit: the site is `jevx/questions/schema.py`, owned by another agent. **Exact one-line change:** `Option.key` at `jevx/questions/schema.py:51` — change `key: str` to `key: str | int | float`, and extend the `:44` arg doc from "Identity of the option within the schema." to "Identity of the option within the schema. ``str`` for named options; ``int``/``float`` for ordinal levels, which `JevXForDecision._normalize_option` builds as `Option(key=option, label=str(option))` (`jevx/modeling_jevx.py:367-369`) and which `project_to_schema` then returns verbatim — so a non-`str` key is a documented, supported path, not a type error. `Option.text()` already falls back to `str(self.key)`, so no other site changes. |
| Q-036 | **PARTIALLY CLOSED — 2 of 4 residues, 2 HANDED-OFF** | **Closed:** `jevx/calibration/base.py:54` `register_calibrator(name: str) -> Callable[[type[BaseCalibrator]], type[BaseCalibrator]]` and `jevx/confidence/base.py:47` `register_confidence_method(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]` — the two decorator factories this pass owns now carry return annotations, with `Callable` added to each module's `typing` import. **Handed off** (owner of `jevx/decisions/**`): `register_decision_head(name: str)` at `jevx/decisions/generic.py:57` needs `-> Callable[[type[T]], type[T]]` (add `Callable` to that module's `typing` import at line 30), and the trivial pass-through `ChoiceHead.build_output` override at `jevx/decisions/choice.py:80-83` (body is exactly `return super().build_output(logits, mask)`) can be deleted outright — `DecisionHead.build_output` in `jevx/decisions/generic.py` is the real implementation. |

**Tally after this pass.** Of the six items the Status Addendum left open: **2 CLOSED**
(Q-024, Q-029), **2 PARTIALLY CLOSED** with the remainder handed off by file ownership
(Q-025: 1/2 halves; Q-036: 2 of 4 residues), **2 HANDED-OFF whole** (Q-028, Q-034). No item
is STILL-OPEN for a reason other than file ownership; none of the six touches measured
behavior — Q-029 is the only code change with any runtime effect at all, and it removes a
side effect rather than adding one.

**Companion work in the same pass (not an audit finding).** The T7-ii tolerance was
characterized and closed under JOB 2: `reports/NUMERICAL_EQUIVALENCE_STUDY.md` (240 cells,
1644 compared rows, **0 argmax flips**), the rule implemented as
`jevx.utils.numerical.batch_equivalence_tolerance`, `tests/unit/test_batch_equivalence_tolerance.py`
(33 tests), `docs/PROOFS.md` Remark 7.4 and `docs/MATHEMATICAL_SPECIFICATION.md` P11.5.
The previously hard-coded `1e-5` contract that the real scenario failed at 1.0967e-05 is
replaced by a measured, configuration-dependent bound — see that report.

---

## Open-Item Closure — Correction (2026-09-25, handoff wave landed)

Append-only, like every section above: the Status Addendum, the Updated tally and the
first Open-Item Closure table are the audit's historical record and are **not** rewritten
here. The six items the first Open-Item Closure pass left **HANDED-OFF** or partially
handed-off have since had their changes applied by the owning agents. The status of
**Q-034** and **Q-036** in that earlier table is therefore superseded: both are now
**CLOSED**. Evidence is re-verified against the working tree in this pass.

| # | Status now | Verified evidence (file:line, re-read this pass) |
|---|---|---|
| Q-034 | **CLOSED** (was HANDED-OFF) | `Option.key` is now `key: str | int | float` at `jevx/questions/schema.py:57` (the one-line change the first closure pass specified verbatim). The `:44-50` arg doc now carries the required ordinal-level rationale: `JevXForDecision._normalize_option` builds `Option(key=option, label=str(option))` (`jevx/modeling_jevx.py:367-369`) and `project_to_schema` returns it verbatim, "so a non-`str` key is a documented, supported path, not a type error". No other site needed changing (`Option.text()` already falls back to `str(self.key)`). |
| Q-036 | **CLOSED** (was 2-of-4 PARTIALLY CLOSED) | All four residues are now annotated/removed. Two landed in the earlier pass: `register_calibrator(name: str) -> Callable[[type[BaseCalibrator]], type[BaseCalibrator]]` at `jevx/calibration/base.py:54` and `register_confidence_method(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]` at `jevx/confidence/base.py:47`. The two handed-off halves have since landed: `register_decision_head(name: str) -> Callable[[type[T]], type[T]]` at `jevx/decisions/generic.py:59` (`Callable` added to that module's `typing` import at `:30`), and the trivial pass-through `ChoiceHead.build_output` override is **deleted** — `jevx/decisions/choice.py` (188 lines) now contains **zero** occurrences of `build_output`, leaving `DecisionHead.build_output` in `jevx/decisions/generic.py` as the single implementation. |

**Also now closed — the remaining handed-off halves of Q-025 and Q-028.** Re-verified
this pass, so the closure set is complete rather than partial:

| # | Status now | Verified evidence (file:line, re-read this pass) |
|---|---|---|
| Q-025 | **CLOSED** (both halves; was 1-of-2) | The second half is the `jevx/modeling_jevx.py:78-82` docstring edit the earlier pass specified: `state_representation` holds the shared projected `H_S [B, D]`, and the per-row fused `H = F(H_S, H_Q, H_C)` is "internal to `forward` and is **not** exposed on the output (`capability_metadata` carries no fused vectors — see docs/API.md)". Consistent with the already-closed first half at `jevx/modeling_outputs.py:88-95` and the class note at `:110-114`. |
| Q-028 | **CLOSED** (was HANDED-OFF) | `_restore_calibration_sidecar` (`jevx/modeling_jevx.py:1250`) now emits the required signal before the `return`: at `:1263-1276`, when `calibration_file is None` and `self.calibration_engine.method != "none"`, it raises a `RuntimeWarning` naming the method and the path and stating that "the calibrator stays UNFITTED and calibration is a no-op" (`stacklevel=2`). The optional-sidecar `except Exception` is kept; a local-directory reload is unaffected. |

### Updated tally after this wave

| Severity | Findings | Closed | Open | Environment-blocked |
|---|---:|---:|---:|---:|
| P0 | 2 | 2 | 0 | 0 |
| P1 | 18 | 18 | 0 | 0 |
| P2 | 16 | 16 | 0 | 0 |
| **Total (Q-001…Q-036)** | **36** | **36** | **0** | **0** |
| Known #1–#11 | 11 | 10 | 0 | 1 |

**All 36 audit findings are now closed; none remains open.** No finding was added or
merged in this wave, so the per-severity counts (P0 2 / P1 18 / P2 16) are unchanged from
the original audit — only the Closed column moves, from 30/36 to 36/36. The one
environment-blocked item (Known #11: quantized / cross-device / GPU) is unchanged and is
still **never** counted as passing: those remain NOT PRODUCED in this CPU-only,
no-`bitsandbytes` environment. The earlier "Updated tally" section above is retained
verbatim as the record of the 30/36 state at its date.

**One regression found and fixed in this wave (not an audit finding).** The six
§94 golden-reference fixtures were captured under the pre-§5.f option formula, so the
new `option_normalize=True` default moved them past the 1e-6 tolerance and all six
golden tests failed. The drift was proven to come from that one intentional change alone
— re-deriving with `option_normalize=False` reproduces the old fixtures **bitwise**
(max |Δp| = 0.000e+00 on all six models) — so the fixtures were regenerated through the
documented generator `vendor/reports/_gen_golden_fixtures.py`, with the intent recorded
in `tests/golden/test_golden_references.py`. Only `probs` and `predicted_indices`
moved; the `state_representation` checksums are unchanged, which is the expected
signature of an option-branch-only change. Tolerances and assertions were not touched.


## Closure note — JevX-Next wave (2026-09-26)

No new finding is opened by this wave, and no previously-closed finding is
reopened. What follows is what the wave did to the audit's own surface, plus the
one defect it found by execution rather than by review.

### Findings touched

| Finding | Status after this wave |
|---|---|
| Q-004 (P1) — "every documented suite count is stale" | **re-verified and re-closed**: `docs/TESTING.md` counts and runtimes are recounted from this wave's own detached run (`docs/TESTING.md` §1, §6) and the four new test files are listed. The finding's shape is unchanged — it is a standing obligation, not a one-time fix. |
| Q-010 (P1) — §31 surface and the decision_type config pin are shipped but untested | **extended, still closed**: the five new `JevXConfig` fields, `forward(condition_options=)`, `set_interaction_scale`/`get_interaction_scale`, the curriculum API and the gate API all ship with dedicated test files (`test_head_next.py` 77, `test_state_only_path.py` 9, `test_curriculum.py` 35, `test_conditioning_gate.py` 45). |
| Q-011 (P1) — public API symbols with no test reference | **extended, still closed**: no new untested public symbol. `__all__` for `jevx.training` exports 35 names; all are reachable from the four test files above. |
| Q-030 (P2) — runtimes quoted across docs are stale | **re-verified and re-closed** for the suite line; see `docs/TESTING.md` §6. Benchmark runtimes quoted elsewhere are unchanged and remain dated. |
| Q-032 (P2) — `API.md` §15 "every documented raise site is asserted by tests" is overbroad | **unchanged** (still closed with its stated wording), and this wave did not repeat the overbroad phrasing: `docs/API.md` §14.6/§14.7 list the new raise sites as "raises", not as "every site tested". |
| Known #11 — quantized / CUDA / cross-device | **still environment-blocked and still never counted as passing.** This wave executed on the same CPU-only host and changed nothing here. |

### Two defects found by execution

| Defect | Severity | Status |
|---|---|---|
| **State-only degeneracy through the duplicate-option tie.** `condition_options=False` substitutes exact zeros for the option rows to keep the `[B, K, D]` shape contract. Zeros are "identical" for every schema, so the tie fired and averaged the K **per-class-by-construction** state scores into one value. Measured logit spread **0.0**; and because the tie gives every logit the same value, `∂L/∂m = Σ_k (p_k − y_k) = 1 − 1 = 0`, so the analytic gradient into every state-pathway parameter is **zero** (float32 round-off ~3e-07 is all that survives, against 2.7 with the tie off) and cannot be trained out. | **P1 — FIXED** | `JevXForDecision.forward` passes `duplicate_tie=condition`, suppressing the tie on the state-only path **and only there**; every other call keeps the documented per-schema tie bit-for-bit. Measured spread 0.0 → **0.2409**. Guarded by `tests/unit/test_state_only_path.py` (9 tests). |
| **`run_curriculum` Stage B never re-enabled the option pathway.** The per-call flag was a construction-time constant; the λ ramp was recorded and applied to a buffer the forward never read, because the flag — not the scale — is what gates the residual. Cells E/F of the first matrix were therefore "state-only for 4 epochs plus an inert ramp". | **P1 — FIXED** | `_StageController` owns the per-step flag and the per-step scale, both pure functions of the stage; `EpochRecord.condition_options` records what the epoch ran under. **Four regression tests, each verified to fail against the defect restored in place.** Cells E and F re-run; public API unchanged. |

### The honesty obligations this wave kept

* The two negative results are stated in **Known Limitations** and in the report's
  own ablation table, not buried: B (nonlinear state path, interaction off) does
  **not** reproduce the MLP probe (−1.11 / −0.33 / +0.48 / **−5.60** pt), and
  **C ≤ B on all four datasets** (−0.52 / −0.15 / −0.14 / **−6.96** pt). The
  curriculum + gate win on RAVDESS is reported **with** the fact that they still
  sit below both probes and only +1.96 pt above the interaction-free arm with
  overlapping ranges.
* The conditioning stack's cost is stated as measured: **50.6 %** of trainable
  parameters at D=128 (55.5 % at D=768) and **18.3×–59.3×** the MLP probe's
  per-example latency. The "23x–35x" summary in
  `reports/HEAD_ABLATION_MATRIX.md` §6 does **not** survive recomputation from
  the JSONs; the exact range is recorded instead, and the discrepancy is flagged
  rather than smoothed.
* Superseded rows are **labelled in place**, not deleted: the E/F rows in
  `reports/HEAD_ABLATION_MATRIX.md` §1–§4 and §8 remain as the audit's dated
  record with a pointer to §11, and the four
  `benchmarks/results/head_ablation_<dataset>.json` files are untouched.

### Tally after this wave

Unchanged: **36 findings (Q-001…Q-036), 36 closed, 0 open**, plus 10 of 11 known
findings closed and Known #11 environment-blocked. This wave added 166 tests
across four files and closed nothing that was open, because nothing was open.
The change is in the evidence base, not in the count.

## Program closure — 2026-09-26 (gap-closure program, WAVE-1 … WAVE-6)

The finding tally above is unchanged (**36 findings Q-001…Q-036, 36 closed, 0 open**;
10 of 11 known findings closed, Known #11 environment-blocked). The gap-closure program
was a separate register, walked in full in `reports/FINAL_VERIFICATION.md` § "Gate
Refresh — 2026-09-26 (gap-closure program, WAVE-1 … WAVE-6)". This section is the
quality-audit's copy of that walk, with the deciding evidence and the audit's own status
vocabulary.

**Status vocabulary used here:** `CLOSED` — the gap no longer exists, and a test fails if
it returns. `ANSWERED-NEGATIVE` — the gap is real, was investigated, and the honest
answer is **no**; nothing is "fixed" and nothing is claimed as a win.
`ENVIRONMENT-BLOCKED` — the question cannot be answered on this host, is reported as
blocked everywhere, and is never counted as passing.

| Gap | Status | Deciding evidence (one line) |
|---|---|---|
| G0.1 | **CLOSED** | `schema_encoder_width` and `conditioning_mode="constant"` are **wired into the module tree / the forward**, not accepted-and-ignored (`modeling_jevx.py:666-674`); default `state_dict` unchanged bit for bit. |
| G0.2 | **CLOSED** | The `(mode, enabled)` pair is **total**: 6/6 pairs end in agreement or a named error, no silent third outcome (`configuration_jevx.py:373-395`; `test_conditioning_mode_precedence.py`). |
| G1 | **ANSWERED-NEGATIVE** | B (MLP state path, interaction off) **≠ P2** on 3 of 4 datasets: −1.11 / −0.33 / +0.48 / **−5.60 pt**; not short of capacity (49 536 params vs P2's 17 028 trunk) nor of gradient (17×–70×). |
| G2 | **ANSWERED-NEGATIVE** | **C ≤ B on all four datasets** (−0.52 / −0.15 / −0.14 / **−6.96 pt**): option conditioning carried no measurable information on top of the state path anywhere in the matrix. |
| G3 | **ANSWERED-NEGATIVE** | Gate + curriculum win on **RAVDESS only** (F − C = +8.92 pt, 5/5, non-overlapping) and still land **below P1 and P2**; elsewhere inside one seed std. |
| G4 | **ANSWERED-NEGATIVE** | Audio A0–A6, 5 seeds: shipped A0 **0.2376** is the worst of ten cells; init refuted (A4 −5.92 pt), capacity not the limit (A6 better), **budget is the lever** (A5@20ep **0.5416**). |
| G5 | **ANSWERED-NEGATIVE** | Vision V0–V6, 65 cell-runs: V2 linear probe **0.9688** vs shipped V0 **0.9412** (5/5 disjoint); cost is the question/ConcatFusion half; the published **0.9087 does not reproduce at HEAD**. |
| G6 | **CLOSED — on one cell only** | First level-4 cell earned: **bert @ `google-bert/bert-base-uncased`**, all 7 criteria PASS, 109 482 240 backbone params. `deit-tiny` / `roberta-base` ship no pooler weights. Breadth: **1 family, 1 dataset**. |
| G7 | **CLOSED** | The conversion reference is mask-driven, not position-indexed; the six-cell grid (2 strategies × 3 paddings) agrees **bitwise** (`test_conversion_equivalence_left_pad.py`). |
| G8 | **CLOSED** | Supervised-step dispatch reads the **signature**; the old duck-probe silently bound a collated batch to `state` on `JevXForDecision`. The trainer now shares one `_call_model` (`test_supervised_step_dispatch.py`). |
| G9 | **CLOSED** | `shared_question_schema_encoder` is RESERVED and **rejected by a named `ValueError`** at construction — the config can never silently accept a switch that does nothing. |
| G10 | **CLOSED** | `conditioning_scale_init` mapping documented **and** tested: config `0.0` = *"unspecified"* → head **λ = 1.0**; a persistent buffer must keep the pre-flag default. |
| G11 | **CLOSED** | `BaseAdapter._normalize_sequence` deduplicated to one canonical definition (source pin + CNN path intact). |
| G12 | **CLOSED** | `LinearProbePathway` docstring no longer claims an init it never runs; a constructed pathway contributes exactly nothing. |
| G13 | **CLOSED** | Confidence fit state is a **persisted `_fitted` bool buffer**; pre-`_fitted` checkpoints still load unchanged under `strict=True`. |
| G14 | **CLOSED** | `has_inf` handles complex tensors like `has_nan`, without changing the integer/bool answer. |
| G15 | **CLOSED** | `_resolve_dtype`'s unreachable string special cases deleted; no accepted name changes. |
| G16 | **CLOSED** | `batch_equivalence_tolerance` raises `ValueError` (not `KeyError`) for a real dtype with no safety factor; all dtype values **bitwise unchanged**. |
| G17 | **CLOSED** | `decision_type` is a **declared, validated, round-tripping config field** (was `from_pretrained`-only, so it neither validated nor round-tripped). |
| G18 | **ANSWERED-NEGATIVE** | The suite exists and ran: 11 questions / 4 datasets / 5 seeds / **35 paired comparisons → 0 wins**; the one non-overlapping comparison is a **−16.92 pp loss for `full`**. The conditioning half is **80.8–89.7 %** of trainable parameters. |
| G19 | **ENVIRONMENT-BLOCKED** | Restated verbatim from `reports/ENVIRONMENT_BLOCKED.md` §2 (2 rows: quantized, cross-device). Never counted as passing, never simulated. |
| G20 | **CLOSED** | Duplicate-option tie is **on-device**: **3456 comparisons bitwise** against the kept host-transfer oracle, **zero host transfers** (counter-instrumented), median speedup asserted **≥ 3.0×**; eval `base_forward` is 1 forward not 2; trainer shares `_call_model`. |

**Tally: 22 gaps — 15 CLOSED, 6 ANSWERED-NEGATIVE, 1 ENVIRONMENT-BLOCKED.** G9, G10 and
G20 carry **reconstructed labels** (the register itself is not in the tree; see the
*Provenance of the register* subsection of the gate refresh). Their evidence is real and
cited; only the label is inferred.

### What the audit's own obligations required of this program

| Obligation | How it was kept |
|---|---|
| Never let a negative result disappear | The **6** ANSWERED-NEGATIVE gaps are the first table in the gate refresh, not a footnote, and each carries the measured number and the seed counts. |
| Never count an unrunnable claim as passing | G19 is ENVIRONMENT-BLOCKED in every artifact that mentions it, with the canonical table reproduced verbatim. |
| A documented claim that the code contradicts is worse than a missing one | Consolidation swept every live doc against the code and corrected **6** such claims: `state_pathway=False` (not a valid value), the config-level π roster (only `("linear","mlp")`; `"none"` is head-only), the (5.g2) "returns `{}`" claim (false at the shipped `"off"` default), the `condition_options` inline comment (lumped `"constant"` with `"full"`), the vision `0.56×` wall-time basis (ratio-to-V2 quoted as ratio-to-V0), and the stale `configuration_jevx.py` line numbers + "(default `full`)" in that same snippet. |
| A default change must be documented and reversible | The `conditioning_mode` default is flipped to `"off"` with the **one-line revert** stated in `docs/API.md` §3.1, spec §5.i and `CHANGELOG.md`. |
| Goldens must not be silently re-baselined | The six `GOLDEN_SPECS` fixtures pin `conditioning_mode="full"` and were **deliberately not regenerated**; recorded as a Known Limitation. |
