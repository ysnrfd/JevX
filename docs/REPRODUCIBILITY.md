# Reproducibility (§87)

One-page reproduction block for every claim in this repository. Each value below
was *measured* or *observed* on the stated host and is quotable from the cited
source. No number here is hand-written.

## 1. Environment of record

| Item | Value | Source |
|---|---|---|
| Date of record | 2026-09-24 | all reports |
| OS | Microsoft Windows 11 Pro 10.0.26100 (win32 10.0.26100 x64) | REPOSITORY_RECONNAISSANCE.md; reports/MATHEMATICAL_VERIFICATION.md |
| CPU | Intel Core i5-6300U @ 2.40 GHz (Intel64 Family 6 Model 78 Stepping 3), 4 logical cores (2C/4T, 2015 mobile) | REPOSITORY_RECONNAISSANCE.md |
| RAM | 7.9 GiB reported (~8 GB); psutil read 8,476 MB in the benchmark runs | REPOSITORY_RECONNAISSANCE.md; reports/LATENCY_BENCHMARKS.md |
| GPU | **none** — `torch.cuda.is_available() = False`, no MPS, no `nvidia-smi` (CPU-only wheel `2.14.0+cpu`) | REPOSITORY_RECONNAISSANCE.md |
| Python | 3.14.3 (MSC v.1944, 64-bit), `C:/Python314/python` | REPOSITORY_RECONNAISSANCE.md |
| torch | 2.14.0+cpu (torch threads = 2 during benchmark runs; `torch.version.cuda = None`) | REPOSITORY_RECONNAISSANCE.md; reports/LATENCY_BENCHMARKS.md |
| transformers | 5.16.1 — installed package matches the vendored clone `vendor/transformers` @ tag v5.16.1, SHA `93c8b7b485963a10800c91f55304db6be211c2bd` | REPOSITORY_RECONNAISSANCE.md; reports/TRANSFORMERS_SOURCE_AUDIT.md |
| tokenizers | 0.23.2 | REPOSITORY_RECONNAISSANCE.md |
| safetensors | 0.8.0 | REPOSITORY_RECONNAISSANCE.md |
| datasets | 4.4.1 | REPOSITORY_RECONNAISSANCE.md |
| pytest | 9.0.3 | REPOSITORY_RECONNAISSANCE.md; reports/MATHEMATICAL_VERIFICATION.md |
| numpy | 2.3.5 | REPOSITORY_RECONNAISSANCE.md |
| dtype | float32 throughout (no mixed-precision path validated) | reports/DATASET_BENCHMARKS.md; docs/LIMITATIONS.md §6 |
| Extra deps used by Phase-92 + examples | soundfile 0.14.0, sentencepiece 0.2.2 (installed this phase; `librosa`/`torchaudio` absent and not needed) | reports/REAL_MODEL_VALIDATION.md |
| Network | huggingface.co and pypi.org HTTP 200 (unauthenticated, rate-limited); first run downloads checkpoints | REPOSITORY_RECONNAISSANCE.md |

Reporting rule (docs/LIMITATIONS.md §1): every number carries machine, versions,
dtype, batch shape and thread count — numbers without that tuple are not results.

## 2. Checkpoints: ids → resolved revision SHAs

18 executed checkpoints (reports/REAL_MODEL_VALIDATION.md Appendix B; resolved
huggingface_hub revisions also stored per-id in `vendor/reports/_matrix_env_probe.json`):

| model id | revision sha |
|---|---|
| `hf-internal-testing/tiny-random-bert` | `f171d7baecaf37b5da5a3616d8833b9969753535` |
| `hf-internal-testing/tiny-random-RobertaModel` | `5f1d755e93e9fe46b9bb90404f1e810515b2410a` |
| `hf-internal-testing/tiny-random-DistilBertModel` | `cc4425ad0676f3ec00e8bffe485fe83cae61041a` |
| `hf-internal-testing/tiny-random-AlbertModel` | `34a63314686b64aaeb595ddb95006f1ff2ffda17` |
| `hf-internal-testing/tiny-random-DebertaV2Model` | `9dab549e7d9f719c1438cdd3928d59545c72b0d0` |
| `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| `hf-internal-testing/tiny-random-t5` | `2f582cd79ed5795b71539951d237945bc1c5ac7e` |
| `hf-internal-testing/tiny-random-BartForConditionalGeneration` | `45bbb767e8601574fe69ce80e82ed295716c9076` |
| `sshleifer/tiny-gpt2` | `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` |
| `yujiepan/qwen2-tiny-random` | `b01b9c82aaf1efb4c26c94b6342d611b397245ff` |
| `hf-internal-testing/tiny-random-LlamaForCausalLM` | `9fb191250dd56d0ba7ec9785a025ed29c03d5998` |
| `hf-internal-testing/tiny-random-ViTModel` | `31817b7a64ebc3333fcd4801dfbb356ab07b13dd` |
| `hf-internal-testing/tiny-random-CLIPModel` | `703b075aa101387f504f075dca2e0c78c99ee420` |
| `hf-internal-testing/tiny-random-SiglipModel` | `5f4a673ccb0e7dc620d13a946dd99f95dba45653` |
| `hf-internal-testing/tiny-random-Wav2Vec2Model` | `7a998ee3ee0619a52828a79c3eed6872fd053f37` |
| `hf-internal-testing/tiny-random-ASTModel` | `75e68f956f6f2c0709b01e596e7a6aecb1b29dce` |
| `hf-internal-testing/tiny-random-WhisperForConditionalGeneration` | `598101b885b24508042d9292e54aa04bff96318e` |
| `katuni4ka/tiny-random-llava-next` | `7ca7d02e9a5da08ddd26b306b9482632578dc471` |

All 18 ids returned HTTP 200 before execution
(`vendor/reports/_matrix_env_probe.json`). **Benchmark-specific checkpoints**:

| id | revision sha | params | role | source |
|---|---|---|---|---|
| `google/bert_uncased_L-2_H-128_A-2` | `30b0a37ccaaa32f332884b96992754e246e48c5f` | 4,385,920 frozen | quality-suite backbone, SST-2 + ag_news (EXP-1/2/7) | reports/DATASET_BENCHMARKS.md §1, §8 |
| `distilgpt2` | `2290a62682d06624634c1f46a6ad5be0f47f38aa` | 81,912,576 | EXP-2 generative baseline B3 (zero-shot, n=1) | reports/DATASET_BENCHMARKS.md §1 |
| `facebook/deit-tiny-patch16-224` | `b3428f18dcc7b543470d07f14b4a4157815d1880` | 5,561,472 frozen | EXP-V vision quality backbone (Imagenette) | reports/MODALITY_BENCHMARKS.md §1 |
| `superb/wav2vec2-base-superb-ks` | `372e0486cd83e6f0c05c20a27262e9ca09450d24` | 94,371,712 frozen | EXP-A audio quality backbone (RAVDESS) | reports/MODALITY_BENCHMARKS.md §1 |

Note (failure policy, reports/DATASET_BENCHMARKS.md §1): the planned primary
`prajjwal1/bert-tiny` exists (sha `6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837`)
but its `config.json` has **no `model_type` key**, so `AutoConfig` refuses it; the
benchmark therefore uses `google/bert_uncased_L-2_H-128_A-2` (same L=2/H=128/A=2
architecture, real pretrained weights).

## 3. Dataset: id, revision, splits, seeds

| Item | Value | Source |
|---|---|---|
| Dataset | `stanfordnlp/sst2`, pinned via `load_dataset(..., revision=...)` | reports/DATASET_BENCHMARKS.md §1 |
| Revision | `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb` | same |
| License | Hub card declares `license: ["unknown"]` (recorded as-is from the API, not inferred) | same |
| train split | 3,000 rows — stratified carve from the official SST-2 **train** split (seed 0); 1,327 neg / 1,673 pos | §2 |
| calibration split | 1,000 rows — stratified carve from official train, **disjoint** from train (drawn from the remainder, seed 0+1); 442 / 558 | §2 |
| test split | **872 rows** — the official validation split used in full (requested cap 1,500 was not binding); 428 / 444 | §2 |
| Split construction | deterministic (`benchmarks/bench_common.py`: `stratified_indices` / `make_splits`, largest-remainder quotas + seeded `np.random.RandomState`); disjointness asserted at construction and re-verified by the smoke tests | §2 |
| Task framing | one Choice question: `"Is the sentiment of this review positive or negative?"`, schema `["negative", "positive"]` (index 0 = label 0 — no label remapping anywhere) | §2 |
| Training seeds | EXP-1 / EXP-2: **3 seeds {0, 1, 2}** (mean±sample-std reported); EXP-3…EXP-6 ablations: **single seed 0** (split seed also 0) — one draw per configuration, reported as single-sample differences | reports/DATASET_BENCHMARKS.md §1; reports/ABLATIONS.md seed table |
| Hyperparameters | epochs=4, lr=1e-3 (AdamW, `HeadTrainer` defaults — chosen a priori, **no hyperparameter search on any split**), batch=64, max_length=128, grad_clip=1.0; JevX: `freeze_base_model=True`, `pooling_method="cls"` (primary), `fusion_type="concat"`, widths = backbone width (128); 1,247,361 trainable JevX params (measured) | §1 |
| Split protocol | heads fit ONLY on train; calibrators (temperature/vector/isotonic) and the energy-confidence fit ONLY on calibration; test used ONLY for final metrics — never for fitting, tuning or selection | §2 |

Other seed policies in the repo:

| Where | Seed | Source |
|---|---|---|
| Golden fixtures / tests | `SEED = 0` (`random`/`numpy`/`torch.manual_seed`) | tests/golden/test_golden_references.py |
| Theorem/property trials | `torch.manual_seed(20260924 + trial)`; local `torch.Generator`s; calibrator fits RNG-free (temperature grid, isotonic PAVA) or seeded (vector LBFGS seed 0) | reports/MATHEMATICAL_VERIFICATION.md |
| `HeadTrainer` default | `seed=42` (torch + loader generator) | jevx/training/trainer.py |
| conftest tiny fixtures | `torch.manual_seed(0..4)` per fixture | tests/conftest.py |

Extension datasets (same split/seed discipline — heads on train only,
calibrators on calibration only, test untouched):

| Dataset | Revision | Splits (train/cal/test) | Seeds | Source |
|---|---|---|---|---|
| `fancyzhx/ag_news` | `eb185aade064a813bc0b7f42de02595523103ca4` | 3,000 / 1,000 / first 1,500 of official test | {0,1,2} | reports/DATASET_BENCHMARKS.md §8 (EXP-7) |
| `frgfm/imagenette` (config `imagenette2-160`) | `a8a496123b40fa739da2acf9b0dae339d30c7bae` (+ S3 data tarball sha256 `64d0c485…02fe5`) | 800 / 300 / 500 (official val subsample) | {0,1,2} | reports/MODALITY_BENCHMARKS.md §1 (EXP-V) |
| `Codec-SUPERB/RAVDESS` | `46aa36c5fdeb1b6d63b746b74fde528a540044b2` | 700 / 240 / 500 (disjoint remainder) | {0,1,2} | reports/MODALITY_BENCHMARKS.md §1 (EXP-A) |

## 4. Benchmark entry points, CLI arguments, result files

| Entry point | CLI (as published) | Writes | Measured runtime (this host) |
|---|---|---|---|
| `benchmarks/run_quality.py` | `--backbone`, `--dataset`, `--dataset-revision`, `--seeds` (default `0,1,2`), `--ablation-seed` (0), `--train-size 3000`, `--cal-size 1000`, `--test-size 1500`, `--epochs 4`, `--lr 1e-3`, `--batch-size 64`, `--max-length 128`, `--fusion concat`, `--pooling cls`, `--experiments all`, `--skip-b3`, `--skip-latency`, `--latency-iters 100`, `--output-dir`, `--run-label` | `benchmarks/results/exp1_…exp6_*.json`, `latency_heads_only.json`, `run_summary*.json` | full run **646.54 s** (incl. 150.97 s first dataset download) + EXP-3 rerun **118.11 s** = **764.65 s**; dev runs: mini 26.52 s, medium 48.64 s (reports/DATASET_BENCHMARKS.md §1) |
| `benchmarks/run_latency.py` | `--skip-load`, `--quick` (smoke run) | `benchmarks/results/latency_*_<stamp>.json` + `*_latest.json` | **~4 min** (reports/LATENCY_BENCHMARKS.md) |
| `benchmarks/run_memory.py` | `--questions` (default 32) | `benchmarks/results/memory_*_<stamp>.json` + `*_latest.json` | **~40 s** (reports/MEMORY_BENCHMARKS.md) |

Published quality invocation (reports/DATASET_BENCHMARKS.md §1):

```bash
python benchmarks/run_quality.py --train-size 3000 --cal-size 1000 --test-size 1500 \
  --seeds 0,1,2 --epochs 4 --lr 1e-3 --batch-size 64 --max-length 128 \
  --pooling cls --fusion concat --experiments all
```

Machine-readable validation artifacts: `vendor/reports/_jevx_matrix_results.json`
(format `jevx-matrix-results-v1`, regenerated on every test run via atexit) and
`vendor/reports/_matrix_env_probe.json` (per-id resolved shas).

## 5. Test suite and validation re-runs

| Command | Result on this host | Expected wall time | Network |
|---|---|---|---|
| `python -m pytest` (full; `pytest.ini` scopes to `tests/`) | **1119 passed, 1 skipped, 3 warnings in 1380.17s (0:23:00)** — executed 2026-09-25 for the breadth & quality wave (tree collects 1120 — docs/TESTING.md §1; verbatim summary in reports/FINAL_VERIFICATION.md gate refresh). Historical runs on the same host, kept as history: `883 passed, 1 skipped … 2393.61s` (final verification pass), `768 passed, 1 skipped … 264.90s`, `… 251.43s`, `… 233.31s` (pre-Qwen-battery; the last quoted in reports/MATHEMATICAL_VERIFICATION.md) | **~23 min** (Qwen3.5 battery ~22 min of it) | yes on first run (checkpoint downloads; cached afterwards); network failures → documented skips, never silent passes |
| `python -m pytest tests/unit -q` | fast subset, network-free (in-memory fixtures) | ~1–2 min | no |
| `python -m pytest tests/numerical -q` | **42 passed in 8.57 s** (2026-09-25); earlier quoted run: 41 passed in 8.48 s (reports/MATHEMATICAL_VERIFICATION.md) | ~10 s | no |
| `python -m pytest tests/golden -q` | re-derives 6 golden records, tolerance 1e-6 | ~58 s | yes (checkpoints) |
| `python -m pytest tests/integration/test_matrix_text.py -q -s` | 11 text families | ~1–2 min | yes |
| `python -m pytest tests/integration/test_matrix_vision_audio.py -q -s` | 7 vision/audio/multimodal families | ~1–2 min | yes |
| `python vendor/reports/_probe_matrix_env.py` | env + checkpoint existence/sha probe → `_matrix_env_probe.json` | seconds | yes (API) |
| `python vendor/reports/_gen_golden_fixtures.py` | **deliberate** golden-fixture regeneration (run ONLY to regenerate; see docs/TESTING.md) | ~30 s | yes |
| `python vendor/reports/_gen_fixture_assets.py` | idempotent modality fixtures (`input_image_224.png`, `input_audio_1s.wav`) | seconds | no |

Environment-blocked rows (never faked — reports/FAILURE_ANALYSIS.md): row 15
quantized inference (CPU-only torch build; `bitsandbytes` absent) and row 11
cross-device input (no second torch device — the one documented suite skip).

## 6. What "reproduced" means here

- Same-host reruns of the suite are deterministic within-process
  (bitwise-equal eval probabilities, `tests/unit/test_padding_determinism.py`);
  **cross-process** bitwise identity is a documented non-goal (kernel/thread
  partitioning) — see reports/FAILURE_ANALYSIS.md §74.
- Benchmark numbers are host- and load-specific: the latency/memory hosts ran
  concurrent agents, cells with IQR/median > 0.25 were flagged noisy and retried
  once, and differences below ~1 ms are treated as noise
  (reports/LATENCY_BENCHMARKS.md).
- Quality numbers are sample estimates on n=872 test rows with sampling error
  (reports/ABLATIONS.md honesty summary; docs/LIMITATIONS.md §4/§10).
