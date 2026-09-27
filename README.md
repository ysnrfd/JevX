# JevX — a typed probabilistic decision layer over Hugging Face Transformers

JevX wraps a Hugging Face backbone so that a **model + state + question + schema**
evaluates to a **typed probabilistic decision**:

```
J_theta(S, Q, C) -> D        # state S, question Q, schema C, decision D
```

Concretely: given any Transformers-compatible checkpoint, JevX encodes the state
`S` once through the backbone, encodes the question(s) `Q` and the schema options
`C`, fuses them, runs a typed decision head (`choice` / `score` / `noul`), applies
optional post-hoc calibration, computes a confidence, and projects the result back
into your declared schema — returning `probs`, a schema-member decision, a
confidence, and an optional abstention flag per question. The formal definitions
live in [docs/MATHEMATICAL_SPECIFICATION.md](docs/MATHEMATICAL_SPECIFICATION.md)
and [docs/PROOFS.md](docs/PROOFS.md).

```python
from jevx import JevX

model = JevX.from_pretrained("hf-internal-testing/tiny-random-bert")
out = model.choice(
    "The crew repaired the bridge support before the storm.",
    questions=["Is this content safe to deploy?"],
    options=[["safe", "unsafe"]],
)
d = out.decisions[0]
print(d.predicted_value, d.probabilities, float(d.confidence[0]))
```

## Honest status (2026-09-24)

| Aspect | Status |
|---|---|
| Test suite | **1284 passed, 1 skipped, 1 environment-blocked failure, 3 warnings in 1380.03s (0:23:00)** (2026-09-26, 1286 collected — the 1 failure is disk exhaustion on C: during a multi-GB Qwen3.5 shard write; the test re-run alone with `--basetemp` on D: is `1 passed in 102.05s`) on this host — the skip is the documented cross-device test on a CPU-only machine (no second torch device); the runtime is dominated by the Qwen3.5-0.8B integration battery; see [docs/TESTING.md](docs/TESTING.md) |
| Real checkpoints executed | **18/18 green** through the full §92 protocol (load → real-modality input → decision invariants → save/load equality → `base_forward` vs raw AutoModel → shared-state single forward → extraction correctness vs an independent manual reference); 17 of the 18 carry **random weights** (pipeline fixtures), `sentence-transformers/all-MiniLM-L6-v2` is a real trained checkpoint — see [reports/REAL_MODEL_VALIDATION.md](reports/REAL_MODEL_VALIDATION.md) |
| Architecture breadth | **70/70 architectures executed in memory — 70 PASS / 0 PARTIAL / 0 FAIL**, 6 families excluded with cited reasons (transformers 5.16.1, torch 2.14.0+cpu, 19.25 s). Scope is executability, not accuracy: the models are **randomly initialized**, no checkpoints or downloads. This wave also fixed **5 detector/adapter defects** and the **CNN hidden-width rule** that had made every CNN backbone unbuildable (4 families PARTIAL → PASS) — [reports/ARCHITECTURE_BREADTH.md](reports/ARCHITECTURE_BREADTH.md), `benchmarks/results/breadth_sweep.json` |
| Support levels | 13 families at **level 3** (executed + validated), 7 at **level 2** (lossy-pooling convention pinned), none at 1/0 — per-family evidence in [TRANSFORMERS_COMPATIBILITY_MATRIX.md](TRANSFORMERS_COMPATIBILITY_MATRIX.md); level 4 additionally requires a real-dataset quality result |
| Benchmarks | Quality on **4 datasets × 3 backbones** — SST-2 + ag_news (text, 3 seeds), Imagenette (vision, 3 seeds), RAVDESS (audio, 3 seeds) — plus ablations (single seed), latency and memory sweeps, all executed on CPU — headline numbers in [docs/BENCHMARKS.md](docs/BENCHMARKS.md); all ablation rows are single-draw observations. **Audio is not closed by the option-pathway fix**: under the spec §5.f unit-direction option row RAVDESS moved 0.1353 → 0.2220 accuracy (+8.67 pt) and NLL 2.3097 → 2.0787, but stays **12.5 pt below the linear probe** (0.3467) on the same features, and the synthetic control **loses 0.20 pt** (reported, not dropped). The shipped audio JSON predates §5.f and is kept as a dated record. The new `LinearProbePathway` head gains **+0.25 pt** on ag_news and does **not** beat the MLP probe |
| Proof checks | T1–T8 executable instantiations green ([reports/MATHEMATICAL_VERIFICATION.md](reports/MATHEMATICAL_VERIFICATION.md)) |
| Environment caveats | CPU-only (`torch 2.14.0+cpu`, no CUDA/MPS), Windows 11, 4 logical cores, ~8 GB RAM. GPU paths, quantized inference and cross-device inputs are **untested here — reported as untested, never as passing** ([docs/LIMITATIONS.md](docs/LIMITATIONS.md) §1) |

Scope of the universality statement (verbatim from
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) §2): *"HF Transformers-compatible models
with a meaningful differentiable representation."* The 709 inventoried architectures
in `vendor/reports/architecture_inventory.json` are **inventory, not support** —
support is per-family, per-level, evidence-based
([docs/SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md)).

## Install / local usage

Packaging lives in [pyproject.toml](pyproject.toml) (PEP 621, setuptools
backend). Both commands below were run and verified on this host:

```bash
pip install -e .          # runtime deps: torch, transformers, numpy, huggingface_hub
pip install -e ".[dev]"   # + dev extra (pytest)
```

- **Version:** `0.1.0` — single-sourced across `pyproject.toml`,
  `jevx.__version__`, `jevx.modeling_jevx.__version__` and
  `JevXConfig().version`, enforced by
  [tests/unit/test_version_consistency.py](tests/unit/test_version_consistency.py).
- **License:** MIT — see [LICENSE](LICENSE).
- **Changelog:** [CHANGELOG.md](CHANGELOG.md).
- **Python:** developed and tested on Python **3.14.3** with `torch` 2.14.0+cpu
  and `transformers` 5.16.1 (Windows 11, CPU-only). `pyproject.toml` declares
  `requires-python = ">=3.14"` — the only interpreter this tree has been
  executed on.

Without installing (straight from a checkout):

- Run any script **from the repository root** (Python puts the cwd on
  `sys.path`, so `import jevx` resolves), or
- Set `PYTHONPATH` to the repository root:
  `set PYTHONPATH=C:\path\to\JevX` (Windows) / `export PYTHONPATH=/path/to/JevX`
- The `examples/*.py` files do this themselves: each inserts the repo root into
  `sys.path` in its first lines, so they also work when invoked by absolute path.

Runtime dependencies (versions measured on this host, see
[REPOSITORY_RECONNAISSANCE.md](REPOSITORY_RECONNAISSANCE.md)): `torch` 2.14.0+cpu,
`transformers` 5.16.1, `tokenizers` 0.23.2, `safetensors` 0.8.0, `numpy` 2.3.5;
examples additionally use `soundfile` 0.14.0 (audio) and `Pillow` (vision).

## 60-second quickstart

Requires network on first run (checkpoint download; warm-cache load measured
3,955.8 ms median for `tiny-random-bert`, reports/LATENCY_BENCHMARKS.md §b). From the repo root:

```bash
python - <<'PY'
import torch
from jevx import JevX

model = JevX.from_pretrained("hf-internal-testing/tiny-random-bert")
model.eval()
options = [["safe", "unsafe"], ["approve", "review", "reject"]]
with torch.no_grad():
    out = model.choice(
        "The maintenance crew inspected the bridge deck and repaired the damage.",
        questions=["Is this content safe to deploy?", "Which disposition fits?"],
        options=options,
    )
for q, opts, dec in zip(["safe?", "disposition?"], options, out.decisions):
    print(q, "->", dec.predicted_value,
          "p=", [round(float(p), 4) for p in dec.probabilities[0, :len(opts)]],
          "conf=", round(float(dec.confidence[0]), 4))
assert out.decisions[0].predicted_value in ("safe", "unsafe")
assert abs(float(out.decisions[0].probabilities[0].sum()) - 1.0) < 1e-6
print("OK")
PY
```

Seven runnable examples (encoder, decoder, encoder-decoder, semantic embedding,
vision, audio, multimodal) live in [examples/](examples/README.md); each prints
real decisions and asserts its invariants — running them is testing.

## Documentation map

**Specification & theory (do not rewrite):**
[MATHEMATICAL_SPECIFICATION.md](docs/MATHEMATICAL_SPECIFICATION.md) ·
[PROOFS.md](docs/PROOFS.md) · [RESEARCH.md](docs/RESEARCH.md) ·
[ARCHITECTURE.md](docs/ARCHITECTURE.md) (12 Mermaid diagrams) ·
[LIMITATIONS.md](docs/LIMITATIONS.md) (the honesty contract — it wins any conflict)

**User & reference docs:**
[API.md](docs/API.md) · [TESTING.md](docs/TESTING.md) · [TRAINING.md](docs/TRAINING.md) ·
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) ·
[TRANSFORMERS_INTEGRATION.md](docs/TRANSFORMERS_INTEGRATION.md) ·
[UNIVERSAL_CONVERSION.md](docs/UNIVERSAL_CONVERSION.md) ·
[SUPPORTED_MODELS.md](docs/SUPPORTED_MODELS.md) ·
[CALIBRATION.md](docs/CALIBRATION.md) · [CONFIDENCE.md](docs/CONFIDENCE.md) ·
[BENCHMARKS.md](docs/BENCHMARKS.md) · [FAILURE_MODES.md](docs/FAILURE_MODES.md)

**Reports (measured results):**
[REAL_MODEL_VALIDATION.md](reports/REAL_MODEL_VALIDATION.md) ·
[DATASET_BENCHMARKS.md](reports/DATASET_BENCHMARKS.md) ·
[ABLATIONS.md](reports/ABLATIONS.md) ·
[LATENCY_BENCHMARKS.md](reports/LATENCY_BENCHMARKS.md) ·
[MEMORY_BENCHMARKS.md](reports/MEMORY_BENCHMARKS.md) ·
[FAILURE_ANALYSIS.md](reports/FAILURE_ANALYSIS.md) ·
[TRANSFORMERS_SOURCE_AUDIT.md](reports/TRANSFORMERS_SOURCE_AUDIT.md) ·
[MATHEMATICAL_VERIFICATION.md](reports/MATHEMATICAL_VERIFICATION.md)

**Root-level:** [TRANSFORMERS_COMPATIBILITY_MATRIX.md](TRANSFORMERS_COMPATIBILITY_MATRIX.md)
(live support-level source) · [REPOSITORY_RECONNAISSANCE.md](REPOSITORY_RECONNAISSANCE.md)
(environment of record)

## Project layout

```
JevX/
├── jevx/                    # the package
│   ├── modeling_jevx.py     # JevXForDecision (alias JevX): factory, forward, save/load
│   ├── configuration_jevx.py# JevXConfig (model_type="jevx")
│   ├── modeling_outputs.py  # JevXOutput / DecisionOutput (transformers ModelOutput)
│   ├── adapters/            # state→inputs + representation extraction (registry chain)
│   ├── auto/                # AutoJevX + session-local AutoConfig/AutoModel registration
│   ├── capabilities/        # introspection detector, CapabilityReport, structured errors
│   ├── questions/           # QuestionEncoder (E_Q), SchemaEncoder (E_C), Option
│   ├── representations/     # pooling strategies, Projection, fusion kinds
│   ├── decisions/           # choice/score/noul heads + registry (+ 2 experimental)
│   ├── calibration/         # none/temperature/vector/isotonic + CalibrationEngine
│   ├── confidence/          # max_probability/margin/entropy/energy/learned engines
│   ├── training/            # losses, HeadTrainer, freeze control, DecisionDataset
│   └── utils/               # stable numerics + shape/finiteness validators
├── tests/                   # unit / integration / golden / numerical suites
├── benchmarks/              # run_quality.py, run_latency.py, run_memory.py + results/*.json
├── docs/                    # specification, reference docs (this set)
├── reports/                 # executed measurement reports
├── examples/                # 7 executable examples (each asserts invariants)
├── vendor/transformers/     # read-only source clone @ v5.16.1 (SHA 93c8b7b4…)
├── vendor/reports/          # audit scripts, inventories, _jevx_matrix_results.json
├── pytest.ini               # scopes collection to tests/ (vendor clone excluded)
├── TRANSFORMERS_COMPATIBILITY_MATRIX.md
└── REPOSITORY_RECONNAISSANCE.md
```

## Tests

```bash
python -m pytest        # full suite — 1119 passed, 1 skipped (~23 min on this host; the Qwen3.5 battery dominates)
python -m pytest tests/unit -q          # fast unit subset (no network needed)
python -m pytest tests/golden -q        # golden-fixture re-derivation (network)
```

`pytest.ini` scopes collection to `tests/` — without it, a bare `python -m pytest`
would also walk `vendor/transformers` and produce 780 pre-existing collection
errors before any test runs. Details, taxonomy and the golden-regeneration
procedure: [docs/TESTING.md](docs/TESTING.md).

## Reproducibility

Every reported number carries machine, versions, dtype, seeds and protocol:
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) is the one-page reproduction
block (environment, checkpoint revision SHAs, dataset id+revision+splits+seeds,
benchmark entry points with expected runtimes). Benchmark re-runs:

```bash
python benchmarks/run_quality.py   # full quality suite (~12.7 min measured incl. first download)
python benchmarks/run_latency.py   # ~4 min
python benchmarks/run_memory.py    # ~40 s
```
