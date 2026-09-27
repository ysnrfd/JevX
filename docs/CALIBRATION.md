# Calibration — implementation and measured results

Two halves: (A) what the code does (verified in `jevx/calibration/` and
asserted by tests), (B) what was *measured* on real data (every number quoted
from [reports/DATASET_BENCHMARKS.md](../reports/DATASET_BENCHMARKS.md) or
[reports/ABLATIONS.md](../reports/ABLATIONS.md)). API signatures:
[docs/API.md](API.md) §11. Epistemic rules: calibration is an aggregate
frequency property with sampling error — docs/LIMITATIONS.md §4 governs any
interpretation here.

## 1. What the code does (fact)

Pipeline position: `P' = Cal(P)` runs **after** the head produces logits and
**before** confidence and schema projection (`jevx/modeling_jevx.py::forward`).
It is applied only when `config.calibration_method != "none"` **and** the
calibrator `is_fitted`; applied decisions carry
`calibration_metadata = {"method", "applied": True}`.

| Method | What it fits | Fit algorithm | Argmax preserved? |
|---|---|---|---|
| `none` (default) | nothing | — | yes (identity) |
| `temperature` | scalar `T > 0` (stored as `log_temperature`) | deterministic log-grid T ∈ [1e-2, 1e2], 400 points + 6 halving refinements minimizing NLL — **RNG-free** | **yes, exactly** (softmax of `l/T` is monotone in each `l_k` for T>0 — spec §8.1) |
| `vector` | per-class affine `scale·z + bias` | full-batch LBFGS, seeded (`torch.manual_seed(0)`) | **not claimed** — may move argmax |
| `isotonic` | top-label isotonic map on (confidence, correctness) | deterministic PAVA with tie grouping | **not claimed** — may move argmax |

**Schema/type invariant (fact + tests):** every registered method maps
`[N, K] → [N, K]` with identical dtype and class count — calibration changes
values, never the schema, the decision type, or K
(`BaseCalibrator.calibrate` docstring; `tests/unit/test_calibration.py::
test_calibration_preserves_schema`, `::test_temperature_preserves_argmax`,
`tests/numerical/test_theorem_verification.py::
test_t8_calibration_then_project_preserves_membership`).

**Fit protocol (fact):** `fit(logits, labels)` is documented as *"on a
CALIBRATION split — never test"* in `BaseCalibrator.fit`,
`CalibrationEngine.fit` and the package docstring; the benchmark protocol
instantiates it literally — heads fit ONLY on train, calibrators ONLY on the
1,000-row calibration split, test (872 rows) used ONLY for final metrics
(reports/DATASET_BENCHMARKS.md §2).

**Fit-time validation:** `validate_fit_inputs` requires float `[N, K]` logits
and long `[N]` labels in `[0, K)` (`{0,1}` for K=1), equal lengths —
TypeErrors/ValueErrors name the tensor.

## 2. Sidecar serialization (fact)

Calibrator fit state cannot ride the safetensors weights: fit-data-shaped
buffers (isotonic breakpoints, vector scale/bias) are *dropped* by v5's loader
on shape mismatch (`core_model_loading.set_param_for_module`, cited in
`JevXForDecision.save_pretrained`). Therefore:

* `save_pretrained` writes `jevx_calibration.pt` **only when**
  `calibration_method != "none"` — payload `{format: "jevx-calibration-v1",
  method, state_dict}`;
* `model.state_dict()` filters `calibrator.*` keys symmetrically so no
  missing/unexpected-key noise appears;
* `from_pretrained` re-materializes calibrator buffers off the meta device,
  then restores the sidecar from the local dir (or Hub) — raising `ValueError`
  if the format marker is missing or the stored method disagrees with
  `config.calibration_method`;
* confidence fit state (scalar `energy_lo/hi`) has fixed shapes and rides the
  regular state_dict.

Enforced by `tests/unit/test_serialization_jevx.py` (5 tests: sidecar
round-trip at rel 1e-12, method-mismatch raise, tied causal-LM round trip, ...)
and failure row 16 of reports/FAILURE_ANALYSIS.md (fitted isotonic:
`is_fitted=True`, breakpoints bitwise-equal via `torch.equal`, decisions equal
within 1e-6, metadata restored).

## 3. Measured results — EXP-1 headline (3 seeds, mean ± sample std)

Protocol: frozen `google/bert_uncased_L-2_H-128_A-2`, SST-2, train 3000 /
cal 1000 / test 872, ECE/MCE with 15 bins. Quoted verbatim from
reports/DATASET_BENCHMARKS.md §3 (source JSON
`benchmarks/results/exp1_jevx_quality.json`):

| Metric | Uncalibrated | Temperature | Isotonic |
|---|---|---|---|
| Accuracy | 0.7011 ± 0.0104 | 0.7011 ± 0.0104 | 0.6950 ± 0.0180 |
| Macro-F1 | 0.6998 ± 0.0107 | 0.6998 ± 0.0107 | 0.6936 ± 0.0184 |
| NLL ↓ | 0.5759 ± 0.0067 | **0.5742 ± 0.0080** | 0.5807 ± 0.0081 |
| Brier ↓ | 0.3940 ± 0.0064 | **0.3926 ± 0.0072** | 0.3971 ± 0.0073 |
| ECE(15) ↓ | 0.0320 ± 0.0068 | **0.0234 ± 0.0003** | 0.0363 ± 0.0077 |
| MCE(15) ↓ | 0.0717 ± 0.0008 | **0.0517 ± 0.0133** | 0.2935 ± 0.3235 |

**Observed (measured), quoted:** temperature preserved accuracy
**bit-identically in all 3 seeds** — the argmax-preservation property
reproduced on real data; it lowered ECE 0.0320→0.0234 and MCE
0.0717→0.0517 and slightly lowered mean NLL/Brier. Isotonic **changed
decisions** in seeds 0 and 1 (0.7122→0.7144 up; 0.6995→0.6789 down — both
directions reported), and its MCE was high and unstable
(**0.2935 ± 0.3235**), consistent with a coarse step map fitted on n=1,000
calibration rows (LIMITATIONS §4).

## 4. Measured results — EXP-5 calibrator ablation (SINGLE seed 0)

reports/ABLATIONS.md EXP-5 (same trained heads; every calibrator fitted on the
1000-row calibration split, evaluated on the 872-row test split; source JSON
`benchmarks/results/exp5_calibration.json`):

| Calibrator | acc | NLL ↓ | ECE(15) ↓ | MCE(15) ↓ | Brier ↓ |
|---|---|---|---|---|---|
| none | 0.7122 | 0.5683 | 0.0371 | 0.0719 | 0.3866 |
| **temperature** | 0.7122 | **0.5654** | **0.0233** | **0.0478** | **0.3847** |
| vector | **0.7179** | 0.5685 | 0.0374 | 0.0601 | 0.3873 |
| isotonic | 0.7144 | 0.5716 | 0.0397 | **0.6667** | 0.3888 |

Single-seed winners (ABLATIONS.md): temperature on NLL/ECE/MCE/Brier; vector
on accuracy. Observed: temperature preserved accuracy exactly (0.7122 →
0.7122); vector changed decisions (+5 examples); isotonic's MCE measured
**0.6667** — an order of magnitude worse than every other row, its coarse step
map putting a large confidence gap into a small high-confidence bin
(n_cal = 1000 → few distinct breakpoints — the LIMITATIONS §4 caveat visible
in the data).

## 5. Honest interpretation

* **Temperature does not change decisions** — reading its identical accuracy
  as "more accurate" is a category error (LIMITATIONS §4); its measured benefit
  here is probability quality (NLL/Brier/ECE/MCE), not task accuracy.
* **Isotonic MCE instability is measured, not hypothetical**: 0.6667 (seed 0)
  and 0.2935 ± 0.3235 across 3 seeds — small calibration sets produce coarse
  step maps; the report states both directions of its decision changes.
* All ECE/MCE numbers are estimators with sampling error and binning
  dependence on n=872 (ABLATIONS.md honesty summary; LIMITATIONS §4/§10); no
  distribution-free calibration claim exists anywhere in this repository
  (MATHEMATICAL_VERIFICATION.md "What this does NOT verify" §4).
* Seed policy: EXP-1 = 3 seeds {0,1,2}; EXP-5 = single seed 0 — see
  [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) §3.
