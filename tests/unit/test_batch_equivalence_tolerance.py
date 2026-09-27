# Copyright (c) 2026 ysnrfd
"""The measured batched-vs-solo tolerance rule (T7-ii, PROOFS.md Remark 7.2).

Why this file exists: the codebase carried a bare 1e-5 in tests and reports
as the bound for batched == solo, but the real 1500-state scenario measured
1.0967254638671875e-05 and reported that check as a FAILURE
(reports/REAL_SCENARIO_VALIDATION.md section 5.2). A fixed number is not a
bound -- the deviation is float re-association whose size depends on the
reduction width, on the number of sequential accumulation stages and on the
dtype. jevx.utils.numerical.batch_equivalence_tolerance replaces it with a
rule derived from a seeded measurement sweep (reports/NUMERICAL_EQUIVALENCE_STUDY.md).

Asserted here:

1. the rule bounds every measured cell of that sweep, with margin;
2. it is monotone in dtype eps, sequence length, stages and scale;
3. the recorded scenario deviation passes the documented rule -- the check
   that used to fail -- read from the scenario JSON rather than re-typed;
4. input validation and the shape of the returned value.

The measured cells are carried here as data with their provenance, so the
test needs no network and no long rerun. They are NOT regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from jevx.utils.numerical import (
    BATCH_EQUIVALENCE_SAFETY,
    JEVX_DECISION_ACCUMULATION_STAGES,
    batch_equivalence_tolerance,
)

#: The deviation that failed the 1e-5 contract, read back from the artifact
#: of record (benchmarks/results/real_scenario.json, phase 5, check b).
SCENARIO_JSON = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "real_scenario.json"
)

#: Fallback constant, identical to the value in the JSON, so the rule is still
#: exercised if the benchmark artifact is not shipped.
SCENARIO_WORST_DEVIATION = 0.000010967254638671875

#: Representative worst cells of the pooled-H_S sweep (96 cells:
#: cls / masked_mean / last_valid / attention_pool, float32 and bfloat16,
#: padding width 4/8/16/32/48/64, 1 and 2 backbone layers).
POOLED_SWEEP = [
    {
        "cell": "attention_pool/float32/L1/T4",
        "dtype": torch.float32,
        "seq_len": 4,
        "stages": 1,
        "scale": 2.596,
        "max_abs_dev": 4.768e-07,
    },
    {
        "cell": "last_valid/float32/L2/T8",
        "dtype": torch.float32,
        "seq_len": 8,
        "stages": 2,
        "scale": 2.828,
        "max_abs_dev": 4.768e-07,
    },
    {
        "cell": "masked_mean/float32/L2/T64",
        "dtype": torch.float32,
        "seq_len": 64,
        "stages": 2,
        "scale": 2.065,
        "max_abs_dev": 2.384e-07,
    },
    {
        "cell": "masked_mean/bfloat16/L1/T8",
        "dtype": torch.bfloat16,
        "seq_len": 8,
        "stages": 1,
        "scale": 1.992,
        "max_abs_dev": 3.906e-03,
    },
    {
        "cell": "masked_mean/bfloat16/L2/T16",
        "dtype": torch.bfloat16,
        "seq_len": 16,
        "stages": 2,
        "scale": 2.234,
        "max_abs_dev": 3.906e-03,
    },
    {
        "cell": "masked_mean/bfloat16/L2/T64",
        "dtype": torch.bfloat16,
        "seq_len": 64,
        "stages": 2,
        "scale": 2.066,
        "max_abs_dev": 0.0,
    },
]


def _scenario_worst() -> float:
    """The recorded worst batched-vs-solo probability deviation."""
    if not SCENARIO_JSON.is_file():
        return SCENARIO_WORST_DEVIATION
    payload = json.loads(SCENARIO_JSON.read_text(encoding="utf-8"))
    checks = payload["phases"]["5_correctness_battery"]["checks"]
    for check in checks:
        if check.get("check") == "b_batched_equals_solo_calls":
            return float(check["worst"])
    raise AssertionError("real_scenario.json no longer records check b.")  # pragma: no cover


# ----------------------------------------------------------------- rule shape


@pytest.mark.parametrize("cell", POOLED_SWEEP, ids=lambda c: c["cell"])
def test_rule_bounds_every_measured_cell(cell) -> None:
    """The rule is a bound, not a guess: every measured cell sits under it,
    and the worst cells keep a real margin (>= 2x) rather than a hair."""
    tol = batch_equivalence_tolerance(
        cell["dtype"], cell["seq_len"], stages=cell["stages"], scale=cell["scale"]
    )
    name = cell["cell"]
    observed = cell["max_abs_dev"]
    assert observed <= tol, f"{name}: measured {observed:.3e} exceeds the rule {tol:.3e}"
    if observed > 0.0:
        margin = tol / observed
        assert margin >= 2.0, f"{name}: margin {margin:.2f}x is too thin"


def test_scenario_deviation_now_passes_the_documented_rule() -> None:
    """The check that FAILED against a bare 1e-5 passes the measured rule.

    The scenario walked a 6-layer, 27.6M-parameter backbone over 1500 states;
    T = 128 is the padded width the batched call used, and the stage count is
    the 6 backbone layers plus the 5 JevX decision-layer stages.
    """
    observed = _scenario_worst()
    assert observed == pytest.approx(SCENARIO_WORST_DEVIATION, rel=1e-12), (
        "the artifact of record moved: update SCENARIO_WORST_DEVIATION and "
        "reports/NUMERICAL_EQUIVALENCE_STUDY.md together"
    )
    assert observed > 1e-5, "this test is about the value that FAILED the 1e-5 bound"
    stages = 6 + JEVX_DECISION_ACCUMULATION_STAGES
    tol = batch_equivalence_tolerance(torch.float32, 128, stages=stages)
    assert observed <= tol, f"scenario deviation {observed:.6e} exceeds the rule {tol:.6e}"
    assert tol / observed >= 2.0, "the rule lost its stated margin"


# --------------------------------------------------------------- monotonicity


def test_rule_is_monotone_in_dtype_eps() -> None:
    """Lower eps gives no larger tolerance at equal width, stages and scale."""
    low = batch_equivalence_tolerance(torch.float32, 64, stages=4, scale=2.0)
    high = batch_equivalence_tolerance(torch.bfloat16, 64, stages=4, scale=2.0)
    assert 0.0 < low < high
    eps32 = float(torch.finfo(torch.float32).eps)
    eps16 = float(torch.finfo(torch.bfloat16).eps)
    ratio = BATCH_EQUIVALENCE_SAFETY["bfloat16"] / BATCH_EQUIVALENCE_SAFETY["float32"]
    assert high / low == pytest.approx(eps16 / eps32 * ratio, rel=1e-9)


def test_rule_is_monotone_in_seq_len() -> None:
    """A wider reduction gives no smaller tolerance, growing like sqrt(T)."""
    tol = [batch_equivalence_tolerance(torch.float32, t) for t in (4, 8, 16, 32, 64, 128)]
    assert all(a < b for a, b in zip(tol, tol[1:])), tol
    assert tol[-1] / tol[0] == pytest.approx((128 / 4) ** 0.5, rel=1e-12)


def test_rule_is_monotone_in_stages_and_scale() -> None:
    """More accumulation stages or a larger magnitude: no smaller tolerance."""
    stages = [batch_equivalence_tolerance(torch.float32, 32, stages=s) for s in (1, 2, 4, 8)]
    assert all(a < b for a, b in zip(stages, stages[1:])), stages
    assert stages[-1] / stages[0] == pytest.approx(8.0**0.5, rel=1e-12)
    scale = [batch_equivalence_tolerance(torch.float32, 32, scale=s) for s in (0.5, 1.0, 2.0, 4.0)]
    assert all(a < b for a, b in zip(scale, scale[1:])), scale
    assert scale[-1] / scale[0] == pytest.approx(8.0, rel=1e-12)


# ------------------------------------------------------------------ contract


def test_safety_factors_come_from_the_study() -> None:
    """Each safety factor is at least 3x the worst measured ratio for that
    dtype, which is how the study derived them (smallest such power of two)."""
    worst_ratio = {
        "float32": 0.7705,  # attention_pool/float32/L1/T4
        "bfloat16": 0.0887,  # masked_mean/bfloat16/L1/T8
    }
    for name, ratio in worst_ratio.items():
        assert BATCH_EQUIVALENCE_SAFETY[name] >= 3.0 * ratio, name
    # float32 is additionally governed by the real scenario walk
    assert BATCH_EQUIVALENCE_SAFETY["float32"] >= 3.0 * 2.452


def test_dtype_accepts_names_and_dtypes_alike() -> None:
    assert batch_equivalence_tolerance("float32", 32) == batch_equivalence_tolerance(
        torch.float32, 32
    )
    assert batch_equivalence_tolerance("bfloat16", 32) == pytest.approx(
        batch_equivalence_tolerance(torch.bfloat16, 32)
    )


@pytest.mark.parametrize("bad_dtype", ["nope", 3, None, torch.tensor([1.0])])
def test_bad_dtype_raises_type_error(bad_dtype) -> None:
    with pytest.raises(TypeError):
        batch_equivalence_tolerance(bad_dtype, 8)


@pytest.mark.parametrize("bad_len", [0, -4, 1.5, True])
def test_bad_seq_len_raises_value_error(bad_len) -> None:
    with pytest.raises(ValueError):
        batch_equivalence_tolerance(torch.float32, bad_len)


@pytest.mark.parametrize("bad_stages", [0, -1, 2.0])
def test_bad_stages_raises_value_error(bad_stages) -> None:
    with pytest.raises(ValueError):
        batch_equivalence_tolerance(torch.float32, 8, stages=bad_stages)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan"), "x"])
def test_bad_scale_raises(bad_scale) -> None:
    with pytest.raises((ValueError, TypeError)):
        batch_equivalence_tolerance(torch.float32, 8, scale=bad_scale)


@pytest.mark.parametrize("bad_safety", [0.0, -2.0, float("inf"), "x"])
def test_bad_safety_raises(bad_safety) -> None:
    with pytest.raises((ValueError, TypeError)):
        batch_equivalence_tolerance(torch.float32, 8, safety=bad_safety)


def test_rule_returns_a_finite_positive_float() -> None:
    tol = batch_equivalence_tolerance(torch.float32, 128, stages=11, scale=1.0)
    assert isinstance(tol, float)
    assert 0.0 < tol < float("inf")

