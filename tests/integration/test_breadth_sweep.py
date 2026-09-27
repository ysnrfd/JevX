# Copyright (c) 2026 ysnrfd
"""Execute the full JevX decision path across the architecture-breadth sweep.

This is the *pipeline* breadth gate: for every family in
``benchmarks/run_breadth_sweep.FAMILIES`` a tiny random-weight model is built
in memory from the real ``transformers`` config class and run through

    ``JevXCapabilityDetector.detect`` -> ``resolve_adapter`` ->
    ``adapter.prepare_inputs`` -> ``adapter.extract_representation`` ->
    ``jevx.representations.pooling.pool`` -> ``JevXForDecision.choice(K=3)``

with every invariant asserted per family (pooled ``[B, D]`` and finite,
probabilities summing to 1 within 1e-5, ``argmax`` inside the schema,
confidence in ``[0, 1]``, ``state_proj`` width equal to the backbone hidden
width, and a recorded capability verdict + representation field per family).

**No downloads and no checkpoints**: every model is randomly initialized in
memory, so a PASS validates the PIPELINE, never a checkpoint's accuracy. The
structure assertions live in the same pass, so the file builds each tiny model
exactly ONCE -- measured runtime is dominated by model construction.

The sweep is the hermetic gate behind the breadth claim in
``reports/ARCHITECTURE_BREADTH.md`` and the "pipeline-executed" column of
``TRANSFORMERS_COMPATIBILITY_MATRIX.md``.

Reproduce outside pytest::

    python -m benchmarks.run_breadth_sweep --out benchmarks/results/breadth_sweep.json
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import pytest

from benchmarks.run_breadth_sweep import (
    EXCLUSIONS,
    FAMILIES,
    HIDDEN_WIDTH_BLOCKER,
    run_family,
)

#: Per-family wall-clock budget (seconds). ~5x the slowest measured family
#: (vits, dominated by its HiFi-GAN vocoder) so ordinary CPU contention on a
#: shared box does not trip it, while a regression that adds a heavy forward
#: does.
PER_FAMILY_BUDGET_S = 30.0

#: Families whose extraction ran in full while the DECISION step could not be
#: built: the CNN backbones declare a per-stage ``hidden_sizes`` LIST and
#: ``_HIDDEN_WIDTH_KEYS`` probed only scalar widths, so ``_hidden_width``
#: returned ``None``. That blocker is RESOLVED -- ``_hidden_width`` now has the
#: last-active-stage rule and the final-convolution rule
#: (``tests/unit/test_cnn_hidden_width.py``), and ``JevXForDecision`` runs the
#: full decision path on all four. The names are kept as an asserted
#: expectation: these four must now PASS (``test_breadth_sweep_end_to_end``
#: asserts it), so a regression that reintroduces the blocker fails loudly
#: instead of quietly returning the rows to PARTIAL.
RESOLVED_DECISION_BLOCK: Tuple[str, ...] = (
    "convnext",
    "resnet",
    "segformer",
    "mobilenet_v2",
)

#: Minimum number of families per architectural group, so a shrinking family
#: list is caught rather than silently narrowing the claim.
REQUIRED_GROUPS: Dict[str, int] = {
    "encoder": 16,
    "decoder": 21,
    "encoder_decoder": 9,
    "vision": 11,
    "audio": 7,
    "multimodal": 5,
}

#: Excluded families must be a small, explainable tail.
MAX_EXCLUSIONS = 12


def test_family_set_is_broad_enough() -> None:
    """The sweep still covers >= 40 architectures in every major group."""
    assert len(FAMILIES) >= 40, f"only {len(FAMILIES)} families in the sweep"
    counts: Dict[str, int] = {}
    for spec in FAMILIES:
        counts[spec.group] = counts.get(spec.group, 0) + 1
    for group, minimum in REQUIRED_GROUPS.items():
        assert counts.get(group, 0) >= minimum, (
            f"group {group!r} has {counts.get(group, 0)} families, expected >= {minimum}"
        )
    assert len({spec.key for spec in FAMILIES}) == len(FAMILIES), "duplicate family key"
    assert len(EXCLUSIONS) <= MAX_EXCLUSIONS, (
        f"{len(EXCLUSIONS)} exclusions is too many to be a tail; the target set "
        "should be executed, not excused"
    )


def test_every_exclusion_states_a_specific_reason() -> None:
    """Exclusions must name the checked source, not shrug."""
    assert EXCLUSIONS, "the exclusion list is part of the claim; it must not be empty"
    for entry in EXCLUSIONS:
        reason = entry["reason"]
        assert entry["family"], "every exclusion names a family"
        assert len(reason) > 40, f"{entry['family']}: reason too vague ({reason!r})"
        assert any(
            marker in reason
            for marker in ("modeling_", "configuration_", "auto_mappings",
                           "inventory", "measured")
        ), f"{entry['family']}: the reason must cite where it was checked"


def test_breadth_sweep_end_to_end() -> None:
    """Run the full sweep; every family must PASS (or be a known PARTIAL)."""
    started = time.perf_counter()
    rows: List[dict] = []
    for spec in FAMILIES:
        row = run_family(spec)
        rows.append(row)
        assert row["elapsed_s"] <= PER_FAMILY_BUDGET_S, (
            f"{spec.key} took {row['elapsed_s']}s, over the {PER_FAMILY_BUDGET_S}s "
            "per-family budget"
        )
        # No silent path: every executed family records a verdict, a named
        # representation field and a resolved adapter.
        assert row["status"] in ("PASS", "PARTIAL"), f"{spec.key}: {row}"
        assert row["verdict"] in ("supported", "experimental"), (
            f"{spec.key}: verdict {row['verdict']!r} (reasons: {row['capability_reasons']})"
        )
        assert row["adapter"], f"{spec.key}: no adapter resolved"
        assert row["representation_field"], (
            f"{spec.key}: no representation_field identified -- a silent fallback"
        )
        assert row["pooled_shape"] is not None and len(row["pooled_shape"]) == 2, (
            f"{spec.key}: pooled must be [B, D], got {row['pooled_shape']}"
        )
        assert row["pooled_finite"] is True, f"{spec.key}: non-finite pooled"
        assert row["hidden_width"] == row["pooled_shape"][-1], spec.key
    elapsed = time.perf_counter() - started

    failed = [r for r in rows if r["status"] == "FAIL"]
    partial = [r for r in rows if r["status"] == "PARTIAL"]
    unexpected_partial = [r for r in partial if r["family"] not in RESOLVED_DECISION_BLOCK]
    assert not failed, "sweep failures:\n" + "\n".join(
        f"  {r['family']}: {r['error']}" for r in failed
    )
    assert not unexpected_partial, (
        "families blocked at the decision step outside the documented set:\n"
        + "\n".join(f"  {r['family']}: {r['decision_blocked']}" for r in unexpected_partial)
    )
    # The four CNN rows are EXPECTED TO PASS now: the width blocker is closed,
    # so a PARTIAL here is a regression, not a known limitation. The decision
    # invariants below then run on them too (state_proj width == hidden width).
    regressed = [
        f for f in RESOLVED_DECISION_BLOCK
        if next((r["status"] for r in rows if r["family"] == f), None) != "PASS"
    ]
    assert not regressed, (
        "the CNN decision blocker is closed -- these must PASS, not PARTIAL:\n"
        + "\n".join(
            "  %s: %s" % (
                r["family"],
                r.get("decision_blocked") or r.get("error") or r["status"],
            )
            for r in rows
            if r["family"] in RESOLVED_DECISION_BLOCK
        )
    )
    for row in partial:
        assert HIDDEN_WIDTH_BLOCKER in row["decision_blocked"], (
            f"{row['family']} is PARTIAL for an undocumented reason: "
            f"{row['decision_blocked']}"
        )

    # Decision-path invariants, for every family that reached the wrapper.
    for row in rows:
        if row["status"] != "PASS":
            continue
        assert abs(row["prob_sum"] - 1.0) <= 1e-5, (
            f"{row['family']}: probabilities sum to {row['prob_sum']}"
        )
        assert 0 <= row["predicted_index"] < 3, row["family"]
        assert 0.0 <= row["confidence"] <= 1.0, row["family"]
        assert row["state_proj_width"] == row["hidden_width"], (
            f"{row['family']}: state_proj width {row['state_proj_width']} != "
            f"hidden {row['hidden_width']}"
        )
    slowest = max(rows, key=lambda r: r["elapsed_s"])
    print(
        f"\nBREADTH SWEEP | families={len(rows)} "
        f"pass={sum(1 for r in rows if r['status'] == 'PASS')} "
        f"partial={len(partial)} excluded={len(EXCLUSIONS)} "
        f"wall={elapsed:.1f}s slowest={slowest['family']}({slowest['elapsed_s']}s)"
    )
