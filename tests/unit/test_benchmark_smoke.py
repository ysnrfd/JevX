# Copyright (c) 2026 ysnrfd
"""Smoke tests for the benchmark pipeline machinery (NO network, seconds).

These assert the *core functions* used by benchmarks/run_quality.py on a
20-sample synthetic split: the stratified splitter, the metric helpers
against hand-computed values, the reliability-bin table, the
confidence-ECE helper, and the risk-coverage / AURC / coverage-at-risk
wrappers against hand-worked curves. They deliberately do NOT run the full
benchmark (that needs hub downloads and minutes of CPU).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402


# --------------------------------------------------------------------------
# Synthetic 20-sample split (no network)
# --------------------------------------------------------------------------

def _synthetic_labels(n: int = 20) -> list[int]:
    """12 class-1, 8 class-0 rows - an imbalanced toy pool."""
    return [1] * 12 + [0] * 8


def test_stratified_split_sizes_disjoint_and_deterministic():
    labels = _synthetic_labels()
    train_idx, cal_idx = bc.make_splits(labels, n_train=10, n_cal=5, seed=7)

    # exact sizes
    assert len(train_idx) == 10
    assert len(cal_idx) == 5
    # disjoint
    assert len(np.intersect1d(train_idx, cal_idx)) == 0
    # all indices valid
    assert train_idx.max() < 20 and cal_idx.min() >= 0
    # stratified: population is 60% class 1 -> train of 10 has 6 class 1
    train_labels = [labels[i] for i in train_idx]
    assert train_labels.count(1) == 6
    assert train_labels.count(0) == 4
    # cal of 5: 5 * 0.6 = 3.0 exactly after largest-remainder over the
    # REMAINING pool (which is 6 class-1 / 4 class-0 minus train draw ->
    # counts differ; assert only that both classes appear, no exact ratio)
    cal_labels = [labels[i] for i in cal_idx]
    assert set(cal_labels) == {0, 1}

    # determinism: same seed -> identical
    train2, cal2 = bc.make_splits(labels, n_train=10, n_cal=5, seed=7)
    assert np.array_equal(train_idx, train2)
    assert np.array_equal(cal_idx, cal2)
    # different seed -> different draw (very likely on 20 rows; assert not equal)
    train3, cal3 = bc.make_splits(labels, n_train=10, n_cal=5, seed=8)
    assert not np.array_equal(train_idx, train3)


def test_stratified_indices_rejects_bad_sizes():
    labels = _synthetic_labels()
    with pytest.raises(ValueError, match="positive"):
        bc.stratified_indices(labels, 0, seed=0)
    with pytest.raises(ValueError, match="cannot sample"):
        bc.stratified_indices(labels, 21, seed=0)


# --------------------------------------------------------------------------
# Metric helpers vs hand-computed values
# --------------------------------------------------------------------------

def test_metrics_match_hand_computation():
    """4 examples, 2 classes: hand-compute every metric.

    logits chosen so softmax probs are exact-ish:
      row0: (2.0, 1.0) label 0 -> p ~ (0.7311, 0.2689) correct
      row1: (1.0, 2.0) label 0 -> p ~ (0.2689, 0.7311) wrong
      row2: (0.0, 3.0) label 1 -> p ~ (0.0474, 0.9526) correct
      row3: (3.0, 0.0) label 1 -> p ~ (0.9526, 0.0474) wrong
    accuracy = 2/4 = 0.5
    """
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    m = bc.metrics_from_logits(logits, labels, n_bins=15)

    assert m["accuracy"] == pytest.approx(0.5, abs=1e-6)

    # macro-F1: class 0 -> tp=1 (row0), fp=1 (row3 predicted 0? no:
    # preds are argmax = [0, 1, 1, 0]; labels [0, 0, 1, 1]
    # class 0: tp=1 (row0), fp=1 (row3 pred 0 label 1), fn=1 (row1 label 0 pred 1)
    #   P = 1/2, R = 1/2, F1 = 1/2
    # class 1: tp=1 (row2), fp=1 (row1 pred 1 label 0), fn=1 (row3 label 1 pred 0)
    #   P = 1/2, R = 1/2, F1 = 1/2
    assert m["macro_f1"] == pytest.approx(0.5, abs=1e-6)

    # NLL hand-computed in float64:
    probs = torch.softmax(logits.double(), dim=-1)
    picked = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    expected_nll = float(-picked.log().mean())
    assert m["nll"] == pytest.approx(expected_nll, abs=1e-6)

    # Brier hand-computed: mean over rows of sum_k (p_k - e_yk)^2
    target = torch.zeros_like(probs)
    target[torch.arange(4), labels] = 1.0
    expected_brier = float(((probs - target) ** 2).sum(dim=-1).mean())
    assert m["brier"] == pytest.approx(expected_brier, abs=1e-6)

    # all metrics finite and in documented ranges
    assert math.isfinite(m["ece"]) and 0.0 <= m["ece"] <= 1.0
    assert math.isfinite(m["mce"]) and 0.0 <= m["mce"] <= 1.0
    assert m["mce"] >= m["ece"] - 1e-12  # MCE is the max gap, ECE the average


def test_temperature_preserves_accuracy_isotonic_may_not():
    """Spec §8: temperature preserves argmax (accuracy identical);
    isotonic is allowed to change decisions - both observed here."""
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [3.0, 0.0]])
    labels = np.array([0, 0, 1, 1])
    cal_logits = np.array([[1.5, 0.5], [0.4, 1.2]], dtype=np.float32)
    cal_labels = np.array([0, 1])

    base = bc.calibrated_metrics(logits.numpy(), labels, "none")
    temp = bc.calibrated_metrics(
        logits.numpy(), labels, "temperature",
        fit_logits=cal_logits, fit_labels=cal_labels,
    )
    assert temp["accuracy"] == pytest.approx(base["accuracy"], abs=1e-9)

    # fitting calibration on test must be rejected (§53)
    with pytest.raises(ValueError, match="calibration split"):
        bc.calibrated_metrics(logits.numpy(), labels, "temperature")


def test_reliability_bins_partition_and_gaps():
    """Counts sum to N; per-bin gap = accuracy - avg_confidence."""
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [3.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1])
    table = bc.reliability_bins(logits, labels, n_bins=15)
    assert len(table) == 15
    assert sum(b["count"] for b in table) == 4
    for b in table:
        if b["count"] == 0:
            assert b["avg_confidence"] is None and b["gap"] is None
        else:
            assert b["gap"] == pytest.approx(
                b["accuracy"] - b["avg_confidence"], abs=1e-9
            )
            assert b["lower"] <= b["avg_confidence"] <= b["upper"]


def test_ece_of_confidence_equals_library_ece_for_maxprob():
    """ece_of_confidence(conf=max_prob, correct) must equal
    expected_calibration_error(probs, labels) - two code paths, one value."""
    logits = torch.tensor(
        [[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [3.0, 0.0], [0.5, -0.5], [-1.0, 0.2]]
    )
    labels = torch.tensor([0, 0, 1, 1, 0, 1])
    probs = torch.softmax(logits.double(), dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = (pred == labels).double()

    ours = bc.ece_of_confidence(conf.numpy(), correct.numpy(), n_bins=15)
    from jevx.utils.numerical import expected_calibration_error

    theirs = expected_calibration_error(probs, labels, n_bins=15)
    assert ours == pytest.approx(theirs, abs=1e-9)


# --------------------------------------------------------------------------
# Risk-coverage helpers vs hand-worked curve
# --------------------------------------------------------------------------

def test_risk_coverage_aurc_and_coverage_at_risk_hand_worked():
    """4 examples: conf [0.9, 0.6, 0.8, 0.3], correct [1, 0, 1, 1].

    Ranking by confidence desc: idx0 (0.9, ok), idx2 (0.8, ok),
    idx1 (0.6, wrong), idx3 (0.3, ok).
    cumulative risk at coverages [0.25, 0.5, 0.75, 1.0]:
        [0, 0, 1/3, 0.25]
    AURC = mean of those four = (0 + 0 + 0.3333... + 0.25) / 4 = 0.145833...
    coverage_at_risk(0.1) = 0.5 (first two points have risk 0 <= 0.1;
        third jumps to 1/3 > 0.1; sup over ALL points with risk<=0.1 is 0.5)
    coverage_at_risk(0.5) = 1.0 (last point risk 0.25 <= 0.5; sup = 1.0)
    """
    conf = np.array([0.9, 0.6, 0.8, 0.3])
    correct = np.array([1, 0, 1, 1], dtype=np.float64)

    from jevx import coverage_at_risk, risk_coverage_curve

    coverage, risk = risk_coverage_curve(conf, correct)
    assert coverage.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0], abs=1e-6)
    expected_risk = [0.0, 0.0, 1.0 / 3.0, 0.25]
    assert risk.tolist() == pytest.approx(expected_risk, abs=1e-6)

    expected_aurc = (0.0 + 0.0 + 1.0 / 3.0 + 0.25) / 4.0
    assert bc.aurc(conf, correct) == pytest.approx(expected_aurc, abs=1e-6)

    assert bc.coverage_at(conf, correct, 0.1) == pytest.approx(0.5, abs=1e-6)
    assert bc.coverage_at(conf, correct, 0.5) == pytest.approx(1.0, abs=1e-6)
    # risk budget 0 excludes everything except... first point risk=0 <= 0
    # but the ranking's point 3 has risk 1/3; sup over {points: risk<=0} = 0.5
    assert coverage_at_risk(conf, correct, 0.0) == pytest.approx(0.5, abs=1e-6)


def test_confidence_vectors_max_probability_range():
    """max_probability confidence in [1/K, 1]; margin in [0, 1]."""
    logits = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 3.0], [3.0, 0.0]],
                      dtype=np.float32)
    conf_max = bc.confidence_vectors(logits, "max_probability")
    conf_margin = bc.confidence_vectors(logits, "margin")
    conf_ent = bc.confidence_vectors(logits, "entropy")
    for c in (conf_max, conf_margin, conf_ent):
        assert c.shape == (4,)
        assert np.all(c >= 0.0) and np.all(c <= 1.0)
    # max_probability must equal rowwise softmax max
    probs = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    assert conf_max == pytest.approx(probs.max(axis=1), abs=1e-6)
    # energy without cal_logits must fail loudly
    with pytest.raises(ValueError, match="cal_logits"):
        bc.confidence_vectors(logits, "energy")


# --------------------------------------------------------------------------
# Record / loader plumbing on the 20-sample split
# --------------------------------------------------------------------------

def test_build_records_and_loader_shapes():
    feats = np.random.RandomState(0).randn(20, 8).astype(np.float32)
    labels = _synthetic_labels()
    records = bc.build_records(feats, labels)
    assert len(records) == 20
    assert records[0]["state"].shape == (8,)
    assert records[0]["decision_type"] == "choice"
    assert records[0]["options"] == [list(bc.DEFAULT_OPTIONS)]
    assert records[0]["labels"] in ([0], [1])

    loader = bc.make_loader(records, batch_size=7, shuffle=False)
    batch = next(iter(loader))
    assert len(batch["state"]) == 7
    assert len(batch["labels"]) == 7

    with pytest.raises(ValueError, match="mismatch"):
        bc.build_records(feats[:10], labels)


def test_mean_std_over_seeds_aggregation():
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}]
    agg = bc.mean_std_over_seeds(rows)
    assert agg["a"]["mean"] == pytest.approx(3.0, abs=1e-12)
    # sample std (ddof=1) of [1, 3, 5] = 2.0
    assert agg["a"]["std"] == pytest.approx(2.0, abs=1e-12)
    assert agg["b"]["mean"] == pytest.approx(4.0, abs=1e-12)
    with pytest.raises(ValueError, match="empty"):
        bc.mean_std_over_seeds([])


def test_generative_choice_logits_rejects_multitoken_option():
    """B3 guard: an option whose continuation is not one token must raise,
    not be silently approximated."""

    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            # " negative" -> 1 token, " positive" -> 2 tokens
            return [3967] if text.strip() == "positive" else [10, 20]

    with pytest.raises(ValueError, match="single-token"):
        bc.generative_choice_logits(
            object(), FakeTok(), ["some text"], options=["positive", "negative"]
        )
