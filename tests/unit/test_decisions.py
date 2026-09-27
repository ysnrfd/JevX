# Copyright (c) 2026 ysnrfd
"""Unit tests for the typed decision heads: shapes, masks, extremes,
gradients, registry dispatch and schema projection."""

from __future__ import annotations

import pytest
import torch

from jevx.decisions import (
    DECISION_HEAD_REGISTRY,
    ChoiceHead,
    DecisionHead,
    MultiLabelHead,
    NoulHead,
    RegressionHead,
    ScoreHead,
    available_decision_heads,
    build_decision_head,
    get_decision_head,
    mask_from_num_options,
    project_to_schema,
    register_decision_head,
)


# ---------------------------------------------------------------------------
# Forward shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("num_options", [1, 2, 5, 12])
@pytest.mark.parametrize("interaction", ["dot", "bilinear", "mlp"])
def test_choice_forward_shapes(batch, num_options, interaction):
    torch.manual_seed(batch * 100 + num_options)
    d = 6
    head = ChoiceHead(d, interaction=interaction)
    fused = torch.randn(batch, d)
    options = torch.randn(batch, num_options, d)
    out = head(fused, options)

    assert out.logits.shape == (batch, num_options)
    assert out.probabilities.shape == (batch, num_options)
    assert out.predicted_index.shape == (batch,)
    assert out.predicted_index.dtype == torch.long
    assert torch.isfinite(out.logits).all()
    assert (out.probabilities >= 0).all()
    assert torch.allclose(
        out.probabilities.sum(dim=-1),
        torch.ones(batch),
        atol=1e-6,
    )
    assert (out.predicted_index >= 0).all()
    assert (out.predicted_index < num_options).all()
    # logits of a plain (unmasked) forward carry no fill values
    assert out.logits.min() > torch.finfo(torch.float32).min / 2


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("num_levels", [1, 2, 5, 12])
def test_score_forward_shapes_fixed_levels(batch, num_levels):
    torch.manual_seed(batch + num_levels)
    d = 6
    head = ScoreHead(d, num_levels=num_levels)
    fused = torch.randn(batch, d)
    out = head(fused)

    assert out.logits.shape == (batch, num_levels)
    assert out.probabilities.shape == (batch, num_levels)
    assert out.predicted_index.shape == (batch,)
    assert out.expected_score.shape == (batch,)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(batch), atol=1e-6)
    assert (out.predicted_index >= 0).all() and (out.predicted_index < num_levels).all()


@pytest.mark.parametrize("batch", [1, 2, 8])
@pytest.mark.parametrize("num_levels", [1, 2, 5, 12])
def test_score_forward_shapes_with_options(batch, num_levels):
    torch.manual_seed(batch * 7 + num_levels)
    d = 6
    head = ScoreHead(d, interaction="dot")
    fused = torch.randn(batch, d)
    options = torch.randn(batch, num_levels, d)
    out = head(fused, options)
    assert out.logits.shape == (batch, num_levels)
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(batch), atol=1e-6)


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_noul_forward_shapes(batch):
    torch.manual_seed(batch)
    d = 6
    head = NoulHead(d)
    out = head(torch.randn(batch, d))
    assert out.logits.shape == (batch, 1)
    assert out.probabilities.shape == (batch, 1)
    assert out.predicted_index.shape == (batch,)
    assert (out.predicted_index <= 1).all()


# ---------------------------------------------------------------------------
# Variable K with masks
# ---------------------------------------------------------------------------


def _random_mask(batch: int, width: int, generator: torch.Generator) -> torch.Tensor:
    counts = torch.randint(1, width + 1, (batch,), generator=generator)
    return mask_from_num_options(counts, width)


def test_variable_k_mask_invalid_probs_exactly_zero():
    torch.manual_seed(0)
    b, kmax, d = 5, 7, 8
    head = ChoiceHead(d, interaction="bilinear")
    fused = torch.randn(b, d)
    options = torch.randn(b, kmax, d)
    gen = torch.Generator().manual_seed(123)
    mask = _random_mask(b, kmax, gen)

    out = head(fused, options, option_mask=mask)
    assert (out.probabilities[~mask] == 0.0).all(), "invalid probs must be exactly 0"
    sums = out.probabilities.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(b), atol=1e-6), "valid probs must sum to 1"
    # predicted index always addresses a valid option
    rows = torch.arange(b)
    assert mask[rows, out.predicted_index].all()


def test_variable_k_num_options_matches_mask():
    torch.manual_seed(0)
    b, kmax, d = 4, 6, 8
    head = ChoiceHead(d, interaction="dot")
    fused = torch.randn(b, d)
    options = torch.randn(b, kmax, d)
    counts = torch.tensor([1, 3, 6, 2])

    out_counts = head(fused, options, num_options=counts)
    out_mask = head(fused, options, option_mask=mask_from_num_options(counts, kmax))
    assert torch.equal(out_counts.probabilities, out_mask.probabilities)
    assert torch.equal(out_counts.predicted_index, out_mask.predicted_index)
    for i, c in enumerate(counts.tolist()):
        assert int(out_counts.predicted_index[i]) < c


def test_variable_k_score_with_mask_keeps_bounds():
    torch.manual_seed(0)
    b, kmax, d = 4, 5, 8
    head = ScoreHead(d, num_levels=kmax)
    fused = torch.randn(b, d)
    counts = torch.tensor([5, 1, 3, 4])
    out = head(fused, num_options=counts)
    mask = mask_from_num_options(counts, kmax)
    assert (out.probabilities[~mask] == 0.0).all()
    assert torch.allclose(out.probabilities.sum(-1), torch.ones(b), atol=1e-6)
    # expected score uses only valid levels: bounded by (valid_count - 1)
    for i, c in enumerate(counts.tolist()):
        s_i = float(out.expected_score[i].detach())
        assert 0.0 - 1e-5 <= s_i <= (c - 1) + 1e-5


# ---------------------------------------------------------------------------
# Extreme logits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("magnitude", [1e4, 1e6])
def test_extreme_logits_no_nan_valid_probs(magnitude):
    head = ScoreHead(4, num_levels=4)
    for row in (
        [magnitude, -magnitude, 0.0, 5.0],
        [-magnitude, -magnitude, magnitude, -magnitude],
        [0.0, 0.0, 0.0, 0.0],
    ):
        logits = torch.tensor([row], dtype=torch.float32)
        out = head.build_output(logits)
        assert torch.isfinite(out.probabilities).all()
        assert (out.probabilities >= 0).all()
        assert torch.allclose(out.probabilities.sum(-1), torch.ones(1), atol=1e-6)
        assert 0 <= int(out.predicted_index) < 4
        assert 0.0 - 1e-4 <= float(out.expected_score) <= 3.0 + 1e-4


@pytest.mark.parametrize("magnitude", [1e4, 1e6])
def test_extreme_logits_with_mask(magnitude):
    head = ScoreHead(4, num_levels=4)
    logits = torch.tensor([[magnitude, -magnitude, magnitude, -magnitude]])
    mask = torch.tensor([[True, False, True, False]])
    out = head.build_output(logits, mask)
    assert (out.probabilities[0, 1] == 0.0) and (out.probabilities[0, 3] == 0.0)
    assert torch.isfinite(out.probabilities).all()
    assert torch.allclose(
        out.probabilities[0, mask[0]].sum(), torch.tensor(1.0), atol=1e-6
    )
    assert int(out.predicted_index) in (0, 2)


def test_extreme_inputs_through_forward_stay_finite():
    torch.manual_seed(0)
    d = 8
    head = NoulHead(d)
    out = head(torch.randn(4, d) * 1e3)
    assert torch.isfinite(out.logits).all()
    assert (out.probabilities >= 0).all() and (out.probabilities <= 1).all()
    score_head = ScoreHead(d, num_levels=3)
    out2 = score_head(torch.randn(4, d) * 1e4)
    assert torch.isfinite(out2.probabilities).all()
    assert torch.allclose(out2.probabilities.sum(-1), torch.ones(4), atol=1e-6)


# ---------------------------------------------------------------------------
# Score bounds / Noul range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_levels", [1, 2, 5, 12])
def test_score_expected_value_bounds(num_levels):
    torch.manual_seed(num_levels)
    d = 6
    head = ScoreHead(d, num_levels=num_levels)
    for _ in range(5):
        fused = torch.randn(8, d) * float(torch.randint(1, 50, (1,)))
        out = head(fused)
        s = out.expected_score
        assert torch.isfinite(s).all()
        assert (s >= 0.0 - 1e-5).all()
        assert (s <= (num_levels - 1) + 1e-5).all()


def test_noul_range_and_predictions():
    torch.manual_seed(0)
    d = 6
    head = NoulHead(d)
    out = head(torch.randn(16, d))
    assert (out.probabilities > 0.0).all(), "sigmoid must be strictly positive at these scales"
    assert (out.probabilities < 1.0).all(), "sigmoid must be strictly below 1 at these scales"
    assert set(out.predicted_index.unique().tolist()) <= {0, 1}
    expected = (out.probabilities >= 0.5).long().reshape(-1)
    assert torch.equal(out.predicted_index, expected)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def test_gradients_flow_to_fused_and_options():
    torch.manual_seed(0)
    b, k, d = 3, 4, 8
    for interaction in ("dot", "bilinear", "mlp"):
        head = ChoiceHead(d, interaction=interaction)
        fused = torch.randn(b, d, requires_grad=True)
        options = torch.randn(b, k, d, requires_grad=True)
        out = head(fused, options)
        assert out.logits.requires_grad
        loss = out.logits.sum() + (out.probabilities * torch.arange(k)).sum()
        grads = torch.autograd.grad(loss, [fused, options], retain_graph=True)
        for g in grads:
            assert g is not None
            assert torch.isfinite(g).all()
            assert g.abs().sum() > 0
        if interaction != "dot":
            param_grads = torch.autograd.grad(
                loss, list(head.parameters()), allow_unused=True
            )
            for g in param_grads:
                assert g is not None and torch.isfinite(g).all()


def test_gradients_flow_for_score_and_noul():
    torch.manual_seed(0)
    d = 6
    for head in (ScoreHead(d, num_levels=5), NoulHead(d)):
        fused = torch.randn(4, d, requires_grad=True)
        out = head(fused)
        loss = out.logits.sum()
        (g,) = torch.autograd.grad(loss, [fused])
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_masked_positions_get_no_gradient():
    torch.manual_seed(0)
    d = 4
    head = ChoiceHead(d, interaction="dot")
    fused = torch.randn(2, d, requires_grad=True)
    options = torch.randn(2, 3, d, requires_grad=True)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    out = head(fused, options, option_mask=mask)
    # Weighted loss (plain sum of probs is the constant 1 and would give
    # vacuous zeros): gradients must be exactly zero at masked-out options.
    weights = torch.arange(3, dtype=out.probabilities.dtype)
    (g_options,) = torch.autograd.grad(
        (out.probabilities * weights).sum(), [options]
    )
    assert (g_options[~mask] == 0).all()
    assert g_options[mask].abs().sum() > 0
    assert torch.isfinite(g_options).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_dispatch_by_name():
    for name, cls in [
        ("choice", ChoiceHead),
        ("score", ScoreHead),
        ("noul", NoulHead),
        ("multi_label", MultiLabelHead),
        ("regression", RegressionHead),
    ]:
        assert name in available_decision_heads()
        assert get_decision_head(name) is cls
    assert build_decision_head("choice", hidden_size=4, interaction="dot").__class__ is ChoiceHead
    assert build_decision_head("score", hidden_size=4, num_levels=3).__class__ is ScoreHead
    assert build_decision_head("noul", hidden_size=4).__class__ is NoulHead
    assert isinstance(build_decision_head("choice", hidden_size=4), DecisionHead)


def test_unknown_registry_name_informative_error():
    with pytest.raises(ValueError, match="unknown decision head"):
        get_decision_head("nope")
    with pytest.raises(ValueError, match="choice"):
        get_decision_head("nope")  # message must list available names


def test_custom_head_registration_and_duplicate_error():
    name = "unit_test_probe_head"
    try:
        @register_decision_head(name)
        class ProbeHead(NoulHead):
            pass

        assert get_decision_head(name) is ProbeHead
        assert build_decision_head(name, hidden_size=3).__class__ is ProbeHead
        with pytest.raises(ValueError, match="already registered"):
            register_decision_head("choice")(ProbeHead)
    finally:
        DECISION_HEAD_REGISTRY.pop(name, None)


def test_interaction_name_validated():
    with pytest.raises(ValueError, match="interaction"):
        ChoiceHead(4, interaction="dragon")
    with pytest.raises(ValueError, match="hidden_size"):
        ChoiceHead(None, interaction="bilinear")


# ---------------------------------------------------------------------------
# Schema validation + projection
# ---------------------------------------------------------------------------


def test_choice_requires_option_features():
    head = ChoiceHead(4)
    with pytest.raises(ValueError, match="option_features"):
        head(torch.randn(2, 4))


def test_noul_rejects_options():
    head = NoulHead(4)
    with pytest.raises(ValueError, match="not applicable"):
        head(torch.randn(2, 4), torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="not applicable"):
        head(torch.randn(2, 4), option_mask=torch.ones(2, 1, dtype=torch.bool))


def test_score_requires_one_schema_source():
    head = ScoreHead(4)
    with pytest.raises(ValueError, match="num_levels"):
        head(torch.randn(2, 4))


def test_batch_alignment_error():
    head = ChoiceHead(4, interaction="dot")
    with pytest.raises(ValueError, match="batch size mismatch"):
        head(torch.randn(3, 4), torch.randn(2, 5, 4))


def test_hidden_dim_mismatch_error():
    head = ChoiceHead(4, interaction="dot")
    with pytest.raises(ValueError, match="hidden dim"):
        head(torch.randn(2, 4), torch.randn(2, 5, 7))


def test_num_options_out_of_range_errors():
    head = ChoiceHead(4, interaction="dot")
    options = torch.randn(2, 3, 4)
    fused = torch.randn(2, 4)
    with pytest.raises(ValueError, match="exceeds"):
        head(fused, options, num_options=torch.tensor([3, 4]))
    with pytest.raises(ValueError, match="needs >= 1"):
        head(fused, options, num_options=torch.tensor([3, 0]))


def test_all_masked_row_error():
    head = ChoiceHead(4, interaction="dot")
    options = torch.randn(2, 3, 4)
    fused = torch.randn(2, 4)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    with pytest.raises(ValueError, match="no valid option"):
        head(fused, options, option_mask=mask)


def test_mask_count_disagreement_error():
    head = ChoiceHead(4, interaction="dot")
    options = torch.randn(2, 3, 4)
    fused = torch.randn(2, 4)
    mask = torch.tensor([[True, False, False], [True, True, False]])
    with pytest.raises(ValueError, match="disagrees"):
        head(fused, options, option_mask=mask, num_options=torch.tensor([2, 2]))


def test_mask_dtype_validated():
    head = ChoiceHead(4, interaction="dot")
    options = torch.randn(2, 3, 4)
    fused = torch.randn(2, 4)
    with pytest.raises(TypeError, match="torch.bool"):
        head(fused, options, option_mask=torch.ones(2, 3, dtype=torch.uint8))


def test_nonfinite_inputs_rejected():
    head = ChoiceHead(4, interaction="dot")
    fused = torch.randn(2, 4)
    fused[0, 1] = float("nan")
    options = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="fused"):
        head(fused, options)
    with pytest.raises(ValueError, match="NaN"):
        head(fused, options)


def test_project_to_schema_valid_and_invalid():
    keys = ["a", "b", "c", "d", "e", "f"]
    idx = torch.tensor([0, 5, 2])
    assert project_to_schema(idx, keys) == ["a", "f", "c"]
    assert project_to_schema(torch.tensor(1), ["x", "y"]) == ["y"]
    assert project_to_schema(0, ["only"]) == ["only"]
    with pytest.raises(ValueError, match="not in the decision schema"):
        project_to_schema(torch.tensor([9]), keys)
    with pytest.raises(ValueError, match="not in the decision schema"):
        project_to_schema(torch.tensor([-1]), keys)
    with pytest.raises(ValueError, match="empty"):
        project_to_schema(torch.tensor([0]), [])


def test_head_project_method():
    head = ChoiceHead(4, interaction="dot")
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    indices = head.project(probs)
    assert torch.equal(indices, torch.tensor([1, 0]))
    selected = head.project(probs, option_keys=["x", "y", "z"])
    assert selected == ["y", "x"]
    with pytest.raises(ValueError, match="not in the decision schema"):
        head.project(probs, option_keys=["only-one"])


# ---------------------------------------------------------------------------
# Generic (experimental) heads
# ---------------------------------------------------------------------------


def test_multilabel_head_smoke():
    torch.manual_seed(0)
    head = MultiLabelHead(4, num_labels=3)
    fused = torch.randn(5, 4)
    mask = torch.tensor(
        [[True, True, True]] * 4 + [[True, True, False]], dtype=torch.bool
    )
    out = head(fused, option_mask=mask)
    assert out.logits.shape == (5, 3)
    assert out.probabilities.shape == (5, 3)
    assert (out.probabilities[~mask] == 0.0).all()
    assert out.predicted_value.shape == (5, 3)
    assert out.predicted_value.dtype == torch.bool
    assert out.predicted_index is None
    with pytest.raises(ValueError, match="option_features"):
        head(fused, torch.randn(5, 3, 4))


def test_regression_head_smoke():
    torch.manual_seed(0)
    head = RegressionHead(4)
    fused = torch.randn(5, 4)
    out = head(fused)
    assert out.logits.shape == (5, 1)
    assert out.probabilities is None
    assert out.predicted_index is None
    assert torch.equal(out.predicted_value, out.logits.squeeze(-1))
    with pytest.raises(ValueError, match="not applicable"):
        head(fused, option_mask=torch.ones(5, 2, dtype=torch.bool))
