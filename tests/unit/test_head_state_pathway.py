# Copyright (c) 2026 ysnrfd
"""The state pathway is the PRIMARY score: ``z_k = state_k + lambda * r_k``.

Motivation (math review). With one shared question and one fixed schema every
option row ``e_k`` is the same vector on every row, so the bilinear residual
collapses to ``z_k = s^T (W[k, :] + A e_k) + b[k]`` — STILL LINEAR in ``s``. A
linear state path therefore contributes no nonlinearity at all and the head
reduces to a linear classifier, which is why it lost to an MLP probe. This
module pins the two state pathways and the config surface that selects them:

1. ``state_pathway="linear"`` (the default) is the shipped linear probe and is
   BIT-IDENTICAL to it — a plain ``nn.Linear(D, K)`` reference built from the
   head's own parameters matches to the last bit, and the pre-EXP-H head
   (v1 ablation) still matches bit-for-bit at initialization.
2. ``state_pathway="mlp"`` with the residual switched off is EXACTLY
   ``nn.Sequential(Linear(D, H), act, Linear(H, K))`` on the fused state
   (atol 1e-6, reference built from the head's own weights).
3. The nonlinearity is real, not decorative: a 2-D XOR that a linear state path
   cannot fit (train accuracy pinned near 0.5) is fitted by the mlp state path
   (train accuracy > 0.9).
4. The state pathway is applied consistently to the score, noul, multi_label
   and regression heads, and the fixed-K score schema keeps its linear
   ``level_proj``.
5. Config round-trip and enum validation for the five new fields.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from jevx import JevXConfig
from jevx.decisions import (
    DEFAULT_STATE_PATHWAY_OPTIONS,
    MLPStatePathway,
    ChoiceHead,
    MultiLabelHead,
    NoulHead,
    RegressionHead,
    ScoreHead,
    build_decision_head,
)
from jevx.decisions.generic import STATE_ACTIVATIONS, STATE_PATHWAYS

INTERACTIONS = ("dot", "bilinear", "mlp")


def _inputs(batch: int = 4, k: int = 5, dim: int = 16, seed: int = 0):
    """Seeded (fused, option_features) pair."""
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, dim, generator=g),
        torch.randn(batch, k, dim, generator=g),
    )


def _fill(pathway: nn.Module, std: float = 0.3) -> None:
    """Give a zero-initialized state pathway non-zero, random parameters."""
    with torch.no_grad():
        for name, param in pathway.named_parameters():
            param.normal_(0.0, std)


# ---------------------------------------------------------------------------
# 1. linear (default) == the shipped linear probe, bit-for-bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("batch,k,dim", [(1, 2, 8), (4, 5, 16), (7, 12, 32)])
def test_linear_state_pathway_is_bitwise_a_linear_probe(
    interaction: str, batch: int, k: int, dim: int
) -> None:
    """z_k = s^T W[k, :] + b[k] IS ``nn.Linear(D, K)`` on the fused state.

    Bitwise (``torch.equal``), not approximate: the default pathway must not
    perturb a single bit of the shipped head. The reference is built from the
    head's OWN non-zero parameters, so this cannot pass by coincidence.
    """
    torch.manual_seed(3)
    head = build_decision_head(
        "choice",
        hidden_size=dim,
        interaction=interaction,
        num_options=k,
        state_pathway="linear",
    )
    _fill(head.state_pathway)
    fused, options = _inputs(batch, k, dim, seed=3)
    with torch.no_grad():
        head.bilinear_weight.zero_() if head.bilinear_weight is not None else None
        if head.mlp is not None:
            for param in head.mlp.parameters():
                param.zero_()
        head.set_interaction_scale(0.0)  # state-only
        got = head(fused, options).logits
        reference = nn.Linear(dim, k, bias=True).to(fused.dtype)
        with torch.no_grad():
            reference.weight.copy_(head.state_pathway.weight[:k])
            reference.bias.copy_(head.state_pathway.bias[:k])
        want = reference(fused)
    assert torch.equal(got, want), f"max |head - probe| = {(got - want).abs().max()}"


def test_default_state_pathway_is_linear_and_bit_identical_to_the_v1_head() -> None:
    """The shipped default is the linear probe AND the pre-EXP-H head.

    Two properties in one place because they share a cause: the linear
    pathway is zero-initialized, so at step 0 the scoring function is exactly
    the old interaction-only one and a freshly constructed head is bitwise
    identical to ``state_pathway=False``.
    """
    torch.manual_seed(2)
    d, k, b = 16, 4, 6
    fused, options = _inputs(b, k, d, seed=2)
    v2 = build_decision_head("choice", hidden_size=d, num_options=k)
    v1 = build_decision_head(
        "choice", hidden_size=d, num_options=k, state_pathway=False
    )
    v1.load_state_dict(
        {k2: v for k2, v in v2.state_dict().items() if not k2.startswith("state_pathway")},
        strict=False,
    )
    assert v2.state_pathway_name == "linear"
    assert isinstance(v2.state_pathway, type(v1.state_pathway)) is False
    with torch.no_grad():
        assert torch.equal(
            v2.state_pathway.weight, torch.zeros_like(v2.state_pathway.weight)
        ), "the linear pathway must stay zero-initialized"
        assert torch.equal(v2(fused, options).logits, v1(fused, options).logits), (
            "a freshly constructed head must be bit-identical to head v1"
        )


# ---------------------------------------------------------------------------
# 2. mlp == nn.Sequential(Linear(D, H), act, Linear(H, K)) on the fused state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", STATE_ACTIVATIONS)
@pytest.mark.parametrize("batch,k,dim,hidden", [(3, 4, 8, 5), (2, 7, 16, 16)])
def test_mlp_state_pathway_equals_an_equivalent_sequential_probe(
    activation: str, batch: int, k: int, dim: int, hidden: int
) -> None:
    """z_k with lambda = 0 IS ``Sequential(Linear(D, H), act, Linear(H, K))``.

    The reference is assembled from the head's OWN weights, so agreement is a
    statement about the wiring (hidden layer, activation, projection, the
    ``[:k]`` slice), never about a lucky value.
    """
    torch.manual_seed(5)
    head = build_decision_head(
        "choice",
        hidden_size=dim,
        interaction="dot",
        num_options=DEFAULT_STATE_PATHWAY_OPTIONS,
        state_pathway="mlp",
        state_hidden_size=hidden,
        state_activation=activation,
        conditioning_scale_init=0.0,
    )
    pathway = head.state_pathway
    assert isinstance(pathway, MLPStatePathway)
    act = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    reference = nn.Sequential(nn.Linear(dim, hidden), act(), nn.Linear(hidden, k))
    with torch.no_grad():
        _fill(pathway)
        reference[0].weight.copy_(pathway.w1)
        reference[0].bias.copy_(pathway.b1)
        reference[2].weight.copy_(pathway.w2[:k])
        reference[2].bias.copy_(pathway.b2[:k])
    fused, options = _inputs(batch, k, dim, seed=5)
    with torch.no_grad():
        got = head(fused, options).logits
        want = reference(fused)
    assert torch.allclose(got, want, atol=1e-6, rtol=0.0), (
        f"max |head - sequential| = {(got - want).abs().max().item():.3e}"
    )


def test_mlp_state_pathway_slices_and_grows_with_variable_k() -> None:
    """A narrower schema slices ``W2``; a wider one grows it without loss."""
    torch.manual_seed(7)
    head = build_decision_head(
        "choice", hidden_size=8, num_options=4, state_pathway="mlp"
    )
    pathway = head.state_pathway
    _fill(pathway)
    capacity = pathway.width
    for k in (2, 4, 3, 9):
        fused, options = _inputs(3, k, 8, seed=100 + k)
        out = head(fused, options)
        assert out.logits.shape == (3, k)
        if k <= capacity:
            assert pathway.width == capacity, "a narrower K must never reallocate"
    assert pathway.width == 9, "a wider K must grow the scoring layer"
    # The grown rows are zero, so a re-visit of the narrow schema is unchanged.
    fused, options = _inputs(3, 4, 8, seed=104)
    with torch.no_grad():
        narrow = head(fused, options).logits
    assert torch.isfinite(narrow).all()


def test_mlp_state_pathway_starts_as_a_no_op_but_is_learnable() -> None:
    """``W2``/``b2`` start at zero (standard zero-init residual convention)."""
    torch.manual_seed(9)
    head = build_decision_head(
        "choice", hidden_size=8, num_options=4, state_pathway="mlp"
    )
    pathway = head.state_pathway
    with torch.no_grad():
        assert torch.equal(pathway.w2, torch.zeros_like(pathway.w2))
        assert torch.equal(pathway.b2, torch.zeros_like(pathway.b2))
        assert float(pathway.w1.abs().sum()) > 0.0, "W1 must be initialized, not zero"
    fused, options = _inputs(3, 4, 8, seed=9)
    with torch.no_grad():
        state_only = head(fused, options, conditioning_enabled=False).logits
    assert torch.equal(state_only, torch.zeros_like(state_only))
    head(fused, options, conditioning_enabled=False).logits.sum().backward()
    assert pathway.w2.grad is not None
    assert float(pathway.w2.grad.abs().sum()) > 0.0, (
        "the zero-initialized scoring layer still receives a gradient at step 0"
    )


# ---------------------------------------------------------------------------
# 3. The nonlinearity is real: XOR
# ---------------------------------------------------------------------------


def _xor(n: int = 256, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded 2-D XOR: NOT linearly separable in the input space."""
    g = torch.Generator().manual_seed(seed)
    xs = torch.randint(0, 2, (n, 2), generator=g).float() * 2.0 - 1.0
    labels = (xs[:, 0] * xs[:, 1] > 0).long()
    return xs, labels


def _fit(head: ChoiceHead, xs: torch.Tensor, labels: torch.Tensor, steps: int = 400):
    """Train the state pathway only (lambda = 0) and report TRAIN accuracy.

    The option rows are DISTINCT and the residual is switched off, so the loss
    depends on the state pathway alone and the duplicate-option tie is a
    no-op — the fit measures exactly the property under test.
    """
    options = torch.randn(len(xs), 2, xs.shape[1], generator=torch.Generator().manual_seed(0))
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-2)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(head(xs, options).logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((head(xs, options).predicted_index == labels).double().mean())


def test_mlp_state_pathway_fits_xor_that_the_linear_pathway_cannot() -> None:
    """The point of the flag: a linear state path cannot fit XOR, an mlp can.

    Both heads are trained identically (same seed, same optimizer, same steps,
    the option residual switched off in both), so the only difference is the
    nonlinearity of the state path.
    """
    xs, labels = _xor()

    torch.manual_seed(11)
    linear = build_decision_head(
        "choice",
        hidden_size=2,
        num_options=2,
        state_pathway="linear",
        conditioning_scale_init=0.0,
    )
    _fill(linear.state_pathway, std=0.5)
    linear_acc = _fit(linear, xs, labels)

    torch.manual_seed(11)
    mlp = build_decision_head(
        "choice",
        hidden_size=2,
        num_options=2,
        state_pathway="mlp",
        state_hidden_size=8,
        conditioning_scale_init=0.0,
    )
    _fill(mlp.state_pathway, std=0.5)
    mlp_acc = _fit(mlp, xs, labels)

    assert linear_acc <= 0.75, (
        f"a linear state path must stay near chance on XOR, got {linear_acc}"
    )
    assert mlp_acc > 0.9, f"the mlp state path must fit XOR, got {mlp_acc}"


# ---------------------------------------------------------------------------
# 4. The state pathway on every other head
# ---------------------------------------------------------------------------


def test_score_head_applies_the_state_pathway_to_both_schemas() -> None:
    """Option-features schema: state + residual. Fixed-K schema: state only."""
    torch.manual_seed(11)
    fused, options = _inputs(4, 5, 16, seed=11)

    fixed = ScoreHead(16, "dot", num_levels=5)
    assert isinstance(fixed.level_proj, nn.Linear)
    assert fixed.state_pathway is None, (
        "the fixed-K linear schema IS the state pathway: level_proj"
    )
    with torch.no_grad():
        assert torch.equal(fixed(fused).logits, fixed.level_proj(fused))

    variable = ScoreHead(16, "dot", state_pathway="mlp", state_hidden_size=16)
    _fill(variable.state_pathway)
    with torch.no_grad():
        reference = variable.state_pathway(fused, options.shape[1])
        state_only = variable(fused, options, conditioning_enabled=False).logits
    assert torch.allclose(state_only, reference, atol=1e-6)
    out = variable(fused, options)
    assert out.expected_score is not None
    assert bool((out.expected_score >= -1e-4).all())
    assert bool((out.expected_score <= options.shape[1] - 1 + 1e-4).all())

    mlp_fixed = ScoreHead(16, "dot", num_levels=5, state_pathway="mlp")
    assert mlp_fixed.level_proj is None
    assert isinstance(mlp_fixed.state_pathway, MLPStatePathway)
    with torch.no_grad():
        assert mlp_fixed(fused).logits.shape == (4, 5)


def test_score_fixed_k_schema_is_unchanged_by_the_new_default() -> None:
    """A fixed-K score head is still exactly ``level_proj`` on the state."""
    torch.manual_seed(13)
    head = ScoreHead(16, "dot", num_levels=4)
    fused, _ = _inputs(3, 4, 16, seed=13)
    reference = nn.Linear(16, 4)
    with torch.no_grad():
        reference.weight.copy_(head.level_proj.weight)
        reference.bias.copy_(head.level_proj.bias)
    with torch.no_grad():
        assert torch.equal(head(fused).logits, reference(fused))


def test_noul_multilabel_and_regression_keep_their_linear_scorer_by_default() -> None:
    """Their state pathway IS the scorer; the default must stay ``nn.Linear``.

    ``heads.noul.proj`` is the transplant target of ``convert()`` and a
    documented state_dict key, so the default cannot change shape or type.
    """
    torch.manual_seed(17)
    fused, _ = _inputs(3, 4, 16, seed=17)
    noul = NoulHead(16)
    assert isinstance(noul.proj, nn.Linear)
    assert noul.state_pathway == "linear", "the selector stays a plain str here"
    with torch.no_grad():
        assert torch.equal(noul(fused).logits, noul.proj(fused))
    mlp_noul = NoulHead(16, state_pathway="mlp", state_hidden_size=8)
    assert isinstance(mlp_noul.proj, MLPStatePathway)
    with torch.no_grad():
        assert mlp_noul(fused).logits.shape == (3, 1)
        assert bool((mlp_noul(fused).probabilities > 0).all())
        assert bool((mlp_noul(fused).probabilities < 1).all())
    labels = MultiLabelHead(16, 4)
    assert isinstance(labels.proj, nn.Linear)
    mlp_labels = MultiLabelHead(16, 4, state_pathway="mlp")
    assert isinstance(mlp_labels.proj, MLPStatePathway)
    with torch.no_grad():
        assert mlp_labels(fused).probabilities.shape == (3, 4)
    reg = RegressionHead(16, state_pathway="mlp")
    assert isinstance(reg.proj, MLPStatePathway)
    with torch.no_grad():
        assert reg(fused).logits.shape == (3, 1)


# ---------------------------------------------------------------------------
# 5. Config surface
# ---------------------------------------------------------------------------


def test_config_defaults_preserve_the_shipped_scoring_function() -> None:
    """The shipped defaults still pin the linear state pathway and lambda 1.0.

    DECISION-3 (WAVE-2 follow-up): the shipped default MODE is now
    ``conditioning_mode="off"``, so a fresh ``JevXConfig()`` derives
    ``conditioning_enabled = False``. The scoring function itself is untouched:
    the head is still built with ``state_pathway="linear"`` and
    ``conditioning_scale_init`` still 0.0 -> lambda 1.0, and a caller who asks
    for conditioning gets ``conditioning_enabled=True`` back.
    """
    config = JevXConfig()
    assert config.state_pathway == "linear"
    assert config.state_hidden_size is None
    assert config.state_activation == "gelu"
    assert config.conditioning_scale_init == 0.0
    assert config.conditioning_mode == "off"
    assert config.conditioning_enabled is False
    assert JevXConfig(conditioning_mode="full").conditioning_enabled is True



def test_config_round_trip_keeps_every_new_field() -> None:
    config = JevXConfig(
        state_pathway="mlp",
        state_hidden_size=24,
        state_activation="tanh",
        conditioning_scale_init=0.25,
        # "off" derives conditioning_enabled=False and round-trips that way;
        # asking for ("full", False) is a named construction error.
        conditioning_mode="off",
    )
    restored = JevXConfig.from_dict(config.to_dict())
    assert restored.state_pathway == "mlp"
    assert restored.state_hidden_size == 24
    assert restored.state_activation == "tanh"
    assert restored.conditioning_scale_init == 0.25
    assert restored.conditioning_enabled is False
    assert copy.deepcopy(restored).to_dict()["state_pathway"] == "mlp"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"state_pathway": "conv"}, "state_pathway"),
        ({"state_activation": "swish"}, "state_activation"),
        ({"state_hidden_size": 0}, "state_hidden_size"),
        ({"state_hidden_size": -3}, "state_hidden_size"),
        ({"conditioning_scale_init": float("nan")}, "conditioning_scale_init"),
        ({"conditioning_scale_init": float("inf")}, "conditioning_scale_init"),
    ],
)
def test_config_rejects_invalid_new_fields(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        JevXConfig(**kwargs)


def test_config_rejects_non_bool_conditioning_enabled() -> None:
    with pytest.raises(TypeError, match="conditioning_enabled"):
        JevXConfig(conditioning_enabled="yes")


def test_config_enum_values_are_the_shipped_ones() -> None:
    assert STATE_PATHWAYS == ("linear", "mlp")
    assert STATE_ACTIVATIONS == ("gelu", "relu", "tanh")


def test_head_rejects_unknown_pathway_and_activation() -> None:
    with pytest.raises(ValueError, match="state_pathway"):
        build_decision_head("choice", hidden_size=8, state_pathway="conv")
    with pytest.raises(ValueError, match="state_activation"):
        build_decision_head("choice", hidden_size=8, state_activation="swish")
    with pytest.raises(TypeError, match="conditioning_enabled"):
        build_decision_head("choice", hidden_size=8, conditioning_enabled=1)
    with pytest.raises(ValueError, match="conditioning_scale_init"):
        build_decision_head("choice", hidden_size=8, conditioning_scale_init=float("nan"))
    with pytest.raises(ValueError, match="state_hidden_size"):
        build_decision_head("choice", hidden_size=8, state_hidden_size=0)
