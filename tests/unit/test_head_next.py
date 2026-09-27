# Copyright (c) 2026 ysnrfd
"""The option interaction is a RESIDUAL: ``z_k = state_k + lambda * r_k``.

This module pins the half of the scoring identity that the state-pathway flag
introduced:

1. ``interaction_scale`` (the lambda buffer) — ``set_interaction_scale(0.0)``
   makes the residual EXACTLY zero (not merely small) and stops the option
   gradients; lambda = 1 with a zeroed state pathway reproduces the
   interaction-only path bit-for-bit; lambda scales the residual exactly.
2. ``conditioning_enabled=False`` on a head call is the same statement, and it
   never mutates the head.
3. Every theorem the DecisionOutput contract promises still holds on the new
   path — per-question simplex, masked columns exactly 0, argmax inside the
   valid set, score bounds, noul interior, the duplicate-option tie, variable-K,
   the confidence range and the fp16/bf16/fp32 dtypes.
4. Gradients reach the state pathway and the option pathway INDEPENDENTLY.
5. ``JevXForDecision.forward(condition_options=False)`` makes the call
   state-only end to end (interaction off AND the option rows neutralized
   before fusion), with a schema present.
6. Backward compatibility: a state_dict written by the pre-flag head (no
   ``interaction_scale`` key) still loads with ``strict=False`` and produces
   today's numbers. Same precedent as the ``option_normalize`` flag, which also
   added a constructor switch without changing any tensor layout.
"""

from __future__ import annotations

import copy
import io

import pytest
import torch
import torch.nn as nn

from jevx import JevXConfig, JevXForDecision
from jevx.decisions import (
    ChoiceHead,
    NoulHead,
    ScoreHead,
    build_decision_head,
    mask_from_num_options,
)

INTERACTIONS = ("dot", "bilinear", "mlp")
HIDDEN = 32  # matches the tiny_bert fixture
STATE = "hello world"


def _inputs(batch: int = 4, k: int = 5, dim: int = 16, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(batch, dim, generator=g),
        torch.randn(batch, k, dim, generator=g),
    )


def _fill(pathway: nn.Module, std: float = 0.3) -> None:
    """Give a zero-initialized state pathway non-zero, random parameters."""
    with torch.no_grad():
        for param in pathway.parameters():
            param.normal_(0.0, std)


def build_wrapper(backbone, tokenizer, **config_overrides) -> JevXForDecision:
    """Tiny wrapper with a deep-copied backbone (fixtures stay read-only)."""
    overrides = dict(
        decision_hidden_size=HIDDEN,
        question_hidden_size=HIDDEN,
        schema_hidden_size=HIDDEN,
    )
    overrides.update(config_overrides)
    wrapper = JevXForDecision(
        JevXConfig(**overrides),
        backbone=copy.deepcopy(backbone),
        tokenizer=tokenizer,
    )
    return wrapper.eval()


@pytest.fixture()
def wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    # EXPLICIT conditioning_mode="full". This module pins the per-call
    # `condition_options` switch, and several of its assertions are about the
    # CONDITIONED call (e.g. "True and the default are byte-identical"). Since
    # DECISION-3 the shipped default is "off" (a state-only call), so the
    # conditioned baseline has to be requested. The model still exercises the
    # same shipped forward; only the mode is pinned. See
    # tests/unit/test_conditioning_off_matches_state_only.py for the "off"
    # side and the DECISION-3 default.
    return build_wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="full")


# ---------------------------------------------------------------------------
# 1. lambda / interaction_scale
# ---------------------------------------------------------------------------


def test_lambda_defaults_to_one_so_a_fresh_head_is_the_shipped_head() -> None:
    head = build_decision_head("choice", hidden_size=8, num_options=4)
    assert head.get_interaction_scale == 1.0
    assert "interaction_scale" in dict(head.named_buffers())
    assert not any(
        name.endswith("interaction_scale") for name, _ in head.named_parameters()
    ), "lambda is a curriculum control, not a trainable weight"


def test_set_interaction_scale_zero_makes_the_residual_exactly_zero() -> None:
    """lambda = 0 gives EXACTLY the state pathway — bitwise, not approximately."""
    torch.manual_seed(0)
    head = build_decision_head(
        "choice", hidden_size=16, interaction="bilinear", num_options=4
    )
    _fill(head.state_pathway)
    fused, options = _inputs(4, 4, 16, seed=0)
    head.set_interaction_scale(0.0)
    assert head.get_interaction_scale == 0.0
    with torch.no_grad():
        got = head(fused, options).logits
        want = head.state_pathway(fused, options.shape[1])
    assert torch.equal(got, want), (
        "lambda = 0 must leave the state pathway bit-for-bit, never a small residual"
    )
    with torch.no_grad():
        assert torch.equal(
            got, head(fused, options, conditioning_enabled=True).logits
        ), "a zero scale stays zero for this call"


def test_lambda_one_with_a_zeroed_state_path_is_the_interaction_only_path() -> None:
    """Mask the state term properly: zero the pathway, keep lambda = 1."""
    torch.manual_seed(1)
    head = build_decision_head(
        "choice", hidden_size=16, interaction="bilinear", num_options=4
    )
    fused, options = _inputs(4, 4, 16, seed=1)
    with torch.no_grad():
        head.state_pathway.weight.zero_()
        head.state_pathway.bias.zero_()
        assert head.get_interaction_scale == 1.0
        got = head(fused, options).logits
        want = head.interaction_logits(fused, options)
    assert torch.equal(got, want), (
        "a zeroed state pathway at lambda = 1 IS the interaction-only path"
    )


@pytest.mark.parametrize("lam", [0.5, 2.0, -1.5])
def test_lambda_scales_the_residual_exactly(lam: float) -> None:
    """z_k = state_k + lambda * r_k, to the bit."""
    torch.manual_seed(2)
    head = build_decision_head(
        "choice", hidden_size=16, interaction="mlp", num_options=4
    )
    _fill(head.state_pathway)
    fused, options = _inputs(4, 4, 16, seed=2)
    head.set_interaction_scale(lam)
    with torch.no_grad():
        got = head(fused, options).logits
        want = head.state_pathway(fused, 4) + lam * head.interaction_logits(
            fused, options
        )
    assert torch.allclose(got, want, atol=1e-6, rtol=0.0)


def test_set_interaction_scale_rejects_non_finite_values() -> None:
    head = build_decision_head("choice", hidden_size=8, num_options=4)
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="set_interaction_scale"):
            head.set_interaction_scale(bad)
    assert head.get_interaction_scale == 1.0, "a rejected value must not be applied"


def test_conditioning_enabled_false_is_exactly_the_state_pathway() -> None:
    torch.manual_seed(3)
    head = build_decision_head(
        "choice", hidden_size=16, interaction="bilinear", num_options=4
    )
    _fill(head.state_pathway)
    fused, options = _inputs(4, 4, 16, seed=3)
    with torch.no_grad():
        off = head(fused, options, conditioning_enabled=False).logits
        want = head.state_pathway(fused, 4)
        on = head(fused, options, conditioning_enabled=True).logits
    assert torch.equal(off, want)
    assert not torch.equal(on, want), "the flag must actually change something"
    assert head.conditioning_enabled is True, "a per-call flag never mutates the head"


def test_head_built_with_conditioning_enabled_false_starts_state_only() -> None:
    head = build_decision_head(
        "choice",
        hidden_size=16,
        num_options=4,
        state_pathway="mlp",
        conditioning_enabled=False,
    )
    assert head.conditioning_enabled is False
    fused, options = _inputs(3, 4, 16, seed=4)
    _fill(head.state_pathway)
    with torch.no_grad():
        got = head(fused, options).logits
        want = head.state_pathway(fused, 4)
        restored = head(fused, options, conditioning_enabled=True).logits
    assert torch.allclose(got, want, atol=1e-6)
    assert not torch.equal(restored, got), (
        "a per-call True re-enables the residual for that call only"
    )


# ---------------------------------------------------------------------------
# 2. Theorem invariants on the new path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_choice_theorems_hold_on_the_new_state_path(
    interaction: str, pathway: str
) -> None:
    """Simplex / mask / index-in-schema with the residual active."""
    torch.manual_seed(5)
    head = build_decision_head(
        "choice",
        hidden_size=16,
        interaction=interaction,
        num_options=6,
        state_pathway=pathway,
    )
    _fill(head.state_pathway)
    fused, options = _inputs(5, 6, 16, seed=5)
    counts = torch.tensor([1, 3, 6, 2, 5])
    mask = mask_from_num_options(counts, 6)
    out = head(fused, options, option_mask=mask)
    probs = out.probabilities
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-6), (
        "each question normalizes to 1 over its VALID options only"
    )
    assert torch.all(probs[~mask] == 0.0), "invalid options are exactly 0"
    rows = torch.arange(5)
    assert bool(mask[rows, out.predicted_index].all()), (
        "argmax must address a valid option"
    )


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_duplicate_option_tie_is_still_enforced_on_the_new_path(pathway: str) -> None:
    """Identical VALID option rows must receive identical scores."""
    torch.manual_seed(6)
    head = build_decision_head(
        "choice", hidden_size=16, num_options=4, state_pathway=pathway
    )
    _fill(head.state_pathway)
    fused, options = _inputs(3, 4, 16, seed=6)
    options = options.clone()
    options[:, 1, :] = options[:, 0, :]
    with torch.no_grad():
        logits = head(fused, options).logits
    assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[:, 3:] = False
    with torch.no_grad():
        masked = head(fused, options, option_mask=mask).logits
    assert torch.allclose(masked[:, 0], masked[:, 1], atol=1e-6), (
        "the tie holds on the valid columns of a padded batch"
    )


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_variable_k_is_unchanged_on_the_new_path(pathway: str) -> None:
    torch.manual_seed(7)
    head = build_decision_head(
        "choice", hidden_size=12, num_options=16, state_pathway=pathway
    )
    capacity = head.state_pathway.width
    for k in (2, 5, 3, 9):
        fused, options = _inputs(4, k, 12, seed=200 + k)
        out = head(fused, options)
        assert out.logits.shape == (4, k)
        if k <= capacity:
            assert head.state_pathway.width == capacity, (
                "a narrower schema must slice, never reallocate"
            )


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_score_bounds_and_noul_interior_on_the_new_path(pathway: str) -> None:
    torch.manual_seed(8)
    fused, options = _inputs(4, 5, 16, seed=8)
    score = ScoreHead(16, "dot", state_pathway=pathway, state_hidden_size=16)
    _fill(score.state_pathway)
    out = score(fused, options)
    k = options.shape[1]
    assert out.expected_score is not None
    assert bool((out.expected_score >= -1e-4).all())
    assert bool((out.expected_score <= k - 1 + 1e-4).all()), (
        "T3: the expected score is a convex combination of the levels"
    )
    assert torch.allclose(out.probabilities.sum(dim=-1), torch.ones(4), atol=1e-6)
    noul = NoulHead(16, state_pathway=pathway, state_hidden_size=16)
    probs = noul(fused).probabilities
    assert bool((probs > 0.0).all()) and bool((probs < 1.0).all()), (
        "T4: a noul probability is strictly interior for every finite logit"
    )


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_dtypes_survive_the_new_path(pathway: str, dtype) -> None:
    torch.manual_seed(9)
    head = build_decision_head(
        "choice", hidden_size=16, num_options=5, state_pathway=pathway
    ).to(dtype)
    _fill(head.state_pathway)
    g = torch.Generator().manual_seed(9)
    fused = torch.randn(3, 16, generator=g).to(dtype)
    options = torch.randn(3, 5, 16, generator=g).to(dtype)
    try:
        out = head(fused, options)
    except (TypeError, RuntimeError) as exc:  # dtype unsupported on this build
        if dtype is torch.float16:
            pytest.skip(f"cpu {dtype} unsupported: {exc}")
        raise
    assert out.logits.dtype == dtype
    assert torch.isfinite(out.logits).all(), dtype
    probs = out.probabilities.float()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-2), dtype


# ---------------------------------------------------------------------------
# 3. Gradient flow, separately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_gradients_reach_the_state_and_the_option_pathway(pathway: str) -> None:
    torch.manual_seed(10)
    head = build_decision_head(
        "choice",
        hidden_size=16,
        interaction="bilinear",
        num_options=4,
        state_pathway=pathway,
    )
    _fill(head.state_pathway)
    fused, options = _inputs(6, 4, 16, seed=10)
    fused = fused.clone().requires_grad_(True)
    options = options.clone().requires_grad_(True)
    out = head(fused, options)
    loss = (out.probabilities * torch.arange(4, dtype=torch.float32)).sum()
    loss.backward()
    assert float(fused.grad.abs().sum()) > 0.0, "the state path must be live"
    assert float(options.grad.abs().sum()) > 0.0, "the option path must be live"
    grads = {
        name: p.grad
        for name, p in head.state_pathway.named_parameters()
        if p.grad is not None
    }
    assert grads and all(float(g.abs().sum()) > 0.0 for g in grads.values())


@pytest.mark.parametrize("pathway", ["linear", "mlp"])
def test_zeroing_lambda_stops_the_option_gradient_only(pathway: str) -> None:
    """lambda = 0 severs the option rows and leaves the state path trainable."""
    torch.manual_seed(12)
    head = build_decision_head(
        "choice",
        hidden_size=16,
        interaction="bilinear",
        num_options=4,
        state_pathway=pathway,
    )
    _fill(head.state_pathway)
    head.set_interaction_scale(0.0)
    fused, options = _inputs(6, 4, 16, seed=12)
    options = options.clone().requires_grad_(True)
    head(fused, options).logits.sum().backward()
    assert options.grad is None or float(options.grad.abs().sum()) == 0.0, (
        "a zero lambda must send NO gradient to the option rows"
    )
    state_grads = [
        p.grad for _, p in head.state_pathway.named_parameters() if p.grad is not None
    ]
    assert state_grads and all(float(g.abs().sum()) > 0.0 for g in state_grads)


def test_conditioning_enabled_false_stops_the_option_gradient() -> None:
    torch.manual_seed(13)
    head = build_decision_head(
        "choice", hidden_size=16, interaction="bilinear", num_options=4
    )
    _fill(head.state_pathway)
    fused, options = _inputs(4, 4, 16, seed=13)
    options = options.clone().requires_grad_(True)
    head(fused, options, conditioning_enabled=False).logits.sum().backward()
    assert options.grad is None or float(options.grad.abs().sum()) == 0.0
    bilinear_grad = head.bilinear_weight.grad
    assert bilinear_grad is None or float(bilinear_grad.abs().sum()) == 0.0


# ---------------------------------------------------------------------------
# 4. Model level: condition_options
# ---------------------------------------------------------------------------


def test_condition_options_is_a_pure_addition(wrapper: JevXForDecision) -> None:
    """``True`` and the default (None) are byte-identical to a plain call."""
    with torch.no_grad():
        default = wrapper.choice(STATE, ["which one"], options=[["a", "b", "c"]])
        explicit = wrapper.choice(
            STATE, ["which one"], options=[["a", "b", "c"]], condition_options=True
        )
    assert torch.equal(default.decisions[0].logits, explicit.decisions[0].logits)
    assert torch.equal(
        default.decisions[0].predicted_index, explicit.decisions[0].predicted_index
    )


def test_condition_options_false_is_the_state_only_path(wrapper: JevXForDecision) -> None:
    """The whole call is state-only: interaction off AND option rows neutral.

    Since DECISION-3 / P0-2 a state-only call is state-only in BOTH halves of the
    conditioning path: the schema rows never enter fusion AND the question rows
    handed to fusion are exact zeros, so the fused state of the state-only call
    is NOT the conditioned call's fused state any more. The reference is
    therefore built from the fused state the STATE-ONLY call itself produced -
    the same head fed EXACT zero option features AND the same
    ``duplicate_tie=False`` the wrapper passes on that path (the zeros are a
    shape placeholder, not a schema, so the duplicate-option tie must not fire
    - otherwise the K per-class state scores get averaged into one and the
    reference IS the degenerate model).

    What is still asserted unchanged: the option branch never feeds back into
    ``fused`` (the state-only fused is the same whether a schema is present or
    not - see ``test_condition_options_false_is_invariant_to_the_schema``).
    """
    head = wrapper.heads["choice"]
    _fill(head.state_pathway)
    captured: dict[str, torch.Tensor] = {}
    original = head.compute_logits

    def spy(fused, option_features=None, **kwargs):
        captured["fused"] = fused.detach().clone()
        captured["options"] = option_features.detach().clone()
        return original(fused, option_features, **kwargs)

    with torch.no_grad():
        head.compute_logits = spy  # type: ignore[method-assign]
        conditioned = wrapper.choice(STATE, ["which one"], options=[["a", "b", "c"]])
        conditioned_fused = captured["fused"].clone()
        conditioned_options = captured["options"].clone()
        # The spy stays installed for the state-only call: `captured` must hold
        # the tensors THAT call produced, because the two calls no longer share
        # a fused state (the question rows differ).
        off = wrapper.choice(
            STATE, ["which one"], options=[["a", "b", "c"]], condition_options=False
        )
        head.compute_logits = original  # type: ignore[method-assign]
    zeros = torch.zeros_like(captured["options"])
    assert float(zeros.abs().sum()) == 0.0
    with torch.no_grad():
        state_only = head(
            captured["fused"],
            zeros,
            conditioning_enabled=False,
            duplicate_tie=False,
        ).logits
    assert torch.equal(off.decisions[0].logits, state_only), (
        "condition_options=False must equal the state pathway on the fused state"
    )
    assert torch.allclose(
        off.decisions[0].probabilities.sum(dim=-1), torch.ones(1), atol=1e-6
    )
    with torch.no_grad():
        conditioned_logits = head(conditioned_fused, conditioned_options).logits
    assert not torch.equal(conditioned_logits, state_only), (
        "the residual must actually be switched off, not merely small"
    )
    # The state-only call's fused state differs from the conditioned one: the
    # question path is neutralized too (P0-2). Control for the assertion above.
    assert not torch.equal(captured["fused"], conditioned_fused), (
        "the state-only fused state is identical to the conditioned one, so "
        "the question path was NOT neutralized"
    )
    # The conditioned call is still a conditioned call.
    assert not torch.equal(conditioned.decisions[0].logits, off.decisions[0].logits)

def test_condition_options_false_is_invariant_to_the_schema(wrapper: JevXForDecision) -> None:
    """Different option texts (same K) must give bit-identical decisions."""
    with torch.no_grad():
        a = wrapper.choice(
            STATE, ["which one"], options=[["a", "b", "c"]], condition_options=False
        )
        b = wrapper.choice(
            STATE, ["which one"], options=[["the", "cat", "sat"]], condition_options=False
        )
    assert torch.equal(a.decisions[0].logits, b.decisions[0].logits), (
        "the option rows must contribute NOTHING when conditioning is off"
    )
    assert torch.equal(
        a.decisions[0].predicted_index, b.decisions[0].predicted_index
    )


def test_condition_options_false_keeps_the_schema_contract(wrapper: JevXForDecision) -> None:
    """K columns, masked padding and the schema metadata survive the switch."""
    with torch.no_grad():
        out = wrapper.choice(
            STATE,
            ["which one", "which one"],
            options=[["a", "b", "c"], ["a", "b"]],
            condition_options=False,
        )
    first, second = out.decisions
    assert first.logits.shape == (1, 3)
    assert second.logits.shape == (1, 3)
    assert torch.all(second.probabilities[0, 2:] == 0.0), "padding stays exactly 0"
    assert torch.allclose(
        second.probabilities[0, :2].sum(), torch.ones(1), atol=1e-6
    )
    assert second.schema_metadata["option_keys"] == ["a", "b"]
    assert int(first.predicted_index[0]) < 3
    for decision in (first, second):
        assert 0.0 <= float(decision.confidence[0]) <= 1.0, "confidence range"


def test_condition_options_false_on_score_and_noul(wrapper: JevXForDecision) -> None:
    with torch.no_grad():
        scored = wrapper.score(
            STATE,
            ["how many"],
            options=[["a", "b", "c", "the"]],
            condition_options=False,
        )
        noul = wrapper.noul(STATE, ["is it true"], condition_options=False)
    score = float(scored.decisions[0].expected_score[0])
    assert -1e-4 <= score <= 3 + 1e-4, "T3 bounds survive the state-only call"
    probability = float(noul.decisions[0].probabilities[0])
    assert 0.0 < probability < 1.0, "T4 interior survives the state-only call"


def test_condition_options_must_be_a_bool_or_none(wrapper: JevXForDecision) -> None:
    with pytest.raises(TypeError, match="condition_options"):
        wrapper.choice(
            STATE, ["which one"], options=[["a", "b"]], condition_options="no"
        )


def test_config_conditioning_mode_off_makes_the_default_call_state_only(
    tiny_bert, tiny_tokenizer
) -> None:
    # conditioning_mode is the source of truth: "off" derives
    # conditioning_enabled=False (the pair ("full", False) is a named
    # construction error), so the state-only head is what gets built.
    model = build_wrapper(tiny_bert, tiny_tokenizer, conditioning_mode="off")
    with torch.no_grad():
        out = model.choice(STATE, ["which one"], options=[["a", "b", "c"]])
        same = model.choice(
            STATE, ["which one"], options=[["a", "b", "c"]], condition_options=False
        )
    assert model.heads["choice"].conditioning_enabled is False
    assert torch.equal(out.decisions[0].logits, same.decisions[0].logits)


def test_config_state_pathway_reaches_the_heads(tiny_bert, tiny_tokenizer) -> None:
    model = build_wrapper(
        tiny_bert,
        tiny_tokenizer,
        state_pathway="mlp",
        state_hidden_size=8,
        state_activation="tanh",
    )
    for name, head in model.heads.items():
        assert head.state_pathway_name == "mlp", name
        assert head.state_hidden_size == 8, name
    with torch.no_grad():
        out = model.choice(STATE, ["which one"], options=[["a", "b", "c"]])
    assert torch.isfinite(out.decisions[0].logits).all()
    assert torch.allclose(
        out.decisions[0].probabilities.sum(dim=-1), torch.ones(1), atol=1e-6
    )


def test_default_model_builds_the_pre_flag_heads(wrapper: JevXForDecision) -> None:
    """Default config -> the shipped scoring function, bit-for-bit.

    The head is evaluated by hand from its own parameters and the logits are
    compared bitwise: that IS the statement "this is still
    ``z_k = s^T W[k, :] + b_k + r_k``".
    """
    for head in wrapper.heads.values():
        assert head.state_pathway_name == "linear"
        assert head.get_interaction_scale == 1.0
    head = wrapper.heads["choice"]
    _fill(head.state_pathway)
    fused, options = _inputs(3, 4, HIDDEN, seed=21)
    with torch.no_grad():
        got = head(fused, options).logits
        want = head.state_pathway(fused, 4) + head.interaction_logits(fused, options)
    assert torch.equal(got, want)


def test_default_config_passes_no_head_kwargs() -> None:
    """The conditioned default passes NO head kwargs; the shipped default (off) does.

    DECISION-3: ``JevXConfig()`` is now ``conditioning_mode="off"``, so its
    DERIVED ``conditioning_enabled`` is ``False`` and ``_state_pathway_kwargs``
    returns an explicit dict. The property that was actually being pinned - a
    default model is built with the PRE-FLAG head constructor defaults, i.e. the
    same state_dict and the same lambda - is asserted below for BOTH modes and
    is stronger than the old ``== {}`` shorthand.

    ``"full"`` is the mode that reproduces the pre-flag construction exactly.
    """
    assert JevXForDecision._state_pathway_kwargs(
        JevXConfig(conditioning_mode="full")
    ) == {}

    shipped = JevXForDecision._state_pathway_kwargs(JevXConfig())
    assert shipped["conditioning_enabled"] is False
    assert shipped["conditioning_scale_init"] == 1.0  # config 0.0 = "unspecified"
    assert shipped["state_pathway"] == "linear"
    assert shipped["state_activation"] == "gelu"
    assert shipped["state_hidden_size"] is None

    kwargs = JevXForDecision._state_pathway_kwargs(
        JevXConfig(conditioning_mode="full", conditioning_scale_init=0.25,
                   state_pathway="mlp")
    )
    assert kwargs["conditioning_scale_init"] == 0.25
    assert kwargs["state_pathway"] == "mlp"
    # 0.0 means "unspecified" and keeps the head default of 1.0.
    assert (
        JevXForDecision._state_pathway_kwargs(
            JevXConfig(conditioning_mode="full", state_pathway="mlp")
        )["conditioning_scale_init"]
        == 1.0
    )


def test_both_default_modes_build_the_same_head_state_dict(
    tiny_bert, tiny_tokenizer
) -> None:
    """The head parameters do not depend on the conditioning MODE.

    DECISION-3 changed the shipped default from ``"full"`` to ``"off"``, which
    makes ``_state_pathway_kwargs`` forward ``conditioning_enabled=False`` to the
    head constructors. That value is a head ATTRIBUTE, never a tensor, so the
    default change cannot move a single weight. This is the invariant that makes
    DECISION-3 a default change rather than an architecture change, and it is
    asserted here on two real wrappers built from the same seed.
    """
    def build(mode):
        torch.manual_seed(31)
        return build_wrapper(
            tiny_bert, tiny_tokenizer, conditioning_mode=mode
        ).state_dict()

    off = build("off")
    full = build("full")
    assert set(off) == set(full), "the mode changed the shipped key layout"
    for name in sorted(off):
        assert off[name].shape == full[name].shape, name
        assert torch.equal(off[name], full[name]), (
            f"{name} differs between conditioning_mode='off' and 'full'"
        )
    # And the head attribute is the only thing that moved.
    assert full["heads.choice.interaction_scale"].shape == torch.Size([])


# ---------------------------------------------------------------------------
# 5. Backward compatibility of the state_dict
# ---------------------------------------------------------------------------


def test_pre_flag_state_dict_loads_and_reproduces_todays_numbers() -> None:
    """An old checkpoint (no ``interaction_scale`` key) still loads.

    Migration note: the lambda buffer is the only new tensor, it is
    zero-dimensional, and the head's own default of 1.0 is the pre-flag
    scoring function — so an old state_dict loads with ``strict=False``, keeps
    lambda = 1 and reproduces the old logits bit-for-bit. The
    ``option_normalize`` flag is the same precedent: a new constructor switch
    that adds no tensor to the shipped layout.
    """
    torch.manual_seed(22)
    d, k, b = 12, 4, 3
    head = build_decision_head(
        "choice", hidden_size=d, interaction="bilinear", num_options=k
    )
    _fill(head.state_pathway)
    fused, options = _inputs(b, k, d, seed=22)
    with torch.no_grad():
        before = head(fused, options).logits.clone()
    # The pre-flag state dict: every key except the new scalar buffer.
    old_state = {
        name: tensor.clone()
        for name, tensor in head.state_dict().items()
        if name != "interaction_scale"
    }
    assert "interaction_scale" not in old_state
    assert set(old_state) == {
        "bilinear_weight",
        "state_pathway.weight",
        "state_pathway.bias",
    }

    reloaded = build_decision_head(
        "choice", hidden_size=d, interaction="bilinear", num_options=k
    )
    incompatible = reloaded.load_state_dict(old_state, strict=False)
    assert incompatible.missing_keys == ["interaction_scale"]
    assert incompatible.unexpected_keys == []
    assert reloaded.get_interaction_scale == 1.0, (
        "the missing key keeps the constructor value, which is the pre-flag lambda"
    )
    with torch.no_grad():
        assert torch.equal(reloaded(fused, options).logits, before), (
            "a pre-flag checkpoint must reproduce today's numbers exactly"
        )


def test_state_dict_round_trip_keeps_the_ramped_lambda() -> None:
    torch.manual_seed(23)
    head = build_decision_head(
        "choice", hidden_size=8, interaction="dot", num_options=4
    )
    head.set_interaction_scale(0.25)
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    assert float(state["interaction_scale"]) == pytest.approx(0.25)
    restored = build_decision_head(
        "choice", hidden_size=8, interaction="dot", num_options=4
    )
    restored.load_state_dict(state)
    assert restored.get_interaction_scale == pytest.approx(0.25)
    assert isinstance(head, ChoiceHead)
