# Copyright (c) 2026 ysnrfd
"""Option-pathway invariants: the option row is a DIRECTION, never a magnitude.

EXP-H (``benchmarks/run_head_quality.py`) measured that on FROZEN AUDIO features
(RAVDESS / wav2vec2) the option/schema pathway - not the head - is what collapses
the choice head: bypassing it alone (P4) recovered +7.3 pt over the shipped head.
The mechanism is scale: ``Projection`` ends every option row with a LayerNorm, so
``||o_k|| = sqrt(D)`` whatever the option says, those rows are LEARNABLE, and the
head's option-interaction residual therefore starts SATURATED
(``max|z| = 21.6``, ``std(z|) = 4.28`` against a uniform NLL of ``ln 8 = 2.079``).

The landed fix normalizes the option row to a unit direction before the
projection (``Fusion.option_normalize=True`, the default):

    c     = F_kind(s, q)
    o-hat = o_k / max(||o_k||, eps)
    g     = sigmoid(W_g c + b_g)
    e_k   = g (*) W_o o-hat_k + (1 - g) (*) c

This module pins every property that fix must not break:

1. NEUTRAL-AT-INIT EQUIVALENCE - a zeroed option branch leaves exactly the
   shipped-v2 pathway (``e_k = (1 - g) (*) c``, identical for every k), so an
   untrained head is exactly the state-only probe it claims to superset.
   ``option_normalize=False`` reproduces the pre-fix formula verbatim.
2. MASKED OPTION ROWS ARE EXACTLY ZERO - including an all-zero padded row
   (the eps floor; never NaN) and garbage in the padded positions.
3. INVARIANTS / THEOREMS - per-question simplex, invalid probability exactly
   0, predicted index inside the valid set, padding invariance, option
   independence, and batch-row (per-question) purity.
4. GRADIENT FLOW - into the option rows, the option projection and the gate;
   masked rows get exact zero gradient.
5. DETERMINISM - repeated forwards are bit-identical, and two identically
   seeded constructions agree bit-for-bit on parameters and outputs.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from jevx.decisions import ChoiceHead
from jevx.representations import FUSION_REGISTRY, build_fusion

HIDDEN = 16
#: Every registered fusion kind; the option formula is shared by all of them.
KINDS = sorted(FUSION_REGISTRY)


def _inputs(batch: int = 3, k: int = 4, dim: int = HIDDEN, seed: int = 11):
    """Seeded (state, question, options, option_mask)."""
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, dim, generator=g)
    question = torch.randn(batch, dim, generator=g)
    options = torch.randn(batch, k, dim, generator=g)
    mask = torch.ones(batch, k, dtype=torch.bool)
    mask[0, -1] = False
    return state, question, options, mask


def _zero_option_branch(fusion) -> None:
    """Make the option branch exactly neutral: W_o o_k == 0 for every k."""
    with torch.no_grad():
        fusion.option_proj.weight.zero_()
        fusion.option_proj.bias.zero_()


# ------------------------------------------------------ 1. neutral at init


@pytest.mark.parametrize("kind", KINDS)
def test_neutral_option_branch_equals_shipped_v2_pathway(kind: str) -> None:
    """With the option branch zeroed the formula reduces to
    ``e_k = (1 - g) (*) c`` - identical for every k and byte-identical to the
    pre-fix formula, so an untrained head is exactly the state-only probe the
    head claims to superset."""
    torch.manual_seed(3)
    fusion = build_fusion(kind, HIDDEN)
    _zero_option_branch(fusion)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        out = fusion(state, question, options, mask)
        context = fusion.context(state, question)
        gate = torch.sigmoid(fusion.option_gate(context)).unsqueeze(1)
        expected = (1.0 - gate) * context.unsqueeze(1)
        expected = expected.masked_fill(~mask.unsqueeze(-1), 0.0)
    assert torch.equal(out.option_features, expected), (
        "a zeroed option branch must leave exactly the shipped-v2 pathway"
    )
    live = out.option_features[0, : mask.shape[1] - 1]
    for j in range(1, live.shape[0]):
        assert torch.equal(live[0], live[j]), (
            "a neutral option pathway must not distinguish any option"
        )


def test_single_valid_option_reduces_to_the_state_pathway() -> None:
    """Zero option branch + zeroed interaction + one valid option per row =>
    the logit IS the additive state pathway, exactly. This is the
    probe-equivalence the head claims (it is a strict superset of a linear
    probe on the fused state) and it holds on the landed option formula."""
    torch.manual_seed(5)
    fusion = build_fusion("concat", HIDDEN)
    head = ChoiceHead(HIDDEN, interaction="bilinear", num_options=4)
    with torch.no_grad():
        head.bilinear_weight.zero_()
        head.state_pathway.weight.copy_(
            torch.arange(4, dtype=torch.float32).unsqueeze(1).expand(4, HIDDEN)
        )
    _zero_option_branch(fusion)
    state, question, options, _ = _inputs()
    one_valid = torch.tensor([[True, False, False, False]] * 3)
    with torch.no_grad():
        out = fusion(state, question, options, one_valid)
        decision = head(out.fused, out.option_features, option_mask=one_valid)
        probe = torch.nn.functional.linear(
            out.fused, head.state_pathway.weight[:4], head.state_pathway.bias[:4]
        )
    assert torch.allclose(decision.logits[:, :1], probe[:, :1], atol=1e-6), (
        "with a neutral option pathway the head must equal its state pathway"
    )
    assert torch.equal(decision.predicted_index, torch.zeros(3, dtype=torch.long)), (
        "the single valid option is the only legal prediction"
    )


def test_neutral_option_pathway_collapses_to_the_duplicate_tie() -> None:
    """A feature-level NEUTRAL option pathway makes every option row identical,
    so the documented duplicate-option rule ties all K columns to one score and
    the gradient to every parameter becomes exactly zero.

    This is the measured reason the feature-level "neutral at init" candidate is
    degenerate (benchmark cell p7_option_neutral: RAVDESS 0.1107, BELOW the
    0.134 majority rate, NLL frozen at ln 8). Neutrality is only meaningful at
    the RESIDUAL, where the option rows stay distinct (benchmark cell
    p12_neutral_residual)."""
    torch.manual_seed(5)
    fusion = build_fusion("concat", HIDDEN)
    head = ChoiceHead(HIDDEN, interaction="bilinear", num_options=4)
    _zero_option_branch(fusion)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        out = fusion(state, question, options, mask)
        assert torch.isfinite(out.option_features).all()
        logits = head(out.fused, out.option_features, option_mask=mask).logits
    # The tie rule groups VALID options only (a padded column may hold
    # anything), so row 0 is compared over its three valid options.
    assert torch.allclose(logits[0, :3], logits[0, :1].expand(3), atol=1e-6), (
        "identical VALID option features must receive identical scores"
    )
    assert torch.allclose(logits[1], logits[1, :1].expand(4), atol=1e-6)
    options_g = options.detach().clone().requires_grad_(True)
    out2 = fusion(state, question, options_g, mask)
    head(out2.fused, out2.option_features, option_mask=mask).logits.sum().backward()
    assert float(options_g.grad.abs().sum()) == 0.0, (
        "a tied head sends NO gradient to the option rows - the model can "
        "never leave the uniform distribution"
    )


def test_option_normalize_false_reproduces_the_pre_fix_formula() -> None:
    """``option_normalize=False`` IS the pre-EXP-H formula - the ablation arm
    run as the ``p15_option_unnormalized`` cell."""
    torch.manual_seed(7)
    state, question, options, mask = _inputs()
    torch.manual_seed(9)
    off = build_fusion("concat", HIDDEN, option_normalize=False)
    torch.manual_seed(9)
    on = build_fusion("concat", HIDDEN, option_normalize=True)
    with torch.no_grad():
        a = off(state, question, options, mask)
        b = on(state, question, options, mask)
        gate = torch.sigmoid(off.option_gate(a.fused)).unsqueeze(1)
        projected = off.option_proj(options)
        legacy = gate * projected + (1.0 - gate) * a.fused.unsqueeze(1)
        legacy = legacy.masked_fill(~mask.unsqueeze(-1), 0.0)
    assert torch.equal(a.option_features, legacy)
    assert not torch.equal(a.option_features, b.option_features), (
        "the landed option_normalize=True must actually change the formula"
    )


def test_option_branch_is_scale_free_in_the_option_row() -> None:
    """The defining property: scaling an option row by any POSITIVE scalar
    leaves the option features unchanged (identity, not magnitude). A negative
    scalar flips the direction and is NOT equivalent, so the contract is
    scale-free, not sign-invariant."""
    torch.manual_seed(13)
    fusion = build_fusion("concat", HIDDEN)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        base = fusion(state, question, options, mask)
        scaled = fusion(state, question, options * 37.5, mask)
        flipped = fusion(state, question, -options, mask)
    assert torch.allclose(base.option_features, scaled.option_features, atol=1e-6)
    assert not torch.allclose(
        base.option_features, flipped.option_features, atol=1e-3
    )


# ------------------------------------------------------------- 2. masking


@pytest.mark.parametrize("kind", KINDS)
def test_masked_option_rows_are_exactly_zero_and_finite(kind: str) -> None:
    """``masked_fill`` still runs LAST, so an invalid option row is exactly 0
    even when it holds garbage - and an ALL-ZERO option row normalizes to an
    exact zero vector through the eps floor instead of 0/0 = NaN."""
    torch.manual_seed(17)
    fusion = build_fusion(kind, HIDDEN)
    state, question, options, mask = _inputs()
    options = options.clone()
    options[~mask] = 1e9
    options[0, -1] = 0.0
    with torch.no_grad():
        out = fusion(state, question, options, mask)
    assert torch.isfinite(out.option_features).all(), (
        "a zero-norm option row must not produce NaN/Inf"
    )
    assert torch.all(out.option_features[~mask] == 0.0)


def test_masked_rows_zero_on_a_wider_padded_schema() -> None:
    state, question, options, _ = _inputs(batch=2, k=6)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True] * 6]
    )
    for kind in KINDS:
        fusion = build_fusion(kind, HIDDEN)
        with torch.no_grad():
            out = fusion(state, question, options, mask)
        assert torch.all(out.option_features[~mask] == 0.0), kind


# ------------------------------------------------- 3. invariants / theorems


@pytest.mark.parametrize("kind", KINDS)
def test_choice_head_invariants_hold_on_the_landed_option_features(
    kind: str,
) -> None:
    """Per-question simplex, invalid probability exactly 0, predicted index
    inside the valid set (DecisionOutput contract / spec Def 6)."""
    torch.manual_seed(23)
    fusion = build_fusion(kind, HIDDEN)
    head = ChoiceHead(HIDDEN, interaction="bilinear")
    state, question, options, mask = _inputs()
    with torch.no_grad():
        out = fusion(state, question, options, mask)
        decision = head(out.fused, out.option_features, option_mask=mask)
    probs = decision.probabilities
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.all(probs[~mask] == 0.0)
    assert bool(
        mask.gather(1, decision.predicted_index.unsqueeze(1)).all()
    ), "predicted_index must stay inside the valid option set"


def test_padding_invariance_of_the_valid_outputs() -> None:
    """Perturbing the PADDED option rows cannot move any valid output."""
    torch.manual_seed(29)
    state, question, options, mask = _inputs()
    for kind in KINDS:
        fusion = build_fusion(kind, HIDDEN)
        with torch.no_grad():
            base = fusion(state, question, options, mask)
            perturbed = options.clone()
            perturbed[~mask] = 1e4
            again = fusion(state, question, perturbed, mask)
        assert torch.equal(base.fused, again.fused), kind
        assert torch.equal(
            base.option_features[mask], again.option_features[mask]
        ), kind


def test_option_independence_and_per_question_purity() -> None:
    """Rows of e_k never mix (option independence), and batch row b's output
    depends only on row b's inputs (D5.1 purity)."""
    torch.manual_seed(31)
    fusion = build_fusion("concat", HIDDEN)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        base = fusion(state, question, options, mask)
        edited = options.clone()
        edited[0, 0] += 5.0
        other = fusion(state, question, edited, mask)
        single = fusion(state[1:2], question[1:2], options[1:2], mask[1:2])
    keep = torch.ones_like(mask)
    keep[0, 0] = False
    assert torch.equal(base.option_features[keep], other.option_features[keep])
    assert not torch.equal(
        base.option_features[0, 0], other.option_features[0, 0]
    )
    # A batch of 3 and a batch of 1 can differ in float32 GEMM blocking, so
    # purity is a numerical-equality claim (< 1e-6), not a bit-equality one.
    assert torch.allclose(base.fused[1:2], single.fused, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        base.option_features[1:2], single.option_features, atol=1e-6, rtol=0.0
    )


# ------------------------------------------------------------ 4. gradients


@pytest.mark.parametrize("kind", KINDS)
def test_gradient_flow_reaches_options_projection_and_gate(kind: str) -> None:
    """Every option-pathway input receives a finite, non-trivial gradient -
    normalizing the option row must not sever the schema encoder's path."""
    torch.manual_seed(37)
    fusion = build_fusion(kind, HIDDEN)
    state, question, options, mask = _inputs()
    options = options.clone().requires_grad_(True)
    out = fusion(state, question, options, mask)
    (out.fused.sum() + out.option_features.sum()).backward()
    assert options.grad is not None
    assert torch.isfinite(options.grad).all()
    assert float(options.grad[mask].abs().sum()) > 0.0
    assert torch.all(options.grad[~mask] == 0.0), (
        "masked option rows are constant 0, so they carry no gradient"
    )
    grads = {
        name: p.grad
        for name, p in fusion.named_parameters()
        if p.grad is not None
    }
    assert grads, "no fusion parameter received a gradient"
    for name, grad in grads.items():
        assert torch.isfinite(grad).all(), name
    for required in (
        "option_proj.weight",
        "option_proj.bias",
        "option_gate.weight",
    ):
        assert required in grads, f"{required} received no gradient"
        assert float(grads[required].abs().sum()) > 0.0, required


def test_option_row_gradient_does_not_vanish_at_any_option_scale() -> None:
    """The strict-superset property the neutral-at-init contract needs: a huge
    schema row must not zero the option branch's gradient."""
    torch.manual_seed(41)
    state, question, options, mask = _inputs()
    totals = []
    for scale in (1.0, 1e3):
        fusion = build_fusion("concat", HIDDEN)
        scaled = (options * scale).clone().requires_grad_(True)
        fusion(state, question, scaled, mask).option_features.sum().backward()
        totals.append(float(scaled.grad.abs().sum()))
    assert all(t > 0.0 for t in totals)


def test_full_stack_trains_through_the_option_pathway() -> None:
    """A short optimization must move the option projection through the landed
    formula and drop the loss below the uniform reference - the guard against a
    severed or frozen option pathway (the RAVDESS failure mode)."""
    torch.manual_seed(43)
    fusion = build_fusion("concat", HIDDEN)
    head = ChoiceHead(HIDDEN, interaction="bilinear")
    opt = torch.optim.AdamW(fusion.parameters(), lr=1e-2)
    state, question, options, mask = _inputs(batch=8, k=4, seed=47)
    target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    before = fusion.option_proj.weight.detach().clone()
    first = None
    for _ in range(5):
        out = fusion(state, question, options, mask)
        decision = head(out.fused, out.option_features, option_mask=mask)
        loss = torch.nn.functional.cross_entropy(decision.logits, target)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert not torch.equal(fusion.option_proj.weight.detach(), before), (
        "the option projection never moved - the option pathway is severed"
    )
    assert float(loss.detach()) < first
    assert float(loss.detach()) < math.log(4)


# ---------------------------------------------------------- 5. determinism


@pytest.mark.parametrize("kind", KINDS)
def test_forward_is_bit_identical_across_repeated_calls(kind: str) -> None:
    torch.manual_seed(47)
    fusion = build_fusion(kind, HIDDEN)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        first = fusion(state, question, options, mask)
        second = fusion(state, question, options, mask)
    assert torch.equal(first.fused, second.fused)
    assert torch.equal(first.option_features, second.option_features)


def test_two_identically_seeded_constructions_agree_bit_for_bit() -> None:
    state, question, options, mask = _inputs()
    built = []
    for flag in (True, False):
        torch.manual_seed(53)
        fusion = build_fusion("concat", HIDDEN, option_normalize=flag)
        with torch.no_grad():
            out = fusion(state, question, options, mask)
        built.append((fusion.option_proj.weight.detach().clone(), out))
    (w_true, o_true), (w_false, o_false) = built
    assert torch.equal(w_true, w_false), "seeding must fix the parameters"
    assert not torch.equal(
        o_true.option_features, o_false.option_features
    ), "the two flags must produce different option features"
    torch.manual_seed(53)
    again = build_fusion("concat", HIDDEN, option_normalize=True)
    with torch.no_grad():
        out_again = again(state, question, options, mask)
    assert torch.equal(again.option_proj.weight.detach(), w_true)
    assert torch.equal(out_again.option_features, o_true.option_features)


def test_train_and_eval_modes_agree() -> None:
    """The landed formula has no train/eval-dependent term, so `.train()` must
    not change a forward - a property the rejected option-pathway-dropout
    candidate did not have."""
    torch.manual_seed(59)
    fusion = build_fusion("concat", HIDDEN)
    state, question, options, mask = _inputs()
    with torch.no_grad():
        fusion.eval()
        a = fusion(state, question, options, mask)
        fusion.train()
        b = fusion(state, question, options, mask)
    assert torch.equal(a.option_features, b.option_features)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
def test_dtype_is_preserved_and_low_precision_is_safe(dtype) -> None:
    """An all-zero option row in low precision must not turn the eps floor
    into a NaN, and the masked rows stay exactly 0 in every dtype."""
    fusion = build_fusion("concat", HIDDEN).to(dtype)
    state, question, options, mask = _inputs()
    state = state.to(dtype)
    question = question.to(dtype)
    options = options.to(dtype)
    options[~mask] = 0.0
    with torch.no_grad():
        out = fusion(state, question, options, mask)
    assert out.option_features.dtype == dtype
    assert torch.isfinite(out.option_features).all(), dtype
    assert torch.all(out.option_features[~mask] == 0.0), dtype


def test_state_dict_is_unchanged_by_the_fix() -> None:
    """The fix adds no parameters, so checkpoints written before it still load
    (state_dict keys and shapes are identical)."""
    torch.manual_seed(61)
    before = build_fusion("concat", HIDDEN, option_normalize=False).state_dict()
    after = build_fusion("concat", HIDDEN, option_normalize=True).state_dict()
    assert set(before) == set(after)
    assert all(before[k].shape == after[k].shape for k in before)
    fresh = build_fusion("concat", HIDDEN, option_normalize=True)
    fresh.load_state_dict(before)
    assert copy.deepcopy(fresh).option_normalize is True
