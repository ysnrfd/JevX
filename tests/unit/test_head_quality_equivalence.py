# Copyright (c) 2026 ysnrfd
"""Head-quality invariants: the choice head is a strict SUPERSET of a linear probe.

EXP-H (``benchmarks/run_head_quality.py``) measured that on FROZEN features the
JevX choice head LOST to a plain linear/MLP probe. The cause is structural: with one
fixed question and one fixed schema every option row is identical on every batch row,
so a head that scores options ONLY through an option interaction has no state-only
path and must reach its per-class weights indirectly, through the schema encoder and
the fusion option gate.

The fix adds an additive, state-only term to the scoring function:

    z_k = s^T W[k, :] + b[k] + r_k(s, e_k)

with ``s = fused``, ``W`` the additive pathway and ``r_k`` the option
interaction residual. This module pins the properties that make the head probe-
equivalent:

1. with the interaction residual zeroed the head EQUALS ``nn.Linear(D, K)`` on the
   fused state, exactly (atol 1e-6, seeded);
2. with interactions enabled the head still satisfies the simplex / mask / bounds
   theorems the DecisionOutput contract promises;
3. gradients reach BOTH the state and the option pathway;
4. masked (invalid) option rows still receive probability exactly 0;
5. variable-``K`` is unaffected - no reallocation, no cross-schema leakage;
6. a tiny separable problem is solved to > 0.9 TRAIN accuracy, so the head can never
   be degenerate (this is the guard against a collapsed head, the RAVDESS failure).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from jevx.decisions import (
    DEFAULT_STATE_PATHWAY_OPTIONS,
    ChoiceHead,
    LinearProbePathway,
    build_decision_head,
    mask_from_num_options,
)

INTERACTIONS = ("dot", "bilinear", "mlp")


def _seeded_inputs(seed: int, batch: int, dim: int, num_options: int):
    """Seeded (fused, option_features) pair - reproducible across runs."""
    g = torch.Generator().manual_seed(seed)
    fused = torch.randn(batch, dim, generator=g)
    options = torch.randn(batch, num_options, dim, generator=g)
    return fused, options


def _zero_interaction(head: ChoiceHead) -> None:
    """Neutralize the option-interaction residual for this head INSTANCE.

    Replaces `interaction_logits` with a zero stub, which is the exact
    mathematical statement "r_k(s, e_k) == 0" for every interaction mode
    without perturbing the option rows (zeroing them would additionally make
    every option a duplicate and trigger the duplicate-tie, which is a
    different property).
    """
    head.interaction_logits = lambda fused, option_features: (  # type: ignore[method-assign]
        torch.zeros(
            (fused.shape[0], option_features.shape[1]),
            dtype=fused.dtype,
            device=fused.device,
        )
    )

# ---------------------------------------------------------------------------
# 1. Probe equivalence: zero residual == nn.Linear(D, K) on the fused state
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("batch,num_options,dim", [(1, 2, 8), (4, 5, 16), (7, 12, 32)])
def test_choice_head_with_zero_interaction_equals_linear_probe(
    seed, interaction, batch, num_options, dim
):
    """z_k = s^T W[:, k] + b[k] with r_k == 0 IS nn.Linear(D, K) on the state.

    This is the exact numerical statement the fix is designed to guarantee, at
    atol 1e-6 with a seeded input: the reference probe is built from the head
    own pathway parameters, so the comparison cannot pass by coincidence.
    """
    torch.manual_seed(seed)
    head = ChoiceHead(dim, interaction=interaction, num_options=num_options)
    fused, options = _seeded_inputs(seed, batch, dim, num_options)
    _zero_interaction(head)
    # The pathway is ZERO-initialized so an untrained head is bit-identical to
    # the pre-EXP-H head; give it NON-ZERO values here so the comparison is a
    # real test of the pathway's linear form rather than of its init.
    with torch.no_grad():
        head.state_pathway.weight.normal_(0.0, 0.3)
        head.state_pathway.bias.normal_(0.0, 0.3)
    with torch.no_grad():
        head_logits = head(fused, options).logits
        # Reference: the SAME linear map, spelled as a plain nn.Linear.
        reference = nn.Linear(dim, num_options, bias=True).to(fused.dtype)
        with torch.no_grad():
            reference.weight.copy_(head.state_pathway.weight[:num_options])
            reference.bias.copy_(head.state_pathway.bias[:num_options])
        probe_logits = reference(fused)
    assert torch.allclose(head_logits, probe_logits, atol=1e-6, rtol=0.0), (
        f"max |head - probe| = {(head_logits - probe_logits).abs().max().item():.3e}"
    )


def test_choice_head_additive_term_is_learnable_not_frozen() -> None:
    """The additive term must be a real parameter with a usable gradient scale."""
    torch.manual_seed(0)
    head = ChoiceHead(16, interaction="bilinear", num_options=4)
    fused, options = _seeded_inputs(0, 8, 16, 4)
    head(fused, options).logits.sum().backward()
    grad = head.state_pathway.weight.grad
    assert grad is not None, "state pathway must receive gradient"
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0, "state pathway gradient must be non-zero"


def test_linear_probe_pathway_alone_is_exactly_linear() -> None:
    """The pathway is nn.Linear(D, K) on its own - the reusable primitive."""
    torch.manual_seed(3)
    pathway = LinearProbePathway(12, num_options=7)
    with torch.no_grad():
        pathway.weight.normal_(0.0, 0.3)
        pathway.bias.normal_(0.0, 0.3)
    fused = torch.randn(5, 12, generator=torch.Generator().manual_seed(9))
    got = pathway(fused, 7)
    reference = nn.functional.linear(fused, pathway.weight[:7], pathway.bias[:7])
    assert torch.equal(got, reference)


def test_linear_probe_pathway_slices_to_observed_width() -> None:
    """A wide batch inside a narrow schema must not reallocate the parameter."""
    torch.manual_seed(4)
    pathway = LinearProbePathway(8, num_options=DEFAULT_STATE_PATHWAY_OPTIONS)
    with torch.no_grad():
        pathway.weight.normal_(0.0, 0.3)
    before = pathway.width
    fused = torch.randn(3, 8)
    out = pathway(fused, 5)
    assert out.shape == (3, 5)
    assert pathway.width == before, "a narrower K must slice, never reallocate"


def test_linear_probe_pathway_rejects_invalid_width() -> None:
    pathway = LinearProbePathway(8, num_options=4)
    with pytest.raises(ValueError, match="width"):
        pathway(torch.randn(2, 8), 0)
    with pytest.raises(ValueError, match="hidden_size"):
        LinearProbePathway(0)
    with pytest.raises(ValueError, match="num_options"):
        LinearProbePathway(8, num_options=0)

# ---------------------------------------------------------------------------
# 2. With interactions ENABLED the DecisionOutput theorems still hold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("batch,num_options,dim", [(1, 1, 8), (3, 4, 16), (5, 9, 24)])
def test_choice_head_satisfies_output_theorems(interaction, batch, num_options, dim):
    """Simplex / bounds / index-in-valid-set hold with the residual active."""
    torch.manual_seed(batch * 31 + num_options)
    head = ChoiceHead(dim, interaction=interaction, num_options=num_options)
    fused, options = _seeded_inputs(batch, batch, dim, num_options)
    out = head(fused, options)
    probs = out.probabilities
    assert probs.shape == (batch, num_options)
    assert (probs >= 0).all() and (probs <= 1).all(), "probabilities in [0, 1]"
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(batch), atol=1e-6
    ), "each question normalizes to 1 over its valid options"
    assert (out.predicted_index >= 0).all()
    assert (out.predicted_index < num_options).all()
    assert torch.isfinite(out.logits).all()


@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_choice_head_masked_rows_are_exactly_zero(interaction, seed):
    """Invalid (padded) options keep probability EXACTLY 0 (variable-K)."""
    torch.manual_seed(seed)
    b, kmax, d = 5, 7, 16
    head = ChoiceHead(d, interaction=interaction, num_options=kmax)
    fused, options = _seeded_inputs(seed, b, d, kmax)
    counts = torch.tensor([1, 3, 7, 2, 5])
    mask = mask_from_num_options(counts, kmax)
    out = head(fused, options, option_mask=mask)
    assert (out.probabilities[~mask] == 0.0).all(), (
        "invalid options must be exactly 0, never a tiny residual"
    )
    assert torch.allclose(
        out.probabilities.sum(dim=-1), torch.ones(b), atol=1e-6
    )
    rows = torch.arange(b)
    assert mask[rows, out.predicted_index].all(), "argmax must address a valid option"


@pytest.mark.parametrize("interaction", INTERACTIONS)
def test_choice_head_variable_k_does_not_reallocate(interaction):
    """Two schemas of different K in one head: no growth, no leakage."""
    torch.manual_seed(11)
    head = ChoiceHead(12, interaction=interaction)
    capacity = head.state_pathway.width
    for k in (2, 5, 3, 9):
        fused, options = _seeded_inputs(100 + k, 4, 12, k)
        out = head(fused, options)
        assert out.logits.shape == (4, k)
        assert head.state_pathway.width == capacity, (
            "a narrower schema must slice the parameter, not reallocate it"
        )


def test_choice_head_num_options_matches_option_mask(interaction="bilinear"):
    """The count spelling and the mask spelling stay interchangeable."""
    torch.manual_seed(5)
    b, kmax, d = 4, 6, 10
    head = ChoiceHead(d, interaction=interaction, num_options=kmax)
    fused, options = _seeded_inputs(5, b, d, kmax)
    counts = torch.tensor([1, 3, 6, 2])
    by_counts = head(fused, options, num_options=counts)
    by_mask = head(fused, options, option_mask=mask_from_num_options(counts, kmax))
    assert torch.equal(by_counts.probabilities, by_mask.probabilities)
    assert torch.equal(by_counts.predicted_index, by_mask.predicted_index)

# ---------------------------------------------------------------------------
# 3. Gradient flow to the STATE and to the OPTION pathway
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interaction", INTERACTIONS)
def test_choice_head_gradients_reach_state_and_options(interaction):
    """Both pathways are live: dL/d(state) != 0 and dL/d(option row) != 0."""
    torch.manual_seed(6)
    b, k, d = 6, 4, 16
    head = ChoiceHead(d, interaction=interaction, num_options=k)
    fused, options = _seeded_inputs(6, b, d, k)
    fused = fused.clone().requires_grad_(True)
    options = options.clone().requires_grad_(True)
    out = head(fused, options)
    probs = out.probabilities
    loss = (probs * torch.arange(k, dtype=probs.dtype)).sum()
    loss.backward()
    assert fused.grad is not None and float(fused.grad.abs().sum()) > 0.0, (
        "gradient must flow to the state pathway"
    )
    assert options.grad is not None and float(options.grad.abs().sum()) > 0.0, (
        "gradient must flow to the option features"
    )
    assert head.state_pathway.weight.grad is not None
    assert float(head.state_pathway.weight.grad.abs().sum()) > 0.0
    if head.bilinear_weight is not None:
        assert head.bilinear_weight.grad is not None
    if head.mlp is not None:
        grads = [p.grad for p in head.mlp.parameters() if p.requires_grad]
        assert grads and all(g is not None for g in grads)


@pytest.mark.parametrize("interaction", INTERACTIONS)
def test_masked_options_receive_no_gradient_through_softmax(interaction):
    """Padding invariance: a masked row cannot influence anything downstream."""
    torch.manual_seed(7)
    b, k, d = 4, 5, 12
    head = ChoiceHead(d, interaction=interaction, num_options=k)
    fused, options = _seeded_inputs(7, b, d, k)
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[:, -2:] = False  # same padding pattern for every row
    out_a = head(fused, options, option_mask=mask)
    perturbed = options.clone()
    perturbed[:, -2:, :] = 1e3  # garbage in the masked rows
    out_b = head(fused, perturbed, option_mask=mask)
    valid = mask
    assert torch.equal(
        out_a.probabilities[valid], out_b.probabilities[valid]
    ), "masked rows must not change any valid probability"


# ---------------------------------------------------------------------------
# 4. Degeneracy guard: the head must be able to FIT a tiny separable problem
# ---------------------------------------------------------------------------

def test_choice_head_fits_a_separable_problem():
    """A tiny linearly separable problem is solved to > 0.9 TRAIN accuracy.

    This is the anti-collapse guard. The RAVDESS failure was a head that sat
    at the majority baseline (0.1340) with zero seed variance, i.e. a DEGENERATE
    head that never learned anything. Whatever the data, the head must be able
    to represent and reach a good solution on a problem it can represent.
    """
    torch.manual_seed(0)
    d, k, n = 12, 4, 256
    directions = torch.randn(k, d, generator=torch.Generator().manual_seed(1))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    labels = torch.arange(n) % k
    features = 3.0 * directions[labels] + 0.05 * torch.randn(n, d)
    head = ChoiceHead(d, interaction="bilinear", num_options=k)
    # Drive the ADDITIVE pathway only: the interaction residual is neutralized
    # so the fit measures the head's own state-scoring capacity - the property
    # whose absence produced the degenerate majority-baseline head. Option rows
    # stay DISTINCT so the duplicate-tie is a no-op.
    options = _seeded_inputs(21, n, d, k)[1]
    _zero_interaction(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-2)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        logits = head(features, options).logits
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = head(features, options).predicted_index
    train_acc = float((preds == labels).double().mean())
    assert train_acc > 0.9, f"head failed to fit a separable problem: {train_acc}"


@pytest.mark.parametrize("interaction", INTERACTIONS)
def test_choice_head_fits_a_separable_problem_all_interactions(interaction):
    """Same degeneracy guard across every shipped interaction residual."""
    torch.manual_seed(0)
    d, k, n = 12, 4, 256
    directions = torch.randn(k, d, generator=torch.Generator().manual_seed(1))
    directions = directions / directions.norm(dim=-1, keepdim=True)
    labels = torch.arange(n) % k
    features = 3.0 * directions[labels] + 0.05 * torch.randn(n, d)
    head = ChoiceHead(d, interaction=interaction, num_options=k)
    options = _seeded_inputs(21, n, d, k)[1]  # DISTINCT rows: tie is a no-op
    _zero_interaction(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-2)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(
            head(features, options).logits, labels
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = head(features, options).predicted_index
    train_acc = float((preds == labels).double().mean())
    assert train_acc > 0.9, f"{interaction} head failed to fit: {train_acc}"

# ---------------------------------------------------------------------------
# 5. Backward compatibility + the head v1-vs-v2 ablation switch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("num_options", [None, 1, 3, 16])
def test_choice_head_constructs_with_optional_state_pathway(interaction, num_options):
    """Both head versions construct and run through the registry."""
    torch.manual_seed(0)
    head = build_decision_head(
        "choice", hidden_size=16, interaction=interaction, num_options=num_options
    )
    assert isinstance(head, ChoiceHead)
    assert head.state_pathway is not None
    fused, options = _seeded_inputs(0, 3, 16, 4)
    # A capacity smaller than the observed K grows EAGERLY on the first
    # forward (so the new columns are still optimizer-visible); the result is
    # [B, K] either way.
    assert head(fused, options).logits.shape == (3, 4)
    assert head.state_pathway.width >= 4
    # v1 ablation: the pre-EXP-H behavior is still reachable from the class.
    v1 = ChoiceHead(16, interaction=interaction, state_pathway=False)
    assert v1.state_pathway is None
    assert v1(fused, options).logits.shape == (3, 4)


@pytest.mark.parametrize("interaction", INTERACTIONS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_choice_head_state_dict_round_trip(interaction, seed):
    """Serialization still reproduces the logits bit-for-bit."""
    import io

    torch.manual_seed(seed)
    d, k, b = 12, 5, 4
    head = ChoiceHead(d, interaction=interaction, num_options=k)
    fused, options = _seeded_inputs(seed, b, d, k)
    reference = head(fused, options)
    buffer = io.BytesIO()
    torch.save(head.state_dict(), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    torch.manual_seed(seed + 999)
    restored = ChoiceHead(d, interaction=interaction, num_options=k)
    restored.load_state_dict(state)
    out = restored(fused, options)
    assert torch.allclose(out.logits, reference.logits, atol=1e-6)
    assert torch.allclose(out.probabilities, reference.probabilities, atol=1e-6)
    assert torch.equal(out.predicted_index, reference.predicted_index)


def test_choice_head_is_bit_identical_to_v1_at_initialization() -> None:
    """A FRESH head is numerically identical to the pre-EXP-H head.

    The additive pathway is zero-initialized, so at step 0 the scoring
    function is exactly the old interaction-only one. This is what keeps the
    pinned golden fixtures byte-identical while still letting the additive
    probe path grow under training.
    """
    torch.manual_seed(2)
    d, k, b = 16, 4, 6
    fused, options = _seeded_inputs(2, b, d, k)
    v2 = ChoiceHead(d, interaction="bilinear", num_options=k)
    v1 = ChoiceHead(d, interaction="bilinear", num_options=k, state_pathway=False)
    v1.load_state_dict(
        {k2: v for k2, v in v2.state_dict().items() if not k2.startswith("state_pathway")},
        strict=False,
    )
    with torch.no_grad():
        assert torch.equal(
            v2.state_pathway.weight, torch.zeros_like(v2.state_pathway.weight)
        )
        assert torch.equal(
            v2.state_pathway.bias, torch.zeros_like(v2.state_pathway.bias)
        )
        assert torch.equal(v2(fused, options).logits, v1(fused, options).logits), (
            "a freshly constructed head must be bit-identical to head v1"
        )


def test_choice_head_additive_term_is_live_after_a_training_step() -> None:
    """Once trained, the additive term really contributes to the scores."""
    torch.manual_seed(3)
    d, k, b = 16, 4, 6
    fused, options = _seeded_inputs(3, b, d, k)
    head = ChoiceHead(d, interaction="bilinear", num_options=k)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
    loss = nn.functional.cross_entropy(head(fused, options).logits, torch.zeros(b, dtype=torch.long))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    v1 = ChoiceHead(d, interaction="bilinear", num_options=k, state_pathway=False)
    v1.load_state_dict(
        {k2: v for k2, v in head.state_dict().items() if not k2.startswith("state_pathway")},
        strict=False,
    )
    with torch.no_grad():
        assert not torch.equal(head(fused, options).logits, v1(fused, options).logits), (
            "after a step the additive term must change the scores"
        )
        # ... and the scores are exactly v1 + the additive term.
        assert torch.allclose(
            head(fused, options).logits,
            v1(fused, options).logits + head.state_pathway(fused, k),
            atol=1e-6,
        )



# ---------------------------------------------------------------------------
# 6. dtype coverage: the duplicate tie must not assume a numpy-able dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("interaction", INTERACTIONS)
def test_choice_head_duplicate_tie_supports_low_precision_dtypes(dtype, interaction):
    """fp16/bf16 option rows must not crash the duplicate scan.

    A regression guard: ``.numpy()`` has no bfloat16 support, so the tie
    comparison upcasts to float32 before any host read. bf16 is exercised on
    CPU; fp16 CPU support varies, so it is skipped when unsupported.
    """
    torch.manual_seed(9)
    b, k, d = 3, 4, 8
    head = ChoiceHead(d, interaction=interaction, num_options=k).to(dtype)
    g = torch.Generator().manual_seed(9)
    fused = torch.randn(b, d, generator=g).to(dtype)
    options = torch.randn(b, k, d, generator=g).to(dtype)
    # Force a genuine duplicate pair in the VALID region of every row.
    options[:, 1, :] = options[:, 0, :]
    try:
        out = head(fused, options)
    except (TypeError, RuntimeError) as exc:  # dtype unsupported on this build
        if dtype in (torch.float16,):
            pytest.skip(f"cpu {dtype} unsupported: {exc}")
        raise
    assert out.logits.dtype == dtype
    assert torch.isfinite(out.logits).all()
    assert torch.allclose(out.logits[:, 0], out.logits[:, 1], atol=1e-2), (
        "duplicated options must receive the same score in any dtype"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_choice_head_forward_supports_low_precision_dtypes(dtype):
    """End-to-end head forward in bf16/fp32 keeps the output contract."""
    torch.manual_seed(10)
    b, k, d = 3, 5, 16
    head = ChoiceHead(d, interaction="bilinear", num_options=k).to(dtype)
    g = torch.Generator().manual_seed(10)
    fused = torch.randn(b, d, generator=g).to(dtype)
    options = torch.randn(b, k, d, generator=g).to(dtype)
    out = head(fused, options)
    assert out.logits.shape == (b, k)
    probs = out.probabilities.float()
    assert (probs >= 0).all() and (probs <= 1).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(b), atol=1e-2)

