# Copyright (c) 2026 ysnrfd
"""The STATE-ONLY path (``condition_options=False``) must not be degenerate.

Defect this module pins. With ``condition_options=False`` the wrapper hands
the head EXACT-ZERO option features of the original ``[B, K, D]`` shape (to
keep the K-column / mask / schema-metadata contract) and
``ChoiceHead._tie_duplicate_options`` fired on them, so all K columns looked
like one duplicate group. The K state-only logits came out IDENTICAL (measured
per-row spread exactly ``0.0``): the softmax was uniform, and because the tie
gives every logit in the group the same value, the gradient into every
state-pathway parameter is ANALYTICALLY ZERO (only float32 round-off, ~3e-07
on a ``[6, 8]`` batch, survives - against ~2.7 for the same batch with the tie
off), so a state-only call could never be trained out of the uniform
distribution. Curriculum Stage A and the gate's state-only arm both measure
that arm, so the whole A-F ablation would have been reading a degenerate
model.

Corrected semantics: the duplicate-option tie is a statement about the
SCHEMA'S option rows. ``ChoiceHead.compute_logits(..., duplicate_tie=False)``
declares that the rows handed to the head are a shape placeholder rather than
a schema; ``JevXForDecision.forward`` passes it exactly when it substituted
the zeros. The per-CLASS state term ``state_k = s^T W[k, :] + b[k]`` is free
by construction and is never averaged. With conditioning ON the tie is left
on, so genuinely duplicated options still score identically (unchanged
behaviour).
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from jevx import JevXConfig, JevXForDecision
from jevx.decisions import ChoiceHead, build_decision_head

STATES = ["hello world", "foo bar", "baz cat", "the a", "hello cat", "foo a"]
SCHEMA = [["hello", "world", "foo"]]
OTHER_SCHEMA = [["bar", "baz", "cat"]]


def _wrapper(backbone, tokenizer, **overrides) -> JevXForDecision:
    """A tiny in-memory wrapper (the same recipe the curriculum tests use)."""
    cfg = JevXConfig(**overrides)
    model = JevXForDecision(
        cfg, backbone=copy.deepcopy(backbone), tokenizer=tokenizer
    )
    return model.eval()


def _nonzero_state_pathway(head: ChoiceHead, seed: int = 1234) -> None:
    """Give the ZERO-INIT state pathway deterministic, non-zero parameters.

    The shipped ``LinearProbePathway`` starts at exactly zero (so a fresh head
    is bit-identical to the pre-EXP-H one), which would mask the tie on its
    own: ``state_k == 0`` for every k whatever the tie does. Filling it first
    isolates the tie, which is what these tests measure.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        head.state_pathway.weight.copy_(
            torch.randn(head.state_pathway.width, head.hidden_size, generator=g) * 0.1
        )
        head.state_pathway.bias.copy_(
            torch.randn(head.state_pathway.width, generator=g) * 0.1
        )


def _state_only_logits(model: JevXForDecision, states, options) -> list[torch.Tensor]:
    """Raw ``[K]`` logits of every state-only decision of one call."""
    with torch.no_grad():
        out = model.choice(
            states,
            ["pick one"],
            options=options,
            condition_options=False,
        )
    return [d.logits.reshape(-1).clone() for d in out.decisions]


def _row_spread(rows: list[torch.Tensor]) -> float:
    """Largest within-row (max - min) over the batch: 0.0 == all equal."""
    return max(float(row.max() - row.min()) for row in rows)

# ---------------------------------------------------------------------------
# 1. The regression: state-only logits must not be all equal
# ---------------------------------------------------------------------------


def test_state_only_logits_are_not_all_equal_on_a_real_wrapper(
    tiny_bert, tiny_tokenizer
) -> None:
    """Seeded wrapper, state-only call: strictly positive per-row spread.

    The zeroed state pathway is filled first, because the shipped pathway is
    ZERO-INIT and would make every column equal for a reason that has nothing
    to do with the tie. With the defect present this assertion measured 0.0 at
    every lambda; the predicted index was constant 0 for every state.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _nonzero_state_pathway(model.heads["choice"])

    for lam in (1.0, 0.5, 0.0):
        model.heads["choice"].set_interaction_scale(lam)
        rows = _state_only_logits(model, STATES, SCHEMA)
        assert _row_spread(rows) > 0.0, (
            f"condition_options=False collapsed the K state-only logits to one "
            f"value at lambda={lam} (spread 0.0)"
        )
        picks = {int(row.argmax()) for row in rows}
        assert len(picks) > 1, (
            f"the predicted index is constant ({picks}) at lambda={lam}: a "
            "constant argmax over K identical logits is the degenerate model"
        )


def test_duplicate_tie_is_exactly_what_collapsed_the_state_only_path(
    tiny_bert, tiny_tokenizer
) -> None:
    """The control: on IDENTICAL tensors the tie is the whole difference.

    ``duplicate_tie=True`` is the pre-fix code path and must still produce a
    spread of exactly 0.0 on the state-only tensors; ``duplicate_tie=False``
    must not. If the first ever stops holding, the defect came back; if the
    second stops holding, the fix stopped working.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    head = model.heads["choice"]
    _nonzero_state_pathway(head)
    d = head.hidden_size
    g = torch.Generator().manual_seed(7)
    fused = torch.randn(4, d, generator=g)
    # Exactly what the wrapper substitutes: zeros of the [B, K, D] shape.
    zeros = torch.zeros(4, 3, d)

    with torch.no_grad():
        tied = head.compute_logits(
            fused, zeros, conditioning_enabled=False, duplicate_tie=True
        )
        free = head.compute_logits(
            fused, zeros, conditioning_enabled=False, duplicate_tie=False
        )
    tied_spread = float(
        (tied.max(dim=-1).values - tied.min(dim=-1).values).max()
    )
    free_spread = float(
        (free.max(dim=-1).values - free.min(dim=-1).values).max()
    )
    assert tied_spread == 0.0, (
        "zeroed option rows are one duplicate group, so the pre-fix path "
        f"still collapses the state term (measured {tied_spread})"
    )
    assert free_spread > 0.0, (
        "duplicate_tie=False must leave the per-class state term intact "
        f"(measured {free_spread})"
    )
    assert torch.equal(free, head._state_scores(fused, 3)[0]), (
        "with the tie off and no residual, z_k IS the state pathway: "
        "the state term is per-class by construction and is never averaged"
    )


def test_state_only_gradient_reaches_the_state_pathway(
    tiny_bert, tiny_tokenizer
) -> None:
    """Step 0: the state-only loss must actually reach the pathway.

    This is the sharpest statement of the defect. The tie gives every logit
    in the group the SAME value, so ``dL/dz_k`` is the same for all k and
    their sum - the only thing the per-class state pathway can respond to -
    is zero; the analytic gradient is therefore ZERO and only float32
    round-off (~1e-07) survives. The control below runs the SAME batch
    through the head with the tie forced back on and pins that contrast, so
    the assertion is about MAGNITUDE rather than a bare ``> 0`` that
    round-off would satisfy.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    model.train()
    out = model.choice(
        STATES,
        ["pick one"],
        options=SCHEMA,
        condition_options=False,
    )
    logits = torch.stack([d.logits.reshape(-1) for d in out.decisions])
    # Deliberately UNBALANCED labels: with a ZERO-init state pathway the
    # bias gradient is sum_batch (p_k - y_k), which cancels to 0 for a
    # class-balanced target set for a reason that has nothing to do with
    # the tie. One label per class would hide the signal this test is
    # about.
    targets = torch.tensor([0, 0, 0, 0, 0, 1])
    nn.functional.cross_entropy(logits, targets).backward()
    head = model.heads["choice"]
    live = {
        name: float(param.grad.detach().abs().max())
        for name, param in head.state_pathway.named_parameters()
    }

    # The tied control: same head, same labels, tie forced back ON.
    # Gradients ACCUMULATE, so clear the wrapper's contribution first.
    head.zero_grad(set_to_none=True)
    g = torch.Generator().manual_seed(3)
    zeros = torch.zeros(len(STATES), logits.shape[1], head.hidden_size)
    tied_logits = head.compute_logits(
        torch.randn(len(STATES), head.hidden_size, generator=g),
        zeros,
        conditioning_enabled=False,
        duplicate_tie=True,
    )
    nn.functional.cross_entropy(tied_logits, targets).backward()
    tied = float(head.state_pathway.weight.grad.detach().abs().max())
    assert tied < 1e-5, (
        "the tied control should be (near) zero; it measured "
        f"{tied} - the degeneracy no longer reproduces"
    )

    for name, magnitude in live.items():
        assert magnitude > 1e-3, (
            f"{name} received a ROUND-OFF-LEVEL state-only gradient "
            f"(max {magnitude:.3e}, against a tied {tied:.3e}): the state "
            f"pathway cannot train"
        )


# ---------------------------------------------------------------------------
# 2. The tie must still fire where it is a statement about the SCHEMA
# ---------------------------------------------------------------------------


def test_duplicate_option_tie_still_fires_with_conditioning_on() -> None:
    """Two GENUINELY duplicated options still receive the same score.

    The state pathway is deliberately non-zero here, so the assertion is the
    strong one: the TOTAL logits z_k = state_k + lambda*r_k tie, not merely
    the residual. This is the behaviour the pre-existing duplicate-tie tests
    pin and the fix must not weaken.
    """
    torch.manual_seed(6)
    head = build_decision_head(
        "choice", hidden_size=16, num_options=4, state_pathway="linear"
    )
    _nonzero_state_pathway(head, seed=6)
    g = torch.Generator().manual_seed(6)
    fused = torch.randn(3, 16, generator=g)
    options = torch.randn(3, 4, 16, generator=g)
    options[:, 1, :] = options[:, 0, :]  # a genuine duplicate
    with torch.no_grad():
        logits = head(fused, options).logits
    assert torch.allclose(logits[:, 0], logits[:, 1], atol=1e-6), (
        "identical VALID option rows must still receive identical scores "
        f"(got {logits[:, 0].tolist()} vs {logits[:, 1].tolist()})"
    )
    assert not torch.allclose(logits[:, 0], logits[:, 2], atol=1e-6), (
        "distinct options must NOT be tied: the tie may not over-fire"
    )
    # The masked/padded case is unchanged too.
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[:, 3:] = False
    with torch.no_grad():
        masked = head(fused, options, option_mask=mask).logits
    assert torch.allclose(masked[:, 0], masked[:, 1], atol=1e-6)


def test_duplicate_option_tie_fires_end_to_end_through_the_wrapper(
    tiny_bert, tiny_tokenizer
) -> None:
    """The same rule on the real wrapper, on a schema that really repeats.

    condition_options=True leaves the tie on, so a schema listing the same
    option twice must score its two columns identically, while the distinct
    third column must not be dragged in.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _nonzero_state_pathway(model.heads["choice"])
    schema = [["hello", "hello", "world"]]
    with torch.no_grad():
        out = model.choice(
            STATES,
            ["pick one"],
            options=schema,
            condition_options=True,
        )
    for decision in out.decisions:
        logits = decision.logits.reshape(-1)
        assert torch.allclose(logits[0], logits[1], atol=1e-6), (
        "the repeated option must tie on the real wrapper: "
        f"{logits.tolist()}"
        )
        assert not torch.allclose(logits[0], logits[2], atol=1e-6), (
            "the DISTINCT option must not be tied"
        )

# ---------------------------------------------------------------------------
# 3. State-only TRAINING: the arm must actually learn
# ---------------------------------------------------------------------------


TRAIN_STATES = [
    "hello hello", "hello world", "world world", "world foo",
    "foo foo", "foo bar", "bar bar", "bar baz",
]
TRAIN_LABELS = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
TRAIN_OPTIONS = [["hello", "world", "foo"]]


def _state_only_logits_for(model, states, options) -> torch.Tensor:
    """Stacked ``[B, K]`` state-only logits of one call.
    """
    with torch.no_grad():
        out = model.choice(
            states,
            ["pick one"],
            options=options,
            condition_options=False,
        )
    return torch.stack([d.logits.reshape(-1) for d in out.decisions])


def _train_state_only(model, steps: int = 60, lr: float = 0.05) -> torch.Tensor:
    """Fit the state-only arm on a token-separable problem.

    Only the JevX layers train (the backbone is frozen by
    freeze_base_model), and the loss is the plain cross-entropy of the
    STATE-ONLY logits, i.e. exactly what curriculum Stage A optimises.
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model.choice(
            TRAIN_STATES,
            ["pick one"],
            options=TRAIN_OPTIONS,
            condition_options=False,
        )
        logits = torch.stack([d.logits.reshape(-1) for d in out.decisions])
        nn.functional.cross_entropy(logits, TRAIN_LABELS).backward()
        optimizer.step()
    model.eval()
    return _state_only_logits_for(model, TRAIN_STATES, TRAIN_OPTIONS)


def test_state_only_training_improves_accuracy_over_its_init(
    tiny_bert, tiny_tokenizer
) -> None:
    """Training the state-only arm must move accuracy above its init.

    At init the state pathway is ZERO, so every column is equal, the softmax
    is uniform and the argmax is constant: accuracy is chance. With the tie
    defect present this test could not pass at all - every gradient was
    exactly zero, so the arm stayed at chance forever.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)

    init_logits = _state_only_logits_for(model, TRAIN_STATES, TRAIN_OPTIONS)
    init_acc = float((init_logits.argmax(dim=-1) == TRAIN_LABELS).double().mean())

    final_logits = _train_state_only(model)
    final_acc = float((final_logits.argmax(dim=-1) == TRAIN_LABELS).double().mean())
    assert final_acc > init_acc + 0.4, (
        f"the state-only arm did not learn: accuracy {init_acc} -> {final_acc}"
    )
    spread = float(
        (final_logits.max(dim=-1).values - final_logits.min(dim=-1).values).max()
    )
    assert spread > 0.0, (
        "after training the state-only logits must still differ across options"
    )

# ---------------------------------------------------------------------------
# 4. The masked / simplex / theorem invariants still hold state-only
# ---------------------------------------------------------------------------


def test_state_only_path_preserves_mask_and_simplex_invariants(
    tiny_bert, tiny_tokenizer
) -> None:
    """Masking, the per-question simplex and argmax-in-valid-set, state-only.

    A variable-K batch (two questions with 2 and 4 options) exercises the
    padded-column path: the mask is still honoured, probabilities are exactly
    0 on invalid columns and still sum to 1, and the predicted index still
    addresses a valid option. These are the runtime ValidationEngine checks;
    the state-only path must satisfy them with non-degenerate logits.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _nonzero_state_pathway(model.heads["choice"])
    schemas = [["hello", "world"], ["a", "b", "c", "foo"]]
    with torch.no_grad():
        out = model.choice(
            STATES,
            ["pick one", "pick two"],
            options=schemas,
            condition_options=False,
        )
    assert len(out.decisions) == 2 * len(STATES)
    # question-major: q0 has K=2, q1 has K=4 (padded to the batch Kmax=4)
    for pair in range(len(STATES)):
        for q, k in ((0, 2), (1, 4)):
            decision = out.decisions[pair * 2 + q]
            logits = decision.logits.reshape(-1)
            probs = decision.probabilities.reshape(-1)
            assert logits.shape == (4,)
            assert torch.isfinite(logits).all()
            assert torch.allclose(
                probs.sum(), torch.tensor(1.0), atol=1e-6
            ), "each question normalizes to 1 over its VALID options"
            if k < probs.numel():
                assert float(probs[k:].abs().max()) == 0.0, (
                    "padded columns must be exactly 0 on the state-only path")
            assert 0 <= int(decision.predicted_index) < k, (
                "argmax must address a valid option")
            assert logits[:k].max() >= logits[:k].min()


def test_state_only_logits_stay_schema_independent(tiny_bert, tiny_tokenizer) -> None:
    """Two different schemas, one state-only call: identical logits.

    The state-only arm must read the STATE and not the schema. This is the
    property curriculum Stage A actually needs, and it is the assertion the
    curriculum suite makes through the real wrapper - now on logits that are
    NOT all equal, so it is a real statement rather than a tautology.
    """
    torch.manual_seed(0)
    model = _wrapper(tiny_bert, tiny_tokenizer)
    _nonzero_state_pathway(model.heads["choice"])
    a = _state_only_logits(model, STATES, SCHEMA)
    b = _state_only_logits(model, STATES, OTHER_SCHEMA)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), (
        "the schema changed the state-only logits"
    )
    assert _row_spread(a) > 0.0, (
        "schema-independence must not be the trivial all-equal case"
    )


# ---------------------------------------------------------------------------
# 5. D3: the config 0.0 -> head 1.0 mapping, pinned
# ---------------------------------------------------------------------------


def test_fresh_model_maps_config_zero_conditioning_scale_to_one(
    tiny_bert, tiny_tokenizer
) -> None:
    """A config of conditioning_scale_init=0.0 builds a head at lambda=1.0.

    This mapping is DELIBERATE and is the answer to the config/head
    inconsistency: JevXConfig.conditioning_scale_init defaults to 0.0, but
    the heads' own default lambda is 1.0, which is the pre-flag scoring
    function z_k = state_k + r_k, so 0.0 is read as 'unspecified'. Changing it
    to forward 0.0 verbatim would silently break PRE-EXISTING checkpoints:
    interaction_scale is a persistent buffer, so a checkpoint written before
    it existed loads with strict=False and KEEPS the constructor value - it
    would quietly lose its residual. A non-zero config value is forwarded
    verbatim, and lambda=0 stays reachable at runtime through
    set_interaction_scale(0.0) / condition_options=False.
    """
    assert JevXConfig().conditioning_scale_init == 0.0
    torch.manual_seed(0)
    default_model = _wrapper(tiny_bert, tiny_tokenizer)
    assert default_model.config.conditioning_scale_init == 0.0
    assert default_model.heads["choice"].get_interaction_scale == 1.0, (
        "config 0.0 on a FRESH model must build the head at lambda=1.0"
    )
    assert default_model.heads["choice"].get_interaction_scale == (
        JevXForDecision._state_pathway_kwargs(JevXConfig()).get(
            "conditioning_scale_init", 1.0)
        )

    torch.manual_seed(0)
    ramped = _wrapper(tiny_bert, tiny_tokenizer, conditioning_scale_init=0.6)
    assert ramped.heads["choice"].get_interaction_scale == pytest.approx(0.6), (
        "a non-zero configured lambda is forwarded verbatim"
    )
    # 0.0 is still reachable at runtime, which is the state-only path.
    ramped.heads["choice"].set_interaction_scale(0.0)
    assert ramped.heads["choice"].get_interaction_scale == 0.0

