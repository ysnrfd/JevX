# Copyright (c) 2026 ysnrfd
"""Unit tests for ``JevXForDecision``: construction, forward contract, gradients.

All models are in-memory (fixtures in ``tests/conftest.py``); no network.
The session-scoped ``tiny_bert`` fixture is treated as read-only: every
wrapper wraps a ``deepcopy`` so freeze/unfreeze/backward never mutate the
shared fixture. Expectations follow the ``jevx/modeling_jevx.py`` module
docstring (execution flow, decisions layout ``b*m + j``, abstention rules).
"""

from __future__ import annotations

import copy

import pytest
import torch

from jevx import JevXConfig, JevXForDecision, coverage_at_risk, risk_coverage_curve
from jevx.decisions import available_decision_heads
from jevx.modeling_outputs import DecisionOutput, JevXOutput

HIDDEN = 32  # matches the tiny_bert fixture's hidden_size
STATE = "hello world"
STATE2 = "foo bar baz"


def build_wrapper(backbone, tokenizer=None, **config_overrides) -> JevXForDecision:
    """Wrap ``backbone`` (deep-copied) with a tiny decision config and eval mode."""
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
    wrapper.eval()
    return wrapper


@pytest.fixture()
def wrapper(tiny_bert, tiny_tokenizer) -> JevXForDecision:
    return build_wrapper(tiny_bert, tokenizer=tiny_tokenizer)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_heads_and_capability_report(wrapper) -> None:
    report = wrapper.capability_report
    assert report is not None
    assert report.jevx_conversion == "supported"
    assert report.model_type == "bert"
    assert report.representation_field == "pooler_output"
    assert report.architecture
    assert report.reasons
    # heads built for all three primitives
    assert set(wrapper.heads) == {"choice", "score", "noul"}
    assert list(wrapper.config.decision_primitives) == ["choice", "score", "noul"]
    # family adapter resolved (bidirectional encoder, no decoder)
    assert wrapper.adapter.name == "encoder"


def test_freeze_base_model_true_freezes_backbone(tiny_bert, tiny_tokenizer) -> None:
    frozen = build_wrapper(tiny_bert, tiny_tokenizer, freeze_base_model=True)
    assert all(not p.requires_grad for p in frozen.model.parameters())
    trainable = list(frozen.trainable_parameters())
    assert trainable, "JevX-side parameters must be yielded"
    trainable_ids = {id(p) for p in trainable}
    backbone_ids = {id(p) for p in frozen.model.parameters()}
    assert trainable_ids.isdisjoint(backbone_ids)
    # yields exactly the parameters NOT named model.*
    expected = {
        id(p) for name, p in frozen.named_parameters() if not name.startswith("model.")
    }
    assert trainable_ids == expected


def test_freeze_base_model_false_keeps_backbone_trainable(tiny_bert, tiny_tokenizer) -> None:
    open_ = build_wrapper(tiny_bert, tiny_tokenizer, freeze_base_model=False)
    assert all(p.requires_grad for p in open_.model.parameters())


# ---------------------------------------------------------------------------
# Forward: choice / score / noul
# ---------------------------------------------------------------------------


def test_choice_forward_contract(wrapper) -> None:
    with torch.no_grad():
        out = wrapper.choice(
            STATE, ["q1", "q2"], options=[["a", "b"], ["c", "d", "e"]]
        )
    assert isinstance(out, JevXOutput)
    assert len(out.decisions) == 2
    key_sets = (["a", "b"], ["c", "d", "e"])
    for dec, keys in zip(out.decisions, key_sets):
        assert isinstance(dec, DecisionOutput)
        probs = dec.probabilities
        assert (probs >= 0).all()
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
        k = len(keys)
        assert 0 <= int(dec.predicted_index[0]) < k
        conf = float(dec.confidence[0])
        assert 0.0 <= conf <= 1.0
        assert dec.predicted_value in keys
        assert dec.abstained is None  # no threshold given
        assert dec.logits.shape[0] == 1  # head batch semantics, leading dim 1
        assert dec.schema_metadata["decision_type"] == "choice"
        assert dec.schema_metadata["option_keys"] == keys
    # output-level representation fields (docstring layout)
    assert out.state_representation.shape == (1, HIDDEN)
    assert out.question_representations.shape == (1, 2, HIDDEN)
    assert out.schema_representations is not None
    assert out.schema_representations.shape[0] == 2  # B*m rows
    assert out.schema_representations.shape[-1] == HIDDEN
    meta = out.capability_metadata
    assert meta is not None
    assert meta["pair_ordering"] == "flat index = b*m + j (state-major)"
    assert meta["num_states"] == 1
    assert meta["questions_per_state"] == 2
    assert meta["backbone_forward_calls"] == 1


def test_score_expected_score_bounds(wrapper) -> None:
    ks = (3, 5)
    with torch.no_grad():
        out = wrapper.score(
            STATE,
            ["s1", "s2"],
            options=[["0", "1", "2"], ["0", "1", "2", "3", "4"]],
        )
    for dec, k in zip(out.decisions, ks):
        assert dec.expected_score is not None
        es = float(dec.expected_score[0])
        assert -1e-6 <= es <= (k - 1) + 1e-6
        assert float(dec.probabilities.sum()) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= int(dec.predicted_index[0]) < k
        assert dec.schema_metadata["decision_type"] == "score"


def test_noul_range_and_options_forbidden(wrapper) -> None:
    with torch.no_grad():
        out = wrapper.noul(STATE, ["n1", "n2"])
    assert len(out.decisions) == 2
    for dec in out.decisions:
        p = float(dec.probabilities[0, 0])
        assert 0.0 < p < 1.0
        assert int(dec.predicted_index[0]) in (0, 1)
        assert dec.expected_score is None
        assert dec.schema_metadata["decision_type"] == "noul"
    # options-for-noul raises the documented ValueError (bug fix: used to TypeError)
    with pytest.raises(ValueError, match="takes no options") as exc:
        wrapper.noul(STATE, ["n1"], options=[["a", "b"]])
    msg = str(exc.value)
    assert "noul" in msg and "choice" in msg  # actionable: what to do instead


# ---------------------------------------------------------------------------
# Variable K, shared state, independence, mixed types, batch ordering
# ---------------------------------------------------------------------------


def test_variable_k_in_one_call(wrapper) -> None:
    ks = (2, 5, 12)
    options = [[f"q{j}-opt{i}" for i in range(k)] for j, k in enumerate(ks)]
    with torch.no_grad():
        out = wrapper.choice(STATE, ["k2", "k5", "k12"], options=options)
    assert len(out.decisions) == 3
    kmax = max(ks)
    for dec, k in zip(out.decisions, ks):
        probs = dec.probabilities
        assert probs.shape[-1] == kmax  # one layout padded to K_max
        assert (probs >= 0).all()
        # masked positions are exposed and exactly 0
        assert bool((probs[0, k:] == 0).all())
        # per-question normalization over the valid slice
        assert float(probs[0, :k].sum()) == pytest.approx(1.0, abs=1e-6)
        assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)
        assert 0 <= int(dec.predicted_index[0]) < k


def test_shared_state_single_backbone_call_and_batching(wrapper) -> None:
    qs = [f"question number {i}" for i in range(8)]
    opts = [[f"q{i}-a", f"q{i}-b", f"q{i}-c"] for i in range(8)]
    counter = {"n": 0}

    def hook(module, args, output):  # test instrument, not a mock
        counter["n"] += 1

    handle = wrapper.model.register_forward_hook(hook)
    try:
        for m in (1, 2, 4, 8):
            before = counter["n"]
            out = wrapper.choice(STATE, qs[:m], options=opts[:m])
            assert len(out.decisions) == m
            # THE shared-state property: exactly ONE backbone forward per call
            assert counter["n"] - before == 1, f"m={m} triggered multiple backbone calls"
    finally:
        handle.remove()

    # batched (one call, m=4) equals per-question one-at-a-time runs
    m = 4
    with torch.no_grad():
        batched = wrapper.choice(STATE, qs[:m], options=opts[:m])
        for j in range(m):
            solo = wrapper.choice(STATE, [qs[j]], options=[opts[j]])
            k = len(opts[j])
            assert torch.allclose(
                batched.decisions[j].probabilities[0, :k],
                solo.decisions[0].probabilities[0, :k],
                atol=1e-5,
                rtol=0.0,
            ), f"question {j}: batched != solo"
            assert torch.allclose(
                batched.decisions[j].confidence,
                solo.decisions[0].confidence,
                atol=1e-5,
                rtol=0.0,
            )


def test_question_independence_across_schemas(wrapper) -> None:
    with torch.no_grad():
        base = wrapper.choice(STATE, ["q one", "q two"], options=[["a", "b"], ["c", "d"]])
        perturbed = wrapper.choice(
            STATE, ["q one", "q two"], options=[["a", "b"], ["c", "d", "e", "f"]]
        )
    # q1 (K=2) unaffected by q2's schema growing to K=4
    assert torch.allclose(
        base.decisions[0].probabilities[0, :2],
        perturbed.decisions[0].probabilities[0, :2],
        atol=1e-6,
        rtol=0.0,
    )
    # (only q1's invariance is the contract — q2 is the perturbed question and
    # its distribution legitimately renormalizes over its new K=4 schema.)


def test_mixed_decision_types_one_call(wrapper) -> None:
    with torch.no_grad():
        out = wrapper.decide(
            STATE,
            questions=["pick", "rate", "flag"],
            decision_type=["choice", "score", "noul"],
            options=[["yes", "no"], ["1", "2", "3"], None],
        )
    assert len(out.decisions) == 3
    assert [d.schema_metadata["decision_type"] for d in out.decisions] == [
        "choice",
        "score",
        "noul",
    ]
    choice_dec, score_dec, noul_dec = out.decisions
    assert choice_dec.expected_score is None
    assert choice_dec.predicted_value in ("yes", "no")
    es = float(score_dec.expected_score[0])
    assert -1e-6 <= es <= 2 + 1e-6  # K=3 levels -> [0, 2]
    p = float(noul_dec.probabilities[0, 0])
    assert 0.0 < p < 1.0
    # softmax rows normalize on their own; noul is a sigmoid probability,
    # not a distribution (DecisionOutput contract: sum-to-1 except noul)
    for dec, k in zip(out.decisions[:2], (2, 3)):
        assert float(dec.probabilities[0, :k].sum()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < float(noul_dec.probabilities[0, 0]) < 1.0


def test_unknown_decision_type_lists_registered_heads(wrapper) -> None:
    with pytest.raises(ValueError) as exc:
        wrapper.decide(
            STATE, questions=["q"], decision_type="bogus", options=[["a", "b"]]
        )
    msg = str(exc.value)
    assert "bogus" in msg
    assert "registered heads are" in msg
    for name in available_decision_heads():
        assert name in msg


def test_batch_states_state_major_ordering(wrapper) -> None:
    states = [STATE, STATE2]
    qs = ["q one", "q two"]
    opts = [["a", "b"], ["c", "d", "e"]]
    with torch.no_grad():
        out = wrapper.choice(states, qs, options=opts)
    assert len(out.decisions) == 4  # B*m
    assert out.capability_metadata["num_states"] == 2
    assert out.capability_metadata["questions_per_state"] == 2
    assert out.state_representation.shape == (2, HIDDEN)
    # states must actually differ, else the ordering check is vacuous
    assert not torch.allclose(out.state_representation[0], out.state_representation[1])
    # entry b*m + j equals the same (state b, question j) computed alone
    for b, state in enumerate(states):
        for j in range(2):
            with torch.no_grad():
                solo = wrapper.choice(state, [qs[j]], options=[opts[j]])
            k = len(opts[j])
            got = out.decisions[b * 2 + j]
            assert torch.allclose(
                got.probabilities[0, :k],
                solo.decisions[0].probabilities[0, :k],
                atol=1e-5,
                rtol=0.0,
            ), f"entry {b}*2+{j} does not match its solo run"


# ---------------------------------------------------------------------------
# Abstention + failure transparency
# ---------------------------------------------------------------------------


def test_abstention_threshold_behavior(wrapper) -> None:
    # Zeroing the choice interaction makes every logit 0 -> uniform probs ->
    # confidence = 1/K, deterministically below 0.99 regardless of RNG stream.
    wrapper.heads["choice"].bilinear_weight.data.zero_()
    options = [["a", "b", "c"]]
    with torch.no_grad():
        base = wrapper.choice(STATE, ["q1"], options=options)
        assert float(base.decisions[0].confidence[0]) < 0.99

        abstained = wrapper.choice(
            STATE, ["q1"], options=options, confidence_threshold=0.99
        )
        d = abstained.decisions[0]
        assert bool(d.abstained[0]) is True
        assert d.predicted_index is None
        assert d.predicted_value is None
        assert d.expected_score is None
        # probabilities untouched by abstention
        assert torch.equal(d.probabilities, base.decisions[0].probabilities)

        committed = wrapper.choice(
            STATE, ["q1"], options=options, confidence_threshold=0.0
        )
        c = committed.decisions[0]
        assert bool(c.abstained[0]) is False
        assert c.predicted_index is not None
        assert c.predicted_value is not None

    with pytest.raises(ValueError, match="confidence_threshold"):
        wrapper.choice(STATE, ["q1"], options=[["a", "b"]], confidence_threshold=1.5)
    with pytest.raises(ValueError, match="confidence_threshold"):
        wrapper.choice(STATE, ["q1"], options=[["a", "b"]], confidence_threshold=-0.1)


def test_failure_messages_are_actionable(wrapper) -> None:
    with pytest.raises(ValueError) as exc:
        wrapper.forward(questions=["q"], options=[["a", "b"]])  # no state at all
    assert "state=" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        wrapper.forward(
            state=STATE,
            questions=["q"],
            input_ids=torch.ones(1, 2, dtype=torch.long),
        )
    assert "not both" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        wrapper.decide(STATE, questions=["q"], decision_type="choice")  # no options
    msg = str(exc.value)
    assert "requires a schema" in msg and "options" in msg

    with pytest.raises(ValueError) as exc:
        wrapper.choice(STATE, questions=[], options=[])
    assert "at least one question" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        wrapper.decide(STATE, questions=[{"wrong_key": "x"}], options=[["a"]])
    msg = str(exc.value)
    assert "'text'" in msg  # names the missing key
    assert "decision_type" in msg  # points at the per-question API instead


# ---------------------------------------------------------------------------
# Preserve-base-model (§31) and gradient contract (§69)
# ---------------------------------------------------------------------------


def test_base_forward_bitwise_equal_to_raw_backbone(tiny_bert, tiny_tokenizer) -> None:
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer)
    input_ids = torch.tensor([[2, 3, 4, 5], [6, 7, 1, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with torch.no_grad():
        wrapped_out = wrapper.base_forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        raw_out = tiny_bert(input_ids=input_ids, attention_mask=attention_mask)
    max_abs_error = float(
        (wrapped_out.last_hidden_state - raw_out.last_hidden_state).abs().max()
    )
    assert max_abs_error == 0.0
    assert torch.equal(wrapped_out.last_hidden_state, raw_out.last_hidden_state)
    assert torch.equal(wrapped_out.pooler_output, raw_out.pooler_output)
    # the wrapper embedding accessors must be preserved weights too
    assert torch.equal(
        wrapper.model.embeddings.word_embeddings.weight,
        tiny_bert.embeddings.word_embeddings.weight,
    )


def test_gradient_contract_backbone_frozen(tiny_bert, tiny_tokenizer) -> None:
    wrapper = build_wrapper(tiny_bert, tiny_tokenizer, freeze_base_model=True)
    out = wrapper.choice(STATE, ["q1", "q2"], options=[["a", "b"], ["c", "d", "e"]])
    loss = out.decisions[0].logits.sum() + out.decisions[1].logits.sum()
    loss.backward()
    # §69 Mode A: no backbone gradient is ever produced
    assert all(p.grad is None for p in wrapper.model.parameters())
    # JevX-side parameters receive non-None, nonzero gradients somewhere
    jevx_side = [
        (name, p)
        for name, p in wrapper.named_parameters()
        if not name.startswith("model.")
    ]
    with_grad = [p for _, p in jevx_side if p.grad is not None]
    assert with_grad, "JevX-side parameters must receive gradients"
    assert any(float(p.grad.abs().sum()) > 0 for p in with_grad)
    # flags unchanged by backward
    assert all(not p.requires_grad for p in wrapper.model.parameters())


def test_gradient_contract_backbone_unfrozen(tiny_bert, tiny_tokenizer) -> None:
    # EXPLICIT conditioning_mode="full": this pins the Mode-B gradient contract
    # (an unfrozen backbone RECEIVES a nonzero gradient). Since DECISION-3 the
    # shipped default is "off" - a state-only call whose option residual is off -
    # and on a FRESH head the zero-initialized linear state pathway makes
    # d(logits)/d(fused) exactly zero, so the default path transmits no gradient
    # to the backbone at all. The conditioned path is the one this contract is
    # about; the state-only gradient contract is pinned in
    # tests/unit/test_state_only_path.py::test_state_only_gradient_reaches_the_state_pathway
    # and in tests/unit/test_conditioning_off_matches_state_only.py.
    wrapper = build_wrapper(
        tiny_bert, tiny_tokenizer, freeze_base_model=False, conditioning_mode="full"
    )
    out = wrapper.choice(STATE, ["q1"], options=[["a", "b"]])
    out.decisions[0].logits.sum().backward()
    grads = [p.grad for p in wrapper.model.parameters()]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)


# ---------------------------------------------------------------------------
# Risk-coverage utilities (§62) — hand-computed examples
# ---------------------------------------------------------------------------


def test_risk_coverage_curve_hand_computed() -> None:
    confidences = [0.9, 0.8, 0.5, 0.2]
    correct = [True, True, False, False]
    coverage, risk = risk_coverage_curve(confidences, correct)
    # already sorted desc: errors 0,0,1,1 -> cumsum/i
    assert coverage.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert risk.tolist() == pytest.approx([0.0, 0.0, 1.0 / 3.0, 0.5])
    assert coverage.dtype == torch.float32 and risk.dtype == torch.float32
    # stable tie-breaking keeps original order
    cov2, risk2 = risk_coverage_curve([0.5, 0.5], [True, False])
    assert cov2.tolist() == pytest.approx([0.5, 1.0])
    assert risk2.tolist() == pytest.approx([0.0, 0.5])
    # non-sorted input is ranked by confidence first
    cov3, risk3 = risk_coverage_curve([0.2, 0.9], [False, True])
    assert cov3.tolist() == pytest.approx([0.5, 1.0])
    assert risk3.tolist() == pytest.approx([0.0, 0.5])
    # empty input -> empty curve
    c0, r0 = risk_coverage_curve([], [])
    assert c0.numel() == 0 and r0.numel() == 0


def test_coverage_at_risk_hand_computed() -> None:
    confidences = [0.9, 0.8, 0.5, 0.2]
    correct = [True, True, False, False]
    assert coverage_at_risk(confidences, correct, 0.0) == pytest.approx(0.5)
    assert coverage_at_risk(confidences, correct, 0.3) == pytest.approx(0.5)
    assert coverage_at_risk(confidences, correct, 0.5) == pytest.approx(1.0)
    # top decision wrong and zero budget -> nothing coverable
    assert coverage_at_risk([0.9], [False], 0.0) == 0.0
    assert coverage_at_risk([], [], 0.5) == 0.0


def test_risk_utility_error_cases() -> None:
    with pytest.raises(ValueError, match="confidences"):
        risk_coverage_curve([[0.9, 0.8]], [True, False])
    with pytest.raises(ValueError, match="correct_mask"):
        risk_coverage_curve([0.9, 0.8], [True])
    with pytest.raises(ValueError, match="max_risk"):
        coverage_at_risk([0.9], [True], 1.5)
    with pytest.raises(TypeError, match="max_risk"):
        coverage_at_risk([0.9], [True], "high")
