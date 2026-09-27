# Copyright (c) 2026 ysnrfd
"""Unit tests for ``jevx.conversion``: conversion reports, equivalence,
transplant honesty, and the ``convert()`` orchestration API.

All models are in-memory constructions (fixtures in ``tests/conftest.py``);
the one directory-based case saves a raw ``BertModel`` with
``save_pretrained`` into ``tmp_path`` — fully offline, no network. The
session-scoped fixtures are treated as read-only: every conversion wraps a
``deepcopy`` so freeze/eval/transplant never mutate shared state.

The assertions encode the honesty invariants of
``jevx/conversion/report.py``: a report can never claim equivalence it did
not measure, and never claim a transplant that mapped nothing; skipped
checks are not passed.
"""

from __future__ import annotations

import copy
import json

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification

import jevx
from jevx import (
    ConversionMode,
    ConversionReport,
    EquivalenceKind,
    JevXForDecision,
    convert,
)
import jevx.conversion as conversion_pkg
from jevx.conversion import (
    CompatReport,
    check_base_equivalence,
    check_decision_roundtrip,
    check_representation_equivalence,
    probe_head_compat,
    transplant_classifier_head,
    transplant_state_encoder_proj,
)
from jevx.decisions import build_decision_head

HIDDEN = 32  # matches the tiny_bert fixture's hidden_size
STATE = "hello world"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _inputs() -> dict:
    """Fixed pre-tokenized batch (ids within the fixture vocab)."""
    return {
        "input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
    }


def _tiny_classifier(hidden: int = HIDDEN, num_labels: int = 2) -> BertForSequenceClassification:
    """Local ``BertForSequenceClassification`` (offline, no downloads).

    The RNG state is saved/restored so constructing the head cannot shift
    global randomness for later tests.
    """
    config = BertConfig(
        vocab_size=99,
        hidden_size=hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=num_labels,
    )
    state = torch.get_rng_state()
    try:
        torch.manual_seed(7)
        return BertForSequenceClassification(config).eval()
    finally:
        torch.set_rng_state(state)


def _score_head(num_levels: int, hidden: int = HIDDEN):
    """A ``ScoreHead`` with the option-independent ``level_proj: Linear(D->K)``."""
    return build_decision_head(
        "score", hidden_size=hidden, interaction="dot", num_levels=num_levels
    )


def _assert_no_unmeasured_passes(report: ConversionReport) -> None:
    """Honesty invariant 1/2, as a reusable assertion for every report."""
    for check in report.checks:
        assert set(("name", "value", "tolerance", "passed")) <= set(check)
        if check["passed"] is True:
            assert check["value"] is not None, check
            assert check["tolerance"] is not None, check
        if check["passed"] is None:
            assert check.get("reason"), check


# ---------------------------------------------------------------------------
# convert(): mode, backbone identity, report round trip
# ---------------------------------------------------------------------------


def test_convert_in_memory_bert_mode_and_report(tiny_bert, tiny_tokenizer) -> None:
    model, report = convert(copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer)
    assert isinstance(model, JevXForDecision)
    assert isinstance(report, ConversionReport)
    # fresh wrap: backbone verbatim, JevX heads freshly initialized
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.mapped_params == {}
    assert report.source.startswith("BertModel")
    assert report.target == "JevXForDecision"
    assert report.jevx_version == jevx.__version__
    # new params are exactly the JevX-side (non-backbone) keys
    assert report.new_params
    assert all(not key.startswith("model.") for key in report.new_params)
    jevx_keys = {
        k for k in model.state_dict() if not k.startswith("model.")
    }
    assert set(report.new_params) == jevx_keys
    # a pre-wrap reference snapshot exists and passed -> FUNCTIONAL measured
    functional = [c for c in report.checks if c["name"] == "functional_equivalence"]
    assert functional, report.checks
    assert functional[0]["passed"] is True
    assert functional[0]["value"] == 0.0
    assert functional[0]["tolerance"] == 0.0
    assert report.equivalence is EquivalenceKind.FUNCTIONAL
    _assert_no_unmeasured_passes(report)
    assert "representation_reuse" in report.summary()


def test_convert_does_not_touch_backbone_weights(tiny_bert) -> None:
    model = copy.deepcopy(tiny_bert)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wrapper, _ = convert(model)
    after = wrapper.model.state_dict()
    assert set(after) == set(before)
    for key, value in before.items():
        assert torch.equal(after[key], value), key
    # and the shared fixture itself is bit-identical to the embedded backbone
    for key, value in tiny_bert.state_dict().items():
        assert torch.equal(after[key], value), key


def test_report_json_roundtrip_exact(tmp_path) -> None:
    report = ConversionReport(
        mode=ConversionMode.WEIGHT_TRANSPLANT,
        equivalence=EquivalenceKind.NONE,
        source="BertForSequenceClassification",
        target="JevXForDecision",
        mapped_params={"classifier.bias": "heads.noul.proj.bias"},
        unmapped_source=["classifier.weight"],
        new_params=["state_proj.proj.weight"],
        checks=[
            {"name": "measured", "value": 0.0, "tolerance": 0.0, "passed": True},
            {"name": "failed", "value": 1.5, "tolerance": 0.0, "passed": False},
            {
                "name": "skipped",
                "value": None,
                "tolerance": None,
                "passed": None,
                "reason": "no featurizer",
            },
        ],
        warnings=["bias-only: class priors carried, weight stays fresh"],
        jevx_version=jevx.__version__,
    )
    path = tmp_path / "report.json"
    report.save(path)
    # raw JSON agrees with to_dict() exactly
    assert json.loads(path.read_text(encoding="utf-8")) == report.to_dict()
    loaded = ConversionReport.load(path)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.mode is ConversionMode.WEIGHT_TRANSPLANT
    assert loaded.equivalence is EquivalenceKind.NONE
    assert loaded.checks[2]["reason"] == "no featurizer"
    # plain-string construction coerces into the enums
    coerced = ConversionReport(
        mode="weight_transplant",
        equivalence="none",
        source="s",
        mapped_params={"a": "b"},
    )
    assert coerced.mode is ConversionMode.WEIGHT_TRANSPLANT
    assert coerced.equivalence is EquivalenceKind.NONE
    # summary phrasing
    assert "transplanted 1 tensor(s)" in report.summary()
    empty = ConversionReport(mode="representation_reuse", source="s")
    assert "no transplant" in empty.summary()


# ---------------------------------------------------------------------------
# equivalence checkers
# ---------------------------------------------------------------------------


def test_check_base_equivalence_zero_and_can_fail(tiny_bert) -> None:
    wrapper, _ = convert(copy.deepcopy(tiny_bert))
    inputs = _inputs()
    result = check_base_equivalence(wrapper, tiny_bert, inputs)
    assert result["max_abs"] == 0.0
    assert result["mean_abs"] == 0.0
    assert result["rel"] == 0.0
    assert result["bitwise"] is True
    assert result["passed"] is True
    assert result["tolerance"] == 0.0
    assert result["n_tensors"] >= 1
    # a perturbed reference must FAIL — the checker can measure disagreement
    perturbed = copy.deepcopy(tiny_bert)
    perturbed.embeddings.word_embeddings.weight.data.add_(1.0)
    bad = check_base_equivalence(wrapper, perturbed, inputs)
    assert bad["passed"] is False
    assert bad["max_abs"] > 0.0
    # structural contract: empty inputs are a caller error, not a verdict
    with pytest.raises(ValueError, match="non-empty"):
        check_base_equivalence(wrapper, tiny_bert, {})


def test_check_decision_roundtrip_identical(tiny_bert, tiny_tokenizer) -> None:
    wrapper, _ = convert(copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer)
    result = check_decision_roundtrip(
        wrapper, state=STATE, questions=["q1?"], options=[["a", "b"]]
    )
    assert result["max_abs"] == 0.0
    assert result["tolerance"] == 0.0
    assert result["bitwise"] is True
    assert result["passed"] is True
    assert result["n_tensors"] > 0


def test_check_representation_equivalence_native_and_forced(tiny_bert) -> None:
    inputs = _inputs()
    # default: BERT exposes pooler_output, adapter takes the native field and
    # the reference is that same field of an independent raw forward
    wrapper, _ = convert(copy.deepcopy(tiny_bert))
    native = check_representation_equivalence(wrapper, inputs)
    assert native["passed"] is True, native
    assert native["value"] <= native["tolerance"]
    assert native["field"] == "pooler_output"
    assert native["reference"] == "raw_output.pooler_output"
    assert "ulp" in native["rationale"]
    # forced sequence pooling: independent fp32 masked-mean reference
    forced, _ = convert(copy.deepcopy(tiny_bert), pooling_method="masked_mean")
    pooled = check_representation_equivalence(forced, inputs)
    assert pooled["passed"] is True, pooled
    assert pooled["strategy"] == "masked_mean"
    assert pooled["value"] <= pooled["tolerance"]
    assert "independent" in pooled["reference"]


# ---------------------------------------------------------------------------
# transplant: the true mappings
# ---------------------------------------------------------------------------


def test_transplant_linear_full_mapping(tiny_bert, tiny_tokenizer) -> None:
    wrapper, _ = convert(copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer)
    # the wrapper's ScoreHead is replaced by the fixed-level schema so the
    # target exposes the option-independent Linear(D -> K)
    wrapper.heads["score"] = _score_head(num_levels=2)
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=2)

    report = transplant_classifier_head(clf, wrapper.heads["score"])
    assert report.mapped_params == {
        "classifier.weight": "level_proj.weight",
        "classifier.bias": "level_proj.bias",
    }
    # mapped target weights EQUAL the source rows, numerically
    target = wrapper.heads["score"]
    assert torch.equal(target.level_proj.weight, clf.classifier.weight)
    assert torch.equal(target.level_proj.bias, clf.classifier.bias)
    # every target parameter was written -> HEAD_ATTACHMENT
    assert report.mode is ConversionMode.HEAD_ATTACHMENT
    assert report.unmapped_source == []  # bert.* backbone copy excluded
    assert report.new_params == []
    assert report.equivalence is EquivalenceKind.NONE  # never claims agreement
    identity = [c for c in report.checks if c["name"] == "transplant_tensor_identity"]
    assert identity and identity[0]["passed"] is True and identity[0]["value"] == 0.0
    _assert_no_unmeasured_passes(report)
    # the graft never breaks the untouched choice path
    out = wrapper.choice(STATE, ["q1?"], options=[["a", "b"]])
    assert len(out.decisions) == 1


def test_transplant_binary_gauge_into_noul(tiny_bert) -> None:
    wrapper, _ = convert(copy.deepcopy(tiny_bert))
    head = wrapper.heads["noul"]  # proj: Linear(D -> 1)
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=2)

    report = transplant_classifier_head(clf, head)
    assert report.mapped_params == {
        "classifier.weight": "proj.weight",
        "classifier.bias": "proj.bias",
    }
    # exact under softmax gauge invariance: z = z1 - z0
    expected_weight = (clf.classifier.weight[1] - clf.classifier.weight[0]).detach()
    assert torch.equal(head.proj.weight.data[0], expected_weight)
    expected_bias = float((clf.classifier.bias[1] - clf.classifier.bias[0]).detach())
    assert float(head.proj.bias.data[0].detach()) == expected_bias
    assert report.mode is ConversionMode.HEAD_ATTACHMENT
    assert any("gauge" in w for w in report.warnings)
    assert report.equivalence is EquivalenceKind.NONE
    _assert_no_unmeasured_passes(report)


def test_transplant_bias_only_when_widths_differ() -> None:
    clf = _tiny_classifier(hidden=64, num_labels=2)  # D_src = 64
    target = _score_head(num_levels=2)  # D_tgt = 32
    weight_before = target.level_proj.weight.detach().clone()

    report = transplant_classifier_head(clf, target)
    assert report.mode is ConversionMode.WEIGHT_TRANSPLANT  # partial write
    assert report.mapped_params == {"classifier.bias": "level_proj.bias"}
    assert torch.equal(target.level_proj.weight, weight_before)  # untouched
    assert torch.equal(target.level_proj.bias, clf.classifier.bias)
    assert "classifier.weight" in report.unmapped_source
    # probabilistic rationale is documented in the warnings
    assert any("log p(y)" in w and "prior" in w for w in report.warnings)
    assert any("NOT an equivalence claim" in w for w in report.warnings)
    _assert_no_unmeasured_passes(report)


def test_transplant_k_mismatch_no_mapping() -> None:
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=3)  # K_src = 3
    target = _score_head(num_levels=2)  # K_tgt = 2
    snapshot = {k: v.detach().clone() for k, v in target.state_dict().items()}

    report = transplant_classifier_head(clf, target)
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.mapped_params == {}
    assert set(report.unmapped_source) >= {"classifier.weight", "classifier.bias"}
    assert report.warnings
    assert any("NOT a transplant" in w for w in report.warnings)
    # target weights unchanged (pre-transplant snapshot)
    for key, value in snapshot.items():
        assert torch.equal(target.state_dict()[key], value), key
    assert "no transplant" in report.summary()
    _assert_no_unmeasured_passes(report)


def test_transplant_choice_head_refused(tiny_bert, tiny_tokenizer) -> None:
    """ChoiceHead has no option-independent linear -> honest no-mapping."""
    wrapper, _ = convert(copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer)
    choice = wrapper.heads["choice"]
    snapshot = {k: v.detach().clone() for k, v in choice.state_dict().items()}
    clf = _tiny_classifier()

    report = transplant_classifier_head(clf, choice)
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.mapped_params == {}
    assert report.unmapped_source
    assert any("option-independent" in w for w in report.warnings)
    assert "no transplant" in report.summary()
    assert report.equivalence is EquivalenceKind.NONE
    for key, value in snapshot.items():
        assert torch.equal(choice.state_dict()[key], value), key
    # the wrapper still works after the refused graft
    out = wrapper.choice(STATE, ["q1?"], options=[["a", "b"]])
    assert len(out.decisions) == 1


def test_transplant_unsupported_source_head() -> None:
    """A source with no classifier-like Linear maps nothing, with a warning."""
    target = _score_head(num_levels=2)
    snapshot = {k: v.detach().clone() for k, v in target.state_dict().items()}

    report = transplant_classifier_head(nn.ReLU(), target)
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.mapped_params == {}
    assert report.warnings
    assert any("NOT a transplant" in w for w in report.warnings)
    assert "no transplant" in report.summary()
    for key, value in snapshot.items():
        assert torch.equal(target.state_dict()[key], value), key
    # non-modules are caller errors, not reports
    with pytest.raises(TypeError, match="nn.Module"):
        transplant_classifier_head(42, target)


# ---------------------------------------------------------------------------
# probe_head_compat
# ---------------------------------------------------------------------------


def test_probe_head_compat_shapes_and_errors() -> None:
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=2)
    target = _score_head(num_levels=2)

    compat = probe_head_compat(clf, target)
    assert isinstance(compat, CompatReport)
    assert compat.source == "BertForSequenceClassification(classifier)"
    assert compat.source_attr == "classifier"
    assert compat.target_attr == "level_proj"
    assert compat.source_shape == (2, HIDDEN)
    assert compat.target_shape == (2, HIDDEN)
    assert compat.source_out_features == compat.target_out_features == 2
    assert compat.source_in_features == compat.target_in_features == HIDDEN
    assert compat.source_bias is compat.target_bias is True
    assert compat.compatible is True
    assert compat.reasons == []
    assert compat.to_dict()["compatible"] is True

    # garbage input -> informative errors
    with pytest.raises(TypeError, match="nn.Module"):
        probe_head_compat(42, target)
    with pytest.raises(TypeError, match="nn.Module"):
        probe_head_compat(clf, 42)
    with pytest.raises(ValueError, match="classifier"):
        probe_head_compat(nn.ReLU(), target)

    # a legitimate head without a linear scorer is reported, not raised
    choice = build_decision_head("choice", hidden_size=HIDDEN)
    refused = probe_head_compat(clf, choice)
    assert refused.compatible is False
    assert refused.target_attr is None
    assert refused.reasons

    # mismatched shapes are spelled out
    wide = probe_head_compat(_tiny_classifier(hidden=64, num_labels=2), target)
    assert wide.compatible is False
    assert any("state width mismatch" in r for r in wide.reasons)


# ---------------------------------------------------------------------------
# transplant_state_encoder_proj
# ---------------------------------------------------------------------------


def test_transplant_state_encoder_proj_exact_and_backbone_protection(tiny_bert) -> None:
    wrapper, _ = convert(copy.deepcopy(tiny_bert))
    # exact key+shape match into a chosen submodule
    source = nn.Linear(HIDDEN, HIDDEN)
    report = transplant_state_encoder_proj(source, wrapper.state_proj.proj)
    assert report.mapped_params == {"weight": "weight", "bias": "bias"}
    assert torch.equal(wrapper.state_proj.proj.weight, source.weight)
    assert torch.equal(wrapper.state_proj.proj.bias, source.bias)
    assert report.mode is ConversionMode.HEAD_ATTACHMENT  # fully written
    identity = [c for c in report.checks if c["name"] == "transplant_tensor_identity"]
    assert identity and identity[0]["passed"] is True

    # no guessing: into the Projection wrapper the bare keys don't exist
    guess = transplant_state_encoder_proj(source, wrapper.state_proj)
    assert guess.mapped_params == {}
    assert guess.mode is ConversionMode.REPRESENTATION_REUSE
    assert any("NOT a transplant" in w for w in guess.warnings)

    # the backbone is never overwritten, even on exact key matches
    holder = nn.Module()
    holder.model = copy.deepcopy(tiny_bert)
    before = {k: v.detach().clone() for k, v in wrapper.model.state_dict().items()}
    protected = transplant_state_encoder_proj(holder, wrapper)
    assert protected.mapped_params == {}
    assert any("backbone" in w for w in protected.warnings)
    assert set(protected.unmapped_source) == set(holder.state_dict())
    for key, value in before.items():
        assert torch.equal(wrapper.model.state_dict()[key], value), key
    _assert_no_unmeasured_passes(protected)


# ---------------------------------------------------------------------------
# convert() with a source head
# ---------------------------------------------------------------------------


def test_convert_with_source_head_transplants(tiny_bert, tiny_tokenizer) -> None:
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=2)
    model, report = convert(
        copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer, source_head=clf
    )
    # ladder: choice (no linear) and score (no level_proj) cannot map; the
    # noul head's Linear(D -> 1) takes the binary-gauge graft
    assert report.mode is ConversionMode.HEAD_ATTACHMENT
    assert set(report.mapped_params) == {"classifier.weight", "classifier.bias"}
    assert all(key.startswith("heads.noul.") for key in report.mapped_params.values())
    head = model.heads["noul"]
    expected = (clf.classifier.weight[1] - clf.classifier.weight[0]).detach()
    assert torch.equal(head.proj.weight.data[0], expected)
    # the ladder taken is recorded, provenance is recorded
    assert any("heads[choice]" in w for w in report.warnings)
    assert any("head graft" in w for w in report.warnings)
    assert report.equivalence is EquivalenceKind.FUNCTIONAL  # measured, not assumed
    _assert_no_unmeasured_passes(report)
    # choice path still runs after the graft
    out = model.choice(STATE, ["q1?"], options=[["a", "b"]])
    assert len(out.decisions) == 1


def test_convert_unmappable_source_head_stays_reuse(tiny_bert, tiny_tokenizer) -> None:
    clf = _tiny_classifier(hidden=HIDDEN, num_labels=3)  # K=3 matches nothing
    model, report = convert(
        copy.deepcopy(tiny_bert), tokenizer=tiny_tokenizer, source_head=clf
    )
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.mapped_params == {}
    assert any("could not be transplanted" in w for w in report.warnings)
    assert any("NOT a transplant" in w for w in report.warnings)
    assert "no transplant" in report.summary()
    assert report.unmapped_source  # the head inventory is reported
    _assert_no_unmeasured_passes(report)
    out = model.choice(STATE, ["q1?"], options=[["a", "b"]])
    assert len(out.decisions) == 1


# ---------------------------------------------------------------------------
# convert(): equivalence bookkeeping on an id/path input
# ---------------------------------------------------------------------------


def test_convert_local_dir_equivalence_measured(tmp_path, tiny_bert) -> None:
    source_dir = tmp_path / "base"
    tiny_bert.save_pretrained(source_dir)

    model, report = convert(str(source_dir), equivalence_check=True)
    assert isinstance(model, JevXForDecision)
    assert report.mode is ConversionMode.REPRESENTATION_REUSE
    assert report.source == str(source_dir)
    functional = [c for c in report.checks if c["name"] == "functional_equivalence"]
    assert functional, report.checks
    # tokenizer-less local dir -> seeded id probe; independent base reload
    # measured a bitwise-identical wrap
    assert functional[0]["passed"] is True
    assert functional[0]["value"] == 0.0
    assert functional[0]["tolerance"] == 0.0
    assert report.equivalence is EquivalenceKind.FUNCTIONAL
    _assert_no_unmeasured_passes(report)
    # equivalence_check=False records nothing (no measurement -> no claim)
    _, off = convert(str(source_dir), equivalence_check=False)
    assert off.checks == []
    assert off.equivalence is EquivalenceKind.NONE


def test_convert_supplied_probe_inputs(tiny_bert) -> None:
    probe = _inputs()
    model, report = convert(copy.deepcopy(tiny_bert), probe_inputs=probe)
    functional = [c for c in report.checks if c["name"] == "functional_equivalence"]
    assert functional and functional[0]["passed"] is True
    assert report.equivalence is EquivalenceKind.FUNCTIONAL
    # the pre-wrap snapshot used exactly the supplied probe
    result = check_base_equivalence(model, copy.deepcopy(tiny_bert), probe)
    assert result["max_abs"] == 0.0
    with pytest.raises(ValueError, match="non-empty"):
        convert(copy.deepcopy(tiny_bert), probe_inputs={})


# ---------------------------------------------------------------------------
# convert(): argument partitioning (mirrors the factory)
# ---------------------------------------------------------------------------


def test_convert_argument_partition_errors(tiny_bert) -> None:
    # backbone-loader kwargs are meaningless for an in-memory module
    with pytest.raises(TypeError, match="loader kwargs"):
        convert(copy.deepcopy(tiny_bert), dtype=torch.float32)
    # unknown kwargs name all three groups
    with pytest.raises(TypeError, match="unexpected keyword"):
        convert(copy.deepcopy(tiny_bert), not_a_kwarg=1)
    # not an id/module at all
    with pytest.raises(TypeError, match="model id"):
        convert(42)
    # an already-wrapped JevX model is not a conversion source
    wrapper, _ = convert(copy.deepcopy(tiny_bert))
    with pytest.raises(TypeError, match="JevXForDecision"):
        convert(wrapper)


# ---------------------------------------------------------------------------
# report honesty invariants
# ---------------------------------------------------------------------------


def test_report_honesty_invariants() -> None:
    # a transplant mode with no mapping is rejected
    with pytest.raises(ValueError, match="mapped_params"):
        ConversionReport(mode=ConversionMode.WEIGHT_TRANSPLANT)
    with pytest.raises(ValueError, match="mapped_params"):
        ConversionReport(mode="head_attachment")
    # claiming equivalence without a measured pass is rejected
    with pytest.raises(ValueError, match="did not measure"):
        ConversionReport(mode=ConversionMode.REPRESENTATION_REUSE, equivalence="functional")
    # a pass without a measured value is rejected at record time...
    report = ConversionReport(mode=ConversionMode.REPRESENTATION_REUSE)
    with pytest.raises(ValueError, match="measured value"):
        report.add_check("unmeasured", None, 0.0, True)
    # ...and at construction time (direct dict entry)
    with pytest.raises(ValueError, match="measured value"):
        ConversionReport(
            mode=ConversionMode.REPRESENTATION_REUSE,
            checks=[{"name": "bad", "value": None, "tolerance": None, "passed": True}],
        )
    # a skip without a reason is rejected
    with pytest.raises(ValueError, match="reason"):
        report.add_check("skipped", None, None, None)
    # claiming equivalence with only skips is rejected
    report.add_check("skipped", None, None, None, reason="no featurizer available")
    with pytest.raises(ValueError, match="measure"):
        report.claim_equivalence(EquivalenceKind.FUNCTIONAL)
    assert report.equivalence is EquivalenceKind.NONE
    # a measured pass unlocks the claim
    report.add_check("measured", 0.0, 0.0, True)
    report.claim_equivalence(EquivalenceKind.FUNCTIONAL)
    assert report.equivalence is EquivalenceKind.FUNCTIONAL
    # unknown enum values fail loudly
    with pytest.raises(ValueError):
        ConversionReport(mode="teleportation")


# ---------------------------------------------------------------------------
# package exports
# ---------------------------------------------------------------------------


def test_package_exports() -> None:
    assert jevx.ConversionMode is ConversionMode
    assert jevx.ConversionReport is ConversionReport
    assert jevx.EquivalenceKind is EquivalenceKind
    assert jevx.convert is convert
    for name in ("ConversionMode", "ConversionReport", "EquivalenceKind", "convert"):
        assert name in jevx.__all__
    for name in conversion_pkg.__all__:
        assert hasattr(conversion_pkg, name), name
    # the reserved mode exists for taxonomy completeness but is never assigned
    assert ConversionMode.ARCHITECTURAL_CONVERSION.value == "architectural_conversion"
