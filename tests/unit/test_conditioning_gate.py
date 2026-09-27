# Copyright (c) 2026 ysnrfd
"""Unit tests for the task-level conditioning gate.

The gate answers one question per TASK: should decisions be scored with the
option pathway on (conditioned) or off (state-only)? The measurement is
delta-NLL (NLL_conditioned - NLL_state_only) on a calibration split, and the
threshold comes from the one-standard-error protocol on that same split -
never from a reported evaluation split (the no-test-leakage guard is tested
here too).

Everything below is in-memory and tiny: the probe model is a real, TRAINED
two-pathway decision head (an additive state probe plus an option-interaction
residual, the same algebra as ``jevx.decisions.ChoiceHead``, driven through the
same ``set_interaction_scale`` hook) wrapped in a forward that honours the same
contract the gate uses on ``JevXForDecision``
(``state/questions/options/decision_type/condition_options``). No network, no
downloads, sub-second.
"""

from __future__ import annotations

import inspect
import math
import zlib
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jevx.modeling_outputs import DecisionOutput
from jevx.training import decision_collate_fn
from jevx.training.conditioning_gate import (
    ConditioningGate,
    ConditioningGateConfig,
    apply_gate,
    assert_split_not_leakage,
    fit_tau,
    measure_delta_nll,
    measure_modes,
)
from jevx.training.losses import choice_loss

# ---------------------------------------------------------------------------
# A real (tiny, trained) two-pathway decision model
# ---------------------------------------------------------------------------

DIM = 8
OPTIONS = 3
#: Size of the stand-in schema encoder's direction table (index by crc32).
VOCAB = 64


def _option_slot(token: str) -> int:
    """Deterministic (process-independent) index of an option token."""
    return zlib.crc32(token.encode("utf-8")) % VOCAB


class ProbeDecisionModel(nn.Module):
    """``z_k = state_probe(fused)_k + scale * <w_k, [fused ; e_k]>``.

    A miniature of the shipped ``ChoiceHead`` algebra: an ADDITIVE state
    pathway (``Linear(D, K)`` on the fused state) plus an option-interaction
    RESIDUAL scaled by ``set_interaction_scale`` (the same lambda the landed
    head exposes). ``condition_options=False`` drops the residual exactly, which
    is the state-only scoring mode the gate measures; ``condition_options=None``
    falls back to ``config.conditioning_enabled``, mirroring the wrapper's
    contract.

    Every parameter is initialized from a LOCAL generator, so the model is
    reproducible without touching the global RNG.
    """

    def __init__(
        self, seed: int = 0, num_options: int = OPTIONS, dim: int = DIM
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_options = num_options
        generator = torch.Generator().manual_seed(seed)
        self.state_map_weight = nn.Parameter(
            torch.eye(dim) + 0.15 * torch.randn(dim, dim, generator=generator)
        )
        self.state_probe_weight = nn.Parameter(
            0.3 * torch.randn(num_options, dim, generator=generator)
        )
        self.state_probe_bias = nn.Parameter(torch.zeros(num_options))
        self.interaction_weight = nn.Parameter(
            torch.randn(num_options, 2 * dim, generator=generator) / math.sqrt(2 * dim)
        )
        self.interaction_bias = nn.Parameter(torch.zeros(num_options))
        self.option_table = nn.Parameter(
            torch.randn(VOCAB, dim, generator=generator) / math.sqrt(dim)
        )
        self._interaction_scale = 1.0
        self.config = SimpleNamespace(conditioning_enabled=True)
        self.forward_calls = 0

    # -- interaction scale (the landed head exposes the same API) ------------

    def set_interaction_scale(self, value: float) -> None:
        """Force the option-interaction residual on/off (the harm harness)."""
        self._interaction_scale = float(value)

    @property
    def get_interaction_scale(self) -> float:
        """Current interaction-residual scale."""
        return self._interaction_scale

    # -- forward -------------------------------------------------------------

    def option_features(self, options: Sequence[Any]) -> torch.Tensor:
        """Schema encoder: option tokens -> ``[K, D]`` directions."""
        return self.option_table[
            torch.as_tensor(
                [_option_slot(str(token)) for token in options], dtype=torch.long
            )
        ]

    def logits_for(
        self, states: torch.Tensor, option_features: torch.Tensor, conditioned: bool
    ) -> torch.Tensor:
        """``[B, K]`` raw scores; ``conditioned=False`` is the state pathway."""
        fused = states @ self.state_map_weight.t()  # [B, D]
        logits = fused @ self.state_probe_weight.t() + self.state_probe_bias
        if not conditioned:
            return logits
        pairs = torch.cat(
            [
                fused.unsqueeze(1).expand(-1, option_features.shape[0], -1),
                option_features.unsqueeze(0).expand(fused.shape[0], -1, -1),
            ],
            dim=-1,
        )  # [B, K, 2D]
        residual = torch.einsum("bkd,kd->bk", pairs, self.interaction_weight)
        return logits + self._interaction_scale * (
            residual + self.interaction_bias
        )

    def forward(
        self,
        state: Any = None,
        questions: Any = None,
        decision_type: str = "choice",
        options: Any = None,
        *,
        condition_options: bool | None = None,
        **kwargs: Any,
    ) -> SimpleNamespace:
        if kwargs:
            raise TypeError(
                f"ProbeDecisionModel.forward: unexpected keyword(s) {sorted(kwargs)}."
            )
        if decision_type != "choice":
            raise ValueError(
                f"ProbeDecisionModel: only 'choice' is modelled, got {decision_type!r}."
            )
        if state is None or questions is None or options is None:
            raise ValueError(
                "ProbeDecisionModel.forward: state, questions and options are required."
            )
        self.forward_calls += 1
        flag = (
            bool(self.config.conditioning_enabled)
            if condition_options is None
            else bool(condition_options)
        )
        states = torch.as_tensor(state, dtype=torch.float32)
        if states.dim() == 1:
            states = states.unsqueeze(0)
        schema = list(options)
        question_texts = (
            [questions] if isinstance(questions, str) else list(questions)
        )
        if len(question_texts) != len(schema):
            raise ValueError(
                f"ProbeDecisionModel: {len(question_texts)} question(s) but "
                f"{len(schema)} schema(s)."
            )
        decisions: list[DecisionOutput] = []
        for option_row in schema:
            keys = [str(token) for token in option_row]
            features = self.option_features(option_row)
            logits = self.logits_for(states, features, flag)
            probabilities = torch.softmax(logits, dim=-1)
            for b in range(int(states.shape[0])):
                decisions.append(
                    DecisionOutput(
                        logits=logits[b : b + 1],
                        probabilities=probabilities[b : b + 1],
                        predicted_index=probabilities[b : b + 1].argmax(dim=-1),
                        schema_metadata={
                            "decision_type": "choice",
                            "option_keys": keys,
                        },
                    )
                )
        return SimpleNamespace(
            decisions=decisions,
            capability_metadata={"decision_types": ["choice"] * len(decisions)},
        )


# ---------------------------------------------------------------------------
# Training the probes (plain Adam, fixed group order -> deterministic)
# ---------------------------------------------------------------------------


def _features_of(model: ProbeDecisionModel, options: Sequence[Any]) -> torch.Tensor:
    with torch.no_grad():
        return model.option_features(options).detach()


def _group_records(
    model: ProbeDecisionModel, records: list[dict]
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``[(states [B, D], option_features [K, D], targets [B])]`` per group."""
    groups: dict[tuple[Any, ...], list[dict]] = {}
    for record in records:
        key = (tuple(record["questions"]), repr(record["options"]))
        groups.setdefault(key, []).append(record)
    out = []
    for members in groups.values():
        states = torch.as_tensor(
            [member["state"] for member in members], dtype=torch.float32
        )
        targets = torch.as_tensor(
            [int(member["labels"][0]) for member in members], dtype=torch.long
        )
        out.append((states, _features_of(model, members[0]["options"][0]), targets))
    return out


def _train(
    model: ProbeDecisionModel,
    records: list[dict],
    *,
    steps: int = 200,
    lr: float = 0.05,
) -> ProbeDecisionModel:
    """Fit the two pathways on a fixed group order (full-batch Adam)."""
    groups = _group_records(model, records)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = sum(
            choice_loss(model.logits_for(states, features, True), targets)
            for states, features, targets in groups
        )
        loss.backward()
        optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _lure_schema_records(
    n: int, *, informative_state: bool, separation: float = 4.0
) -> list[dict]:
    """Two schemas whose gold option is ALWAYS the one whose text is "yes".

    Schema A is ``["yes", "no", "maybe"]`` (gold 0) and schema B is
    ``["no", "maybe", "yes"]`` (gold 2), so the option TEXT alone identifies the
    answer: the interaction has a single consistent rule to learn ("the yes
    option wins") and, when the state marks the schema, the state pathway has
    to learn the same rule from the state.

    Args:
        n: Number of records; they alternate between the two schemas.
        informative_state: ``True`` -> the state is a one-hot marking which
            schema the record uses, so a state probe CAN learn the task (the
            "real decision problem" arm used by the harmful-head test).
            ``False`` -> every state is the zero vector, so the state pathway
            carries nothing and only the option pathway can answer.
        separation: Magnitude of the informative state's one-hot.
    """
    schemas = [
        (["yes", "no", "maybe"], 0),
        (["no", "maybe", "yes"], 2),
    ]
    records = []
    for index in range(n):
        options, gold = schemas[index % len(schemas)]
        state = [0.0] * DIM
        if informative_state:
            state[index % len(schemas)] = separation
        records.append(
            {
                "state": state,
                "questions": ["choose"],
                "decision_type": "choice",
                "options": [options],
                "labels": [gold],
            }
        )
    return records


def _loader(records: list[dict], batch_size: int) -> DataLoader:
    return DataLoader(
        records,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=decision_collate_fn,
    )


def _gate(model: ProbeDecisionModel, **overrides: Any) -> ConditioningGate:
    fields: dict[str, Any] = {"min_samples": 0, "split_name": "calibration"}
    fields.update(overrides)
    return ConditioningGate(model, ConditioningGateConfig(**fields))


#: The lambda forced onto a trained option pathway to turn it into a LURE (it
#: then votes confidently for the wrong option). This is the harm harness: a
#: conditioning scale, not a different model.
HARMFUL_SCALE = -8.0


def _trained_pipeline(
    seed: int, *, informative_state: bool = True, scale: float = 1.0, n: int = 40
) -> tuple[ProbeDecisionModel, list[dict], DataLoader]:
    """A trained two-pathway head on a seeded n-record calibration split."""
    model = ProbeDecisionModel(seed=seed)
    records = _lure_schema_records(n, informative_state=informative_state)
    _train(model, records)
    model.set_interaction_scale(scale)
    return model, records, _loader(records, batch_size=8)


def _built_pipeline(seed: int) -> tuple[ProbeDecisionModel, list[dict], DataLoader]:
    """The harmful pipeline shared by the determinism / flag / io tests."""
    return _trained_pipeline(seed, informative_state=True, scale=HARMFUL_SCALE)


# ---------------------------------------------------------------------------
# 1. The measurement: conditioning helps when the options carry information
# ---------------------------------------------------------------------------


def test_delta_nll_is_negative_when_options_carry_information() -> None:
    """One constant state, two schemas: only the option pathway can answer.

    The state is the zero vector, so the state pathway gives every option the
    same score and cannot beat chance, while the trained interaction learns
    "the yes option wins". Delta-NLL must be clearly negative, the conditioned
    mode must be right and the state-only mode worse on BOTH NLL and accuracy.
    """
    model, records, loader = _trained_pipeline(
        3, informative_state=False, scale=1.0, n=24
    )
    assert len(records) == 24

    delta = measure_delta_nll(model, loader, decision_type="choice")
    measurement = measure_modes(model, loader, decision_type="choice")

    assert measurement.num_samples == 24
    assert measurement.num_batches == 3
    assert delta == pytest.approx(measurement.delta_nll, abs=0.0)
    assert delta < -0.2, (
        "the option pathway carries the answer here, so Delta-NLL must be "
        f"clearly negative, got {delta}"
    )
    assert measurement.conditioned_nll < 0.2
    assert measurement.state_only_nll > measurement.conditioned_nll + 0.2
    assert measurement.conditioned_accuracy == 1.0
    # The gold is never the same option index in both schemas, so a state-only
    # (constant-state) probe cannot be right on both.
    assert measurement.state_only_accuracy <= 0.5 + 1e-9
    assert measurement.state_only_accuracy < measurement.conditioned_accuracy


# ---------------------------------------------------------------------------
# 2. A harmful option pathway -> state_only wins, and accuracy follows
# ---------------------------------------------------------------------------


def test_harmful_option_pathway_picks_state_only_with_better_accuracy() -> None:
    """Trained probes + a forced bad interaction scale (a lure).

    Both pathways are trained on a real decision problem (the state marks the
    schema, so the state probe learns it), then the interaction residual is
    forced to vote for the WRONG option. The gate must pick ``state_only`` and
    the mode it picked must measure at least as accurate as the one it
    rejected - the EXP-H failure mode, asserted on accuracy and not only NLL.
    """
    model, _, loader = _built_pipeline(11)

    delta = measure_delta_nll(model, loader, decision_type="choice")
    measurement = measure_modes(model, loader, decision_type="choice")

    assert delta > 0.0, f"a forced-bad interaction must raise the NLL, got {delta}"
    assert measurement.state_only_accuracy >= measurement.conditioned_accuracy
    assert measurement.state_only_nll < measurement.conditioned_nll
    assert measurement.state_only_accuracy > 0.5, (
        "fixture must keep a usable state probe, otherwise the comparison is vacuous"
    )

    gate = _gate(model)
    assert gate.decide(loader) == "state_only"
    assert gate.measurement is not None
    assert gate.measurement.delta_nll == pytest.approx(delta, abs=0.0)
    assert gate.decision == "state_only"
    # The chosen mode is the more accurate one on the same split.
    assert measurement.state_only_accuracy >= measurement.conditioned_accuracy


def test_zero_interaction_is_exactly_neutral_so_the_gate_keeps_conditioning() -> None:
    """At interaction scale 0 the two modes are the SAME function.

    Delta-NLL is then exactly 0 (bit-identical logits), which is inside the noise
    band for every non-negative tau - the gate must not switch on a difference
    it cannot measure, so the shipped behaviour is kept.
    """
    model, _, loader = _trained_pipeline(17, informative_state=True, scale=0.0, n=24)

    measurement = measure_modes(model, loader, decision_type="choice")
    assert measurement.delta_nll == 0.0
    assert measurement.conditioned_nll == measurement.state_only_nll
    assert measurement.conditioned_accuracy == measurement.state_only_accuracy

    gate = _gate(model)
    assert gate.decide(loader) == "conditioned"
    assert gate.tau == 0.0
    assert gate.config.tau_source is not None
    assert gate.config.tau_source.startswith("one_se(")


# ---------------------------------------------------------------------------
# 3. The tau protocol
# ---------------------------------------------------------------------------


def test_fit_tau_is_z_times_the_standard_error() -> None:
    """Seeded synthetic case: tau = z * std(d, ddof=1) / sqrt(B), exactly."""
    deltas = [0.4, 0.5, 0.6, 0.7, 0.8]
    mean = 0.6
    variance = sum((value - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    expected_se = math.sqrt(variance) / math.sqrt(len(deltas))

    fit = fit_tau(deltas, z=1.0)
    assert fit.standard_error == pytest.approx(expected_se, rel=1e-12)
    assert fit.tau == pytest.approx(1.0 * expected_se, rel=1e-12)
    assert fit.num_batches == 5
    assert fit.z == 1.0
    assert fit.source == f"one_se(z=1, se={expected_se:.6g}, batches=5)"

    doubled = fit_tau(deltas, z=2.0)
    assert doubled.tau == pytest.approx(2.0 * expected_se, rel=1e-12)
    assert doubled.tau == pytest.approx(2.0 * fit.tau, rel=1e-12)


def test_fit_tau_one_se_rule_keeps_a_within_noise_effect() -> None:
    """A useless-but-noisy interaction does NOT clear the one-SE bar.

    Seeded synthetic per-batch Delta-NLL: mean 0.01, spread 0.2, so the
    standard error is an order of magnitude above the mean effect. The rule
    reads ``delta > tau``; here it must not fire.
    """
    generator = torch.Generator().manual_seed(2024)
    noise = torch.randn(12, generator=generator) * 0.2
    # Centred noise + a small positive mean: the mean is pinned to 0.01 while
    # the batch-to-batch spread stays 0.2, so the effect stays inside the noise.
    deltas = [float(value) for value in (noise - noise.mean() + 0.01)]
    fit = fit_tau(deltas, z=1.0)
    delta_nll = math.fsum(deltas) / len(deltas)

    assert fit.standard_error > abs(delta_nll), (
        f"fixture must be within noise: |mean|={abs(delta_nll):.4f} vs "
        f"SE={fit.standard_error:.4f}"
    )
    assert delta_nll <= fit.tau
    assert not (delta_nll > fit.tau)  # the rule reads exactly this

    # A clearly useless effect measured over enough batches: a mean of 0.5
    # against the same noise floor DOES clear it.
    assert 0.5 > fit.standard_error


def test_fit_tau_large_effect_switches_and_se_is_unestimable_flagged() -> None:
    """A large, consistent effect clears tau; a single batch says so loudly."""
    noisy = [0.30, 0.31, 0.32, 0.33]
    mean = math.fsum(noisy) / len(noisy)
    variance = sum((value - mean) ** 2 for value in noisy) / (len(noisy) - 1)
    expected_se = math.sqrt(variance) / math.sqrt(len(noisy))

    fit = fit_tau(noisy, z=1.0)
    assert fit.standard_error == pytest.approx(expected_se, rel=1e-12)
    assert fit.tau == pytest.approx(expected_se, rel=1e-12)  # consistent -> tiny
    assert mean > fit.tau  # a large effect switches

    single = fit_tau([0.5])
    assert single.standard_error == 0.0
    assert single.tau == 0.0
    assert "se_unestimable" in single.source


def test_fit_tau_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_tau([])
    with pytest.raises(ValueError, match="finite"):
        fit_tau([0.1, float("nan")])
    with pytest.raises(ValueError, match="z"):
        fit_tau([0.1, 0.2], z=-1.0)
    with pytest.raises(TypeError, match="sequence"):
        fit_tau(0.1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. The no-test-leakage split guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "split_name",
    ["test", "TEST", "ravdess_test", "test_split", "holdout", "eval", "final"],
)
def test_guard_rejects_test_split_names(split_name: str) -> None:
    with pytest.raises(ValueError, match="no-test-leakage"):
        assert_split_not_leakage(split_name)


@pytest.mark.parametrize("split_name", ["calibration", "val", "dev", "contest"])
def test_guard_allows_non_test_split_names(split_name: str) -> None:
    assert_split_not_leakage(split_name)


def test_measure_and_decide_refuse_a_test_split() -> None:
    model, _, loader = _built_pipeline(2)
    gate = _gate(model)
    # The caller re-tags the split after construction (config validation runs
    # at construction time); measure/decide must re-check every time.
    gate.config.split_name = "ravdess_test"
    with pytest.raises(ValueError, match="no-test-leakage"):
        gate.measure(loader)
    with pytest.raises(ValueError, match="no-test-leakage"):
        gate.decide(loader)
    with pytest.raises(ValueError, match="no-test-leakage"):
        measure_delta_nll(model, loader, split_name="test")
    with pytest.raises(ValueError, match="no-test-leakage"):
        ConditioningGateConfig(split_name="test")


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


def test_same_seed_gives_the_same_delta_and_decision() -> None:
    """Two identically-seeded pipelines agree bit-for-bit on the verdict."""
    deltas, decisions = [], []
    for _ in range(2):
        model, _, loader = _built_pipeline(21)
        gate = _gate(model)
        decisions.append(gate.decide(loader))
        deltas.append(gate.measurement.delta_nll if gate.measurement else None)
    assert decisions[0] == decisions[1] == "state_only"
    assert deltas[0] == deltas[1]


def test_repeated_measure_on_one_gate_is_bit_identical() -> None:
    model, _, loader = _built_pipeline(22)
    gate = _gate(model)
    first = gate.measure(loader)
    second = gate.measure(loader)
    assert first == second
    assert first.delta_nll == second.delta_nll
    assert first.per_batch_delta == second.per_batch_delta
    assert gate.decide(loader) == "state_only"


def test_large_effect_decision_is_stable_across_seeds() -> None:
    """A large measured effect gives the same verdict on every seed."""
    decisions = []
    for seed in (31, 32, 33):
        model, _, loader = _built_pipeline(seed)
        gate = _gate(model)
        decisions.append(gate.decide(loader))
        assert gate.measurement is not None
        assert gate.measurement.delta_nll > gate.tau, (
            "fixture must keep a large effect so stability is meaningful"
        )
    assert decisions == ["state_only", "state_only", "state_only"]


# ---------------------------------------------------------------------------
# 6. The flag the gate returns is the flag the forward honours
# ---------------------------------------------------------------------------


def _predictions(
    model: ProbeDecisionModel, loader: DataLoader, flag: bool
) -> tuple[list[int], list[int]]:
    """Run the forward at a fixed flag; return (predictions, gold).

    Samples are grouped by schema exactly the way the gate groups them (one
    forward call per distinct schema, state-major inside the group), so the
    predictions line up with the gold one-for-one.
    """
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []
    with torch.no_grad():
        for batch in loader:
            groups: dict[str, tuple[list[Any], list[Any], list[Any]]] = {}
            for state, options, labels in zip(
                batch["state"], batch["options"], batch["labels"]
            ):
                key = repr(options)
                slot = groups.setdefault(key, ([], [], []))
                slot[0].append(state)
                slot[1].append(options)
                slot[2].append(labels)
            for states, options_list, labels_list in groups.values():
                out = model(
                    state=states,
                    questions=batch["questions"][0],
                    options=options_list[0],
                    decision_type="choice",
                    condition_options=flag,
                )
                assert len(out.decisions) == len(states)
                for decision, labels in zip(out.decisions, labels_list):
                    predictions.append(int(decision.predicted_index[0]))
                    gold.append(int(labels[0]))
    return predictions, gold

@pytest.mark.parametrize("seed", [41, 42, 43])
def test_condition_options_for_reproduces_the_chosen_mode(seed: int) -> None:
    """The returned flag selects EXACTLY the mode the gate measured.

    The gate's recorded accuracy for the chosen mode must equal the accuracy
    obtained by running the forward with ``condition_options_for(model)``, and
    must differ from the rejected mode's - the gate and the forward cannot
    disagree about what was measured.
    """
    model, _, loader = _built_pipeline(seed)
    gate = _gate(model)
    decision = gate.decide(loader)
    flag = gate.condition_options_for(model)
    assert flag == (decision == "conditioned")

    measurement = gate.measurement
    assert measurement is not None
    conditioned_pred, conditioned_gold = _predictions(model, loader, True)
    state_only_pred, state_only_gold = _predictions(model, loader, False)
    assert conditioned_gold == state_only_gold
    assert len(conditioned_pred) == len(conditioned_gold)

    conditioned_accuracy = sum(
        int(p == g) for p, g in zip(conditioned_pred, conditioned_gold)
    ) / len(conditioned_gold)
    state_only_accuracy = sum(
        int(p == g) for p, g in zip(state_only_pred, state_only_gold)
    ) / len(state_only_gold)

    if flag:
        assert conditioned_accuracy == pytest.approx(
            measurement.conditioned_accuracy, abs=1e-12
        )
        assert conditioned_accuracy != pytest.approx(
            measurement.state_only_accuracy, abs=1e-12
        )
    else:
        assert state_only_accuracy == pytest.approx(
            measurement.state_only_accuracy, abs=1e-12
        )
        assert state_only_accuracy >= conditioned_accuracy


def test_condition_options_for_requires_a_decision() -> None:
    model = ProbeDecisionModel(seed=1)
    gate = _gate(model)
    with pytest.raises(RuntimeError, match="no decision yet"):
        gate.condition_options_for(model)


def test_apply_gate_sets_config_and_round_trips_the_flag() -> None:
    model, _, loader = _built_pipeline(51)
    gate = _gate(model)

    with pytest.raises(RuntimeError, match="pending"):
        apply_gate(model, gate)  # an undecided gate must not set a flag

    gate.decide(loader)
    flag = apply_gate(model, gate)
    assert flag is False
    assert model.config.conditioning_enabled is False
    # With the config flag flipped and no per-call override, the forward
    # reproduces the chosen mode.
    with torch.no_grad():
        default_pred, gold = _predictions(
            model, loader, model.config.conditioning_enabled
        )
        state_only_pred, _ = _predictions(model, loader, False)
    assert default_pred == state_only_pred
    assert gold


# ---------------------------------------------------------------------------
# 7. Serialization
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip() -> None:
    model, _, loader = _built_pipeline(61)
    gate = _gate(model, tie_tolerance=0.01, min_samples=4, z=0.5)
    gate.decide(loader)
    payload = gate.to_dict()

    assert payload["decision"] == "state_only"
    assert payload["split_name"] == "calibration"
    assert payload["rule"] == "state_only_if_delta_nll_exceeds"
    assert payload["min_samples"] == 4
    assert payload["tie_tolerance"] == 0.01
    assert payload["z"] == 0.5
    assert payload["forbidden_split_tokens"] and isinstance(
        payload["forbidden_split_tokens"], list
    )
    assert payload["measurement"]["num_samples"] == 40

    restored = ConditioningGate.from_dict(payload, model)
    assert restored.to_dict() == payload
    assert restored.config == gate.config
    assert restored.measurement == gate.measurement
    assert restored.decision == gate.decision
    assert restored.condition_options_for(model) == gate.condition_options_for(model)
    assert restored.summary() == gate.summary()


def test_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        ConditioningGate.from_dict({"tau": 0.1, "nope": 1})


def test_summary_reports_the_measurement_and_the_protocol() -> None:
    model, _, loader = _built_pipeline(71)
    gate = _gate(model)
    pending_summary = gate.summary()
    assert "pending" in pending_summary
    assert "(none yet" in pending_summary

    gate.decide(loader)
    text = gate.summary()
    assert "state_only" in text
    assert "calibration" in text
    assert "one standard error" in text
    assert "tau = z * SE" in text
    assert "conditioned=" in text and "state_only=" in text
    assert "samples/batches: 40/5" in text


# ---------------------------------------------------------------------------
# 8. Configuration contract
# ---------------------------------------------------------------------------


def test_config_defaults_match_the_documented_gate() -> None:
    config = ConditioningGateConfig()
    assert config.tau is None
    assert config.rule == "state_only_if_delta_nll_exceeds"
    assert config.split_name == "calibration"
    assert config.min_samples == 32
    assert config.tie_tolerance == 0.0
    assert config.decision == "pending"
    assert config.delta_nll is None
    assert config.tau_source is None


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"rule": "whatever"}, ValueError, "rule"),
        ({"decision": "maybe"}, ValueError, "decision"),
        ({"tau": -1.0}, ValueError, "tau"),
        ({"z": -0.5}, ValueError, "z"),
        ({"tie_tolerance": -1.0}, ValueError, "tie_tolerance"),
        ({"min_samples": -1}, ValueError, "min_samples"),
        ({"decision_type": "nope"}, ValueError, "decision_type"),
        ({"z": "1"}, TypeError, "z"),
    ],
)
def test_config_validation(kwargs: dict, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        ConditioningGateConfig(**kwargs)


def test_min_samples_blocks_the_switch_and_says_why() -> None:
    """Too little calibration data -> keep the shipped behaviour, say why."""
    model, _, loader = _built_pipeline(81)
    gate = _gate(model, min_samples=1000)
    assert gate.decide(loader) == "conditioned"
    assert gate.measurement is not None
    assert gate.measurement.delta_nll > 0.0  # the effect is there ...
    assert "min_samples_unmet" in (gate.config.tau_source or "")
    assert "kept=conditioned" in (gate.config.tau_source or "")


def test_tie_tolerance_keeps_conditioning_on_a_marginal_effect() -> None:
    """A delta within tie_tolerance of tau is a tie -> keep conditioning."""
    model, _, loader = _built_pipeline(91)
    strict = _gate(model)
    assert strict.decide(loader) == "state_only"
    assert strict.measurement is not None
    assert strict.tau is not None
    delta = strict.measurement.delta_nll
    assert delta > strict.tau

    # A tolerance that covers the whole margin: delta <= tau + tie_tolerance.
    padded = _gate(model, tie_tolerance=delta - strict.tau + 1.0)
    assert padded.decide(loader) == "conditioned"
    assert padded.measurement is not None
    assert padded.measurement.delta_nll == pytest.approx(delta, abs=0.0)


def test_pinned_tau_is_used_verbatim_and_refitted_after_unpinning() -> None:
    model, _, loader = _built_pipeline(101)
    gate = _gate(model)
    gate.set_tau(1e9)  # unreachable bar: never switch
    assert gate.decide(loader) == "conditioned"
    assert gate.config.tau_source == "explicit"
    assert gate.tau == 1e9

    gate.set_tau(0.0)  # any positive effect now switches
    assert gate.decide(loader) == "state_only"
    assert gate.config.tau == 0.0

    gate.set_tau(None)  # back to the protocol, and RE-FIT (not stale)
    assert gate.decide(loader) == "state_only"
    assert gate.config.tau_source is not None
    assert gate.config.tau_source.startswith("one_se(")


def test_measure_rejects_a_model_without_the_flag() -> None:
    class NoFlag(torch.nn.Module):
        def forward(
            self, state=None, questions=None, decision_type=None, options=None
        ):
            raise AssertionError("must not be called")

    model, _, loader = _built_pipeline(2)
    with pytest.raises(TypeError, match="condition_options"):
        measure_delta_nll(NoFlag(), loader)
    # The real model is fine.
    assert isinstance(measure_delta_nll(model, loader), float)


def test_measure_rejects_unknown_decision_type_and_empty_loader() -> None:
    model, _, loader = _built_pipeline(2)
    with pytest.raises(ValueError, match="decision_type"):
        measure_delta_nll(model, loader, decision_type="vibes")
    with pytest.raises(ValueError, match="no batches"):
        measure_delta_nll(model, [], decision_type="choice")


# ---------------------------------------------------------------------------
# 9. Integration with the real wrapper (JevXForDecision)
# ---------------------------------------------------------------------------


def _has_condition_options() -> bool:
    from jevx import JevXForDecision

    return "condition_options" in inspect.signature(
        JevXForDecision.forward
    ).parameters


@pytest.mark.skipif(
    not _has_condition_options(),
    reason=(
        "JevXForDecision.forward(condition_options=...) is not on disk yet; the "
        "gate's forward contract is covered by the ProbeDecisionModel tests until "
        "it lands"
    ),
)
def test_gate_flag_reproduces_real_wrapper_decisions(
    tiny_bert, tiny_tokenizer
) -> None:
    """Tie the gate to the REAL forward: the flag picks the chosen mode."""
    import copy

    from jevx import JevXConfig, JevXForDecision

    hidden = 32
    options = ["alpha", "beta", "gamma"]
    records = []
    for index in range(12):
        records.append(
            {
                "state": "hello world" if index % 2 else "foo bar baz",
                "questions": ["option"],
                "decision_type": "choice",
                "options": [options],
                "labels": [index % 3],
            }
        )
    loader = _loader(records, batch_size=6)

    model = JevXForDecision(
        JevXConfig(
            decision_hidden_size=hidden,
            question_hidden_size=hidden,
            schema_hidden_size=hidden,
        ),
        backbone=copy.deepcopy(tiny_bert),
        tokenizer=tiny_tokenizer,
    )
    model.eval()

    gate = _gate(model)
    decision = gate.decide(loader)
    flag = gate.condition_options_for(model)
    assert flag == (decision == "conditioned")

    def predictions(value: bool) -> list[int]:
        out: list[int] = []
        with torch.no_grad():
            for batch in loader:
                result = model(
                    state=batch["state"],
                    questions=batch["questions"][0],
                    options=batch["options"][0],
                    decision_type="choice",
                    condition_options=value,
                )
                assert len(result.decisions) == len(batch["state"])
                for entry in result.decisions:
                    out.append(int(entry.predicted_index[0]))
        return out

    measurement = gate.measurement
    assert measurement is not None
    gold = [int(record["labels"][0]) for record in records]
    conditioned = predictions(True)
    state_only = predictions(False)
    assert len(conditioned) == len(state_only) == len(gold)

    chosen = conditioned if flag else state_only
    chosen_accuracy = sum(int(p == g) for p, g in zip(chosen, gold)) / len(gold)
    other = state_only if flag else conditioned
    other_accuracy = sum(int(p == g) for p, g in zip(other, gold)) / len(gold)

    if flag:
        assert chosen_accuracy == pytest.approx(
            measurement.conditioned_accuracy, abs=1e-12
        )
    else:
        assert chosen_accuracy == pytest.approx(
            measurement.state_only_accuracy, abs=1e-12
        )
        assert chosen_accuracy >= other_accuracy
