# Copyright (c) 2026 ysnrfd
"""JevX-Next ablation matrix: does the nonlinear state path + the option
residual + the adaptive gate + the curriculum actually pay?

The hypothesis under test (design review) is that the choice head's deficit
against a plain MLP probe on frozen features has TWO causes:

  (a) MISSING NONLINEAR STATE CAPACITY. With one fixed question and one fixed
      schema every option row ``e_k`` is constant across batch rows, so the
      bilinear residual ``s^T A e_k`` collapses to ``s^T A e_k`` - a constant
      vector - and ``z_k = s^T (W_k + A e_k) + b_k`` is still LINEAR in ``s``.
      A linear state pathway therefore supplies no nonlinearity at all;
  (b) JOINT-OPTIMIZATION INEFFICIENCY of the option pathway (it has to be
      useful immediately, from step 0, or it is ignored).

The matrix holds the DATA, the SPLITS, the SEEDS, the optimizer/lr/epochs/batch
and the loss fixed, and varies exactly one thing per column:

  ==============  ==========  ==========  =====  ===========
  Cell            State path  Interaction  Gate   Curriculum
  ==============  ==========  ==========  =====  ===========
  A               linear      on          no     no
  B               mlp         OFF         no     no
  C               mlp         on          no     no
  D               mlp         on          YES    no
  E               mlp         on          no     YES
  F               mlp         on          YES    YES
  ==============  ==========  ==========  =====  ===========

plus the reference cells on the IDENTICAL features / splits / seeds:

  P1  p1_linear            Linear[D -> K] probe on the pooled state
  P2  p2_mlp               Linear[D -> D] -> ReLU -> Linear[D -> K] probe
  P4  p4_state_only_bypass the question/schema pathway bypassed (a fixed,
                            state-independent option basis)

``interaction OFF`` is enforced EXPLICITLY (``conditioning_enabled=False`` in
the config AND ``set_interaction_scale(0.0)`` on the head, both asserted at
run time) - never by relying on a config default: a fresh model maps
``JevXConfig.conditioning_scale_init == 0.0`` to a head lambda of 1.0 by
design (``JevXForDecision._state_pathway_kwargs``), so the default is NOT
"off".

The backbone is frozen and featurized ONCE per split (cache reused from the
EXP-H runs), so no backbone is in the loop: this is head-vs-probe on the same
numbers.

Usage (from the repo root):
    python benchmarks/run_head_ablation.py --datasets synthetic ag_news ravdess
    python benchmarks/run_head_ablation.py --datasets all --seeds 0,1,2,3,4
    python benchmarks/run_head_ablation.py --smoke

Result files : benchmarks/results/head_ablation_<dataset>.json
               benchmarks/results/head_ablation_all.json
Cached features: benchmarks/cache/head_quality/<dataset>_features.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks import bench_common as bc  # noqa: E402
from benchmarks import run_head_quality as hq  # noqa: E402

from jevx.decisions import build_decision_head  # noqa: E402
from jevx.modeling_outputs import DecisionOutput, JevXOutput  # noqa: E402
from jevx.questions import Option  # noqa: E402
from jevx.training import (  # noqa: E402
    ConditioningGate,
    ConditioningGateConfig,
    CurriculumConfig,
    HeadTrainer,
    apply_gate,
    make_supervised_step_fn,
    run_curriculum,
)
from jevx.training.curriculum import _logits_from_output  # noqa: E402

CACHE_DIR = REPO_ROOT / "benchmarks" / "cache" / "head_quality"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

#: Latency protocol: batch 1, warmup then timed iterations.
LATENCY_WARMUP = 20
LATENCY_ITERS = 120

#: The gate is measured on the CALIBRATION split only (never on the reported
#: test split). The name must pass the shipped no-test-leakage guard.
GATE_SPLIT_NAME = "calibration"


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


CELLS: Dict[str, Dict[str, Any]] = {
    # -- references (identical features / splits / seeds) -------------------
    "p1_linear": {
        "kind": "reference",
        "state_pathway": None,
        "interaction": None,
        "gate": False,
        "curriculum": False,
        "label": "P1 Linear probe [D->K] on the pooled state",
    },
    "p2_mlp": {
        "kind": "reference",
        "state_pathway": None,
        "interaction": None,
        "gate": False,
        "curriculum": False,
        "label": "P2 MLP probe [D->D]->ReLU->[D->K] on the pooled state",
    },
    "p4_state_only_bypass": {
        "kind": "reference",
        "state_pathway": "linear",
        "interaction": None,
        "gate": False,
        "curriculum": False,
        "label": "P4 JevX with the question/schema pathway bypassed (fixed basis)",
    },
    # -- the JevX-Next matrix ----------------------------------------------
    "A_linear_interact": {
        "kind": "jevx",
        "state_pathway": "linear",
        "interaction": True,
        "gate": False,
        "curriculum": False,
        "label": "A JevX: linear state path + option interaction ON",
    },
    "B_mlp_no_interact": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": False,
        "gate": False,
        "curriculum": False,
        "label": "B JevX: MLP state path + option interaction OFF",
    },
    "C_mlp_interact": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": True,
        "gate": False,
        "curriculum": False,
        "label": "C JevX: MLP state path + option interaction ON",
    },
    "D_gate": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": True,
        "gate": True,
        "curriculum": False,
        "label": "D JevX: MLP state path + interaction + adaptive gate (cal-only)",
    },
    "E_curriculum": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": True,
        "gate": False,
        "curriculum": True,
        "label": "E JevX: MLP state path + interaction + Stage A/B curriculum",
    },
    "F_gate_curriculum": {
        "kind": "jevx",
        "state_pathway": "mlp",
        "interaction": True,
        "gate": True,
        "curriculum": True,
        "label": "F JevX: MLP state path + interaction + gate + curriculum",
    },
}

#: Canonical execution order (references first, then the matrix A..F).
ALL_CELLS: Tuple[str, ...] = (
    "p1_linear",
    "p2_mlp",
    "p4_state_only_bypass",
    "A_linear_interact",
    "B_mlp_no_interact",
    "C_mlp_interact",
    "D_gate",
    "E_curriculum",
    "F_gate_curriculum",
)

#: Curriculum cells spend the SAME total epoch budget as every other cell:
#: ``stage_a_epochs + stage_b_epochs == DATASETS[...]["epochs"]``.
STAGE_A_EPOCHS = 2
STAGE_B_EPOCHS = 2

#: The interaction mode under test for every JevX cell (the SHIPPED default).
INTERACTION_MODE = "bilinear"

#: The λ a conditioned cell starts and ends at. Pinned explicitly so no config
#: default is in play (config 0.0 maps to head 1.0 by design).
LAMBDA_ON = 1.0
LAMBDA_OFF = 0.0


# --------------------------------------------------------------------------
# The wrapper: the shipped decision path on precomputed pooled features
# --------------------------------------------------------------------------


METRIC_KEYS: Tuple[str, ...] = (
    "accuracy",
    "macro_f1",
    "nll",
    "ece",
    "brier",
    "mce",
    "train_loss_first",
    "train_loss_last",
    "val_loss_first",
    "val_loss_last",
    "val_accuracy_last",
    "trainable_params",
)


def numeric_aggregate(
    per_seed: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """``{metric: {mean, std}}`` over the numeric per-seed metrics only.

    ``bench_common.mean_std_over_seeds`` walks EVERY key of the row, which the
    rich rows of this harness (curves, grad norms, wiring) do not satisfy, so
    the aggregation is restricted to :data:`METRIC_KEYS`.
    """
    if not per_seed:
        raise ValueError("numeric_aggregate: empty input")
    out: Dict[str, Dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = [
            float(row[key]) for row in per_seed if row.get(key) is not None
        ]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        out[key] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
    return out


def _states_to_tensor(states: Sequence[Any]) -> torch.Tensor:
    """Stack precomputed pooled features into ``[B, D]`` float32."""
    if isinstance(states, np.ndarray):
        array = np.asarray(states, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return torch.from_numpy(array)
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in states]
    if not rows:
        raise ValueError("no states to score")
    return torch.from_numpy(np.ascontiguousarray(np.stack(rows, axis=0)))


def _first_text(value: Any, fallback: str) -> str:
    """Normalize the ``questions`` spellings to the benchmark's one fixed question.

    The same question reaches this wrapper as a bare ``str``, a per-record
    ``[q]``, a per-question ``[[q]]`` or a full ``[B][m]`` nest, depending on
    which caller drove it (HeadTrainer batch mapping, the curriculum's
    ``_forward_kwargs``, the gate's ``_group_batch``). The benchmark has ONE
    question for every record, so unwrapping to the first string is exact; the
    length bound keeps a malformed nest a loud failure rather than an infinite
    walk.
    """
    node = value
    for _ in range(6):
        if node is None:
            return fallback
        if isinstance(node, str):
            return node
        if isinstance(node, Mapping):
            if isinstance(node.get("text"), str):
                return str(node["text"])
            return fallback
        if not isinstance(node, Sequence):
            return fallback
        items = list(node)
        if not items:
            return fallback
        node = items[0]
    return fallback


def _first_schema(value: Any, fallback: Sequence[str]) -> List[str]:
    """Normalize the ``options`` spellings to the benchmark's one fixed schema.

    The schema arrives as ``[K]`` (one call), ``[1][K]`` (per record) or
    ``[B][m][K]`` (the full collated nest); the answer is the innermost list
    whose entries are all strings, which is the option list itself.
    """
    node = value
    for _ in range(6):
        if node is None or isinstance(node, Mapping):
            return list(fallback)
        if isinstance(node, str):
            return [node]
        if not isinstance(node, Sequence):
            return list(fallback)
        items = list(node)
        if not items:
            return list(fallback)
        if all(isinstance(item, str) for item in items):
            return [str(item) for item in items]
        node = items[0]
    return list(fallback)


class FeatureJevX(nn.Module):
    """The shipped ``JevXForDecision`` decision path on POOLED features.

    Steps 2-8 of ``JevXForDecision.forward`` (``state_proj`` -> question
    encoder/proj -> schema encoder/proj -> ``fusion`` -> ``heads["choice"]``)
    with ``state`` replaced by the precomputed pooled feature row, which is
    exactly what ``benchmarks/bench_common.FeaturizedJevX`` does for the
    featurize-once protocol. The addition over that wrapper is the shipped
    per-call ``condition_options`` flag, wired EXACTLY as
    ``JevXForDecision.forward`` wires it:

    * fusion receives ``options=None`` / ``option_mask=None`` when the flag is
      ``False``, so the schema rows are removed BEFORE fusion and the fused
      state is byte-identical to the conditioned call's;
    * the head receives EXACT ZEROS of the original ``[B, K, D]`` shape so the
      K-column contract is unchanged;
    * the head is told ``duplicate_tie=False`` on a state-only call, because
      exact zeros are "identical" for every schema and the tie would average
      the K per-class state scores into one value.

    It speaks the call convention the curriculum and the gate use
    (``state=`` / ``questions=`` / ``options=`` / ``decision_type=`` /
    ``condition_options=``) and returns a real :class:`JevXOutput` whose
    ``decisions`` is one :class:`DecisionOutput` per (state, question) pair in
    the documented state-major order, so ``measure_modes`` /
    ``_logits_from_output`` / ``HeadTrainer`` can all drive it unchanged.
    ``model.config`` is the real ``JevXConfig``, so :func:`apply_gate` writes
    ``config.conditioning_enabled`` and the ``condition_options=None`` default
    picks it up - the gate path is the shipped one, not a re-implementation.
    """

    def __init__(self, model: Any, question: str, options: Sequence[str]) -> None:
        super().__init__()
        self.jevx = model
        self.question = question
        self.options = list(options)

    @property
    def config(self):
        """The real ``JevXConfig`` (what ``apply_gate`` writes into)."""
        return self.jevx.config

    @property
    def head(self):
        return self.jevx.heads["choice"]

    def _question_rows(self, batch_size: int, question: str) -> torch.Tensor:
        """The question row, re-encoded every call (its gradients must flow)."""
        m = self.jevx
        encoded = m.question_encoder.encode_texts([question], tokenizer=m.tokenizer)
        rows = m.question_proj(m.question_encoder(encoded))  # [1, D]
        return rows.expand(batch_size, -1).contiguous()

    def _schema_rows(
        self, batch_size: int, options: Sequence[str]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """The option rows, re-encoded every call (gradients must flow)."""
        m = self.jevx
        row = [Option(key=str(o), label=str(o)) for o in options]
        feats, mask = m.schema_encoder.encode([row], tokenizer=m.tokenizer)
        feats = m.schema_proj(feats)  # [1, K, D]
        option_mask = mask if (mask is not None and not bool(mask.all())) else None
        feats = feats.expand(batch_size, -1, -1).contiguous()
        mask_b = (
            option_mask.expand(batch_size, -1).contiguous()
            if option_mask is not None
            else None
        )
        return feats, mask_b

    def forward(
        self,
        state: Any = None,
        questions: Any = None,
        decision_type: Any = None,
        options: Any = None,
        *,
        condition_options: Optional[bool] = None,
        **kwargs: Any,
    ) -> JevXOutput:
        # ``HeadTrainer``/``make_supervised_step_fn`` try ``model(batch)`` before
        # ``model(**batch)``; accept the collated mapping in the positional slot
        # so both spellings work, exactly like the shipped duck-typed contract.
        if isinstance(state, Mapping):
            batch = dict(state)
            batch.update(kwargs)
            state = batch.get("state")
            questions = batch.get("questions")
            options = batch.get("options")
            if batch.get("decision_type") is not None:
                decision_type = batch.get("decision_type")
            if "condition_options" in batch:
                condition_options = batch.get("condition_options")
        if state is None:
            raise ValueError("FeatureJevX: no `state` (pooled feature) was given.")
        if decision_type is not None:
            values = (
                decision_type
                if isinstance(decision_type, (list, tuple))
                else [decision_type]
            )
            if {str(v) for v in values} != {"choice"}:
                raise ValueError(
                    "FeatureJevX: this benchmark only scores `choice`; got "
                    f"{decision_type!r}."
                )
        if condition_options is None:
            condition = bool(
                getattr(self.jevx.config, "conditioning_enabled", True)
            )
        else:
            condition = bool(condition_options)

        question = _first_text(questions, self.question)
        schema = _first_schema(options, self.options)
        feats = _states_to_tensor(state)
        batch_size = int(feats.shape[0])

        m = self.jevx
        state_repr = m.state_proj(feats)  # [B, D]
        question_rows = self._question_rows(batch_size, question)
        option_features, option_mask = self._schema_rows(batch_size, schema)
        fused_output = m.fusion(
            state_repr,
            question_rows,
            option_features if condition else None,
            option_mask if condition else None,
        )
        head_options = fused_output.option_features
        if head_options is None and option_features is not None:
            head_options = torch.zeros_like(option_features)
        head_output = m.heads["choice"](
            fused_output.fused,
            head_options,
            option_mask=option_mask,
            conditioning_enabled=condition,
            duplicate_tie=condition,
        )
        logits = head_output.logits  # [B, K]
        decisions = [
            DecisionOutput(
                logits=logits[row : row + 1],
                probabilities=head_output.probabilities[row : row + 1],
                predicted_index=head_output.predicted_index[row : row + 1],
                schema_metadata={"option_keys": list(schema)},
            )
            for row in range(batch_size)
        ]
        return JevXOutput(
            decisions=decisions,
            state_representation=state_repr,
            question_representations=question_rows,
            schema_representations=option_features,
        )


class LogitView(nn.Module):
    """View a :class:`FeatureJevX` as a plain ``[B, K]`` logit module.

    Mirrors ``jevx.training.curriculum._CurriculumModelView``: the model is held
    as a child so ``parameters()`` / ``train()`` / ``eval()`` stay pointed at
    the real module, while ``forward`` returns the stacked logits that
    ``HeadTrainer`` and ``bench_common.collect_logits`` expect.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return _logits_from_output(self.model(batch))


# --------------------------------------------------------------------------
# Parameter accounting and per-module gradient norms
# --------------------------------------------------------------------------

#: Which reported module a named parameter belongs to. The reported groups are
#: deliberately overlapping: ``option_pathway`` is the SUM of the interaction
#: residual, the schema pathway and the fusion (everything only the option
#: conditioning can reach), while ``fusion``/``schema``/``state_pathway`` are
#: also reported on their own so a single stage's movement is visible.
_INTERACTION_PREFIXES = ("bilinear_weight", "mlp.")


def parameter_group(name: str) -> str:
    """Map a full parameter name to one of the reported module groups.

    Wrapper prefixes (``LogitView.model.`` / ``FeatureJevX.jevx.``) and the
    ``heads.<name>.`` head prefix are stripped first, so the head's OWN
    parameters - the state pathway and the interaction residual - land in their
    own groups rather than in a generic "head" bucket.
    """
    for prefix in ("model.jevx.", "jevx.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.startswith("heads."):
        parts = name.split(".", 2)
        name = parts[2] if len(parts) == 3 else ""
    if name.startswith("state_pathway."):
        return "state_pathway"
    if name.startswith("state_proj."):
        return "state_proj"
    if name.startswith("question_encoder.") or name.startswith("question_proj."):
        return "question"
    if name.startswith("schema_encoder.") or name.startswith("schema_proj."):
        return "schema"
    if name.startswith("fusion."):
        return "fusion"
    if name == "bilinear_weight" or name.startswith("mlp."):
        return "option_interaction"
    if name == "interaction_scale":
        return "interaction_scale_buffer"
    if name:
        return "head_other"
    return "other"


#: Groups that only the OPTION CONDITIONING can reach (``option_pathway``).
_OPTION_GROUPS = ("option_interaction", "schema", "fusion")
def group_parameter_counts(module: nn.Module) -> Dict[str, int]:
    """Trainable parameter count per reported group (plus the total)."""
    counts: Dict[str, int] = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_group(name)
        counts[group] = counts.get(group, 0) + int(parameter.numel())
    counts["option_pathway"] = sum(counts.get(g, 0) for g in _OPTION_GROUPS)
    counts["state_stack"] = counts.get("state_pathway", 0) + counts.get(
        "state_proj", 0
    )
    counts["total"] = sum(
        v for k, v in counts.items() if k not in ("option_pathway", "state_stack")
    )
    return counts


def group_grad_norms(
    module: nn.Module, per_param: Mapping[str, float]
) -> Dict[str, Any]:
    """Aggregate per-parameter grad norms into the reported module groups."""
    groups: Dict[str, float] = {}
    for name, value in per_param.items():
        group = parameter_group(name)
        groups[group] = groups.get(group, 0.0) + value * value
    result = {k: math.sqrt(v) for k, v in groups.items()}
    result["option_pathway"] = math.sqrt(
        sum(result.get(g, 0.0) ** 2 for g in _OPTION_GROUPS)
    )
    result["state_stack"] = math.sqrt(
        result.get("state_pathway", 0.0) ** 2 + result.get("state_proj", 0.0) ** 2
    )
    result["total"] = math.sqrt(sum(v * v for v in per_param.values()))
    return result


class GradProbe:
    """Per-parameter grad norms at the END of epoch 1 and of the final epoch.

    A ``register_post_accumulate_grad_hook`` fires right after the gradient is
    accumulated into ``.grad`` and BEFORE ``clip_grad_norm_``/``optimizer.step``
    run, so the numbers are the gradient the optimizer actually consumed
    (un-clipped, pre-step) rather than a re-derivation. Per parameter the
    MAXIMUM over the epoch's steps is kept, so the number is the strongest
    gradient the module saw in that epoch, not a single arbitrary step.
    """

    def __init__(self, module: nn.Module, epochs: int) -> None:
        self.module = module
        self.epochs = max(1, int(epochs))
        self.epoch = 1
        self.buckets: Dict[int, Dict[str, float]] = {}
        self._handles: List[Any] = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            self._handles.append(
                parameter.register_post_accumulate_grad_hook(self._make_hook(name))
            )

    def _make_hook(self, name: str) -> Callable[..., None]:
        def hook(grad: torch.Tensor) -> None:
            bucket = self.buckets.setdefault(self.epoch, {})
            bucket[name] = max(
                bucket.get(name, 0.0),
                float(grad.detach().to(dtype=torch.float64).norm()),
            )

        return hook

    def advance(self, _record: Any = None) -> None:
        """Move to the next epoch (stop past the last one)."""
        if self.epoch < self.epochs:
            self.epoch += 1

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            "epoch1": group_grad_norms(self.module, self.buckets.get(1, {})),
            "final": group_grad_norms(
                self.module, self.buckets.get(self.epochs, self.buckets.get(1, {}))
            ),
        }

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # pragma: no cover - best effort teardown
                pass


class EpochTicker:
    """Map-style loader wrapper that advances a :class:`GradProbe` every
    ``steps_per_epoch`` batches.

    ``run_curriculum`` calls ``HeadTrainer.fit(..., epochs=1)`` once per
    curriculum epoch and exposes NO per-epoch callback, so the flat path's
    callback is not available there; counting batches is the one hook both
    drivers share.
    """

    def __init__(self, loader: Any, steps_per_epoch: int, probe: GradProbe) -> None:
        self.loader = loader
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.probe = probe
        # CUMULATIVE across iterations: ``run_curriculum`` calls
        # ``fit(epochs=1)`` once per curriculum epoch, and each call iterates
        # the loader from scratch, so a per-``__iter__`` counter would never
        # reach one epoch's worth of steps.
        self._count = 0

    @property
    def generator(self):
        return getattr(self.loader, "generator", None)

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if self._count and self._count % self.steps_per_epoch == 0:
                self.probe.advance()
            self._count += 1
            yield batch


# --------------------------------------------------------------------------
# Training: ONE shared protocol for every cell
# --------------------------------------------------------------------------


def _loader(
    records: List[Dict[str, Any]],
    batch_size: int,
    *,
    shuffle: bool,
    seed: Optional[int] = None,
):
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    return bc.make_loader(records, batch_size, shuffle=shuffle, generator=generator)


def train_flat(
    view: nn.Module,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    *,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> Tuple[HeadTrainer, List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """HeadTrainer + choice_loss, the EXP-H protocol, with the grad probe on.

    Identical optimizer / lr / grad-clip / epochs / batch to
    ``bench_common.train_heads``; the only addition is the per-module gradient
    probe and an eval loader so the train/val gap is measurable.
    """
    torch.manual_seed(seed)
    trainer = HeadTrainer(view, lr=lr, seed=seed)
    train_loader = _loader(train_records, batch_size, shuffle=True, seed=seed)
    eval_loader = _loader(eval_records, batch_size, shuffle=False)
    probe = GradProbe(view, epochs)
    step = make_supervised_step_fn(view)
    history = trainer.fit(
        train_loader,
        epochs=epochs,
        eval_loader=eval_loader,
        loss_fn=step,
        callback=probe.advance,
    )
    grads = probe.snapshot()
    probe.close()
    return trainer, list(history.epochs), grads


def train_curriculum(
    model: FeatureJevX,
    view: LogitView,
    train_records: List[Dict[str, Any]],
    eval_records: List[Dict[str, Any]],
    *,
    seed: int,
    lr: float,
    epochs: int,
    batch_size: int,
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """The two-stage curriculum (Stage A state-only -> Stage B lambda ramp).

    Same optimizer / lr / grad-clip / loss / total epoch budget as
    :func:`train_flat`; ``stage_a_epochs + stage_b_epochs == epochs``.
    """
    torch.manual_seed(seed)
    raw_loader = _loader(train_records, batch_size, shuffle=True, seed=seed)
    eval_loader = _loader(eval_records, batch_size, shuffle=False)
    probe = GradProbe(view, epochs)
    train_loader = EpochTicker(raw_loader, len(raw_loader), probe)
    curriculum = CurriculumConfig(
        stage_a_epochs=STAGE_A_EPOCHS,
        stage_b_epochs=STAGE_B_EPOCHS,
        lambda_start=LAMBDA_OFF,
        lambda_end=LAMBDA_ON,
        lambda_schedule="linear",
        lr=lr,
        weight_decay=0.0,
        grad_clip=1.0,
        seed=seed,
        decision_type="choice",
    )
    history = run_curriculum(model, train_loader, {"cal": eval_loader},
                             curriculum=curriculum)
    grads = probe.snapshot()
    probe.close()
    return history, list(history.to_dict()["records"]), grads


# --------------------------------------------------------------------------
# Model construction (one base per config per seed, deep-copied per cell)
# --------------------------------------------------------------------------


def _new_jevx(cfg: Dict[str, Any], in_dim: int, **config_kwargs: Any):
    """A fresh JevX per (config, seed), built from the frozen backbone.

    The decision/question/schema widths are pinned to the cached feature width
    when it differs from the tiny BERT's 128, so ``state_proj`` accepts the
    precomputed features unchanged (audio: wav2vec2 emits 768).
    """
    kwargs: Dict[str, Any] = {}
    if in_dim != 128:
        kwargs.update(
            decision_hidden_size=in_dim,
            question_hidden_size=in_dim,
            schema_hidden_size=in_dim,
        )
    kwargs.update(config_kwargs)
    backbone = cfg["backbone"] or hq.TEXT_BACKBONE
    model = bc.JevX.from_pretrained(
        backbone,
        freeze_base_model=True,
        pooling_method=cfg["pooling"],
        fusion_type=cfg["fusion"],
        **kwargs,
    )
    model.eval()
    return model


def build_jevx_cell(
    cell: str,
    spec: Mapping[str, Any],
    cfg: Dict[str, Any],
    in_dim: int,
    base: Any,
    question: str,
    options: Sequence[str],
) -> Tuple[FeatureJevX, Dict[str, Any]]:
    """Instantiate one JevX cell from the shared base; record its wiring."""
    state_pathway = spec["state_pathway"]
    interaction_on = bool(spec["interaction"])
    model = copy.deepcopy(base)
    # Re-wire the choice head through the shipped registry so the cell's
    # configuration is expressed with the SHIPPED constructor, not a patch.
    model.heads["choice"] = build_decision_head(
        "choice",
        hidden_size=int(model.config.decision_hidden_size),
        interaction=INTERACTION_MODE,
        state_pathway=state_pathway,
        conditioning_scale_init=LAMBDA_ON,
        conditioning_enabled=True,
    )
    head = model.heads["choice"]
    # Keep the config's selector in step with the head that was actually built,
    # so the recorded wiring and `config.conditioning_enabled` (what the gate
    # writes) describe the real cell.
    model.config.state_pathway = head.state_pathway_name
    model.config.state_hidden_size = head.state_hidden_size
    model.config.state_activation = head.state_activation
    model.config.conditioning_enabled = True
    if not interaction_on:
        # EXPLICIT off: both switches, neither a config default.
        model.config.conditioning_enabled = False
        head.conditioning_enabled = False
        head.set_interaction_scale(LAMBDA_OFF)
    else:
        head.set_interaction_scale(LAMBDA_ON)
    wiring = {
        "state_pathway_configured": str(state_pathway),
        "state_pathway_built": str(getattr(head, "state_pathway_name", "?")),
        "state_pathway_module": type(
            getattr(head, "state_pathway", None)
        ).__name__,
        "state_hidden_size": int(head.state_hidden_size),
        "state_activation": str(head.state_activation),
        "interaction_mode": INTERACTION_MODE,
        "interaction_scale": head.get_interaction_scale,
        "conditioning_enabled_config": bool(model.config.conditioning_enabled),
        "conditioning_enabled_head": bool(head.conditioning_enabled),
    }
    # Assert the enforcement, so a silent default change cannot hide here.
    assert wiring["state_pathway_built"] == str(state_pathway), wiring
    if not interaction_on:
        assert wiring["interaction_scale"] == LAMBDA_OFF, wiring
        assert wiring["conditioning_enabled_config"] is False, wiring
    return FeatureJevX(model, question, options), wiring


def build_reference_cell(
    cell: str, cfg: Dict[str, Any], feats: Mapping[str, np.ndarray],
    num_classes: int, question: str, options: Sequence[str], seed: int,
    base: Any,
) -> nn.Module:
    """P1 / P2 / P4 - the reference cells, built exactly as EXP-H builds them."""
    torch.manual_seed(seed)
    in_dim = int(feats["train"].shape[1])
    if cell == "p1_linear":
        return bc.LinearProbe(in_dim, num_classes)
    if cell == "p2_mlp":
        torch.manual_seed(seed + 10_000)  # mirrors the B2 seeding of run_exp7
        return bc.MLPProbe(in_dim, num_classes)
    if cell == "p4_state_only_bypass":
        return hq.JevXStack(copy.deepcopy(base), question, options, state_only=True)
    raise ValueError("unknown reference cell " + repr(cell))


# --------------------------------------------------------------------------
# Diagnostics + latency
# --------------------------------------------------------------------------


def state_only_diagnostic(
    model: FeatureJevX, records: List[Dict[str, Any]], batch_size: int
) -> Dict[str, float]:
    """Check the state-only path is NOT degenerate on the real stack.

    ``state_only`` = ``condition_options=False``; the shipped fix suppresses
    the duplicate-option tie there, so the K logits must DISCRIMINATE the
    options (spread > 0) instead of collapsing to one value. ``max_abs_diff``
    compares the two ways of reaching the same state-only scoring function
    (``condition_options=False`` vs ``set_interaction_scale(0.0)`` with
    conditioning on) and must be 0.
    """
    module = model
    module.eval()
    batch = {
        key: [record[key] for record in records[:batch_size]]
        for key in ("state", "questions", "options", "decision_type")
    }
    with torch.no_grad():
        state_only = _logits_from_output(module(batch, condition_options=False))
        head = module.head
        saved = head.get_interaction_scale
        head.set_interaction_scale(LAMBDA_OFF)
        zero_lambda = _logits_from_output(module(batch, condition_options=True))
        head.set_interaction_scale(saved)
        conditioned = _logits_from_output(module(batch, condition_options=True))
    spread = (state_only.max(dim=-1).values - state_only.min(dim=-1).values)
    return {
        "state_only_logit_spread_mean": float(spread.mean()),
        "state_only_logit_spread_min": float(spread.min()),
        "state_only_logit_spread_max": float(spread.max()),
        "state_only_tied_rows": int((spread <= 0.0).sum()),
        "state_only_vs_zero_lambda_max_abs_diff": float(
            (state_only - zero_lambda).abs().max()
        ),
        "conditioned_logit_spread_mean": float(
            (conditioned.max(dim=-1).values - conditioned.min(dim=-1).values).mean()
        ),
    }


def measure_cell_latency(
    module: nn.Module, records: List[Dict[str, Any]], batch_size: int,
    head_only: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Batch-1 latency: warmup then timed iterations, e2e and head-only."""
    row = {
        key: record[key]
        for key in ("state", "questions", "options", "decision_type")
        for record in records[:1]
    }
    # A [1, D] matrix (not a 1-D vector): the batch-1 forward must keep the
    # rank-2 state contract every module validates.
    row["state"] = np.asarray([np.asarray(row["state"], dtype=np.float32)])
    out: Dict[str, Any] = {"protocol": {
        "batch_size": 1, "warmup": LATENCY_WARMUP, "iters": LATENCY_ITERS,
        "note": "pooled features are precomputed, so the JevX e2e path is the "
                "decision stack (state_proj -> question/schema -> fusion -> head); "
                "the frozen backbone is NOT in this measurement.",
    }}
    module.eval()
    with torch.no_grad():
        out["e2e"] = bc.measure_latency(
            lambda: module(row), warmup=LATENCY_WARMUP, iters=LATENCY_ITERS
        )
    if head_only is not None:
        with torch.no_grad():
            out["head_only"] = bc.measure_latency(
                head_only, warmup=LATENCY_WARMUP, iters=LATENCY_ITERS
            )
    return out


def _jevx_head_only_call(model: FeatureJevX) -> Callable[[], Any]:
    """A batch-1 closure that runs ONLY ``heads['choice']`` on cached tensors."""
    m = model.jevx
    feats = torch.zeros(1, int(m.config.decision_hidden_size), dtype=torch.float32)
    option_rows = torch.zeros(
        1, len(model.options), int(m.config.decision_hidden_size),
        dtype=torch.float32,
    )
    head = m.heads["choice"]

    def call() -> Any:
        return head(feats, option_rows)

    return call


# --------------------------------------------------------------------------
# One cell, one seed
# --------------------------------------------------------------------------


def run_cell(
    cell: str,
    *,
    cfg: Dict[str, Any],
    feats: Mapping[str, np.ndarray],
    labels: Mapping[str, List[int]],
    num_classes: int,
    question: str,
    options: Sequence[str],
    seed: int,
    bases: Mapping[str, Any],
    with_latency: bool,
) -> Dict[str, Any]:
    """Train + score ONE cell for ONE seed under the shared protocol."""
    spec = CELLS[cell]
    in_dim = int(feats["train"].shape[1])
    train_records = bc.build_records(feats["train"], labels["train"], question, options)
    cal_records = bc.build_records(feats["cal"], labels["cal"], question, options)
    test_records = bc.build_records(feats["test"], labels["test"], question, options)
    epochs = int(cfg["epochs"])

    wiring: Dict[str, Any] = {}
    gate_log: Dict[str, Any] = {}
    grads: Dict[str, Dict[str, float]] = {}
    diagnostics: Dict[str, float] = {}
    curve: List[Dict[str, Any]] = []
    gate_measurement: Dict[str, Any] = {}

    torch.manual_seed(seed)
    if spec["kind"] == "reference":
        base = bases["linear"]
        module = build_reference_cell(
            cell, cfg, feats, num_classes, question, options, seed, base
        )
        trainer, history, grads = train_flat(
            module, train_records, cal_records, seed=seed, lr=cfg["lr"],
            epochs=epochs, batch_size=cfg["batch_size"],
        )
        view = module
        curve = [dict(h) for h in history]
        feature_model = None
    else:
        base = bases[str(spec["state_pathway"])]
        feature_model, wiring = build_jevx_cell(
            cell, spec, cfg, in_dim, base, question, options
        )
        view = LogitView(feature_model)
        if spec["curriculum"]:
            history, curve, grads = train_curriculum(
                feature_model, view, train_records, cal_records, seed=seed,
                lr=cfg["lr"], epochs=epochs, batch_size=cfg["batch_size"],
            )
        else:
            _, history, grads = train_flat(
                view, train_records, cal_records, seed=seed, lr=cfg["lr"],
                epochs=epochs, batch_size=cfg["batch_size"],
            )
            curve = [dict(h) for h in history]

        if spec["gate"]:
            # The gate sees ONLY the calibration split. The reported test
            # metrics below are computed AFTER the verdict, never fed to it.
            cal_loader = _loader(cal_records, cfg["batch_size"], shuffle=False)
            gate = ConditioningGate(
                feature_model,
                ConditioningGateConfig(split_name=GATE_SPLIT_NAME,
                                       decision_type="choice"),
            )
            before = _logits_from_output(
                feature_model(_single(cal_records, cfg["batch_size"]),
                              condition_options=True)
            )
            before_y = np.asarray(
                [record["labels"][0] for record in cal_records[: len(before)]],
                dtype=np.int64,
            )
            decision = gate.decide(cal_loader)
            flag = apply_gate(feature_model, gate)
            after = _logits_from_output(
                feature_model(_single(cal_records, cfg["batch_size"]),
                              condition_options=None)
            )
            after_y = before_y
            measurement = gate.measurement
            gate_measurement = {
                "conditioned_nll": float(measurement.conditioned_nll),
                "state_only_nll": float(measurement.state_only_nll),
                "delta_nll": float(measurement.delta_nll),
                "conditioned_accuracy": float(measurement.conditioned_accuracy),
                "state_only_accuracy": float(measurement.state_only_accuracy),
                "tau": float(gate.config.tau) if gate.config.tau is not None else None,
                "tau_source": gate.config.tau_source,
                "decision": decision,
                "applied_flag": bool(flag),
                "config_conditioning_enabled": bool(
                    feature_model.config.conditioning_enabled
                ),
                "num_samples": int(measurement.num_samples),
                "num_batches": int(measurement.num_batches),
                "per_batch_delta_mean": float(
                    sum(measurement.per_batch_delta)
                    / max(1, len(measurement.per_batch_delta))
                ),
                "per_batch_delta": [float(v) for v in measurement.per_batch_delta],
                "cal_nll_before_apply": float(
                    bc.metrics_from_logits(before, before_y)["nll"]
                ),
                "cal_nll_after_apply": float(
                    bc.metrics_from_logits(after, after_y)["nll"]
                ),
            }
            gate_log = dict(gate_measurement)
        diagnostics = state_only_diagnostic(
            feature_model, test_records, min(cfg["batch_size"], 64)
        )

    test_logits, test_labels = bc.collect_logits(
        view, test_records, batch_size=cfg["batch_size"]
    )
    m = bc.metrics_from_logits(test_logits, test_labels)
    row: Dict[str, Any] = {
        "seed": seed,
        "cell": cell,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "nll": m["nll"],
        "ece": m["ece"],
        "brier": m["brier"],
        "mce": m["mce"],
        "train_loss_first": float(curve[0].get("train_loss", curve[0].get("loss"))),
        "train_loss_last": float(curve[-1].get("train_loss", curve[-1].get("loss"))),
        "val_loss_first": (
            float(curve[0]["val_loss"]) if "val_loss" in curve[0] else None
        ),
        "val_loss_last": (
            float(curve[-1]["val_loss"]) if "val_loss" in curve[-1] else None
        ),
        "val_accuracy_last": (
            float(curve[-1]["val_accuracy"]) if "val_accuracy" in curve[-1] else None
        ),
        "convergence_curve": curve,
        "grad_norms": grads,
        "parameter_groups": group_parameter_counts(
            feature_model if feature_model is not None else module
        ),
        "trainable_params": bc.count_trainable(
            feature_model if feature_model is not None else module
        ),
        "state_only_diagnostic": diagnostics,
        "wiring": wiring,
        "gate": gate_log,
    }
    if with_latency:
        head_only = (
            _jevx_head_only_call(feature_model) if feature_model is not None else None
        )
        row["latency"] = measure_cell_latency(
            view, test_records, cfg["batch_size"], head_only
        )
    del view, feature_model
    return row


def _single(records: List[Dict[str, Any]], batch_size: int) -> Dict[str, List[Any]]:
    """A single collated batch of at most ``batch_size`` records."""
    chunk = records[:batch_size]
    return {
        key: [record[key] for record in chunk]
        for key in ("state", "questions", "options", "decision_type")
    }


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split(chr(10))[0])
    p.add_argument("--datasets", nargs="+",
                   default=["synthetic", "ag_news", "ravdess"],
                   help="subset of " + str(sorted(hq.DATASETS)) + ", or all")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--cells", nargs="+", default=list(ALL_CELLS))
    p.add_argument("--smoke", action="store_true",
                   help="tiny sizes for pipeline validation (never reportable)")
    p.add_argument("--output-dir", default=str(RESULTS_DIR))
    p.add_argument("--recompute-features", action="store_true")
    p.add_argument("--latency-seeds", default="0",
                   help="seeds on which the batch-1 latency probe runs")
    return p.parse_args(argv)


def run_dataset(name: str, args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(hq.DATASETS[name])
    if args.smoke:
        cfg["train_size"] = min(cfg["train_size"], 64)
        cfg["cal_size"] = min(cfg["cal_size"], 32)
        cfg["test_size"] = min(cfg["test_size"], 48)
        cfg["epochs"] = 2
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    latency_seeds = {int(s) for s in args.latency_seeds.split(",") if s.strip()}
    cells = [c for c in ALL_CELLS if c in set(args.cells)]
    t0 = time.perf_counter()
    print("", flush=True)
    print("[=== " + name + " ===] protocol from " + cfg["source"], flush=True)

    if cfg["modality"] == "synthetic":
        feats, labels, meta = hq.synthetic_task(cfg)
        source = "computed"
    else:
        feats, labels, meta, source = hq.build_features(
            name, cfg, recompute=args.recompute_features
        )
    num_classes = len(cfg["options"])
    in_dim = int(feats["train"].shape[1])

    rows_by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cells}
    failures: Dict[str, str] = {}
    t_cells = time.perf_counter()
    for seed in seeds:
        print("  -- seed " + str(seed), flush=True)
        # One base per state-pathway configuration per seed, deep-copied per
        # cell: cells sharing a configuration start from byte-identical weights.
        bases: Dict[str, Any] = {}
        for pathway in ("linear", "mlp"):
            torch.manual_seed(seed)
            bases[pathway] = hq._new_jevx(cfg, in_dim=in_dim)
        for cell in cells:
            ct = time.perf_counter()
            try:
                rows_by_cell[cell].append(run_cell(
                    cell, cfg=cfg, feats=feats, labels=labels,
                    num_classes=num_classes, question=cfg["question"],
                    options=cfg["options"], seed=seed, bases=bases,
                    with_latency=seed in latency_seeds,
                ))
            except Exception as exc:  # noqa: BLE001 - recorded, never silent
                failures[cell + "/seed" + str(seed)] = (
                    type(exc).__name__ + ": " + str(exc)
                )
                print("[FAILED] " + name + "/" + cell + "/" + str(seed) + ": " +
                      failures[cell + "/seed" + str(seed)], flush=True)
                traceback.print_exc()
                continue
            row = rows_by_cell[cell][-1]
            print("     " + cell.ljust(22) + " acc=" +
                  ("%.4f" % row["accuracy"]) +
                  "  f1=" + ("%.4f" % row["macro_f1"]) +
                  "  nll=" + ("%.4f" % row["nll"]) +
                  "  (" + str(round(time.perf_counter() - ct, 1)) + "s)", flush=True)
        del bases
    results: Dict[str, Any] = {}
    for cell in cells:
        rows = rows_by_cell[cell]
        if not rows:
            continue
        results[cell] = {
            "label": CELLS[cell]["label"],
            "spec": dict(CELLS[cell]),
            "per_seed": rows,
            "aggregate": numeric_aggregate(rows),
            "parameter_groups": rows[0]["parameter_groups"],
            "state_only_diagnostic": rows[0]["state_only_diagnostic"],
            "gate": rows[0]["gate"],
            "latency": rows[0].get("latency"),
            "grad_norms_epoch1": rows[0]["grad_norms"].get("epoch1"),
            "grad_norms_final": rows[0]["grad_norms"].get("final"),
        }
        agg = results[cell]["aggregate"]
        print("  " + cell.ljust(22) + " acc=" +
              ("%.4f+-%.4f" % (agg["accuracy"]["mean"], agg["accuracy"]["std"])) +
              "  f1=" + ("%.4f" % agg["macro_f1"]["mean"]) +
              "  nll=" + ("%.4f" % agg["nll"]["mean"]) +
              "  (" + str(round(time.perf_counter() - t_cells, 1)) + "s)", flush=True)

    ranking = sorted(
        ((c, results[c]["aggregate"]["accuracy"]["mean"]) for c in results),
        key=lambda kv: -kv[1],
    )
    payload: Dict[str, Any] = {
        "experiment": "jevx_next_ablation_matrix",
        "dataset": name,
        "smoke": bool(args.smoke),
        "protocol": {
            "seeds": seeds,
            "cells": cells,
            "epochs": cfg["epochs"],
            "stage_a_epochs": STAGE_A_EPOCHS if cfg["epochs"] == hq.DATASETS[name]["epochs"] else None,
            "stage_b_epochs": STAGE_B_EPOCHS if cfg["epochs"] == hq.DATASETS[name]["epochs"] else None,
            "lr": cfg["lr"],
            "batch_size": cfg["batch_size"],
            "train_n": int(feats["train"].shape[0]),
            "cal_n": int(feats["cal"].shape[0]),
            "test_n": int(feats["test"].shape[0]),
            "feature_dim": in_dim,
            "num_classes": num_classes,
            "question": cfg["question"],
            "options": list(cfg["options"]),
            "optimizer": "torch.optim.AdamW (HeadTrainer default), grad_clip=1.0",
            "loss": "jevx.training.choice_loss (masked cross-entropy)",
            "gate_split": GATE_SPLIT_NAME,
            "latency": {"batch_size": 1, "warmup": LATENCY_WARMUP,
                        "iters": LATENCY_ITERS},
            "interaction_mode": INTERACTION_MODE,
            "lambda_on": LAMBDA_ON,
            "lambda_off": LAMBDA_OFF,
            "note": ("IDENTICAL features / splits / seeds / optimizer / lr / epochs / "
                     "batch / loss for every cell; the backbone is frozen and "
                     "featurized once, so the only thing that varies is the scoring "
                     "function (+ gate / curriculum). `interaction OFF` is enforced "
                     "explicitly with config.conditioning_enabled=False AND "
                     "set_interaction_scale(0.0), both asserted."),
        },
        "features": dict(meta, source=source),
        "cells": results,
        "ranking_by_accuracy": [{"cell": c, "accuracy": a} for c, a in ranking],
        "failures": failures,
        "environment": bc.environment_info(),
        "wall_s": round(time.perf_counter() - t0, 2),
    }
    return payload


def combined_path(out_dir: Path, smoke: bool) -> Path:
    return out_dir / ("head_ablation_all" + ("_smoke" if smoke else "") + ".json")


def write_combined(
    out_dir: Path, smoke: bool, fresh: Dict[str, Any]
) -> Path:
    """Merge the per-dataset payloads into one index file.

    The per-dataset JSONs are the primary artifacts; this index MERGES rather
    than overwrites, so invoking the harness for one dataset does not discard
    the results already on disk for the others.
    """
    path = combined_path(out_dir, smoke)
    merged: Dict[str, Any] = {}
    if path.exists():
        try:
            merged = dict(json.loads(path.read_text(encoding="utf-8")).get("datasets", {}))
        except (OSError, ValueError):  # pragma: no cover - unreadable index
            merged = {}
    merged.update(fresh)
    # Re-read every per-dataset payload on disk, so the index is complete
    # whichever subset this invocation ran.
    for source in sorted(out_dir.glob("head_ablation_*.json")):
        name = source.stem[len("head_ablation_") :]
        if name == "all" or name.endswith("_smoke") != smoke:
            continue
        try:
            merged[name] = json.loads(source.read_text(encoding="utf-8"))
        except ValueError:  # pragma: no cover - unreadable payload
            continue
    bc.write_json(path, {
        "experiment": "jevx_next_ablation_matrix",
        "datasets": merged,
        "wall_s": round(
            sum(float(p.get("wall_s", 0.0)) for p in merged.values()), 2
        ),
    })
    print("[wrote] " + str(path), flush=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    names = sorted(hq.DATASETS) if args.datasets == ["all"] else list(args.datasets)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    everything: Dict[str, Any] = {}
    for name in names:
        if name not in hq.DATASETS:
            print("[skip] unknown dataset " + repr(name), flush=True)
            continue
        payload = run_dataset(name, args)
        suffix = "_smoke" if args.smoke else ""
        path = out_dir / ("head_ablation_" + name + suffix + ".json")
        bc.write_json(path, payload)
        everything[name] = payload
        print("[wrote] " + str(path) + " (" + str(payload["wall_s"]) +
              "s, failures=" + str(len(payload["failures"])) + ")", flush=True)
    write_combined(out_dir, args.smoke, everything)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
