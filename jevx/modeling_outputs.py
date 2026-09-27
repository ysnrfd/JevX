# Copyright (c) 2026 ysnrfd
"""ModelOutput dataclasses for the JevX decision layer.

Verified against transformers v5.16.1 (``transformers/utils/generic.py``):

* ``ModelOutput`` subclasses must be ``@dataclass`` decorated, may have at
  most ONE required field, and every other field must default to ``None``.
* Fields whose value is ``None`` are omitted from the dict / ``to_tuple()``.
* ``return_dict=False`` tuple mode still exists: the ``@can_return_tuple``
  decorator converts model outputs via ``output.to_tuple()``, which returns
  the non-``None`` fields in declaration order. There is no ``to_dict`` on
  v5 ``ModelOutput`` — use ``dict(output.items())`` instead.
* The FIRST field must be a tensor (or a non-iterable); passing a ``dict``
  as the first field would be re-interpreted as a key/value mapping of field
  names. Non-tensor metadata (dicts, plain objects) therefore lives in
  trailing ``Any``-typed fields, never in the first position, and tensors are
  only ever stored in tensor-typed fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from transformers.utils import ModelOutput

__all__ = ["DecisionOutput", "JevXOutput"]


@dataclass
class DecisionOutput(ModelOutput):
    """Output of a single decision head / decision primitive.

    Shapes use ``B`` = batch size (number of questions) and ``K`` = number of
    options/levels for that decision (variable across rows via ``option_mask``).

    Args:
        logits: Raw scores ``[B, K]`` (``[B, 1]`` for Noul/Binary). Returned
            UNMASKED: invalid positions keep their raw values so downstream
            calibration sees true logits; use ``probabilities`` (which are
            mask-aware) for anything option-sensitive.
        probabilities: Mask-aware probabilities ``[B, K]`` in ``[0, 1]``;
            rows sum to 1 (softmax) except Noul/Binary where the single
            column is ``sigmoid(logit) = P(yes)``. Invalid masked positions
            are exactly 0.
        confidence: Optional per-question confidence ``[B]`` in ``[0, 1]``.
            Filled by :class:`jevx.confidence.ConfidenceEngine`, not by the
            heads (``None`` straight out of a head).
        predicted_index: Predicted option/level index ``[B]``
            (``torch.long``); for Noul/Binary it is the thresholded decision
            ``1[p_yes >= 0.5]``. Always in ``[0, K)`` / ``{0, 1}``.
        expected_score: For Score decisions, the expectation
            ``s = sum_k k * p_k`` over levels ``0..K-1`` (``[B]``, bounded by
            ``[0, K-1]``). ``None`` for other decision types.
        predicted_value: Caller-filled, schema-projected decision value
            (e.g. the option key chosen by ``project_to_schema``); any
            non-tensor payload. Heads leave it ``None`` — mapping indices to
            the schema C is the caller's job (Project step).
        abstained: Optional boolean mask ``[B]`` — ``True`` where the
            decision layer abstained instead of committing. Produced by the
            abstention policy, not by the heads.
        calibration_metadata: Free-form non-tensor metadata about an applied
            calibration (method name, temperature, fit statistics, ...).
        schema_metadata: Free-form non-tensor metadata about the decision
            schema C (option keys, level names, ...).
    """

    logits: torch.Tensor
    probabilities: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    predicted_index: torch.Tensor | None = None
    expected_score: torch.Tensor | None = None
    predicted_value: Any = None
    abstained: torch.Tensor | None = None
    calibration_metadata: Any = None
    schema_metadata: Any = None


@dataclass
class JevXOutput(ModelOutput):
    """Full output of the JevX wrapper model (``H -> P' -> D`` pipeline).

    Args:
        decisions: One :class:`DecisionOutput` per question, or stacked head
            tensors ``[B, K, ...]``. Required first field.
        state_representation: Shared *projected state* ``H_S [B, D]``
            (``H_S = Proj(E_M(S))``) — the same value every question of a
            state is answered against (T7/A7.2). Always populated by
            :class:`jevx.modeling_jevx.JevXForDecision`. This is **not**
            the per-row fused ``H = F(H_S, H_Q, H_C)``: the fused state is
            internal to the forward pass and is not exposed on the output
            (``capability_metadata`` carries no fused vectors — see
            docs/API.md).
        question_representations: Per-question embeddings ``[B, L, D_q]``
            or ``[B, D_q]`` depending on the representation strategy.
        schema_representations: Schema/option embeddings ``[B, K, D_c]``.
        base_output: Raw output of the underlying HF model, passed through
            untouched (its type belongs to the base model; kept as ``Any``).
        hidden_states: Per-layer hidden states (tuple of ``[B, L, D]``);
            only populated when ``config.output_hidden_states=True``.
        attentions: Per-layer attention weights (tuple of
            ``[B, H, L, L]``); only populated when
            ``config.output_attentions=True`` (and attention implementation
            allows it).
        capability_metadata: Non-tensor dict describing which decision
            capabilities/primitives were exercised for this batch.

    Heavy optional fields default to ``None`` so the common path allocates
    nothing extra; the modeling layer fills them only when the corresponding
    config flag is set — except ``state_representation``, which the
    wrapper always populates (it is the shared state, not an opt-in
    diagnostic). Tensors live exclusively in tensor-typed fields;
    ``capability_metadata``/``base_output`` carry non-tensor (or foreign)
    payloads and are omitted from ``to_tuple()`` when ``None``.
    """

    decisions: list[DecisionOutput] | torch.Tensor
    state_representation: torch.Tensor | None = None
    question_representations: torch.Tensor | None = None
    schema_representations: torch.Tensor | None = None
    base_output: Any = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    capability_metadata: dict[str, Any] | None = None
