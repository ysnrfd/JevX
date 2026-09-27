# Copyright (c) 2026 ysnrfd
"""Typed decision heads with a pluggable registry.

Built-ins: ``"choice"`` (softmax over options), ``"score"`` (softmax over
ordinal levels + expected score), ``"noul"`` (binary sigmoid), plus the
experimental ``"multi_label"`` and ``"regression"`` heads. New primitives
register with ``@register_decision_head("name")`` and are built by name via
:func:`build_decision_head`.

Scoring identity shared by every head::

    z_k = state_k + lambda * r_k(s_or_h_s, e_k)

with ``state_k`` the PRIMARY state score (:class:`LinearProbePathway` for
``state_pathway="linear"``, :class:`MLPStatePathway` for ``"mlp"``),
``r_k`` the option-interaction residual and ``lambda = interaction_scale`` the
buffer driven by :meth:`DecisionHead.set_interaction_scale`.
"""

from .choice import ChoiceHead
from .generic import (
    DECISION_HEAD_REGISTRY,
    DEFAULT_STATE_PATHWAY_OPTIONS,
    STATE_ACTIVATIONS,
    STATE_PATHWAYS,
    DecisionHead,
    LinearProbePathway,
    MLPStatePathway,
    MultiLabelHead,
    RegressionHead,
    available_decision_heads,
    build_decision_head,
    get_decision_head,
    mask_from_num_options,
    project_to_schema,
    register_decision_head,
)
from .noul import NoulHead
from .score import ScoreHead

__all__ = [
    "DECISION_HEAD_REGISTRY",
    "DEFAULT_STATE_PATHWAY_OPTIONS",
    "MLPStatePathway",
    "STATE_ACTIVATIONS",
    "STATE_PATHWAYS",
    "ChoiceHead",
    "DecisionHead",
    "LinearProbePathway",
    "MultiLabelHead",
    "NoulHead",
    "RegressionHead",
    "ScoreHead",
    "available_decision_heads",
    "build_decision_head",
    "get_decision_head",
    "mask_from_num_options",
    "project_to_schema",
    "register_decision_head",
]
