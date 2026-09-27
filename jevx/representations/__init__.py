# Copyright (c) 2026 ysnrfd
"""Representation machinery: pooling, projection and fusion.

* :func:`jevx.representations.pool.pool` — the mask-aware pooling operators
  of Definition 4 (transformers v5 ships none; audit Q7).
* :class:`jevx.representations.projection.Projection` — width maps
  ``D_model -> D_decision``.
* :func:`jevx.representations.fusion.build_fusion` — the ``F(H_S, H_Q, H_C)``
  variants of Definition 5.
"""

from .fusion import (
    FUSION_REGISTRY,
    Fusion,
    FusionOutput,
    AdditiveFusion,
    BilinearFusion,
    ConcatFusion,
    DotFusion,
    GatedFusion,
    MLPFusion,
    available_fusions,
    build_fusion,
)
from .pooling import (
    POOLING_STRATEGIES,
    AttentionPooler,
    LearnerPooler,
    available_pooling_strategies,
    pool,
)
from .projection import Projection

__all__ = [
    "FUSION_REGISTRY",
    "POOLING_STRATEGIES",
    "AdditiveFusion",
    "AttentionPooler",
    "BilinearFusion",
    "ConcatFusion",
    "DotFusion",
    "Fusion",
    "FusionOutput",
    "GatedFusion",
    "LearnerPooler",
    "MLPFusion",
    "Projection",
    "available_fusions",
    "available_pooling_strategies",
    "build_fusion",
    "pool",
]
