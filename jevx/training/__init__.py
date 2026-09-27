# Copyright (c) 2026 ysnrfd
"""JevX training pipeline: losses, composite objective, freeze control,
decision-task data plumbing and a minimal supervised head trainer.

Import surface::

    from jevx.training import (
        choice_loss, score_loss, noul_loss, multilabel_loss,
        DecisionObjective, l2_regularization,
        freeze_module, unfreeze_module, freeze_backbone, unfreeze_backbone,
        unfreeze_last_n_layers, count_parameters,
        DecisionDataset, decision_collate_fn, make_supervised_step_fn,
        accuracy, macro_f1, confusion_matrix,
        HeadTrainer, History,
        ConditioningGate, ConditioningGateConfig, measure_delta_nll, apply_gate,
        CurriculumConfig, CurriculumHistory, EpochRecord,
        interaction_scale, read_interaction_scale, run_curriculum,
    )
"""

from .conditioning_gate import (
    ConditioningGate,
    ConditioningGateConfig,
    GateMeasurement,
    TauFit,
    apply_gate,
    assert_split_not_leakage,
    fit_tau,
    measure_delta_nll,
    measure_modes,
)
from .curriculum import (
    CurriculumConfig,
    CurriculumHistory,
    EpochRecord,
    interaction_scale,
    read_interaction_scale,
    run_curriculum,
)
from .data import (
    DecisionDataset,
    accuracy,
    confusion_matrix,
    decision_collate_fn,
    macro_f1,
    make_supervised_step_fn,
)
from .freeze import (
    count_parameters,
    freeze_backbone,
    freeze_module,
    unfreeze_backbone,
    unfreeze_last_n_layers,
    unfreeze_module,
)
from .losses import choice_loss, multilabel_loss, noul_loss, score_loss
from .objective import DecisionObjective, l2_regularization
from .trainer import HeadTrainer, History

__all__ = [
    "ConditioningGate",
    "ConditioningGateConfig",
    "CurriculumConfig",
    "CurriculumHistory",
    "DecisionDataset",
    "DecisionObjective",
    "EpochRecord",
    "GateMeasurement",
    "HeadTrainer",
    "History",
    "TauFit",
    "accuracy",
    "apply_gate",
    "assert_split_not_leakage",
    "choice_loss",
    "confusion_matrix",
    "count_parameters",
    "decision_collate_fn",
    "fit_tau",
    "freeze_backbone",
    "freeze_module",
    "interaction_scale",
    "l2_regularization",
    "macro_f1",
    "make_supervised_step_fn",
    "measure_delta_nll",
    "measure_modes",
    "multilabel_loss",
    "noul_loss",
    "read_interaction_scale",
    "run_curriculum",
    "score_loss",
    "unfreeze_backbone",
    "unfreeze_last_n_layers",
    "unfreeze_module",
]
