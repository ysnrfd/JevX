# Copyright (c) 2026 ysnrfd
"""JevX utilities: shape/finiteness validators and stable numerical helpers."""

from .numerical import (
    BATCH_EQUIVALENCE_SAFETY,
    JEVX_DECISION_ACCUMULATION_STAGES,
    LOW_PRECISION_DTYPES,
    batch_equivalence_tolerance,
    brier_score,
    expected_calibration_error,
    masked_softmax,
    maximum_calibration_error,
    negative_log_likelihood,
    one_hot,
    stable_log_softmax,
    stable_softmax,
)
from .shapes import (
    assert_batch_alignment,
    assert_same_device_dtype,
    assert_shape,
    check_finite,
    has_inf,
    has_nan,
)

__all__ = [
    "BATCH_EQUIVALENCE_SAFETY",
    "JEVX_DECISION_ACCUMULATION_STAGES",
    "LOW_PRECISION_DTYPES",
    "assert_batch_alignment",
    "assert_same_device_dtype",
    "assert_shape",
    "batch_equivalence_tolerance",
    "brier_score",
    "check_finite",
    "expected_calibration_error",
    "has_inf",
    "has_nan",
    "masked_softmax",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "one_hot",
    "stable_log_softmax",
    "stable_softmax",
]
