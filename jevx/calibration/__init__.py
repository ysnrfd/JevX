# Copyright (c) 2026 ysnrfd
"""Post-hoc calibration: ``P' = Cal(P)`` without changing schema or type.

``CalibrationEngine(method)`` resolves a registered method (``"none"``,
``"temperature"``, ``"vector"``, ``"isotonic"``) and exposes ``fit`` (on a
CALIBRATION split — never test), ``calibrate``/``apply``, ``evaluate``
(``nll``, ``ece``, ``mce``, ``brier``) and ``save``/``load`` round trips.
See ``jevx.calibration.base`` for the schema invariant.
"""

from .base import (
    CALIBRATOR_REGISTRY,
    BaseCalibrator,
    CalibrationEngine,
    IdentityCalibrator,
    available_calibrators,
    build_calibrator,
    get_calibrator,
    register_calibrator,
    validate_fit_inputs,
)
from .isotonic import IsotonicCalibrator, pava_isotonic
from .temperature import TemperatureCalibrator
from .vector import VectorCalibrator

__all__ = [
    "CALIBRATOR_REGISTRY",
    "BaseCalibrator",
    "CalibrationEngine",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "TemperatureCalibrator",
    "VectorCalibrator",
    "available_calibrators",
    "build_calibrator",
    "get_calibrator",
    "pava_isotonic",
    "register_calibrator",
    "validate_fit_inputs",
]
