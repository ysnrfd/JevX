# Copyright (c) 2026 ysnrfd
"""Conversion reports: the machine-readable record of *what a conversion did*.

This module formalizes the JevX "weight conversion principle"
(``docs/ARCHITECTURE.md`` §5): a checkpoint conversion is one of several
*distinguishable* operations, and the report makes the distinction explicit
and machine-checkable. JevX never claims a lossless weight conversion of the
whole system — the backbone maps 1:1, the decision system is built anew.

Modes (:class:`ConversionMode`) — the semantic contract of one conversion:

* ``REPRESENTATION_REUSE`` — backbone weights are reused verbatim (identity
  map on tensors) and every JevX-side module (fusion, heads, encoders) is
  freshly initialized. **No** source tensor was written into a JevX module.
* ``WEIGHT_TRANSPLANT`` — a *subset* of an existing JevX module's parameters
  was overwritten with tensors derived from a source checkpoint head (a
  warm start per ARCHITECTURE §5.3); untouched parameters stay fresh.
* ``HEAD_ATTACHMENT`` — *every* trainable parameter of the target module is
  now source-derived: the source head fully parameterizes the target head
  (a narrow "head graft"; still not an identity claim about the whole
  system — see the equivalence contract below).
* ``ARCHITECTURAL_CONVERSION`` — reserved for future structural rewrites
  (re-parameterizing/merging/remapping architecture). **NEVER auto-assigned
  today**: no code path in ``jevx.conversion`` rewrites architecture, so a
  report bearing this mode could only be built manually by a caller that
  performed such a rewrite.

Equivalence (:class:`EquivalenceKind`) — what was *measured*, never assumed:

* ``FUNCTIONAL`` — the wrapped backbone computes the same function as the
  source model on identical inputs (measured by
  :func:`jevx.conversion.equivalence.check_base_equivalence` /
  ``check_functional_equivalence`` on sample inputs).
* ``BEHAVIORAL`` — full decision outputs agree between two systems (e.g.
  pre/post save-reload, or source-task-head vs JevX decision outputs on the
  same probes). :func:`jevx.conversion.equivalence.check_decision_roundtrip`
  measures same-call determinism, which is the local prerequisite;
  cross-system behavioral equivalence must be measured explicitly and is
  never claimed by conversion alone.
* ``NONE`` — nothing was measured (or what was measured did not pass).

**Honesty invariant** (enforced in code, asserted in tests — a report can
never claim equivalence it did not measure):

1. a check entry with ``passed=True`` must carry a measured ``value`` and a
   ``tolerance`` it was judged against (``add_check`` raises otherwise);
2. a skipped check must carry a ``reason`` and has ``passed=None`` —
   *skipped is not passed*;
3. ``equivalence != NONE`` requires at least one passing, measured check
   (``claim_equivalence`` / ``__post_init__`` raise otherwise);
4. ``mode`` in {``WEIGHT_TRANSPLANT``, ``HEAD_ATTACHMENT``} requires
   non-empty ``mapped_params`` — never claim a transplant that mapped
   nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["ConversionMode", "ConversionReport", "EquivalenceKind"]


class ConversionMode(str, Enum):
    """What kind of operation the conversion actually was (see module doc)."""

    REPRESENTATION_REUSE = "representation_reuse"
    WEIGHT_TRANSPLANT = "weight_transplant"
    ARCHITECTURAL_CONVERSION = "architectural_conversion"
    HEAD_ATTACHMENT = "head_attachment"


class EquivalenceKind(str, Enum):
    """Which equivalence claim was *measured* (see module doc)."""

    FUNCTIONAL = "functional"
    BEHAVIORAL = "behavioral"
    NONE = "none"


def _jevx_version() -> str:
    """JevX version, imported lazily to avoid an import cycle at package init."""
    from jevx import __version__  # noqa: PLC0415 - deliberate lazy import

    return __version__


@dataclass
class ConversionReport:
    """Machine-readable record of one conversion (spec §63/§70/§71 semantics).

    Args:
        mode: Which operation was performed — ``REPRESENTATION_REUSE``
            (backbone verbatim, heads fresh), ``WEIGHT_TRANSPLANT`` (subset
            of target parameters overwritten), ``HEAD_ATTACHMENT`` (target
            fully source-parameterized) or ``ARCHITECTURAL_CONVERSION``
            (reserved; never auto-assigned). A transplant/attachment mode
            with no mapped parameters is rejected (honesty invariant 4).
        equivalence: What was measured — ``FUNCTIONAL`` /
            ``BEHAVIORAL`` / ``NONE``. Claiming a non-``NONE`` kind
            requires at least one passing check with a measured value
            (honesty invariants 1–3).
        source: Source model id or class this conversion started from.
        target: Target JevX class (normally ``"JevXForDecision"``).
        mapped_params: ``source key -> target key`` for every tensor that
            was transplanted. Keys on the target side are relative to the
            module that was written (the converter prefixes the enclosing
            ``heads.<name>.`` path when it transplants into a wrapper).
        unmapped_source: Source parameter keys that found NO counterpart —
            an honest inventory of what was *not* converted.
        new_params: Target parameter keys with no source counterpart —
            freshly initialized JevX-side modules.
        checks: Measurement records ``{name, value, tolerance, passed}``
            with an optional ``reason`` for skipped checks
            (``passed=None``). ``value is None`` only ever appears with
            ``passed=None`` + ``reason`` (honesty invariant 1).
        warnings: Human-readable caveats (transform formulas applied,
            fallback ladders taken, things that were NOT claimed).
        jevx_version: ``jevx.__version__`` at report creation (defaults
            lazily to the installed package version).

    Raises:
        ValueError: An honesty invariant is violated (see module docstring).
    """

    mode: ConversionMode
    equivalence: EquivalenceKind = EquivalenceKind.NONE
    source: str = ""
    target: str = "JevXForDecision"
    mapped_params: Dict[str, str] = field(default_factory=dict)
    unmapped_source: List[str] = field(default_factory=list)
    new_params: List[str] = field(default_factory=list)
    checks: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    jevx_version: str = ""

    def __post_init__(self) -> None:
        # Accept the plain-string spellings too (JSON round trips, callers).
        if not isinstance(self.mode, ConversionMode):
            self.mode = ConversionMode(self.mode)
        if not isinstance(self.equivalence, EquivalenceKind):
            self.equivalence = EquivalenceKind(self.equivalence)
        if not self.jevx_version:
            self.jevx_version = _jevx_version()
        if not isinstance(self.mapped_params, dict):
            raise TypeError(
                f"mapped_params: expected dict[str, str], got "
                f"{type(self.mapped_params).__name__}."
            )
        if self.mode in (
            ConversionMode.WEIGHT_TRANSPLANT,
            ConversionMode.HEAD_ATTACHMENT,
        ) and not self.mapped_params:
            raise ValueError(
                f"mode={self.mode.value!r} but mapped_params is empty — a report may "
                "never claim a transplant that mapped nothing (use "
                "REPRESENTATION_REUSE, the honest mode for 'no mapping found')."
            )
        for check in self.checks:
            self._validate_check(check)
        if self.equivalence is not EquivalenceKind.NONE and not self._passing_checks():
            raise ValueError(
                f"equivalence={self.equivalence.value!r} but no passing check with a "
                "measured value exists — a report may never claim equivalence it did "
                "not measure."
            )

    # -- honesty-enforcing mutators ------------------------------------------

    @staticmethod
    def _validate_check(check: Dict[str, Any]) -> None:
        """Structural honesty of one check entry (invariants 1/2)."""
        if not isinstance(check, dict):
            raise TypeError(
                f"checks: every entry must be a dict, got {type(check).__name__}."
            )
        missing = [k for k in ("name", "value", "tolerance", "passed") if k not in check]
        if missing:
            raise ValueError(f"check entry {check.get('name', check)!r}: missing {missing}.")
        passed = check["passed"]
        if passed is not None and not isinstance(passed, bool):
            raise TypeError(
                f"check {check['name']!r}: passed must be bool or None, got "
                f"{type(passed).__name__}."
            )
        if passed is True and (check["value"] is None or check["tolerance"] is None):
            raise ValueError(
                f"check {check['name']!r}: passed=True requires a measured value and a "
                "tolerance — a report may never claim a pass it did not measure."
            )
        if passed is None and not check.get("reason"):
            raise ValueError(
                f"check {check['name']!r}: skipped checks (passed=None) must state a "
                "reason."
            )

    def add_check(
        self,
        name: str,
        value: Optional[float],
        tolerance: Optional[float],
        passed: Optional[bool],
        *,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record one measurement (or one explicit skip) in ``checks``.

        Contract (honey invariant 1/2): a boolean verdict must be backed by
        a measured ``value`` and the ``tolerance``/expectation it was judged
        against; ``passed=None`` (skipped) must carry a ``reason``. Skipped
        is never passed.

        Returns:
            The stored check dict.
        """
        if not isinstance(name, str) or not name:
            raise TypeError(f"check name must be a non-empty str, got {name!r}.")
        if passed is not None and not isinstance(passed, bool):
            raise TypeError(
                f"check {name!r}: passed must be bool or None, got "
                f"{type(passed).__name__}."
            )
        if passed is True:
            if value is None or tolerance is None:
                raise ValueError(
                    f"check {name!r}: passed=True requires a measured value and a "
                    "tolerance — a report may never claim a pass it did not measure "
                    "(record an explicit skip with passed=None and a reason instead)."
                )
        if passed is None and not reason:
            raise ValueError(
                f"check {name!r}: skipped checks (passed=None) must state a reason."
            )
        entry: Dict[str, Any] = {
            "name": name,
            "value": None if value is None else float(value),
            "tolerance": None if tolerance is None else float(tolerance),
            "passed": passed,
        }
        if reason is not None:
            entry["reason"] = str(reason)
        self.checks.append(entry)
        return entry

    def claim_equivalence(self, kind: EquivalenceKind) -> "ConversionReport":
        """Set ``equivalence`` to a measured kind.

        Raises:
            ValueError: ``kind`` is ``FUNCTIONAL``/``BEHAVIORAL`` but no
                passing measured check exists — claims require evidence.
        """
        if not isinstance(kind, EquivalenceKind):
            kind = EquivalenceKind(kind)
        if kind is EquivalenceKind.NONE:
            self.equivalence = EquivalenceKind.NONE
            return self
        if not self._passing_checks():
            raise ValueError(
                f"cannot claim equivalence={kind.value!r}: no passing check with a "
                "measured value has been recorded. Equivalence is earned by "
                "measurement, never by assertion."
            )
        self.equivalence = kind
        return self

    def _passing_checks(self) -> List[Dict[str, Any]]:
        return [
            c
            for c in self.checks
            if c.get("passed") is True and c.get("value") is not None
        ]

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict; :meth:`load` of this dict reproduces the report."""
        return {
            "mode": self.mode.value,
            "equivalence": self.equivalence.value,
            "source": self.source,
            "target": self.target,
            "mapped_params": dict(self.mapped_params),
            "unmapped_source": list(self.unmapped_source),
            "new_params": list(self.new_params),
            "checks": [dict(c) for c in self.checks],
            "warnings": list(self.warnings),
            "jevx_version": self.jevx_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversionReport":
        """Rebuild a report from :meth:`to_dict` output (validates invariants)."""
        if not isinstance(data, dict):
            raise TypeError(f"from_dict: expected a dict, got {type(data).__name__}.")
        known = {
            "mode",
            "equivalence",
            "source",
            "target",
            "mapped_params",
            "unmapped_source",
            "new_params",
            "checks",
            "warnings",
            "jevx_version",
        }
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"from_dict: unknown report fields {unknown}.")
        return cls(
            mode=ConversionMode(data["mode"]),
            equivalence=EquivalenceKind(data.get("equivalence", "none")),
            source=data.get("source", ""),
            target=data.get("target", "JevXForDecision"),
            mapped_params=dict(data.get("mapped_params") or {}),
            unmapped_source=list(data.get("unmapped_source") or []),
            new_params=list(data.get("new_params") or []),
            checks=[dict(c) for c in (data.get("checks") or [])],
            warnings=list(data.get("warnings") or []),
            jevx_version=data.get("jevx_version", ""),
        )

    def save(self, path: Union[str, Path]) -> Path:
        """Write the report as JSON to ``path``; returns the resolved path."""
        target = Path(path)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ConversionReport":
        """Read a report written by :meth:`save` (round-trips exactly)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- human summary --------------------------------------------------------

    def summary(self) -> str:
        """One-line human summary (never overstates: skips and non-mapping show)."""
        passed = len(self._passing_checks())
        failed = len([c for c in self.checks if c.get("passed") is False])
        skipped = len([c for c in self.checks if c.get("passed") is None])
        if self.mapped_params:
            transplant = f"transplanted {len(self.mapped_params)} tensor(s)"
        else:
            transplant = "no transplant (0 tensors mapped)"
        return (
            f"{self.mode.value}: {self.source or '<unknown>'} -> {self.target}; "
            f"equivalence={self.equivalence.value} "
            f"({passed} passed, {failed} failed, {skipped} skipped); "
            f"{transplant}; {len(self.unmapped_source)} unmapped source, "
            f"{len(self.new_params)} new target params; {len(self.warnings)} warning(s)"
        )
