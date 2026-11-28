"""Post-calculation constraint profiles over immutable RAW records.

The RAW dataset stores calculable points regardless of engineering acceptance.
This module applies acceptance criteria **after** calculation and returns a
separate annotation object.  It never filters records, never mutates a
:class:`~turbine_engine.raw.RawCandidate`, and never calls an evaluator.

A constraint rule contains an explicit access path, optional minimum/maximum,
and an optional normalisation scale.  Missing, null, non-numeric, and non-finite
actual values are ``NOT_CHECKED``, never a pass.  A profile such as
``RHO_LE_0_65_OPT_IN`` is available only when explicitly requested; it is not a
search bound and is not applied automatically.

The built-in helpers intentionally contain no BOOK/CURRENT/RELAXED engineering
numbers.  The project owner supplies those profiles as explicit metadata.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from turbine_engine.raw import (
    CalculationStatus,
    ConstraintStatus,
    RawCandidate,
    copy_json_value,
)


_MISSING = object()


def _finite_float(name: str, value: Any, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ConstraintRule:
    """One lower/upper acceptance rule.

    ``path`` is a dotted string or a tuple of mapping keys (and list indexes).
    ``scale`` defaults to the largest absolute bound (or 1.0 when only zero
    bounds are present); normalised violation is raw violation divided by that
    positive scale.  Supplies ``scale`` explicitly when a different
    normalisation is desired.
    """

    path: str | Sequence[str | int]
    name: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            parts: tuple[str | int, ...] = tuple(self.path.split("."))
        elif isinstance(self.path, Sequence) and not isinstance(
            self.path, (str, bytes, bytearray)
        ):
            parts = tuple(self.path)
        else:
            raise ValueError("path must be a dotted string or a sequence")
        if not parts or any(
            (type(part) is not str or not part) and type(part) is not int
            for part in parts
        ):
            raise ValueError("path must contain non-empty string or integer keys")
        object.__setattr__(self, "path", parts)

        name = self.name
        if name is None:
            name = ".".join(str(part) for part in parts)
        if type(name) is not str or not name:
            raise ValueError("rule name must be a non-empty string")
        object.__setattr__(self, "name", name)

        minimum = _finite_float("minimum", self.minimum, allow_none=True)
        maximum = _finite_float("maximum", self.maximum, allow_none=True)
        if minimum is None and maximum is None:
            raise ValueError("a constraint rule needs minimum and/or maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("minimum must be <= maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

        scale = _finite_float("scale", self.scale, allow_none=True)
        if scale is not None and scale <= 0.0:
            raise ValueError("scale must be > 0")
        object.__setattr__(self, "scale", scale)

    @property
    def effective_scale(self) -> float:
        """Positive normalisation scale used for the violation."""

        if self.scale is not None:
            return self.scale
        bounds = [
            abs(value)
            for value in (self.minimum, self.maximum)
            if value is not None
        ]
        largest = max(bounds) if bounds else 0.0
        return largest if largest > 0.0 else 1.0

    def to_dict(self) -> dict[str, Any]:
        """Fresh plain-dict representation of the rule."""

        return {
            "name": self.name,
            "path": list(self.path),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "scale": self.effective_scale,
        }


@dataclass(frozen=True)
class ConstraintProfile:
    """A named, ordered set of acceptance rules."""

    name: str
    rules: tuple[ConstraintRule, ...]

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("profile name must be a non-empty string")
        rules = tuple(self.rules)
        if not rules:
            raise ValueError("a constraint profile needs at least one rule")
        if any(not isinstance(rule, ConstraintRule) for rule in rules):
            raise TypeError("rules must be ConstraintRule instances")
        names = [rule.name for rule in rules]
        if len(set(names)) != len(names):
            raise ValueError("constraint rule names must be unique within a profile")
        object.__setattr__(self, "rules", rules)

    @classmethod
    def from_mapping(
        cls, name: str, rules: Mapping[str, Mapping[str, Any]]
    ) -> "ConstraintProfile":
        """Build a profile from ``{rule_name: rule_kwargs}``.

        ``rule_kwargs`` must contain ``path`` and at least one bound.  The
        mapping key becomes the rule name unless an explicit ``name`` is given.
        """

        built: list[ConstraintRule] = []
        for rule_name, specification in rules.items():
            if not isinstance(specification, Mapping):
                raise TypeError(f"rule {rule_name!r} specification must be a mapping")
            values = dict(specification)
            values.setdefault("name", rule_name)
            built.append(ConstraintRule(**values))
        return cls(name=name, rules=tuple(built))

    def to_dict(self) -> dict[str, Any]:
        """Fresh plain-dict representation of the profile."""

        return {
            "name": self.name,
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(frozen=True)
class ConstraintCheck:
    """One evaluated acceptance rule."""

    name: str
    path: tuple[str | int, ...]
    actual: Any
    minimum: float | None
    maximum: float | None
    lower_margin: float | None
    upper_margin: float | None
    normalized_violation: float | None
    checked: bool
    passed: bool | None
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Fresh plain-dict representation of the check."""

        return {
            "name": self.name,
            "path": list(self.path),
            "actual": copy_json_value(self.actual, path="$.actual"),
            "min": self.minimum,
            "max": self.maximum,
            "lower_margin": self.lower_margin,
            "upper_margin": self.upper_margin,
            "normalized_violation": self.normalized_violation,
            "checked": self.checked,
            "pass": self.passed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AnnotatedCandidate:
    """A RAW record plus post-calculation constraint annotations."""

    raw: RawCandidate
    profile_name: str
    constraint_status: ConstraintStatus
    feasible: bool | None
    checks: tuple[ConstraintCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw, RawCandidate):
            raise TypeError("raw must be a RawCandidate")
        try:
            status = ConstraintStatus(self.constraint_status)
        except ValueError as exc:
            raise ValueError(f"unknown constraint status: {exc}") from exc
        object.__setattr__(self, "constraint_status", status)
        if self.feasible is not None and type(self.feasible) is not bool:
            raise ValueError("feasible must be True, False, or None")
        if status is ConstraintStatus.PASS and self.feasible is not True:
            raise ValueError("PASS requires feasible=True")
        if status is ConstraintStatus.FAIL and self.feasible is not False:
            raise ValueError("FAIL requires feasible=False")
        if status is ConstraintStatus.NOT_CHECKED and self.feasible is not None:
            raise ValueError("NOT_CHECKED requires feasible=None")
        object.__setattr__(self, "checks", tuple(self.checks))

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Return annotations; include the untouched RAW record only on request."""

        result: dict[str, Any] = {
            "candidate_id": self.raw.candidate_id,
            "run_id": self.raw.run_id,
            "record_hash": self.raw.record_hash,
            "calculation_status": self.raw.calculation_status.value,
            "constraint_profile": self.profile_name,
            "constraint_status": self.constraint_status.value,
            "feasible": self.feasible,
            "checks": [check.to_dict() for check in self.checks],
        }
        if include_raw:
            result["raw"] = self.raw.to_dict()
        return result


def _resolve_path(value: Any, path: Sequence[str | int]) -> Any:
    current = value
    for part in path:
        if type(part) is int:
            if not isinstance(current, list) or part < 0 or part >= len(current):
                return _MISSING
            current = current[part]
        else:
            if not isinstance(current, Mapping) or part not in current:
                return _MISSING
            current = current[part]
    return current


def _check_rule(raw: RawCandidate, rule: ConstraintRule) -> ConstraintCheck:
    value = _resolve_path(raw.to_dict(), rule.path)
    if value is _MISSING:
        return ConstraintCheck(
            name=rule.name,
            path=rule.path,
            actual=None,
            minimum=rule.minimum,
            maximum=rule.maximum,
            lower_margin=None,
            upper_margin=None,
            normalized_violation=None,
            checked=False,
            passed=None,
            reason="missing",
        )
    if value is None:
        return ConstraintCheck(
            name=rule.name,
            path=rule.path,
            actual=None,
            minimum=rule.minimum,
            maximum=rule.maximum,
            lower_margin=None,
            upper_margin=None,
            normalized_violation=None,
            checked=False,
            passed=None,
            reason="null",
        )

    if type(value) is bool or not isinstance(value, (int, float)):
        actual_value = value
        try:
            actual_value = copy_json_value(value, path=f"$.{rule.name}")
        except ValueError:
            actual_value = None
        return ConstraintCheck(
            name=rule.name,
            path=rule.path,
            actual=actual_value,
            minimum=rule.minimum,
            maximum=rule.maximum,
            lower_margin=None,
            upper_margin=None,
            normalized_violation=None,
            checked=False,
            passed=None,
            reason="non_numeric",
        )

    actual = float(value)
    if not math.isfinite(actual):
        return ConstraintCheck(
            name=rule.name,
            path=rule.path,
            actual=None,
            minimum=rule.minimum,
            maximum=rule.maximum,
            lower_margin=None,
            upper_margin=None,
            normalized_violation=None,
            checked=False,
            passed=None,
            reason="non_finite",
        )

    lower_margin = (
        actual - rule.minimum if rule.minimum is not None else None
    )
    upper_margin = (
        rule.maximum - actual if rule.maximum is not None else None
    )
    lower_violation = (
        max(0.0, (rule.minimum - actual) / rule.effective_scale)
        if rule.minimum is not None
        else 0.0
    )
    upper_violation = (
        max(0.0, (actual - rule.maximum) / rule.effective_scale)
        if rule.maximum is not None
        else 0.0
    )
    normalized_violation = max(lower_violation, upper_violation)
    passed = (rule.minimum is None or actual >= rule.minimum) and (
        rule.maximum is None or actual <= rule.maximum
    )
    return ConstraintCheck(
        name=rule.name,
        path=rule.path,
        actual=actual,
        minimum=rule.minimum,
        maximum=rule.maximum,
        lower_margin=lower_margin,
        upper_margin=upper_margin,
        normalized_violation=normalized_violation,
        checked=True,
        passed=passed,
        reason=None,
    )


def annotate_candidate(
    candidate: RawCandidate, profile: ConstraintProfile
) -> AnnotatedCandidate:
    """Apply ``profile`` to one RAW record without mutation or recalculation.

    If the calculation did not succeed, the overall constraint status is
    ``NOT_CHECKED`` even if a partial output happens to satisfy a rule.  For a
    successful calculation, any checked failure yields ``FAIL``; otherwise any
    unchecked rule yields ``NOT_CHECKED``; otherwise the result is ``PASS``.
    """

    if not isinstance(candidate, RawCandidate):
        raise TypeError("candidate must be a RawCandidate")
    if not isinstance(profile, ConstraintProfile):
        raise TypeError("profile must be a ConstraintProfile")

    checks = tuple(_check_rule(candidate, rule) for rule in profile.rules)
    if candidate.calculation_status is not CalculationStatus.SUCCESS:
        status = ConstraintStatus.NOT_CHECKED
        feasible: bool | None = None
    elif any(check.checked and check.passed is False for check in checks):
        status = ConstraintStatus.FAIL
        feasible = False
    elif any(not check.checked for check in checks):
        status = ConstraintStatus.NOT_CHECKED
        feasible = None
    else:
        status = ConstraintStatus.PASS
        feasible = True
    return AnnotatedCandidate(
        raw=candidate,
        profile_name=profile.name,
        constraint_status=status,
        feasible=feasible,
        checks=checks,
    )


def annotate_records(
    records: Iterable[RawCandidate], profile: ConstraintProfile
) -> tuple[AnnotatedCandidate, ...]:
    """Annotate every record in order; no record is filtered or dropped."""

    if not isinstance(profile, ConstraintProfile):
        raise TypeError("profile must be a ConstraintProfile")
    return tuple(annotate_candidate(record, profile) for record in records)


RHO_LE_0_65_PROFILE_NAME = "RHO_LE_0_65_OPT_IN"


def rho_065_profile(*, path: str | Sequence[str | int] = "outputs.rho") -> ConstraintProfile:
    """Return the opt-in ``rho <= 0.65`` annotation profile.

    This profile is never applied automatically and is never a search bound.
    Callers may pass a different explicit ``path`` if their evaluator stores
    the reaction ratio elsewhere.
    """

    return ConstraintProfile(
        name=RHO_LE_0_65_PROFILE_NAME,
        rules=(
            ConstraintRule(
                path=path,
                name="rho",
                maximum=0.65,
                scale=0.65,
            ),
        ),
    )


__all__ = [
    "ConstraintRule",
    "ConstraintProfile",
    "ConstraintCheck",
    "AnnotatedCandidate",
    "annotate_candidate",
    "annotate_records",
    "rho_065_profile",
    "RHO_LE_0_65_PROFILE_NAME",
]
