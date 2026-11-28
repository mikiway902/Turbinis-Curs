"""Pure post-calculation analysis for annotated candidate rows.

SCOPE
=====
This module consumes **already calculated and annotated candidate rows**.  It
does not run a cycle, a sweep, a solver, a constraint engine, or a backend.  It
contains no physical model and no acceptance-criterion default: every
constraint verdict, bound, scale, objective direction, and candidate input is
explicit.  The module is deliberately independent of any raw-dataset writer so
that it can be used with the generic mapping adapter below or with
:class:`CandidateRow` values produced by another stage.

The analysis pipeline is separated into three distinct levels:

``calculation_status``
    Whether the numerical calculation itself succeeded.  A successful
    calculation says nothing about engineering acceptability.

``feasibility_status``
    Engineering screening only.  ``SCREENED_FEASIBLE`` means every supplied
    constraint is explicitly ``PASS``.  A missing or ``NOT_CHECKED``
    constraint, including an absent material/life/model criterion, yields
    ``UNVERIFIED``; it is never silently counted as a pass.  A row with no
    constraints is ``UNVERIFIED``.

``fully_verified``
    Never inferred.  It requires an explicit
    ``verification_status="VERIFIED"`` annotation **and** a successful
    calculation **and** complete screening feasibility.  This module cannot
    promote a row to fully verified on its own.

ANNOTATED ROW SCHEMA
====================
:meth:`CandidateRow.from_mapping` accepts a mapping with these required keys:

``candidate_id``
    Non-empty string identifying the candidate.  Duplicate identifiers are not
    rejected; Pareto analysis preserves duplicate equivalent candidates.
``architecture``
    Non-empty string architecture label, used for grouping and paired-input
    matching.
``calculation_status``
    Exactly ``"SUCCESS"``, ``"NUMERICAL_FAILURE"`` or
    ``"NON_PHYSICAL_STATE"`` (or the matching :class:`CalculationStatus`).
``constraints``
    Mapping from constraint name to a mapping with an explicit ``status`` of
    exactly ``"PASS"``, ``"FAIL"`` or ``"NOT_CHECKED"`` (or the matching
    :class:`ConstraintStatus`).  ``PASS``/``FAIL`` require a finite ``actual``
    and at least one finite bound (``lower`` and/or ``upper``).  The optional
    ``scale`` must be finite and strictly positive and is the only accepted
    normalization scale.  ``bounds=(lower, upper)`` is accepted as an
    alternative spelling when ``lower``/``upper`` are not also supplied.
    Margins and normalized violations are derived, not taken from the input.
    Numeric fields accept real numbers (not numeric strings or booleans) and
    reject integers that cannot be represented exactly as float, so a verdict
    cannot change through silent rounding.
``metrics``
    Mapping from metric name to a real number or ``None``.  Values are stored
    as supplied, including non-finite values; consumers exclude non-finite or
    missing metrics instead of replacing them with zero.  The same exact-real
    conversion rule as constraints applies.

Optional keys are ``inputs`` (JSON-style mapping used by controlled sensitivity;
default empty), ``pi_compressor`` (finite number or ``None``), ``model_variant``
(non-empty string or ``None``), ``verification_status`` (exactly
``"NOT_VERIFIED"`` or ``"VERIFIED"``; default ``"NOT_VERIFIED"``),
``failure_stage`` and ``failure_reason`` (non-empty strings or ``None``).
``REQUIRED_ROW_KEYS`` and ``OPTIONAL_ROW_KEYS`` expose this contract.  The
analysis functions additionally accept an optional ``required_constraints``
profile.  A constraint named in that profile but absent from a row is treated
as unknown, never as a pass; the row is excluded from the default Pareto front
and left unranked in near-feasibility.  This is how a caller declares that,
for example, a material/life criterion applies even when an upstream annotator
omitted it.

FEASIBILITY ANNEXATION
======================
For each constraint the annotation stores ``actual``, ``lower``, ``upper``,
``lower_margin = actual - lower``, ``upper_margin = upper - actual``, and a
nonnegative ``violation``.  A normalized violation is available only when an
explicit positive ``scale`` was supplied.  There is no implicit unit scale, no
bound-width normalization, and no fabricated tolerance.  ``NOT_CHECKED``
constraints have a ``None`` violation even if values are present; unknown
criteria therefore cannot contribute a zero violation silently.  A derived
margin that overflows is rejected explicitly, and a normalized violation that
overflows or underflows to zero is returned as ``None`` and left unranked
rather than being treated as a valid zero or infinity.

PARETO
======
:func:`pareto_front` implements pairwise nondominance over explicit
:class:`Objective` names and ``MINIMIZE``/``MAXIMIZE`` directions.  Missing or
non-finite objective values exclude a row; they are never zero-filled.  By
default only rows with ``SUCCESS`` calculation status and complete screening
feasibility are considered; engineering-infeasible, unchecked, and
calculation-failed rows can be included only through explicit flags.  No
weighted sum, score, or hidden objective transformation is applied.

NEAR-FEASIBILITY
================
:func:`rank_near_feasible` returns the full per-constraint violation vector and
the normalized vector.  Ranking uses only explicitly normalized violations and
is a deterministic lexicographic worst-violation ordering; it is not a hidden
weighted score.  Rows with unknown constraints or a violated constraint lacking
an explicit scale are returned unranked with a reason, not ranked as if the
unknown contribution were zero.

CONTROLLED SENSITIVITY
======================
:func:`paired_sensitivity` compares exactly two rows with identical
architecture and model variant whose input mappings have identical keys and
identical values for every input except the explicitly named parameter.  No
tolerance is applied.  A zero input difference is refused.  A zero baseline
metric yields ``relative_change=None``, never infinity.  This is a paired local
finite difference, not a causal claim about arbitrary sweep correlations.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping


#: Keys that :meth:`CandidateRow.from_mapping` requires.
REQUIRED_ROW_KEYS = frozenset(
    {"candidate_id", "architecture", "calculation_status", "constraints", "metrics"}
)

#: Keys that :meth:`CandidateRow.from_mapping` accepts in addition.
OPTIONAL_ROW_KEYS = frozenset(
    {
        "inputs",
        "pi_compressor",
        "model_variant",
        "verification_status",
        "failure_stage",
        "failure_reason",
    }
)


class AnalysisError(ValueError):
    """Raised when analysis input is missing, malformed, or inconsistent."""


class ConstraintError(AnalysisError):
    """Raised for an invalid or contradictory constraint annotation."""


class SensitivityError(AnalysisError):
    """Raised when two rows cannot form a controlled paired comparison."""


class CalculationStatus(str, Enum):
    """Explicit numerical-calculation outcome."""

    SUCCESS = "SUCCESS"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    NON_PHYSICAL_STATE = "NON_PHYSICAL_STATE"


class ConstraintStatus(str, Enum):
    """Explicit engineering-constraint verdict."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_CHECKED = "NOT_CHECKED"


class FeasibilityStatus(str, Enum):
    """Engineering screening level; never a physical validation claim."""

    CALCULATION_FAILED = "CALCULATION_FAILED"
    ENGINEERING_INFEASIBLE = "ENGINEERING_INFEASIBLE"
    UNVERIFIED = "UNVERIFIED"
    SCREENED_FEASIBLE = "SCREENED_FEASIBLE"


class VerificationStatus(str, Enum):
    """Explicit external verification marker."""

    NOT_VERIFIED = "NOT_VERIFIED"
    VERIFIED = "VERIFIED"


class Direction(str, Enum):
    """Objective direction for Pareto dominance."""

    MINIMIZE = "MINIMIZE"
    MAXIMIZE = "MAXIMIZE"


def _real_to_float(
    name: str, value: Any, *, allow_nonfinite: bool
) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise AnalysisError(f"{name} must be a real number, got {value!r}")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AnalysisError(
            f"{name} cannot be represented as float, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        if allow_nonfinite:
            return result
        raise AnalysisError(f"{name} must be finite, got {result!r}")
    if result != value:
        raise AnalysisError(
            f"{name} cannot be represented exactly as float, got {value!r}; "
            "use a value that is exactly representable"
        )
    return result


def _finite_float(name: str, value: Any) -> float:
    return _real_to_float(name, value, allow_nonfinite=False)


def _optional_finite_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _finite_float(name, value)


def _metric_float(name: str, value: Any) -> float | None:
    if value is None:
        return None
    return _real_to_float(name, value, allow_nonfinite=True)


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise AnalysisError(f"{name} must be a bool, got {value!r}")
    return value


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnalysisError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _optional_nonempty_string(name: str, value: Any) -> str | None:
    if value is None:
        return None
    return _nonempty_string(name, value)


def _constraint_status(name: str, value: Any) -> ConstraintStatus:
    if isinstance(value, ConstraintStatus):
        return value
    if isinstance(value, str):
        try:
            return ConstraintStatus(value)
        except ValueError:
            pass
    raise ConstraintError(
        f"{name} must be exactly one of "
        f"{[member.value for member in ConstraintStatus]}, got {value!r}"
    )


def _calculation_status(value: Any) -> CalculationStatus:
    if isinstance(value, CalculationStatus):
        return value
    if isinstance(value, str):
        try:
            return CalculationStatus(value)
        except ValueError:
            pass
    raise AnalysisError(
        "calculation_status must be exactly one of "
        f"{[member.value for member in CalculationStatus]}, got {value!r}"
    )


def _verification_status(value: Any) -> VerificationStatus:
    if isinstance(value, VerificationStatus):
        return value
    if isinstance(value, str):
        try:
            return VerificationStatus(value)
        except ValueError:
            pass
    raise AnalysisError(
        "verification_status must be exactly one of "
        f"{[member.value for member in VerificationStatus]}, got {value!r}"
    )


def _freeze_value(value: Any, *, path: str = "value") -> Any:
    """Freeze a JSON-style input value without aliasing opaque objects.

    Only ``None``, booleans, integers, finite floats, strings, sequences, and
    string-keyed mappings are accepted.  Bytes, bytearrays, sets, NumPy arrays,
    and arbitrary mutable objects are rejected explicitly rather than being
    stored as aliases.
    """

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AnalysisError(f"{path} must be a finite float, got {value!r}")
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            key = _nonempty_string(f"{path} key", key)
            frozen[key] = _freeze_value(item, path=f"{path}[{key!r}]")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise AnalysisError(
        f"{path} has unsupported type {type(value).__name__}; inputs support "
        "only JSON-style scalars, sequences, and string-keyed mappings"
    )


def _freeze_mapping(
    name: str, value: Any, *, freeze_values: bool = False
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisError(f"{name} must be a mapping, got {value!r}")
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        key = _nonempty_string(f"{name} key", key)
        frozen[key] = _freeze_value(item) if freeze_values else item
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ConstraintEvaluation:
    """Immutable annotated constraint with derived margins and violation.

    ``status`` is authoritative and must agree with the supplied numeric
    bounds.  ``PASS`` means every supplied bound is satisfied; ``FAIL`` means at
    least one supplied bound is violated; ``NOT_CHECKED`` makes no verdict.
    ``scale`` is the explicit denominator for a normalized violation and is
    never inferred from the bound width or from the actual value.
    """

    name: str
    status: ConstraintStatus
    actual: float | None = None
    lower: float | None = None
    upper: float | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        name = _nonempty_string("constraint name", self.name)
        status = _constraint_status(f"{name}.status", self.status)
        actual = _optional_finite_float(f"{name}.actual", self.actual)
        lower = _optional_finite_float(f"{name}.lower", self.lower)
        upper = _optional_finite_float(f"{name}.upper", self.upper)
        scale = _optional_finite_float(f"{name}.scale", self.scale)

        if lower is not None and upper is not None and lower > upper:
            raise ConstraintError(
                f"{name} has lower bound {lower!r} greater than upper bound "
                f"{upper!r}"
            )
        if scale is not None and scale <= 0.0:
            raise ConstraintError(
                f"{name}.scale must be > 0, got {scale!r}"
            )

        # Two finite operands can still overflow on subtraction.  Reject that
        # explicitly rather than exposing an infinite margin/violation.
        for label, margin in (
            (
                "lower_margin",
                actual - lower
                if actual is not None and lower is not None
                else None,
            ),
            (
                "upper_margin",
                upper - actual
                if actual is not None and upper is not None
                else None,
            ),
        ):
            if margin is not None and not math.isfinite(margin):
                raise ConstraintError(
                    f"{name}.{label} is non-finite: actual={actual!r}, "
                    f"lower={lower!r}, upper={upper!r}"
                )

        if status in (ConstraintStatus.PASS, ConstraintStatus.FAIL):
            if actual is None:
                raise ConstraintError(
                    f"{name} is {status.value} but has no finite actual value"
                )
            if lower is None and upper is None:
                raise ConstraintError(
                    f"{name} is {status.value} but has no finite bound"
                )
            violated = (
                (lower is not None and actual < lower)
                or (upper is not None and actual > upper)
            )
            if status is ConstraintStatus.PASS and violated:
                raise ConstraintError(
                    f"{name} is marked PASS but actual={actual!r} violates "
                    f"lower={lower!r}, upper={upper!r}"
                )
            if status is ConstraintStatus.FAIL and not violated:
                raise ConstraintError(
                    f"{name} is marked FAIL but actual={actual!r} satisfies "
                    f"lower={lower!r}, upper={upper!r}"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "scale", scale)

    @property
    def lower_margin(self) -> float | None:
        """``actual - lower`` when both are known."""

        if self.actual is None or self.lower is None:
            return None
        return self.actual - self.lower

    @property
    def upper_margin(self) -> float | None:
        """``upper - actual`` when both are known."""

        if self.actual is None or self.upper is None:
            return None
        return self.upper - self.actual

    @property
    def violation(self) -> float | None:
        """Nonnegative bound violation, or ``None`` for an unknown verdict.

        A ``NOT_CHECKED`` constraint has no violation even when actual and
        bounds happen to be present, because the annotation made no verdict.
        """

        if self.status is ConstraintStatus.NOT_CHECKED:
            return None
        lower_violation = (
            self.lower - self.actual
            if self.lower is not None and self.actual is not None
            else 0.0
        )
        upper_violation = (
            self.actual - self.upper
            if self.upper is not None and self.actual is not None
            else 0.0
        )
        return max(lower_violation, upper_violation, 0.0)

    @property
    def normalized_violation(self) -> float | None:
        """``violation / scale`` using only the explicit scale.

        Zero violation needs no scale and is returned as ``0.0``.  A positive
        violation without an explicit scale returns ``None`` rather than an
        invented normalization.  A quotient that overflows or underflows to
        zero is also returned as ``None`` so near-feasibility cannot silently
        treat it as zero violation.
        """

        violation = self.violation
        if violation is None:
            return None
        if violation == 0.0:
            return 0.0
        if self.scale is None:
            return None
        result = violation / self.scale
        if not math.isfinite(result) or result == 0.0:
            return None
        return result


def _constraint_from_mapping(
    name: str, value: Any
) -> ConstraintEvaluation:
    if isinstance(value, ConstraintEvaluation):
        if value.name != name:
            raise ConstraintError(
                f"constraint key {name!r} does not match "
                f"ConstraintEvaluation.name {value.name!r}"
            )
        return value
    if not isinstance(value, Mapping):
        raise ConstraintError(
            f"constraints[{name!r}] must be a mapping or "
            f"ConstraintEvaluation, got {value!r}"
        )
    if "status" not in value:
        raise ConstraintError(
            f"constraints[{name!r}] is missing the explicit 'status' key"
        )

    lower = value.get("lower")
    upper = value.get("upper")
    if "bounds" in value:
        if lower is not None or upper is not None:
            raise ConstraintError(
                f"constraints[{name!r}] supplies both 'bounds' and "
                "'lower'/'upper'"
            )
        bounds = value["bounds"]
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ConstraintError(
                f"constraints[{name!r}].bounds must be a two-item sequence"
            )
        lower, upper = bounds

    return ConstraintEvaluation(
        name=name,
        status=value["status"],
        actual=value.get("actual"),
        lower=lower,
        upper=upper,
        scale=value.get("scale"),
    )


@dataclass(frozen=True)
class CandidateRow:
    """Immutable normalized annotated candidate row.

    Use :meth:`from_mapping` for generic row mappings.  Direct construction
    requires already normalized :class:`ConstraintEvaluation` values; the
    constructor validates the remaining fields.  All mappings are copied into
    read-only proxies and input values are recursively frozen where practical.
    """

    candidate_id: str
    architecture: str
    calculation_status: CalculationStatus
    constraints: Mapping[str, ConstraintEvaluation]
    metrics: Mapping[str, float | None]
    inputs: Mapping[str, Any] = field(default_factory=dict)
    pi_compressor: float | None = None
    model_variant: str | None = None
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    failure_stage: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        candidate_id = _nonempty_string("candidate_id", self.candidate_id)
        architecture = _nonempty_string("architecture", self.architecture)
        calculation_status = _calculation_status(self.calculation_status)

        if not isinstance(self.constraints, Mapping):
            raise AnalysisError("constraints must be a mapping")
        constraints: dict[str, ConstraintEvaluation] = {}
        for key, value in self.constraints.items():
            name = _nonempty_string("constraint key", key)
            if not isinstance(value, ConstraintEvaluation):
                raise AnalysisError(
                    f"constraints[{name!r}] must be ConstraintEvaluation; "
                    "use CandidateRow.from_mapping for raw mappings"
                )
            if value.name != name:
                raise ConstraintError(
                    f"constraint key {name!r} does not match "
                    f"ConstraintEvaluation.name {value.name!r}"
                )
            constraints[name] = value
        constraints = dict(sorted(constraints.items()))

        if not isinstance(self.metrics, Mapping):
            raise AnalysisError("metrics must be a mapping")
        metrics: dict[str, float | None] = {}
        for key, value in self.metrics.items():
            name = _nonempty_string("metric key", key)
            metrics[name] = _metric_float(f"metrics[{name!r}]", value)

        if not isinstance(self.inputs, Mapping):
            raise AnalysisError("inputs must be a mapping")
        inputs = _freeze_mapping("inputs", self.inputs, freeze_values=True)

        pi_compressor = _optional_finite_float(
            "pi_compressor", self.pi_compressor
        )
        model_variant = _optional_nonempty_string(
            "model_variant", self.model_variant
        )
        verification_status = _verification_status(self.verification_status)
        failure_stage = _optional_nonempty_string(
            "failure_stage", self.failure_stage
        )
        failure_reason = _optional_nonempty_string(
            "failure_reason", self.failure_reason
        )

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "calculation_status", calculation_status)
        object.__setattr__(self, "constraints", MappingProxyType(constraints))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "pi_compressor", pi_compressor)
        object.__setattr__(self, "model_variant", model_variant)
        object.__setattr__(self, "verification_status", verification_status)
        object.__setattr__(self, "failure_stage", failure_stage)
        object.__setattr__(self, "failure_reason", failure_reason)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> CandidateRow:
        """Build a row from the documented generic annotated-row mapping.

        The explicit ``status`` strings are authoritative.  No boolean field,
        truthiness rule, or default verdict is accepted, so an absent or
        unknown criterion cannot become a pass by omission.
        """

        if not isinstance(row, Mapping):
            raise AnalysisError(
                f"candidate row must be a mapping, got {row!r}"
            )
        missing = sorted(REQUIRED_ROW_KEYS.difference(row))
        if missing:
            raise AnalysisError(
                f"candidate row is missing required key(s): {missing}"
            )
        unknown = sorted(set(row).difference(REQUIRED_ROW_KEYS | OPTIONAL_ROW_KEYS))
        if unknown:
            raise AnalysisError(
                f"candidate row contains unsupported key(s): {unknown}"
            )

        raw_constraints = row["constraints"]
        if not isinstance(raw_constraints, Mapping):
            raise ConstraintError(
                f"constraints must be a mapping, got {raw_constraints!r}"
            )
        constraints: dict[str, ConstraintEvaluation] = {}
        for key, value in raw_constraints.items():
            name = _nonempty_string("constraint key", key)
            constraints[name] = _constraint_from_mapping(name, value)

        raw_metrics = row["metrics"]
        if not isinstance(raw_metrics, Mapping):
            raise AnalysisError(
                f"metrics must be a mapping, got {raw_metrics!r}"
            )

        raw_inputs = row.get("inputs", {})
        if raw_inputs is None:
            raw_inputs = {}
        if not isinstance(raw_inputs, Mapping):
            raise AnalysisError(
                f"inputs must be a mapping, got {raw_inputs!r}"
            )

        return cls(
            candidate_id=row["candidate_id"],
            architecture=row["architecture"],
            calculation_status=row["calculation_status"],
            constraints=constraints,
            metrics=raw_metrics,
            inputs=raw_inputs,
            pi_compressor=row.get("pi_compressor"),
            model_variant=row.get("model_variant"),
            verification_status=row.get(
                "verification_status", VerificationStatus.NOT_VERIFIED
            ),
            failure_stage=row.get("failure_stage"),
            failure_reason=row.get("failure_reason"),
        )

    @property
    def calculation_succeeded(self) -> bool:
        """True only for an explicit ``SUCCESS`` calculation status."""

        return self.calculation_status is CalculationStatus.SUCCESS

    @property
    def failed_constraints(self) -> tuple[str, ...]:
        """Sorted names of explicitly failed constraints."""

        return tuple(
            name
            for name, item in self.constraints.items()
            if item.status is ConstraintStatus.FAIL
        )

    @property
    def unchecked_constraints(self) -> tuple[str, ...]:
        """Sorted names of constraints explicitly marked ``NOT_CHECKED``."""

        return tuple(
            name
            for name, item in self.constraints.items()
            if item.status is ConstraintStatus.NOT_CHECKED
        )

    @property
    def screening_feasible(self) -> bool:
        """Complete engineering screening feasibility, not validation.

        Requires a successful calculation, at least one constraint, no failed
        constraint, and no unchecked constraint.  An empty constraint mapping
        is explicitly not feasible because it cannot be screened.  The row
        alone cannot know which criteria were supposed to be present; analysis
        callers can additionally pass ``required_constraints`` to treat an
        absent criterion as unknown.
        """

        return (
            self.calculation_succeeded
            and bool(self.constraints)
            and not self.failed_constraints
            and not self.unchecked_constraints
        )

    @property
    def fully_verified(self) -> bool:
        """Explicit verification plus complete screening feasibility.

        This module never promotes a row on its own.  The external
        ``verification_status`` marker must be supplied explicitly.
        """

        return (
            self.verification_status is VerificationStatus.VERIFIED
            and self.screening_feasible
        )

    @property
    def feasibility_status(self) -> FeasibilityStatus:
        """Screening classification derived from explicit annotations."""

        if not self.calculation_succeeded:
            return FeasibilityStatus.CALCULATION_FAILED
        if self.failed_constraints:
            return FeasibilityStatus.ENGINEERING_INFEASIBLE
        if not self.constraints or self.unchecked_constraints:
            return FeasibilityStatus.UNVERIFIED
        return FeasibilityStatus.SCREENED_FEASIBLE


def normalize_candidate_rows(
    rows: Iterable[Mapping[str, Any] | CandidateRow],
) -> tuple[CandidateRow, ...]:
    """Normalize generic mappings or pass through existing rows."""

    result: list[CandidateRow] = []
    for row in rows:
        if isinstance(row, CandidateRow):
            result.append(row)
        elif isinstance(row, Mapping):
            result.append(CandidateRow.from_mapping(row))
        else:
            raise AnalysisError(
                f"candidate row must be a mapping or CandidateRow, got {row!r}"
            )
    return tuple(result)


@dataclass(frozen=True)
class ConstraintStatistics:
    """Per-constraint counts with explicit denominators.

    ``denominator_rows`` is every row in the design group, including rows where
    the constraint is absent.  ``absent_rows`` is therefore not silently
    treated as checked or as a pass.  ``checked_rows`` is the pass+fail subset
    used by the checked-only fractions.
    """

    constraint: str
    denominator_rows: int
    present_rows: int
    absent_rows: int
    pass_count: int
    fail_count: int
    not_checked_count: int

    def __post_init__(self) -> None:
        name = _nonempty_string("constraint", self.constraint)
        counts = (
            self.denominator_rows,
            self.present_rows,
            self.absent_rows,
            self.pass_count,
            self.fail_count,
            self.not_checked_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise AnalysisError("constraint statistics counts must be integers")
        if any(value < 0 for value in counts):
            raise AnalysisError("constraint statistics counts must be >= 0")
        if self.present_rows != (
            self.pass_count + self.fail_count + self.not_checked_count
        ):
            raise AnalysisError(
                f"{name}: present_rows must equal pass+fail+not_checked"
            )
        if self.denominator_rows != self.present_rows + self.absent_rows:
            raise AnalysisError(
                f"{name}: denominator_rows must equal present+absent"
            )
        object.__setattr__(self, "constraint", name)

    @property
    def checked_rows(self) -> int:
        """Rows with an explicit pass/fail verdict (the checked denominator)."""

        return self.pass_count + self.fail_count

    @property
    def unchecked_or_absent_rows(self) -> int:
        """Rows whose constraint status is unknown or absent."""

        return self.not_checked_count + self.absent_rows

    @property
    def pass_fraction_of_checked(self) -> float | None:
        """``pass_count / checked_rows``, or ``None`` if nothing was checked."""

        if self.checked_rows == 0:
            return None
        return self.pass_count / self.checked_rows

    @property
    def fail_fraction_of_checked(self) -> float | None:
        """``fail_count / checked_rows``, or ``None`` if nothing was checked."""

        if self.checked_rows == 0:
            return None
        return self.fail_count / self.checked_rows


@dataclass(frozen=True)
class DesignGroupStatistics:
    """Statistics for one ``(architecture, pi_compressor)`` group.

    ``failure_reason_counts`` and ``failure_stage_counts`` use only rows whose
    calculation failed; their denominator is :attr:`calculation_failures`.
    ``feasibility_counts`` covers every row in the group.  Constraint
    statistics carry their own explicit denominators.
    """

    architecture: str
    pi_compressor: float | None
    total_rows: int
    calculation_success: int
    calculation_failures: int
    feasibility_counts: Mapping[FeasibilityStatus, int]
    failure_reason_counts: Mapping[str, int]
    failure_stage_counts: Mapping[str, int]
    constraint_statistics: Mapping[str, ConstraintStatistics]

    def __post_init__(self) -> None:
        architecture = _nonempty_string("architecture", self.architecture)
        pi_compressor = _optional_finite_float(
            "pi_compressor", self.pi_compressor
        )
        if self.total_rows != self.calculation_success + self.calculation_failures:
            raise AnalysisError(
                "total_rows must equal calculation_success + "
                "calculation_failures"
            )
        if sum(self.feasibility_counts.values()) != self.total_rows:
            raise AnalysisError(
                "feasibility_counts must sum to total_rows"
            )
        if sum(self.failure_reason_counts.values()) != self.calculation_failures:
            raise AnalysisError(
                "failure_reason_counts must sum to calculation_failures"
            )
        if sum(self.failure_stage_counts.values()) != self.calculation_failures:
            raise AnalysisError(
                "failure_stage_counts must sum to calculation_failures"
            )

        frozen_feasibility: dict[FeasibilityStatus, int] = {}
        for key, value in self.feasibility_counts.items():
            if not isinstance(key, FeasibilityStatus):
                raise AnalysisError(
                    f"feasibility count key must be FeasibilityStatus, got {key!r}"
                )
            frozen_feasibility[key] = value
        for member in FeasibilityStatus:
            frozen_feasibility.setdefault(member, 0)

        object.__setattr__(self, "architecture", architecture)
        object.__setattr__(self, "pi_compressor", pi_compressor)
        object.__setattr__(
            self, "feasibility_counts", MappingProxyType(frozen_feasibility)
        )
        object.__setattr__(
            self,
            "failure_reason_counts",
            MappingProxyType(dict(sorted(self.failure_reason_counts.items()))),
        )
        object.__setattr__(
            self,
            "failure_stage_counts",
            MappingProxyType(dict(sorted(self.failure_stage_counts.items()))),
        )
        object.__setattr__(
            self,
            "constraint_statistics",
            MappingProxyType(dict(sorted(self.constraint_statistics.items()))),
        )


def _count(items: Iterable[str]) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return MappingProxyType(dict(sorted(counts.items())))


def _profile_feasibility_status(
    row: CandidateRow, required: frozenset[str]
) -> FeasibilityStatus:
    if not row.calculation_succeeded:
        return FeasibilityStatus.CALCULATION_FAILED
    if row.failed_constraints:
        return FeasibilityStatus.ENGINEERING_INFEASIBLE
    if (
        not row.constraints
        or row.unchecked_constraints
        or required.difference(row.constraints)
    ):
        return FeasibilityStatus.UNVERIFIED
    return FeasibilityStatus.SCREENED_FEASIBLE


def summarize_design_groups(
    candidates: Iterable[Mapping[str, Any] | CandidateRow],
    *,
    required_constraints: Iterable[str] | None = None,
) -> tuple[DesignGroupStatistics, ...]:
    """Summarize rows by architecture and compressor pressure ratio.

    The statistics are descriptive only.  No group is declared feasible.  Every
    count carries an explicit denominator through the group fields and
    :class:`ConstraintStatistics`.  When ``required_constraints`` is supplied,
    those names are included in every group even when absent from all rows;
    their absence is then visible as an explicit unknown count rather than
    disappearing from the comparison.
    """

    rows = normalize_candidate_rows(candidates)
    required = _normalise_required_constraints(required_constraints)
    grouped: dict[tuple[str, float | None], list[CandidateRow]] = {}
    for row in rows:
        grouped.setdefault((row.architecture, row.pi_compressor), []).append(row)

    # Use the union across the whole analysed dataset (plus the declared
    # required profile) so that a constraint not present in one group still
    # appears there with an explicit absent count instead of disappearing from
    # the comparison.
    all_constraint_names = sorted(
        {name for row in rows for name in row.constraints} | required
    )

    def group_key(item: tuple[str, float | None]) -> tuple[str, int, float]:
        architecture, pi_compressor = item
        return (
            architecture,
            0 if pi_compressor is None else 1,
            0.0 if pi_compressor is None else pi_compressor,
        )

    result: list[DesignGroupStatistics] = []
    for key in sorted(grouped, key=group_key):
        group_rows = grouped[key]
        feasibility_counts = {member: 0 for member in FeasibilityStatus}
        for row in group_rows:
            feasibility_counts[
                _profile_feasibility_status(row, required)
            ] += 1

        calculation_failures = [
            row for row in group_rows if not row.calculation_succeeded
        ]
        failure_reasons = _count(
            row.failure_reason or "<not_recorded>" for row in calculation_failures
        )
        failure_stages = _count(
            row.failure_stage or "<not_recorded>" for row in calculation_failures
        )

        constraint_statistics: dict[str, ConstraintStatistics] = {}
        for name in all_constraint_names:
            statuses = [
                row.constraints[name].status
                for row in group_rows
                if name in row.constraints
            ]
            present_rows = len(statuses)
            pass_count = statuses.count(ConstraintStatus.PASS)
            fail_count = statuses.count(ConstraintStatus.FAIL)
            not_checked_count = statuses.count(ConstraintStatus.NOT_CHECKED)
            constraint_statistics[name] = ConstraintStatistics(
                constraint=name,
                denominator_rows=len(group_rows),
                present_rows=present_rows,
                absent_rows=len(group_rows) - present_rows,
                pass_count=pass_count,
                fail_count=fail_count,
                not_checked_count=not_checked_count,
            )

        result.append(
            DesignGroupStatistics(
                architecture=key[0],
                pi_compressor=key[1],
                total_rows=len(group_rows),
                calculation_success=len(group_rows) - len(calculation_failures),
                calculation_failures=len(calculation_failures),
                feasibility_counts=feasibility_counts,
                failure_reason_counts=failure_reasons,
                failure_stage_counts=failure_stages,
                constraint_statistics=constraint_statistics,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class Objective:
    """Explicit Pareto objective name and direction."""

    metric: str
    direction: Direction

    def __post_init__(self) -> None:
        metric = _nonempty_string("objective metric", self.metric)
        if isinstance(self.direction, Direction):
            direction = self.direction
        elif isinstance(self.direction, str):
            try:
                direction = Direction(self.direction)
            except ValueError as exc:
                raise AnalysisError(
                    "objective direction must be exactly MINIMIZE or "
                    f"MAXIMIZE, got {self.direction!r}"
                ) from exc
        else:
            raise AnalysisError(
                "objective direction must be exactly MINIMIZE or MAXIMIZE, "
                f"got {self.direction!r}"
            )
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True)
class ParetoExclusion:
    """A row excluded from Pareto consideration by an explicit policy test."""

    candidate_id: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _nonempty_string("candidate_id", self.candidate_id),
        )
        object.__setattr__(
            self, "reason", _nonempty_string("reason", self.reason)
        )


@dataclass(frozen=True)
class ParetoResult:
    """Immutable Pareto front plus policy/objective exclusions.

    ``front`` preserves the eligible input order.  Equivalent duplicate
    candidates are retained; they do not dominate one another.  ``excluded``
    records policy exclusions and rows missing usable objective values; rows
    merely dominated by another eligible row are not listed there.
    """

    objectives: tuple[Objective, ...]
    front: tuple[CandidateRow, ...]
    excluded: tuple[ParetoExclusion, ...]


def _normalise_required_constraints(
    required_constraints: Iterable[str] | None,
) -> frozenset[str]:
    if required_constraints is None:
        return frozenset()
    if isinstance(required_constraints, (str, bytes)):
        raise AnalysisError(
            "required_constraints must be an iterable of constraint names"
        )
    try:
        items = tuple(required_constraints)
    except TypeError as exc:
        raise AnalysisError(
            "required_constraints must be an iterable of constraint names"
        ) from exc
    names: set[str] = set()
    for item in items:
        names.add(_nonempty_string("required constraint name", item))
    return frozenset(names)


def _normalise_objectives(objectives: Iterable[Objective]) -> tuple[Objective, ...]:
    if isinstance(objectives, (str, bytes)):
        raise AnalysisError("objectives must be an iterable of Objective values")
    try:
        items = tuple(objectives)
    except TypeError as exc:
        raise AnalysisError(
            "objectives must be an iterable of Objective values"
        ) from exc
    if not items:
        raise AnalysisError("at least one Pareto objective is required")
    result: list[Objective] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, Objective):
            raise AnalysisError(
                f"objectives must contain Objective values, got {item!r}"
            )
        if item.metric in seen:
            raise AnalysisError(
                f"duplicate Pareto objective metric {item.metric!r}"
            )
        seen.add(item.metric)
        result.append(item)
    return tuple(result)


def _objective_values(
    row: CandidateRow, objectives: tuple[Objective, ...]
) -> tuple[tuple[float, ...] | None, str | None]:
    values: list[float] = []
    for objective in objectives:
        value = row.metrics.get(objective.metric)
        if value is None:
            return None, f"objective {objective.metric!r} is missing"
        if not math.isfinite(value):
            return None, f"objective {objective.metric!r} is non-finite"
        if objective.direction is Direction.MAXIMIZE:
            values.append(-value)
        else:
            values.append(value)
    return tuple(values), None


def _dominates(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    no_worse = all(a <= b for a, b in zip(left, right))
    strictly_better = any(a < b for a, b in zip(left, right))
    return no_worse and strictly_better


def pareto_front(
    candidates: Iterable[Mapping[str, Any] | CandidateRow],
    objectives: Iterable[Objective],
    *,
    include_calculation_failures: bool = False,
    include_infeasible: bool = False,
    include_unchecked: bool = False,
    required_constraints: Iterable[str] | None = None,
) -> ParetoResult:
    """Return the nondominated front for explicit objectives.

    Parameters
    ----------
    candidates:
        Annotated rows or generic row mappings.
    objectives:
        Explicit :class:`Objective` values.  Duplicate metric names are
        rejected.  Maximization is converted internally to an equivalent
        minimization for dominance testing only; no weighted score is formed.
    required_constraints:
        Optional declared set of applicable constraint names.  Under the
        default ``include_unchecked=False`` policy, a row missing any required
        constraint is excluded as unchecked; absence is never treated as pass.
    include_calculation_failures:
        If false (default), rows whose ``calculation_status`` is not
        ``SUCCESS`` are excluded.
    include_infeasible:
        If false (default), rows with any explicitly failed constraint are
        excluded.
    include_unchecked:
        If false (default), rows with no constraints or any ``NOT_CHECKED``
        constraint are excluded.  This prevents unknown material/life/model
        criteria from entering a front as if they had passed.

    Missing or non-finite objective values always exclude a row.  They are
    never replaced with zero.  The returned front preserves eligible input
    order and keeps all equivalent duplicates.
    """

    include_calculation_failures = _require_bool(
        "include_calculation_failures", include_calculation_failures
    )
    include_infeasible = _require_bool(
        "include_infeasible", include_infeasible
    )
    include_unchecked = _require_bool("include_unchecked", include_unchecked)

    rows = normalize_candidate_rows(candidates)
    normalized_objectives = _normalise_objectives(objectives)
    required = _normalise_required_constraints(required_constraints)

    eligible: list[tuple[CandidateRow, tuple[float, ...]]] = []
    excluded: list[ParetoExclusion] = []
    for row in rows:
        if (
            not row.calculation_succeeded
            and not include_calculation_failures
        ):
            excluded.append(
                ParetoExclusion(
                    row.candidate_id,
                    f"calculation_status={row.calculation_status.value}",
                )
            )
            continue
        if row.failed_constraints and not include_infeasible:
            excluded.append(
                ParetoExclusion(
                    row.candidate_id,
                    "engineering-infeasible constraints: "
                    + ", ".join(row.failed_constraints),
                )
            )
            continue
        if not include_unchecked:
            if not row.constraints and not required:
                excluded.append(
                    ParetoExclusion(
                        row.candidate_id,
                        "no constraints supplied; screening is unverified",
                    )
                )
                continue
            missing_required = tuple(
                sorted(required.difference(row.constraints))
            )
            if missing_required:
                excluded.append(
                    ParetoExclusion(
                        row.candidate_id,
                        "missing required constraints: "
                        + ", ".join(missing_required),
                    )
                )
                continue
            if row.unchecked_constraints:
                excluded.append(
                    ParetoExclusion(
                        row.candidate_id,
                        "unchecked constraints: "
                        + ", ".join(row.unchecked_constraints),
                    )
                )
                continue
        values, reason = _objective_values(row, normalized_objectives)
        if values is None:
            excluded.append(ParetoExclusion(row.candidate_id, reason or ""))
            continue
        eligible.append((row, values))

    front: list[CandidateRow] = []
    for index, (row, values) in enumerate(eligible):
        dominated = False
        for other_index, (_, other_values) in enumerate(eligible):
            if index == other_index:
                continue
            if _dominates(other_values, values):
                dominated = True
                break
        if not dominated:
            front.append(row)

    return ParetoResult(
        objectives=normalized_objectives,
        front=tuple(front),
        excluded=tuple(excluded),
    )


@dataclass(frozen=True)
class NearFeasibilityEntry:
    """Full violation vector and optional deterministic near-feasibility rank.

    ``violation_vector`` contains every constraint name and its nonnegative
    bound violation, or ``None`` for ``NOT_CHECKED``.  ``normalized_violation``
    is populated only through explicit per-constraint scales.  ``rank`` is
    ``None`` whenever the row cannot be ranked without inventing information;
    ``unranked_reason`` then explains why.
    """

    candidate: CandidateRow
    violation_vector: Mapping[str, float | None]
    normalized_violation_vector: Mapping[str, float | None]
    unknown_constraints: tuple[str, ...]
    missing_normalization: tuple[str, ...]
    rank: int | None
    unranked_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, CandidateRow):
            raise AnalysisError("candidate must be a CandidateRow")
        for name, vector in (
            ("violation_vector", self.violation_vector),
            (
                "normalized_violation_vector",
                self.normalized_violation_vector,
            ),
        ):
            if not isinstance(vector, Mapping):
                raise AnalysisError(f"{name} must be a mapping")
            if not set(vector).issuperset(self.candidate.constraints):
                raise AnalysisError(
                    f"{name} must contain every candidate constraint name"
                )
        if set(self.violation_vector) != set(
            self.normalized_violation_vector
        ):
            raise AnalysisError(
                "violation vectors must contain identical constraint names"
            )
        if self.rank is not None:
            if self.rank < 1:
                raise AnalysisError("rank must be >= 1 or None")
            if self.unranked_reason is not None:
                raise AnalysisError(
                    "a ranked entry cannot have an unranked_reason"
                )
        else:
            _nonempty_string("unranked_reason", self.unranked_reason)
        object.__setattr__(
            self, "violation_vector", MappingProxyType(dict(self.violation_vector))
        )
        object.__setattr__(
            self,
            "normalized_violation_vector",
            MappingProxyType(dict(self.normalized_violation_vector)),
        )
        object.__setattr__(
            self,
            "unknown_constraints",
            tuple(sorted(self.unknown_constraints)),
        )
        object.__setattr__(
            self,
            "missing_normalization",
            tuple(sorted(self.missing_normalization)),
        )


@dataclass(frozen=True)
class NearFeasibilityReport:
    """Immutable near-feasibility report."""

    entries: tuple[NearFeasibilityEntry, ...]

    @property
    def ranked_entries(self) -> tuple[NearFeasibilityEntry, ...]:
        """Entries with a rank, in ascending rank order."""

        return tuple(entry for entry in self.entries if entry.rank is not None)

    @property
    def unranked_entries(self) -> tuple[NearFeasibilityEntry, ...]:
        """Entries without a rank, in deterministic input order."""

        return tuple(entry for entry in self.entries if entry.rank is None)


def rank_near_feasible(
    candidates: Iterable[Mapping[str, Any] | CandidateRow],
    *,
    include_calculation_failures: bool = False,
    required_constraints: Iterable[str] | None = None,
) -> NearFeasibilityReport:
    """Return full violation vectors and a deterministic near-feasibility rank.

    Ranking rule: for each rankable row, take its complete vector of normalized
    violations, sort the values from worst to least, and order those tuples
    lexicographically from smallest to largest (then candidate id, then input
    position).  This is a transparent worst-violation ordering, not a weighted
    sum and not a hidden scale.  A row with a ``NOT_CHECKED`` constraint, a
    failed constraint lacking an explicit ``scale``, a missing required
    constraint, or no constraints at all is unranked and reports the reason.
    Unknown constraints never contribute an implicit zero.  When
    ``required_constraints`` is supplied, absent names appear in the full
    violation vector as ``None``.
    """

    include_calculation_failures = _require_bool(
        "include_calculation_failures", include_calculation_failures
    )

    rows = normalize_candidate_rows(candidates)
    required = _normalise_required_constraints(required_constraints)
    unranked_entries: list[NearFeasibilityEntry] = []
    rankable: list[
        tuple[tuple[float, ...], str, int, NearFeasibilityEntry]
    ] = []

    for index, row in enumerate(rows):
        if not row.calculation_succeeded and not include_calculation_failures:
            continue

        names = tuple(sorted(set(row.constraints) | required))
        violation_vector: dict[str, float | None] = {}
        normalized_vector: dict[str, float | None] = {}
        for name in names:
            evaluation = row.constraints.get(name)
            violation_vector[name] = (
                evaluation.violation if evaluation is not None else None
            )
            normalized_vector[name] = (
                evaluation.normalized_violation
                if evaluation is not None
                else None
            )
        missing_required = tuple(
            sorted(required.difference(row.constraints))
        )
        unknown = tuple(
            name for name in names if violation_vector[name] is None
        )
        missing_normalization = tuple(
            name
            for name in names
            if violation_vector[name] is not None
            and violation_vector[name] > 0.0
            and normalized_vector[name] is None
        )

        reason: str | None = None
        if not row.calculation_succeeded:
            reason = f"calculation_status={row.calculation_status.value}"
        elif not row.constraints and not required:
            reason = "no constraints supplied; screening is unverified"
        elif missing_required:
            reason = (
                "missing required constraints: "
                + ", ".join(missing_required)
            )
        elif unknown:
            reason = "unchecked constraints: " + ", ".join(unknown)
        elif missing_normalization:
            reason = (
                "missing, underflowing, or non-finite explicit scale for "
                "violated constraints: "
                + ", ".join(missing_normalization)
            )

        entry = NearFeasibilityEntry(
            candidate=row,
            violation_vector=violation_vector,
            normalized_violation_vector=normalized_vector,
            unknown_constraints=unknown,
            missing_normalization=missing_normalization,
            rank=None,
            unranked_reason=reason or "not rankable",
        )
        if reason is None:
            key = tuple(
                sorted(
                    (
                        value
                        for value in normalized_vector.values()
                        if value is not None
                    ),
                    reverse=True,
                )
            )
            rankable.append((key, row.candidate_id, index, entry))
        else:
            unranked_entries.append(entry)

    rankable.sort(key=lambda item: (item[0], item[1], item[2]))
    ranked_entries: list[NearFeasibilityEntry] = []
    for rank, (_, _, _, entry) in enumerate(rankable, start=1):
        ranked_entries.append(
            NearFeasibilityEntry(
                candidate=entry.candidate,
                violation_vector=entry.violation_vector,
                normalized_violation_vector=entry.normalized_violation_vector,
                unknown_constraints=entry.unknown_constraints,
                missing_normalization=entry.missing_normalization,
                rank=rank,
                unranked_reason=None,
            )
        )

    return NearFeasibilityReport(
        entries=tuple(ranked_entries + unranked_entries)
    )


@dataclass(frozen=True)
class SensitivityMetric:
    """Paired metric difference, slope, and optional relative change."""

    metric: str
    baseline: float | None
    perturbed: float | None
    difference: float | None
    derivative: float | None
    relative_change: float | None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metric", _nonempty_string("metric", self.metric)
        )


@dataclass(frozen=True)
class PairedSensitivity:
    """Controlled paired finite-difference result.

    ``dx`` is the signed difference of the designated input parameter.  A
    positive metric difference divided by ``dx`` is the local slope.  The
    relative change is only populated when the caller explicitly requests it
    and the baseline metric is nonzero.
    """

    baseline_id: str
    perturbed_id: str
    architecture: str
    model_variant: str | None
    parameter: str
    baseline_parameter: float
    perturbed_parameter: float
    dx: float
    relative: bool
    metrics: Mapping[str, SensitivityMetric]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "baseline_id", _nonempty_string("baseline_id", self.baseline_id)
        )
        object.__setattr__(
            self,
            "perturbed_id",
            _nonempty_string("perturbed_id", self.perturbed_id),
        )
        object.__setattr__(
            self, "architecture", _nonempty_string("architecture", self.architecture)
        )
        object.__setattr__(
            self, "parameter", _nonempty_string("parameter", self.parameter)
        )
        object.__setattr__(
            self, "metrics", MappingProxyType(dict(self.metrics))
        )


def _sensitivity_metric_names(
    baseline: CandidateRow,
    perturbed: CandidateRow,
    metric_names: Iterable[str] | None,
) -> tuple[str, ...]:
    if metric_names is None:
        return tuple(sorted(set(baseline.metrics) | set(perturbed.metrics)))
    if isinstance(metric_names, (str, bytes)):
        raise SensitivityError("metric_names must be an iterable of names")
    try:
        names = tuple(metric_names)
    except TypeError as exc:
        raise SensitivityError(
            "metric_names must be an iterable of names"
        ) from exc
    if not names:
        raise SensitivityError("metric_names must not be empty")
    result: list[str] = []
    seen: set[str] = set()
    for name in names:
        name = _nonempty_string("metric name", name)
        if name in seen:
            raise SensitivityError(f"duplicate metric name {name!r}")
        seen.add(name)
        result.append(name)
    return tuple(result)


def paired_sensitivity(
    baseline: CandidateRow,
    perturbed: CandidateRow,
    *,
    parameter: str,
    metric_names: Iterable[str] | None = None,
    relative: bool = False,
) -> PairedSensitivity:
    """Compare exactly two controlled rows differing in one input parameter.

    Both rows must have ``SUCCESS`` calculation status, the same architecture,
    the same model variant, and identical input keys and values except for
    ``parameter``.  Matching is exact; no tolerance is invented.  The parameter
    must be finite in both rows and ``dx`` must be nonzero.  Metrics are
    compared only when finite in both rows; otherwise ``difference`` and the
    derivatives are ``None`` with an explicit reason.  If ``relative`` is true,
    ``relative_change = difference / abs(baseline)`` when the baseline is
    nonzero; a zero baseline yields ``None``, not infinity.

    This function deliberately does not accept a sweep or a set of rows.  It
    refuses unpaired correlation as evidence of a controlled sensitivity.
    """

    if not isinstance(baseline, CandidateRow) or not isinstance(
        perturbed, CandidateRow
    ):
        raise SensitivityError("both rows must be CandidateRow values")
    try:
        parameter = _nonempty_string("parameter", parameter)
    except AnalysisError as exc:
        raise SensitivityError(str(exc)) from exc
    if not isinstance(relative, bool):
        raise SensitivityError("relative must be a bool")
    if baseline.architecture != perturbed.architecture:
        raise SensitivityError(
            "paired rows must have the same architecture: "
            f"{baseline.architecture!r} != {perturbed.architecture!r}"
        )
    if baseline.model_variant != perturbed.model_variant:
        raise SensitivityError(
            "paired rows must have the same model_variant: "
            f"{baseline.model_variant!r} != {perturbed.model_variant!r}"
        )
    if not baseline.calculation_succeeded:
        raise SensitivityError(
            f"baseline {baseline.candidate_id!r} did not calculate successfully"
        )
    if not perturbed.calculation_succeeded:
        raise SensitivityError(
            f"perturbed {perturbed.candidate_id!r} did not calculate successfully"
        )

    baseline_keys = set(baseline.inputs)
    perturbed_keys = set(perturbed.inputs)
    if baseline_keys != perturbed_keys:
        missing = sorted(baseline_keys - perturbed_keys)
        extra = sorted(perturbed_keys - baseline_keys)
        raise SensitivityError(
            "paired input mappings must have identical keys; "
            f"missing in perturbed={missing}, extra in perturbed={extra}"
        )
    if parameter not in baseline_keys:
        raise SensitivityError(
            f"parameter {parameter!r} is absent from the input mappings"
        )
    for key in sorted(baseline_keys - {parameter}):
        if baseline.inputs[key] != perturbed.inputs[key]:
            raise SensitivityError(
                f"unpaired inputs differ at {key!r}: "
                f"{baseline.inputs[key]!r} != {perturbed.inputs[key]!r}"
            )

    try:
        baseline_parameter = _finite_float(
            f"baseline input {parameter!r}", baseline.inputs[parameter]
        )
        perturbed_parameter = _finite_float(
            f"perturbed input {parameter!r}", perturbed.inputs[parameter]
        )
    except AnalysisError as exc:
        raise SensitivityError(str(exc)) from exc
    dx = perturbed_parameter - baseline_parameter
    if not math.isfinite(dx):
        raise SensitivityError("parameter difference is non-finite")
    if dx == 0.0:
        raise SensitivityError(
            f"parameter {parameter!r} has zero difference; no finite "
            "difference quotient exists"
        )

    names = _sensitivity_metric_names(baseline, perturbed, metric_names)
    metrics: dict[str, SensitivityMetric] = {}
    for name in names:
        baseline_value = baseline.metrics.get(name)
        perturbed_value = perturbed.metrics.get(name)
        baseline_usable = (
            baseline_value is not None and math.isfinite(baseline_value)
        )
        perturbed_usable = (
            perturbed_value is not None and math.isfinite(perturbed_value)
        )
        reason: str | None = None
        difference: float | None = None
        derivative: float | None = None
        relative_change: float | None = None
        if baseline_usable and perturbed_usable:
            assert baseline_value is not None and perturbed_value is not None
            difference = perturbed_value - baseline_value
            if not math.isfinite(difference):
                raise SensitivityError(
                    f"metric {name!r} difference is non-finite"
                )
            derivative = difference / dx
            if not math.isfinite(derivative):
                raise SensitivityError(
                    f"metric {name!r} derivative is non-finite"
                )
            if relative:
                if baseline_value != 0.0:
                    relative_change = difference / abs(baseline_value)
                    if not math.isfinite(relative_change):
                        raise SensitivityError(
                            f"metric {name!r} relative change is non-finite"
                        )
        else:
            reason = "metric is missing or non-finite in baseline/perturbed"

        metrics[name] = SensitivityMetric(
            metric=name,
            baseline=baseline_value,
            perturbed=perturbed_value,
            difference=difference,
            derivative=derivative,
            relative_change=relative_change,
            unavailable_reason=reason,
        )

    return PairedSensitivity(
        baseline_id=baseline.candidate_id,
        perturbed_id=perturbed.candidate_id,
        architecture=baseline.architecture,
        model_variant=baseline.model_variant,
        parameter=parameter,
        baseline_parameter=baseline_parameter,
        perturbed_parameter=perturbed_parameter,
        dx=dx,
        relative=relative,
        metrics=metrics,
    )


__all__ = [
    "AnalysisError",
    "CalculationStatus",
    "CandidateRow",
    "ConstraintError",
    "ConstraintEvaluation",
    "ConstraintStatistics",
    "ConstraintStatus",
    "DesignGroupStatistics",
    "Direction",
    "FeasibilityStatus",
    "NearFeasibilityEntry",
    "NearFeasibilityReport",
    "OPTIONAL_ROW_KEYS",
    "Objective",
    "PairedSensitivity",
    "ParetoExclusion",
    "ParetoResult",
    "REQUIRED_ROW_KEYS",
    "SensitivityError",
    "SensitivityMetric",
    "VerificationStatus",
    "normalize_candidate_rows",
    "pareto_front",
    "paired_sensitivity",
    "rank_near_feasible",
    "summarize_design_groups",
]
