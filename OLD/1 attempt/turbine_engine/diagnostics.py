"""Explicit numerical-failure diagnostics shared by extracted solvers.

The extracted solvers preserve the notebook equations, evaluation order,
iteration limits and tolerances.  When a fixed-point iteration or a scalar
solve stops because its limit was reached, because its secant denominator
stalled, or because no valid stage evaluation remains, that is a numerical
outcome, not a successful result.  This module provides one typed, structured
way to report it without changing any physical equation.

Statuses
--------
``NUMERICAL_FAILURE``
    The solve did not close: iteration/stall/no-valid-evaluation cases.
``NON_PHYSICAL_STATE``
    A state is mathematically usable but non-physical.  The extracted solvers
    use the namespace's own ``НетРешения`` for invalid evaluated states; this
    status is available so callers can keep the two cases separate and never
    turn a numerical failure into an engineering acceptance filter.

The record is JSON friendly: mappings become plain dictionaries, sequences
become lists, real numbers become finite ``float``/``int``, and non-finite
numbers become ``None`` so no ``NaN``/``Infinity`` can be serialized.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Mapping

NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
NON_PHYSICAL_STATE = "NON_PHYSICAL_STATE"
STATUSES: tuple[str, ...] = (NUMERICAL_FAILURE, NON_PHYSICAL_STATE)


def sanitize_state(value: Any) -> Any:
    """Return a JSON-friendly copy of ``value`` with non-finite numbers as None.

    Supported values are mappings, sequences, strings, booleans and real
    numbers.  Unknown objects are omitted (``None``) rather than serialized by
    accident.  The helper does not retain references to caller-owned objects.
    """

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): sanitize_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_state(item) for item in value]
    return None


class SolverDiagnosticsError(Exception):
    """Typed numerical result with machine-readable diagnostic attributes.

    Parameters
    ----------
    message:
        Human-readable description.  It should state what did not close.
    stage:
        Stable identifier of the solver stage, for example
        ``"baseline.распределить_π"``.
    reason:
        Stable short reason such as ``"pi_secant_iteration_exhausted"``.
    status:
        One of :data:`STATUSES`; defaults to :data:`NUMERICAL_FAILURE`.
    last_valid_state:
        Last finite iterate/inputs available when the solve stopped.  It is
        sanitized through :func:`sanitize_state`.
    """

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        reason: str,
        status: str = NUMERICAL_FAILURE,
        last_valid_state: Mapping[str, Any] | None = None,
    ) -> None:
        if status not in STATUSES:
            raise ValueError(
                f"unknown solver diagnostic status {status!r}; "
                f"expected one of {STATUSES!r}"
            )
        self.stage = str(stage)
        self.reason = str(reason)
        self.status = status
        self.last_valid_state = sanitize_state(last_valid_state or {})
        super().__init__(message)

    def as_record(self) -> dict[str, Any]:
        """Return the diagnostic as a JSON-friendly dictionary."""

        return {
            "error": type(self).__name__,
            "stage": self.stage,
            "reason": self.reason,
            "status": self.status,
            "last_valid_state": sanitize_state(self.last_valid_state),
        }

    def __str__(self) -> str:
        message = super().__str__()
        return (
            f"{message} "
            f"[stage={self.stage}, reason={self.reason}, status={self.status}]"
        )


class NumericalConvergenceError(SolverDiagnosticsError):
    """A solve did not close to its existing tolerance.

    This is raised for iteration-limit exhaustion, a stalled secant
    denominator, or a final iteration with no valid evaluated state.  It never
    means that the candidate violates an engineering constraint.
    """


__all__ = [
    "NUMERICAL_FAILURE",
    "NON_PHYSICAL_STATE",
    "NumericalConvergenceError",
    "STATUSES",
    "SolverDiagnosticsError",
    "sanitize_state",
]
