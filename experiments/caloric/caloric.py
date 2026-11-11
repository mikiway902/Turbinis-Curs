"""Opt-in source-backed true-``cp`` caloric model and variable-``cp`` units.

STATUS: ``NOT_PHYSICALLY_VALIDATED``.  This module is a faithful software
transcription of the source true-``cp`` expression on the pages identified in
the task context; it is not a validation of that expression over the whole
temperature range, and it makes no bladed-stage, map, cooling, material, or
operating-range claim.

Why a separate module
=====================
The existing cycle and component modules use the legacy **mean** ``cp``
polynomial (``turbine_engine.cycle.calc_cp_kerosene``).  The source's **true**
``cp`` is an independent fit; by authority it is *not* the temperature
derivative of that mean polynomial and is *not* obtained by substituting
``cp(T)`` into the mean-``cp`` equations.  This module therefore leaves every
legacy equation and record untouched and is an explicitly opt-in alternative.

Source expression
=================
The source gives ``cp`` in kcal/(kg K); the module converts with the source
factor ``SOURCE_CP_UNIT_FACTOR = 4187`` J/kcal.  With excess-air ratio
``a = alpha`` (``alpha = +inf`` for dry air), the transcription is exactly:

``T < 750``:

.. code-block:: text

    (0.0174/a + 0.2407)
    + (0.0093/a + 0.0193) * (2.5e-3*T - 0.875)
    + (2e-3 - 1.056e-3/(a - 0.2))
      * (2.5e-5*T*T - 0.0275*T + 6.5625)

``T >= 750``:

.. code-block:: text

    (0.0267/a + 0.26)
    + (0.0133/a + 0.032) * (1.176e-3*T - 0.88235)
    - (0.374e-2 + 0.94e-2/(a*a + 10))
      * (5.5556e-6*T*T - 1.3056e-2*T + 6.67)

The reciprocal terms use their analytic ``a = +inf`` limits (zero), so dry air
is evaluated without ``inf * 0``.  ``alpha < 1`` (the rich branch) is outside
the visibly supported source chart and is rejected explicitly as an
unsupported source domain, not as an engineering acceptance filter.

The two branch polynomials do not have to meet with equal value or slope at
750 K.  ``cp`` is deliberately **not** smoothed: the lower branch is used below
750 K and the upper branch from 750 K onward.  Enthalpy and thermal entropy are
still continuous because they are integrals; the primitive is split exactly at
750 K and anchored at ``T_ref``.

Math implemented
================
For each branch ``cp(T) = c0 + c1*T + c2*T**2`` (all coefficients in SI units):

.. code-block:: text

    h(T)   = integral[T_ref, T] cp dT                 [J/kg]
    phi(T) = integral[T_ref, T] cp/T dT               [J/(kg K)]
    k(T)   = cp(T) / (cp(T) - R)

The component relations are:

.. code-block:: text

    compressor: integral[T_in, T_out] cp/T dT = R*ln(pi_c)/eta_poly
                work = h(T_out) - h(T_in) > 0

    turbine:    integral[T_out, T_in] cp/T dT = eta_poly*R*ln(pi_t)
                work = h(T_in) - h(T_out) > 0

For the inverse turbine, a prescribed positive work first determines ``T_out``
from ``h(T_in) - h(T_out) = work`` and then
``pi_t = exp((phi(T_in) - phi(T_out))/(eta_poly*R))``.  Pressure ratios are
strictly greater than one; efficiency is in ``(0, 1]``.  No engineering
acceptance range is applied.

Numerics
========
Integrals use analytic quadratic primitives.  Differences use factored
polynomial forms and ``log1p`` for ``ln(T2/T1)`` and ``ln(pi)`` so small
differences remain stable.  The bounded root solve is deterministic bisection
with a **relative residual closure** criterion, not an absolute temperature
width.  If adjacent representable temperatures are reached, the better
evaluated endpoint is returned only when it meets the caller-requested relative
closure; otherwise :class:`FloatingResolutionError` is raised rather than
returning a no-op or knowingly wrong work value.  There is no hidden
resolution-tolerance relaxation.

Brackets toward ``T -> 0`` are expanded by geometric halving, so a legitimate
deep expansion below ``0.2 * T_in`` is reached without ever jumping through
zero.  A bracket-expansion cap, a non-positive temperature limit, a
non-positive ``cp``, ``cp <= R`` on a requested interval, and a non-finite
intermediate all produce explicit typed outcomes rather than a clamped
temperature or a silently returned last iterate.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any

from turbine_engine.architecture import StationState
from turbine_engine.diagnostics import (
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
    sanitize_state,
)

#: Source conversion from kcal to J.
SOURCE_CP_UNIT_FACTOR = 4187.0

#: Exact temperature split between the two source ``cp`` polynomials.
CP_BRANCH_TEMPERATURE = 750.0

#: Unique identifier of this transcribed and integrated source variant.
SOURCE_VARIANT = "textbook_true_cp_integrated_v1"

#: Human-readable provenance for the transcribed source expression.
SOURCE_REFERENCE = (
    "true-cp chart (p.54) and appendix 6 (p.57) of the provided textbook, "
    "confirmed from the source image; p.57 contains the separate true-cp and "
    "mean-cp fits, while p.20 is a fuel-balance use of the mean cp and is not "
    "a true-cp source"
)

#: Explicit evidence status; this module is not physical validation.
SOURCE_VALIDATION_STATUS = "NOT_PHYSICALLY_VALIDATED"

#: Default relative closure tolerance for the solved scalar residual.
DEFAULT_SOLVER_TOL = 1e-10

#: Default maximum bisection iterations.  This is not a physical iteration
#: count and is deliberately modest for a bracketed scalar solve.
DEFAULT_SOLVER_MAX_ITER = 200

#: Hard numerical cap on bounded bracket expansion (doubling the bracket gap).
MAX_BRACKET_EXPANSIONS = 64

_DEFAULT_STATE_RTOL = 1e-9
_DEFAULT_STATE_CP_ATOL = 1e-6
_DEFAULT_STATE_K_ATOL = 1e-9


class CaloricModelError(ValueError):
    """Invalid model input, source-domain input, or incompatible state."""


class IncompatibleStateError(CaloricModelError):
    """A station state's ``cp``/``k`` do not match the selected model."""


class CaloricModelDomainError(CaloricModelError):
    """A mathematically or source-domain unusable caloric state.

    This is intentionally distinct from
    :class:`turbine_engine.diagnostics.NumericalConvergenceError`: a model
    domain limit is not an exhausted solver iteration.  ``status`` is
    :data:`turbine_engine.diagnostics.NON_PHYSICAL_STATE` for structured
    reporting.
    """

    status = NON_PHYSICAL_STATE

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        reason: str,
        last_valid_state: dict[str, Any] | None = None,
    ) -> None:
        self.stage = str(stage)
        self.reason = str(reason)
        self.last_valid_state = sanitize_state(last_valid_state or {})
        super().__init__(message)

    def as_record(self) -> dict[str, Any]:
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


class FloatingResolutionError(NumericalConvergenceError):
    """The requested change is below representable temperature resolution.

    The bisection bracket reached adjacent floating-point temperatures and the
    best evaluated endpoint still missed the caller-requested relative closure.
    The call does not return a no-op temperature or a knowingly wrong work
    value.  ``last_valid_state`` records the final bracket, its residuals, and
    the achieved relative residual.
    """


def _strict_real(name: str, value: Any, *, allow_infinite: bool = False) -> float:
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, numbers.Real):
        raise CaloricModelError(
            f"{name} must be a real number, got {value!r}"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CaloricModelError(
            f"{name} must be a real number, got {value!r}"
        ) from exc
    if math.isnan(result) or result == -math.inf:
        raise CaloricModelError(f"{name} must be a finite real number, got {value!r}")
    if not allow_infinite and not math.isfinite(result):
        raise CaloricModelError(f"{name} must be finite, got {value!r}")
    return result


def _strict_positive(name: str, value: Any) -> float:
    result = _strict_real(name, value)
    if result <= 0.0:
        raise CaloricModelError(f"{name} must be > 0, got {value!r}")
    return result


def _strict_alpha(value: Any) -> float:
    result = _strict_real("alpha", value, allow_infinite=True)
    if result < 1.0:
        raise CaloricModelError(
            "alpha must be >= 1 or +inf; the rich branch (alpha < 1) is not "
            f"supported by the source transcription, got {value!r}"
        )
    return result


def _strict_iterations(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CaloricModelError(
            f"max_iter must be a positive integer, got {value!r}"
        )
    if value <= 0:
        raise CaloricModelError(f"max_iter must be > 0, got {value!r}")
    return int(value)


def _strict_tolerance(value: Any) -> float:
    result = _strict_positive("tol", value)
    if not math.isfinite(result):
        raise CaloricModelError(f"tol must be finite, got {value!r}")
    return result


def _finite(value: float, *, stage: str, reason: str, state: dict[str, Any]) -> float:
    if not math.isfinite(value):
        raise CaloricModelDomainError(
            f"non-finite numeric result in {stage}: {value!r}",
            stage=stage,
            reason=reason,
            last_valid_state=state,
        )
    return value


def _poly(coeffs: tuple[float, float, float], temperature: float) -> float:
    c0, c1, c2 = coeffs
    return c0 + c1 * temperature + c2 * temperature * temperature


def _quadratic_h_difference(
    coeffs: tuple[float, float, float], t_low: float, t_high: float
) -> float:
    """``integral[t_low, t_high] (c0 + c1*T + c2*T^2) dT``.

    The factored form is algebraically exact and avoids subtracting two large
    near-equal primitives when the interval is small.
    """

    c0, c1, c2 = coeffs
    delta = t_high - t_low
    return delta * (
        c0
        + 0.5 * c1 * (t_low + t_high)
        + (c2 / 3.0)
        * (t_low * t_low + t_low * t_high + t_high * t_high)
    )


def _quadratic_phi_difference(
    coeffs: tuple[float, float, float], t_low: float, t_high: float
) -> float:
    """``integral[t_low, t_high] (c0/T + c1 + c2*T) dT``."""

    c0, c1, c2 = coeffs
    delta = t_high - t_low
    log_ratio = math.log1p(delta / t_low)
    return (
        c0 * log_ratio
        + c1 * delta
        + 0.5 * c2 * delta * (t_high + t_low)
    )


def _segment_minimum(
    coeffs: tuple[float, float, float], t_low: float, t_high: float
) -> float:
    """Minimum of one quadratic on a closed segment, including its vertex."""

    c0, c1, c2 = coeffs
    candidates = [
        c0 + c1 * t_low + c2 * t_low * t_low,
        c0 + c1 * t_high + c2 * t_high * t_high,
    ]
    if c2 != 0.0:
        vertex = -c1 / (2.0 * c2)
        if t_low < vertex < t_high:
            candidates.append(c0 + c1 * vertex + c2 * vertex * vertex)
    return min(candidates)


@dataclass(frozen=True)
class CaloricGasModel:
    """Immutable true-``cp`` source model for one fixed excess-air ratio.

    ``R`` [J/(kg K)], ``alpha`` [-] and ``T_ref`` [K] are caller-explicit.
    ``T_ref`` is only an integration anchor; it is not a physical validity
    limit.  All temperatures must be finite and positive.  The model does not
    impose an upper or lower physical temperature range; instead it evaluates
    the source polynomial and raises a typed domain error if ``cp`` is not
    finite and positive (or not above ``R`` where ``k`` is requested).
    """

    R: float
    alpha: float
    T_ref: float = 288.0

    def __post_init__(self) -> None:
        R = _strict_positive("R", self.R)
        alpha = _strict_alpha(self.alpha)
        T_ref = _strict_positive("T_ref", self.T_ref)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "T_ref", T_ref)

        # The reference state itself must be usable by this caloric model.
        cp_ref = self.cp(T_ref)
        if cp_ref <= R:
            raise CaloricModelError(
                f"cp(T_ref) must be > R, got cp={cp_ref!r}, R={R!r}"
            )
        self.k(T_ref)  # validates the finite isentropic-exponent branch

    @property
    def source_variant(self) -> str:
        return SOURCE_VARIANT

    @property
    def validation_status(self) -> str:
        return SOURCE_VALIDATION_STATUS

    @staticmethod
    def _reciprocal_alpha(alpha: float) -> float:
        return 0.0 if alpha == math.inf else 1.0 / alpha

    @staticmethod
    def _reciprocal_alpha_minus_0p2(alpha: float) -> float:
        return 0.0 if alpha == math.inf else 1.0 / (alpha - 0.2)

    @staticmethod
    def _reciprocal_alpha_sq_plus_10(alpha: float) -> float:
        if alpha == math.inf:
            return 0.0
        return 1.0 / (alpha * alpha + 10.0)

    def _low_coefficients(self) -> tuple[float, float, float]:
        """SI coefficients ``c0 + c1*T + c2*T^2`` for ``T < 750``."""

        alpha = self.alpha
        factor = SOURCE_CP_UNIT_FACTOR
        inv_a = self._reciprocal_alpha(alpha)
        inv_a_minus = self._reciprocal_alpha_minus_0p2(alpha)

        b0 = 0.0174 * inv_a + 0.2407
        b1 = 0.0093 * inv_a + 0.0193
        b2 = 2.0e-3 - 1.056e-3 * inv_a_minus

        return (
            factor * (b0 - 0.875 * b1 + 6.5625 * b2),
            factor * (2.5e-3 * b1 - 0.0275 * b2),
            factor * (2.5e-5 * b2),
        )

    def _high_coefficients(self) -> tuple[float, float, float]:
        """SI coefficients ``c0 + c1*T + c2*T^2`` for ``T >= 750``."""

        alpha = self.alpha
        factor = SOURCE_CP_UNIT_FACTOR
        inv_a = self._reciprocal_alpha(alpha)
        inv_a_sq = self._reciprocal_alpha_sq_plus_10(alpha)

        d0 = 0.0267 * inv_a + 0.26
        d1 = 0.0133 * inv_a + 0.032
        d2 = 0.374e-2 + 0.94e-2 * inv_a_sq

        return (
            factor * (d0 - 0.88235 * d1 - 6.67 * d2),
            factor * (1.176e-3 * d1 + 1.3056e-2 * d2),
            factor * (-5.5556e-6 * d2),
        )

    def _coefficients(self, temperature: float) -> tuple[float, float, float]:
        if temperature < CP_BRANCH_TEMPERATURE:
            return self._low_coefficients()
        return self._high_coefficients()

    def _cp_from_coefficients(
        self, coeffs: tuple[float, float, float], temperature: float
    ) -> float:
        return _poly(coeffs, temperature)

    def cp(self, T: float) -> float:
        """True source ``cp(T)`` in J/(kg K), without branch smoothing."""

        temperature = _strict_positive("T", T)
        coeffs = self._coefficients(temperature)
        cp = self._cp_from_coefficients(coeffs, temperature)
        cp = _finite(
            cp,
            stage="caloric.cp",
            reason="cp_overflow",
            state={"T": temperature, "alpha": self.alpha},
        )
        if cp <= 0.0:
            raise CaloricModelDomainError(
                f"cp must be positive, got {cp!r} at T={temperature!r}",
                stage="caloric.cp",
                reason="cp_nonpositive",
                last_valid_state={"T": temperature, "alpha": self.alpha},
            )
        return cp

    def k(self, T: float) -> float:
        """``k = cp/(cp-R)``; raises when ``cp <= R``."""

        temperature = _strict_positive("T", T)
        cp = self.cp(temperature)
        denominator = cp - self.R
        if denominator <= 0.0:
            raise CaloricModelDomainError(
                f"cp must be > R for k, got cp={cp!r}, R={self.R!r} "
                f"at T={temperature!r}",
                stage="caloric.k",
                reason="cp_not_above_R",
                last_valid_state={
                    "T": temperature,
                    "cp": cp,
                    "R": self.R,
                },
            )
        k = cp / denominator
        return _finite(
            k,
            stage="caloric.k",
            reason="k_overflow",
            state={"T": temperature, "cp": cp, "R": self.R},
        )

    def _segment_minimum_checked(
        self,
        coeffs: tuple[float, float, float],
        t_low: float,
        t_high: float,
        *,
        stage: str,
    ) -> float:
        value = _segment_minimum(coeffs, t_low, t_high)
        return _finite(
            value,
            stage=stage,
            reason="cp_polynomial_overflow",
            state={"T_low": t_low, "T_high": t_high, "alpha": self.alpha},
        )

    def _minimum_cp_on_interval(self, t_low: float, t_high: float) -> float:
        split = CP_BRANCH_TEMPERATURE
        stage = "caloric.cp_interval"
        if t_high < split:
            return self._segment_minimum_checked(
                self._low_coefficients(), t_low, t_high, stage=stage
            )
        if t_low >= split:
            return self._segment_minimum_checked(
                self._high_coefficients(), t_low, t_high, stage=stage
            )
        low_min = self._segment_minimum_checked(
            self._low_coefficients(), t_low, split, stage=stage
        )
        high_min = self._segment_minimum_checked(
            self._high_coefficients(), split, t_high, stage=stage
        )
        return min(low_min, high_min)

    def ensure_cp_above(
        self, T_1: float, T_2: float, floor: float = 0.0
    ) -> None:
        """Require ``cp > floor`` over the whole selected interval.

        The quadratic branch is checked at both endpoints and at its vertex
        when the vertex lies inside each segment.  This is stronger than
        checking endpoint ``cp`` values only.
        """

        t_1 = _strict_positive("T_1", T_1)
        t_2 = _strict_positive("T_2", T_2)
        bound = _strict_real("floor", floor)
        t_low, t_high = (t_1, t_2) if t_1 <= t_2 else (t_2, t_1)
        minimum = self._minimum_cp_on_interval(t_low, t_high)
        if minimum <= bound:
            reason = "cp_nonpositive" if bound == 0.0 else "cp_not_above_floor"
            raise CaloricModelDomainError(
                f"cp must be > {bound!r} over [{t_low!r}, {t_high!r}], "
                f"minimum is {minimum!r}",
                stage="caloric.cp_interval",
                reason=reason,
                last_valid_state={
                    "T_low": t_low,
                    "T_high": t_high,
                    "floor": bound,
                    "minimum_cp": minimum,
                    "alpha": self.alpha,
                },
            )

    def _h_primitive(self, temperature: float) -> float:
        """Continuous SI enthalpy primitive, split and offset at 750 K."""

        if temperature < CP_BRANCH_TEMPERATURE:
            c0, c1, c2 = self._low_coefficients()
            return (
                c0 * temperature
                + 0.5 * c1 * temperature * temperature
                + (c2 / 3.0) * temperature * temperature * temperature
            )

        c0, c1, c2 = self._high_coefficients()
        high_value = (
            c0 * temperature
            + 0.5 * c1 * temperature * temperature
            + (c2 / 3.0) * temperature * temperature * temperature
        )
        lc0, lc1, lc2 = self._low_coefficients()
        split = CP_BRANCH_TEMPERATURE
        low_at_split = (
            lc0 * split
            + 0.5 * lc1 * split * split
            + (lc2 / 3.0) * split * split * split
        )
        high_at_split = (
            c0 * split
            + 0.5 * c1 * split * split
            + (c2 / 3.0) * split * split * split
        )
        return high_value + (low_at_split - high_at_split)

    def _phi_primitive(self, temperature: float) -> float:
        """Continuous SI thermal-entropy primitive, split/offset at 750 K."""

        if temperature < CP_BRANCH_TEMPERATURE:
            c0, c1, c2 = self._low_coefficients()
            return (
                c0 * math.log(temperature)
                + c1 * temperature
                + 0.5 * c2 * temperature * temperature
            )

        c0, c1, c2 = self._high_coefficients()
        high_value = (
            c0 * math.log(temperature)
            + c1 * temperature
            + 0.5 * c2 * temperature * temperature
        )
        lc0, lc1, lc2 = self._low_coefficients()
        split = CP_BRANCH_TEMPERATURE
        low_at_split = (
            lc0 * math.log(split)
            + lc1 * split
            + 0.5 * lc2 * split * split
        )
        high_at_split = (
            c0 * math.log(split)
            + c1 * split
            + 0.5 * c2 * split * split
        )
        return high_value + (low_at_split - high_at_split)

    def h(self, T: float) -> float:
        """``integral[T_ref, T] cp dT`` in J/kg, with ``h(T_ref) = 0``."""

        temperature = _strict_positive("T", T)
        self.ensure_cp_above(self.T_ref, temperature, 0.0)
        value = self._h_primitive(temperature) - self._h_primitive(self.T_ref)
        return _finite(
            value,
            stage="caloric.h",
            reason="h_overflow",
            state={"T": temperature, "T_ref": self.T_ref},
        )

    def phi(self, T: float) -> float:
        """``integral[T_ref, T] cp/T dT`` in J/(kg K), ``phi(T_ref)=0``."""

        temperature = _strict_positive("T", T)
        self.ensure_cp_above(self.T_ref, temperature, 0.0)
        value = self._phi_primitive(temperature) - self._phi_primitive(self.T_ref)
        return _finite(
            value,
            stage="caloric.phi",
            reason="phi_overflow",
            state={"T": temperature, "T_ref": self.T_ref},
        )

    def h_difference(self, T_1: float, T_2: float) -> float:
        """Stable signed ``integral[T_1, T_2] cp dT`` in J/kg."""

        t_1 = _strict_positive("T_1", T_1)
        t_2 = _strict_positive("T_2", T_2)
        if t_1 == t_2:
            self.ensure_cp_above(t_1, t_2, 0.0)
            return 0.0
        self.ensure_cp_above(t_1, t_2, 0.0)
        if t_1 > t_2:
            value = -self.h_difference(t_2, t_1)
        elif t_2 <= CP_BRANCH_TEMPERATURE:
            value = _quadratic_h_difference(
                self._low_coefficients(), t_1, t_2
            )
        elif t_1 >= CP_BRANCH_TEMPERATURE:
            value = _quadratic_h_difference(
                self._high_coefficients(), t_1, t_2
            )
        else:
            value = _quadratic_h_difference(
                self._low_coefficients(), t_1, CP_BRANCH_TEMPERATURE
            ) + _quadratic_h_difference(
                self._high_coefficients(), CP_BRANCH_TEMPERATURE, t_2
            )
        return _finite(
            value,
            stage="caloric.h_difference",
            reason="h_overflow",
            state={"T_1": t_1, "T_2": t_2},
        )

    def phi_difference(self, T_1: float, T_2: float) -> float:
        """Stable signed ``integral[T_1, T_2] cp/T dT`` in J/(kg K)."""

        t_1 = _strict_positive("T_1", T_1)
        t_2 = _strict_positive("T_2", T_2)
        if t_1 == t_2:
            self.ensure_cp_above(t_1, t_2, 0.0)
            return 0.0
        self.ensure_cp_above(t_1, t_2, 0.0)
        if t_1 > t_2:
            value = -self.phi_difference(t_2, t_1)
        elif t_2 <= CP_BRANCH_TEMPERATURE:
            value = _quadratic_phi_difference(
                self._low_coefficients(), t_1, t_2
            )
        elif t_1 >= CP_BRANCH_TEMPERATURE:
            value = _quadratic_phi_difference(
                self._high_coefficients(), t_1, t_2
            )
        else:
            value = _quadratic_phi_difference(
                self._low_coefficients(), t_1, CP_BRANCH_TEMPERATURE
            ) + _quadratic_phi_difference(
                self._high_coefficients(), CP_BRANCH_TEMPERATURE, t_2
            )
        return _finite(
            value,
            stage="caloric.phi_difference",
            reason="phi_overflow",
            state={"T_1": t_1, "T_2": t_2},
        )

    def state(
        self,
        T: float,
        p: float,
        mass_flow: float,
        identifier: str | None = None,
    ) -> StationState:
        """Build a ``StationState`` whose ``cp``/``k`` come from this model."""

        temperature = _strict_positive("T", T)
        pressure = _strict_positive("p", p)
        mass = _strict_positive("mass_flow", mass_flow)
        return StationState(
            T_total=temperature,
            p_total=pressure,
            mass_flow=mass,
            cp=self.cp(temperature),
            k=self.k(temperature),
            identifier=identifier,
        )

    def validate_state(
        self,
        state: StationState,
        *,
        rtol: float = _DEFAULT_STATE_RTOL,
        atol_cp: float = _DEFAULT_STATE_CP_ATOL,
        atol_k: float = _DEFAULT_STATE_K_ATOL,
    ) -> StationState:
        """Reject an inlet whose stored ``cp``/``k`` do not match this model."""

        if not isinstance(state, StationState):
            raise TypeError(
                f"state must be a StationState, got {type(state).__name__}"
            )
        expected_cp = self.cp(state.T_total)
        expected_k = self.k(state.T_total)
        if not math.isclose(state.cp, expected_cp, rel_tol=rtol, abs_tol=atol_cp):
            raise IncompatibleStateError(
                "inlet.cp is incompatible with the selected caloric model at "
                f"T={state.T_total!r}: got {state.cp!r}, expected "
                f"{expected_cp!r}"
            )
        if not math.isclose(state.k, expected_k, rel_tol=rtol, abs_tol=atol_k):
            raise IncompatibleStateError(
                "inlet.k is incompatible with the selected caloric model at "
                f"T={state.T_total!r}: got {state.k!r}, expected "
                f"{expected_k!r}"
            )
        return state


@dataclass(frozen=True)
class VariableCpResult:
    """Immutable result of one opt-in variable-``cp`` component.

    ``component`` is ``"compressor"`` or ``"turbine"``.  ``pressure_ratio`` is
    the compressor ratio ``p_out/p_in`` or turbine ratio ``p_in/p_out`` and is
    strictly greater than one.  ``specific_work`` is positive in J/kg (its
    ``work`` property is the same value); ``power = specific_work *
    inlet.mass_flow`` in W.

    ``energy_residual`` is ``h_out - h_in - work`` for a compressor and
    ``h_in - h_out - work`` for a turbine, in J/kg.  ``polytropic_residual`` is
    the entropy integral minus its polytropic target in J/(kg K).  These are
    independent unfiltered closure diagnostics, not acceptance criteria.

    ``model_variant``, ``alpha``, ``reference_temperature``, ``gas_constant``
    and ``validation_status`` record the selected caloric model.  The outlet
    ``StationState`` carries its own model-evaluated ``cp`` and ``k`` at the
    actual outlet temperature; they are deliberately not forced to equal the
    inlet values.
    """

    inlet: StationState
    outlet: StationState
    component: str
    pressure_ratio: float
    eta_poly: float
    specific_work: float
    power: float
    energy_residual: float
    polytropic_residual: float
    model_variant: str
    alpha: float
    reference_temperature: float
    gas_constant: float
    validation_status: str = SOURCE_VALIDATION_STATUS

    def __post_init__(self) -> None:
        if not isinstance(self.inlet, StationState):
            raise TypeError(
                f"inlet must be a StationState, got {type(self.inlet).__name__}"
            )
        if not isinstance(self.outlet, StationState):
            raise TypeError(
                f"outlet must be a StationState, got {type(self.outlet).__name__}"
            )
        if self.component not in ("compressor", "turbine"):
            raise ValueError(
                "component must be 'compressor' or 'turbine', got "
                f"{self.component!r}"
            )

        pressure_ratio = _strict_real("pressure_ratio", self.pressure_ratio)
        eta_poly = _strict_real("eta_poly", self.eta_poly)
        work = _strict_real("specific_work", self.specific_work)
        power = _strict_real("power", self.power)
        energy_residual = _strict_real("energy_residual", self.energy_residual)
        polytropic_residual = _strict_real(
            "polytropic_residual", self.polytropic_residual
        )
        alpha = _strict_alpha(self.alpha)
        reference_temperature = _strict_positive(
            "reference_temperature", self.reference_temperature
        )
        gas_constant = _strict_positive("gas_constant", self.gas_constant)
        if not isinstance(self.model_variant, str) or not self.model_variant.strip():
            raise ValueError(
                f"model_variant must be a non-empty string, got {self.model_variant!r}"
            )
        if (
            not isinstance(self.validation_status, str)
            or not self.validation_status.strip()
        ):
            raise ValueError(
                "validation_status must be a non-empty string, got "
                f"{self.validation_status!r}"
            )

        object.__setattr__(self, "pressure_ratio", pressure_ratio)
        object.__setattr__(self, "eta_poly", eta_poly)
        object.__setattr__(self, "specific_work", work)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "energy_residual", energy_residual)
        object.__setattr__(self, "polytropic_residual", polytropic_residual)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "reference_temperature", reference_temperature)
        object.__setattr__(self, "gas_constant", gas_constant)

        if pressure_ratio <= 1.0:
            raise ValueError(
                f"pressure_ratio must be > 1, got {pressure_ratio!r}"
            )
        if not 0.0 < eta_poly <= 1.0:
            raise ValueError(f"eta_poly must be in (0, 1], got {eta_poly!r}")
        if work <= 0.0:
            raise ValueError(f"specific_work must be > 0, got {work!r}")

        expected_power = work * self.inlet.mass_flow
        if not math.isclose(power, expected_power, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                "power must equal specific_work * inlet.mass_flow, got "
                f"{power!r} and {expected_power!r}"
            )
        if not math.isclose(
            self.outlet.mass_flow,
            self.inlet.mass_flow,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "outlet.mass_flow must equal inlet.mass_flow, got "
                f"{self.outlet.mass_flow!r} and {self.inlet.mass_flow!r}"
            )

        if self.component == "compressor":
            expected_pressure = self.inlet.p_total * pressure_ratio
            if self.outlet.T_total <= self.inlet.T_total:
                raise ValueError(
                    "a variable-cp compressor outlet must be hotter than its inlet"
                )
        else:
            expected_pressure = self.inlet.p_total / pressure_ratio
            if self.outlet.T_total >= self.inlet.T_total:
                raise ValueError(
                    "a variable-cp turbine outlet must be cooler than its inlet"
                )
        if not math.isclose(
            self.outlet.p_total,
            expected_pressure,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "outlet.p_total is inconsistent with pressure_ratio: got "
                f"{self.outlet.p_total!r}, expected {expected_pressure!r}"
            )

    @property
    def work(self) -> float:
        """Alias of ``specific_work`` in J/kg."""

        return self.specific_work

    @property
    def component_kind(self) -> str:
        return self.component


def _validate_component_inputs(
    model: Any,
    inlet: StationState,
    pressure_ratio: Any,
    eta_poly: Any,
    tol: Any,
    max_iter: Any,
) -> tuple[StationState, float, float, float, int]:
    state = model.validate_state(inlet)
    ratio = _strict_real("pressure_ratio", pressure_ratio)
    if ratio <= 1.0:
        raise CaloricModelError(
            f"pressure_ratio must be > 1, got {pressure_ratio!r}"
        )
    eta = _strict_real("eta_poly", eta_poly)
    if not 0.0 < eta <= 1.0:
        raise CaloricModelError(
            f"eta_poly must be in (0, 1], got {eta_poly!r}"
        )
    tolerance = _strict_tolerance(tol)
    iterations = _strict_iterations(max_iter)
    return state, ratio, eta, tolerance, iterations


def _bisect_increasing(
    function: Any,
    lower: float,
    upper: float,
    relative_tol: float,
    max_iter: int,
    *,
    target_scale: float,
    stage: str,
) -> tuple[float, int]:
    """Deterministic bisection for a monotone increasing function.

    The bracket is assumed valid: ``f(lower) <= 0 <= f(upper)``.  Convergence
    is a **relative residual** criterion, ``abs(f) <= relative_tol *
    abs(target_scale)``, not an absolute temperature width.  A root is returned
    only after an evaluated iterate; an iteration limit raises
    :class:`NumericalConvergenceError` with the last bracket and residuals.

    If adjacent representable temperatures are reached, the better evaluated
    endpoint is returned only when its achieved relative residual satisfies the
    caller-requested ``relative_tol``.  Otherwise the request is below
    meaningful floating-point temperature resolution and
    :class:`FloatingResolutionError` is raised.  There is no hidden relaxation
    of the requested tolerance.
    """

    scale = abs(float(target_scale))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"target_scale must be finite and positive, got {target_scale!r}"
        )
    closure = relative_tol * scale

    f_lower = function(lower)
    f_upper = function(upper)
    if not (math.isfinite(f_lower) and math.isfinite(f_upper)):
        raise NumericalConvergenceError(
            "non-finite residual at the initial bracket",
            stage=stage,
            reason="non_finite_initial_residual",
            last_valid_state={
                "bracket_lower_T": lower,
                "bracket_upper_T": upper,
                "residual_lower": f_lower,
                "residual_upper": f_upper,
                "target_scale": scale,
            },
        )
    if f_lower > 0.0 or f_upper < 0.0:
        raise NumericalConvergenceError(
            "invalid initial bracket for the monotone solve",
            stage=stage,
            reason="invalid_initial_bracket",
            last_valid_state={
                "bracket_lower_T": lower,
                "bracket_upper_T": upper,
                "residual_lower": f_lower,
                "residual_upper": f_upper,
                "target_scale": scale,
            },
        )
    if f_lower == 0.0:
        return lower, 0
    if f_upper == 0.0:
        return upper, 0

    last_state = {
        "bracket_lower_T": lower,
        "bracket_upper_T": upper,
        "residual_lower": f_lower,
        "residual_upper": f_upper,
        "target_scale": scale,
        "closure_tolerance": closure,
    }

    for iteration in range(1, max_iter + 1):
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint == lower or midpoint == upper:
            # The bracket is already at floating-point resolution.  Select the
            # better evaluated endpoint and require the caller-requested
            # relative closure; otherwise report the numerical resolution
            # limit without any hidden relaxation.
            root = lower if abs(f_lower) <= abs(f_upper) else upper
            best_residual = f_lower if root == lower else f_upper
            achieved = abs(best_residual) / scale
            resolution_state = {
                "bracket_lower_T": lower,
                "bracket_upper_T": upper,
                "residual_lower": f_lower,
                "residual_upper": f_upper,
                "best_T": root,
                "best_residual": best_residual,
                "achieved_relative_residual": achieved,
                "target_scale": scale,
                "requested_relative_tol": relative_tol,
                "ulp_at_best_T": math.ulp(root),
            }
            if achieved <= relative_tol:
                return root, iteration - 1
            raise FloatingResolutionError(
                "the requested change is below representable temperature "
                f"resolution: best relative residual {achieved!r} exceeds "
                f"requested {relative_tol!r}",
                stage=stage,
                reason="floating_temperature_resolution_limit",
                last_valid_state=resolution_state,
            )

        f_mid = function(midpoint)
        if not math.isfinite(f_mid):
            raise NumericalConvergenceError(
                "non-finite residual during bisection",
                stage=stage,
                reason="non_finite_residual",
                last_valid_state=last_state,
            )
        last_state = {
            "bracket_lower_T": lower,
            "bracket_upper_T": upper,
            "midpoint_T": midpoint,
            "residual_lower": f_lower,
            "residual_midpoint": f_mid,
            "residual_upper": f_upper,
            "target_scale": scale,
            "closure_tolerance": closure,
            "iteration": iteration,
        }

        if abs(f_mid) <= closure:
            return midpoint, iteration
        if f_mid < 0.0:
            lower = midpoint
            f_lower = f_mid
        else:
            upper = midpoint
            f_upper = f_mid

    best_residual = min(abs(f_lower), abs(f_upper))
    last_state = dict(last_state)
    last_state["best_relative_residual"] = best_residual / scale
    raise NumericalConvergenceError(
        "bisection did not reach relative closure "
        f"{relative_tol!r} within {max_iter} iterations",
        stage=stage,
        reason="bisection_iteration_limit",
        last_valid_state=last_state,
    )


def _refine_bracket_before_domain_boundary(
    model: Any,
    fixed_T: float,
    valid_T: float,
    invalid_T: float,
    residual_at: Any,
    max_iter: int,
    *,
    stage: str,
    reason: str,
) -> float:
    """Find a valid bracket endpoint before an invalid-``cp`` trial.

    ``valid_T`` must already have a valid ``cp > R`` interval to ``fixed_T``
    and a negative bracket residual.  ``invalid_T`` is an initial trial that
    exceeded the caloric domain.  The routine bisects between them, rejecting
    only intervals that remain invalid.  It returns a valid endpoint whose
    residual reached zero or above.  If the valid side is exhausted down to
    floating-point resolution without reaching the residual, the target is
    proven unreachable before the genuine ``cp > R`` domain boundary and a
    :class:`CaloricModelDomainError` is raised.  Iteration-limit exhaustion is
    instead a :class:`NumericalConvergenceError`; a trial overshoot is never
    itself treated as an infeasible request.
    """

    last_valid = valid_T
    last_residual = residual_at(valid_T)
    last_domain_error: CaloricModelDomainError | None = None
    current_invalid = invalid_T

    for _ in range(max_iter):
        midpoint = last_valid + 0.5 * (current_invalid - last_valid)
        if midpoint == last_valid or midpoint == current_invalid:
            raise CaloricModelDomainError(
                "the requested target is not reachable before the genuine "
                "cp > R domain boundary; the initial invalid trial was "
                "refined toward the last valid temperature",
                stage=stage,
                reason=reason,
                last_valid_state={
                    "fixed_T": fixed_T,
                    "T_last_valid": last_valid,
                    "residual_at_last_valid": last_residual,
                    "T_invalid_trial": current_invalid,
                    "domain_reason": (
                        None
                        if last_domain_error is None
                        else last_domain_error.reason
                    ),
                    "domain_state": (
                        None
                        if last_domain_error is None
                        else last_domain_error.last_valid_state
                    ),
                },
            )
        try:
            model.ensure_cp_above(fixed_T, midpoint, model.R)
        except CaloricModelDomainError as exc:
            last_domain_error = exc
            current_invalid = midpoint
            continue

        residual = residual_at(midpoint)
        if residual >= 0.0:
            return midpoint
        last_valid = midpoint
        last_residual = residual

    raise NumericalConvergenceError(
        "domain-boundary bracket refinement did not reach representable "
        "resolution within its iteration limit",
        stage=stage,
        reason="domain_boundary_refinement_iteration_limit",
        last_valid_state={
            "fixed_T": fixed_T,
            "T_last_valid": last_valid,
            "residual_at_last_valid": last_residual,
            "T_invalid_trial": current_invalid,
        },
    )


def _result(
    model: Any,
    inlet: StationState,
    outlet: StationState,
    component: str,
    pressure_ratio: float,
    eta_poly: float,
    work: float,
    energy_residual: float,
    polytropic_residual: float,
) -> VariableCpResult:
    power = work * inlet.mass_flow
    if not math.isfinite(power) or power <= 0.0:
        raise CaloricModelDomainError(
            f"{component} power is not finite and positive",
            stage=f"caloric.{component}.power",
            reason="power_underflow" if power == 0.0 else "power_overflow",
            last_valid_state={
                "specific_work": work,
                "mass_flow": inlet.mass_flow,
                "power": power,
            },
        )
    return VariableCpResult(
        inlet=inlet,
        outlet=outlet,
        component=component,
        pressure_ratio=pressure_ratio,
        eta_poly=eta_poly,
        specific_work=work,
        power=power,
        energy_residual=energy_residual,
        polytropic_residual=polytropic_residual,
        model_variant=model.source_variant,
        alpha=model.alpha,
        reference_temperature=model.T_ref,
        gas_constant=model.R,
        validation_status=getattr(
            model, "validation_status", SOURCE_VALIDATION_STATUS
        ),
    )


def compress_variable_cp(
    model: CaloricGasModel,
    inlet: StationState,
    pressure_ratio: float,
    eta_poly: float,
    *,
    tol: float = DEFAULT_SOLVER_TOL,
    max_iter: int = DEFAULT_SOLVER_MAX_ITER,
    identifier: str | None = None,
) -> VariableCpResult:
    """Compress one inlet state with the true-``cp`` caloric model.

    The root satisfies ``integral[T_in, T_out] cp/T dT =
    R*ln(pressure_ratio)/eta_poly``.  Pressure ratio is ``p_out/p_in > 1``.
    ``tol`` is the relative residual closure tolerance; ``max_iter`` bounds the
    deterministic bisection.  A request below representable temperature
    resolution raises :class:`FloatingResolutionError`.  The inlet ``cp``/``k``
    must already match ``model``; use ``model.state(...)`` to construct it.
    The outlet is a new state with the model's actual outlet ``cp``/``k``.
    """

    state, ratio, eta, tolerance, iterations = _validate_component_inputs(
        model, inlet, pressure_ratio, eta_poly, tol, max_iter
    )
    T_in = state.T_total
    log_ratio = math.log1p(ratio - 1.0)
    target_entropy = model.R * log_ratio / eta
    if not math.isfinite(target_entropy) or target_entropy <= 0.0:
        raise CaloricModelDomainError(
            "compressor entropy target is not finite and positive",
            stage="caloric.compressor.target",
            reason="invalid_entropy_target",
            last_valid_state={
                "T_in": T_in,
                "pressure_ratio": ratio,
                "target_entropy": target_entropy,
            },
        )

    T_upper = T_in * 1.1
    if not math.isfinite(T_upper) or T_upper <= T_in:
        raise CaloricModelDomainError(
            "compressor bracket cannot advance above the inlet temperature",
            stage="caloric.compressor.bracket",
            reason="bracket_temperature_not_above_inlet",
            last_valid_state={"T_in": T_in, "T_upper": T_upper},
        )

    def residual_at(T: float) -> float:
        return model.phi_difference(T_in, T) - target_entropy

    bracket_found = False
    T_last_valid = T_in
    for _ in range(MAX_BRACKET_EXPANSIONS):
        try:
            model.ensure_cp_above(T_in, T_upper, model.R)
        except CaloricModelDomainError:
            # The trial overshot the cp > R boundary.  Do not classify the
            # request as invalid yet: refine toward the last valid temperature.
            T_upper = _refine_bracket_before_domain_boundary(
                model,
                T_in,
                T_last_valid,
                T_upper,
                residual_at,
                iterations,
                stage="caloric.compressor.bracket",
                reason="entropy_target_not_reachable_before_cp_domain",
            )
            bracket_found = True
            break

        residual = residual_at(T_upper)
        if residual >= 0.0:
            bracket_found = True
            break
        T_last_valid = T_upper
        new_upper = T_in + 2.0 * (T_upper - T_in)
        if not math.isfinite(new_upper) or new_upper <= T_upper:
            raise CaloricModelDomainError(
                "compressor bracket expansion reached a non-finite limit",
                stage="caloric.compressor.bracket",
                reason="bracket_temperature_overflow",
                last_valid_state={
                    "T_in": T_in,
                    "T_upper": T_upper,
                    "target_entropy": target_entropy,
                    "residual": residual,
                },
            )
        T_upper = new_upper

    if not bracket_found:
        raise NumericalConvergenceError(
            "compressor bracket expansion exhausted without enclosing the "
            "entropy target",
            stage="caloric.compressor.bracket",
            reason="bracket_expansion_exhausted",
            last_valid_state={
                "T_in": T_in,
                "T_upper": T_upper,
                "target_entropy": target_entropy,
            },
        )

    T_out, _ = _bisect_increasing(
        residual_at,
        T_in,
        T_upper,
        tolerance,
        iterations,
        target_scale=target_entropy,
        stage="caloric.compressor.bisection",
    )

    entropy_change = model.phi_difference(T_in, T_out)
    work = model.h_difference(T_in, T_out)
    h_in = model.h(T_in)
    h_out = model.h(T_out)
    energy_residual = (h_out - h_in) - work
    polytropic_residual = entropy_change - target_entropy
    if work <= 0.0:
        raise CaloricModelDomainError(
            "compressor work is not positive at the solved outlet",
            stage="caloric.compressor.work",
            reason="nonpositive_work",
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "work": work,
            },
        )
    p_out = state.p_total * ratio
    if not math.isfinite(p_out):
        raise CaloricModelDomainError(
            "compressor outlet pressure overflowed",
            stage="caloric.compressor.pressure",
            reason="pressure_overflow",
            last_valid_state={
                "p_in": state.p_total,
                "pressure_ratio": ratio,
            },
        )

    outlet = model.state(T_out, p_out, state.mass_flow, identifier=identifier)
    return _result(
        model,
        state,
        outlet,
        "compressor",
        ratio,
        eta,
        work,
        energy_residual,
        polytropic_residual,
    )


def expand_variable_cp(
    model: CaloricGasModel,
    inlet: StationState,
    pressure_ratio: float,
    eta_poly: float,
    *,
    tol: float = DEFAULT_SOLVER_TOL,
    max_iter: int = DEFAULT_SOLVER_MAX_ITER,
    identifier: str | None = None,
) -> VariableCpResult:
    """Expand one inlet state with the true-``cp`` caloric model.

    The root satisfies ``integral[T_out, T_in] cp/T dT =
    eta_poly*R*ln(pressure_ratio)``.  ``pressure_ratio`` is the expansion
    ratio ``p_in/p_out > 1``.  ``tol`` is the relative residual closure
    tolerance; ``max_iter`` bounds the deterministic bisection.  The inlet
    ``cp``/``k`` must match ``model``.
    """

    state, ratio, eta, tolerance, iterations = _validate_component_inputs(
        model, inlet, pressure_ratio, eta_poly, tol, max_iter
    )
    T_in = state.T_total
    log_ratio = math.log1p(ratio - 1.0)
    target_entropy = eta * model.R * log_ratio
    if not math.isfinite(target_entropy) or target_entropy <= 0.0:
        raise CaloricModelDomainError(
            "turbine entropy target is not finite and positive",
            stage="caloric.turbine.target",
            reason="invalid_entropy_target",
            last_valid_state={
                "T_in": T_in,
                "pressure_ratio": ratio,
                "target_entropy": target_entropy,
            },
        )

    T_lower = T_in * 0.9
    if not math.isfinite(T_lower) or T_lower <= 0.0:
        raise CaloricModelDomainError(
            "turbine bracket cannot advance below the inlet toward T > 0",
            stage="caloric.turbine.bracket",
            reason="bracket_temperature_not_positive",
            last_valid_state={"T_in": T_in, "T_lower": T_lower},
        )

    def bracket_residual_at(T: float) -> float:
        return model.phi_difference(T, T_in) - target_entropy

    bracket_found = False
    T_last_valid = T_in
    for _ in range(MAX_BRACKET_EXPANSIONS):
        try:
            model.ensure_cp_above(T_lower, T_in, model.R)
        except CaloricModelDomainError:
            # The trial overshot an invalid cp interval.  Refine toward the
            # last valid temperature before deciding the target is unreachable.
            T_lower = _refine_bracket_before_domain_boundary(
                model,
                T_in,
                T_last_valid,
                T_lower,
                bracket_residual_at,
                iterations,
                stage="caloric.turbine.bracket",
                reason="entropy_target_not_reachable_before_cp_domain",
            )
            bracket_found = True
            break

        residual = bracket_residual_at(T_lower)
        if residual >= 0.0:
            bracket_found = True
            break
        T_last_valid = T_lower
        # Approach T -> 0 geometrically; never add a step that jumps past or
        # through zero.  The bracket length remains meaningful for legitimate
        # deep expansions such as T_out < 0.2*T_in.
        new_lower = T_lower * 0.5
        if (
            not math.isfinite(new_lower)
            or new_lower <= 0.0
            or new_lower >= T_lower
        ):
            raise CaloricModelDomainError(
                "turbine bracket expansion would reach T <= 0; the requested "
                "enthalpy/entropy drop is not available inside the source "
                "model domain",
                stage="caloric.turbine.bracket",
                reason="bracket_reached_nonpositive_temperature",
                last_valid_state={
                    "T_in": T_in,
                    "T_lower": T_lower,
                    "target_entropy": target_entropy,
                    "residual": residual,
                },
            )
        T_lower = new_lower

    if not bracket_found:
        raise NumericalConvergenceError(
            "turbine bracket expansion exhausted without enclosing the "
            "entropy target",
            stage="caloric.turbine.bracket",
            reason="bracket_expansion_exhausted",
            last_valid_state={
                "T_in": T_in,
                "T_lower": T_lower,
                "target_entropy": target_entropy,
            },
        )

    def residual_at(T: float) -> float:
        # Increasing in T: phi(T, T_in) decreases as T rises toward T_in.
        return target_entropy - model.phi_difference(T, T_in)

    T_out, _ = _bisect_increasing(
        residual_at,
        T_lower,
        T_in,
        tolerance,
        iterations,
        target_scale=target_entropy,
        stage="caloric.turbine.bisection",
    )

    entropy_change = model.phi_difference(T_out, T_in)
    work = model.h_difference(T_out, T_in)
    h_in = model.h(T_in)
    h_out = model.h(T_out)
    energy_residual = (h_in - h_out) - work
    polytropic_residual = entropy_change - target_entropy
    if work <= 0.0:
        raise CaloricModelDomainError(
            "turbine work is not positive at the solved outlet",
            stage="caloric.turbine.work",
            reason="nonpositive_work",
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "work": work,
            },
        )
    p_out = state.p_total / ratio
    if not math.isfinite(p_out) or p_out <= 0.0:
        raise CaloricModelDomainError(
            "turbine outlet pressure is not finite and positive",
            stage="caloric.turbine.pressure",
            reason="pressure_nonpositive",
            last_valid_state={
                "p_in": state.p_total,
                "pressure_ratio": ratio,
            },
        )

    outlet = model.state(T_out, p_out, state.mass_flow, identifier=identifier)
    return _result(
        model,
        state,
        outlet,
        "turbine",
        ratio,
        eta,
        work,
        energy_residual,
        polytropic_residual,
    )


def turbine_pressure_ratio_for_work_variable_cp(
    model: CaloricGasModel,
    inlet: StationState,
    work_required: float,
    eta_poly: float,
    *,
    tol: float = DEFAULT_SOLVER_TOL,
    max_iter: int = DEFAULT_SOLVER_MAX_ITER,
    identifier: str | None = None,
) -> VariableCpResult:
    """Inverse true-``cp`` turbine for a prescribed positive specific work.

    First solve ``h(T_in) - h(T_out) = work_required``, then compute
    ``pressure_ratio = exp((phi(T_in)-phi(T_out))/(eta_poly*R))``.  The result
    must have ``pressure_ratio > 1``.  ``tol`` is the relative residual closure
    tolerance.  A work demand that cannot be reached before the temperature
    bracket reaches ``T <= 0`` (or an invalid ``cp`` interval) raises
    :class:`CaloricModelDomainError`, not a solver-iteration failure.  A demand
    below representable temperature resolution raises
    :class:`FloatingResolutionError`.
    """

    state = model.validate_state(inlet)
    work = _strict_positive("work_required", work_required)
    eta = _strict_real("eta_poly", eta_poly)
    if not 0.0 < eta <= 1.0:
        raise CaloricModelError(
            f"eta_poly must be in (0, 1], got {eta_poly!r}"
        )
    tolerance = _strict_tolerance(tol)
    iterations = _strict_iterations(max_iter)

    T_in = state.T_total
    T_lower = T_in * 0.9
    if not math.isfinite(T_lower) or T_lower <= 0.0:
        raise CaloricModelDomainError(
            "inverse turbine bracket cannot advance below the inlet toward "
            "T > 0",
            stage="caloric.turbine_inverse.bracket",
            reason="bracket_temperature_not_positive",
            last_valid_state={"T_in": T_in, "T_lower": T_lower},
        )

    def bracket_residual_at(T: float) -> float:
        return model.h_difference(T, T_in) - work

    bracket_found = False
    T_last_valid = T_in
    for _ in range(MAX_BRACKET_EXPANSIONS):
        try:
            model.ensure_cp_above(T_lower, T_in, model.R)
        except CaloricModelDomainError:
            # Refine an invalid trial toward the last valid temperature before
            # deciding that the prescribed work is physically unreachable.
            T_lower = _refine_bracket_before_domain_boundary(
                model,
                T_in,
                T_last_valid,
                T_lower,
                bracket_residual_at,
                iterations,
                stage="caloric.turbine_inverse.bracket",
                reason="prescribed_work_not_reachable_before_cp_domain",
            )
            bracket_found = True
            break

        available = bracket_residual_at(T_lower)
        if available >= 0.0:
            bracket_found = True
            break
        T_last_valid = T_lower
        # Geometric approach to T -> 0, never jumping through zero.
        new_lower = T_lower * 0.5
        if (
            not math.isfinite(new_lower)
            or new_lower <= 0.0
            or new_lower >= T_lower
        ):
            raise CaloricModelDomainError(
                "prescribed turbine work is not available inside the source "
                "model domain: the temperature bracket would reach T <= 0",
                stage="caloric.turbine_inverse.bracket",
                reason="prescribed_work_not_reachable",
                last_valid_state={
                    "T_in": T_in,
                    "T_lower": T_lower,
                    "work_required": work,
                    "available_work": available,
                },
            )
        T_lower = new_lower

    if not bracket_found:
        # Geometric halving reached the bounded expansion cap without finding
        # the prescribed enthalpy drop.  This is a physically unavailable-work
        # domain outcome for the inverse problem, not a bisection failure.
        raise CaloricModelDomainError(
            "prescribed turbine work is not available inside the source model "
            "domain after bounded geometric bracket expansion",
            stage="caloric.turbine_inverse.bracket",
            reason="prescribed_work_not_reachable",
            last_valid_state={
                "T_in": T_in,
                "T_lower": T_lower,
                "work_required": work,
                "available_work": available,
            },
        )

    def residual_at(T: float) -> float:
        # Increasing in T: h_difference(T, T_in) decreases as T rises.
        return work - model.h_difference(T, T_in)

    T_out, _ = _bisect_increasing(
        residual_at,
        T_lower,
        T_in,
        tolerance,
        iterations,
        target_scale=work,
        stage="caloric.turbine_inverse.bisection",
    )

    work_solved = model.h_difference(T_out, T_in)
    phi_drop = model.phi_difference(T_out, T_in)
    # Both factors are individually finite and positive, but their product can
    # underflow to zero (for example eta_poly=5e-324 and R=1e-12).  Report the
    # unusable product explicitly instead of dividing by it.
    denominator = eta * model.R
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CaloricModelDomainError(
            "inverse turbine pressure-ratio denominator eta_poly*R is not "
            "finite and positive",
            stage="caloric.turbine_inverse.pressure",
            reason=(
                "pressure_denominator_underflow"
                if denominator == 0.0
                else "invalid_pressure_denominator"
            ),
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "eta_poly": eta,
                "R": model.R,
                "denominator": denominator,
                "phi_drop": phi_drop,
            },
        )
    exponent = phi_drop / denominator
    if not math.isfinite(exponent):
        raise CaloricModelDomainError(
            "inverse turbine pressure-ratio exponent overflowed",
            stage="caloric.turbine_inverse.pressure",
            reason="pressure_ratio_overflow",
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "eta_poly": eta,
                "R": model.R,
                "denominator": denominator,
                "phi_drop": phi_drop,
                "exponent": exponent,
            },
        )
    try:
        ratio = 1.0 + math.expm1(exponent)
    except OverflowError as exc:
        raise CaloricModelDomainError(
            "inverse turbine pressure ratio overflowed",
            stage="caloric.turbine_inverse.pressure",
            reason="pressure_ratio_overflow",
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "eta_poly": eta,
                "R": model.R,
                "denominator": denominator,
                "phi_drop": phi_drop,
                "exponent": exponent,
            },
        ) from exc
    if not math.isfinite(ratio) or ratio <= 1.0:
        raise CaloricModelDomainError(
            "prescribed work does not yield a finite pressure ratio > 1",
            stage="caloric.turbine_inverse.pressure",
            reason="pressure_ratio_not_above_one",
            last_valid_state={
                "T_in": T_in,
                "T_out": T_out,
                "eta_poly": eta,
                "R": model.R,
                "denominator": denominator,
                "phi_drop": phi_drop,
                "exponent": exponent,
                "pressure_ratio": ratio,
            },
        )

    p_out = state.p_total / ratio
    if not math.isfinite(p_out) or p_out <= 0.0:
        raise CaloricModelDomainError(
            "inverse turbine outlet pressure is not finite and positive",
            stage="caloric.turbine_inverse.pressure",
            reason="pressure_nonpositive",
            last_valid_state={
                "p_in": state.p_total,
                "pressure_ratio": ratio,
            },
        )

    h_in = model.h(T_in)
    h_out = model.h(T_out)
    energy_residual = (h_in - h_out) - work_solved
    polytropic_residual = phi_drop - eta * model.R * math.log1p(ratio - 1.0)
    outlet = model.state(T_out, p_out, state.mass_flow, identifier=identifier)
    return _result(
        model,
        state,
        outlet,
        "turbine",
        ratio,
        eta,
        work_solved,
        energy_residual,
        polytropic_residual,
    )


__all__ = [
    "CP_BRANCH_TEMPERATURE",
    "CaloricGasModel",
    "CaloricModelDomainError",
    "CaloricModelError",
    "DEFAULT_SOLVER_MAX_ITER",
    "DEFAULT_SOLVER_TOL",
    "FloatingResolutionError",
    "IncompatibleStateError",
    "MAX_BRACKET_EXPANSIONS",
    "SOURCE_CP_UNIT_FACTOR",
    "SOURCE_REFERENCE",
    "SOURCE_VALIDATION_STATUS",
    "SOURCE_VARIANT",
    "VariableCpResult",
    "compress_variable_cp",
    "expand_variable_cp",
    "turbine_pressure_ratio_for_work_variable_cp",
]
