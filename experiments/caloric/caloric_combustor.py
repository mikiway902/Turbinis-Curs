"""Opt-in uncooled lean true-``cp`` integral combustor closure.

STATUS: ``NOT_PHYSICALLY_VALIDATED``.  This module is an independent 0D
energy-balance closure built on the opt-in source true-``cp`` model in
:mod:`experiments.caloric.caloric`.  It does not rewire the legacy combustor, the
legacy mean-``cp`` cycle, or any saved RAW fixture, and it is not a
thermochemical validation.

Energy equation
===============
For one kilogram of air, with fuel/air mass ratio ``f`` and excess-air ratio
``alpha = 1/(f*l0)`` (``alpha = +inf`` at ``f = 0``):

.. code-block:: text

    r(f) = (1 + f) * h_g(T_g, alpha(f)) - h_air(T_c)
           - f * (eta_combustion * Q_lhv + h_fuel)

where

- ``h_g`` and ``h_air`` are ``integral[T_ref, T] cp_true dT`` from the
  :class:`experiments.caloric.caloric.CaloricGasModel` for the product mixture and
  dry air respectively;
- ``T_c`` is the inlet total temperature (never overridden by this module);
- ``T_g`` is the caller-specified target combustor-exit total temperature;
- ``h_fuel`` is the caller's fuel sensible enthalpy per kilogram of fuel,
  referenced to the same explicit ``T_ref``;
- ``eta_combustion`` applies only to the chemical lower-heating-value term,
  not to fuel sensible enthalpy.

There is no cooling, bleed, leakage, pressure-loss map, composition solve, or
rich-mixture extrapolation.  The lean bracket is ``0 <= f <= 1/l0`` so that
``alpha >= 1``, the visibly supported source domain.  ``R`` and ``T_ref`` are
caller-explicit and fixed for all gas states in one closure.

The root is found by deterministic bounded bisection.  Convergence is a
relative residual test against the largest evaluated energy term, not an
arbitrary absolute joule threshold.  A request below representable fuel/air
resolution raises :class:`experiments.caloric.caloric.FloatingResolutionError`; an
iteration limit raises
:class:`turbine_engine.diagnostics.NumericalConvergenceError`.  No last
iterate is returned as if it had converged and no tolerance is relaxed.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, NoReturn

from turbine_engine.architecture import StationState
from experiments.caloric.caloric import (
    DEFAULT_SOLVER_MAX_ITER,
    DEFAULT_SOLVER_TOL,
    CaloricGasModel,
    CaloricModelDomainError,
    CaloricModelError,
    FloatingResolutionError,
)
from turbine_engine.diagnostics import (
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
)

#: Unique identifier of this closure variant.
MODEL_VARIANT = "uncooled_lean_true_cp_integral_combustor_v1"

#: Explicit evidence status; this is not a physical validation.
VALIDATION_STATUS = "NOT_PHYSICALLY_VALIDATED"

#: Fixed assumptions that apply to every result.
LIMITATIONS: tuple[str, ...] = (
    VALIDATION_STATUS,
    "common_R_and_composition_approximation",
    "fuel_sensible_enthalpy_referenced_to_caller_T_ref",
    "no_maps_no_cooling_no_bleed_no_leak",
)

class CaloricCombustorError(CaloricModelError):
    """Invalid combustor input or an unusable lean closure request."""


def _strict_real(name: str, value: Any) -> float:
    if isinstance(value, (bool, str, bytes)) or not isinstance(value, numbers.Real):
        raise CaloricModelError(f"{name} must be a real number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CaloricModelError(
            f"{name} must be a real number, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise CaloricModelError(f"{name} must be finite, got {value!r}")
    return result


def _strict_positive(name: str, value: Any) -> float:
    result = _strict_real(name, value)
    if result <= 0.0:
        raise CaloricModelError(f"{name} must be > 0, got {value!r}")
    return result


def _strict_unit_interval(name: str, value: Any) -> float:
    result = _strict_real(name, value)
    if not 0.0 < result <= 1.0:
        raise CaloricModelError(f"{name} must be in (0, 1], got {value!r}")
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
    return _strict_positive("tol", value)


def _finite(value: float, *, stage: str, reason: str, state: dict[str, Any]) -> float:
    if not math.isfinite(value):
        raise CaloricModelDomainError(
            f"non-finite or unusable numeric result in {stage}: {value!r}",
            stage=stage,
            reason=reason,
            last_valid_state=state,
        )
    return value


@dataclass(frozen=True)
class CaloricCombustorSpec:
    """Caller-explicit combustor physics for one lean closure.

    Every field is required; there are no physical defaults.  ``totalT`` is the
    target combustor-exit total temperature [K].  ``pressure_recovery`` and
    ``eta_combustion`` are in ``(0, 1]``.  ``l0`` is the stoichiometric air/fuel
    mass ratio [kg air/kg fuel], ``Q_lhv_J_per_kg`` is the fuel lower heating
    value, and ``fuel_sensible_enthalpy_J_per_kg`` is the fuel sensible
    enthalpy referenced to the same explicit ``T_ref`` used by the closure.

    A negative fuel sensible enthalpy is allowed, but
    ``eta_combustion * Q_lhv + fuel_sensible_enthalpy`` must be finite and
    strictly positive; otherwise no fuel-sourced heat would remain.
    """

    totalT: float
    pressure_recovery: float
    l0: float
    Q_lhv_J_per_kg: float
    eta_combustion: float
    fuel_sensible_enthalpy_J_per_kg: float

    def __post_init__(self) -> None:
        totalT = _strict_positive("totalT", self.totalT)
        pressure_recovery = _strict_unit_interval(
            "pressure_recovery", self.pressure_recovery
        )
        l0 = _strict_positive("l0", self.l0)
        Q_lhv = _strict_positive("Q_lhv_J_per_kg", self.Q_lhv_J_per_kg)
        eta = _strict_unit_interval("eta_combustion", self.eta_combustion)
        h_fuel = _strict_real(
            "fuel_sensible_enthalpy_J_per_kg",
            self.fuel_sensible_enthalpy_J_per_kg,
        )
        effective_fuel_heat = eta * Q_lhv + h_fuel
        if not math.isfinite(effective_fuel_heat) or effective_fuel_heat <= 0.0:
            raise CaloricModelDomainError(
                "eta_combustion*Q_lhv + fuel_sensible_enthalpy must be finite "
                "and strictly positive",
                stage="caloric.combustor.spec",
                reason="nonpositive_effective_fuel_heat",
                last_valid_state={
                    "eta_combustion": eta,
                    "Q_lhv_J_per_kg": Q_lhv,
                    "fuel_sensible_enthalpy_J_per_kg": h_fuel,
                    "effective_fuel_heat": effective_fuel_heat,
                },
            )
        lean_f_limit = _lean_f_limit(l0)
        if not math.isfinite(lean_f_limit):
            raise CaloricModelDomainError(
                "the lean fuel/air limit 1/l0 is not finite",
                stage="caloric.combustor.spec",
                reason="lean_f_limit_overflow",
                last_valid_state={"l0": l0, "lean_f_limit": lean_f_limit},
            )

        object.__setattr__(self, "totalT", totalT)
        object.__setattr__(self, "pressure_recovery", pressure_recovery)
        object.__setattr__(self, "l0", l0)
        object.__setattr__(self, "Q_lhv_J_per_kg", Q_lhv)
        object.__setattr__(self, "eta_combustion", eta)
        object.__setattr__(
            self, "fuel_sensible_enthalpy_J_per_kg", h_fuel
        )

    @property
    def lean_f_limit(self) -> float:
        """Stoichiometric fuel/air mass ratio ``1/l0``."""

        return _lean_f_limit(self.l0)

    @property
    def target_total_T(self) -> float:
        """Alias of the target outlet total temperature ``totalT``."""

        return self.totalT


def _lean_f_limit(l0: float) -> float:
    try:
        return 1.0 / l0
    except OverflowError:
        return math.inf


@dataclass(frozen=True)
class CaloricCombustorEnthalpies:
    """Measured specific enthalpy terms for one closure, in J/kg air.

    ``gas_product_J_per_kg`` and ``air_inlet_J_per_kg`` are per kilogram of
    their own gas.  The remaining terms are already per kilogram of air.  This
    keeps effective chemical heat and fuel sensible heat separately visible.
    """

    gas_product_J_per_kg: float
    air_inlet_J_per_kg: float
    weighted_gas_product_J_per_kg_air: float
    effective_chemical_J_per_kg_air: float
    fuel_sensible_J_per_kg_air: float

    def __post_init__(self) -> None:
        for name in (
            "gas_product_J_per_kg",
            "air_inlet_J_per_kg",
            "weighted_gas_product_J_per_kg_air",
            "effective_chemical_J_per_kg_air",
            "fuel_sensible_J_per_kg_air",
        ):
            object.__setattr__(
                self, name, _strict_real(name, getattr(self, name))
            )


@dataclass(frozen=True)
class CaloricCombustorResult:
    """Immutable result of one independent lean true-``cp`` combustor closure.

    ``f`` is kilograms of fuel per kilogram of air and ``alpha = 1/(f*l0)``
    (``+inf`` only at ``f = 0``).  ``mfuel = f * inlet.mass_flow`` and
    ``mgas = (1 + f) * inlet.mass_flow``.  ``energy_residual_specific_...`` and
    ``energy_residual_absolute_W`` are measured from the stored evaluated terms;
    they are not recomputed or forced to zero.  ``mass_residual_kg_per_s`` is
    likewise the measured ``inlet.mass_flow + mfuel - mgas``.

    ``spec`` is stored unchanged for provenance.  The result intentionally has
    no complete custom serializer; a future adapter can build one from these
    explicit fields.
    """

    inlet: StationState
    outlet: StationState
    f: float
    alpha: float
    gas_model: CaloricGasModel
    mfuel: float
    mgas: float
    enthalpies: CaloricCombustorEnthalpies
    h_fuel: float
    gross_chemical_W: float
    effective_chemical_W: float
    fuel_sensible_W: float
    energy_residual_specific_J_per_kg_air: float
    energy_residual_absolute_W: float
    mass_residual_kg_per_s: float
    iterations: int
    relative_residual: float
    reference_T: float
    R: float
    model_variant: str
    limitations: tuple[str, ...]
    spec: CaloricCombustorSpec

    def __post_init__(self) -> None:
        if not isinstance(self.inlet, StationState):
            raise TypeError(
                f"inlet must be a StationState, got {type(self.inlet).__name__}"
            )
        if not isinstance(self.outlet, StationState):
            raise TypeError(
                f"outlet must be a StationState, got {type(self.outlet).__name__}"
            )
        if not isinstance(self.gas_model, CaloricGasModel):
            raise TypeError(
                "gas_model must be a CaloricGasModel, got "
                f"{type(self.gas_model).__name__}"
            )
        if not isinstance(self.enthalpies, CaloricCombustorEnthalpies):
            raise TypeError(
                "enthalpies must be CaloricCombustorEnthalpies, got "
                f"{type(self.enthalpies).__name__}"
            )
        if not isinstance(self.spec, CaloricCombustorSpec):
            raise TypeError(
                "spec must be a CaloricCombustorSpec, got "
                f"{type(self.spec).__name__}"
            )

        f = _strict_real("f", self.f)
        if f < 0.0 or f > self.spec.lean_f_limit:
            raise ValueError(
                f"f must be in [0, 1/l0], got {f!r} and "
                f"1/l0={self.spec.lean_f_limit!r}"
            )
        alpha = self.alpha
        if f == 0.0:
            if alpha != math.inf:
                raise ValueError("alpha must be +inf when f is zero")
        else:
            alpha = _strict_real("alpha", alpha)
            if alpha < 1.0:
                raise ValueError(f"alpha must be >= 1, got {alpha!r}")

        mfuel = _strict_real("mfuel", self.mfuel)
        mgas = _strict_positive("mgas", self.mgas)
        if mfuel < 0.0:
            raise ValueError(f"mfuel must be >= 0, got {mfuel!r}")
        h_fuel = _strict_real("h_fuel", self.h_fuel)
        gross = _strict_real("gross_chemical_W", self.gross_chemical_W)
        effective = _strict_real(
            "effective_chemical_W", self.effective_chemical_W
        )
        sensible = _strict_real("fuel_sensible_W", self.fuel_sensible_W)
        specific = _strict_real(
            "energy_residual_specific_J_per_kg_air",
            self.energy_residual_specific_J_per_kg_air,
        )
        absolute = _strict_real(
            "energy_residual_absolute_W", self.energy_residual_absolute_W
        )
        mass_residual = _strict_real(
            "mass_residual_kg_per_s", self.mass_residual_kg_per_s
        )
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 0
        ):
            raise ValueError(
                f"iterations must be a non-negative integer, got {self.iterations!r}"
            )
        relative = _strict_real("relative_residual", self.relative_residual)
        if relative < 0.0:
            raise ValueError(
                f"relative_residual must be >= 0, got {relative!r}"
            )
        reference_T = _strict_positive("reference_T", self.reference_T)
        gas_constant = _strict_positive("R", self.R)
        if not isinstance(self.model_variant, str) or not self.model_variant.strip():
            raise ValueError(
                "model_variant must be a non-empty string, got "
                f"{self.model_variant!r}"
            )
        if not isinstance(self.limitations, tuple) or not self.limitations:
            raise ValueError("limitations must be a non-empty tuple")
        for item in self.limitations:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"limitations entries must be non-empty strings, got {item!r}"
                )
        if VALIDATION_STATUS not in self.limitations:
            raise ValueError(
                f"limitations must include {VALIDATION_STATUS!r}"
            )

        object.__setattr__(self, "f", f)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "mfuel", mfuel)
        object.__setattr__(self, "mgas", mgas)
        object.__setattr__(self, "h_fuel", h_fuel)
        object.__setattr__(self, "gross_chemical_W", gross)
        object.__setattr__(self, "effective_chemical_W", effective)
        object.__setattr__(self, "fuel_sensible_W", sensible)
        object.__setattr__(
            self, "energy_residual_specific_J_per_kg_air", specific
        )
        object.__setattr__(
            self, "energy_residual_absolute_W", absolute
        )
        object.__setattr__(self, "mass_residual_kg_per_s", mass_residual)
        object.__setattr__(self, "relative_residual", relative)
        object.__setattr__(self, "reference_T", reference_T)
        object.__setattr__(self, "R", gas_constant)

        # Measured identities, not a physical PASS criterion and not a
        # replacement of the residual by zero.
        if not math.isclose(
            mfuel,
            f * self.inlet.mass_flow,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("mfuel must equal f * inlet.mass_flow")
        if not math.isclose(
            mgas,
            (1.0 + f) * self.inlet.mass_flow,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError("mgas must equal (1+f) * inlet.mass_flow")
        if not math.isclose(
            absolute,
            specific * self.inlet.mass_flow,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "energy_residual_absolute_W must equal specific_residual * "
                "inlet.mass_flow"
            )
        expected_mass_residual = (
            self.inlet.mass_flow + mfuel
        ) - mgas
        if not math.isclose(
            mass_residual,
            expected_mass_residual,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                "mass_residual_kg_per_s must be the measured inlet+fuel-gas "
                "difference"
            )


@dataclass(frozen=True)
class _Evaluation:
    f: float
    alpha: float
    gas_model: CaloricGasModel
    gas_product_J_per_kg: float
    weighted_gas_product_J_per_kg_air: float
    air_inlet_J_per_kg: float
    effective_chemical_J_per_kg_air: float
    fuel_sensible_J_per_kg_air: float
    residual: float
    scale: float
    cp_outlet: float
    k_outlet: float

    @property
    def relative_residual(self) -> float:
        return abs(self.residual) / self.scale


def _alpha_from_fraction(f: float, l0: float) -> float:
    """Return ``alpha = 1/(f*l0)`` with ``+inf`` at ``f = 0``.

    The reciprocal's denominator and result are checked explicitly so an
    underflow/overflow becomes a typed domain state instead of an
    unstructured ``ZeroDivisionError`` or a silent air limit.
    """

    if f == 0.0:
        return math.inf
    scaled = f * l0
    if scaled <= 0.0 or not math.isfinite(scaled):
        reason = (
            "alpha_denominator_underflow"
            if scaled <= 0.0
            else "alpha_denominator_overflow"
        )
        raise CaloricModelDomainError(
            "the reciprocal excess-air denominator f*l0 is not finite and "
            "positive",
            stage="caloric.combustor.alpha",
            reason=reason,
            last_valid_state={"f": f, "l0": l0, "f_l0": scaled},
        )
    if scaled >= 1.0:
        # The bracket endpoint is exactly stoichiometric; guard the reciprocal
        # against a one-ulp product above one.
        return 1.0
    try:
        alpha = 1.0 / scaled
    except OverflowError as exc:
        raise CaloricModelDomainError(
            "the reciprocal excess-air ratio overflowed",
            stage="caloric.combustor.alpha",
            reason="alpha_reciprocal_overflow",
            last_valid_state={"f": f, "l0": l0, "f_l0": scaled},
        ) from exc
    if not math.isfinite(alpha):
        raise CaloricModelDomainError(
            "the reciprocal excess-air ratio is not finite",
            stage="caloric.combustor.alpha",
            reason="alpha_reciprocal_overflow",
            last_valid_state={"f": f, "l0": l0, "f_l0": scaled, "alpha": alpha},
        )
    if alpha <= 0.0:
        raise CaloricModelDomainError(
            "the reciprocal excess-air ratio underflowed",
            stage="caloric.combustor.alpha",
            reason="alpha_reciprocal_underflow",
            last_valid_state={"f": f, "l0": l0, "f_l0": scaled, "alpha": alpha},
        )
    return alpha


def _no_lean_root(
    spec: CaloricCombustorSpec,
    inlet: StationState,
    f_upper: float,
    lower: _Evaluation,
    upper: _Evaluation | None,
) -> NoReturn:
    state = {
        "inlet_T_total": inlet.T_total,
        "inlet_p_total": inlet.p_total,
        "inlet_mass_flow": inlet.mass_flow,
        "spec": {
            "totalT": spec.totalT,
            "pressure_recovery": spec.pressure_recovery,
            "l0": spec.l0,
            "Q_lhv_J_per_kg": spec.Q_lhv_J_per_kg,
            "eta_combustion": spec.eta_combustion,
            "fuel_sensible_enthalpy_J_per_kg": (
                spec.fuel_sensible_enthalpy_J_per_kg
            ),
        },
        "evaluated_endpoints": {
            "f_lower": lower.f,
            "alpha_lower": lower.alpha,
            "residual_lower": lower.residual,
            "relative_residual_lower": lower.relative_residual,
            "f_upper": f_upper,
            "alpha_upper": None if upper is None else upper.alpha,
            "residual_upper": None if upper is None else upper.residual,
            "relative_residual_upper": (
                None if upper is None else upper.relative_residual
            ),
        },
    }
    raise CaloricModelDomainError(
        "no fuel/air ratio in the lean bracket [0, 1/l0] closes the target "
        "combustor-exit temperature; rich or extrapolated closure was not "
        "invented",
        stage="caloric.combustor",
        reason="no_lean_fuel_root",
        last_valid_state=state,
    )


def _bisect_lean_root(
    evaluate: Callable[[float], _Evaluation],
    lower: _Evaluation,
    upper: _Evaluation,
    tolerance: float,
    max_iter: int,
) -> tuple[_Evaluation, int]:
    """Deterministic bisection for a sign-changing lean residual."""

    if lower.residual == 0.0 or abs(lower.residual) <= tolerance * lower.scale:
        return lower, 0
    if upper.residual == 0.0 or abs(upper.residual) <= tolerance * upper.scale:
        return upper, 0
    if (lower.residual > 0.0) == (upper.residual > 0.0):
        raise ValueError("bisection bracket does not change sign")

    last_state = {
        "bracket_lower_f": lower.f,
        "bracket_upper_f": upper.f,
        "residual_lower": lower.residual,
        "residual_upper": upper.residual,
        "scale_lower": lower.scale,
        "scale_upper": upper.scale,
    }

    for iteration in range(1, max_iter + 1):
        midpoint = lower.f + 0.5 * (upper.f - lower.f)
        if midpoint == lower.f or midpoint == upper.f:
            best = (
                lower
                if lower.relative_residual <= upper.relative_residual
                else upper
            )
            achieved = best.relative_residual
            resolution_state = {
                "bracket_lower_f": lower.f,
                "bracket_upper_f": upper.f,
                "residual_lower": lower.residual,
                "residual_upper": upper.residual,
                "best_f": best.f,
                "best_residual": best.residual,
                "achieved_relative_residual": achieved,
                "requested_relative_tol": tolerance,
                "ulp_at_best_f": math.ulp(best.f),
            }
            if achieved <= tolerance:
                return best, iteration - 1
            raise FloatingResolutionError(
                "the requested fuel/air change is below representable "
                f"resolution: best relative residual {achieved!r} exceeds "
                f"requested {tolerance!r}",
                stage="caloric.combustor.bisection",
                reason="floating_fuel_fraction_resolution_limit",
                last_valid_state=resolution_state,
            )

        midpoint_eval = evaluate(midpoint)
        last_state = {
            "bracket_lower_f": lower.f,
            "bracket_upper_f": upper.f,
            "midpoint_f": midpoint,
            "residual_lower": lower.residual,
            "residual_midpoint": midpoint_eval.residual,
            "residual_upper": upper.residual,
            "scale_midpoint": midpoint_eval.scale,
            "requested_relative_tol": tolerance,
            "iteration": iteration,
        }
        if (
            midpoint_eval.residual == 0.0
            or abs(midpoint_eval.residual)
            <= tolerance * midpoint_eval.scale
        ):
            return midpoint_eval, iteration
        if (midpoint_eval.residual > 0.0) == (lower.residual > 0.0):
            lower = midpoint_eval
        else:
            upper = midpoint_eval

    best = lower if lower.relative_residual <= upper.relative_residual else upper
    last_state = dict(last_state)
    last_state["best_relative_residual"] = best.relative_residual
    raise NumericalConvergenceError(
        "combustor bisection did not reach relative closure "
        f"{tolerance!r} within {max_iter} iterations",
        stage="caloric.combustor.bisection",
        reason="bisection_iteration_limit",
        last_valid_state=last_state,
    )


def close_caloric_combustor(
    inlet: StationState,
    spec: CaloricCombustorSpec,
    *,
    R: float,
    T_ref: float,
    tol: float = DEFAULT_SOLVER_TOL,
    max_iter: int = DEFAULT_SOLVER_MAX_ITER,
) -> CaloricCombustorResult:
    """Close one independent uncooled lean true-``cp`` combustor.

    Parameters
    ----------
    inlet:
        Station state with total temperature ``T_c``, total pressure ``p_c`` and
        mass flow.  Its stored ``cp`` and ``k`` must match the dry-air
        :class:`CaloricGasModel` built from ``R`` and ``T_ref``.  The inlet
        temperature is used as supplied and cannot be overridden.
    spec:
        Required physical inputs.  ``spec.totalT`` is the target outlet total
        temperature and must be strictly above ``inlet.T_total``.
    R:
        Caller-explicit gas constant [J/(kg K)] shared by air and products.
    T_ref:
        Caller-explicit enthalpy reference temperature [K].
    tol:
        Relative energy-residual closure tolerance.
    max_iter:
        Maximum deterministic bisection iterations.

    The returned fuel/air ratio is independent of ``inlet.mass_flow``; changing
    the flow scales ``mfuel``, ``mgas`` and absolute powers but not ``f``.
    """

    if not isinstance(spec, CaloricCombustorSpec):
        raise TypeError(
            f"spec must be a CaloricCombustorSpec, got {type(spec).__name__}"
        )
    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
        )

    air_model = CaloricGasModel(R, math.inf, T_ref)
    state = air_model.validate_state(inlet)
    tolerance = _strict_tolerance(tol)
    iterations_limit = _strict_iterations(max_iter)

    if spec.totalT <= state.T_total:
        raise CaloricModelDomainError(
            "the target combustor-exit total temperature must be strictly "
            "above the inlet total temperature",
            stage="caloric.combustor.domain",
            reason="target_temperature_not_above_inlet",
            last_valid_state={
                "inlet_T_total": state.T_total,
                "target_totalT": spec.totalT,
            },
        )

    # Validate the enthalpy intervals explicitly before evaluating any fuel
    # trial.  cp must be positive over the whole integrated interval.
    air_model.ensure_cp_above(air_model.T_ref, state.T_total, 0.0)
    air_model.ensure_cp_above(air_model.T_ref, spec.totalT, 0.0)
    h_air = air_model.h(state.T_total)
    h_fuel = spec.fuel_sensible_enthalpy_J_per_kg
    heat_per_fuel = spec.eta_combustion * spec.Q_lhv_J_per_kg

    def evaluate(f: float) -> _Evaluation:
        alpha = _alpha_from_fraction(f, spec.l0)
        gas_model = CaloricGasModel(R, alpha, T_ref)
        # The outlet cp/k are actual model values at the target temperature;
        # k also enforces cp > R.
        cp_outlet = gas_model.cp(spec.totalT)
        k_outlet = gas_model.k(spec.totalT)
        gas_h = gas_model.h(spec.totalT)
        weighted_gas = (1.0 + f) * gas_h
        effective_chemical = f * heat_per_fuel
        fuel_sensible = f * h_fuel
        residual = (
            weighted_gas - h_air - effective_chemical - fuel_sensible
        )
        scale = max(
            abs(weighted_gas),
            abs(h_air),
            abs(effective_chemical),
            abs(fuel_sensible),
        )
        residual = _finite(
            residual,
            stage="caloric.combustor.energy",
            reason="energy_residual_overflow",
            state={
                "f": f,
                "weighted_gas": weighted_gas,
                "h_air": h_air,
                "effective_chemical": effective_chemical,
                "fuel_sensible": fuel_sensible,
            },
        )
        scale = _finite(
            scale,
            stage="caloric.combustor.energy",
            reason="energy_scale_overflow",
            state={
                "f": f,
                "weighted_gas": weighted_gas,
                "h_air": h_air,
                "effective_chemical": effective_chemical,
                "fuel_sensible": fuel_sensible,
            },
        )
        if scale <= 0.0:
            raise CaloricModelDomainError(
                "the combustor energy scale is not positive",
                stage="caloric.combustor.energy",
                reason="energy_scale_nonpositive",
                last_valid_state={"f": f, "scale": scale},
            )
        return _Evaluation(
            f=f,
            alpha=alpha,
            gas_model=gas_model,
            gas_product_J_per_kg=gas_h,
            weighted_gas_product_J_per_kg_air=weighted_gas,
            air_inlet_J_per_kg=h_air,
            effective_chemical_J_per_kg_air=effective_chemical,
            fuel_sensible_J_per_kg_air=fuel_sensible,
            residual=residual,
            scale=scale,
            cp_outlet=cp_outlet,
            k_outlet=k_outlet,
        )

    lower = evaluate(0.0)
    f_upper = spec.lean_f_limit
    if not math.isfinite(f_upper) or f_upper <= 0.0:
        _no_lean_root(spec, state, f_upper, lower, None)
    upper = evaluate(f_upper)
    if (
        lower.residual != 0.0
        and upper.residual != 0.0
        and (lower.residual > 0.0) == (upper.residual > 0.0)
    ):
        _no_lean_root(spec, state, f_upper, lower, upper)

    root, iterations = _bisect_lean_root(
        evaluate,
        lower,
        upper,
        tolerance,
        iterations_limit,
    )

    mfuel = _finite(
        root.f * state.mass_flow,
        stage="caloric.combustor.mass",
        reason="fuel_flow_overflow",
        state={"f": root.f, "mass_flow": state.mass_flow},
    )
    mgas = _finite(
        (1.0 + root.f) * state.mass_flow,
        stage="caloric.combustor.mass",
        reason="gas_flow_overflow",
        state={"f": root.f, "mass_flow": state.mass_flow},
    )
    if mfuel < 0.0 or mgas <= 0.0:
        raise CaloricModelDomainError(
            "combustor mass flows must be non-negative and positive",
            stage="caloric.combustor.mass",
            reason="invalid_mass_flow",
            last_valid_state={"mfuel": mfuel, "mgas": mgas},
        )

    pressure_out = _finite(
        state.p_total * spec.pressure_recovery,
        stage="caloric.combustor.pressure",
        reason="pressure_overflow",
        state={
            "p_in": state.p_total,
            "pressure_recovery": spec.pressure_recovery,
        },
    )
    if pressure_out <= 0.0:
        raise CaloricModelDomainError(
            "combustor outlet pressure is not finite and positive",
            stage="caloric.combustor.pressure",
            reason="pressure_underflow",
            last_valid_state={
                "p_in": state.p_total,
                "pressure_recovery": spec.pressure_recovery,
                "p_out": pressure_out,
            },
        )

    gross_chemical = _finite(
        mfuel * spec.Q_lhv_J_per_kg,
        stage="caloric.combustor.power",
        reason="gross_chemical_power_overflow",
        state={
            "mfuel": mfuel,
            "Q_lhv_J_per_kg": spec.Q_lhv_J_per_kg,
        },
    )
    effective_chemical = _finite(
        mfuel * heat_per_fuel,
        stage="caloric.combustor.power",
        reason="effective_chemical_power_overflow",
        state={"mfuel": mfuel, "heat_per_fuel": heat_per_fuel},
    )
    fuel_sensible_power = _finite(
        mfuel * h_fuel,
        stage="caloric.combustor.power",
        reason="fuel_sensible_power_overflow",
        state={"mfuel": mfuel, "h_fuel": h_fuel},
    )
    absolute_residual = _finite(
        root.residual * state.mass_flow,
        stage="caloric.combustor.energy",
        reason="absolute_energy_residual_overflow",
        state={"residual": root.residual, "mass_flow": state.mass_flow},
    )
    mass_residual = (state.mass_flow + mfuel) - mgas

    outlet = root.gas_model.state(
        spec.totalT, pressure_out, mgas, identifier=None
    )
    enthalpies = CaloricCombustorEnthalpies(
        gas_product_J_per_kg=root.gas_product_J_per_kg,
        air_inlet_J_per_kg=root.air_inlet_J_per_kg,
        weighted_gas_product_J_per_kg_air=(
            root.weighted_gas_product_J_per_kg_air
        ),
        effective_chemical_J_per_kg_air=(
            root.effective_chemical_J_per_kg_air
        ),
        fuel_sensible_J_per_kg_air=root.fuel_sensible_J_per_kg_air,
    )

    return CaloricCombustorResult(
        inlet=state,
        outlet=outlet,
        f=root.f,
        alpha=root.alpha,
        gas_model=root.gas_model,
        mfuel=mfuel,
        mgas=mgas,
        enthalpies=enthalpies,
        h_fuel=h_fuel,
        gross_chemical_W=gross_chemical,
        effective_chemical_W=effective_chemical,
        fuel_sensible_W=fuel_sensible_power,
        energy_residual_specific_J_per_kg_air=root.residual,
        energy_residual_absolute_W=absolute_residual,
        mass_residual_kg_per_s=mass_residual,
        iterations=iterations,
        relative_residual=root.relative_residual,
        reference_T=T_ref,
        R=R,
        model_variant=MODEL_VARIANT,
        limitations=LIMITATIONS,
        spec=spec,
    )


__all__ = [
    "CaloricCombustorEnthalpies",
    "CaloricCombustorError",
    "CaloricCombustorResult",
    "CaloricCombustorSpec",
    "LIMITATIONS",
    "MODEL_VARIANT",
    "VALIDATION_STATUS",
    "close_caloric_combustor",
]
