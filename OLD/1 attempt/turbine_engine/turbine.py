"""Reusable 0D/mean lumped turbine-expansion unit.

SCOPE
=====
This is the existing single-point cycle's **total-to-total lumped expansion
relation** as a stand-alone component consuming
:class:`~turbine_engine.architecture.StationState`.  It is not a bladed turbine
stage, a stage-inventory model, a velocity-triangle or meanline stage
calculation, a cooling/bleed model, an operating map, a speed/diameter model,
or a new architecture cycle.  It applies one expansion ratio to one inlet total
state and returns one outlet total state.

PROVENANCE
==========
Both equations are taken from :mod:`turbine_engine.cycle` without change.  For
an inlet total state with ``cp``, ``k``, ``T_in`` and ``p_in``:

.. code-block:: text

    eta   = calc_eta_expansion(pressure_ratio, inlet.k, eta_turbine_poly)
    w     = inlet.cp * inlet.T_total
            * (1 - pressure_ratio ** ((1 - inlet.k) / inlet.k)) * eta
    T_out = inlet.T_total - w / inlet.cp
    p_out = inlet.p_total / pressure_ratio

``pressure_ratio`` is the **expansion ratio** ``p_in_total / p_out_total``:
greater than one for expansion, exactly one for no pressure change, and less
than one only as a mathematical continuation (it represents a pressure-rising,
work-absorbing state, not a turbine design point).  Every finite positive value
is accepted without an operating-map range or a preferred design interval; the
equation itself decides the sign of the work.  A ratio of exactly one makes
:func:`calc_eta_expansion` degenerate (its numerator and denominator both
vanish), so the helper's explicit error is retained instead of inventing a
limiting efficiency.

``cp`` and ``k`` are taken from the inlet state because ``StationState`` does
not carry gas composition, fuel/air ratio, or cooling-flow information.  They
are preserved unchanged at the outlet; this is a unit relation, not a
gas-property update.  ``specific_work`` is positive for expansion and negative
for the mathematical pressure-rising branch.  The stored work is re-evaluated
from the produced outlet temperature as ``inlet.cp * (inlet.T_total -
outlet.T_total)``, which is the same source expression to floating-point
roundoff and keeps the work/temperature pair exactly consistent.

:func:`turbine_pressure_ratio_for_work` is the analytical inverse of the same
constant-``cp``/``k`` relation for a prescribed positive specific work.  It is
used by the opt-in architecture reference cycles for the gas-generator turbine;
it adds no stage, map, or cooling model and does not change the cycle anchor.

Do **not** wire this unit into ``calculate_cycle_point``: the cycle's coupled
nozzle iteration owns its turbine pressure ratio, and the cycle anchor remains
the authority.  Future turbine assembly/stage inventories, cooling splits, and
stage-by-stage pressure ratios are unknown and are not represented here.  A
result record that contradicts its own equations raises ``ValueError``; the
degenerate ratio of exactly one instead propagates the retained helper's
``CycleSolverError``.  The standalone unit itself adds no operating-map limits.

OPT-IN TEXTBOOK PROCESS PROPERTIES
==================================
``gas_property_basis="textbook_true_cp"`` requires explicit lean ``alpha`` and
``R``; the inlet local ``cp``/``k`` must match the published true-cp
correlation at the inlet temperature.  Process ``cp``/``k`` are evaluated at
``0.5 * (T_in + T_out)``.  The retained source expression is

.. code-block:: text

    w = cp_mean * T_in * (1 - pi ** ((1 - k_mean) / k_mean)) * eta_from_poly
    T_out = T_in - w / cp_mean

Forward expansion (``pi > 1``) solves that relation by a bounded fixed point;
outlet ``cp``/``k`` are local true values at ``T_out`` and the process mean is
kept in :class:`TurbineProcessProperties`.  The prescribed-work inverse solves
``T_out = T_in - w_required / cp_mean`` and applies the retained ratio formula
``pi = (1 - w / (cp_mean * T_in)) ** (-k_mean / ((k_mean - 1) * eta))``.
Defaults are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from turbine_engine.architecture import StationState
from turbine_engine.cycle import calc_eta_expansion
from turbine_engine.diagnostics import (
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
    SolverDiagnosticsError,
)

# Opt-in property bases for the existing component.  The default is unchanged.
TURBINE_GAS_PROPERTY_BASIS_LEGACY = "legacy_constant_cp"
TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK = "textbook_true_cp"
TURBINE_GAS_PROPERTY_BASES = (
    TURBINE_GAS_PROPERTY_BASIS_LEGACY,
    TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
)

# Explicit opt-in numerical controls (the legacy path keeps its own values).
TEXTBOOK_TEMPERATURE_TOL_K = 1e-9
TEXTBOOK_WORK_RELATIVE_TOL = 1e-10
TEXTBOOK_MAX_ITER = 100


def _as_finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_pressure_ratio(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


def _resolve_gas_property_basis(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"gas_property_basis must be a string, got {value!r}; "
            f"expected one of {TURBINE_GAS_PROPERTY_BASES!r}"
        )
    if value not in TURBINE_GAS_PROPERTY_BASES:
        raise ValueError(
            f"unknown gas_property_basis {value!r}; "
            f"expected one of {TURBINE_GAS_PROPERTY_BASES!r}"
        )
    return value


def _as_lean_alpha(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"alpha must be a real number, got {value!r}"
        ) from exc
    if math.isnan(result) or result < 1.0:
        raise ValueError(
            "the published true-cp fit requires alpha >= 1 or inf, "
            f"got {value!r}"
        )
    return result


def _textbook_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
    # Lazy import keeps the legacy import graph and matches cycle's bridge.
    from turbine_engine.gas import calc_textbook_gas_cp_k

    return calc_textbook_gas_cp_k(T, alpha, R)


def _resolve_textbook_inputs(
    alpha: Any, R: Any
) -> tuple[float, float]:
    if alpha is None or R is None:
        raise ValueError(
            "alpha and R must be supplied explicitly with "
            "gas_property_basis='textbook_true_cp'; alpha is not inferred from cp"
        )
    alpha_value = _as_lean_alpha(alpha)
    R_value = _as_finite_float("R", R)
    if R_value <= 0.0:
        raise ValueError(f"R must be > 0, got {R_value!r}")
    return alpha_value, R_value


def _validate_textbook_inlet(
    inlet: StationState, alpha: float, R: float
) -> None:
    cp, k = _textbook_cp_k(inlet.T_total, alpha, R)
    if not (
        math.isclose(inlet.cp, cp, rel_tol=1e-9, abs_tol=1e-9)
        and math.isclose(inlet.k, k, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise ValueError(
            "inlet.cp/k do not match the published textbook true-cp values at "
            f"inlet.T_total={inlet.T_total!r} for alpha={alpha!r}, R={R!r}: "
            f"got ({inlet.cp!r}, {inlet.k!r}) and ({cp!r}, {k!r})"
        )


def _nonphysical(
    message: str, reason: str, state: dict[str, Any]
) -> None:
    raise SolverDiagnosticsError(
        message,
        stage="turbine.textbook_true_cp",
        reason=reason,
        status=NON_PHYSICAL_STATE,
        last_valid_state=state,
    )


@dataclass(frozen=True)
class TurbineProcessProperties:
    """Opt-in process-mean true-cp record for one turbine expansion.

    Endpoint temperatures, process-mean ``cp``/``k``, composition, residual
    diagnostics, and (when requested) requested/achieved work are kept
    separate from the local ``StationState`` values.
    """

    T_in: float
    T_out: float
    T_mean: float
    cp: float
    k: float
    alpha: float
    gas_constant_R: float
    method: str
    tolerance_K: float
    temperature_residual_K: float
    pressure_formula_temperature_residual_K: float
    requested_specific_work: float | None = None
    achieved_specific_work: float | None = None
    work_relative_residual: float | None = None

    def __post_init__(self) -> None:
        for name in ("T_in", "T_out", "T_mean", "cp"):
            value = _as_finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value!r}")
            object.__setattr__(self, name, value)
        k = _as_finite_float("k", self.k)
        if k <= 1.0:
            raise ValueError(f"k must be > 1, got {k!r}")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "alpha", _as_lean_alpha(self.alpha))
        R = _as_finite_float("gas_constant_R", self.gas_constant_R)
        if R <= 0.0:
            raise ValueError(f"gas_constant_R must be > 0, got {R!r}")
        object.__setattr__(self, "gas_constant_R", R)
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError(f"method must be a non-empty string, got {self.method!r}")
        tolerance = _as_finite_float("tolerance_K", self.tolerance_K)
        if tolerance < 0.0:
            raise ValueError(f"tolerance_K must be >= 0, got {tolerance!r}")
        object.__setattr__(self, "tolerance_K", tolerance)
        for name in ("temperature_residual_K", "pressure_formula_temperature_residual_K"):
            residual = _as_finite_float(name, getattr(self, name))
            if abs(residual) > tolerance:
                raise ValueError(f"{name} exceeds tolerance_K: {residual!r} > {tolerance!r}")
            object.__setattr__(self, name, residual)

        if self.T_mean != 0.5 * (self.T_in + self.T_out):
            raise ValueError("T_mean must equal 0.5 * (T_in + T_out)")
        cp, k_source = _textbook_cp_k(self.T_mean, self.alpha, self.gas_constant_R)
        if self.cp != cp or self.k != k_source:
            raise ValueError(
                f"cp/k must equal the published true-cp values at T_mean, got "
                f"({self.cp!r}, {self.k!r}) and ({cp!r}, {k_source!r})"
            )

        work = (self.requested_specific_work, self.achieved_specific_work, self.work_relative_residual)
        if all(value is None for value in work):
            return
        if any(value is None for value in work):
            raise ValueError("requested, achieved and relative work fields must be all None or all provided")
        requested, achieved, relative = (
            _as_finite_float("requested_specific_work", work[0]),
            _as_finite_float("achieved_specific_work", work[1]),
            _as_finite_float("work_relative_residual", work[2]),
        )
        if requested <= 0.0:
            raise ValueError(f"requested_specific_work must be > 0, got {requested!r}")
        expected_achieved = self.cp * (self.T_in - self.T_out)
        expected_relative = abs(achieved - requested) / requested
        if achieved != expected_achieved:
            raise ValueError(
                "achieved_specific_work must equal cp_mean * (T_in - T_out), got "
                f"{achieved!r} and {expected_achieved!r}"
            )
        if relative != expected_relative:
            raise ValueError(
                "work_relative_residual must equal abs(achieved - requested) / requested, got "
                f"{relative!r} and {expected_relative!r}"
            )
        if relative > TEXTBOOK_WORK_RELATIVE_TOL:
            raise ValueError(
                f"work_relative_residual exceeds {TEXTBOOK_WORK_RELATIVE_TOL!r}: {relative!r}"
            )
        object.__setattr__(self, "requested_specific_work", requested)
        object.__setattr__(self, "achieved_specific_work", achieved)
        object.__setattr__(self, "work_relative_residual", relative)



@dataclass(frozen=True)
class TurbineResult:
    """Immutable result of one lumped 0D turbine-expansion unit.

    ``pressure_ratio`` is the expansion ratio ``inlet.p_total /
    outlet.p_total``.  ``specific_work`` is the per-mass work/drop (J/kg),
    positive for expansion and negative for the mathematical pressure-rising
    branch; ``power = specific_work * inlet.mass_flow`` (W).
    ``efficiency`` is the total-to-total efficiency returned by
    :func:`turbine_engine.cycle.calc_eta_expansion` for the supplied
    ``eta_turbine_poly``.  ``residual`` is the unfiltered arithmetic diagnostic
    ``inlet.p_total / outlet.p_total - pressure_ratio``; it must equal that
    expression but is not required to be exactly zero and is not an operating
    point validity check.

    Every field is checked for internal consistency.  A record that contradicts
    these relationships raises ``ValueError``; a pressure ratio of exactly one
    propagates the retained ``CycleSolverError`` because the efficiency is
    undefined.  ``outlet.mass_flow``, ``outlet.cp`` and ``outlet.k`` must equal
    the inlet values, and ``specific_work`` must equal ``inlet.cp *
    (inlet.T_total - outlet.T_total)`` exactly.  ``outlet.T_total`` must equal
    the source expansion temperature ``inlet.T_total - energy_drop / inlet.cp``
    for ``energy_drop = inlet.cp * inlet.T_total *
    (1 - pressure_ratio ** ((1 - inlet.k) / inlet.k)) * efficiency``.
    """

    inlet: StationState
    outlet: StationState
    pressure_ratio: float
    eta_turbine_poly: float
    specific_work: float
    power: float
    efficiency: float
    residual: float
    gas_property_basis: str = TURBINE_GAS_PROPERTY_BASIS_LEGACY
    process_properties: TurbineProcessProperties | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inlet, StationState):
            raise TypeError(
                f"inlet must be a StationState, got {type(self.inlet).__name__}"
            )
        if not isinstance(self.outlet, StationState):
            raise TypeError(
                f"outlet must be a StationState, got {type(self.outlet).__name__}"
            )

        pressure_ratio = _as_pressure_ratio(
            "pressure_ratio", self.pressure_ratio
        )
        eta_turbine_poly = _as_finite_float(
            "eta_turbine_poly", self.eta_turbine_poly
        )
        specific_work = _as_finite_float("specific_work", self.specific_work)
        power = _as_finite_float("power", self.power)
        efficiency = _as_finite_float("efficiency", self.efficiency)
        residual = _as_finite_float("residual", self.residual)
        basis = _resolve_gas_property_basis(self.gas_property_basis)
        process_properties = self.process_properties
        if process_properties is not None and not isinstance(
            process_properties, TurbineProcessProperties
        ):
            raise ValueError(
                "process_properties must be None or a "
                "TurbineProcessProperties record"
            )
        if basis == TURBINE_GAS_PROPERTY_BASIS_LEGACY:
            if process_properties is not None:
                raise ValueError(
                    "process_properties must be None for the "
                    "legacy_constant_cp basis"
                )
        elif process_properties is None:
            raise ValueError(
                "textbook_true_cp requires process_properties"
            )

        object.__setattr__(self, "pressure_ratio", pressure_ratio)
        object.__setattr__(self, "eta_turbine_poly", eta_turbine_poly)
        object.__setattr__(self, "specific_work", specific_work)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "efficiency", efficiency)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "gas_property_basis", basis)

        if self.outlet.mass_flow != self.inlet.mass_flow:
            raise ValueError(
                "outlet.mass_flow must equal inlet.mass_flow, got "
                f"{self.outlet.mass_flow!r} and {self.inlet.mass_flow!r}"
            )

        expected_pressure = self.inlet.p_total / self.pressure_ratio
        if self.outlet.p_total != expected_pressure:
            raise ValueError(
                "outlet.p_total must equal inlet.p_total / pressure_ratio, "
                f"got {self.outlet.p_total!r} and {expected_pressure!r}"
            )

        if basis == TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK:
            self._validate_textbook_energy(process_properties)
            expected_residual = (
                self.inlet.p_total / self.outlet.p_total - self.pressure_ratio
            )
            if self.residual != expected_residual:
                raise ValueError(
                    "residual must equal inlet.p_total / outlet.p_total - "
                    f"pressure_ratio, got {self.residual!r} and "
                    f"{expected_residual!r}"
                )
            return

        if self.outlet.cp != self.inlet.cp:
            raise ValueError(
                "outlet.cp must equal inlet.cp, got "
                f"{self.outlet.cp!r} and {self.inlet.cp!r}"
            )
        if self.outlet.k != self.inlet.k:
            raise ValueError(
                "outlet.k must equal inlet.k, got "
                f"{self.outlet.k!r} and {self.inlet.k!r}"
            )

        expected_work = self.inlet.cp * (
            self.inlet.T_total - self.outlet.T_total
        )
        if self.specific_work != expected_work:
            raise ValueError(
                "specific_work must equal inlet.cp * (inlet.T_total - "
                f"outlet.T_total), got {self.specific_work!r} and "
                f"{expected_work!r}"
            )

        expected_power = self.specific_work * self.inlet.mass_flow
        if self.power != expected_power:
            raise ValueError(
                "power must equal specific_work * inlet.mass_flow, got "
                f"{self.power!r} and {expected_power!r}"
            )

        expected_efficiency = calc_eta_expansion(
            self.pressure_ratio, self.inlet.k, self.eta_turbine_poly
        )
        if self.efficiency != expected_efficiency:
            raise ValueError(
                "efficiency must equal calc_eta_expansion(pressure_ratio, "
                "inlet.k, eta_turbine_poly), got "
                f"{self.efficiency!r} and {expected_efficiency!r}"
            )

        expected_energy_drop = (
            self.inlet.cp
            * self.inlet.T_total
            * (
                1.0
                - self.pressure_ratio ** ((1.0 - self.inlet.k) / self.inlet.k)
            )
            * expected_efficiency
        )
        expected_outlet_temperature = (
            self.inlet.T_total - expected_energy_drop / self.inlet.cp
        )
        if self.outlet.T_total != expected_outlet_temperature:
            raise ValueError(
                "outlet.T_total must equal inlet.T_total - energy_drop / "
                "inlet.cp for energy_drop = inlet.cp * inlet.T_total * "
                "(1 - pressure_ratio ** ((1 - inlet.k) / inlet.k)) * "
                "efficiency, got "
                f"{self.outlet.T_total!r} and {expected_outlet_temperature!r}"
            )

        expected_residual = (
            self.inlet.p_total / self.outlet.p_total - self.pressure_ratio
        )
        if self.residual != expected_residual:
            raise ValueError(
                "residual must equal inlet.p_total / outlet.p_total - "
                f"pressure_ratio, got {self.residual!r} and "
                f"{expected_residual!r}"
            )

    def _validate_textbook_energy(
        self, process: TurbineProcessProperties
    ) -> None:
        if (process.T_in, process.T_out) != (
            self.inlet.T_total,
            self.outlet.T_total,
        ):
            raise ValueError(
                "process_properties.T_in/T_out must equal the endpoint states, got "
                f"({process.T_in!r}, {process.T_out!r}) and "
                f"({self.inlet.T_total!r}, {self.outlet.T_total!r})"
            )
        _validate_textbook_inlet(
            self.inlet, process.alpha, process.gas_constant_R
        )

        expected_work = process.cp * (
            self.inlet.T_total - self.outlet.T_total
        )
        if self.specific_work != expected_work or self.specific_work <= 0.0:
            raise ValueError(
                "specific_work must equal positive process-mean cp * "
                f"(inlet.T_total - outlet.T_total), got {self.specific_work!r} "
                f"and {expected_work!r}"
            )
        if self.power != self.specific_work * self.inlet.mass_flow:
            raise ValueError(
                "power must equal specific_work * inlet.mass_flow, got "
                f"{self.power!r} and {self.specific_work * self.inlet.mass_flow!r}"
            )

        expected_efficiency = calc_eta_expansion(
            self.pressure_ratio, process.k, self.eta_turbine_poly
        )
        if self.efficiency != expected_efficiency:
            raise ValueError(
                "efficiency must equal calc_eta_expansion(pressure_ratio, "
                "process-mean k, eta_turbine_poly), got "
                f"{self.efficiency!r} and {expected_efficiency!r}"
            )

        local_cp, local_k = _textbook_cp_k(
            self.outlet.T_total, process.alpha, process.gas_constant_R
        )
        if (self.outlet.cp, self.outlet.k) != (local_cp, local_k):
            raise ValueError(
                "outlet.cp/k must be the local textbook true-cp values at "
                f"outlet.T_total={self.outlet.T_total!r}, got "
                f"({self.outlet.cp!r}, {self.outlet.k!r}) and "
                f"({local_cp!r}, {local_k!r})"
            )

        predicted = self.inlet.T_total * (
            1.0
            - (1.0 - self.pressure_ratio ** ((1.0 - process.k) / process.k))
            * self.efficiency
        )
        expected_pressure_residual = predicted - self.outlet.T_total
        if abs(expected_pressure_residual) > process.tolerance_K:
            raise ValueError(
                "outlet.T_total does not close the pressure/efficiency "
                f"temperature formula within {process.tolerance_K!r}: got "
                f"{self.outlet.T_total!r} and {predicted!r}"
            )
        if (
            process.pressure_formula_temperature_residual_K
            != expected_pressure_residual
        ):
            raise ValueError(
                "process_properties.pressure_formula_temperature_residual_K "
                "must equal the recomputed pressure/efficiency temperature "
                "residual, got "
                f"{process.pressure_formula_temperature_residual_K!r} and "
                f"{expected_pressure_residual!r}"
            )
        if process.requested_specific_work is None:
            expected_temperature_residual = expected_pressure_residual
        else:
            expected_temperature_residual = (
                self.inlet.T_total
                - process.requested_specific_work / process.cp
                - self.outlet.T_total
            )
        if process.temperature_residual_K != expected_temperature_residual:
            raise ValueError(
                "process_properties.temperature_residual_K does not match "
                "the recomputed fixed-point equation residual, got "
                f"{process.temperature_residual_K!r} and "
                f"{expected_temperature_residual!r}"
            )


def _textbook_forward_state(
    inlet: StationState,
    ratio: float,
    eta_poly: float,
    alpha: float,
    R: float,
) -> dict[str, Any]:
    """Bounded process-mean true-cp forward fixed point."""

    if ratio <= 1.0:
        raise ValueError(
            "textbook_true_cp forward expansion requires "
            f"pressure_ratio > 1, got {ratio!r}"
        )
    T_in = inlet.T_total
    cp0, k0 = _textbook_cp_k(T_in, alpha, R)
    eta0 = calc_eta_expansion(ratio, k0, eta_poly)
    candidate = T_in * (
        1.0 - (1.0 - ratio ** ((1.0 - k0) / k0)) * eta0
    )
    if not math.isfinite(candidate) or candidate <= 0.0:
        _nonphysical(
            "textbook turbine initial expansion temperature is invalid",
            "turbine_initial_temperature_invalid",
            {"T_in": T_in, "pressure_ratio": ratio, "T_candidate": candidate},
        )

    last: dict[str, Any] = {}
    for iteration in range(TEXTBOOK_MAX_ITER):
        T_mean = 0.5 * (T_in + candidate)
        cp, k = _textbook_cp_k(T_mean, alpha, R)
        eta = calc_eta_expansion(ratio, k, eta_poly)
        predicted = T_in * (
            1.0 - (1.0 - ratio ** ((1.0 - k) / k)) * eta
        )
        residual = predicted - candidate
        work = cp * (T_in - candidate)
        last = {
            "iteration": iteration + 1,
            "T_evaluated": candidate,
            "T_proposed": predicted,
            "residual_K": residual,
        }
        if abs(residual) <= TEXTBOOK_TEMPERATURE_TOL_K:
            if not math.isfinite(work) or work <= 0.0:
                _nonphysical(
                    "textbook turbine returned state has non-positive work",
                    "turbine_work_not_positive",
                    {**last, "specific_work": work},
                )
            return {
                "T_out": candidate,
                "T_mean": T_mean,
                "cp": cp,
                "k": k,
                "efficiency": eta,
                "specific_work": work,
                "temperature_residual_K": residual,
                "pressure_formula_temperature_residual_K": residual,
            }
        if not math.isfinite(predicted) or predicted <= 0.0:
            _nonphysical(
                "textbook turbine expansion produced an invalid outlet temperature",
                "turbine_outlet_temperature_invalid",
                last,
            )
        candidate = predicted

    raise NumericalConvergenceError(
        "turbine.textbook_true_cp forward temperature iteration exhausted",
        stage="turbine.textbook_true_cp",
        reason="forward_temperature_iteration_exhausted",
        last_valid_state={
            "iteration_limit": TEXTBOOK_MAX_ITER,
            "tolerance_K": TEXTBOOK_TEMPERATURE_TOL_K,
            "converged": False,
            **last,
        },
    )


def _textbook_inverse_state(
    inlet: StationState,
    required: float,
    eta_poly: float,
    alpha: float,
    R: float,
) -> dict[str, Any]:
    """Bounded process-mean true-cp prescribed-work fixed point."""

    T_in = inlet.T_total
    cp_in, _ = _textbook_cp_k(T_in, alpha, R)
    candidate = T_in - required / cp_in
    if not math.isfinite(candidate) or candidate <= 0.0:
        raise ValueError(
            "required_specific_work exceeds the textbook process-mean "
            f"expansion capacity at T_in={T_in!r}: got {required!r} J/kg"
        )

    last: dict[str, Any] = {}
    for iteration in range(TEXTBOOK_MAX_ITER):
        T_mean = 0.5 * (T_in + candidate)
        cp, k = _textbook_cp_k(T_mean, alpha, R)
        predicted = T_in - required / cp
        residual = predicted - candidate
        achieved = cp * (T_in - candidate)
        relative = abs(achieved - required) / required
        last = {
            "iteration": iteration + 1,
            "T_evaluated": candidate,
            "T_proposed": predicted,
            "residual_K": residual,
            "achieved_specific_work": achieved,
            "work_relative_residual": relative,
        }
        if (
            abs(residual) <= TEXTBOOK_TEMPERATURE_TOL_K
            and relative <= TEXTBOOK_WORK_RELATIVE_TOL
        ):
            break
        if not math.isfinite(predicted) or predicted <= 0.0:
            raise ValueError(
                "required_specific_work exceeds the textbook process-mean "
                "expansion capacity at the evaluated mean state: "
                f"required={required!r} J/kg, evaluated={last!r}"
            )
        candidate = predicted
    else:
        raise NumericalConvergenceError(
            "turbine.textbook_true_cp inverse temperature iteration exhausted",
            stage="turbine.textbook_true_cp",
            reason="inverse_temperature_iteration_exhausted",
            last_valid_state={
                "iteration_limit": TEXTBOOK_MAX_ITER,
                "tolerance_K": TEXTBOOK_TEMPERATURE_TOL_K,
                "work_relative_tolerance": TEXTBOOK_WORK_RELATIVE_TOL,
                "converged": False,
                **last,
            },
        )

    base = 1.0 - achieved / (cp * T_in)
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError(
            "required_specific_work exceeds the textbook process-mean "
            f"expansion capacity cp_mean*T_in={cp * T_in!r} J/kg: "
            f"got {required!r}"
        )
    exponent = -k / ((k - 1.0) * eta_poly)
    try:
        pressure_ratio = base**exponent
    except OverflowError as exc:
        raise ValueError(
            "required_specific_work produced an unrepresentable textbook "
            f"expansion ratio (base={base!r}, exponent={exponent!r})"
        ) from exc
    if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
        raise NumericalConvergenceError(
            "required_specific_work produced a non-finite textbook "
            "expansion ratio",
            stage="turbine.textbook_true_cp",
            reason="inverse_pressure_ratio_invalid",
            last_valid_state={
                "base": base,
                "exponent": exponent,
                "pressure_ratio": pressure_ratio,
                **last,
            },
        )
    if pressure_ratio <= 1.0:
        raise ValueError(
            "required_specific_work produced a non-expanding textbook "
            f"pressure ratio {pressure_ratio!r}; required={required!r} J/kg"
        )

    efficiency = calc_eta_expansion(pressure_ratio, k, eta_poly)
    predicted_outlet = T_in * (
        1.0
        - (1.0 - pressure_ratio ** ((1.0 - k) / k)) * efficiency
    )
    pressure_residual = predicted_outlet - candidate
    if abs(pressure_residual) > TEXTBOOK_TEMPERATURE_TOL_K:
        raise NumericalConvergenceError(
            "textbook inverse pressure/efficiency temperature formula did "
            "not close within tolerance",
            stage="turbine.textbook_true_cp",
            reason="inverse_pressure_formula_closure",
            last_valid_state={
                "pressure_ratio": pressure_ratio,
                "predicted_outlet_T": predicted_outlet,
                "evaluated_T": candidate,
                "pressure_formula_residual_K": pressure_residual,
                **last,
            },
        )
    return {
        "T_out": candidate,
        "T_mean": T_mean,
        "cp": cp,
        "k": k,
        "efficiency": efficiency,
        "pressure_ratio": pressure_ratio,
        "specific_work": achieved,
        "requested_specific_work": required,
        "work_relative_residual": relative,
        "temperature_residual_K": residual,
        "pressure_formula_temperature_residual_K": pressure_residual,
    }


def _textbook_result(
    inlet: StationState,
    ratio: float,
    eta_poly: float,
    alpha: float,
    R: float,
    state: dict[str, Any],
    *,
    forward: bool,
) -> TurbineResult:
    """Build and validate the endpoint states from a textbook fixed point."""

    T_out = state["T_out"]
    p_out = inlet.p_total / ratio
    if (
        not math.isfinite(T_out)
        or T_out <= 0.0
        or not math.isfinite(p_out)
        or p_out <= 0.0
    ):
        raise ValueError(
            "textbook turbine produced a nonphysical outlet state "
            f"(T_total={T_out!r}, p_total={p_out!r})"
        )
    local_cp, local_k = _textbook_cp_k(T_out, alpha, R)
    outlet = StationState(
        T_total=T_out,
        p_total=p_out,
        mass_flow=inlet.mass_flow,
        cp=local_cp,
        k=local_k,
    )
    process = TurbineProcessProperties(
        T_in=inlet.T_total,
        T_out=T_out,
        T_mean=state["T_mean"],
        cp=state["cp"],
        k=state["k"],
        alpha=alpha,
        gas_constant_R=R,
        method=(
            "textbook_true_cp_process_mean_forward"
            if forward
            else "textbook_true_cp_process_mean_inverse"
        ),
        tolerance_K=TEXTBOOK_TEMPERATURE_TOL_K,
        temperature_residual_K=state["temperature_residual_K"],
        pressure_formula_temperature_residual_K=state[
            "pressure_formula_temperature_residual_K"
        ],
        requested_specific_work=(
            None if forward else state["requested_specific_work"]
        ),
        achieved_specific_work=(
            None if forward else state["specific_work"]
        ),
        work_relative_residual=(
            None if forward else state["work_relative_residual"]
        ),
    )
    return TurbineResult(
        inlet=inlet,
        outlet=outlet,
        pressure_ratio=ratio,
        eta_turbine_poly=eta_poly,
        specific_work=state["specific_work"],
        power=state["specific_work"] * inlet.mass_flow,
        efficiency=state["efficiency"],
        residual=inlet.p_total / outlet.p_total - ratio,
        gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
        process_properties=process,
    )


def turbine_pressure_ratio_for_work(
    inlet: StationState,
    required_specific_work: float,
    eta_turbine_poly: float,
    *,
    gas_property_basis: str = TURBINE_GAS_PROPERTY_BASIS_LEGACY,
    alpha: float | None = None,
    R: float | None = None,
) -> TurbineResult:
    """Return the expansion needed to produce a prescribed specific work.

    This is the analytical inverse of the retained constant-``cp``/``k``
    polytropic unit relation used by :func:`turbine_expansion`.  For positive
    expansion work the work expression simplifies to

    .. code-block:: text

        w(pi) = cp * T_in * (1 - pi ** (-(k - 1) * eta_poly / k))

    (the total-to-total efficiency numerator and denominator cancel exactly),
    so

    .. code-block:: text

        pi = (1 - w / (cp * T_in)) ** (-k / ((k - 1) * eta_poly)).

    ``required_specific_work`` is the per-kilogram gas work in J/kg and must be
    finite and strictly positive.  A request greater than or equal to
    ``cp * T_in`` cannot be produced by this expansion and raises
    ``ValueError`` instead of returning a nonphysical temperature.  The
    returned record is the same validated :class:`TurbineResult` produced by
    :func:`turbine_expansion`; its ``specific_work`` is the achieved value and
    agrees with the request to floating-point roundoff.  No operating-map,
    stage-count, or design-range limit is applied.

    This inverse is the polynomial/constant-property branch only.  It does not
    infer gas properties or a turbine stage inventory.
    """

    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
        )
    basis = _resolve_gas_property_basis(gas_property_basis)
    required = _as_finite_float(
        "required_specific_work", required_specific_work
    )
    if required <= 0.0:
        raise ValueError(
            "required_specific_work must be > 0, "
            f"got {required_specific_work!r}"
        )
    eta_value = _as_finite_float("eta_turbine_poly", eta_turbine_poly)
    if not 0.0 < eta_value <= 1.0:
        raise ValueError(
            f"eta_turbine_poly must be in (0, 1], got {eta_turbine_poly!r}"
        )

    if basis == TURBINE_GAS_PROPERTY_BASIS_LEGACY:
        if alpha is not None or R is not None:
            raise ValueError(
                "alpha and R are only used with "
                "gas_property_basis='textbook_true_cp'"
            )
        available_per_cp = inlet.cp * inlet.T_total
        base = 1.0 - required / available_per_cp
        if base <= 0.0:
            raise ValueError(
                "required_specific_work exceeds the constant-property "
                f"expansion capacity cp*T_in={available_per_cp!r} J/kg: "
                f"got {required!r}"
            )
        exponent = -inlet.k / ((inlet.k - 1.0) * eta_value)
        try:
            pressure_ratio = base**exponent
        except OverflowError as exc:
            raise ValueError(
                "required_specific_work produced an unrepresentable "
                f"expansion ratio (base={base!r}, exponent={exponent!r})"
            ) from exc
        if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
            raise ValueError(
                "required_specific_work produced a non-finite expansion "
                f"ratio {pressure_ratio!r} for base={base!r}, "
                f"exponent={exponent!r}"
            )

        result = turbine_expansion(inlet, pressure_ratio, eta_value)
        if not math.isclose(
            result.specific_work,
            required,
            rel_tol=1e-10,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "turbine inverse produced inconsistent work: requested "
                f"{required!r} J/kg, achieved {result.specific_work!r} J/kg"
            )
        return result

    alpha_value, R_value = _resolve_textbook_inputs(alpha, R)
    _validate_textbook_inlet(inlet, alpha_value, R_value)
    state = _textbook_inverse_state(
        inlet, required, eta_value, alpha_value, R_value
    )
    return _textbook_result(
        inlet,
        state["pressure_ratio"],
        eta_value,
        alpha_value,
        R_value,
        state,
        forward=False,
    )


def turbine_expansion(
    inlet: StationState,
    pressure_ratio: float,
    eta_turbine_poly: float,
    *,
    gas_property_basis: str = TURBINE_GAS_PROPERTY_BASIS_LEGACY,
    alpha: float | None = None,
    R: float | None = None,
) -> TurbineResult:
    """Apply one lumped 0D total-to-total turbine-expansion unit.

    ``pressure_ratio`` is the explicit expansion ratio
    ``inlet.p_total / outlet.p_total``.  It must be finite and positive; no
    preferred operating range, stage split, or map limit is applied.  A ratio
    of exactly one is rejected by the retained
    :func:`turbine_engine.cycle.calc_eta_expansion` denominator check because
    the total-to-total efficiency is undefined there.

    Nonphysical output temperatures or pressures (non-finite or non-positive)
    raise ``ValueError`` instead of constructing an invalid ``StationState``.
    The opt-in textbook branch requires explicit ``alpha``/``R`` and ``pi > 1``;
    the default legacy branch is unchanged.
    """

    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
        )

    basis = _resolve_gas_property_basis(gas_property_basis)
    if basis == TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK:
        alpha_value, R_value = _resolve_textbook_inputs(alpha, R)
        _validate_textbook_inlet(inlet, alpha_value, R_value)
        ratio = _as_pressure_ratio("pressure_ratio", pressure_ratio)
        eta_value = _as_finite_float("eta_turbine_poly", eta_turbine_poly)
        if not 0.0 < eta_value <= 1.0:
            raise ValueError(
                "eta_turbine_poly must be in (0, 1], "
                f"got {eta_turbine_poly!r}"
            )
        state = _textbook_forward_state(
            inlet, ratio, eta_value, alpha_value, R_value
        )
        return _textbook_result(
            inlet,
            ratio,
            eta_value,
            alpha_value,
            R_value,
            state,
            forward=True,
        )

    if alpha is not None or R is not None:
        raise ValueError(
            "alpha and R are only used with "
            "gas_property_basis='textbook_true_cp'"
        )
    ratio = _as_pressure_ratio("pressure_ratio", pressure_ratio)
    efficiency = calc_eta_expansion(ratio, inlet.k, eta_turbine_poly)
    if not math.isfinite(efficiency):
        raise ValueError(
            "turbine expansion produced non-finite efficiency "
            f"{efficiency!r} for pressure_ratio={ratio!r}, k={inlet.k!r}"
        )

    energy_drop = (
        inlet.cp
        * inlet.T_total
        * (1.0 - ratio ** ((1.0 - inlet.k) / inlet.k))
        * efficiency
    )
    if not math.isfinite(energy_drop):
        raise ValueError(
            "turbine expansion produced non-finite work/drop "
            f"{energy_drop!r} for pressure_ratio={ratio!r}, "
            f"T_in={inlet.T_total!r}, cp={inlet.cp!r}, k={inlet.k!r}"
        )

    outlet_temperature = inlet.T_total - energy_drop / inlet.cp
    outlet_pressure = inlet.p_total / ratio
    if (
        not math.isfinite(outlet_temperature)
        or outlet_temperature <= 0.0
        or not math.isfinite(outlet_pressure)
        or outlet_pressure <= 0.0
    ):
        raise ValueError(
            "turbine expansion produced a nonphysical outlet state "
            f"(T_total={outlet_temperature!r}, "
            f"p_total={outlet_pressure!r}) for pressure_ratio={ratio!r}, "
            f"T_in={inlet.T_total!r}, p_in={inlet.p_total!r}, "
            f"cp={inlet.cp!r}, k={inlet.k!r}"
        )

    outlet = StationState(
        T_total=outlet_temperature,
        p_total=outlet_pressure,
        mass_flow=inlet.mass_flow,
        cp=inlet.cp,
        k=inlet.k,
    )

    # Re-evaluate the work from the produced outlet temperature so the stored
    # work/temperature pair is exactly consistent, as in the compressor unit.
    # This is algebraically the same source expression and differs only by
    # floating-point roundoff.
    specific_work = inlet.cp * (inlet.T_total - outlet.T_total)
    power = specific_work * inlet.mass_flow
    residual = inlet.p_total / outlet.p_total - ratio
    return TurbineResult(
        inlet=inlet,
        outlet=outlet,
        pressure_ratio=ratio,
        eta_turbine_poly=eta_turbine_poly,
        specific_work=specific_work,
        power=power,
        efficiency=efficiency,
        residual=residual,
        gas_property_basis=basis,
        process_properties=None,
    )


__all__ = [
    "TURBINE_GAS_PROPERTY_BASES",
    "TURBINE_GAS_PROPERTY_BASIS_LEGACY",
    "TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK",
    "TEXTBOOK_MAX_ITER",
    "TEXTBOOK_TEMPERATURE_TOL_K",
    "TEXTBOOK_WORK_RELATIVE_TOL",
    "TurbineProcessProperties",
    "TurbineResult",
    "turbine_expansion",
    "turbine_pressure_ratio_for_work",
]
