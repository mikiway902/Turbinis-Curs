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
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from turbine_engine.architecture import StationState
from turbine_engine.cycle import calc_eta_expansion


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

        object.__setattr__(self, "pressure_ratio", pressure_ratio)
        object.__setattr__(self, "eta_turbine_poly", eta_turbine_poly)
        object.__setattr__(self, "specific_work", specific_work)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "efficiency", efficiency)
        object.__setattr__(self, "residual", residual)

        if self.outlet.mass_flow != self.inlet.mass_flow:
            raise ValueError(
                "outlet.mass_flow must equal inlet.mass_flow, got "
                f"{self.outlet.mass_flow!r} and {self.inlet.mass_flow!r}"
            )
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

        expected_pressure = self.inlet.p_total / self.pressure_ratio
        if self.outlet.p_total != expected_pressure:
            raise ValueError(
                "outlet.p_total must equal inlet.p_total / pressure_ratio, "
                f"got {self.outlet.p_total!r} and {expected_pressure!r}"
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


def turbine_pressure_ratio_for_work(
    inlet: StationState,
    required_specific_work: float,
    eta_turbine_poly: float,
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

    available_per_cp = inlet.cp * inlet.T_total
    base = 1.0 - required / available_per_cp
    if base <= 0.0:
        raise ValueError(
            "required_specific_work exceeds the constant-property expansion "
            f"capacity cp*T_in={available_per_cp!r} J/kg: got {required!r}"
        )
    exponent = -inlet.k / ((inlet.k - 1.0) * eta_value)
    try:
        pressure_ratio = base**exponent
    except OverflowError as exc:
        raise ValueError(
            "required_specific_work produced an unrepresentable expansion "
            f"ratio (base={base!r}, exponent={exponent!r})"
        ) from exc
    if not math.isfinite(pressure_ratio) or pressure_ratio <= 0.0:
        raise ValueError(
            "required_specific_work produced a non-finite expansion ratio "
            f"{pressure_ratio!r} for base={base!r}, exponent={exponent!r}"
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


def turbine_expansion(
    inlet: StationState,
    pressure_ratio: float,
    eta_turbine_poly: float,
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
    """

    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
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
    )


__all__ = [
    "TurbineResult",
    "turbine_expansion",
    "turbine_pressure_ratio_for_work",
]
