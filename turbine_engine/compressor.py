"""Reusable 0D compressor-unit and compressor-train components.

SCOPE
=====
This is the existing **0D/mean lumped, variable-k polytropic compressor**
model.  It is not a compressor map, an aerodynamic stage model, a
corrected-flow model, a speed/diameter model, or a geometry model.  A train is
a sequence of independent lumped units in series; it is not a bladed
compressor-stage model.  No preferred operating range, efficiency map, bleed,
or pressure-ratio split rule is inferred here.

PROVENANCE
==========
The temperature equation is the one already used by
:func:`turbine_engine.cycle.calc_compressor`; both paths call the same private
helper in :mod:`turbine_engine.cycle`.  For one unit with inlet total
temperature ``T_in``, pressure ratio ``pr`` and the cycle config:

.. code-block:: text

    T_out = T_in * pr ** ((k - 1) / (k * eta_compressor_poly))
    k     = calc_k_kerosene(0.5 * (T_in + T_out), alpha=inf, R)

The cycle's mean-gas specific work is retained exactly:

.. code-block:: text

    cp = R * k / (k - 1)
    w  = cp * (T_out - T_in) = R * k / (k - 1) * (T_out - T_in)

Each train unit applies that same unit equation to the previous exit state with
the same ``eta_compressor_poly``.  Re-evaluating mean ``k`` after each unit
means a multi-unit train is deliberately **not** algebraically collapsed into
one unit with the product ratio; the supplied ratios remain explicit inputs.
Total pressure is multiplied by each ratio and mass flow is preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from turbine_engine.architecture import StationState
from turbine_engine.cycle import (
    DEFAULT_CONFIG,
    SOLVER_MAX_ITER,
    SOLVER_TOL,
    CycleConfig,
    _compressor_exit_temperature,
)


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


def _normalise_pressure_ratios(
    pressure_ratios: Iterable[float],
) -> tuple[float, ...]:
    if isinstance(pressure_ratios, (str, bytes)):
        raise ValueError(
            "pressure_ratios must be an iterable of finite positive numbers, "
            f"got {pressure_ratios!r}"
        )
    try:
        ratios = tuple(pressure_ratios)
    except TypeError as exc:
        raise ValueError(
            "pressure_ratios must be an iterable of finite positive numbers, "
            f"got {pressure_ratios!r}"
        ) from exc
    if not ratios:
        raise ValueError("pressure_ratios must contain at least one ratio")
    return tuple(
        _as_pressure_ratio(f"pressure_ratios[{index}]", ratio)
        for index, ratio in enumerate(ratios)
    )


@dataclass(frozen=True)
class CompressorResult:
    """Immutable result of one lumped compressor unit or a unit train.

    ``station_states`` holds the exit state of each unit in order; its length
    equals ``len(pressure_ratios)`` and its last element is ``outlet``.
    ``per_unit_specific_work`` is in the same order.  ``specific_work`` is the
    sequential sum of those unit works (J/kg) and
    ``power = specific_work * inlet.mass_flow`` (W).

    ``pressure_ratio_product`` is the arithmetic product of the supplied
    ratios.  ``residual`` is the unfiltered arithmetic diagnostic
    ``outlet.p_total / inlet.p_total - pressure_ratio_product``; it must equal
    that expression but is not required to be zero and is not an operating
    point validity check.  Every field is checked for internal consistency, so
    a record that contradicts these relationships raises ``ValueError``.
    ``cp`` and ``k`` in each exit state are the mean-gas values that produced
    that unit's work, exactly as in the current cycle equation: each unit work
    equals that exit state's ``cp`` times the temperature rise from the
    preceding state (the inlet for the first unit).
    """

    inlet: StationState
    outlet: StationState
    station_states: tuple[StationState, ...]
    pressure_ratios: tuple[float, ...]
    pressure_ratio_product: float
    per_unit_specific_work: tuple[float, ...]
    specific_work: float
    power: float
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
        try:
            station_states = tuple(self.station_states)
            pressure_ratios = tuple(self.pressure_ratios)
            per_unit_specific_work = tuple(self.per_unit_specific_work)
        except TypeError as exc:
            raise ValueError(
                "station_states, pressure_ratios and per_unit_specific_work "
                "must be iterables"
            ) from exc
        if not all(isinstance(state, StationState) for state in station_states):
            raise ValueError(
                "station_states must contain only StationState instances"
            )
        if not (
            len(station_states)
            == len(pressure_ratios)
            == len(per_unit_specific_work)
        ):
            raise ValueError(
                "station_states, pressure_ratios and per_unit_specific_work "
                "must have the same length"
            )
        if not station_states:
            raise ValueError(
                "a compressor result must contain at least one unit"
            )

        pressure_ratios = tuple(
            _as_pressure_ratio(f"pressure_ratios[{index}]", ratio)
            for index, ratio in enumerate(pressure_ratios)
        )
        per_unit_specific_work = tuple(
            _as_finite_float(f"per_unit_specific_work[{index}]", work)
            for index, work in enumerate(per_unit_specific_work)
        )
        pressure_ratio_product = _as_pressure_ratio(
            "pressure_ratio_product", self.pressure_ratio_product
        )
        specific_work = _as_finite_float("specific_work", self.specific_work)
        power = _as_finite_float("power", self.power)
        residual = _as_finite_float("residual", self.residual)

        object.__setattr__(self, "station_states", station_states)
        object.__setattr__(self, "pressure_ratios", pressure_ratios)
        object.__setattr__(
            self, "per_unit_specific_work", per_unit_specific_work
        )
        object.__setattr__(
            self, "pressure_ratio_product", pressure_ratio_product
        )
        object.__setattr__(self, "specific_work", specific_work)
        object.__setattr__(self, "power", power)
        object.__setattr__(self, "residual", residual)

        if self.outlet != station_states[-1]:
            raise ValueError(
                "outlet must equal station_states[-1], got "
                f"{self.outlet!r} and {station_states[-1]!r}"
            )

        previous_pressure = self.inlet.p_total
        for index, (state, ratio) in enumerate(
            zip(station_states, pressure_ratios)
        ):
            if state.mass_flow != self.inlet.mass_flow:
                raise ValueError(
                    f"station_states[{index}].mass_flow must equal "
                    f"inlet.mass_flow, got {state.mass_flow!r} and "
                    f"{self.inlet.mass_flow!r}"
                )
            expected_pressure = previous_pressure * ratio
            if state.p_total != expected_pressure:
                raise ValueError(
                    f"station_states[{index}].p_total must equal the "
                    "preceding pressure times pressure_ratios"
                    f"[{index}], got {state.p_total!r} and "
                    f"{expected_pressure!r}"
                )
            previous_pressure = state.p_total

        previous_temperature = self.inlet.T_total
        for index, (state, work) in enumerate(
            zip(station_states, per_unit_specific_work)
        ):
            expected_work = state.cp * (state.T_total - previous_temperature)
            if work != expected_work:
                raise ValueError(
                    f"per_unit_specific_work[{index}] must equal "
                    f"station_states[{index}].cp * "
                    f"(station_states[{index}].T_total - preceding "
                    f"T_total), got {work!r} and {expected_work!r}"
                )
            previous_temperature = state.T_total

        expected_product = 1.0
        for ratio in pressure_ratios:
            expected_product = expected_product * ratio
        if self.pressure_ratio_product != expected_product:
            raise ValueError(
                "pressure_ratio_product must equal the sequential product "
                f"of pressure_ratios, got {self.pressure_ratio_product!r} "
                f"and {expected_product!r}"
            )

        expected_specific_work = sum(per_unit_specific_work)
        if self.specific_work != expected_specific_work:
            raise ValueError(
                "specific_work must equal sum(per_unit_specific_work), got "
                f"{self.specific_work!r} and {expected_specific_work!r}"
            )

        expected_power = self.specific_work * self.inlet.mass_flow
        if self.power != expected_power:
            raise ValueError(
                "power must equal specific_work * inlet.mass_flow, got "
                f"{self.power!r} and {expected_power!r}"
            )

        expected_residual = (
            self.outlet.p_total / self.inlet.p_total
            - self.pressure_ratio_product
        )
        if self.residual != expected_residual:
            raise ValueError(
                "residual must equal outlet.p_total / inlet.p_total - "
                f"pressure_ratio_product, got {self.residual!r} and "
                f"{expected_residual!r}"
            )


def compressor_train(
    inlet: StationState,
    pressure_ratios: Iterable[float],
    config: CycleConfig = DEFAULT_CONFIG,
) -> CompressorResult:
    """Apply independent lumped 0D compressor units in series.

    ``pressure_ratios`` are explicit finite positive mathematical inputs.  They
    are neither normalised nor constrained to a preferred operating range, and
    their arithmetic product is preserved for diagnostics.  Each unit uses the
    same current cycle polytropic efficiency and re-evaluates mean ``k`` from
    its own inlet and exit temperatures.
    """

    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
        )
    if not isinstance(config, CycleConfig):
        raise TypeError(
            f"config must be a CycleConfig, got {type(config).__name__}"
        )

    ratios = _normalise_pressure_ratios(pressure_ratios)
    state = inlet
    station_states: list[StationState] = []
    per_unit_specific_work: list[float] = []
    product = 1.0

    for ratio in ratios:
        T_out, k = _compressor_exit_temperature(
            state.T_total, ratio, config, SOLVER_TOL, SOLVER_MAX_ITER
        )
        cp = config.R * k / (k - 1.0)
        p_out = state.p_total * ratio
        product = product * ratio
        work = cp * (T_out - state.T_total)
        exit_state = StationState(
            T_total=T_out,
            p_total=p_out,
            mass_flow=state.mass_flow,
            cp=cp,
            k=k,
        )
        station_states.append(exit_state)
        per_unit_specific_work.append(work)
        state = exit_state

    specific_work = sum(per_unit_specific_work)
    power = specific_work * inlet.mass_flow
    residual = state.p_total / inlet.p_total - product
    return CompressorResult(
        inlet=inlet,
        outlet=state,
        station_states=tuple(station_states),
        pressure_ratios=ratios,
        pressure_ratio_product=product,
        per_unit_specific_work=tuple(per_unit_specific_work),
        specific_work=specific_work,
        power=power,
        residual=residual,
    )


def compressor_unit(
    inlet: StationState,
    pressure_ratio: float,
    config: CycleConfig = DEFAULT_CONFIG,
) -> CompressorResult:
    """Apply one independent lumped 0D compressor unit."""

    ratio = _as_pressure_ratio("pressure_ratio", pressure_ratio)
    return compressor_train(inlet, (ratio,), config)


__all__ = [
    "CompressorResult",
    "compressor_train",
    "compressor_unit",
]
