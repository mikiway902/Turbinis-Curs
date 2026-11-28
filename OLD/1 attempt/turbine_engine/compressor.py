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

OPT-IN TEXTBOOK PROCESS PROPERTIES
==================================
``gas_property_basis="textbook_true_cp"`` calls ``calc_compressor`` at each
unit's actual inlet and evaluates published true ``cp`` at
``0.5 * (T_in + T_out)`` for work.  Exit ``cp``/``k`` are local true values at
``T_out``; process means are kept in ``CompressorProcessProperties``.  The
default legacy basis creates no records and is unchanged.
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
    calc_compressor,
)

# Opt-in property bases for the existing component.  The default is unchanged.
COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY = "legacy_mean_cp"
COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK = "textbook_true_cp"
COMPRESSOR_GAS_PROPERTY_BASES = (
    COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY,
    COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
)


def _resolve_gas_property_basis(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            f"gas_property_basis must be a string, got {value!r}; "
            f"expected one of {COMPRESSOR_GAS_PROPERTY_BASES!r}"
        )
    if value not in COMPRESSOR_GAS_PROPERTY_BASES:
        raise ValueError(
            f"unknown gas_property_basis {value!r}; "
            f"expected one of {COMPRESSOR_GAS_PROPERTY_BASES!r}"
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
    # Lazy import mirrors cycle._textbook_cp_k and avoids a new cycle/gas cycle.
    from turbine_engine.gas import calc_textbook_gas_cp_k

    return calc_textbook_gas_cp_k(T, alpha, R)


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
class CompressorProcessProperties:
    """Opt-in process-mean true-cp record for one compressor unit.

    ``fixed_point_residual_K`` is the retained polytropic temperature relation
    residual and must be within ``tolerance_K``; it is not forced to zero.
    """

    T_in: float
    T_out: float
    T_mean: float
    cp: float
    k: float
    pressure_ratio: float
    eta_polytropic: float
    gas_constant_R: float
    alpha: float
    method: str
    fixed_point_residual_K: float
    tolerance_K: float

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
        object.__setattr__(
            self, "pressure_ratio",
            _as_pressure_ratio("pressure_ratio", self.pressure_ratio),
        )
        eta = _as_finite_float("eta_polytropic", self.eta_polytropic)
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"eta_polytropic must be in (0, 1], got {eta!r}")
        object.__setattr__(self, "eta_polytropic", eta)
        R = _as_finite_float("gas_constant_R", self.gas_constant_R)
        if R <= 0.0:
            raise ValueError(f"gas_constant_R must be > 0, got {R!r}")
        object.__setattr__(self, "gas_constant_R", R)
        object.__setattr__(self, "alpha", _as_lean_alpha(self.alpha))
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError(f"method must be a non-empty string, got {self.method!r}")
        residual = _as_finite_float("fixed_point_residual_K", self.fixed_point_residual_K)
        tolerance = _as_finite_float("tolerance_K", self.tolerance_K)
        if tolerance < 0.0:
            raise ValueError(f"tolerance_K must be >= 0, got {tolerance!r}")
        if abs(residual) > tolerance:
            raise ValueError(
                f"fixed_point_residual_K exceeds tolerance_K: {residual!r} > {tolerance!r}"
            )
        object.__setattr__(self, "fixed_point_residual_K", residual)
        object.__setattr__(self, "tolerance_K", tolerance)

        if self.T_mean != 0.5 * (self.T_in + self.T_out):
            raise ValueError("T_mean must equal 0.5 * (T_in + T_out)")
        cp, k_source = _textbook_cp_k(self.T_mean, self.alpha, self.gas_constant_R)
        if self.cp != cp or self.k != k_source:
            raise ValueError(
                "cp/k must equal the published true-cp values at T_mean, got "
                f"({self.cp!r}, {self.k!r}) and ({cp!r}, {k_source!r})"
            )
        predicted = self.T_in * self.pressure_ratio ** (
            (self.k - 1.0) / (self.k * self.eta_polytropic)
        )
        if self.fixed_point_residual_K != predicted - self.T_out:
            raise ValueError(
                "fixed_point_residual_K must equal the retained polytropic "
                f"relation residual, got {self.fixed_point_residual_K!r} and "
                f"{predicted - self.T_out!r}"
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

    The opt-in textbook basis is recorded by ``gas_property_basis`` and one
    :class:`CompressorProcessProperties` per unit in ``process_properties``.
    Its exit-state ``cp``/``k`` are local values at the exit temperature and
    each unit work equals the process-mean record ``cp`` times its temperature
    rise.  With the default legacy basis, ``process_properties`` is empty and
    all legacy invariants above are unchanged.
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
    gas_property_basis: str = COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY
    process_properties: tuple[CompressorProcessProperties, ...] = ()

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
        basis = _resolve_gas_property_basis(self.gas_property_basis)
        try:
            process_properties = tuple(self.process_properties)
        except TypeError as exc:
            raise ValueError(
                "process_properties must be an iterable of "
                "CompressorProcessProperties records"
            ) from exc
        if not all(
            isinstance(record, CompressorProcessProperties)
            for record in process_properties
        ):
            raise ValueError(
                "process_properties must contain only "
                "CompressorProcessProperties records"
            )

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
        object.__setattr__(self, "gas_property_basis", basis)
        object.__setattr__(self, "process_properties", process_properties)

        if basis == COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY:
            if process_properties:
                raise ValueError(
                    "process_properties must be empty for the legacy_mean_cp "
                    "basis"
                )
        elif len(process_properties) != len(station_states):
            raise ValueError(
                "textbook_true_cp requires one process record per unit, got "
                f"{len(process_properties)} records for "
                f"{len(station_states)} units"
            )

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
        if basis == COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY:
            for index, (state, work) in enumerate(
                zip(station_states, per_unit_specific_work)
            ):
                expected_work = state.cp * (
                    state.T_total - previous_temperature
                )
                if work != expected_work:
                    raise ValueError(
                        f"per_unit_specific_work[{index}] must equal "
                        f"station_states[{index}].cp * "
                        f"(station_states[{index}].T_total - preceding "
                        f"T_total), got {work!r} and {expected_work!r}"
                    )
                previous_temperature = state.T_total
        else:
            for index, (state, work, record) in enumerate(
                zip(station_states, per_unit_specific_work, process_properties)
            ):
                if record.T_in != previous_temperature:
                    raise ValueError(
                        f"process_properties[{index}].T_in must equal the "
                        f"preceding T_total, got {record.T_in!r} and "
                        f"{previous_temperature!r}"
                    )
                if record.T_out != state.T_total:
                    raise ValueError(
                        f"process_properties[{index}].T_out must equal "
                        f"station_states[{index}].T_total, got "
                        f"{record.T_out!r} and {state.T_total!r}"
                    )
                expected_work = record.cp * (
                    state.T_total - previous_temperature
                )
                if work != expected_work:
                    raise ValueError(
                        f"per_unit_specific_work[{index}] must equal the "
                        "process-mean cp times the unit temperature rise, got "
                        f"{work!r} and {expected_work!r}"
                    )
                local_cp, local_k = _textbook_cp_k(
                    state.T_total, record.alpha, record.gas_constant_R
                )
                if state.cp != local_cp or state.k != local_k:
                    raise ValueError(
                        f"station_states[{index}].cp/k must be the local "
                        "textbook true-cp values at the exit temperature, got "
                        f"({state.cp!r}, {state.k!r}) and "
                        f"({local_cp!r}, {local_k!r})"
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
    *,
    gas_property_basis: str = COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY,
) -> CompressorResult:
    """Apply independent lumped 0D compressor units in series.

    ``pressure_ratios`` are explicit finite positive mathematical inputs.  They
    are neither normalised nor constrained to a preferred operating range, and
    their arithmetic product is preserved for diagnostics.  Each unit uses the
    same current cycle polytropic efficiency and re-evaluates mean ``k`` from
    its own inlet and exit temperatures.

    ``gas_property_basis`` accepts only
    ``"legacy_mean_cp"`` (default, unchanged legacy arithmetic) or
    ``"textbook_true_cp"`` (opt-in process-mean true-cp variant with per-unit
    :class:`CompressorProcessProperties` records and local exit states).
    """

    if not isinstance(inlet, StationState):
        raise TypeError(
            f"inlet must be a StationState, got {type(inlet).__name__}"
        )
    if not isinstance(config, CycleConfig):
        raise TypeError(
            f"config must be a CycleConfig, got {type(config).__name__}"
        )
    basis = _resolve_gas_property_basis(gas_property_basis)

    ratios = _normalise_pressure_ratios(pressure_ratios)
    state = inlet
    station_states: list[StationState] = []
    per_unit_specific_work: list[float] = []
    process_properties: list[CompressorProcessProperties] = []
    product = 1.0

    for ratio in ratios:
        p_out = state.p_total * ratio
        product = product * ratio
        if basis == COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY:
            T_out, k = _compressor_exit_temperature(
                state.T_total, ratio, config, SOLVER_TOL, SOLVER_MAX_ITER
            )
            cp = config.R * k / (k - 1.0)
            work = cp * (T_out - state.T_total)
            exit_state = StationState(
                T_total=T_out,
                p_total=p_out,
                mass_flow=state.mass_flow,
                cp=cp,
                k=k,
            )
        else:
            T_out, _ = calc_compressor(
                ratio,
                config,
                tol=SOLVER_TOL,
                nmax=SOLVER_MAX_ITER,
                T_in=state.T_total,
                gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
            )
            T_mean = 0.5 * (state.T_total + T_out)
            cp, k = _textbook_cp_k(T_mean, math.inf, config.R)
            work = cp * (T_out - state.T_total)
            predicted = state.T_total * ratio ** (
                (k - 1.0) / (k * config.eta_compressor_poly)
            )
            process_properties.append(
                CompressorProcessProperties(
                    T_in=state.T_total,
                    T_out=T_out,
                    T_mean=T_mean,
                    cp=cp,
                    k=k,
                    pressure_ratio=ratio,
                    eta_polytropic=config.eta_compressor_poly,
                    gas_constant_R=config.R,
                    alpha=math.inf,
                    method=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
                    fixed_point_residual_K=predicted - T_out,
                    tolerance_K=SOLVER_TOL,
                )
            )
            local_cp, local_k = _textbook_cp_k(T_out, math.inf, config.R)
            exit_state = StationState(
                T_total=T_out,
                p_total=p_out,
                mass_flow=state.mass_flow,
                cp=local_cp,
                k=local_k,
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
        gas_property_basis=basis,
        process_properties=tuple(process_properties),
    )


def compressor_unit(
    inlet: StationState,
    pressure_ratio: float,
    config: CycleConfig = DEFAULT_CONFIG,
    *,
    gas_property_basis: str = COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY,
) -> CompressorResult:
    """Apply one independent lumped 0D compressor unit."""

    ratio = _as_pressure_ratio("pressure_ratio", pressure_ratio)
    return compressor_train(
        inlet,
        (ratio,),
        config,
        gas_property_basis=gas_property_basis,
    )


__all__ = [
    "COMPRESSOR_GAS_PROPERTY_BASES",
    "COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY",
    "COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK",
    "CompressorProcessProperties",
    "CompressorResult",
    "compressor_train",
    "compressor_unit",
]
