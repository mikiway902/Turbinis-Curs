"""Executable 0D reference cycles for the three selected architectures.

SCOPE
=====
This module adds an **opt-in reference cycle** on top of the extracted
components.  It does not replace :func:`turbine_engine.cycle.calculate_cycle_point`
and it does not wire the legacy single-point cycle, the baseline, the notebooks,
or the optimizer to a new formulation.  The legacy cycle remains the behavioural
source of truth for the current engine and is only compared where its
assumptions are explicitly reproduced.

The implemented variant is named
``uncooled_no_bleed_0d_ideal_mixing``.  Its assumptions are explicit:

* no turbine cooling flow, no leak, no bleed, and no cooling-air return flow;
* one combustor in every architecture; all compressor air enters it;
* ideal mixing to the requested combustor-exit **total** temperature;
* total-to-total turbine expansions only; the prescribed turbine outlet is a
  total pressure, never a static pressure;
* constant ``cp``/``k`` through each turbine, evaluated once at that turbine's
  inlet total temperature; the compressor keeps the existing variable-k
  mean-property relation;
* one explicit mechanical efficiency per turbine and one explicit gearbox
  efficiency downstream of the pre-reducer power plane;
* fuel/air ratio from the retained :func:`turbine_engine.cycle.calc_g_m`
  relation, or an externally supplied combustor state as a separate closure;
* no compressor map, corrected-flow, speed, diameter, stage count, blade
  geometry, gear ratio, propeller load, or radial-equilibrium model.

The ``alpha`` used for the turbine gas ``cp``/``k`` is not hidden.  The
:class:`PolynomialGasModel` records whether it is the fuel-based value
``1 / (f * l0)`` or the retained empirical ``calc_alpha`` polynomial.

SOURCE EQUATIONS
================
The retained component expressions are:

.. code-block:: text

    compressor unit:  T_out = T_in * pi ** ((k - 1) / (k * eta_c))
                      k     = cp(T_mean) / (cp(T_mean) - R)
                      w_c   = cp(T_mean) * (T_out - T_in)

    combustor:        f     = calc_g_m(T_comp, T_g, config)
                      G_gas = G_air * (1 + f)

    turbine:          eta_t = calc_eta_expansion(pi_t, k, eta_t_poly)
                      w_t   = cp * T_in * (1 - pi_t ** ((1 - k) / k)) * eta_t
                      T_out = T_in - w_t / cp
                      p_out = p_in / pi_t

    shaft A/B:        eta_m * G_gas * w_t - sum(G_air * w_c,i) = N_target
    shaft C (GG):     eta_m_GG * G_gas * w_GG = G_air * w_c
    shaft C (FPT):    eta_m_FPT * G_gas * w_FPT = N_target
    gearbox output:   N_out = N_actual * eta_gearbox

The compressor relation is the current cycle extraction
(:mod:`turbine_engine.cycle`), the turbine relation is
:mod:`turbine_engine.turbine`, and the shaft count/order contract is
:mod:`turbine_engine.shaft`.  The fuel/air relation is the current cycle cell-5
expression, itself an implementation of the book's relative fuel-flow formula
(Mikhal'tsev & Molyakov, *Raschet parametrov tsikla...*, 2014, pp. 19-22).
The turbine total-to-total drop is the turbine book's expression (part 1,
printed p. 18, eq. 8; p. 92, item 140; p. 93, items 150-153).  The shaft
relations follow the same book's prescribed-work closure (printed pp. 5-7,
84, 92-93): in a multishaft engine the compressor turbine is prescribed by
compressor power and the free power turbine by the required output.

TOTAL/STATIC CONTRACT
=====================
Every ``StationState`` in this module is a **total** state (``T_total``,
``p_total``).  The caller supplies ``inlet_total_temperature``,
``inlet_total_pressure`` and ``turbine_outlet_total_pressure``.  No nozzle,
exhaust-velocity, or static/total conversion is performed.  This deliberately
avoids mixing the legacy cycle's nozzle boundary with a total-pressure turbine
boundary.

ARCHITECTURE CLOSURES
=====================
* ``SINGLE_SHAFT_SINGLE_CC``: one compressor unit, one combustor, one turbine
  assembly on the output shaft.  The turbine drives the compressor and the
  requested pre-reducer output.  ``G_air`` is sized from the net specific
  shaft power; if that is non-positive the requested positive output cannot
  be supplied by this closure and a clean
  :class:`ArchitectureCyclePhysicalError` is raised (this is a physical
  inability result, not an acceptance filter).
* ``SINGLE_SHAFT_DUAL_CC``: two compressor units in series with explicit
  per-unit ratios (the product is preserved, not re-split), one combustor, one
  turbine assembly on the output shaft.  Each unit uses its actual inlet
  temperature and pressure; the second unit is not reset to ambient.
* ``FREE_POWER_TURBINE``: one gas-generator compressor and one gas-generator
  turbine on the GG shaft, then a free power turbine on the output shaft.
  The GG turbine work per kilogram of gas is prescribed by
  ``w_c / (eta_m_GG * (1 + f))``.  Its total outlet pressure is obtained by
  inverting the retained expansion relation
  (:func:`turbine_engine.turbine.turbine_pressure_ratio_for_work`).  The actual
  GG outlet :class:`StationState` is passed to the FPT; it is **not** replaced
  by ``T_g`` again.  The FPT expands to the caller's prescribed total outlet
  pressure, and ``G_air`` is sized from the FPT output.  The compressor load is
  never subtracted a second time from the FPT output.

VALIDATION SCOPE / LIMITATIONS
==============================
``ARCHITECTURE_CYCLE_LIMITATIONS`` is stored in every result.  In particular:

* this is **not** a physical validation of any topology;
* ``MISSING_COMPRESSOR_PHYSICS``: no map, blading, corrected flow, speed,
  diameter, diffusion, or surge model;
* no turbine stage count, velocity triangles, cooling, or map;
* no gear ratio, shaft speed, propeller load, accessory loss, or control law;
* the fuel/air relation and the gas-property polynomials retain the current
  project assumptions; they are not independently validated;
* the two gas-property models are deliberately not claimed equivalent: the
  compressor uses mean-temperature variable ``k``, while each turbine holds its
  inlet ``cp``/``k`` constant.  The shaft/mass closure is exact arithmetic for
  the supplied component models; it is not a proof that those models are
  physically complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from turbine_engine.architecture import (
    ARCHITECTURES,
    Architecture,
    StationState,
)
from turbine_engine.compressor import CompressorResult, compressor_train
from turbine_engine.cycle import (
    CycleConfig,
    calc_alpha,
    calc_cp_kerosene,
    calc_g_m,
    calc_k_kerosene,
)
from turbine_engine.shaft import ShaftPowerBalance, evaluate_shaft_power_balance
from turbine_engine.turbine import (
    TurbineResult,
    turbine_expansion,
    turbine_pressure_ratio_for_work,
)


class ArchitectureCycleError(ValueError):
    """Base class for this module's clean domain errors."""


class ArchitectureCyclePhysicalError(ArchitectureCycleError):
    """The supplied mathematical inputs cannot close a positive-output cycle.

    This is not a design/acceptance filter.  It means the requested output
    cannot be produced by the explicitly selected component equations (for
    example a non-positive net specific shaft power or an impossible prescribed
    turbine work).
    """


class AlphaBasis(str, Enum):
    """How turbine gas ``alpha`` is obtained for the ``cp``/``k`` polynomial."""

    FUEL_BASED = "fuel_based"
    EMPIRICAL_POLYNOMIAL = "empirical_polynomial"


#: Stable limitation markers included in every result.
ARCHITECTURE_CYCLE_LIMITATIONS: tuple[str, ...] = (
    "NOT_PHYSICALLY_VALIDATED",
    "UNCOOLED_NO_BLEED_0D_IDEAL_MIXING",
    "MISSING_COMPRESSOR_PHYSICS: no map, blading, corrected flow, speed, "
    "diameter, surge or diffusion model",
    "MISSING_TURBINE_STAGE_PHYSICS: no stage count, velocity triangles, "
    "cooling or map",
    "MISSING_GEARBOX_PROPELLER: no ratio, speed, propeller load or control law",
    "MISSING_COOLING_BLEED: no cooling, leak, bleed or return flow in this variant",
    "MISSING_THERMOCHEMISTRY_VALIDATION: calc_g_m and the selected alpha "
    "polynomial are retained project assumptions, not independently validated",
    "GAS_PROPERTY_BASIS: compressor mean-temperature variable k; turbine "
    "constant cp/k at the turbine inlet total temperature",
    "TOTAL_STATES_ONLY: prescribed turbine outlet is a total pressure",
)


def _as_finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArchitectureCycleError(
            f"{name} must be a real number, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ArchitectureCycleError(
            f"{name} must be finite, got {result!r}"
        )
    return result


def _as_positive_float(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if result <= 0.0:
        raise ArchitectureCycleError(f"{name} must be > 0, got {value!r}")
    return result


def _as_nonnegative_float(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if result < 0.0:
        raise ArchitectureCycleError(f"{name} must be >= 0, got {value!r}")
    return result


def _as_efficiency(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if not 0.0 < result <= 1.0:
        raise ArchitectureCycleError(
            f"{name} must be in (0, 1], got {value!r}"
        )
    return result


def _as_positive_tuple(name: str, values: Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ArchitectureCycleError(
            f"{name} must be an iterable of finite positive numbers, "
            f"got {values!r}"
        )
    try:
        sequence = tuple(values)
    except TypeError as exc:
        raise ArchitectureCycleError(
            f"{name} must be an iterable of finite positive numbers, "
            f"got {values!r}"
        ) from exc
    return tuple(
        _as_positive_float(f"{name}[{index}]", value)
        for index, value in enumerate(sequence)
    )


def _as_efficiency_tuple(
    name: str, values: Iterable[float]
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ArchitectureCycleError(
            f"{name} must be an iterable of efficiencies in (0, 1], "
            f"got {values!r}"
        )
    try:
        sequence = tuple(values)
    except TypeError as exc:
        raise ArchitectureCycleError(
            f"{name} must be an iterable of efficiencies in (0, 1], "
            f"got {values!r}"
        ) from exc
    return tuple(
        _as_efficiency(f"{name}[{index}]", value)
        for index, value in enumerate(sequence)
    )


@dataclass(frozen=True)
class PolynomialGasModel:
    """Explicit gas-property and fuel-relation variant.

    ``R`` is the specific gas constant in J/(kg K); ``l0`` the stoichiometric
    air/fuel mass ratio; ``q_lhv_kj`` the lower heating value in kJ/kg;
    ``eta_combustion`` the combustion completeness; and ``T_ref`` the reference
    temperature used by the retained ``calc_g_m`` relation.  These values are
    not new engineering assumptions supplied by this module: the caller must
    pass them, and the current project values are documented in the README and
    the legacy :class:`~turbine_engine.cycle.CycleConfig`.

    ``alpha_basis`` chooses the excess-air ratio used only for the turbine gas
    ``cp``/``k`` polynomial.  ``FUEL_BASED`` uses ``1 / (f * l0)``;
    ``EMPIRICAL_POLYNOMIAL`` uses the retained ``calc_alpha`` polynomial.  The
    choice is stored in the result and is never silently selected.
    """

    R: float
    l0: float
    q_lhv_kj: float
    eta_combustion: float
    T_ref: float
    alpha_basis: AlphaBasis = AlphaBasis.FUEL_BASED
    model_variant: str = (
        "legacy_cp_polynomial_compressor_mean_turbine_inlet_constant"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "R", _as_positive_float("R", self.R))
        object.__setattr__(self, "l0", _as_positive_float("l0", self.l0))
        object.__setattr__(
            self, "q_lhv_kj", _as_positive_float("q_lhv_kj", self.q_lhv_kj)
        )
        object.__setattr__(
            self,
            "eta_combustion",
            _as_efficiency("eta_combustion", self.eta_combustion),
        )
        object.__setattr__(self, "T_ref", _as_positive_float("T_ref", self.T_ref))

        basis = self.alpha_basis
        if not isinstance(basis, AlphaBasis):
            try:
                basis = AlphaBasis(basis)
            except (TypeError, ValueError) as exc:
                raise ArchitectureCycleError(
                    "alpha_basis must be an AlphaBasis member, "
                    f"got {self.alpha_basis!r}"
                ) from exc
        object.__setattr__(self, "alpha_basis", basis)

        if not isinstance(self.model_variant, str) or not self.model_variant.strip():
            raise ArchitectureCycleError(
                "model_variant must be a non-empty string, "
                f"got {self.model_variant!r}"
            )

    def _legacy_config(self) -> CycleConfig:
        # Only R, l0, q_lhv_kj, eta_combustion and T_ref are read by the
        # retained calc_g_m relation.  Other CycleConfig fields are unused here.
        return CycleConfig(
            R=self.R,
            l0=self.l0,
            q_lhv_kj=self.q_lhv_kj,
            eta_combustion=self.eta_combustion,
            T_ref=self.T_ref,
        )

    def air_cp_k(self, T_total: float) -> tuple[float, float]:
        """Return dry-air ``(cp, k)`` at a total temperature (K)."""

        T = _as_positive_float("T_total", T_total)
        cp = calc_cp_kerosene(T, math.inf)
        k = calc_k_kerosene(T, math.inf, self.R)
        return cp, k

    def gas_cp_k(
        self, T_total: float, alpha: float
    ) -> tuple[float, float]:
        """Return combustion-product ``(cp, k)`` at ``T_total`` and ``alpha``."""

        T = _as_positive_float("T_total", T_total)
        alpha_value = _as_positive_float("alpha", alpha)
        cp = calc_cp_kerosene(T, alpha_value)
        k = calc_k_kerosene(T, alpha_value, self.R)
        return cp, k

    def fuel_air_ratio(
        self, T_compressor_exit: float, T_combustor_exit: float
    ) -> float:
        """Return fuel/air mass ratio from the retained ``calc_g_m`` relation.

        The basis is one kilogram of compressor air, with all compressor air
        entering the combustor (no bleed/leak/cooling in this variant).  The
        returned value is the raw source relation and is not clamped.
        """

        T_in = _as_positive_float("T_compressor_exit", T_compressor_exit)
        T_g = _as_positive_float("T_combustor_exit", T_combustor_exit)
        return calc_g_m(T_in, T_g, self._legacy_config())

    def alpha_for(
        self,
        T_compressor_exit: float,
        T_combustor_exit: float,
        fuel_air_ratio: float,
    ) -> float:
        """Return the explicitly selected turbine-gas alpha basis."""

        f = _as_finite_float("fuel_air_ratio", fuel_air_ratio)
        if f <= 0.0:
            raise ArchitectureCyclePhysicalError(
                "fuel_air_ratio must be > 0 for a combustor with heat addition, "
                f"got {fuel_air_ratio!r}"
            )
        if self.alpha_basis is AlphaBasis.FUEL_BASED:
            return 1.0 / (f * self.l0)
        return calc_alpha(
            _as_positive_float("T_compressor_exit", T_compressor_exit),
            _as_positive_float("T_combustor_exit", T_combustor_exit),
        )


@dataclass(frozen=True)
class ExternalCombustorState:
    """An externally supplied combustor-exit total state and fuel flow.

    This is the supported no-inference fallback: the caller supplies the total
    outlet state, the fuel/air ratio, and the gas ``cp``/``k``.  The module then
    only sizes flow and closes the shafts; it does not infer fuel flow,
    composition, ``alpha``, or gas properties.  The state is not a
    ``StationState`` because its mass flow follows from the solved air flow.
    """

    T_total: float
    p_total: float
    fuel_air_ratio: float
    cp: float
    k: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "T_total", _as_positive_float("T_total", self.T_total)
        )
        object.__setattr__(
            self, "p_total", _as_positive_float("p_total", self.p_total)
        )
        object.__setattr__(
            self,
            "fuel_air_ratio",
            _as_nonnegative_float("fuel_air_ratio", self.fuel_air_ratio),
        )
        object.__setattr__(self, "cp", _as_positive_float("cp", self.cp))
        k = _as_finite_float("k", self.k)
        if k <= 1.0:
            raise ArchitectureCycleError(f"k must be > 1, got {self.k!r}")
        object.__setattr__(self, "k", k)


@dataclass(frozen=True)
class LegacyCombustorClosure:
    """Combustor target temperature and total-pressure recovery.

    ``outlet_total_temperature`` is the requested combustor-exit total
    temperature.  ``total_pressure_recovery`` multiplies the compressor-exit
    total pressure to give the combustor-exit/turbine-inlet total pressure.
    """

    outlet_total_temperature: float
    total_pressure_recovery: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outlet_total_temperature",
            _as_positive_float(
                "outlet_total_temperature", self.outlet_total_temperature
            ),
        )
        object.__setattr__(
            self,
            "total_pressure_recovery",
            _as_efficiency(
                "total_pressure_recovery", self.total_pressure_recovery
            ),
        )


@dataclass(frozen=True)
class ExternalCombustorClosure:
    """Wrap an externally supplied combustor-exit state."""

    state: ExternalCombustorState

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExternalCombustorState):
            raise ArchitectureCycleError(
                "state must be an ExternalCombustorState, "
                f"got {type(self.state).__name__}"
            )


@dataclass(frozen=True)
class ArchitectureCycleInput:
    """Explicit inputs for one architecture-specific reference cycle.

    The compressor pressure ratios are per-unit values in series order.  For
    ``SINGLE_SHAFT_DUAL_CC`` exactly two are required; for the other
    architectures exactly one.  They are not normalised or re-split, and their
    arithmetic product is reported.

    The turbine efficiency tuples are ordered by architecture exactly as in
    :mod:`turbine_engine.shaft`: one value for the single-shaft architectures,
    and ``(GG turbine, free power turbine)`` for ``FREE_POWER_TURBINE``.
    Mechanical efficiencies use the same order.

    ``turbine_outlet_total_pressure`` is the total outlet pressure of the main
    turbine for the single-shaft architectures and of the free power turbine
    for ``FREE_POWER_TURBINE``.  The GG turbine outlet pressure is solved from
    its prescribed work and is not supplied.

    ``combustor`` is either a :class:`LegacyCombustorClosure` (fuel flow from
    ``calc_g_m``) or an :class:`ExternalCombustorClosure` (explicit state and
    fuel flow).
    """

    architecture: Architecture
    inlet_total_temperature: float
    inlet_total_pressure: float
    compressor_pressure_ratios: tuple[float, ...]
    compressor_polytropic_efficiency: float
    combustor: LegacyCombustorClosure | ExternalCombustorClosure
    turbine_outlet_total_pressure: float
    turbine_polytropic_efficiencies: tuple[float, ...]
    turbine_mechanical_efficiencies: tuple[float, ...]
    target_gearbox_input_power: float
    gearbox_efficiency: float
    gas_model: PolynomialGasModel

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, Architecture):
            raise ArchitectureCycleError(
                "architecture must be an Architecture member, "
                f"got {self.architecture!r}"
            )
        descriptor = ARCHITECTURES[self.architecture]

        object.__setattr__(
            self,
            "inlet_total_temperature",
            _as_positive_float(
                "inlet_total_temperature", self.inlet_total_temperature
            ),
        )
        object.__setattr__(
            self,
            "inlet_total_pressure",
            _as_positive_float(
                "inlet_total_pressure", self.inlet_total_pressure
            ),
        )
        object.__setattr__(
            self,
            "compressor_pressure_ratios",
            _as_positive_tuple(
                "compressor_pressure_ratios",
                self.compressor_pressure_ratios,
            ),
        )
        object.__setattr__(
            self,
            "compressor_polytropic_efficiency",
            _as_efficiency(
                "compressor_polytropic_efficiency",
                self.compressor_polytropic_efficiency,
            ),
        )
        if not isinstance(
            self.combustor,
            (LegacyCombustorClosure, ExternalCombustorClosure),
        ):
            raise ArchitectureCycleError(
                "combustor must be LegacyCombustorClosure or "
                "ExternalCombustorClosure, "
                f"got {type(self.combustor).__name__}"
            )
        object.__setattr__(
            self,
            "turbine_outlet_total_pressure",
            _as_positive_float(
                "turbine_outlet_total_pressure",
                self.turbine_outlet_total_pressure,
            ),
        )
        object.__setattr__(
            self,
            "turbine_polytropic_efficiencies",
            _as_efficiency_tuple(
                "turbine_polytropic_efficiencies",
                self.turbine_polytropic_efficiencies,
            ),
        )
        object.__setattr__(
            self,
            "turbine_mechanical_efficiencies",
            _as_efficiency_tuple(
                "turbine_mechanical_efficiencies",
                self.turbine_mechanical_efficiencies,
            ),
        )
        object.__setattr__(
            self,
            "target_gearbox_input_power",
            _as_positive_float(
                "target_gearbox_input_power",
                self.target_gearbox_input_power,
            ),
        )
        object.__setattr__(
            self,
            "gearbox_efficiency",
            _as_efficiency("gearbox_efficiency", self.gearbox_efficiency),
        )
        if not isinstance(self.gas_model, PolynomialGasModel):
            raise ArchitectureCycleError(
                "gas_model must be a PolynomialGasModel, "
                f"got {type(self.gas_model).__name__}"
            )

        if len(self.compressor_pressure_ratios) != descriptor.compressor_units:
            raise ArchitectureCycleError(
                f"{self.architecture.value} requires exactly "
                f"{descriptor.compressor_units} compressor pressure ratio(s), "
                f"got {len(self.compressor_pressure_ratios)}"
            )
        if (
            len(self.turbine_polytropic_efficiencies)
            != descriptor.turbine_assemblies
        ):
            raise ArchitectureCycleError(
                f"{self.architecture.value} requires exactly "
                f"{descriptor.turbine_assemblies} turbine polytropic "
                "efficiency value(s), got "
                f"{len(self.turbine_polytropic_efficiencies)}"
            )
        if (
            len(self.turbine_mechanical_efficiencies)
            != descriptor.turbine_assemblies
        ):
            raise ArchitectureCycleError(
                f"{self.architecture.value} requires exactly "
                f"{descriptor.turbine_assemblies} turbine mechanical "
                "efficiency value(s), got "
                f"{len(self.turbine_mechanical_efficiencies)}"
            )


@dataclass(frozen=True)
class CombustorClosureResult:
    """Mass and state record for the explicit combustor closure.

    ``inlet`` is the compressor-exit air state, ``outlet`` the turbine-inlet
    gas state, ``fuel_mass_flow`` and ``gas_mass_flow`` are absolute flows in
    kg/s, and ``alpha`` is ``None`` only for the external-state closure.  The
    mass balance is ``gas = air + fuel`` with no bleed/leak/cooling terms.
    """

    inlet: StationState
    outlet: StationState
    fuel_air_ratio: float
    alpha: float | None
    alpha_basis: str
    closure: str
    fuel_mass_flow: float
    gas_mass_flow: float
    heat_input: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.inlet, StationState):
            raise ArchitectureCycleError(
                f"inlet must be a StationState, got {type(self.inlet).__name__}"
            )
        if not isinstance(self.outlet, StationState):
            raise ArchitectureCycleError(
                f"outlet must be a StationState, got {type(self.outlet).__name__}"
            )
        f = _as_nonnegative_float("fuel_air_ratio", self.fuel_air_ratio)
        fuel = _as_nonnegative_float("fuel_mass_flow", self.fuel_mass_flow)
        gas = _as_positive_float("gas_mass_flow", self.gas_mass_flow)
        object.__setattr__(self, "fuel_air_ratio", f)
        object.__setattr__(self, "fuel_mass_flow", fuel)
        object.__setattr__(self, "gas_mass_flow", gas)

        alpha = self.alpha
        if alpha is not None:
            alpha = _as_positive_float("alpha", alpha)
            object.__setattr__(self, "alpha", alpha)
        for name in ("alpha_basis", "closure"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ArchitectureCycleError(
                    f"{name} must be a non-empty string, got {value!r}"
                )
        heat_input = self.heat_input
        if heat_input is not None:
            heat_input = _as_nonnegative_float("heat_input", heat_input)
            object.__setattr__(self, "heat_input", heat_input)

        if self.outlet.mass_flow != gas:
            raise ArchitectureCycleError(
                "gas_mass_flow must equal outlet.mass_flow, got "
                f"{gas!r} and {self.outlet.mass_flow!r}"
            )
        expected_fuel = self.inlet.mass_flow * f
        if not math.isclose(
            fuel, expected_fuel, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ArchitectureCycleError(
                "fuel_mass_flow must equal inlet.mass_flow * fuel_air_ratio, "
                f"got {fuel!r} and {expected_fuel!r}"
            )
        expected_gas = self.inlet.mass_flow + fuel
        if not math.isclose(
            gas, expected_gas, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ArchitectureCycleError(
                "gas_mass_flow must equal inlet.mass_flow + fuel_mass_flow, "
                f"got {gas!r} and {expected_gas!r}"
            )

    @property
    def mass_balance_residual(self) -> float:
        """``gas - (air + fuel)`` in kg/s (unfiltered arithmetic)."""

        return self.gas_mass_flow - (
            self.inlet.mass_flow + self.fuel_mass_flow
        )


@dataclass(frozen=True)
class ArchitectureCycleResult:
    """Immutable result of one architecture-specific reference cycle.

    Exact, unfiltered closure residuals are available on the nested component
    records (:attr:`compressor`, :attr:`turbines`, :attr:`shaft`, and
    :attr:`combustor`) and are repeated here for convenient serialization.
    ``net_specific_power_per_air`` is the MAIN net shaft power per kg air for
    the single-shaft topologies and the FPT output power per kg air for the
    free-power topology (the GG shaft is balanced separately).
    ``target_power_status`` is a **reporting** status, not a feasibility filter:
    ``"met"`` means the solved actual gearbox-input power matches the request
    to numerical tolerance; no engineering range or map filter is applied.
    """

    architecture: Architecture
    model_variant: str
    combustor_closure: str
    gas_model: PolynomialGasModel
    validation_scope: tuple[str, ...]
    inlet: StationState
    compressor: CompressorResult
    combustor: CombustorClosureResult
    turbines: tuple[TurbineResult, ...]
    shaft: ShaftPowerBalance
    air_mass_flow: float
    fuel_mass_flow: float
    gas_mass_flow: float
    target_gearbox_input_power: float
    actual_gearbox_input_power: float
    gearbox_output_power: float
    net_specific_power_per_air: float
    pressure_ratio_product: float
    compressor_pressure_residual: float
    turbine_pressure_residuals: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, Architecture):
            raise ArchitectureCycleError(
                "architecture must be an Architecture member, "
                f"got {self.architecture!r}"
            )
        descriptor = ARCHITECTURES[self.architecture]
        if not isinstance(self.inlet, StationState):
            raise ArchitectureCycleError(
                f"inlet must be a StationState, got {type(self.inlet).__name__}"
            )
        if not isinstance(self.compressor, CompressorResult):
            raise ArchitectureCycleError(
                "compressor must be a CompressorResult, "
                f"got {type(self.compressor).__name__}"
            )
        if not isinstance(self.combustor, CombustorClosureResult):
            raise ArchitectureCycleError(
                "combustor must be a CombustorClosureResult, "
                f"got {type(self.combustor).__name__}"
            )
        if not isinstance(self.gas_model, PolynomialGasModel):
            raise ArchitectureCycleError(
                "gas_model must be a PolynomialGasModel, "
                f"got {type(self.gas_model).__name__}"
            )
        if not isinstance(self.shaft, ShaftPowerBalance):
            raise ArchitectureCycleError(
                f"shaft must be a ShaftPowerBalance, got {type(self.shaft).__name__}"
            )
        if self.shaft.architecture is not self.architecture:
            raise ArchitectureCycleError(
                "shaft architecture must match result architecture, got "
                f"{self.shaft.architecture!r} and {self.architecture!r}"
            )

        for name in ("model_variant", "combustor_closure"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ArchitectureCycleError(
                    f"{name} must be a non-empty string, got {value!r}"
                )
        scope = self.validation_scope
        if isinstance(scope, (str, bytes)):
            raise ArchitectureCycleError(
                "validation_scope must be an iterable of non-empty strings"
            )
        try:
            scope = tuple(scope)
        except TypeError as exc:
            raise ArchitectureCycleError(
                "validation_scope must be an iterable of non-empty strings"
            ) from exc
        if not scope or any(
            not isinstance(item, str) or not item.strip() for item in scope
        ):
            raise ArchitectureCycleError(
                "validation_scope must contain non-empty strings"
            )
        object.__setattr__(self, "validation_scope", scope)

        try:
            turbines = tuple(self.turbines)
        except TypeError as exc:
            raise ArchitectureCycleError(
                "turbines must be an iterable of TurbineResult"
            ) from exc
        if not all(isinstance(item, TurbineResult) for item in turbines):
            raise ArchitectureCycleError(
                "turbines must contain only TurbineResult instances"
            )
        if len(turbines) != descriptor.turbine_assemblies:
            raise ArchitectureCycleError(
                f"{self.architecture.value} requires exactly "
                f"{descriptor.turbine_assemblies} turbine result(s), got "
                f"{len(turbines)}"
            )
        object.__setattr__(self, "turbines", turbines)

        air = _as_positive_float("air_mass_flow", self.air_mass_flow)
        fuel = _as_nonnegative_float("fuel_mass_flow", self.fuel_mass_flow)
        gas = _as_positive_float("gas_mass_flow", self.gas_mass_flow)
        target = _as_positive_float(
            "target_gearbox_input_power", self.target_gearbox_input_power
        )
        actual = _as_finite_float(
            "actual_gearbox_input_power", self.actual_gearbox_input_power
        )
        output = _as_finite_float(
            "gearbox_output_power", self.gearbox_output_power
        )
        net_specific = _as_finite_float(
            "net_specific_power_per_air", self.net_specific_power_per_air
        )
        product = _as_positive_float(
            "pressure_ratio_product", self.pressure_ratio_product
        )
        compressor_residual = _as_finite_float(
            "compressor_pressure_residual", self.compressor_pressure_residual
        )
        try:
            turbine_residuals = tuple(
                _as_finite_float(
                    f"turbine_pressure_residuals[{index}]", value
                )
                for index, value in enumerate(self.turbine_pressure_residuals)
            )
        except TypeError as exc:
            raise ArchitectureCycleError(
                "turbine_pressure_residuals must be an iterable of finite numbers"
            ) from exc

        object.__setattr__(self, "air_mass_flow", air)
        object.__setattr__(self, "fuel_mass_flow", fuel)
        object.__setattr__(self, "gas_mass_flow", gas)
        object.__setattr__(self, "target_gearbox_input_power", target)
        object.__setattr__(self, "actual_gearbox_input_power", actual)
        object.__setattr__(self, "gearbox_output_power", output)
        object.__setattr__(self, "net_specific_power_per_air", net_specific)
        object.__setattr__(self, "pressure_ratio_product", product)
        object.__setattr__(
            self, "compressor_pressure_residual", compressor_residual
        )
        object.__setattr__(
            self, "turbine_pressure_residuals", turbine_residuals
        )

        if self.inlet != self.compressor.inlet:
            raise ArchitectureCycleError(
                "inlet must equal compressor.inlet"
            )
        if air != self.compressor.inlet.mass_flow:
            raise ArchitectureCycleError(
                "air_mass_flow must equal compressor.inlet.mass_flow, got "
                f"{air!r} and {self.compressor.inlet.mass_flow!r}"
            )
        if self.combustor.inlet != self.compressor.outlet:
            raise ArchitectureCycleError(
                "combustor.inlet must equal compressor.outlet"
            )
        if fuel != self.combustor.fuel_mass_flow:
            raise ArchitectureCycleError(
                "fuel_mass_flow must equal combustor.fuel_mass_flow"
            )
        if gas != self.combustor.gas_mass_flow:
            raise ArchitectureCycleError(
                "gas_mass_flow must equal combustor.gas_mass_flow"
            )
        expected_product = self.compressor.pressure_ratio_product
        if not math.isclose(
            product, expected_product, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ArchitectureCycleError(
                "pressure_ratio_product must equal the compressor product, got "
                f"{product!r} and {expected_product!r}"
            )
        if compressor_residual != self.compressor.residual:
            raise ArchitectureCycleError(
                "compressor_pressure_residual must equal compressor.residual"
            )
        expected_turbine_residuals = tuple(
            item.residual for item in turbines
        )
        if turbine_residuals != expected_turbine_residuals:
            raise ArchitectureCycleError(
                "turbine_pressure_residuals must equal the turbine residuals"
            )
        if turbines[0].inlet != self.combustor.outlet:
            raise ArchitectureCycleError(
                "the first turbine inlet must equal combustor.outlet"
            )
        for index in range(1, len(turbines)):
            if turbines[index].inlet != turbines[index - 1].outlet:
                raise ArchitectureCycleError(
                    "turbine result " f"{index} inlet must equal the preceding "
                    "turbine outlet"
                )
        for index, turbine in enumerate(turbines):
            if turbine.inlet.mass_flow != gas:
                raise ArchitectureCycleError(
                    f"turbines[{index}].inlet.mass_flow must equal "
                    f"gas_mass_flow, got {turbine.inlet.mass_flow!r} and "
                    f"{gas!r}"
                )
        if target != self.shaft.required_gearbox_input_power:
            raise ArchitectureCycleError(
                "target_gearbox_input_power must equal the shaft request"
            )
        if not math.isclose(
            actual,
            self.shaft.actual_gearbox_input_power,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ArchitectureCycleError(
                "actual_gearbox_input_power must equal the shaft actual input, "
                f"got {actual!r} and {self.shaft.actual_gearbox_input_power!r}"
            )
        if not math.isclose(
            output,
            self.shaft.gearbox_output_power,
            rel_tol=1e-12,
            abs_tol=1e-6,
        ):
            raise ArchitectureCycleError(
                "gearbox_output_power must equal the shaft gearbox output, got "
                f"{output!r} and {self.shaft.gearbox_output_power!r}"
            )
        expected_gearbox_output = actual * self.shaft.gearbox_efficiency
        if not math.isclose(
            output, expected_gearbox_output, rel_tol=1e-12, abs_tol=1e-6
        ):
            raise ArchitectureCycleError(
                "gearbox_output_power must equal actual input * gearbox "
                f"efficiency, got {output!r} and {expected_gearbox_output!r}"
            )

    @property
    def main_turbine(self) -> TurbineResult | None:
        """The single-shaft turbine, or ``None`` for the free-power topology."""

        if self.architecture is Architecture.FREE_POWER_TURBINE:
            return None
        return self.turbines[0]

    @property
    def gas_generator_turbine(self) -> TurbineResult | None:
        """The gas-generator turbine, or ``None`` for single-shaft topologies."""

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return self.turbines[0]

    @property
    def free_power_turbine(self) -> TurbineResult | None:
        """The free power turbine, or ``None`` for single-shaft topologies."""

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return self.turbines[1]

    @property
    def mass_balance_residual(self) -> float:
        """``gas - (air + fuel)`` in kg/s."""

        return self.gas_mass_flow - (self.air_mass_flow + self.fuel_mass_flow)

    @property
    def target_power_residual(self) -> float:
        """Actual pre-reducer power minus the requested target, in W."""

        return self.actual_gearbox_input_power - self.target_gearbox_input_power

    @property
    def target_power_status(self) -> str:
        """Reporting status: ``"met"`` or ``"not_met"`` (not a filter)."""

        if math.isclose(
            self.actual_gearbox_input_power,
            self.target_gearbox_input_power,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            return "met"
        return "not_met"

    def residual_report(self) -> dict[str, float]:
        """Return the unfiltered scalar residuals keyed by name."""

        report: dict[str, float] = {
            "compressor_pressure": self.compressor_pressure_residual,
            "mass_balance": self.mass_balance_residual,
            "target_power": self.target_power_residual,
        }
        for index, value in enumerate(self.turbine_pressure_residuals):
            report[f"turbine_{index}_pressure"] = value
        for name, value in (
            ("shaft_main", self.shaft.main_residual),
            ("shaft_gg", self.shaft.gas_generator_residual),
            ("shaft_fpt", self.shaft.free_power_turbine_residual),
        ):
            if value is not None:
                report[name] = value
        return report


def _compressor_config(
    request: ArchitectureCycleInput, inlet_k: float
) -> CycleConfig:
    """Build the minimal legacy config used by the retained compressor helper.

    ``_compressor_exit_temperature`` reads only ``R``, ``k_air`` (initial
    guess), and ``eta_compressor_poly``.  The other fields are required by the
    retained ``CycleConfig`` contract but are not used by that helper.
    """

    return CycleConfig(
        T_a_full=request.inlet_total_temperature,
        p_a=request.inlet_total_pressure,
        p_a_full=request.inlet_total_pressure,
        k_air=inlet_k,
        R=request.gas_model.R,
        eta_compressor_poly=request.compressor_polytropic_efficiency,
    )


def _air_state(
    gas_model: PolynomialGasModel,
    T_total: float,
    p_total: float,
    mass_flow: float,
    identifier: str = "compressor_inlet",
) -> StationState:
    cp, k = gas_model.air_cp_k(T_total)
    return StationState(
        T_total=T_total,
        p_total=p_total,
        mass_flow=mass_flow,
        cp=cp,
        k=k,
        identifier=identifier,
    )


def _gas_state(
    T_total: float,
    p_total: float,
    mass_flow: float,
    cp: float,
    k: float,
    identifier: str = "turbine_inlet",
) -> StationState:
    return StationState(
        T_total=T_total,
        p_total=p_total,
        mass_flow=mass_flow,
        cp=cp,
        k=k,
        identifier=identifier,
    )


def _resolve_combustor(
    request: ArchitectureCycleInput,
    compressor: CompressorResult,
) -> tuple[float, float, float, float, float | None, str, str, float | None]:
    """Return ``(f, alpha, p_out, cp, k, heat_per_air, basis, closure)``.

    ``heat_per_air`` is in J per kilogram of compressor air and is ``None``
    for the external-state closure, where no heat-input relation is inferred.
    """

    compressor_outlet = compressor.outlet
    closure = request.combustor
    if isinstance(closure, LegacyCombustorClosure):
        T_g = closure.outlet_total_temperature
        if T_g <= compressor_outlet.T_total:
            raise ArchitectureCyclePhysicalError(
                "combustor outlet total temperature must exceed the compressor "
                f"outlet total temperature, got T_g={T_g!r} K and "
                f"T_comp={compressor_outlet.T_total!r} K"
            )
        try:
            f = request.gas_model.fuel_air_ratio(
                compressor_outlet.T_total, T_g
            )
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            raise ArchitectureCyclePhysicalError(
                "calc_g_m is undefined for the supplied combustor "
                f"temperatures T_comp={compressor_outlet.T_total!r} K and "
                f"T_g={T_g!r} K: {exc}"
            ) from exc
        if not math.isfinite(f) or f <= 0.0:
            raise ArchitectureCyclePhysicalError(
                "calc_g_m returned a non-positive/non-finite fuel/air ratio "
                f"{f!r} for T_comp={compressor_outlet.T_total!r} K and "
                f"T_g={T_g!r} K"
            )
        try:
            alpha = request.gas_model.alpha_for(
                compressor_outlet.T_total, T_g, f
            )
            cp, k = request.gas_model.gas_cp_k(T_g, alpha)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            raise ArchitectureCyclePhysicalError(
                "combustor gas properties are undefined for the selected "
                f"alpha basis {request.gas_model.alpha_basis.value!r} and "
                f"T_g={T_g!r} K: {exc}"
            ) from exc
        p_out = (
            compressor_outlet.p_total * closure.total_pressure_recovery
        )
        heat_per_air = f * request.gas_model.q_lhv_kj * 1.0e3 * (
            request.gas_model.eta_combustion
        )
        return (
            f,
            alpha,
            p_out,
            cp,
            k,
            heat_per_air,
            request.gas_model.alpha_basis.value,
            "legacy_calc_g_m_no_bleed",
        )

    if isinstance(closure, ExternalCombustorClosure):
        external = closure.state
        if external.T_total <= compressor_outlet.T_total:
            raise ArchitectureCyclePhysicalError(
                "external combustor outlet total temperature must exceed the "
                "compressor outlet total temperature, got "
                f"T_ext={external.T_total!r} K and "
                f"T_comp={compressor_outlet.T_total!r} K"
            )
        if external.fuel_air_ratio <= 0.0:
            raise ArchitectureCyclePhysicalError(
                "external combustor fuel/air ratio must be > 0 when its "
                "outlet temperature exceeds the compressor outlet, got "
                f"{external.fuel_air_ratio!r}"
            )
        return (
            external.fuel_air_ratio,
            None,
            external.p_total,
            external.cp,
            external.k,
            None,
            "external_state_no_inference",
            "external_state",
        )

    raise ArchitectureCycleError(
        "combustor must be LegacyCombustorClosure or "
        "ExternalCombustorClosure"
    )


def _combustor_result(
    inlet: StationState,
    T_g: float,
    p_g: float,
    cp_g: float,
    k_g: float,
    fuel_air_ratio: float,
    alpha: float | None,
    alpha_basis: str,
    closure: str,
    heat_per_air: float | None,
) -> CombustorClosureResult:
    air = inlet.mass_flow
    fuel = air * fuel_air_ratio
    gas = air + fuel
    outlet = _gas_state(
        T_g,
        p_g,
        gas,
        cp_g,
        k_g,
        identifier="combustor_exit",
    )
    heat_input = None if heat_per_air is None else heat_per_air * air
    return CombustorClosureResult(
        inlet=inlet,
        outlet=outlet,
        fuel_air_ratio=fuel_air_ratio,
        alpha=alpha,
        alpha_basis=alpha_basis,
        closure=closure,
        fuel_mass_flow=fuel,
        gas_mass_flow=gas,
        heat_input=heat_input,
    )


def _physical_error_from_turbine(
    exc: Exception, context: str
) -> ArchitectureCyclePhysicalError:
    return ArchitectureCyclePhysicalError(f"{context}: {exc}")


def calculate_architecture_cycle(
    request: ArchitectureCycleInput,
) -> ArchitectureCycleResult:
    """Evaluate one explicit 0D reference cycle for the selected architecture.

    The function is stateless and performs no sweep, optimization, map filter,
    or acceptance rejection.  It raises :class:`ArchitectureCyclePhysicalError`
    when the requested positive output cannot be closed by the supplied
    equations, and :class:`ArchitectureCycleError` for invalid configuration or
    mathematical-domain inputs.
    """

    if not isinstance(request, ArchitectureCycleInput):
        raise TypeError(
            "request must be an ArchitectureCycleInput, "
            f"got {type(request).__name__}"
        )

    architecture = request.architecture
    descriptor = ARCHITECTURES[architecture]
    gas_model = request.gas_model

    # --- unit-air compressor chain ---------------------------------------
    # Start with 1 kg/s air.  All per-kilogram works are independent of the
    # later flow size; the final states are rebuilt with the solved flow.
    _, inlet_k = gas_model.air_cp_k(
        request.inlet_total_temperature
    )
    inlet_unit = _air_state(
        gas_model,
        request.inlet_total_temperature,
        request.inlet_total_pressure,
        1.0,
    )
    compressor_config = _compressor_config(request, inlet_k)
    compressor_unit = compressor_train(
        inlet_unit,
        request.compressor_pressure_ratios,
        compressor_config,
    )

    (
        fuel_air_ratio,
        alpha,
        p_gas,
        cp_gas,
        k_gas,
        heat_per_air,
        alpha_basis,
        closure_name,
    ) = _resolve_combustor(request, compressor_unit)

    T_gas = (
        request.combustor.outlet_total_temperature
        if isinstance(request.combustor, LegacyCombustorClosure)
        else request.combustor.state.T_total
    )
    gas_unit = _gas_state(
        T_gas,
        p_gas,
        1.0 + fuel_air_ratio,
        cp_gas,
        k_gas,
        identifier="combustor_exit",
    )

    model_variant = "uncooled_no_bleed_0d_ideal_mixing"
    eta_t = request.turbine_polytropic_efficiencies
    eta_m = request.turbine_mechanical_efficiencies
    p_t_out = request.turbine_outlet_total_pressure
    target = request.target_gearbox_input_power

    if architecture in (
        Architecture.SINGLE_SHAFT_SINGLE_CC,
        Architecture.SINGLE_SHAFT_DUAL_CC,
    ):
        expansion_ratio = p_gas / p_t_out
        if not expansion_ratio > 1.0:
            raise ArchitectureCyclePhysicalError(
                "main turbine outlet total pressure must be below the "
                "combustor-exit total pressure, got "
                f"p_in={p_gas!r} Pa and p_out={p_t_out!r} Pa"
            )
        try:
            turbine_unit = turbine_expansion(
                gas_unit, expansion_ratio, eta_t[0]
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc, "main turbine expansion is not physically available"
            ) from exc

        net_per_air = (
            eta_m[0] * (1.0 + fuel_air_ratio) * turbine_unit.specific_work
            - compressor_unit.specific_work
        )
        if not math.isfinite(net_per_air) or net_per_air <= 0.0:
            raise ArchitectureCyclePhysicalError(
                "physical inability to supply a positive target: net specific "
                "shaft power per kg air is non-positive "
                f"({net_per_air!r} J/kg) for target {target!r} W"
            )
        air_flow = target / net_per_air

        # Rebuild all states on the solved flow basis.  The combustor outlet
        # state is the canonical turbine inlet so every mass flow in the chain
        # is the same floating-point value.
        inlet_final = _air_state(
            gas_model,
            request.inlet_total_temperature,
            request.inlet_total_pressure,
            air_flow,
        )
        compressor_final = compressor_train(
            inlet_final,
            request.compressor_pressure_ratios,
            compressor_config,
        )
        combustor_final = _combustor_result(
            compressor_final.outlet,
            T_gas,
            p_gas,
            cp_gas,
            k_gas,
            fuel_air_ratio,
            alpha,
            alpha_basis,
            closure_name,
            heat_per_air,
        )
        try:
            turbine_final = turbine_expansion(
                combustor_final.outlet, expansion_ratio, eta_t[0]
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc, "main turbine expansion is not physically available"
            ) from exc

        compressor_powers = tuple(
            work * air_flow
            for work in compressor_final.per_unit_specific_work
        )
        shaft = evaluate_shaft_power_balance(
            architecture,
            compressor_gas_powers=compressor_powers,
            turbine_gas_powers=(turbine_final.power,),
            mechanical_efficiencies=(eta_m[0],),
            required_gearbox_input_power=target,
            gearbox_efficiency=request.gearbox_efficiency,
        )
        turbines = (turbine_final,)
        net_specific_per_air = (
            eta_m[0]
            * (combustor_final.gas_mass_flow / air_flow)
            * turbine_final.specific_work
            - compressor_final.specific_work
        )
    else:
        # Free power turbine: prescribe the GG turbine work, invert the
        # polytropic relation to obtain its real outlet total state, then feed
        # that state to the FPT.  The compressor load is not subtracted again
        # from the FPT output.
        required_gg_work = compressor_unit.specific_work / (
            eta_m[0] * (1.0 + fuel_air_ratio)
        )
        try:
            gg_unit = turbine_pressure_ratio_for_work(
                gas_unit, required_gg_work, eta_t[0]
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc,
                "GG turbine cannot supply the prescribed compressor work",
            ) from exc
        if not gg_unit.outlet.p_total > p_t_out:
            raise ArchitectureCyclePhysicalError(
                "free power turbine outlet total pressure must be below the "
                "GG turbine outlet total pressure, got "
                f"p_GG_out={gg_unit.outlet.p_total!r} Pa and "
                f"p_FPT_out={p_t_out!r} Pa"
            )
        fpt_ratio = gg_unit.outlet.p_total / p_t_out
        try:
            fpt_unit = turbine_expansion(
                gg_unit.outlet, fpt_ratio, eta_t[1]
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc, "free power turbine expansion is not physically available"
            ) from exc
        fpt_power_per_air = (
            eta_m[1] * (1.0 + fuel_air_ratio) * fpt_unit.specific_work
        )
        if not math.isfinite(fpt_power_per_air) or fpt_power_per_air <= 0.0:
            raise ArchitectureCyclePhysicalError(
                "physical inability to supply a positive target: FPT specific "
                "power per kg air is non-positive "
                f"({fpt_power_per_air!r} J/kg) for target {target!r} W"
            )
        air_flow = target / fpt_power_per_air

        inlet_final = _air_state(
            gas_model,
            request.inlet_total_temperature,
            request.inlet_total_pressure,
            air_flow,
        )
        compressor_final = compressor_train(
            inlet_final,
            request.compressor_pressure_ratios,
            compressor_config,
        )
        combustor_final = _combustor_result(
            compressor_final.outlet,
            T_gas,
            p_gas,
            cp_gas,
            k_gas,
            fuel_air_ratio,
            alpha,
            alpha_basis,
            closure_name,
            heat_per_air,
        )
        required_gg_work_final = compressor_final.specific_work / (
            eta_m[0] * (1.0 + fuel_air_ratio)
        )
        try:
            gg_final = turbine_pressure_ratio_for_work(
                combustor_final.outlet,
                required_gg_work_final,
                eta_t[0],
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc,
                "GG turbine cannot supply the prescribed compressor work",
            ) from exc
        if not gg_final.outlet.p_total > p_t_out:
            raise ArchitectureCyclePhysicalError(
                "free power turbine outlet total pressure must be below the "
                "GG turbine outlet total pressure, got "
                f"p_GG_out={gg_final.outlet.p_total!r} Pa and "
                f"p_FPT_out={p_t_out!r} Pa"
            )
        fpt_ratio_final = gg_final.outlet.p_total / p_t_out
        try:
            fpt_final = turbine_expansion(
                gg_final.outlet, fpt_ratio_final, eta_t[1]
            )
        except ValueError as exc:
            raise _physical_error_from_turbine(
                exc,
                "free power turbine expansion is not physically available",
            ) from exc

        shaft = evaluate_shaft_power_balance(
            architecture,
            compressor_gas_powers=(compressor_final.power,),
            turbine_gas_powers=(gg_final.power, fpt_final.power),
            mechanical_efficiencies=(eta_m[0], eta_m[1]),
            required_gearbox_input_power=target,
            gearbox_efficiency=request.gearbox_efficiency,
        )
        turbines = (gg_final, fpt_final)
        net_specific_per_air = (
            eta_m[1]
            * (combustor_final.gas_mass_flow / air_flow)
            * fpt_final.specific_work
        )

    result = ArchitectureCycleResult(
        architecture=architecture,
        model_variant=model_variant,
        combustor_closure=closure_name,
        gas_model=gas_model,
        validation_scope=ARCHITECTURE_CYCLE_LIMITATIONS,
        inlet=inlet_final,
        compressor=compressor_final,
        combustor=combustor_final,
        turbines=turbines,
        shaft=shaft,
        air_mass_flow=air_flow,
        fuel_mass_flow=combustor_final.fuel_mass_flow,
        gas_mass_flow=combustor_final.gas_mass_flow,
        target_gearbox_input_power=target,
        actual_gearbox_input_power=shaft.actual_gearbox_input_power,
        gearbox_output_power=shaft.gearbox_output_power,
        net_specific_power_per_air=net_specific_per_air,
        pressure_ratio_product=compressor_final.pressure_ratio_product,
        compressor_pressure_residual=compressor_final.residual,
        turbine_pressure_residuals=tuple(
            item.residual for item in turbines
        ),
    )
    return result


__all__ = [
    "ARCHITECTURE_CYCLE_LIMITATIONS",
    "AlphaBasis",
    "ArchitectureCycleError",
    "ArchitectureCycleInput",
    "ArchitectureCyclePhysicalError",
    "ArchitectureCycleResult",
    "CombustorClosureResult",
    "ExternalCombustorClosure",
    "ExternalCombustorState",
    "LegacyCombustorClosure",
    "PolynomialGasModel",
    "calculate_architecture_cycle",
]
