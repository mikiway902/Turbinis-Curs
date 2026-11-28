"""Standalone explicit shaft power balances for the selected architectures.

SCOPE
=====
This module evaluates a **shaft power balance only**.  It does not implement a
cycle, a combustor model, a turbine expansion, a pressure-ratio split, a shaft
speed, a gearbox ratio, a propeller load, a stage inventory, a map, or an
optimizer hook.  All component gas powers and mechanical efficiencies are
explicit inputs in watts and dimensionless respectively.  Mass flow is resolved
externally: the caller passes already mass-flow-weighted gas powers
(``power = specific_work * mass_flow``).  The evaluator never equates specific
works across different mass-flow bases and never infers a stage or a map.

ARCHITECTURES AND ORDERING
==========================
``CC`` in the architecture names means **centrifugal compressor**.  Every
selected architecture has exactly one combustor.  Ordering is fixed and
explicit; there is no name map, stage inference, or pressure-split inference.

* ``SINGLE_SHAFT_DUAL_CC`` -- two compressor units in series and one turbine
  assembly on the MAIN output shaft.  ``compressor_gas_powers`` must contain
  exactly two values in series order; only their sum enters the MAIN balance.
  ``turbine_gas_powers`` and ``mechanical_efficiencies`` must contain exactly
  one value: ``(turbine assembly,)`` and ``(eta_mech,)``.
* ``SINGLE_SHAFT_SINGLE_CC`` -- one compressor unit and one turbine assembly on
  the MAIN output shaft.  Both tuples have exactly one value.
* ``FREE_POWER_TURBINE`` -- one gas-generator compressor on the gas-generator
  shaft and two turbine assemblies.  ``compressor_gas_powers`` has exactly one
  value: ``(gas-generator compressor,)``.  ``turbine_gas_powers`` is ordered
  gas-generator turbine first, free power turbine second:
  ``(GG turbine, FPT)``.  ``mechanical_efficiencies`` has the same order:
  ``(eta_mech_GG, eta_mech_FPT)``.  The free power turbine drives the output;
  the gas-generator shaft does not drive the gearbox.

SIGNS AND CLOSURE
=================
Intended engineering inputs are nonnegative compressor loads and nonnegative
turbine driving powers.  The standalone compressor and turbine equations also
retain signed mathematical branches (for example a pressure-ratio continuation
below one), so provided component gas powers are validated as finite and kept
with their signs rather than clamped.  The required gearbox-input power must be
finite and nonnegative; zero is a valid no-load target, while a negative
"required output" has no defined meaning.

Mechanical efficiency is applied exactly once to each turbine **gas** power::

    P_mech = eta_mech * P_turbine_gas

The independent raw balances are::

    MAIN (single-shaft dual-CC / single-shaft single-CC):
        actual = sum(P_mech) - sum(P_compressor)
        residual = actual - required_gearbox_input_power

    GAS_GENERATOR (free power turbine):
        actual = P_mech_GG - P_compressor_GG
        residual = actual

    FREE_POWER_TURBINE (free power turbine):
        actual = P_mech_FPT
        residual = actual - required_gearbox_input_power

The actual gearbox input is the MAIN net power for the two single-shaft
architectures and the free-power-turbine mechanical power for the free-power
architecture.  The downstream gearbox output is always::

    gearbox_output = actual_gearbox_input * eta_gearbox

It is never ``required_gearbox_input_power * eta_gearbox``.  Residuals are
unfiltered and signed: insufficient turbine power yields a negative residual or
actual input and is retained, not replaced by an exception or a feasibility
profile.  Mechanical losses are represented by the supplied efficiencies; the
evaluator adds no accessory, cooling, bleed, or shaft-loss default.

This helper is not wired into ``calculate_cycle_point``, the baseline, or the
optimizer.  Architecture-specific cycles remain unimplemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from turbine_engine.architecture import Architecture, EngineArchitecture


def _as_finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a real number, got {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_nonnegative_float(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {result!r}")
    return result


def _as_efficiency(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be in (0, 1], got {result!r}")
    return result


def _normalise_values(name: str, values: Iterable[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{name} must be an iterable of finite numbers, got {values!r}"
        )
    try:
        sequence = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of finite numbers, got {values!r}"
        ) from exc
    return tuple(
        _as_finite_float(f"{name}[{index}]", value)
        for index, value in enumerate(sequence)
    )


def _normalise_efficiencies(
    name: str, values: Iterable[float]
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(
            f"{name} must be an iterable of efficiencies in (0, 1], "
            f"got {values!r}"
        )
    try:
        sequence = tuple(values)
    except TypeError as exc:
        raise ValueError(
            f"{name} must be an iterable of efficiencies in (0, 1], "
            f"got {values!r}"
        ) from exc
    return tuple(
        _as_efficiency(f"{name}[{index}]", value)
        for index, value in enumerate(sequence)
    )


#: Required ``(compressor count, turbine count)`` for each architecture.
#:
#: The ordering of the values inside each tuple is documented in the module
#: docstring.  This table deliberately records counts only; it contains no
#: pressure split, stage split, speed, gear ratio, or component map.
_EXPECTED_COUNTS: Mapping[Architecture, tuple[int, int]] = MappingProxyType({
    Architecture.SINGLE_SHAFT_DUAL_CC: (2, 1),
    Architecture.SINGLE_SHAFT_SINGLE_CC: (1, 1),
    Architecture.FREE_POWER_TURBINE: (1, 2),
})


@dataclass(frozen=True)
class ShaftPowerBalance:
    """Immutable architecture-specific shaft power balance.

    Only explicit inputs are stored: the architecture, compressor gas powers,
    turbine gas powers, per-turbine mechanical efficiencies, the required
    gearbox-input power, and the gearbox efficiency.  Every net power,
    residual, and downstream output is a **derived property** and is therefore
    recomputed consistently after :func:`dataclasses.replace`; no redundant
    derived field can be supplied or forged independently.

    Powers are in W.  Efficiencies are dimensionless.  The tuple ordering is
    defined by the architecture (see the module docstring), with the
    free-power-turbine architecture ordered gas generator then free power
    turbine.  A property that is not applicable to the selected architecture
    returns ``None``; applicable values are signed raw arithmetic results and
    are never filtered or clamped.
    """

    architecture: EngineArchitecture
    compressor_gas_powers: tuple[float, ...]
    turbine_gas_powers: tuple[float, ...]
    mechanical_efficiencies: tuple[float, ...]
    required_gearbox_input_power: float
    gearbox_efficiency: float

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, Architecture):
            raise ValueError(
                "architecture must be an Architecture member, "
                f"got {self.architecture!r}"
            )

        compressor_powers = _normalise_values(
            "compressor_gas_powers", self.compressor_gas_powers
        )
        turbine_powers = _normalise_values(
            "turbine_gas_powers", self.turbine_gas_powers
        )
        efficiencies = _normalise_efficiencies(
            "mechanical_efficiencies", self.mechanical_efficiencies
        )
        required_input = _as_nonnegative_float(
            "required_gearbox_input_power",
            self.required_gearbox_input_power,
        )
        gearbox_efficiency = _as_efficiency(
            "gearbox_efficiency", self.gearbox_efficiency
        )

        expected_compressors, expected_turbines = _EXPECTED_COUNTS[
            self.architecture
        ]
        if len(compressor_powers) != expected_compressors:
            raise ValueError(
                f"{self.architecture.value} requires exactly "
                f"{expected_compressors} compressor gas power value(s), got "
                f"{len(compressor_powers)}"
            )
        if len(turbine_powers) != expected_turbines:
            raise ValueError(
                f"{self.architecture.value} requires exactly "
                f"{expected_turbines} turbine gas power value(s), got "
                f"{len(turbine_powers)}"
            )
        if len(efficiencies) != expected_turbines:
            raise ValueError(
                f"{self.architecture.value} requires exactly "
                f"{expected_turbines} mechanical efficiency value(s), got "
                f"{len(efficiencies)}"
            )

        object.__setattr__(self, "compressor_gas_powers", compressor_powers)
        object.__setattr__(self, "turbine_gas_powers", turbine_powers)
        object.__setattr__(self, "mechanical_efficiencies", efficiencies)
        object.__setattr__(
            self, "required_gearbox_input_power", required_input
        )
        object.__setattr__(self, "gearbox_efficiency", gearbox_efficiency)

        derived = [
            *self.turbine_mechanical_powers,
            self.compressor_gas_power_total,
            self.turbine_mechanical_power_total,
            self.actual_gearbox_input_power,
            self.gearbox_output_power,
        ]
        for value in (
            self.main_net_power,
            self.gas_generator_net_power,
            self.free_power_turbine_mechanical_power,
            self.main_residual,
            self.gas_generator_residual,
            self.free_power_turbine_residual,
        ):
            if value is not None:
                derived.append(value)
        for value in derived:
            if not math.isfinite(value):
                raise ValueError(
                    "shaft power balance produced a non-finite derived value "
                    f"for {self.architecture.value}: {value!r}"
                )

    @property
    def turbine_mechanical_powers(self) -> tuple[float, ...]:
        """Per-turbine mechanical powers ``eta_mech * P_turbine_gas`` in W."""

        return tuple(
            power * efficiency
            for power, efficiency in zip(
                self.turbine_gas_powers, self.mechanical_efficiencies
            )
        )

    @property
    def compressor_gas_power_total(self) -> float:
        """Sum of the supplied compressor loads in W."""

        return sum(self.compressor_gas_powers)

    @property
    def turbine_mechanical_power_total(self) -> float:
        """Sum of the per-turbine mechanical powers in W."""

        return sum(self.turbine_mechanical_powers)

    @property
    def main_net_power(self) -> float | None:
        """Single-shaft MAIN net power in W; ``None`` otherwise."""

        if self.architecture is Architecture.FREE_POWER_TURBINE:
            return None
        return (
            self.turbine_mechanical_power_total
            - self.compressor_gas_power_total
        )

    @property
    def gas_generator_net_power(self) -> float | None:
        """Gas-generator available mechanical power minus compressor load.

        Applies only to ``FREE_POWER_TURBINE``; returns ``None`` for the two
        single-shaft architectures.
        """

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return (
            self.turbine_mechanical_powers[0]
            - self.compressor_gas_powers[0]
        )

    @property
    def free_power_turbine_mechanical_power(self) -> float | None:
        """Free-power-turbine mechanical power in W, or ``None`` if absent."""

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return self.turbine_mechanical_powers[1]

    @property
    def actual_gearbox_input_power(self) -> float:
        """Actual pre-reducer power at the gearbox input in W.

        This is the MAIN net power for the single-shaft architectures and the
        free-power-turbine mechanical power for ``FREE_POWER_TURBINE``.  It is
        signed raw arithmetic and may be negative.
        """

        if self.architecture is Architecture.FREE_POWER_TURBINE:
            return self.turbine_mechanical_powers[1]
        return (
            self.turbine_mechanical_power_total
            - self.compressor_gas_power_total
        )

    @property
    def gearbox_output_power(self) -> float:
        """Downstream output ``actual_input * eta_gearbox`` in W."""

        return self.actual_gearbox_input_power * self.gearbox_efficiency

    @property
    def main_residual(self) -> float | None:
        """Single-shaft raw residual in W, or ``None`` otherwise."""

        if self.architecture is Architecture.FREE_POWER_TURBINE:
            return None
        return (
            self.actual_gearbox_input_power
            - self.required_gearbox_input_power
        )

    @property
    def gas_generator_residual(self) -> float | None:
        """Gas-generator raw residual in W, or ``None`` for single-shaft."""

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return (
            self.turbine_mechanical_powers[0]
            - self.compressor_gas_powers[0]
        )

    @property
    def free_power_turbine_residual(self) -> float | None:
        """Free-power-turbine raw residual in W, or ``None`` if absent."""

        if self.architecture is not Architecture.FREE_POWER_TURBINE:
            return None
        return (
            self.actual_gearbox_input_power
            - self.required_gearbox_input_power
        )


def evaluate_shaft_power_balance(
    architecture: EngineArchitecture,
    *,
    compressor_gas_powers: Iterable[float],
    turbine_gas_powers: Iterable[float],
    mechanical_efficiencies: Iterable[float],
    required_gearbox_input_power: float,
    gearbox_efficiency: float,
) -> ShaftPowerBalance:
    """Evaluate one architecture-specific shaft power balance.

    Parameters
    ----------
    architecture:
        One of ``SINGLE_SHAFT_DUAL_CC``, ``SINGLE_SHAFT_SINGLE_CC`` or
        ``FREE_POWER_TURBINE``.  The required tuple counts and order are fixed
        per architecture as documented in the module docstring.
    compressor_gas_powers:
        Compressor gas/aerodynamic powers in W.  For the dual-CC single-shaft
        architecture, two values in series order; for the other architectures,
        one gas-generator compressor value.
    turbine_gas_powers:
        Turbine gas/aerodynamic powers in W before mechanical efficiency.  For
        ``FREE_POWER_TURBINE`` the order is ``(GG turbine, FPT)``.
    mechanical_efficiencies:
        One efficiency per turbine in the same order as ``turbine_gas_powers``.
        Each must be finite and in ``(0, 1]``.  There is no default: the caller
        must supply the architecture's efficiencies explicitly.
    required_gearbox_input_power:
        Required pre-reducer gearbox-input power in W.  Must be finite and
        nonnegative; zero is a valid no-load target.
    gearbox_efficiency:
        Downstream reduction-gearbox efficiency in ``(0, 1]``.  It affects the
        gearbox output property only, not the raw shaft residuals or the actual
        gearbox-input power.

    Returns
    -------
    ShaftPowerBalance
        Immutable record whose net powers, residuals, and downstream output are
        derived properties.  Values are not clamped and engineering-infeasible
        balances are retained with their signs.
    """

    return ShaftPowerBalance(
        architecture=architecture,
        compressor_gas_powers=compressor_gas_powers,
        turbine_gas_powers=turbine_gas_powers,
        mechanical_efficiencies=mechanical_efficiencies,
        required_gearbox_input_power=required_gearbox_input_power,
        gearbox_efficiency=gearbox_efficiency,
    )


__all__ = [
    "ShaftPowerBalance",
    "evaluate_shaft_power_balance",
]
