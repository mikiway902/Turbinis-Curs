"""Explicit station and engine-architecture contracts.

This module introduces the first M2 foundation only: an immutable, validated
:class:`StationState` and explicit topology descriptors for the three target
engine architectures.  It does not implement a cycle, a component model, a
sweep, a shaft-speed relation, a gearbox ratio, or any new numerical
assumption.  The existing legacy cycle remains the behavioural source of truth.

ARCHITECTURE NAMES
==================
``CC`` in :class:`Architecture` means **centrifugal compressor**, not
combustion chamber.  Every target architecture has exactly one combustor in
this contract.

* ``SINGLE_SHAFT_SINGLE_CC`` -- one centrifugal compressor unit, one combustor,
  one turbine assembly on one output shaft.  The turbine shaft drives the
  propeller through a reduction gearbox.
* ``SINGLE_SHAFT_DUAL_CC`` -- two centrifugal compressor units in series, one
  combustor, one turbine assembly on one output shaft.  Both compressor units
  and the turbine share that shaft; the shaft drives the propeller through a
  reduction gearbox.  The contract records the unit count and common shaft
  group; it does not define a pressure-ratio split between the units.
* ``FREE_POWER_TURBINE`` -- one centrifugal compressor unit and a
  gas-generator turbine on one shaft, plus a free power turbine on a separate
  output shaft.  The free-power-turbine shaft drives the propeller through a
  reduction gearbox; the gas-generator shaft does not drive the output.

Compressor and turbine counts are conceptual units/assemblies, not aerodynamic
stage counts.  Unknown shaft speeds, gearbox ratios, gearbox efficiencies,
power splits, propeller loads, and stage inventories are deliberately not
encoded here.  The gearbox/propeller relation is a topology statement only:
the contract does not add a gear ratio, a loss model beyond the existing cycle
``eta_reducer``, or a propeller load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class EngineArchitecture(str, Enum):
    """Target engine topologies; ``CC`` means centrifugal compressor."""

    SINGLE_SHAFT_SINGLE_CC = "SINGLE_SHAFT_SINGLE_CC"
    SINGLE_SHAFT_DUAL_CC = "SINGLE_SHAFT_DUAL_CC"
    FREE_POWER_TURBINE = "FREE_POWER_TURBINE"


# Backwards-compatible short name for callers that prefer ``Architecture``.
Architecture = EngineArchitecture


def _as_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_positive_float(name: str, value: Any) -> float:
    result = _as_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result!r}")
    return result


def _as_count(name: str, value: Any, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return value


def _as_identifier(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"identifier must be None or a non-empty string, got {value!r}"
        )
    return value


@dataclass(frozen=True)
class StationState:
    """Total thermodynamic state at one gas-path station.

    Only quantities explicitly available in the current model are stored:
    total temperature ``T_total`` [K], total pressure ``p_total`` [Pa], mass
    flow ``mass_flow`` [kg/s], specific heat ``cp`` [J/(kg K)], and isentropic
    exponent ``k`` [-].  Fuel/air ratio, composition, and cooling-flow state
    are intentionally not implied.

    ``identifier`` is an optional label (for example ``"compressor_exit"``).
    It must be ``None`` or a non-empty string and is never interpreted.
    """

    T_total: float
    p_total: float
    mass_flow: float
    cp: float
    k: float
    identifier: str | None = None

    def __post_init__(self) -> None:
        for name in ("T_total", "p_total", "mass_flow", "cp"):
            object.__setattr__(
                self, name, _as_positive_float(name, getattr(self, name))
            )

        k = _as_float("k", self.k)
        if k <= 1.0:
            raise ValueError(f"k must be > 1, got {self.k!r}")
        object.__setattr__(self, "k", k)
        object.__setattr__(
            self, "identifier", _as_identifier(self.identifier)
        )


@dataclass(frozen=True)
class ShaftGroup:
    """One mechanically connected shaft group.

    Components listed in the same group rotate together.  The contract stores
    only membership and whether the group drives the useful output.  For the
    output group, "useful output" means a reduction gearbox and then the
    propeller; the gearbox and propeller are downstream, unmodelled elements.
    The contract does not define an absolute speed, a speed ratio, a
    reduction-gear ratio, a gearbox efficiency, a power split, a propeller
    load, or a rotation direction.
    """

    name: str
    compressor_units: int
    turbine_assemblies: int
    drives_output: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"shaft group name must be a non-empty string, got {self.name!r}"
            )
        object.__setattr__(
            self,
            "compressor_units",
            _as_count("compressor_units", self.compressor_units),
        )
        object.__setattr__(
            self,
            "turbine_assemblies",
            _as_count("turbine_assemblies", self.turbine_assemblies),
        )
        if not isinstance(self.drives_output, bool):
            raise ValueError(
                "drives_output must be a bool, "
                f"got {self.drives_output!r}"
            )
        if self.compressor_units == 0 and self.turbine_assemblies == 0:
            raise ValueError(
                "a shaft group must contain at least one compressor unit "
                "or turbine assembly"
            )


#: Canonical topology for each :class:`Architecture`.
#:
#: Each value is ``(compressor_units, combustors, turbine_assemblies,
#: ((group name, (compressor_units, turbine_assemblies, drives_output)), ...))``.
#: The descriptor invariant compares against this table so that an enum member
#: cannot be paired with a contradictory shaft layout, even when the aggregate
#: counts and the single output group happen to be consistent.
_ARCHITECTURE_LAYOUTS: Mapping[
    Architecture,
    tuple[int, int, int, tuple[tuple[str, tuple[int, int, bool]], ...]],
] = MappingProxyType({
    Architecture.SINGLE_SHAFT_SINGLE_CC: (
        1,
        1,
        1,
        (("MAIN", (1, 1, True)),),
    ),
    Architecture.SINGLE_SHAFT_DUAL_CC: (
        2,
        1,
        1,
        (("MAIN", (2, 1, True)),),
    ),
    Architecture.FREE_POWER_TURBINE: (
        1,
        1,
        2,
        (
            ("GAS_GENERATOR", (1, 1, False)),
            ("FREE_POWER_TURBINE", (0, 1, True)),
        ),
    ),
})


@dataclass(frozen=True)
class ArchitectureDescriptor:
    """Explicit topology for one :class:`Architecture`.

    ``compressor_units``, ``combustors`` and ``turbine_assemblies`` are
    conceptual counts and must match the canonical layout for the selected
    architecture.  ``shaft_groups`` records mechanical coupling only: every
    group has exactly one output flag, and all components assigned to a group
    share that shaft.  The output group is the one that drives the reduction
    gearbox and propeller; neither is modelled here.  No component order,
    pressure split, bleed topology, shaft speed, gear ratio, or numerical cycle
    behaviour is defined here.
    """

    architecture: Architecture
    compressor_units: int
    combustors: int
    turbine_assemblies: int
    shaft_groups: tuple[ShaftGroup, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.architecture, Architecture):
            raise ValueError(
                "architecture must be an Architecture member, "
                f"got {self.architecture!r}"
            )
        object.__setattr__(
            self,
            "compressor_units",
            _as_count("compressor_units", self.compressor_units, minimum=1),
        )
        object.__setattr__(
            self,
            "combustors",
            _as_count("combustors", self.combustors, minimum=1),
        )
        object.__setattr__(
            self,
            "turbine_assemblies",
            _as_count(
                "turbine_assemblies", self.turbine_assemblies, minimum=1
            ),
        )

        raw_groups = self.shaft_groups
        if isinstance(raw_groups, (str, bytes)):
            raise ValueError("shaft_groups must be an iterable of ShaftGroup")
        try:
            groups = tuple(raw_groups)
        except TypeError as exc:
            raise ValueError(
                "shaft_groups must be an iterable of ShaftGroup"
            ) from exc
        if not groups:
            raise ValueError("shaft_groups must contain at least one group")
        for group in groups:
            if not isinstance(group, ShaftGroup):
                raise ValueError(
                    "shaft_groups must contain only ShaftGroup instances, "
                    f"got {group!r}"
                )

        names = [group.name for group in groups]
        if len(set(names)) != len(names):
            raise ValueError(f"shaft group names must be unique, got {names!r}")

        compressor_total = sum(group.compressor_units for group in groups)
        if compressor_total != self.compressor_units:
            raise ValueError(
                "shaft group compressor units "
                f"({compressor_total}) must equal compressor_units "
                f"({self.compressor_units})"
            )
        turbine_total = sum(group.turbine_assemblies for group in groups)
        if turbine_total != self.turbine_assemblies:
            raise ValueError(
                "shaft group turbine assemblies "
                f"({turbine_total}) must equal turbine_assemblies "
                f"({self.turbine_assemblies})"
            )

        output_groups = [group for group in groups if group.drives_output]
        if len(output_groups) != 1:
            raise ValueError(
                "exactly one shaft group must drive the useful output, "
                f"got {len(output_groups)}"
            )

        (
            expected_compressor_units,
            expected_combustors,
            expected_turbine_assemblies,
            expected_layout,
        ) = _ARCHITECTURE_LAYOUTS[self.architecture]
        actual_layout = {
            group.name: (
                group.compressor_units,
                group.turbine_assemblies,
                group.drives_output,
            )
            for group in groups
        }
        if (
            self.compressor_units != expected_compressor_units
            or self.combustors != expected_combustors
            or self.turbine_assemblies != expected_turbine_assemblies
            or actual_layout != dict(expected_layout)
        ):
            raise ValueError(
                f"{self.architecture.value} topology must be "
                f"(compressor_units, combustors, turbine_assemblies) = "
                f"({expected_compressor_units}, {expected_combustors}, "
                f"{expected_turbine_assemblies}) with shaft groups "
                f"{dict(expected_layout)!r}, got "
                f"({self.compressor_units}, {self.combustors}, "
                f"{self.turbine_assemblies}) with shaft groups "
                f"{actual_layout!r}"
            )
        object.__setattr__(self, "shaft_groups", groups)


ARCHITECTURES: Mapping[Architecture, ArchitectureDescriptor] = MappingProxyType(
    {
        Architecture.SINGLE_SHAFT_SINGLE_CC: ArchitectureDescriptor(
            architecture=Architecture.SINGLE_SHAFT_SINGLE_CC,
            compressor_units=1,
            combustors=1,
            turbine_assemblies=1,
            shaft_groups=(
                ShaftGroup(
                    name="MAIN",
                    compressor_units=1,
                    turbine_assemblies=1,
                    drives_output=True,
                ),
            ),
        ),
        Architecture.SINGLE_SHAFT_DUAL_CC: ArchitectureDescriptor(
            architecture=Architecture.SINGLE_SHAFT_DUAL_CC,
            compressor_units=2,
            combustors=1,
            turbine_assemblies=1,
            shaft_groups=(
                ShaftGroup(
                    name="MAIN",
                    compressor_units=2,
                    turbine_assemblies=1,
                    drives_output=True,
                ),
            ),
        ),
        Architecture.FREE_POWER_TURBINE: ArchitectureDescriptor(
            architecture=Architecture.FREE_POWER_TURBINE,
            compressor_units=1,
            combustors=1,
            turbine_assemblies=2,
            shaft_groups=(
                ShaftGroup(
                    name="GAS_GENERATOR",
                    compressor_units=1,
                    turbine_assemblies=1,
                    drives_output=False,
                ),
                ShaftGroup(
                    name="FREE_POWER_TURBINE",
                    compressor_units=0,
                    turbine_assemblies=1,
                    drives_output=True,
                ),
            ),
        ),
    }
)


__all__ = [
    "ARCHITECTURES",
    "Architecture",
    "ArchitectureDescriptor",
    "EngineArchitecture",
    "ShaftGroup",
    "StationState",
]
