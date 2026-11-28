"""Standalone scalar combustor heat-addition wrapper.

SCOPE
=====
This module is a thin, equation-preserving wrapper over the two existing
scalar combustor-related equations in :mod:`turbine_engine.cycle`:

.. code-block:: text

    alpha = calc_alpha(T_in, T_gas)
    g_m   = calc_g_m(T_in, T_gas, config)

It returns an immutable :class:`CombustorResult` holding the scalar inlet and
target temperatures (K), the relative fuel flow ``g_m`` and the excess-air
ratio ``alpha``.  It does not convert a mass flow, update a fuel or product
composition, produce ``cp``/``k`` state, apply a pressure loss, or introduce a
separate combustion-efficiency model.  ``g_m`` inherits the current cycle
assumptions inside :func:`turbine_engine.cycle.calc_g_m` (``q_lhv_kj``,
``eta_combustion``, ``l0``, ``T_ref`` and the existing mean-``cp``
polynomials); ``alpha`` inherits the current
:func:`turbine_engine.cycle.calc_alpha` polynomial.  Neither helper is
reimplemented here.

This is **not** a combustor geometry model, a combustor performance map, an
operating-range model, a thermodynamic state model, or a mass-balance model.
It does not accept or produce :class:`~turbine_engine.architecture.StationState`
because the current station contract intentionally does not encode fuel or
composition.  It is not wired into
:func:`turbine_engine.cycle.calculate_cycle_point`; that cycle continues to
call ``calc_alpha`` and ``calc_g_m`` directly, and its outputs are unchanged.

PROVENANCE
==========
Both equations are the existing source-preserving scalar equations of
:mod:`turbine_engine.cycle`, themselves extracted from
``Turbinis-Cycle/Turbinis-Cycle.ipynb`` cell 3 (``calc_alpha``) and cell 5
(``calc_g_m``).  This wrapper adds validation and a record only; it changes no
formula, assumption, tolerance, or numerical behaviour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from turbine_engine.cycle import (
    DEFAULT_CONFIG,
    CycleConfig,
    calc_alpha,
    calc_g_m,
)


def _as_finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_positive_temperature(name: str, value: Any) -> float:
    result = _as_finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return result


@dataclass(frozen=True)
class CombustorResult:
    """Immutable scalar heat-addition result from the existing cycle helpers.

    ``T_in`` is the inlet total temperature and ``T_gas`` the target turbine
    inlet total temperature, both in K.  ``g_m`` is the relative fuel flow
    returned by :func:`turbine_engine.cycle.calc_g_m`; ``alpha`` is the
    excess-air ratio returned by :func:`turbine_engine.cycle.calc_alpha`.
    ``alpha`` is the helper's excess-air ratio, not the cycle's derived
    ``1 / (g_m * l0)`` bookkeeping field.

    Only finite positive temperatures and finite outputs are required.  The
    record does not store the cycle configuration, does not claim a solved
    combustor state, and does not check a physical heat-addition direction.
    Because the configuration is not stored, a manually constructed record is
    not re-evaluated against the helpers; :func:`combustor_heat_addition` is the
    equation-parity construction path.
    """

    T_in: float
    T_gas: float
    g_m: float
    alpha: float

    def __post_init__(self) -> None:
        T_in = _as_positive_temperature("T_in", self.T_in)
        T_gas = _as_positive_temperature("T_gas", self.T_gas)
        g_m = _as_finite_float("g_m", self.g_m)
        alpha = _as_finite_float("alpha", self.alpha)

        object.__setattr__(self, "T_in", T_in)
        object.__setattr__(self, "T_gas", T_gas)
        object.__setattr__(self, "g_m", g_m)
        object.__setattr__(self, "alpha", alpha)


def combustor_heat_addition(
    T_in: float,
    T_gas: float,
    config: CycleConfig = DEFAULT_CONFIG,
) -> CombustorResult:
    """Return the existing scalar ``alpha`` and ``g_m`` for one heat addition.

    ``T_in`` is the compressor-exit total temperature and ``T_gas`` the target
    turbine-inlet total temperature, both in K.  ``config`` is passed unchanged
    to :func:`turbine_engine.cycle.calc_g_m`; it supplies the same current cycle
    constants that ``calculate_cycle_point`` uses.  The function calls exactly

    ``calc_alpha(T_in, T_gas)`` and ``calc_g_m(T_in, T_gas, config)``

    and records their returned scalar values.  It performs no mass-flow
    conversion, composition update, ``cp``/``k`` evaluation, pressure-loss
    calculation, or separate combustion-efficiency model.
    """

    if not isinstance(config, CycleConfig):
        raise TypeError(
            f"config must be a CycleConfig, got {type(config).__name__}"
        )

    T_in = _as_positive_temperature("T_in", T_in)
    T_gas = _as_positive_temperature("T_gas", T_gas)

    alpha = calc_alpha(T_in, T_gas)
    g_m = calc_g_m(T_in, T_gas, config)

    return CombustorResult(
        T_in=T_in,
        T_gas=T_gas,
        g_m=g_m,
        alpha=alpha,
    )


__all__ = [
    "CombustorResult",
    "combustor_heat_addition",
]
