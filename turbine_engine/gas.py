"""Gas-property helper for one turbine stage.

PROVENANCE
==========
This module is a formula-preserving extraction of ``газ(T)``,
``cp_k_ступени(T0, H0)`` and ``H0_по_p(T0, p0, p2)`` from code cell 7 of
``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb``.  It executes no notebook,
imports no NumPy/SciPy, and keeps no mutable global state.

The notebook remains the independent legacy comparison path.  The equations,
the 1 K finite-difference step, the 273.15 K offset, the 50-iteration limit, and
the strict ``abs(cp_new - cp) < 1e-9`` convergence test are copied verbatim.
Like the source, the loop returns its last iterate after 50 iterations without
raising a convergence error.  Tests compare this module with observed legacy
outputs only as software-drift diagnostics, never as physical validation.

The composition function used by ``газ`` is
:func:`turbine_engine.cycle.calc_cp_kerosene`, which is bit-identical to the
legacy notebook ``calc_cp``.

UNITS
=====
SI throughout.  ``T`` and ``T0`` are absolute temperatures in K; ``p0`` and
``p2`` are absolute pressures in Pa; ``H0`` is an isentropic heat drop in J/kg;
``R`` is the gas constant in J/(kg*K); ``cp`` is returned in J/(kg*K); ``k`` is
dimensionless.  ``alpha`` is the excess-air ratio (``math.inf`` for dry air).
``phi`` is the dimensionless stator velocity coefficient.
"""

from __future__ import annotations

import math
from typing import Any

from turbine_engine.cycle import calc_cp_kerosene

# Source-fixed numerical controls (do not tune without changing the notebook).
GAS_FINITE_DIFFERENCE_STEP_K = 1.0
GAS_CELSIUS_OFFSET_K = 273.15
STAGE_CP_K_MAX_ITER = 50
STAGE_CP_K_TOL = 1e-9


class GasPropertyError(RuntimeError):
    """The gas-property expression is undefined for the supplied values."""


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


def _as_alpha(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"alpha must be a real number, got {value!r}") from exc
    if math.isnan(result) or result <= 0.0:
        raise ValueError(f"alpha must be > 0, got {value!r}")
    return result


def calc_gas_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
    """Return true ``cp`` and ``k`` of combustion products at ``T``.

    Equivalent to the notebook ``газ(T)`` with the namespace values ``α_г``,
    ``R``.  The centred 1 K difference uses the supplied ``alpha`` for all three
    ``cp`` evaluations, exactly as the source does.

    Parameters
    ----------
    T:
        Gas temperature in K; must be finite and positive.
    alpha:
        Excess-air ratio (``math.inf`` for dry air); must be positive.
    R:
        Specific gas constant in J/(kg*K); must be finite and positive.

    Returns
    -------
    tuple[float, float]
        ``(cp, k)`` with ``cp`` in J/(kg*K) and ``k = cp / (cp - R)``.
    """

    T_value = _as_positive_float("T", T)
    alpha_value = _as_alpha(alpha)
    R_value = _as_positive_float("R", R)

    dT = GAS_FINITE_DIFFERENCE_STEP_K
    c_pm = float(calc_cp_kerosene(T_value, alpha_value))
    dc_pm = (
        float(calc_cp_kerosene(T_value + dT, alpha_value))
        - float(calc_cp_kerosene(T_value - dT, alpha_value))
    ) / (2 * dT)
    c_p = c_pm + (T_value - GAS_CELSIUS_OFFSET_K) * dc_pm

    denominator = c_p - R_value
    if not math.isfinite(c_p) or not math.isfinite(denominator) or denominator <= 0.0:
        raise GasPropertyError(
            f"invalid cp={c_p!r} for R={R_value!r}: cp - R must be finite and positive"
        )
    return c_p, c_p / denominator


def calc_stage_cp_k(
    T0: float,
    H0: float,
    alpha: float,
    R: float,
    phi: float,
) -> tuple[float, float]:
    """Return stage-average ``cp`` and ``k`` from the notebook iteration.

    Equivalent to ``cp_k_ступени(T0, H0)`` with the namespace values ``α_г``,
    ``R`` and ``φ``.  The mean conditional temperature is
    ``T0 - H0 * phi**2 / (2 * cp)``; the loop stops on ``abs(cp_new - cp) < 1e-9``
    and otherwise returns the 50th iterate without raising, as in the source.

    Parameters
    ----------
    T0:
        Stage-inlet total temperature in K; must be finite and positive.
    H0:
        Available isentropic heat drop in J/kg; must be finite and non-negative.
    alpha:
        Excess-air ratio (``math.inf`` for dry air); must be positive.
    R:
        Specific gas constant in J/(kg*K); must be finite and positive.
    phi:
        Stator velocity coefficient; must be in ``(0, 1]``.
    """

    T0_value = _as_positive_float("T0", T0)
    H0_value = _as_float("H0", H0)
    if H0_value < 0.0:
        raise ValueError(f"H0 must be >= 0, got {H0_value!r}")
    alpha_value = _as_alpha(alpha)
    R_value = _as_positive_float("R", R)
    phi_value = _as_float("phi", phi)
    if not 0.0 < phi_value <= 1.0:
        raise ValueError(f"phi must be in (0, 1], got {phi_value!r}")

    cp, k = calc_gas_cp_k(T0_value, alpha_value, R_value)
    for _ in range(STAGE_CP_K_MAX_ITER):
        if cp == 0.0:
            raise GasPropertyError("stage cp is zero; mean temperature is undefined")
        cp_new, k_new = calc_gas_cp_k(
            T0_value - H0_value * phi_value**2 / (2 * cp),
            alpha_value,
            R_value,
        )
        if abs(cp_new - cp) < STAGE_CP_K_TOL:
            break
        cp, k = cp_new, k_new
    return cp, k


def calc_h0_for_pressure(
    T0: float,
    p0: float,
    p2: float,
    alpha: float,
    R: float,
    phi: float,
) -> float:
    """Return the available isentropic heat drop at a prescribed outlet pressure.

    Equivalent to ``H0_по_p(T0, p0, p2)`` with the namespace values ``α_г``,
    ``R`` and ``φ``.  For each iterate the drop is
    ``H0 = cp * T0 * (1 - (p2 / p0) ** ((k - 1) / k))``; the loop stops on
    ``abs(cp_new - cp) < 1e-9`` and otherwise returns the 50th iterate without
    raising, exactly as in the source.

    Parameters
    ----------
    T0:
        Stage-inlet total temperature in K; must be finite and positive.
    p0:
        Stage-inlet total pressure in Pa; must be finite and positive.
    p2:
        Prescribed stage-outlet static pressure in Pa; must be finite and
        positive.  ``p2 > p0`` is not rejected here: the source formula then
        yields a negative drop, and no clamp or model change is applied.
    alpha:
        Excess-air ratio (``math.inf`` for dry air); must be positive.
    R:
        Specific gas constant in J/(kg*K); must be finite and positive.
    phi:
        Stator velocity coefficient; must be in ``(0, 1]``.
    """

    T0_value = _as_positive_float("T0", T0)
    p0_value = _as_positive_float("p0", p0)
    p2_value = _as_positive_float("p2", p2)
    alpha_value = _as_alpha(alpha)
    R_value = _as_positive_float("R", R)
    phi_value = _as_float("phi", phi)
    if not 0.0 < phi_value <= 1.0:
        raise ValueError(f"phi must be in (0, 1], got {phi_value!r}")

    cp, k = calc_gas_cp_k(T0_value, alpha_value, R_value)
    H0 = 0.0
    for _ in range(STAGE_CP_K_MAX_ITER):
        H0 = cp * T0_value * (1 - (p2_value / p0_value) ** ((k - 1) / k))
        cp_new, k_new = calc_gas_cp_k(
            T0_value - H0 * phi_value**2 / (2 * cp),
            alpha_value,
            R_value,
        )
        if abs(cp_new - cp) < STAGE_CP_K_TOL:
            break
        cp, k = cp_new, k_new
    return H0


__all__ = [
    "GAS_CELSIUS_OFFSET_K",
    "GAS_FINITE_DIFFERENCE_STEP_K",
    "GasPropertyError",
    "STAGE_CP_K_MAX_ITER",
    "STAGE_CP_K_TOL",
    "calc_gas_cp_k",
    "calc_h0_for_pressure",
    "calc_stage_cp_k",
]
