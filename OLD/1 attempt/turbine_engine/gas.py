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
Successful fixed points return the same pre-update ``cp``/``k`` and ``H0`` as
the source.  Unlike the source, exhaustion of either 50-iteration loop raises
:class:`turbine_engine.diagnostics.NumericalConvergenceError` instead of
silently returning the last iterate.  Tests compare this module with observed
legacy outputs only as software-drift diagnostics, never as physical validation.

The composition function used by ``газ`` is
:func:`turbine_engine.cycle.calc_cp_kerosene`, which is bit-identical to the
legacy notebook ``calc_cp``.

OPT-IN TEXTBOOK CORRELATION
===========================
:func:`calc_textbook_gas_cp_k` separately transcribes the published Appendix 6
true-``cp`` fit.  It is opt-in through
``gas_property_basis="textbook_true_cp"``; the legacy notebook path remains the
default and is unchanged.

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
import numbers
from typing import Any

from turbine_engine.cycle import calc_cp_kerosene
from turbine_engine.diagnostics import NumericalConvergenceError

# Source-fixed numerical controls (do not tune without changing the notebook).
GAS_FINITE_DIFFERENCE_STEP_K = 1.0
GAS_CELSIUS_OFFSET_K = 273.15
STAGE_CP_K_MAX_ITER = 50
STAGE_CP_K_TOL = 1e-9

# Opt-in property bases.  The default keeps the legacy notebook path.
GAS_PROPERTY_BASIS_LEGACY = "legacy_derivative"
GAS_PROPERTY_BASIS_TEXTBOOK = "textbook_true_cp"
GAS_PROPERTY_BASES = (GAS_PROPERTY_BASIS_LEGACY, GAS_PROPERTY_BASIS_TEXTBOOK)

# Published true-cp fit: kcal/(kg*K) converted to J/(kg*K), branch at 750 K.
TEXTBOOK_CP_KCAL_TO_J = 4187.0
TEXTBOOK_CP_BRANCH_T_K = 750.0


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


def _as_strict_real(name: str, value: Any, *, allow_infinite: bool = False) -> float:
    """Reject booleans, strings and disallowed non-finite values for the new fit."""

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    result = float(value)
    if math.isnan(result) or (math.isinf(result) and not allow_infinite):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _as_strict_positive_real(name: str, value: Any) -> float:
    result = _as_strict_real(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0, got {result!r}")
    return result


def _resolve_gas_property_basis(value: Any) -> str:
    """Return a validated property-basis selector; reject unknown config."""

    if not isinstance(value, str):
        raise ValueError(
            f"gas_property_basis must be a string, got {value!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    if value not in GAS_PROPERTY_BASES:
        raise ValueError(
            f"unknown gas_property_basis {value!r}; "
            f"expected one of {GAS_PROPERTY_BASES!r}"
        )
    return value


def calc_textbook_gas_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
    """Return the published direct true-``cp`` correlation at ``T``.

    Literal transcription of the Appendix 6 true-``cp`` fit used by the cycle
    and turbine texts: ``cp`` in kcal/(kg*K) is evaluated directly at ``T`` and
    multiplied by 4187 J/kcal.  The published branches meet at 750 K with the
    source's finite jump; no smoothing, mean-temperature derivative or
    process-average substitution is applied.  The fit is published for lean
    products (``alpha >= 1``); ``alpha`` infinity uses the analytic reciprocal
    limit, not a numeric surrogate.  This is a source correlation, not a
    validated extension to rich mixtures or a physical acceptance claim.

    Parameters
    ----------
    T:
        Gas temperature in K; must be finite and positive.
    alpha:
        Excess-air ratio (``math.inf`` for dry air); must be ``>= 1`` or
        infinity, the published lean domain.
    R:
        Specific gas constant in J/(kg*K); must be finite and positive.

    Returns
    -------
    tuple[float, float]
        ``(cp, k)`` with ``cp`` in J/(kg*K) and ``k = cp / (cp - R)``.

    Raises
    ------
    ValueError
        For booleans, strings or non-finite ``T``/``R``, or a non-real
        ``alpha``.
    GasPropertyError
        When ``alpha`` is outside the published lean domain, or the evaluated
        ``cp`` is non-finite or not greater than ``R``.  No value is clamped.
    """

    T_value = _as_strict_positive_real("T", T)
    alpha_value = _as_strict_real("alpha", alpha, allow_infinite=True)
    R_value = _as_strict_positive_real("R", R)
    if alpha_value != math.inf and alpha_value < 1.0:
        raise GasPropertyError(
            "textbook true-cp fit is published for lean products "
            f"(alpha >= 1 or inf), got alpha={alpha_value!r}"
        )

    if T_value < TEXTBOOK_CP_BRANCH_T_K:
        cp_kcal = (
            (0.0174 / alpha_value + 0.2407)
            + (0.0093 / alpha_value + 0.0193) * (0.0025 * T_value - 0.875)
            + (0.002 - 0.001056 / (alpha_value - 0.2))
            * (0.000025 * T_value * T_value - 0.0275 * T_value + 6.5625)
        )
    else:
        cp_kcal = (
            (0.0267 / alpha_value + 0.26)
            + (0.0133 / alpha_value + 0.032) * (0.001176 * T_value - 0.88235)
            - (0.00374 + 0.0094 / (alpha_value * alpha_value + 10.0))
            * (0.0000055556 * T_value * T_value - 0.013056 * T_value + 6.67)
        )

    cp = cp_kcal * TEXTBOOK_CP_KCAL_TO_J
    denominator = cp - R_value
    if not math.isfinite(cp) or not math.isfinite(denominator) or denominator <= 0.0:
        raise GasPropertyError(
            f"invalid cp={cp!r} for R={R_value!r}: cp - R must be finite and positive"
        )
    k = cp / denominator
    if not math.isfinite(k):
        raise GasPropertyError(f"invalid k={k!r} for cp={cp!r}, R={R_value!r}")
    return cp, k


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
    *,
    gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY,
) -> tuple[float, float]:
    """Return stage-average ``cp`` and ``k`` from the notebook iteration.

    Equivalent to ``cp_k_ступени(T0, H0)`` with the namespace values ``α_г``,
    ``R`` and ``φ``.  The mean conditional temperature is
    ``T0 - H0 * phi**2 / (2 * cp)``; the loop stops on ``abs(cp_new - cp) < 1e-9``
    and returns the pre-update ``cp``/``k`` exactly as the source does.
    Exhausting the 50-iteration limit raises
    :class:`turbine_engine.diagnostics.NumericalConvergenceError` instead of
    returning the last iterate without marking non-convergence.

    ``gas_property_basis="legacy_derivative"`` (default) preserves the notebook
    mean-``cp`` finite-difference calls exactly.  Opting into
    ``"textbook_true_cp"`` evaluates the published direct true-``cp``
    correlation at the same conditional temperature on every iteration; the
    book mean-temperature approximation, tolerance and pre-update return
    semantics are unchanged.  The textbook basis is a source correlation and
    is not an additional physical validation.

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
    gas_property_basis:
        ``"legacy_derivative"`` (default) or ``"textbook_true_cp"``.
    """

    basis = _resolve_gas_property_basis(gas_property_basis)
    T0_value = _as_positive_float("T0", T0)
    H0_value = _as_float("H0", H0)
    if H0_value < 0.0:
        raise ValueError(f"H0 must be >= 0, got {H0_value!r}")
    alpha_value = _as_alpha(alpha)
    R_value = _as_positive_float("R", R)
    phi_value = _as_float("phi", phi)
    if not 0.0 < phi_value <= 1.0:
        raise ValueError(f"phi must be in (0, 1], got {phi_value!r}")

    gas_cp_k = (
        calc_gas_cp_k
        if basis == GAS_PROPERTY_BASIS_LEGACY
        else calc_textbook_gas_cp_k
    )
    cp, k = gas_cp_k(T0_value, alpha_value, R_value)
    for iteration in range(STAGE_CP_K_MAX_ITER):
        if cp == 0.0:
            raise GasPropertyError("stage cp is zero; mean temperature is undefined")
        cp_evaluated, k_evaluated = cp, k
        T_mean = T0_value - H0_value * phi_value**2 / (2 * cp_evaluated)
        cp_new, k_new = gas_cp_k(T_mean, alpha_value, R_value)
        if abs(cp_new - cp_evaluated) < STAGE_CP_K_TOL:
            break
        cp, k = cp_new, k_new
    else:
        raise NumericalConvergenceError(
            "stage cp/k fixed-point iteration exhausted its limit",
            stage="gas.calc_stage_cp_k",
            reason="cp_k_iteration_exhausted",
            last_valid_state={
                "gas_property_basis": basis,
                "iterations": iteration + 1,
                "iteration_limit": STAGE_CP_K_MAX_ITER,
                "tolerance": STAGE_CP_K_TOL,
                "cp": cp_evaluated,
                "k": k_evaluated,
                "cp_new": cp_new,
                "k_new": k_new,
                "residual": cp_new - cp_evaluated,
                "T_mean": T_mean,
                "H0": H0_value,
            },
        )
    return cp, k


def calc_h0_for_pressure(
    T0: float,
    p0: float,
    p2: float,
    alpha: float,
    R: float,
    phi: float,
    *,
    gas_property_basis: str = GAS_PROPERTY_BASIS_LEGACY,
) -> float:
    """Return the available isentropic heat drop at a prescribed outlet pressure.

    Equivalent to ``H0_по_p(T0, p0, p2)`` with the namespace values ``α_г``,
    ``R`` and ``φ``.  For each iterate the drop is
    ``H0 = cp * T0 * (1 - (p2 / p0) ** ((k - 1) / k))``; the loop stops on
    ``abs(cp_new - cp) < 1e-9`` and returns the ``H0`` computed in that
    iteration exactly as the source does.  Exhausting the 50-iteration limit
    raises :class:`turbine_engine.diagnostics.NumericalConvergenceError`
    instead of returning the last iterate without marking non-convergence.

    ``gas_property_basis="legacy_derivative"`` (default) preserves the notebook
    mean-``cp`` finite-difference calls exactly.  Opting into
    ``"textbook_true_cp"`` evaluates the published direct true-``cp``
    correlation at the same conditional temperature on every iteration; the
    ``H0`` pressure process, tolerance and stop semantics are unchanged.  The
    textbook basis is a source correlation and is not an additional physical
    validation.

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
    gas_property_basis:
        ``"legacy_derivative"`` (default) or ``"textbook_true_cp"``.
    """

    basis = _resolve_gas_property_basis(gas_property_basis)
    T0_value = _as_positive_float("T0", T0)
    p0_value = _as_positive_float("p0", p0)
    p2_value = _as_positive_float("p2", p2)
    alpha_value = _as_alpha(alpha)
    R_value = _as_positive_float("R", R)
    phi_value = _as_float("phi", phi)
    if not 0.0 < phi_value <= 1.0:
        raise ValueError(f"phi must be in (0, 1], got {phi_value!r}")

    gas_cp_k = (
        calc_gas_cp_k
        if basis == GAS_PROPERTY_BASIS_LEGACY
        else calc_textbook_gas_cp_k
    )
    cp, k = gas_cp_k(T0_value, alpha_value, R_value)
    H0 = 0.0
    for iteration in range(STAGE_CP_K_MAX_ITER):
        H0 = cp * T0_value * (1 - (p2_value / p0_value) ** ((k - 1) / k))
        cp_evaluated, k_evaluated = cp, k
        T_mean = T0_value - H0 * phi_value**2 / (2 * cp_evaluated)
        cp_new, k_new = gas_cp_k(T_mean, alpha_value, R_value)
        if abs(cp_new - cp_evaluated) < STAGE_CP_K_TOL:
            break
        cp, k = cp_new, k_new
    else:
        raise NumericalConvergenceError(
            "stage H0 pressure iteration exhausted its limit",
            stage="gas.calc_h0_for_pressure",
            reason="h0_pressure_iteration_exhausted",
            last_valid_state={
                "gas_property_basis": basis,
                "iterations": iteration + 1,
                "iteration_limit": STAGE_CP_K_MAX_ITER,
                "tolerance": STAGE_CP_K_TOL,
                "cp": cp_evaluated,
                "k": k_evaluated,
                "cp_new": cp_new,
                "k_new": k_new,
                "residual": cp_new - cp_evaluated,
                "T_mean": T_mean,
                "H0": H0,
            },
        )
    return H0


__all__ = [
    "GAS_CELSIUS_OFFSET_K",
    "GAS_FINITE_DIFFERENCE_STEP_K",
    "GAS_PROPERTY_BASES",
    "GAS_PROPERTY_BASIS_LEGACY",
    "GAS_PROPERTY_BASIS_TEXTBOOK",
    "GasPropertyError",
    "STAGE_CP_K_MAX_ITER",
    "STAGE_CP_K_TOL",
    "TEXTBOOK_CP_BRANCH_T_K",
    "TEXTBOOK_CP_KCAL_TO_J",
    "calc_gas_cp_k",
    "calc_h0_for_pressure",
    "calc_stage_cp_k",
    "calc_textbook_gas_cp_k",
]
