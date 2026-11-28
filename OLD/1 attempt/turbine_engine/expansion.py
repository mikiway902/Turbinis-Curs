"""Turbine expansion iteration and preliminary sizing from notebook cell 9.

PROVENANCE
==========
This module is a source-preserving extraction of the numerical block at the
beginning of code cell 9 of
``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`` (items 1-12 and 26-31/46).  It
executes no notebook, imports no NumPy/SciPy/pandas, and keeps no mutable
global state.

The notebook remains the independent legacy comparison path.  The equations,
their evaluation order, the ``π_т`` exponent, the ``0.25`` initial-temperature
exponent, the exact 50-iteration limit, and the strict
``abs(T_т_н - T_т) < 1e-6`` convergence test are copied verbatim.  The
successful fixed point assigns ``T_т`` the last ``T_т_н`` exactly as the
source does.  Unlike the source, exhaustion of the 50-iteration limit raises
:class:`turbine_engine.diagnostics.NumericalConvergenceError` with the last
finite iterate instead of silently returning a non-converged temperature.

The source's pandas display table ``табл_п1_31`` is intentionally not part of
this module; it is presentation only and no caller in the optimizer consumes
it.  All namespace-dependent inputs are explicit arguments, including the gas
callable, so the function has no hidden namespace dependency.  No guards or
clamps are added: native division and square-root failures keep their source
classification.

Tests compare this module with observed legacy cell-9 outputs only as
software-drift diagnostics, never as physical validation.

UNITS
=====
SI throughout.  Temperatures are K, pressures Pa, ``H_t`` and ``c0_t`` are
``J/kg`` and ``m/s``; ``R`` is J/(kg*K), mass flow kg/s, areas m², rotational
speed rpm, and lengths m.  ``D_eta``, ``l_eta`` and ``D_t0`` are diameters and
blade lengths in m.  ``sigma_l`` is the rotor-blade stress in MPa and ``k_n``
carries the source's unit conversion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from turbine_engine.diagnostics import NumericalConvergenceError

GasProperty = Callable[[float], tuple[float, float]]

# Source-fixed numerical control (do not tune without changing the notebook).
EXPANSION_ITER_MAX = 50
EXPANSION_TEMPERATURE_TOL = 1e-6


@dataclass(frozen=True)
class TurbineExpansion:
    """Numerical result of the cell-9 expansion and preliminary sizing block.

    The field names are ASCII equivalents of the notebook identifiers:

    ===================  ==================
    result field         notebook identifier
    ===================  ==================
    ``pi_t``             ``π_т``
    ``T_t``              ``T_т``
    ``H_t``              ``H_т``
    ``cp0``              ``cp0``
    ``k0``               ``k0``
    ``v_t``              ``v_т``
    ``A_t0``             ``A_т0``
    ``n_max0``           ``n_max0``
    ``n_rab0``           ``n_rab0``
    ``c0_t``             ``c0_т``
    ``u_eta``            ``u_η``
    ``D_eta``            ``D_η``
    ``l_eta``            ``l_η``
    ``D_t0``             ``D_т0``
    ===================  ==================
    """

    pi_t: float
    T_t: float
    H_t: float
    cp0: float
    k0: float
    v_t: float
    A_t0: float
    n_max0: float
    n_rab0: float
    c0_t: float
    u_eta: float
    D_eta: float
    l_eta: float
    D_t0: float

    def legacy_values(self) -> dict[str, Any]:
        """Return the values under the exact notebook identifiers."""

        return {
            "π_т": self.pi_t,
            "T_т": self.T_t,
            "H_т": self.H_t,
            "cp0": self.cp0,
            "k0": self.k0,
            "v_т": self.v_t,
            "A_т0": self.A_t0,
            "n_max0": self.n_max0,
            "n_раб0": self.n_rab0,
            "c0_т": self.c0_t,
            "u_η": self.u_eta,
            "D_η": self.D_eta,
            "l_η": self.l_eta,
            "D_т0": self.D_t0,
        }


def calculate_turbine_expansion(
    p_g_total: float,
    p_t: float,
    T_g_total: float,
    eta_l_expected: float,
    *,
    gas: GasProperty,
    R: float,
    G_l_ost: float,
    c_ta: float,
    sigma_l: float,
    k_n: float,
    q_n: float,
    y0: float,
    l_t_D_t: float,
) -> TurbineExpansion:
    """Evaluate the cell-9 expansion iteration and preliminary sizing.

    Parameters
    ----------
    p_g_total, p_t:
        Turbine-inlet total pressure and turbine-outlet static pressure, Pa.
    T_g_total:
        Turbine-inlet total temperature, K.
    eta_l_expected:
        Expected blade efficiency ``η_л_ожид`` used only by the temperature
        iteration.
    gas:
        Callable ``gas(T) -> (cp, k)`` corresponding to the notebook ``газ``.
    R:
        Specific gas constant, J/(kg*K).
    G_l_ost:
        Mass flow through the downstream stages (``G_л_ост``), kg/s.
    c_ta:
        Axial exit velocity ``c_та``, m/s.
    sigma_l:
        Allowable rotor-blade stress ``σ_л``, MPa.
    k_n:
        Source blade-form coefficient (``K_n``), including unit conversion.
    q_n:
        Rotational-speed margin ``n_раб / n_max``.
    y0:
        Optimal velocity ratio ``u_η / c0_т``.
    l_t_D_t:
        Source aspect-ratio estimate ``l_т/D_т``.

    Returns
    -------
    TurbineExpansion
        Every numerical intermediate written by the source block.
    """

    pi_t = p_g_total / p_t  # п. 1
    T_t = T_g_total * pi_t**-0.25  # first iterate, п. 15
    converged = False
    previous_T_t = T_t
    residual = math.inf
    for iteration in range(EXPANSION_ITER_MAX):
        cp0, k0 = gas(0.5 * (T_g_total + T_t))
        H_t = cp0 * T_g_total * (1 - pi_t ** ((1 - k0) / k0))
        T_t_n = T_g_total - H_t * eta_l_expected / cp0
        previous_T_t, residual = T_t, T_t_n - T_t
        if abs(residual) < EXPANSION_TEMPERATURE_TOL:
            converged = True
            break
        T_t = T_t_n
    if not converged:
        raise NumericalConvergenceError(
            "turbine expansion temperature iteration exhausted its limit",
            stage="expansion.calculate_turbine_expansion",
            reason="temperature_iteration_exhausted",
            last_valid_state={
                "iterations": iteration + 1,
                "T_t": T_t,
                "T_t_previous": previous_T_t,
                "T_t_next": T_t_n,
                "residual": residual,
                "tolerance": EXPANSION_TEMPERATURE_TOL,
                "iteration_limit": EXPANSION_ITER_MAX,
                "T_g_total": T_g_total,
                "pi_t": pi_t,
            },
        )
    T_t = T_t_n
    v_t = R * T_t / p_t
    A_t0 = G_l_ost * v_t / c_ta
    n_max0 = math.sqrt(sigma_l / (k_n * A_t0))
    n_rab0 = q_n * n_max0
    # одноступенчатая турбина (п. 26-31)
    c0_t = math.sqrt(2 * H_t)
    u_eta = y0 * c0_t
    D_eta = 60 * u_eta / (math.pi * n_rab0)
    l_eta = A_t0 / (math.pi * D_eta)
    D_t0 = math.sqrt(A_t0 / (math.pi * l_t_D_t))

    return TurbineExpansion(
        pi_t=pi_t,
        T_t=T_t,
        H_t=H_t,
        cp0=cp0,
        k0=k0,
        v_t=v_t,
        A_t0=A_t0,
        n_max0=n_max0,
        n_rab0=n_rab0,
        c0_t=c0_t,
        u_eta=u_eta,
        D_eta=D_eta,
        l_eta=l_eta,
        D_t0=D_t0,
    )


__all__ = [
    "EXPANSION_ITER_MAX",
    "EXPANSION_TEMPERATURE_TOL",
    "GasProperty",
    "TurbineExpansion",
    "calculate_turbine_expansion",
]
