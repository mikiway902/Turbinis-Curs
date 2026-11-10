"""Turbine aggregate performance from the notebook's cell 20.

PROVENANCE
==========
This module is a formula-preserving extraction of the numerical aggregation at
the beginning of code cell 20 of
``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb`` (items 136 and 146-153).  It
executes no notebook, imports no NumPy/SciPy/pandas, and keeps no mutable global
state.

The notebook remains the independent legacy comparison path.  The equations,
their evaluation order, and the gas-property callable are preserved verbatim:

``g_i`` -> ``L_тΣ`` -> ``η_т_турб`` -> ``h_вz`` -> ``η_л_турб`` ->
``p_т_полн`` -> ``cp_т, k_т`` -> ``H_т_полн`` -> ``η_т_полн`` ->
``N_т`` -> ``N_вал``.

``N_вал`` is the pre-reducer shaft power at the gearbox input.  The old source
factor ``η_р`` is no longer folded into it; it is applied only to the explicit
downstream ``N_shaft_reducer_out`` diagnostic.  The notebook cell was updated
to the same pre-reducer closure.

The source's pandas display tables are intentionally not part of this module;
they are presentation only and no caller in the optimizer consumes them.
Functions and scalars that are pure pass-throughs (``газ``, ``H_т``,
``G_c1``, ``T_г_полн``, ``p_г_полн``, ``η_м``, ``G_в``, ``L_к``, ``η_р``)
are explicit arguments so the function has no hidden namespace dependency.

Tests compare this module with the observed legacy cell-20 execution at one
fixed candidate only as a software-drift diagnostic, never as physical
validation.  In particular, the absence of guards or clamps is intentional:
like the source, calculable but engineering-poor inputs are evaluated with the
same arithmetic.

UNITS
=====
SI throughout.  ``L_t_sum``, ``H_t_total`` and ``N_turbine``/``N_shaft`` are
``J/kg`` and ``W`` respectively; temperatures are K; pressures are Pa.  The
efficiency fields are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

GasProperty = Callable[[float], tuple[float, float]]


@dataclass(frozen=True)
class TurbinePerformance:
    """Numerical result of the cell-20 turbine aggregation.

    The field names are ASCII equivalents of the notebook identifiers:

    ===================  ==================
    result field         notebook identifier
    ===================  ==================
    ``g_i``              ``g_i``
    ``L_t_sum``          ``L_тΣ``
    ``eta_t_turb``       ``η_т_турб``
    ``h_v_exit``         ``h_вz``
    ``eta_l_turb``       ``η_л_турб``
    ``p_t_total``        ``p_т_полн``
    ``cp_t``             ``cp_т``
    ``k_t``              ``k_т``
    ``H_t_total``        ``H_т_полн``
    ``eta_t_total``      ``η_т_полн``
    ``N_turbine``        ``N_т``
    ``N_shaft``          ``N_вал`` (pre-reducer, gearbox-input plane)
    ===================  ==================

    ``N_shaft_reducer_out = N_shaft * eta_r`` is an explicit downstream
    gearbox-output field.  It is deliberately not added to
    :meth:`legacy_values`, because the source cell's ``N_вал`` identifier is the
    pre-reducer target used by the active optimizer.
    """

    g_i: tuple[float, ...]
    L_t_sum: float
    eta_t_turb: float
    h_v_exit: float
    eta_l_turb: float
    p_t_total: float
    cp_t: float
    k_t: float
    H_t_total: float
    eta_t_total: float
    N_turbine: float
    N_shaft: float
    N_shaft_reducer_out: float

    def legacy_values(self) -> dict[str, Any]:
        """Return the values under the exact notebook identifiers.

        ``g_i`` is a list, matching the source list comprehension; every other
        value is passed through unchanged.
        """

        return {
            "g_i": list(self.g_i),
            "L_тΣ": self.L_t_sum,
            "η_т_турб": self.eta_t_turb,
            "h_вz": self.h_v_exit,
            "η_л_турб": self.eta_l_turb,
            "p_т_полн": self.p_t_total,
            "cp_т": self.cp_t,
            "k_т": self.k_t,
            "H_т_полн": self.H_t_total,
            "η_т_полн": self.eta_t_total,
            "N_т": self.N_turbine,
            "N_вал": self.N_shaft,
        }


def calculate_turbine_performance(
    stages: Sequence[Mapping[str, Any]],
    *,
    G_c1: float,
    H_t: float,
    T_g_total: float,
    p_g_total: float,
    eta_m: float,
    G_v: float,
    L_k: float,
    eta_r: float,
    gas: GasProperty,
) -> TurbinePerformance:
    """Evaluate the cell-20 turbine aggregation without notebook execution.

    Parameters
    ----------
    stages:
        Stage mappings in flow order.  The final stage provides the exit
        velocity ``c2``, exit total pressure ``p_ст_п`` and exit total
        temperature ``T_ст_п``.  No stage count or field-name guard is added.
    G_c1:
        First-stage nozzle mass flow, kg/s.
    H_t:
        Available turbine heat drop used by the source's power efficiency.
    T_g_total, p_g_total:
        Turbine-inlet total temperature (K) and total pressure (Pa).
    eta_m, eta_r:
        Mechanical efficiency and reduction-gear efficiency.  ``eta_r`` is used
        only for the explicit downstream ``N_shaft_reducer_out`` field; the
        target ``N_shaft`` is at the gearbox input (pre-reducer).
    G_v, L_k:
        Compressor air flow (kg/s) and compressor work (J/kg).
    gas:
        Callable ``gas(T) -> (cp, k)`` corresponding to the notebook ``газ``.

    Returns
    -------
    TurbinePerformance
        All twelve numerical intermediates in source evaluation order.
    """

    g_i = tuple(s["G_c"] / G_c1 for s in stages)
    L_t_sum = sum(g * s["L_т"] for g, s in zip(g_i, stages))
    eta_t_turb = L_t_sum / H_t
    h_v_exit = stages[-1]["c2"] ** 2 / 2
    eta_l_turb = (L_t_sum + h_v_exit) / H_t
    p_t_total = stages[-1]["p_ст_п"]
    cp_t, k_t = gas(0.5 * (T_g_total + stages[-1]["T_ст_п"]))
    H_t_total = cp_t * T_g_total * (
        1 - (p_t_total / p_g_total) ** ((k_t - 1) / k_t)
    )
    eta_t_total = L_t_sum / H_t_total
    N_turbine = L_t_sum * G_c1 * eta_m
    N_shaft = N_turbine - G_v * L_k
    N_shaft_reducer_out = N_shaft * eta_r

    return TurbinePerformance(
        g_i=g_i,
        L_t_sum=L_t_sum,
        eta_t_turb=eta_t_turb,
        h_v_exit=h_v_exit,
        eta_l_turb=eta_l_turb,
        p_t_total=p_t_total,
        cp_t=cp_t,
        k_t=k_t,
        H_t_total=H_t_total,
        eta_t_total=eta_t_total,
        N_turbine=N_turbine,
        N_shaft=N_shaft,
        N_shaft_reducer_out=N_shaft_reducer_out,
    )


__all__ = [
    "GasProperty",
    "TurbinePerformance",
    "calculate_turbine_performance",
]
