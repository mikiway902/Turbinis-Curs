"""Root reaction relation from the notebook radial-profiling cell.

PROVENANCE
==========
This module is a formula-preserving extraction of ``ρ_корня(r, закон)`` from
code cell 14 of ``Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb``.  It executes no
notebook and keeps no mutable global state.

STATUS
======
This is a transcription of the *current* root reaction relation only.  It is
not a validated or accepted radial-equilibrium model, not a physical validation,
and not an engineering acceptance criterion.  The notebook remains the
independent legacy comparison path; tests compare the two only as
software-drift diagnostics.

SOURCE LAWS
===========
The source has exactly two branches:

* ``закон == "Γ"`` uses the (71) pressure relation with the stator-exit
  tangential Mach number.
* every other law value uses the general fallback based on
  ``c1 = c1 * (r_c / r1) ** cos²(α1)``.

Only the exact string ``"Γ"`` selects the first branch, as in the source.

ARITHMETIC
==========
The extraction keeps Python's arithmetic semantics.  A negative intermediate
pressure factor raised to an integer-valued exponent still returns a real
value (so does a negative root radius in the ``Γ`` branch), while a negative
base raised to a non-integer power returns ``complex`` exactly as the notebook
does.  No intermediate sign or finiteness check is applied.  The remaining
checks validate independent inputs (numeric, finite, and the module's existing
positive-domain constraints) or cover arithmetic the source cannot perform
(zero root radius, zero ``R``/``phi``).

UNITS
=====
SI throughout.  Temperatures are K, pressures Pa, heat drops J/kg, velocities
m/s, radii m, ``cp`` J/(kg*K), and ``R`` J/(kg*K).  ``phi`` is the dimensionless
stator velocity coefficient.  The returned root reaction degree is
dimensionless.
"""

from __future__ import annotations

import math
from typing import Any

from turbine_engine.stage import StageResult


class ReactionCalculationError(RuntimeError):
    """The root reaction expression is undefined for the supplied values.

    The source relies on Python's arithmetic exceptions for divisions by zero
    and math-domain errors.  This explicit error keeps the same
    propagate-don't-clamp behaviour for those cases; it is not an engineering
    acceptance filter.  Negative intermediate bases and the ``complex`` powers
    they produce are not treated as errors.
    """


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


def calculate_root_reaction(
    stage: StageResult,
    law: str,
    R: float,
    phi: float,
) -> float | complex:
    """Return the root reaction degree from the current notebook relation.

    This reproduces ``ρ_корня(r, закон)`` operation-for-operation and in the
    source order.  ``R`` and ``phi`` are the active namespace values of the
    notebook; they are passed explicitly so the function stays pure.

    Parameters
    ----------
    stage:
        Typed stage result.  The source fields ``k``, ``cp``, ``T0``, ``p0``,
        ``H0``, ``p2``, ``D_c``, ``l_c``, ``c1``, ``c1u``, ``T1_ad`` and
        ``alpha1_deg`` are read.
    law:
        Profiling law.  Only the exact string ``"Γ"`` selects the constant-
        circulation branch; any other value uses the source's general fallback.
    R:
        Specific gas constant in J/(kg*K); must be finite and positive.
    phi:
        Stator velocity coefficient; must be finite and positive.

    Returns
    -------
    float | complex
        Root reaction degree exactly as the source expression computes it.
        This is a ``float`` in the physical operating range.  A negative
        intermediate base raised to a non-integer power yields a ``complex``
        value, which is the notebook's own Python semantics and is preserved
        rather than rejected.
    """

    k = _as_float("stage.k", stage.k)
    cp = _as_positive_float("stage.cp", stage.cp)
    T0 = _as_positive_float("stage.T0", stage.T0)
    p0 = _as_positive_float("stage.p0", stage.p0)
    H0 = _as_positive_float("stage.H0", stage.H0)
    p2 = _as_positive_float("stage.p2", stage.p2)
    D_c = _as_positive_float("stage.D_c", stage.D_c)
    l_c = _as_float("stage.l_c", stage.l_c)
    c1 = _as_float("stage.c1", stage.c1)
    c1u = _as_float("stage.c1u", stage.c1u)
    T1_ad = _as_float("stage.T1_ad", stage.T1_ad)
    alpha1_deg = _as_float("stage.alpha1_deg", stage.alpha1_deg)

    if k == 1.0:
        raise ReactionCalculationError(f"k must differ from 1, got {k!r}")

    r_c = D_c / 2
    r1 = r_c - 0.5 * l_c
    if r1 == 0.0:
        raise ReactionCalculationError(f"root radius r1 must be nonzero, got {r1!r}")

    if law == "Γ":
        R_value = _as_positive_float("R", R)
        if T1_ad <= 0.0:
            raise ReactionCalculationError(
                f"stage.T1_ad must be > 0, got {T1_ad!r}"
            )
        rotor_mach_denominator = math.sqrt(k * R_value * T1_ad)
        M1u = c1u / rotor_mach_denominator
        pressure_base = 1 + (k - 1) / 2 * M1u**2 * (1 - (r_c / r1) ** 2)
        p1 = stage.p1 * pressure_base ** (k / (k - 1))
    else:
        phi_value = _as_positive_float("phi", phi)
        c1_root = c1 * (r_c / r1) ** (math.cos(math.radians(alpha1_deg)) ** 2)
        pressure_base = 1 - (c1_root / phi_value) ** 2 / (2 * cp * T0)
        p1 = p0 * pressure_base ** (k / (k - 1))

    T1ad = T0 * (p1 / p0) ** ((k - 1) / k)
    return cp * T1ad / H0 * (1 - (p2 / p1) ** ((k - 1) / k))


__all__ = [
    "ReactionCalculationError",
    "calculate_root_reaction",
]
