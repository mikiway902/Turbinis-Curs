"""EXPERIMENTAL read-only entropy budget for the Stage 6B-4.5-B fields.

This module contains diagnostic algebra only.  It does not add an unknown, a
solved equation, a closure, or a boundary condition.  The ideal-gas entropy
difference is ACCEPTED physics; the inherited rotor energy/``psi`` loss and the
book meanline pressure are EXPERIMENTAL/``book_approx`` inputs.  The budget is
a NUMERICAL APPROXIMATION decomposition of the already-evaluated absolute
cross-rotor residual:

    E_s = cp ln(T2/T1) - R ln(p2/p1) - ds_l

and, at one anchor,

    E_s = E_book
          + [-cp ln(T1_a/T1*) + R ln(p1_a/p1*)]
          + [ cp ln(T2_a/T2*) ]
          + [-R ln(p2_a/p2*) ]
          + [-(ds_l_a - h_l*/T2*)].

Asterisks denote the book meanline state.  ``E_book`` is the book-state
evaluation and ``h_l*`` is ``(psi^-2 - 1) w2*^2 / 2`` [J/kg].  Every entropy
term is J/(kg K); temperatures are K and pressures Pa.  The sum is checked
against :func:`turbine.radial.rotor_exit.absolute_entropy_residual`; the module
does not decide whether the experimental closure is acceptable.

The book identities are conditional:

    T2 - T2_ad = h_l/cp,
    cp/R = k/(k-1),
    p2 = p1 (T2_ad/T1)^(k/(k-1)),
    cp ln(T2/T2_ad) = -cp ln(1 - x),  x = h_l/(cp T2).

When an assumption fails the conditional comparison is reported as
``NOT_APPLICABLE`` with a reason; it is never silently forced.  Invalid
positive/finite state, ``cp <= 0``, ``R <= 0``, ``psi`` outside ``(0, 1]``, or
``x >= 1`` raise :class:`DomainError` instead of returning NaN.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import rotor_exit as core

FloatArray = NDArray[np.float64]

ENERGY_IDENTITY_ABS_TOL_K = 1e-9
ENERGY_IDENTITY_REL_TOL = 1e-10
CP_R_IDENTITY_REL_TOL = 1e-10


class DomainError(ValueError):
    """Raised when a diagnostic input is outside the valid domain."""


def _finite(name: str, value: ArrayLike) -> FloatArray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise DomainError(f"{name} must be finite and non-empty")
    return arr


def _positive(name: str, value: ArrayLike) -> FloatArray:
    arr = _finite(name, value)
    if np.any(arr <= 0.0):
        raise DomainError(f"{name} must be strictly positive")
    return arr


def _nonnegative(name: str, value: ArrayLike) -> FloatArray:
    arr = _finite(name, value)
    if np.any(arr < 0.0):
        raise DomainError(f"{name} must be non-negative")
    return arr


def _scalar_positive(name: str, value: float) -> float:
    out = float(_positive(name, value).reshape(()))
    return out


def _state_value(st: dict, *keys: str):
    for key in keys:
        if key in st:
            return st[key]
    return None


def book_rotor_loss(w2: ArrayLike, psi: float) -> ArrayLike:
    """Return the inherited book rotor loss ``(psi^-2 - 1) w2^2 / 2``.

    ``w2`` is relative velocity [m/s] and the result is [J/kg].  ``psi`` must
    be in ``(0, 1]``; ``psi = 1`` gives exactly zero loss.
    """
    psi = _scalar_positive("psi", psi)
    if psi > 1.0:
        raise DomainError("psi must satisfy 0 < psi <= 1")
    w2 = _nonnegative("w2", w2)
    hl = (1.0 / psi**2 - 1.0) * w2**2 / 2.0
    if np.any(~np.isfinite(hl)) or np.any(hl < 0.0):
        raise DomainError("rotor loss must be finite and non-negative")
    return hl


def log_loss_comparison(hl: float, T2: float, cp: float) -> dict:
    """Compare the conditional logarithmic loss entropy with ``hl/T2``.

    ``x = hl/(cp T2)``.  Requires ``0 <= x < 1``.  Returns ``ds_state`` from
    ``-cp ln(1-x)``, the current approximation ``ds_l = hl/T2``, their
    difference, and the leading small-``x`` term ``cp x^2/2``.  This is a
    diagnostic identity, not an alternative loss model.
    """
    hl = float(_nonnegative("hl", hl).reshape(()))
    T2 = _scalar_positive("T2", T2)
    cp = _scalar_positive("cp", cp)
    x = hl / (cp * T2)
    if not (0.0 <= x < 1.0):
        raise DomainError("x = hl/(cp T2) must satisfy 0 <= x < 1")
    ds_state = -cp * math.log1p(-x)
    ds_l = hl / T2
    difference = ds_state - ds_l
    return {
        "x": float(x),
        "ds_state": float(ds_state),
        "ds_l": float(ds_l),
        "difference": float(difference),
        "small_x_approximation": float(0.5 * cp * x * x),
    }


def book_meanline_diagnostics(st: dict, *, R: float, psi: float) -> dict:
    """Evaluate the inherited meanline identities on one book state.

    Required keys: ``T1, T2, p1, p2, w2, cp`` and either ``T2_ад`` or
    ``T2ad``.  ``k`` and ``h_л``/``h_l`` are used when present.  The
    conditional logarithmic identity is marked ``NOT_APPLICABLE`` when its
    assumptions fail; no invalid log is evaluated.
    """
    cp = _scalar_positive("cp", st["cp"])
    R = _scalar_positive("R", R)
    T1 = _scalar_positive("T1", st["T1"])
    T2 = _scalar_positive("T2", st["T2"])
    p1 = _scalar_positive("p1", st["p1"])
    p2 = _scalar_positive("p2", st["p2"])
    w2 = _nonnegative("w2", st["w2"])
    T2_ad_raw = _state_value(st, "T2_ад", "T2ad", "T2_ad")
    if T2_ad_raw is None:
        raise DomainError("book state needs T2_ад or T2ad")
    T2_ad = _scalar_positive("T2_ад", T2_ad_raw)

    hl = float(book_rotor_loss(w2, psi))
    hl_key_value = _state_value(st, "h_л", "h_l")
    if hl_key_value is None:
        hl_check = {"key_present": False, "key_value": None,
                    "key_abs_difference": None, "key_relative_difference": None}
    else:
        key = float(_nonnegative("h_l", hl_key_value))
        abs_diff = key - hl
        rel_diff = abs_diff / hl if hl > 0.0 else abs_diff
        hl_check = {"key_present": True, "key_value": key,
                    "key_abs_difference": float(abs_diff),
                    "key_relative_difference": float(rel_diff)}

    identity_K = (T2 - T2_ad) - hl / cp
    identity_scale = max(abs(T2), abs(T2_ad))
    identity_limit = max(
        ENERGY_IDENTITY_ABS_TOL_K, ENERGY_IDENTITY_REL_TOL * identity_scale
    )
    energy_applicable = abs(identity_K) <= identity_limit

    k_raw = st.get("k")
    if k_raw is None:
        cp_r_identity = {"applicable": False, "reason": "k is absent"}
        pressure_reconstruction = {"applicable": False, "reason": "k is absent"}
    else:
        k = _scalar_positive("k", k_raw)
        if not k > 1.0:
            raise DomainError("k must be greater than 1")
        cp_over_R = cp / R
        k_over_k_minus_1 = k / (k - 1.0)
        cp_r_diff = cp_over_R - k_over_k_minus_1
        cp_r_limit = CP_R_IDENTITY_REL_TOL * max(
            abs(cp_over_R), abs(k_over_k_minus_1), 1.0
        )
        cp_r_identity = {
            "applicable": bool(abs(cp_r_diff) <= cp_r_limit),
            "cp_over_R": float(cp_over_R),
            "k_over_k_minus_1": float(k_over_k_minus_1),
            "absolute_difference": float(cp_r_diff),
            "relative_difference": float(
                abs(cp_r_diff) / max(abs(cp_over_R), abs(k_over_k_minus_1), 1e-30)
            ),
            "reason": None if abs(cp_r_diff) <= cp_r_limit else
                      "cp/R differs from k/(k-1)",
        }
        pressure_recon = p1 * (T2_ad / T1) ** (k / (k - 1.0))
        pressure_residual = pressure_recon - p2
        pressure_reconstruction = {
            "applicable": True,
            "reconstructed_p2": float(pressure_recon),
            "absolute_residual": float(pressure_residual),
            "relative_residual": float(pressure_residual / p2),
            "reason": None,
        }

    ds_state_book = cp * math.log(T2 / T2_ad)
    ds_l = hl / T2
    x = hl / (cp * T2)
    if not (0.0 <= x < 1.0):
        conditional = {
            "x": float(x), "status": "NOT_APPLICABLE",
            "reason": "x = hl/(cp T2) is outside [0, 1)",
            "ds_state": None, "difference": None, "small_x_approximation": None,
        }
    elif not energy_applicable:
        conditional = {
            "x": float(x), "status": "NOT_APPLICABLE",
            "reason": "T2 - T2_ад = h_l/cp is not satisfied",
            "ds_state": None, "difference": None, "small_x_approximation": None,
        }
    else:
        ds_state_cond = -cp * math.log1p(-x)
        conditional = {
            "x": float(x), "status": "APPLICABLE", "reason": None,
            "ds_state": float(ds_state_cond),
            "difference": float(ds_state_cond - ds_l),
            "small_x_approximation": float(0.5 * cp * x * x),
        }

    return {
        "cp": cp,
        "R": R,
        "psi": float(psi),
        "T1": T1,
        "T2": T2,
        "T2_ад": T2_ad,
        "p1": p1,
        "p2": p2,
        "w2": float(w2),
        "hl": {"value": hl, **hl_check},
        "energy_identity_residual_K": float(identity_K),
        "energy_identity_residual_J_kg": float(cp * identity_K),
        "energy_identity_limit_K": float(identity_limit),
        "energy_identity_applicable": bool(energy_applicable),
        "cp_over_R": cp_r_identity,
        "pressure_reconstruction": pressure_reconstruction,
        "ds_state_book": float(ds_state_book),
        "ds_l": float(ds_l),
        "conditional_log": conditional,
    }


def entropy_budget(
    st: dict,
    g: dict,
    f: dict,
    *,
    anchor: int | None = None,
    absolute_residual: ArrayLike | None = None,
) -> dict:
    """Return the signed per-term attribution of ``E_s`` at one anchor.

    ``st`` is the book meanline state; ``g`` and ``f`` are the saved setup and
    :func:`turbine.radial.rotor_exit.fields` output.  ``cp`` must agree between
    ``st`` and ``g``; ``R`` and ``psi`` come from ``g``.  The terms are exactly
    the decomposition in the module docstring; ``sum`` is checked against
    ``absolute_residual`` (or the shared core residual when omitted).
    """
    cp = _scalar_positive("g.cp", g["cp"])
    R = _scalar_positive("g.R", g["R"])
    psi = _scalar_positive("g.psi", g["psi"])
    if psi > 1.0:
        raise DomainError("psi must satisfy 0 < psi <= 1")
    st_cp = _scalar_positive("st.cp", st.get("cp", cp))
    if not math.isclose(st_cp, cp, rel_tol=1e-12, abs_tol=0.0):
        raise DomainError("st.cp and g.cp must be the same")
    if "R" in st and not math.isclose(
        _scalar_positive("st.R", st["R"]), R, rel_tol=1e-12, abs_tol=0.0
    ):
        raise DomainError("st.R and g.R must be the same")

    st_T1 = _scalar_positive("st.T1", st["T1"])
    st_T2 = _scalar_positive("st.T2", st["T2"])
    st_p1 = _scalar_positive("st.p1", st["p1"])
    st_p2 = _scalar_positive("st.p2", st["p2"])
    st_w2 = _nonnegative("st.w2", st["w2"])

    T1 = _positive("g.T1", g["T1"])
    p1 = _positive("g.p1", g["p1"])
    T2 = _positive("f.T2", f["T2"])
    p2 = _positive("f.p", f["p"])
    dsl = _nonnegative("f.dsl", f["dsl"])
    if T1.shape != p1.shape or T1.shape != T2.shape or T1.shape != p2.shape:
        raise DomainError("g.T1, g.p1, f.T2 and f.p must have matching shapes")
    if T1.shape != dsl.shape:
        raise DomainError("f.dsl must match the state shape")

    a = int(g["anchor"]) if anchor is None else int(anchor)
    if not (0 <= a < T1.size):
        raise DomainError("anchor index is outside the fields")

    hl_book = float(book_rotor_loss(st_w2, psi))
    book_ds_l = hl_book / st_T2

    book_offset = (
        cp * math.log(st_T2 / st_T1)
        - R * math.log(st_p2 / st_p1)
        - book_ds_l
    )
    inlet_state_shift = (
        -cp * math.log(float(T1[a]) / st_T1)
        + R * math.log(float(p1[a]) / st_p1)
    )
    exit_temperature_shift = cp * math.log(float(T2[a]) / st_T2)
    exit_pressure_shift = -R * math.log(float(p2[a]) / st_p2)
    loss_shift = -(float(dsl[a]) - book_ds_l)

    terms = {
        "book_offset": float(book_offset),
        "inlet_state_shift": float(inlet_state_shift),
        "exit_temperature_shift": float(exit_temperature_shift),
        "exit_pressure_shift": float(exit_pressure_shift),
        "loss_shift": float(loss_shift),
    }
    total = float(sum(terms.values()))

    if absolute_residual is None:
        residual = core.absolute_entropy_residual(g, f)
    else:
        residual = absolute_residual
    residual = np.asarray(residual, dtype=float)
    if residual.shape != T1.shape or np.any(~np.isfinite(residual)):
        raise DomainError("absolute residual must be finite and match the fields")

    ratios = {
        "T1_a_over_book_T1": float(T1[a]) / st_T1,
        "p1_a_over_book_p1": float(p1[a]) / st_p1,
        "T2_a_over_book_T2": float(T2[a]) / st_T2,
        "p2_a_over_book_p2": float(p2[a]) / st_p2,
        "dsl_a_over_book_ds_l": float(dsl[a]) / book_ds_l if book_ds_l != 0 else None,
    }
    current = {
        "T1_a": float(T1[a]),
        "p1_a": float(p1[a]),
        "T2_a": float(T2[a]),
        "p2_a": float(p2[a]),
        "dsl_a": float(dsl[a]),
    }
    for key in ("ca", "w2", "hl"):
        if key in f and np.asarray(f[key]).shape == T1.shape:
            current[f"{key}_a"] = float(np.asarray(f[key])[a])
    if "c2a" in st and "ca_a" in current:
        ratios["c2a_a_over_book_c2a"] = current["ca_a"] / float(st["c2a"])
    if "w2" in st and "w2_a" in current:
        ratios["w2_a_over_book_w2"] = current["w2_a"] / st_w2
    if hl_book != 0.0 and "hl_a" in current:
        ratios["hl_a_over_book_hl"] = current["hl_a"] / hl_book

    return {
        "anchor": a,
        "terms": terms,
        "sum": total,
        "absolute_entropy_residual": float(residual[a]),
        "mismatch": float(total - residual[a]),
        "absolute": {
            "min": float(np.min(residual)),
            "max": float(np.max(residual)),
            "mean": float(np.mean(residual)),
            "max_abs": float(np.max(np.abs(residual))),
            "span": float(np.max(residual) - np.min(residual)),
        },
        "book": {
            "T1": st_T1,
            "T2": st_T2,
            "p1": st_p1,
            "p2": st_p2,
            "w2": st_w2,
            "hl": hl_book,
            "ds_l": book_ds_l,
        },
        "current": current,
        "ratios": ratios,
    }
