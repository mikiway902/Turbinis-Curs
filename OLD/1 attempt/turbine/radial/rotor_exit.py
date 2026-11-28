"""EXPERIMENTAL rotor-exit closure algebra for system B.

This module owns the pure algebraic part of the Stage 6B-4.5-B coupled
rotor-exit closure so the optimizer does not keep a private copy of physical
relations (project rule: optimizers must not own private equations).

Status: EXPERIMENTAL.  Nothing here is accepted design physics.  The inherited
``book_approx`` stator profiles, inlet pressure law, rotor energy relation and
``psi`` loss entropy are reproduced unchanged; the exact Euler work law is used
to eliminate ``c2u`` per streamline.

Equation count (explicit)
-------------------------
Unknowns are ``y_i = ln(c2a_i / c2a_ref)``, ``N`` of them.  The square branch
solver uses ``N-1`` finite-volume radial-momentum interval residuals plus one
integral mass-flow residual (``N`` equations).  The ``p2`` construction
enforces ``N-1`` relative entropy/state equations plus the prescribed pressure
BC, leaving one absolute cross-rotor entropy condition
:func:`absolute_entropy_residual`.  The complete required set is therefore
``N-1`` momentum + 1 mass + 1 absolute entropy = ``N+1`` conditions on ``N``
unknowns.  The absolute entropy condition is a compatibility residual of the
prescribed control point, not an extra unknown.  ``residual`` and
``jacobian`` below are the square momentum+mass branch solver only.

All interfaces are SI.  Angles in the input ``st`` mapping are degrees.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, eye as sparse_eye

FloatArray = NDArray[np.float64]
PSI_DEFAULT = 0.96


class InvalidState(RuntimeError):
    """Raised when an iterate leaves the positive thermodynamic domain."""


def stator_profile(
    st: dict, law: str, R: float, phi: float, r: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return existing book stator ``c1u, c1a, |c1|, T1`` on radii ``r``.

    ``Gamma`` uses free-vortex swirl; ``alpha1`` uses the book exponential
    profile.  ``R`` and ``phi`` are kept in the signature for interface parity
    with the legacy notebook call.
    """
    r = np.asarray(r, dtype=float)
    r_c = st["D_c"] / 2
    if law == "Γ":
        cu = st["c1u"] * r_c / r
        ca = np.full_like(r, st["c1a"])
    elif law == "α1":
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        cu, ca = st["c1u"] * q, st["c1a"] * q
    else:
        raise ValueError(f"unknown profiling law {law!r}")
    c1 = np.hypot(cu, ca)
    T1 = st["T0"] - c1**2 / (2 * st["cp"])
    return cu, ca, c1, T1


def inlet_pressure(st: dict, law: str, R: float, phi: float, r: ArrayLike) -> FloatArray:
    """Return the existing book stage-inlet static pressure law on ``r``."""
    r = np.asarray(r, dtype=float)
    r_c = st["D_c"] / 2
    k = st["k"]
    if law == "Γ":
        M1u = st["c1u"] / math.sqrt(k * R * st["T1_ад"])
        return st["p1"] * (
            1 + (k - 1) / 2 * M1u**2 * (1 - (r_c / r) ** 2)
        ) ** (k / (k - 1))
    if law == "α1":
        q = (r_c / r) ** (math.cos(math.radians(st["α1"])) ** 2)
        c1 = np.hypot(st["c1u"] * q, st["c1a"] * q)
        return st["p0"] * (1 - (c1 / phi) ** 2 / (2 * st["cp"] * st["T0"])) ** (
            k / (k - 1)
        )
    raise ValueError(f"unknown profiling law {law!r}")


def trapezoid_weights(r: ArrayLike) -> FloatArray:
    """Weights such that ``sum(c_i g_i) == np.trapezoid(g*2*pi*r, r)``."""
    r = np.asarray(r, dtype=float)
    dr = np.diff(r)
    c = np.zeros_like(r)
    c[:-1] += math.pi * r[:-1] * dr
    c[1:] += math.pi * r[1:] * dr
    return c


def build_setup(
    st: dict,
    n: float,
    law: str,
    R: float,
    phi: float,
    N: int,
    bc: str = "mean",
    psi: float = PSI_DEFAULT,
) -> dict:
    """Build the fixed radial stations and algebraic coefficients."""
    N = int(N)
    if N < 5:
        raise ValueError("N must be at least 5")
    if bc not in ("mean", "hub"):
        raise ValueError("bc must be 'mean' or 'hub'")
    r_c, r_l = st["D_c"] / 2, st["D_л"] / 2
    r1 = np.linspace(r_c - 0.5 * st["l_c"], r_c + 0.5 * st["l_c"], N)
    r2 = np.linspace(r_l - 0.5 * st["l_л"], r_l + 0.5 * st["l_л"], N)
    cu1, ca1, c1, T1 = stator_profile(st, law, R, phi, r1)
    p1 = inlet_pressure(st, law, R, phi, r1)
    u1 = math.pi * n * r1 / 30.0
    u2 = math.pi * n * r2 / 30.0
    w1 = np.hypot(ca1, cu1 - u1)

    # Exact Euler work law on every streamline with the existing scalar target.
    c2u = (st["L_u"] - cu1 * u1) / u2
    if not np.all(np.isfinite(c2u)):
        raise ValueError("non-finite Euler tangential velocity")
    vt2 = c2u + u2  # book sign convention used by radial_exit

    cp, k = st["cp"], st["k"]
    mid = N // 2
    s1 = cp * np.log(T1 / T1[mid]) - R * np.log(p1 / p1[mid])
    A = T1 + (w1**2 + u2**2 - u1**2) / (2 * cp)
    Aw = 1.0 / psi**2 - 1.0

    anchor = mid if bc == "mean" else 0
    p_bc = st["p2"]
    rho_scale = st["p2"] / (R * st["T2"])
    a_scale = float(np.max(np.abs(c2u**2 / r2)))
    if not np.isfinite(a_scale) or a_scale <= 0:
        raise ValueError("non-positive pre-solve acceleration scale")
    return {
        "N": N,
        "r1": r1,
        "r2": r2,
        "cu1": cu1,
        "ca1": ca1,
        "T1": T1,
        "p1": p1,
        "u1": u1,
        "u2": u2,
        "w1": w1,
        "cu2": c2u,
        "vt2": vt2,
        "cp": cp,
        "k": k,
        "R": R,
        "psi": psi,
        "Aw": Aw,
        "A": A,
        "s1": s1,
        "mid": mid,
        "anchor": anchor,
        "p_bc": p_bc,
        "ca_ref": st["c2a"],
        "L_u": st["L_u"],
        "G_l": st["G_л"],
        "rho_scale": rho_scale,
        "a_scale": a_scale,
        "radial_scale": rho_scale * a_scale,
        "weights": trapezoid_weights(r2),
        "bc": bc,
        "law": law,
    }


def fields(g: dict, y: ArrayLike) -> dict:
    """Evaluate all algebraic fields from the log axial velocities."""
    y = np.asarray(y, dtype=float)
    if y.shape != (g["N"],) or np.any(~np.isfinite(y)):
        raise InvalidState("y must be a finite vector of length N")
    if np.any(np.abs(y) > g.get("y_bound", 1.0) + 1e-9):
        raise InvalidState("y outside solver bounds")
    ca = g["ca_ref"] * np.exp(y)
    w2 = np.hypot(ca, g["vt2"])
    T2 = g["A"] - w2**2 / (2 * g["cp"])
    if np.any(~np.isfinite(T2)) or np.any(T2 <= 0) or np.any(ca <= 0):
        raise InvalidState("non-positive/non-finite T2 or c2a")
    hl = g["Aw"] * w2**2 / 2.0
    dsl = hl / T2
    s2 = g["s1"] + dsl
    g_state = g["cp"] * np.log(T2) - s2
    a = g["anchor"]
    p2 = g["p_bc"] * np.exp((g_state - g_state[a]) / g["R"])
    rho2 = p2 / (g["R"] * T2)
    if np.any(~np.isfinite(p2)) or np.any(p2 <= 0) or np.any(rho2 <= 0):
        raise InvalidState("non-positive/non-finite p2 or rho2")
    return {
        "ca": ca,
        "w2": w2,
        "T2": T2,
        "hl": hl,
        "dsl": dsl,
        "s2": s2,
        "g_state": g_state,
        "p": p2,
        "rho": rho2,
    }


def absolute_entropy_residual(g: dict, f: dict) -> FloatArray:
    """Return the absolute cross-rotor entropy residual ``E_s`` [J/(kg K)].

    ``E_s = cp ln(T2/T1) - R ln(p2/p1) - Delta_s_l`` with the same absolute
    inlet ``p1, T1`` and exit ``p2, T2`` reference; ``Delta_s_l = h_l / T2``
    and ``h_l = (psi^-2 - 1) w2^2 / 2``.  The ideal-gas entropy difference is
    ACCEPTED PHYSICS; ``Delta_s_l`` is the EXPERIMENTAL/EMPIRICAL book
    approximation; the ``p2`` boundary condition is a SOFTWARE ASSUMPTION.
    This is a compatibility residual of the prescribed control point, not an
    unknown.  At zero swirl/loss (``w2 = 0``, ``T2 = T1``, ``p2 = p1``) it
    vanishes.  Because a constant offset cancels from radial derivatives, the
    ``R_p``/``R_T``/``R_GRE`` equivalence cannot detect it; this absolute value
    is the only check on that offset.  It is not exported through
    ``turbine.radial.__all__``.
    """
    return (
        g["cp"] * np.log(f["T2"] / g["T1"])
        - g["R"] * np.log(f["p"] / g["p1"])
        - f["dsl"]
    )


def derivatives(g: dict, f: dict) -> tuple[FloatArray, ...]:
    """Analytic derivatives of the algebraic fields with respect to ``y``."""
    ca, T2, hl, p = f["ca"], f["T2"], f["hl"], f["p"]
    R, cp, a = g["R"], g["cp"], g["anchor"]
    dT2 = -ca**2 / cp
    dG = -ca**2 * ((1.0 / g["psi"] ** 2) / T2 + hl / (T2**2 * cp))
    p2_dy = np.zeros_like(p)
    p2_da = np.zeros_like(p)
    mask = np.arange(p.size) != a
    p2_dy[mask] = p[mask] * dG[mask] / R
    p2_da[mask] = -p[mask] * dG[a] / R
    drho_dy = p2_dy / (R * T2) - p / (R * T2**2) * dT2
    drho_da = p2_da / (R * T2)
    return dT2, p2_dy, p2_da, drho_dy, drho_da


def residual(y: ArrayLike, g: dict) -> FloatArray:
    """Return the N residuals: N-1 radial FV equations plus one mass equation."""
    try:
        f = fields(g, y)
    except InvalidState:
        return np.full(g["N"], 1e6)
    p, rho, cu2 = f["p"], f["rho"], g["cu2"]
    r = g["r2"]
    dr = np.diff(r)
    fv = (p[1:] - p[:-1]) / dr - 0.5 * (
        rho[:-1] * cu2[:-1] ** 2 / r[:-1] + rho[1:] * cu2[1:] ** 2 / r[1:]
    )
    mass = np.trapezoid(rho * f["ca"] * 2 * math.pi * r, r) / g["G_l"] - 1.0
    return np.concatenate([fv / g["radial_scale"], [mass]])


def jacobian(y: ArrayLike, g: dict) -> csr_matrix:
    """Sparse analytic Jacobian of :func:`residual`."""
    try:
        f = fields(g, y)
    except InvalidState:
        return (sparse_eye(g["N"], format="csr") * 1e6).tocsr()
    N = g["N"]
    r = g["r2"]
    dr = np.diff(r)
    cu2_over_r = g["cu2"] ** 2 / r
    _, p2_dy, p2_da, drho_dy, drho_da = derivatives(g, f)

    rows: list[float] = []
    cols: list[float] = []
    vals: list[float] = []
    k = np.arange(N - 1)
    rows.extend(k.tolist())
    cols.extend(k.tolist())
    vals.extend((-p2_dy[:-1] / dr - 0.5 * cu2_over_r[:-1] * drho_dy[:-1]).tolist())
    rows.extend(k.tolist())
    cols.extend((k + 1).tolist())
    vals.extend((p2_dy[1:] / dr - 0.5 * cu2_over_r[1:] * drho_dy[1:]).tolist())
    # anchor coupling: p2_i depends on y_anchor for every i != anchor
    rows.extend(k.tolist())
    cols.extend([g["anchor"]] * (N - 1))
    vals.extend(
        (
            np.diff(p2_da) / dr
            - 0.5 * (cu2_over_r[:-1] * drho_da[:-1] + cu2_over_r[1:] * drho_da[1:])
        ).tolist()
    )

    # global mass row (dense in y because the integral couples every node)
    weights = g["weights"]
    ca, rho = f["ca"], f["rho"]
    dg_dy = weights * (drho_dy * ca + rho * ca)
    dg_da = weights * (drho_da * ca)
    rows.extend([N - 1] * N)
    cols.extend(np.arange(N).tolist())
    vals.extend((dg_dy / g["G_l"]).tolist())
    rows.extend([N - 1] * N)
    cols.extend([g["anchor"]] * N)
    vals.extend((dg_da / g["G_l"]).tolist())

    J = csr_matrix((vals, (rows, cols)), shape=(N, N))
    J = J / g["radial_scale"]
    # undo the mass-row scaling: the mass equation is already relative
    J.data[J.indptr[N - 1] :] *= g["radial_scale"]
    return J.tocsr()
