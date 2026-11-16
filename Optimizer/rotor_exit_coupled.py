"""Stage 6B-4.5-B: coupled rotor-exit closure (EXPERIMENTAL).

Reduced square system on N equal-span radial nodes.

Unknowns (N)
------------
``y_i = ln(c2a_i / c2a_meanline)`` for ``i = 0..N-1``.  The value is read as a
global axial scale ``lambda_a = exp(y_mid)`` plus ``N-1`` shape deviations
``delta_i = y_i - y_mid`` (``delta_mid = 0``).  ``c2a_i`` stays positive by
construction and the solver bounds are applied to ``y``.

Equations (N) — square branch subsystem
---------------------------------------
This square set is the branch solver only.  The complete compatibility set is
``N+1``: the ``N`` equations below plus the absolute cross-rotor entropy
residual in ``turbine.radial.rotor_exit.absolute_entropy_residual``.  The
extra condition is a compatibility residual of the prescribed control point,
not a solvable unknown.

``N-1`` finite-volume radial-momentum interval equations, using the
trapezoidal face average of ``rho c2u^2/r``

    (p_{i+1} - p_i) / (r_{i+1} - r_i)
        = 0.5 * (rho_i c2u_i^2/r_i + rho_{i+1} c2u_{i+1}^2/r_{i+1})

plus one global integral mass equation

    integral rho2 c2a 2*pi*r dr = G_l.

Radial residuals are divided by one fixed pre-solve scale
``rho_scale * max|c2u^2/r|`` (meanline ``p2/(R T2)`` density, no iterate
maximum); the mass residual stays relative.  The log variables enforce
``c2a > 0``, solver bounds are ``|y| <= 1``, and every field evaluation checks
positivity and finiteness explicitly.

Algebraic eliminations (independently checked, not counted as unknowns)
-----------------------------------------------------------------------
Implemented in ``turbine.radial.rotor_exit`` and called from here:
* ``c2u_i`` from the existing Euler work law at every streamline
  ``L_u(r) = c1u*u1 + c2u*u2`` with the existing scalar stage ``L_u`` target.
* ``T2_i`` from the inherited rotor relative-energy relation
  ``T2 = T1 + (w1^2 - w2^2 + u2^2 - u1^2)/(2 cp)``.
* Rotor loss ``h_l = (psi^-2 - 1) w2^2 / 2`` and the experimental entropy
  increment ``Delta s_l = h_l / T2`` (psi unchanged, no tuning).
* Inlet state entropy ``s1 = cp ln(T1/T1_anchor) - R ln(p1/p1_anchor)``.
  ``h/T`` is *not* used as exact entropy.
* ``s2_target = s1 + Delta s_l``.
* ``p2`` from the ideal-gas entropy difference with one pressure BC.
  Primary BC: ``p2(mean) = st['p2']``.  Alternative sensitivity:
  ``p2(hub) = st['p2']`` (never chosen from reaction/pass).
* ``rho2 = p2/(R T2)``.

Classification
--------------
Engineering model remains EXPERIMENTAL even when all numerical checks pass.
``YES`` requires every acceptance check on the best converged finest-mesh
solution, at least two genuinely independent cold starts on the same branch on
some mesh, a branch continuously refined to the finest mesh, and available
conditioning evidence.  ``NO`` additionally requires the same failure across
mesh and the alternative pressure BC with a clear margin.  Warm/continuation
starts never count as independent, and a failed solver is never evidence of
non-existence.  Anything else is ``INCONCLUSIVE``.

Run::

    python Optimizer/rotor_exit_coupled.py
    python Optimizer/rotor_exit_coupled.py --quick

The module does not modify accepted formulas, baselines, or engineering docs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator, aslinearoperator, onenormest, splu

ROOT = Path(__file__).resolve().parent.parent
OPTIMIZER = Path(__file__).resolve().parent
for _path in (str(OPTIMIZER), str(ROOT / "Turbinis-Calc"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from turbine.radial import rotor_exit as core  # noqa: E402
from turbine.radial.equilibrium import (  # noqa: E402
    general_residual,
    pressure_residual,
    thermodynamic_residual,
)

RESULTS = OPTIMIZER / "results"
PSI = 0.96
Y_BOUND = 1.0
EPS_RRE_TOL = 0.01          # 1 %
MASS_TOL = 0.005            # 0.5 %
WORK_TOL = 1e-6
R_FORM_TOL = 1e-4           # numerical finite-difference consistency, not %-level
SMOOTH_TOL = 0.5            # max |second difference| / max |first difference|
SAME_SOLUTION_RTOL = 1e-2   # max relative c2a difference counted as the same solution
ABS_ENTROPY_TOL = 1e-6      # J/(kg K): strict algebraic closure, not model uncertainty
SOLVER_CLOSURE_TOL = 1e-5   # dimensionless max|core.residual| for a solved branch member
STD_TOL = dict(xtol=1e-10, ftol=1e-10, gtol=1e-10)
LOOSE_TOL = dict(xtol=1e-6, ftol=1e-6, gtol=1e-6)
STAGES = (1, 2, 3)
LAWS = {1: "Γ", 2: "α1", 3: "Γ"}
MESHES = (41, 81, 161, 401, 801)
COND_SEED = 20261109
InvalidState = core.InvalidState


def _stator(st, n, law, R, phi, r):
    """Backward-compatible wrapper for the shared stator profile."""
    return core.stator_profile(st, law, R, phi, r)


def _inlet_pressure(st, law, R, phi, r):
    """Backward-compatible wrapper for the shared inlet pressure law."""
    return core.inlet_pressure(st, law, R, phi, r)


def _trapezoid_weights(r):
    """Backward-compatible wrapper for the shared integral weights."""
    return core.trapezoid_weights(r)


def build_setup(st, n, law, R, phi, N, bc="mean", psi=PSI):
    """Build the fixed radial station and algebraic coefficients."""
    return core.build_setup(st, n, law, R, phi, N, bc=bc, psi=psi)


def fields(g, y):
    """Evaluate all algebraic fields from the log axial velocities."""
    return core.fields(g, y)


def _derivatives(g, f):
    """Backward-compatible wrapper for the shared analytic derivatives."""
    return core.derivatives(g, f)


def residual_vector(y, g):
    """Return the N residuals: N-1 radial FV equations plus one mass equation."""
    return core.residual(y, g)


def jacobian(y, g):
    """Sparse analytic Jacobian of :func:`residual_vector`."""
    return core.jacobian(y, g)



def _closed(run):
    """True only for a valid run whose momentum+mass residual is numerically closed."""
    residual_inf = run.get("equation_residual_inf")
    return bool(
        run.get("valid")
        and residual_inf is not None
        and math.isfinite(residual_inf)
        and residual_inf <= SOLVER_CLOSURE_TOL
    )


def _branch_runs(runs, best):
    """Numerically closed runs geometrically on the best branch."""
    if best is None or not _closed(best):
        return []
    return [r for r in runs if _closed(r) and r.get("same_as_best")]


def _unfinished_near(runs, best):
    """Valid, geometrically near, but numerically unclosed iterates."""
    if best is None:
        return []
    return [
        r for r in runs
        if r.get("valid") and r.get("same_as_best") and not _closed(r)
    ]


def smoothness_index(x):
    """Max |second difference| / max |first difference|; spikes approach 1."""
    x = np.asarray(x, dtype=float)
    d1 = np.diff(x)
    d2 = np.diff(x, 2)
    scale = float(np.max(np.abs(d1)))
    if scale <= 0:
        return 0.0
    return float(np.max(np.abs(d2)) / scale)


def sign_changes(x):
    s = np.sign(np.asarray(x, dtype=float))
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0


def metrics(g, y, st):
    """Independent diagnostics for one solved field."""
    f = fields(g, y)
    r = g["r2"]
    p, T, rho, ca, cu = f["p"], f["T2"], f["rho"], f["ca"], g["cu2"]
    dpr = np.gradient(p, r, edge_order=2) / rho
    denom = float(np.max(np.abs(dpr)))
    if not np.isfinite(denom) or denom <= 0:
        raise InvalidState("non-positive pressure-gradient scale")

    # Accepted pressure-momentum residual, no entropy term.
    R_p = pressure_residual(r, p, T, cu, gas_constant=g["R"])
    # Independent thermodynamic/general forms using the target entropy field.
    R_T = thermodynamic_residual(r, T, f["s2"], cu, cp=g["cp"])
    R_GRE = general_residual(r, T, f["s2"], ca, cu, cp=g["cp"])

    epsilon_RRE = float(np.max(np.abs(R_p)) / denom)
    delta_T = float(np.max(np.abs(R_T - R_p)) / denom)
    delta_GRE = float(np.max(np.abs(R_GRE - R_p)) / denom)

    T2_energy = g["T1"] + (
        g["w1"] ** 2 - f["w2"] ** 2 + g["u2"] ** 2 - g["u1"] ** 2
    ) / (2 * g["cp"])
    energy_res = float(np.max(np.abs(T - T2_energy)) / np.max(T))
    L_u = g["cu1"] * g["u1"] + g["cu2"] * g["u2"]
    work_res = float(np.max(np.abs(L_u / g["L_u"] - 1.0)))
    Gm = float(np.trapezoid(rho * ca * 2 * math.pi * r, r))
    mass_rel = Gm / g["G_l"] - 1.0

    # Absolute cross-rotor entropy compatibility residual (see core docs).
    E_s = core.absolute_entropy_residual(g, f)

    alpha = np.degrees(np.arctan2(ca, cu))
    stage = g.get("stage", st.get("_stage"))
    alpha_lo = 80.0 if stage == 3 else 70.0
    alpha_hi = 90.0
    alpha2_violation = float(max(0.0, alpha_lo - alpha.min(), alpha.max() - alpha_hi))

    finite = bool(
        np.all(np.isfinite(p))
        and np.all(np.isfinite(T))
        and np.all(np.isfinite(rho))
        and np.all(np.isfinite(ca))
        and np.all(np.isfinite(cu))
    )
    smooth = float(max(smoothness_index(p), smoothness_index(T), smoothness_index(ca)))
    min_w2_ca = float(np.min(f["w2"] - ca))

    return {
        "valid": True,
        "epsilon_RRE": epsilon_RRE,
        "mass_rel": float(mass_rel),
        "mass_abs": float(abs(mass_rel)),
        "energy_res": energy_res,
        "work_res": work_res,
        "delta_T": delta_T,
        "delta_GRE": delta_GRE,
        "positive_finite": bool(finite and np.all(p > 0) and np.all(T > 0) and np.all(rho > 0) and np.all(ca > 0)),
        "smoothness": smooth,
        "min_w2_minus_ca": min_w2_ca,
        "alpha2_min": float(alpha.min()),
        "alpha2_max": float(alpha.max()),
        "alpha2_violation": alpha2_violation,
        "dpdr_sign_changes": sign_changes(dpr),
        "p_min": float(p.min()), "p_max": float(p.max()),
        "T_min": float(T.min()), "T_max": float(T.max()),
        "rho_min": float(rho.min()), "rho_max": float(rho.max()),
        "cu_min": float(cu.min()), "cu_max": float(cu.max()),
        "ca_min": float(ca.min()), "ca_max": float(ca.max()),
        "lambda": float(math.exp(y[g["mid"]])),
        "p_hub": float(p[0]), "p_tip": float(p[-1]),
        "rho_hub": float(rho[0]), "rho_tip": float(rho[-1]),
        "T_hub": float(T[0]), "T_tip": float(T[-1]),
        "alpha_hub": float(alpha[0]), "alpha_tip": float(alpha[-1]),
        "R_p_max": float(np.max(np.abs(R_p))),
        "R_T_max": float(np.max(np.abs(R_T))),
        "R_GRE_max": float(np.max(np.abs(R_GRE))),
        "entropy_min": float(np.min(E_s)),
        "entropy_max": float(np.max(E_s)),
        "entropy_mean": float(np.mean(E_s)),
        "entropy_max_abs": float(np.max(np.abs(E_s))),
        "entropy_span": float(np.max(E_s) - np.min(E_s)),
        "entropy_anchor_value": float(E_s[g["anchor"]]),
        "dpdr_scale": denom,
    }


def criteria_from_metrics(m):
    return {
        "epsilon_RRE": m["epsilon_RRE"] <= EPS_RRE_TOL,
        "mass": m["mass_abs"] <= MASS_TOL,
        "work": m["work_res"] <= WORK_TOL,
        "r_forms": m["delta_T"] <= R_FORM_TOL and m["delta_GRE"] <= R_FORM_TOL,
        "positive_finite": m["positive_finite"],
        "smooth": m["smoothness"] <= SMOOTH_TOL,
        "w2_gt_ca": m["min_w2_minus_ca"] > 0,
        "alpha2": m["alpha2_violation"] <= 0,
        "entropy_anchor": m["entropy_max_abs"] <= ABS_ENTROPY_TOL,
    }


def initial_guesses(N):
    """Three substantially different cold starts (different global scales)."""
    xi = np.linspace(0.0, 1.0, N)
    return {
        "uniform": np.zeros(N),
        "hub-biased": np.log(0.75) + 0.20 * (1.0 - 2.0 * xi),
        "tip-biased": np.log(1.25) + 0.20 * (2.0 * xi - 1.0),
    }


def _start_meta(value, name):
    """Normalize a start specification to (y0, kind, origin)."""
    if isinstance(value, dict):
        y0 = value["y"]
        kind = str(value.get("kind", "cold"))
        origin = str(value.get("origin", name))
    else:
        y0 = value
        kind = "cold"
        origin = name
    if kind not in ("cold", "continuation"):
        raise ValueError(f"unknown start_kind {kind!r}")
    return np.asarray(y0, dtype=float), kind, origin


def conditioning_estimate(y, g):
    """Practical 1-norm condition estimate of the N x N square branch Jacobian.

    This is the momentum+mass subsystem Jacobian, not the ``N+1`` full
    compatibility set that includes the absolute entropy residual, so the
    estimate describes the branch solve only.  ``||J||_1`` and ``||J^-1||_1``
    are estimated with deterministic
    ``scipy.sparse.linalg.onenormest`` calls (fixed seed, global RNG state
    restored).  The inverse operator applies a SuperLU factorization and its
    transpose solve.  A negative diagonal pivot ratio is reported as a
    rank-deficiency proxy.  No acceptance threshold is invented here: the
    estimate is evidence, and factorization/finiteness failure makes the
    numerical evidence incomplete.
    """
    result = {
        "method": "onenormest(J) * onenormest(J^-1); J^-1 via SuperLU splu "
                  "solve and transpose solve; fixed seed",
        "available": False,
        "reason": None,
        "factorization_available": False,
        "pivot_proxy": "min|U_ii| / max|U_ii|",
    }
    try:
        J = jacobian(y, g)
    except Exception as exc:  # pragma: no cover - defensive
        result["reason"] = f"Jacobian evaluation failed: {type(exc).__name__}: {exc}"
        return result
    N = int(J.shape[0])
    result["N"] = N
    result["nnz"] = int(J.nnz)
    if J.shape[0] != J.shape[1] or N == 0:
        result["reason"] = "Jacobian is not square"
        return result
    if not np.all(np.isfinite(J.data)):
        result["reason"] = "Jacobian has non-finite entries"
        return result
    try:
        lu = splu(J.tocsc(), permc_spec="COLAMD")
    except Exception as exc:
        result["reason"] = f"factorization failed: {type(exc).__name__}: {exc}"
        return result
    result["factorization_available"] = True
    diag = np.abs(lu.U.diagonal())
    if diag.size == 0 or not np.all(np.isfinite(diag)):
        result["reason"] = "non-finite LU diagonal"
        return result
    min_pivot = float(np.min(diag))
    max_pivot = float(np.max(diag))
    result["min_abs_pivot"] = min_pivot
    result["max_abs_pivot"] = max_pivot
    result["pivot_ratio"] = min_pivot / max_pivot if max_pivot > 0 else 0.0
    result["zero_pivots"] = int(np.sum(diag == 0.0))
    if min_pivot <= 0.0:
        result["reason"] = "singular/rank-deficient LU factorization"
        return result
    inverse = LinearOperator(
        (N, N),
        matvec=lambda b: lu.solve(b),
        rmatvec=lambda b: lu.solve(b, trans="T"),
        dtype=float,
    )
    state = np.random.get_state()
    try:
        np.random.seed(COND_SEED)
        norm_j = float(onenormest(aslinearoperator(J), t=2, itmax=5))
        norm_inv = float(onenormest(inverse, t=2, itmax=5))
    finally:
        np.random.set_state(state)
    result["norm_J_1_estimate"] = norm_j
    result["norm_Jinv_1_estimate"] = norm_inv
    estimate = norm_j * norm_inv
    result["condition_1norm_estimate"] = estimate
    if not (np.isfinite(norm_j) and np.isfinite(norm_inv) and np.isfinite(estimate)):
        result["reason"] = "non-finite norm estimate"
        return result
    if estimate <= 0:
        result["reason"] = "non-positive norm estimate"
        return result
    if N <= 64:  # independent dense cross-check on small systems
        result["dense_cond_1"] = float(np.linalg.cond(J.toarray(), 1))
    result["available"] = True
    return result


def solve_stage(st, n, law, R, phi, N, *, stage=None, bc="mean", starts=None,
                tol=None, max_nfev=400, psi=PSI):
    """Solve one stage on one mesh for every initial guess."""
    g = build_setup(st, n, law, R, phi, N, bc=bc, psi=psi)
    g["stage"] = stage
    st = dict(st)
    st["_stage"] = stage
    if starts is None:
        starts = initial_guesses(N)
    tol = dict(STD_TOL if tol is None else tol)
    runs = []
    for name, start in starts.items():
        y0, kind, origin = _start_meta(start, name)
        try:
            res = least_squares(
                residual_vector,
                y0,
                args=(g,),
                jac=jacobian,
                bounds=(-Y_BOUND, Y_BOUND),
                method="trf",
                x_scale="jac",
                tr_solver="lsmr",
                max_nfev=max_nfev,
                **tol,
            )
            y = np.asarray(res.x, dtype=float)
            try:
                m = metrics(g, y, st)
                crit = criteria_from_metrics(m)
            except InvalidState:
                m = {"valid": False}
                crit = {}
            residual_inf = float(np.max(np.abs(core.residual(y, g))))
            run = {
                "start": name,
                "start_kind": kind,
                "start_origin": origin,
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "nfev": int(res.nfev),
                "njev": int(getattr(res, "njev", -1)),
                "cost": float(res.cost),
                "optimality": float(res.optimality),
                "valid": bool(m.get("valid")),
                "equation_residual_inf": residual_inf,
                "y": y,
                "metrics": m,
                "criteria": crit,
            }
        except Exception as exc:  # solver/domain failure is data, not a crash
            run = {
                "start": name,
                "start_kind": kind,
                "start_origin": origin,
                "success": False,
                "status": -1,
                "message": f"{type(exc).__name__}: {exc}",
                "nfev": 0,
                "njev": -1,
                "cost": float("inf"),
                "optimality": float("inf"),
                "valid": False,
                "equation_residual_inf": float("inf"),
                "y": None,
                "metrics": {"valid": False},
                "criteria": {},
            }
        runs.append(run)

    # Prefer a numerically closed run so a near but unfinished iterate can never
    # become the best and contaminate downstream branch evidence.
    closed = [r for r in runs if _closed(r)]
    converged = [r for r in runs if r["success"] and r["valid"]]
    pool = closed or converged or [r for r in runs if r["valid"]]
    best_run = min(pool, key=lambda r: r["cost"]) if pool else None
    best = None
    if best_run is not None and best_run["valid"]:
        best = dict(best_run)
        best["criteria"] = criteria_from_metrics(best["metrics"])
    if best is not None and best.get("y") is not None:
        best_ca = g["ca_ref"] * np.exp(best["y"])
        best_scale = float(np.max(best_ca))
        for r in runs:
            if r.get("y") is not None and r["valid"]:
                run_ca = g["ca_ref"] * np.exp(r["y"])
                delta = float(np.max(np.abs(run_ca - best_ca)))
                rel = delta / best_scale
                r["delta_c2a_m_s"] = delta
                r["rel_from_best"] = rel
                r["same_as_best"] = bool(rel <= SAME_SOLUTION_RTOL)
            else:
                r["delta_c2a_m_s"] = float("nan")
                r["rel_from_best"] = float("nan")
                r["same_as_best"] = False
        for r in runs:
            r["is_best"] = bool(r is best_run)
        # Branch membership requires numerical closure of both run and best.
        best_closed = _closed(best)
        branch = _branch_runs(runs, best)
        best["solver_closed"] = bool(best_closed)
        best["branch_run_count"] = len(branch)
        best["cold_branch_count"] = sum(1 for r in branch if r["start_kind"] == "cold")
        best["branch_spread_max_m_s"] = max(
            (r["delta_c2a_m_s"] for r in branch), default=float("nan")
        )
        best["branch_spread_max_rel"] = max(
            (r["rel_from_best"] for r in branch), default=float("nan")
        )
        best["conditioning"] = conditioning_estimate(best["y"], g)
    return {"stage": stage, "law": law, "N": N, "bc": bc, "runs": runs,
            "best": best, "g": g}


def _resample_relative(y, size):
    field = np.exp(np.asarray(y, dtype=float))
    return np.interp(
        np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, field.size), field
    )


def branch_analysis(stage_result):
    """Independent-start, cross-mesh branch evidence for one stage.

    Confirmation, spread and inconsistency only use runs and bests that are
    numerically closed (``_closed``).  Geometrically near but unfinished
    iterates are reported separately and never count as branch members.
    """
    meshes = stage_result.get("meshes") or []
    confirmations = []
    inconsistencies = []
    valid_bests = []
    unfinished_bests = []
    unfinished_near = []
    for sol in meshes:
        best = sol.get("best")
        if best is None or not best.get("valid") or best.get("y") is None:
            continue
        best_closed = _closed(best)
        if not best_closed:
            unfinished_bests.append({
                "N": sol["N"],
                "equation_residual_inf": best.get("equation_residual_inf"),
            })
        # Near-branch but numerically unfinished runs, reported separately.
        for r in _unfinished_near(sol.get("runs", []), best):
            unfinished_near.append({
                "N": sol["N"], "start": r["start"],
                "start_kind": r.get("start_kind", "cold"),
                "equation_residual_inf": r.get("equation_residual_inf"),
                "entropy_max_abs": r.get("metrics", {}).get("entropy_max_abs"),
            })
        if not best_closed:
            continue
        valid_bests.append(sol)
        cold_on_branch = [
            r["start"] for r in sol["runs"]
            if _closed(r) and r.get("same_as_best") and r.get("start_kind") == "cold"
        ]
        confirmations.append({
            "N": sol["N"],
            "cold_on_branch": cold_on_branch,
            "cold_on_branch_count": len(cold_on_branch),
        })
        for r in sol["runs"]:
            if not (_closed(r) and r.get("same_as_best") and r["start_kind"] == "cold"):
                continue
            if r.get("criteria") != best.get("criteria"):
                differing = sorted(
                    k for k in set(r["criteria"]) | set(best["criteria"])
                    if r["criteria"].get(k) != best["criteria"].get(k)
                )
                inconsistencies.append({
                    "N": sol["N"], "start": r["start"], "criteria": differing,
                })

    continuity = []
    continuity_ok = False
    first_to_finest_rel = None
    if valid_bests:
        size = int(np.asarray(valid_bests[-1]["best"]["y"]).size)
        finest_field = _resample_relative(valid_bests[-1]["best"]["y"], size)
        finest_scale = float(np.max(finest_field))
        for coarse, fine in zip(valid_bests[:-1], valid_bests[1:]):
            a = _resample_relative(coarse["best"]["y"], size)
            b = _resample_relative(fine["best"]["y"], size)
            rel = float(np.max(np.abs(a - b)) / max(float(np.max(b)), 1e-30))
            continuity.append({
                "N_coarse": coarse["N"], "N_fine": fine["N"],
                "max_relative_c2a_difference": rel,
                "within_same_solution_tolerance": bool(rel <= SAME_SOLUTION_RTOL),
            })
        first_field = _resample_relative(valid_bests[0]["best"]["y"], size)
        first_to_finest_rel = float(
            np.max(np.abs(first_field - finest_field)) / max(finest_scale, 1e-30)
        )
        continuity_ok = bool(
            len(valid_bests) == len(meshes)
            and continuity
            and all(c["within_same_solution_tolerance"] for c in continuity)
            and first_to_finest_rel <= SAME_SOLUTION_RTOL
        )

    confirmed_meshes = [c["N"] for c in confirmations if c["cold_on_branch_count"] >= 2]
    return {
        "confirmed": bool(confirmed_meshes and continuity_ok),
        "confirmed_meshes": confirmed_meshes,
        "confirmations": confirmations,
        "continuity": continuity,
        "continuity_ok": continuity_ok,
        "first_to_finest_relative_c2a_difference": first_to_finest_rel,
        "inconsistent": inconsistencies,
        "unfinished_bests": unfinished_bests,
        "unfinished_near_branch": unfinished_near,
        "solver_closure_tol": SOLVER_CLOSURE_TOL,
        "same_solution_rtol": SAME_SOLUTION_RTOL,
    }


def _raw_value(run, key):
    """Raw scalar behind one criterion, including the combined R-form key."""
    m = run["metrics"]
    if key == "mass":
        return m["mass_abs"]
    if key == "w2_gt_ca":
        return max(0.0, -m["min_w2_minus_ca"])
    if key == "alpha2":
        return m["alpha2_violation"]
    if key == "positive_finite":
        return 0.0 if m["positive_finite"] else 1.0
    if key == "smooth":
        return m["smoothness"]
    if key == "work":
        return m["work_res"]
    if key == "r_forms":
        return max(m["delta_T"], m["delta_GRE"])
    if key == "entropy_anchor":
        return m["entropy_max_abs"]
    return m[key]


def _margin_reason(key, value):
    """Return an INCONCLUSIVE reason when a failure is too close to its cap."""
    if key == "epsilon_RRE" and value < 1.25 * EPS_RRE_TOL:
        return "epsilon_RRE failure is marginal"
    if key == "mass" and value < 1.25 * MASS_TOL:
        return "mass failure is marginal"
    if key == "r_forms" and value < 1.25 * R_FORM_TOL:
        return "r_forms failure is marginal"
    if key == "work" and value < 1.25 * WORK_TOL:
        return "work failure is marginal"
    if key == "smooth" and value < 1.25 * SMOOTH_TOL:
        return "smoothness failure is marginal"
    if key == "alpha2" and value < 0.5:
        return "alpha2 failure is below 0.5 deg"
    if key == "w2_gt_ca" and value < 1.0:
        return "w2-c2a failure is below 1 m/s"
    if key == "entropy_anchor" and value < 1.25 * ABS_ENTROPY_TOL:
        return "entropy_anchor failure is marginal"
    return None


def classify(stage_result):
    """Return (YES|NO|INCONCLUSIVE, reasons) under the corrected evidence rules.

    Warm/continuation starts never count as independent confirmation.  A failed
    solve is never evidence of non-existence.  Solver ``success`` alone is
    neither necessary nor sufficient: a solved branch member requires
    ``valid`` and ``equation_residual_inf <= SOLVER_CLOSURE_TOL``.
    """
    meshes = stage_result.get("meshes") or []
    if not meshes:
        return "INCONCLUSIVE", ["no mesh results"]
    finest = meshes[-1]
    best = finest.get("best")
    if best is None or not best.get("valid"):
        return "INCONCLUSIVE", ["no valid solution at the finest mesh"]
    if not _closed(best):
        return "INCONCLUSIVE", [
            "finest-mesh best is not numerically closed "
            f"(equation_residual_inf > {SOLVER_CLOSURE_TOL:g})"
        ]
    failed = sorted(k for k, ok in best["criteria"].items() if not ok)
    analysis = branch_analysis(stage_result)
    cold_confirmed = analysis["confirmed"]
    continuity_ok = analysis["continuity_ok"]
    conditioning = (finest.get("best") or {}).get("conditioning") or {}
    conditioning_ok = bool(conditioning.get("available"))

    if not failed:
        reasons = []
        if not cold_confirmed:
            reasons.append(
                "fewer than two independent cold starts reached the finest branch "
                "on any mesh with continuous refinement"
            )
        if not conditioning_ok:
            reasons.append("conditioning evidence missing/incomplete at the finest mesh")
        if analysis["inconsistent"]:
            reasons.append("cold starts on the branch disagree on acceptance criteria")
        if reasons:
            return "INCONCLUSIVE", reasons
        return "YES", []

    if "positive_finite" in failed:
        return "INCONCLUSIVE", ["non-finite/negative fields indicate solver failure"]
    if not cold_confirmed:
        return "INCONCLUSIVE", [
            "fewer than two independent cold starts reached the finest branch "
            "on any mesh with continuous refinement"
        ]
    if not continuity_ok:
        return "INCONCLUSIVE", ["branch is not continuously refined to the finest mesh"]
    if not conditioning_ok:
        return "INCONCLUSIVE", ["conditioning evidence missing/incomplete at the finest mesh"]
    if analysis["inconsistent"]:
        return "INCONCLUSIVE", ["cold starts on the branch disagree on acceptance criteria"]

    # NO requires numerically closed branch evidence on every mesh.
    if len(analysis["unfinished_bests"]):
        return "INCONCLUSIVE", [
            "branch best is not numerically closed on every mesh: "
            + ", ".join(str(u["N"]) for u in analysis["unfinished_bests"])
        ]
    mesh_bests = [
        sol["best"] for sol in meshes
        if sol.get("best") is not None and _closed(sol["best"])
    ]
    if not mesh_bests:
        return "INCONCLUSIVE", ["no numerically closed mesh solution"]
    for key in failed:
        if any(b["criteria"].get(key, True) for b in mesh_bests):
            return "INCONCLUSIVE", [f"{key} is not a failure on every mesh"]
        hub = stage_result.get("bc_hub") or {}
        hub_best = hub.get("best")
        if hub_best is None or not hub_best.get("valid"):
            return "INCONCLUSIVE", [f"{key} robustness has no valid alternative-BC solution"]
        if not _closed(hub_best):
            return "INCONCLUSIVE", [
                f"{key} robustness alternative-BC best is not numerically closed"
            ]
        if key not in hub_best["criteria"] or hub_best["criteria"][key]:
            return "INCONCLUSIVE", [f"{key} failure is not robust to the alternative pressure BC"]
        value = _raw_value(best, key)
        reason = _margin_reason(key, value)
        if reason:
            return "INCONCLUSIVE", [reason]

    for key in failed:
        coarse_value = _raw_value(mesh_bests[0], key)
        fine_value = _raw_value(best, key)
        if fine_value <= 0.8 * coarse_value:
            return "INCONCLUSIVE", [
                f"{key} is still improving with mesh ({coarse_value:.4g} -> {fine_value:.4g})"
            ]
    return "NO", ["failure is common to the branch across mesh and BC: " + ", ".join(failed)]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def control_point():
    """Resolve the specified control point, including alpha1_stage1 = 13.10 deg."""
    import optimize_extended as oe
    import optimizer_core as oc
    import radial_check as rc
    import scan5d as s5

    m = oc.load_model()
    ns, _ = oe.build_ns(m, 0.995, 0.995**2 * oe.SIGMA_LIM)
    R, phi = ns["R"], ns["φ"]
    x = s5.base_point()
    x[0], x[1], x[3], x[2], x[8], x[9] = 90000, 0.062, 0.8125, 0.70, 1.04375, 1.25
    x[4] = 11.0
    for _ in range(60):
        oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
        _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
        if rc.profile(st[0], x[0], "Γ", R, phi)["корень"]["a1"] >= 11.15:
            break
        x[4] = min(20.0, x[4] + 0.05)
    oe.eval_candidate(x, ns, "B87_relaxed", 0.995)
    _, st = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    a11 = float(st[0]["α1"])
    if abs(a11 - 13.10) > 0.01:
        raise RuntimeError(f"control point alpha1_stage1={a11:.4f} deg, expected 13.10")
    return dict(st=st, n=float(x[0]), R=float(R), phi=float(phi), x=x, alpha1_stage1=a11)


def provenance():
    """Solver/library provenance for the generated evidence."""
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def run_evidence(meshes=MESHES, stages=STAGES):
    cp = control_point()
    st, n, R, phi = cp["st"], cp["n"], cp["R"], cp["phi"]
    out = {"control_point": {"n": n, "D_hub": 0.062, "w1": 0.70, "w2": 0.8125,
                             "kappa2": 1.04375, "kappa3": 1.25,
                             "alpha1_stage1": cp["alpha1_stage1"],
                             "R": R, "phi": phi, "laws": [LAWS[i] for i in stages]},
           "meshes": list(meshes), "stages": [],
           "provenance": provenance()}
    for stage in stages:
        law = LAWS[stage]
        stage_res = {"stage": stage, "law": law, "meshes": []}
        for N in meshes:
            starts = initial_guesses(N)
            if stage_res["meshes"]:
                prev = stage_res["meshes"][-1]["best"]
                if prev is not None and prev.get("y") is not None:
                    # A warm mesh-continuation start is not an independent guess.
                    starts["warm"] = {
                        "y": np.interp(
                            np.linspace(0.0, 1.0, N),
                            np.linspace(0.0, 1.0, prev["y"].size),
                            prev["y"],
                        ),
                        "kind": "continuation",
                        "origin": f"N={stage_res['meshes'][-1]['N']}",
                    }
            stage_res["meshes"].append(
                solve_stage(st[stage - 1], n, law, R, phi, N, stage=stage,
                            bc="mean", starts=starts)
            )
        # Tolerance sensitivity on the finest mesh, initialized from the branch
        # so it measures tolerance effects rather than the start basin.
        tolN = meshes[-1]
        finest_best = stage_res["meshes"][-1]["best"]
        branch_start = (
            {"y": np.asarray(finest_best["y"], dtype=float),
             "kind": "continuation", "origin": f"N={tolN} best"}
            if finest_best is not None and finest_best.get("y") is not None else None
        )
        tol_starts = {"branch": branch_start} if branch_start else None
        standard = solve_stage(st[stage - 1], n, law, R, phi, tolN, stage=stage,
                               bc="mean", tol=STD_TOL, starts=tol_starts)
        loose = solve_stage(st[stage - 1], n, law, R, phi, tolN, stage=stage,
                            bc="mean", tol=LOOSE_TOL, starts=tol_starts)
        stage_res["tolerance"] = {"N": tolN, "standard": standard, "loose": loose}
        sb, lb = standard.get("best"), loose.get("best")
        if sb and lb and sb.get("valid") and lb.get("valid"):
            sb_ca = standard["g"]["ca_ref"] * np.exp(sb["y"])
            lb_ca = loose["g"]["ca_ref"] * np.exp(lb["y"])
            stage_res["tolerance"]["field_difference_c2a_m_s"] = float(
                np.max(np.abs(sb_ca - lb_ca))
            )
            stage_res["tolerance"]["field_difference_c2a_relative"] = float(
                np.max(np.abs(sb_ca - lb_ca)) / np.max(sb_ca)
            )
        else:
            stage_res["tolerance"]["field_difference_c2a_m_s"] = None
            stage_res["tolerance"]["field_difference_c2a_relative"] = None
        stage_res["bc_hub"] = solve_stage(st[stage - 1], n, law, R, phi, meshes[-1],
                                          stage=stage, bc="hub")
        stage_res["analysis"] = branch_analysis(stage_res)
        stage_res["classification"], stage_res["classification_reasons"] = classify(stage_res)
        out["stages"].append(stage_res)
    return out


def _kb(x):
    return f"{x / 1e3:.2f}"


def _fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float) and not math.isfinite(x):
        return "—"
    return f"{x:.{nd}g}" if isinstance(x, float) else str(x)


def _run_row(stage, law, sol, run):
    m = run.get("metrics", {})
    is_best = bool(run.get("is_best", False))
    cond = (sol.get("best") or {}).get("conditioning", {}) if is_best else {}
    return {
        "stage": stage,
        "law": law,
        "N": sol["N"],
        "bc": sol["bc"],
        "start": run["start"],
        "start_kind": run.get("start_kind", "cold"),
        "start_origin": run.get("start_origin", run["start"]),
        "success": run["success"],
        "status": run["status"],
        "nfev": run["nfev"],
        "cost": run["cost"],
        "optimality": run["optimality"],
        "valid": run["valid"],
        "solver_closed": _closed(run),
        "equation_residual_inf": run.get("equation_residual_inf"),
        "is_best": bool(is_best),
        "epsilon_RRE": m.get("epsilon_RRE"),
        "mass_rel": m.get("mass_rel"),
        "energy_res": m.get("energy_res"),
        "work_res": m.get("work_res"),
        "delta_T": m.get("delta_T"),
        "delta_GRE": m.get("delta_GRE"),
        "r_forms_value": max(m["delta_T"], m["delta_GRE"]) if "delta_T" in m else None,
        "smoothness": m.get("smoothness"),
        "min_w2_minus_ca": m.get("min_w2_minus_ca"),
        "alpha2_min": m.get("alpha2_min"),
        "alpha2_max": m.get("alpha2_max"),
        "alpha2_violation": m.get("alpha2_violation"),
        "lambda": m.get("lambda"),
        "p_min": m.get("p_min"),
        "p_max": m.get("p_max"),
        "T_min": m.get("T_min"),
        "T_max": m.get("T_max"),
        "rho_min": m.get("rho_min"),
        "rho_max": m.get("rho_max"),
        "cu_min": m.get("cu_min"),
        "cu_max": m.get("cu_max"),
        "ca_min": m.get("ca_min"),
        "ca_max": m.get("ca_max"),
        "dpdr_sign_changes": m.get("dpdr_sign_changes"),
        "entropy_min": m.get("entropy_min"),
        "entropy_max": m.get("entropy_max"),
        "entropy_mean": m.get("entropy_mean"),
        "entropy_max_abs": m.get("entropy_max_abs"),
        "entropy_span": m.get("entropy_span"),
        "entropy_anchor_value": m.get("entropy_anchor_value"),
        "entropy_anchor": run.get("criteria", {}).get("entropy_anchor"),
        "delta_c2a_m_s": run.get("delta_c2a_m_s"),
        "rel_from_best": run.get("rel_from_best"),
        "same_as_best": run.get("same_as_best"),
        "conditioning_available": cond.get("available") if is_best else None,
        "condition_1norm_estimate": cond.get("condition_1norm_estimate") if is_best else None,
        "pivot_ratio": cond.get("pivot_ratio") if is_best else None,
        "tol_case": "",
    }


def write_report(out, directory=RESULTS):
    directory.mkdir(parents=True, exist_ok=True)
    md = directory / "stage6b45b_report.md"
    js = directory / "stage6b45b_results.json"
    csv_path = directory / "stage6b45b_runs.csv"
    summary_csv = directory / "stage6b45b_stage_summary.csv"

    rows = []
    for stage_res in out["stages"]:
        for sol in stage_res["meshes"]:
            for run in sol["runs"]:
                rows.append(_run_row(stage_res["stage"], stage_res["law"], sol, run))
        for label in ("standard", "loose"):
            sol = stage_res["tolerance"][label]
            for run in sol["runs"]:
                row = _run_row(stage_res["stage"], stage_res["law"], sol, run)
                row["tol_case"] = label
                rows.append(row)
        for run in stage_res["bc_hub"]["runs"]:
            row = _run_row(stage_res["stage"], stage_res["law"], stage_res["bc_hub"], run)
            row["bc"] = "hub"
            rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0])
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary_rows = []
    for stage_res in out["stages"]:
        best = stage_res["meshes"][-1]["best"]
        m = best["metrics"] if best and best.get("valid") else {}
        cond = (best or {}).get("conditioning", {})
        analysis = stage_res.get("analysis", {})
        summary_rows.append({
            "stage": stage_res["stage"],
            "law": stage_res["law"],
            "classification": stage_res["classification"],
            "N_finest": stage_res["meshes"][-1]["N"],
            "independent_cold_confirmed": analysis.get("confirmed"),
            "cold_confirmation_meshes": ";".join(str(n) for n in analysis.get("confirmed_meshes", [])),
            "branch_continuous": analysis.get("continuity_ok"),
            "epsilon_RRE": m.get("epsilon_RRE"),
            "mass_rel": m.get("mass_rel"),
            "energy_res": m.get("energy_res"),
            "work_res": m.get("work_res"),
            "delta_T": m.get("delta_T"),
            "delta_GRE": m.get("delta_GRE"),
            "smoothness": m.get("smoothness"),
            "min_w2_minus_ca": m.get("min_w2_minus_ca"),
            "alpha2_min": m.get("alpha2_min"),
            "alpha2_max": m.get("alpha2_max"),
            "lambda": m.get("lambda"),
            "dpdr_sign_changes": m.get("dpdr_sign_changes"),
            "entropy_min": m.get("entropy_min"),
            "entropy_max": m.get("entropy_max"),
            "entropy_mean": m.get("entropy_mean"),
            "entropy_max_abs": m.get("entropy_max_abs"),
            "entropy_span": m.get("entropy_span"),
            "entropy_anchor_value": m.get("entropy_anchor_value"),
            "entropy_anchor_pass": (best or {}).get("criteria", {}).get("entropy_anchor"),
            "equation_residual_inf": (best or {}).get("equation_residual_inf"),
            "solver_closed": _closed(best) if best else False,
            "conditioning_available": cond.get("available"),
            "condition_1norm_estimate": cond.get("condition_1norm_estimate"),
            "pivot_ratio": cond.get("pivot_ratio"),
        })
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    lines = []
    lines.append("# Stage 6B-4.5-B — coupled rotor-exit closure\n")
    lines.append("Engineering model: **EXPERIMENTAL**. A numerical pass does not promote "
                 "the loss closure or the equal-span mapping to accepted physics.\n")
    cp = out["control_point"]
    lines.append("## Control point\n")
    lines.append(f"n = {cp['n']:.0f} rpm; D_hub = {cp['D_hub'] * 1e3:.0f} mm; "
                 f"w1 = {cp['w1']}; w2 = {cp['w2']}; kappa2 = {cp['kappa2']}; "
                 f"kappa3 = {cp['kappa3']}; alpha1_stage1 = {cp['alpha1_stage1']:.2f} deg.\n")
    lines.append(f"Laws: {', '.join(f'stage {i}: {LAWS[i]}' for i in STAGES)}.\n")
    prov = out.get("provenance", {})
    lines.append("## Provenance\n")
    lines.append(f"Python {prov.get('python', '—')}; NumPy {prov.get('numpy', '—')}; "
                 f"SciPy {prov.get('scipy', '—')}; platform {prov.get('platform', '—')}.\n")
    lines.append("## Equation count\n")
    lines.append("Unknowns: N axial log velocities `y_i = ln(c2a_i/c2a_ref)`.\n")
    lines.append("The square branch solve uses N-1 finite-volume radial-momentum "
                 "intervals + 1 integral mass equation = N equations. The `p2` "
                 "construction enforces N-1 relative entropy/state equations plus the "
                 "prescribed pressure boundary condition. The complete required set is "
                 "therefore N-1 momentum + 1 mass + 1 absolute cross-rotor entropy "
                 "condition = N+1 conditions on N unknowns. The absolute entropy "
                 "condition is a compatibility residual of the prescribed control point; "
                 "there is no physically allowed extra global unknown. `residual` and its "
                 "analytic Jacobian are the square momentum+mass subsystem only, so their "
                 "conditioning does not describe the N+1 compatibility set. Loss/state and "
                 "the pressure BC are not claimed to be independently satisfied.\n")
    lines.append("## Acceptance thresholds\n")
    lines.append(f"epsilon_RRE <= {EPS_RRE_TOL:.3g}; \\|mass\\| <= {MASS_TOL:.3g}; "
                 f"work <= {WORK_TOL:.3g}; R-form difference <= {R_FORM_TOL:.3g} "
                 "(numerical finite-difference consistency cap, not a physics tolerance); "
                 f"smoothness <= {SMOOTH_TOL:.3g}; p,T,rho positive finite; "
                 "w2>c2a; alpha2 70..90 deg stages 1-2 and 80..90 deg stage 3; "
                 f"entropy anchor max\\|E_s\\| <= {ABS_ENTROPY_TOL:.1e} J/(kg K) "
                 "(strict algebraic numerical closure, not model uncertainty).\n")
    lines.append(f"Solved branch member: valid and `max\\|core.residual\\| <= "
                 f"{SOLVER_CLOSURE_TOL:.0e}` (dimensionless). Solver success alone is "
                 "neither necessary nor sufficient; unfinished iterates are not branch "
                 "confirmations or contradictions.\n")
    lines.append("Classification rules: **YES** requires every finest-mesh criterion "
                 "(including the absolute entropy anchor) to pass, at least two genuinely "
                 "independent cold starts on the same numerically closed branch on some "
                 "mesh, a branch continuously refined to the finest mesh with every best "
                 "numerically closed, conditioning evidence, and no branch inconsistency. "
                 "**NO** additionally requires the same failure on every mesh using "
                 "numerically closed bests, robustness to the alternative pressure BC "
                 "with a numerically closed alternative best, a clear margin and no "
                 "continuing improvement with mesh. Warm/continuation starts are never "
                 "counted as independent. A failed solver is never evidence of "
                 "non-existence. Anything else is **INCONCLUSIVE**. "
                 f"Starts within {SAME_SOLUTION_RTOL:.0%} relative c2a of the best are "
                 "treated as geometrically close for reporting; branch membership still "
                 "requires numerical closure.\n")

    for stage_res in out["stages"]:
        lines.append(f"## Stage {stage_res['stage']} ({stage_res['law']})\n")
        lines.append(f"**Classification: {stage_res['classification']}**")
        if stage_res["classification_reasons"]:
            lines.append(" — " + "; ".join(stage_res["classification_reasons"]))
        lines.append("\n")
        finest = stage_res["meshes"][-1]
        best = finest["best"]
        if best is None or not best.get("valid"):
            lines.append("No valid finest-mesh solution.\n")
            continue
        m = best["metrics"]
        lines.append("| criterion | value | threshold | pass |")
        lines.append("|---|---|---|---|")
        crit = best["criteria"]
        lines.append(f"| epsilon_RRE | {m['epsilon_RRE']:.4%} | <= {EPS_RRE_TOL:.2%} | {_fmt(crit['epsilon_RRE'])} |")
        lines.append(f"| \\|mass\\| | {m['mass_abs']:.4%} | <= {MASS_TOL:.2%} | {_fmt(crit['mass'])} |")
        lines.append(f"| energy residual | {m['energy_res']:.3e} | algebraic | {_fmt(m['energy_res'] <= 1e-12)} |")
        lines.append(f"| work residual | {m['work_res']:.3e} | <= {WORK_TOL:.0e} | {_fmt(crit['work'])} |")
        lines.append(f"| max\\|R_T-R_p\\|/scale | {m['delta_T']:.3e} | <= {R_FORM_TOL:.1e} | {_fmt(m['delta_T'] <= R_FORM_TOL)} |")
        lines.append(f"| max\\|R_GRE-R_p\\|/scale | {m['delta_GRE']:.3e} | <= {R_FORM_TOL:.1e} | {_fmt(m['delta_GRE'] <= R_FORM_TOL)} |")
        lines.append(f"| positive finite | {_fmt(m['positive_finite'])} | yes | {_fmt(crit['positive_finite'])} |")
        lines.append(f"| smoothness | {m['smoothness']:.3g} | <= {SMOOTH_TOL:.3g} | {_fmt(crit['smooth'])} |")
        lines.append(f"| min(w2-c2a) | {m['min_w2_minus_ca']:.3f} m/s | > 0 | {_fmt(crit['w2_gt_ca'])} |")
        lines.append(f"| alpha2 range | {m['alpha2_min']:.2f}..{m['alpha2_max']:.2f} deg | 70..90 / 80..90 | {_fmt(crit['alpha2'])} |")
        lines.append(f"| entropy anchor max\\|E_s\\| | {m['entropy_max_abs']:.3e} J/(kg K) | <= {ABS_ENTROPY_TOL:.1e} | {_fmt(crit['entropy_anchor'])} |")
        lines.append(f"| equation residual max\\|f\\| | {best.get('equation_residual_inf', float('nan')):.3e} | <= {SOLVER_CLOSURE_TOL:.0e} | {_fmt(_closed(best))} |\n")
        lines.append(f"R-form numerical consistency: achieved max\\|R_T-R_p\\|/scale = "
                     f"{m['delta_T']:.3e} and max\\|R_GRE-R_p\\|/scale = {m['delta_GRE']:.3e}, "
                     f"versus the numerical-consistency cap {R_FORM_TOL:.1e}. These are "
                     "finite-difference discretization differences, not percent-level "
                     "physical discrepancies.\n")
        lines.append(f"Ranges: p {_kb(m['p_min'])}..{_kb(m['p_max'])} kPa; "
                     f"T {m['T_min']:.1f}..{m['T_max']:.1f} K; "
                     f"rho {m['rho_min']:.4f}..{m['rho_max']:.4f} kg/m3; "
                     f"c2u {m['cu_min']:.2f}..{m['cu_max']:.2f} m/s; "
                     f"c2a {m['ca_min']:.1f}..{m['ca_max']:.1f} m/s; "
                     f"alpha2 {m['alpha2_min']:.2f}..{m['alpha2_max']:.2f} deg. "
                     f"lambda_a = {m['lambda']:.4f}; dp/dr sign changes = "
                     f"{m['dpdr_sign_changes']}.\n")
        lines.append(f"Independent scales: max\\|(1/rho)dp/dr\\| = {m['dpdr_scale']:.3e} m/s2; "
                     f"max\\|R_p\\| = {m['R_p_max']:.3e}; max\\|R_T\\| = {m['R_T_max']:.3e}; "
                     f"max\\|R_GRE\\| = {m['R_GRE_max']:.3e} m/s2.\n")

        lines.append("### Absolute cross-rotor entropy\n")
        lines.append("| quantity | value [J/(kg K)] |")
        lines.append("|---|---|")
        lines.append(f"| min E_s | {m['entropy_min']:.6e} |")
        lines.append(f"| max E_s | {m['entropy_max']:.6e} |")
        lines.append(f"| mean E_s | {m['entropy_mean']:.6e} |")
        lines.append(f"| max\\|E_s\\| | {m['entropy_max_abs']:.6e} |")
        lines.append(f"| span (max-min) | {m['entropy_span']:.6e} |")
        lines.append(f"| anchor E_s | {m['entropy_anchor_value']:.6e} |")
        lines.append("")
        lines.append("`E_s = cp ln(T2/T1) - R ln(p2/p1) - Delta_s_l` uses absolute inlet "
                     "and exit states. The relative `p2` construction makes `E_s` nearly "
                     "constant across the span, so the derivative equivalence "
                     "`R_p = R_T = R_GRE` cannot detect a constant `E_s` offset. Only the "
                     "anchor/max absolute value tests that offset.\n")

        analysis = stage_res.get("analysis", {})
        lines.append("### Independent cold-start evidence\n")
        lines.append(f"Independent cold-start confirmation: **{_fmt(analysis.get('confirmed'))}**; "
                     f"confirming meshes: "
                     f"{', '.join(str(n) for n in analysis.get('confirmed_meshes', [])) or 'none'}.\n")
        lines.append("| N | cold starts on best branch | names |")
        lines.append("|---|---|---|")
        for conf in analysis.get("confirmations", []):
            lines.append(f"| {conf['N']} | {conf['cold_on_branch_count']} | "
                         f"{', '.join(conf['cold_on_branch']) or '—'} |")
        lines.append("")
        lines.append("| N coarse | N fine | max relative c2a difference | same branch |")
        lines.append("|---|---|---|---|")
        for cont in analysis.get("continuity", []):
            lines.append(f"| {cont['N_coarse']} | {cont['N_fine']} | "
                         f"{cont['max_relative_c2a_difference']:.3e} | "
                         f"{_fmt(cont['within_same_solution_tolerance'])} |")
        lines.append("")
        first_to_finest = analysis.get("first_to_finest_relative_c2a_difference")
        lines.append(f"First-to-finest relative c2a difference: "
                     f"{_fmt(first_to_finest, 3)} (limit {SAME_SOLUTION_RTOL:.0%}).\n")
        if analysis.get("inconsistent"):
            lines.append("Branch inconsistencies: " + "; ".join(
                f"N={i['N']} {i['start']} ({', '.join(i['criteria'])})"
                for i in analysis["inconsistent"]) + ".\n")
        else:
            lines.append("No branch inconsistency among cold starts on the branch.\n")

        lines.append("### Mesh and start evidence\n")
        lines.append("| N | start | kind | success | nfev | cost | optimality | epsilon_RRE | mass | lambda | delta c2a [m/s] | relative | same | eq res max\\|f\\| | solver closed | entropy max\\|E_s\\| |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for sol in stage_res["meshes"]:
            for run in sol["runs"]:
                mm = run.get("metrics", {})
                if run["valid"]:
                    d = run.get("delta_c2a_m_s")
                    rel = run.get("rel_from_best")
                    dtxt = f"{d:.4g}" if d is not None and math.isfinite(d) else "—"
                    rtxt = f"{rel:.3e}" if rel is not None and math.isfinite(rel) else "—"
                    lines.append(
                        f"| {sol['N']} | {run['start']} | {run.get('start_kind', 'cold')} | "
                        f"{_fmt(run['success'])} | {run['nfev']} | "
                        f"{run['cost']:.3e} | {run['optimality']:.3e} | "
                        f"{mm['epsilon_RRE']:.4%} | {mm['mass_abs']:.4%} | {mm['lambda']:.4f} | "
                        f"{dtxt} | {rtxt} | {_fmt(run.get('same_as_best'))} | "
                        f"{_fmt(run.get('equation_residual_inf'), 3)} | {_fmt(_closed(run))} | "
                        f"{_fmt(mm.get('entropy_max_abs'), 3)} |"
                    )
                else:
                    lines.append(f"| {sol['N']} | {run['start']} | {run.get('start_kind', 'cold')} | "
                                 f"{_fmt(run['success'])} | {run['nfev']} | — | — | invalid | — | — | — | — | no | — | no | — |")
        lines.append("")
        branch_runs = _branch_runs(finest["runs"], best)
        non_branch = [r for r in finest["runs"] if r.get("valid") and not r.get("same_as_best")]
        lines.append(f"Best finest branch spread: max c2a difference = "
                     f"{_fmt(best.get('branch_spread_max_m_s'), 4)} m/s "
                     f"({_fmt(best.get('branch_spread_max_rel'), 3)} relative) "
                     f"over {len(branch_runs)} numerically closed branch runs. "
                     "Warm/continuation starts are excluded from independent "
                     "confirmation.\n")
        if non_branch:
            lines.append("Geometrically distant endpoints at the finest mesh: "
                         + "; ".join(
                             f"{r['start']} ({r.get('start_kind', 'cold')}): "
                             f"{r['delta_c2a_m_s']:.4g} m/s, {r['rel_from_best']:.3e} relative, "
                             f"numerically closed: {_fmt(_closed(r))}"
                             for r in non_branch)
                         + ". Distance alone does not establish distinct solved "
                         "branches. Unclosed endpoints are unfinished, not evidence of "
                         "non-existence.\n")

        near = analysis.get("unfinished_near_branch", [])
        lines.append("### Geometrically near but numerically unfinished runs\n")
        if near:
            lines.append("| N | start | kind | eq res max\\|f\\| | entropy max\\|E_s\\| |")
            lines.append("|---|---|---|---|---|")
            for item in near:
                lines.append(f"| {item['N']} | {item['start']} | "
                             f"{item.get('start_kind', 'cold')} | "
                             f"{_fmt(item.get('equation_residual_inf'), 3)} | "
                             f"{_fmt(item.get('entropy_max_abs'), 3)} |")
            lines.append("")
            lines.append("These iterates are geometrically close to the best branch but do "
                         "not satisfy the solver-closure cap, so they are not confirmations, "
                         "branch spread, continuity, inconsistency, or non-existence "
                         "evidence.\n")
        else:
            lines.append("None reported at the analysed meshes.\n")

        lines.append("### Conditioning evidence\n")
        lines.append("Method: deterministic `onenormest(J) * onenormest(J^-1)` with fixed "
                     "seed; `J^-1` applies a SuperLU `splu` factorization and its transpose "
                     "solve. `J` is the N x N square momentum+mass subsystem Jacobian, not "
                     "the N+1 full compatibility set (which adds the absolute entropy "
                     "residual), so the estimate describes the branch solve only. Rank proxy "
                     "is `min|U_ii| / max|U_ii|`. No pass threshold is invented; "
                     "factorization/finiteness failure would make the evidence incomplete.\n")
        lines.append("| N | factorization | cond_1 estimate | pivot ratio | \\|J\\|_1 est. | \\|J^-1\\|_1 est. | dense cond_1 | available |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for sol in stage_res["meshes"]:
            c = (sol.get("best") or {}).get("conditioning", {})
            lines.append(
                f"| {sol['N']} | {_fmt(c.get('factorization_available'))} | "
                f"{_fmt(c.get('condition_1norm_estimate'), 4)} | "
                f"{_fmt(c.get('pivot_ratio'), 4)} | "
                f"{_fmt(c.get('norm_J_1_estimate'), 4)} | "
                f"{_fmt(c.get('norm_Jinv_1_estimate'), 4)} | "
                f"{_fmt(c.get('dense_cond_1'), 4)} | {_fmt(c.get('available'))} |"
            )
        cond_best = best.get("conditioning", {})
        if cond_best.get("reason"):
            lines.append(f"\nConditioning note: {cond_best['reason']}.\n")
        else:
            lines.append("")
        lines.append(f"Finest-mesh conditioning available: **{_fmt(cond_best.get('available'))}**.\n")

        tol = stage_res["tolerance"]
        lines.append(f"### Tolerance sensitivity (N={tol['N']}, initialized from the found branch)\n")
        lines.append("| case | success | optimality | epsilon_RRE | mass | R-form max | lambda |")
        lines.append("|---|---|---|---|---|---|---|")
        for label in ("standard", "loose"):
            sol = tol[label]
            b = sol.get("best")
            if b and b.get("valid"):
                mm = b["metrics"]
                lines.append(
                    f"| {label} | {_fmt(b['success'])} | {b['optimality']:.3e} | "
                    f"{mm['epsilon_RRE']:.4%} | {mm['mass_abs']:.4%} | "
                    f"{max(mm['delta_T'], mm['delta_GRE']):.3e} | {mm['lambda']:.4f} |"
                )
            else:
                lines.append(f"| {label} | no valid solution | — | — | — | — | — |")
        lines.append("")
        fd = tol.get("field_difference_c2a_m_s")
        fdr = tol.get("field_difference_c2a_relative")
        lines.append(f"Standard vs loose field difference: "
                     f"{_fmt(fd, 4)} m/s ({_fmt(fdr, 3)} relative).\n")

        hub = stage_res["bc_hub"]
        hub_best = hub.get("best")
        lines.append("### BC sensitivity (finest mesh, hub anchor vs primary mean anchor)\n")
        if hub_best and hub_best.get("valid"):
            hm = hub_best["metrics"]
            lines.append(f"- primary mean: epsilon_RRE={m['epsilon_RRE']:.4%}, |mass|={m['mass_abs']:.4%}, "
                         f"p_hub={_kb(m['p_hub'])} kPa, rho_hub={m['rho_hub']:.4f}, "
                         f"alpha2 {m['alpha_hub']:.2f}/{m['alpha_tip']:.2f} deg, lambda={m['lambda']:.4f}")
            lines.append(f"- alternative hub: epsilon_RRE={hm['epsilon_RRE']:.4%}, |mass|={hm['mass_abs']:.4%}, "
                         f"p_hub={_kb(hm['p_hub'])} kPa, rho_hub={hm['rho_hub']:.4f}, "
                         f"alpha2 {hm['alpha_hub']:.2f}/{hm['alpha_tip']:.2f} deg, lambda={hm['lambda']:.4f}")
            dp = abs(hm["p_hub"] - m["p_hub"]) / max(abs(m["p_hub"]), 1e-12)
            lines.append(f"- hub-anchor sensitivity: {dp:.3%} on p_hub. The hub anchor is a "
                         "sensitivity case only; it is not selected by reaction/pass.\n")
        else:
            lines.append("- alternative hub anchor produced no valid solution.\n")

    md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    js.write_text(json.dumps(_jsonable(out), indent=2) + "\n", encoding="utf-8")
    return {"markdown": md, "json": js, "csv": csv_path, "summary_csv": summary_csv}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="41/81/161 meshes only")
    parser.add_argument("--meshes", nargs="+", type=int, default=None)
    parser.add_argument("--stages", nargs="+", type=int, default=list(STAGES))
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    meshes = tuple(args.meshes) if args.meshes else ((41, 81, 161) if args.quick else MESHES)
    out = run_evidence(meshes=meshes, stages=tuple(args.stages))
    paths = write_report(out, directory=args.out)
    for stage_res in out["stages"]:
        best = stage_res["meshes"][-1]["best"]
        if best and best.get("valid"):
            m = best["metrics"]
            print(f"stage {stage_res['stage']} ({stage_res['law']}): "
                  f"{stage_res['classification']}  N={stage_res['meshes'][-1]['N']}  "
                  f"epsilon_RRE={m['epsilon_RRE']:.3%}  mass={m['mass_rel']:+.3%}  "
                  f"lambda={m['lambda']:.4f}  alpha2={m['alpha2_min']:.1f}..{m['alpha2_max']:.1f}")
        else:
            print(f"stage {stage_res['stage']} ({stage_res['law']}): "
                  f"{stage_res['classification']} (no valid solution)")
    print("report:", paths["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
