"""Explicit EXPERIMENTAL clearance variant: delta_rel = 0.003, delta_min = 0.15 mm.

Diagnostic entry point only. The legacy book_approx defaults (0.005 / 0.2 mm),
reference baselines and the existing coupled radial evidence are untouched.
No new unknowns, equations or solver: the existing notebook stage calculator
and the existing fixed meanline control point are reused unchanged.

    python Optimizer/clearance_sensitivity.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import optimizer_core as oc  # noqa: E402
import optimize_extended as oe  # noqa: E402
import rotor_exit_coupled as rc  # noqa: E402

CASE = {"delta_rel": 0.003, "delta_min_m": 0.15e-3}
LEGACY = {"delta_rel": 0.005, "delta_min_m": 0.2e-3}
SCENARIO = "A87_relaxed"
Q_N = 0.995
N_TARGET = 110e3
TOL_POWER = 1e-10
TOL_FROZEN = 1e-12
ROTOR_HEIGHT_MIN = 0.016
ALPHA2_STRICT = ((70.0, 90.0), (70.0, 90.0), (80.0, 90.0))
FLOOR_RATIO = CASE["delta_min_m"] / LEGACY["delta_min_m"]
RESULTS = Path(__file__).resolve().parent / "results"
STAGE_INPUT_KEYS = ("T0", "p0", "H0", "ρ", "D_c", "D_л", "G_c", "G_л", "α1")
POSITIVE_KEYS = ("T0", "p0", "T1", "p1", "T2", "p2", "T_ст", "T_ст_п", "p_ст_п",
                 "l_c", "l_л", "c1", "c2a", "w1", "w2", "L_u", "h_з")
LIST_FIELDS = ("T0_K", "p0_Pa", "T1_K", "p1_Pa", "T2_K", "p2_Pa",
               "p_static_heated_Pa", "T_static_heated_K", "rho_static_heated_kg_m3",
               "T_total_heated_K", "p_total_heated_Pa",
               "heights_m", "delta_m", "hz_J_kg", "L_u_J_kg", "L_t_J_kg",
               "G_stator_kg_s", "G_rotor_kg_s")
COMPARISON_KEYS = ("eta_old", "eta_new", "delta_abs", "delta_points", "delta_rel_pct")
STAGE_COUNT = 3
CONTROL_X_LEN = 10
FROZEN_RAW_FIELDS = {
    "l_rotor_m": "heights_m", "T1_K": "T1_K", "T2_K": "T2_K",
    "L_u_J_kg": "L_u_J_kg", "alpha1_deg": "alpha1_deg", "alpha2_deg": "alpha2_deg",
}
SCALAR_FIELDS = ("N_W", "N_rel", "eta_t", "G_air_kg_s", "sigma_max_MPa", "q_n",
                 "u_max_m_s", "u_mean_m_s", "residual_mass_unheated",
                 "residual_energy_unheated", "residual_p2z")
FINITE_LIST_FIELDS = ("alpha1_deg", "alpha2_deg", "mass_static_heated_signed_rel_error")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
ROOT_RHO_METHOD = {
    "method": "approximate_geometry_exponent_estimate",
    "source": "Optimizer/optimize_extended.py:194-199 -> "
              "rho_root = 1 - (1-rho) * (D_c/(D_c-l_c))**(2*cos(alpha1)**2)",
    "assumption": "alpha1 = const on stages 1-2; D_c and l_c from meanline geometry; "
                  "geometric exponent estimate, not a radial field",
    "not_a_radial_profile": True,
    "same_as_baseline": False,
    "baseline_id": "BOOK-APPROX-001",
    "baseline_stored_first_two_rounded": [0.0609, 0.0673],
    "baseline_changed": False,
    "note": "Stage-1 estimate differs from the law-specific baseline diagnostic 0.0609 "
            "(stage 2 agrees to 0.0673). Values are reported as-is, never averaged or "
            "substituted; the physical model and the baseline are unchanged.",
}

PROTECTED_FILES = (
    "Mathcad_ST_cooling_Jupyter.ipynb",
    "Anton-Mathcad_ST_cooling_Jupyter (2).ipynb",
    "Sergey-TVD_pi6_sequential.ipynb",
    "Turbinis-Cycle/Turbinis-Cycle.ipynb",
    "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb",
    "Optimizer/optimize_extended.py",
    "Optimizer/optimizer_core.py",
    "Optimizer/rotor_exit_coupled.py",
)
PROTECTED_GLOBS = (
    "turbine/radial/*.py",
    "reference/baselines/*.json",
    "Optimizer/results/stage6b45b*",
)


def protected_hashes():
    """SHA-256 of every protected source/baseline/evidence artifact."""
    paths = [ROOT / name for name in PROTECTED_FILES]
    for pattern in PROTECTED_GLOBS:
        paths.extend(sorted(ROOT.glob(pattern)))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def stage_inputs(stage, n_rpm):
    """Exact inputs of one existing stage-calculator call; kappa from its own c2a/c1a."""
    out = {key: float(stage[key]) for key in STAGE_INPUT_KEYS}
    out["n_rpm"] = float(n_rpm)
    out["κ"] = float(stage["c2a"] / stage["c1a"])
    return out


def local_losses(ns, inputs):
    """Call the existing notebook stage calculator with identical inputs in one namespace."""
    rows = []
    for s in inputs:
        r = ns["ступень"](s["T0"], s["p0"], s["H0"], s["ρ"], s["D_c"], s["D_л"],
                          s["n_rpm"], s["G_c"], s["G_л"], s["α1"], s["κ"])
        rows.append({
            "delta_m": float(r["δ_л"]), "hz_J_kg": float(r["h_з"]),
            "l_rotor_m": float(r["l_л"]), "T1_K": float(r["T1"]),
            "T2_K": float(r["T2"]), "L_u_J_kg": float(r["L_u"]),
            "alpha1_deg": float(r["α1"]), "alpha2_deg": float(r["α2"]),
        })
    return rows


def _close(a, b):
    """Tight deterministic-equality check for recomputed evidence."""
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15)


def compare_frozen(inputs, ns_old, ns_new):
    """First-order fixed-field comparison: same inputs, old vs new clearance only."""
    inputs = list(inputs)
    if len(inputs) != STAGE_COUNT:
        raise ValueError(f"compare_frozen: {len(inputs)} inputs, expected {STAGE_COUNT}")
    old = local_losses(ns_old, inputs)
    new = local_losses(ns_new, inputs)
    if len(old) != STAGE_COUNT or len(new) != STAGE_COUNT:
        raise ValueError("compare_frozen: stage calculator returned the wrong row count")
    rows, errors, unchanged = [], [], True
    for i, (s, a, b) in enumerate(zip(inputs, old, new), 1):
        for label, r in (("old", a), ("new", b)):
            for key, value in r.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value):
                    raise ValueError(f"stage {i} {label}.{key}={value!r} is not finite")
            for key in ("l_rotor_m", "T1_K", "T2_K", "L_u_J_kg", "delta_m", "hz_J_kg"):
                if r[key] <= 0:
                    raise ValueError(f"stage {i} {label}.{key}={r[key]!r} is not positive")
        # hz is linear in delta/l and every other factor is identical, so the
        # expected ratio is delta_new/delta_old on both floor and relative branches.
        expected = b["delta_m"] / a["delta_m"]
        ratio = b["hz_J_kg"] / a["hz_J_kg"]
        if not (math.isfinite(expected) and expected > 0
                and math.isfinite(ratio) and ratio > 0):
            raise ValueError(f"stage {i}: non-finite hz or delta ratio")
        errors.append(abs(ratio - expected))
        same = all(a[key] == b[key] for key in
                   ("T1_K", "T2_K", "L_u_J_kg", "alpha1_deg", "alpha2_deg"))
        unchanged = unchanged and same
        rows.append({"inputs": dict(s), "old": a, "new": b, "ratio": ratio,
                     "floor_branch": (_close(a["delta_m"], LEGACY["delta_min_m"])
                                      and _close(b["delta_m"], CASE["delta_min_m"])),
                     "local_fields_unchanged": same})
    max_err = max(errors)
    if not math.isfinite(max_err):
        raise ValueError("compare_frozen: non-finite max ratio error")
    return {"stages": rows, "all_unchanged": unchanged, "max_ratio_error": max_err}


def _case(model, x, delta_rel, delta_min):
    ns, _ = oe.build_ns(model, Q_N, Q_N**2 * oe.SIGMA_LIM)
    ns["δ_отн"], ns["δ_мин"] = float(delta_rel), float(delta_min)
    result = oe.eval_candidate(np.asarray(x, dtype=float).copy(), ns, SCENARIO, Q_N)
    if result["status"] != "ok":
        raise RuntimeError(f"existing eval_candidate failed: {result['status']} {result['err']}")
    _, stages = ns["распределить_π_const_hub"](x[0], x[1], ns["π_т"] ** (1 / ns["z"]))
    if abs(stages[0]["l_c"] - result["lc1"]) > 0.0:
        raise RuntimeError("recomputed stage fields disagree with eval_candidate")
    return ns, result, stages


def _validate_stages(stages, label):
    for i, stage in enumerate(stages, 1):
        for key in POSITIVE_KEYS:
            value = float(stage[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} stage {i}: {key}={value!r} is not finite positive")


def _meanline(result, stages, ns, x):
    gas_r = float(ns["R"])
    p_static, t_static, rho_static, mass_signed = [], [], [], []
    t_total, p_total = [], []
    for s in stages:
        area = math.pi * float(s["D_л"]) * float(s["l_л"])
        p_s, t_s = float(s["p2"]), float(s["T_ст"])
        rho_s = p_s / (gas_r * t_s)  # static heated exit state (book p. 93/125)
        p_static.append(p_s)
        t_static.append(t_s)
        rho_static.append(rho_s)
        # Signed dimensionless diagnostic rho*A*c2a/G_л - 1. With the existing
        # identity A = G_л*v2/(c2a*pi*D_л) it equals T2/T_ст - 1.
        mass_signed.append(rho_s * area * float(s["c2a"]) / float(s["G_л"]) - 1.0)
        # Total state is reported separately; it is not a density.
        t_total.append(float(s["T_ст_п"]))
        p_total.append(float(s["p_ст_п"]))
    alpha2 = [float(s["α2"]) for s in stages]
    return {
        "x": [float(v) for v in x],
        "stage_inputs": [stage_inputs(s, float(x[0])) for s in stages],
        "N_W": float(result["N"]), "N_rel": float(result["N"] / N_TARGET - 1.0),
        "eta_t": float(result["eta_t"]), "eta_ok": bool(result["eta_t"] >= 0.90),
        "G_air_kg_s": float(ns["G_в"]),
        "G_stator_kg_s": [float(s["G_c"]) for s in stages],
        "G_rotor_kg_s": [float(s["G_л"]) for s in stages],
        "heights_m": [float(s["l_л"]) for s in stages],
        "delta_m": [float(s["δ_л"]) for s in stages],
        "hz_J_kg": [float(s["h_з"]) for s in stages],
        "L_u_J_kg": [float(s["L_u"]) for s in stages],
        "L_t_J_kg": [float(s["L_т"]) for s in stages],
        "T0_K": [float(s["T0"]) for s in stages],
        "p0_Pa": [float(s["p0"]) for s in stages],
        "T1_K": [float(s["T1"]) for s in stages],
        "p1_Pa": [float(s["p1"]) for s in stages],
        "T2_K": [float(s["T2"]) for s in stages],
        "p2_Pa": [float(s["p2"]) for s in stages],
        "p_static_heated_Pa": p_static,
        "T_static_heated_K": t_static,
        "rho_static_heated_kg_m3": rho_static,
        "mass_static_heated_signed_rel_error": mass_signed,
        "T_total_heated_K": t_total,
        "p_total_heated_Pa": p_total,
        "alpha1_deg": [float(s["α1"]) for s in stages],
        "alpha2_deg": alpha2,
        "strict_alpha2": all(lo <= a <= hi for (lo, hi), a in zip(ALPHA2_STRICT, alpha2)),
        "sigma_max_MPa": float(result["sigma"]), "q_n": float(result["qn"]),
        "u_max_m_s": float(result["u"]),
        "u_mean_m_s": sum(0.5 * (float(s["u1"]) + float(s["u2"])) for s in stages) / len(stages),
        "residual_mass_unheated": float(result["cont"]),
        "residual_energy_unheated": float(result["energy"]),
        "residual_p2z": float(result["p2z"]),
        "eval_feasible": bool(result["feasible"]),
        "eval_violations": [f"{v['name']}:{v['amount']:.6g}" for v in result["violations"]],
        "rho_root_stages_1_2": [float(v) for v in result["rho_root"]],
    }


def _comparison(old, new):
    eta_old, eta_new = old["eta_t"], new["eta_t"]
    return {
        "eta_old": eta_old, "eta_new": eta_new,
        "delta_abs": eta_new - eta_old,
        "delta_points": 100.0 * (eta_new - eta_old),
        "delta_rel_pct": 100.0 * (eta_new / eta_old - 1.0),
        "power_old_kW": old["N_W"] / 1e3, "power_new_kW": new["N_W"] / 1e3,
    }


def _power_closure(meanline):
    """Power residual is recomputed from N_W; a stored N_rel must reproduce it."""
    residual = meanline["N_W"] / N_TARGET - 1.0
    return abs(residual) <= TOL_POWER and _close(meanline["N_rel"], residual)


def _positive_fields_ok(meanline):
    for key in LIST_FIELDS:
        if not all(math.isfinite(v) and v > 0 for v in meanline[key]):
            return False
    for key in ("G_air_kg_s", "eta_t", "sigma_max_MPa", "q_n", "u_max_m_s", "u_mean_m_s"):
        if not (math.isfinite(meanline[key]) and meanline[key] > 0):
            return False
    return all(math.isfinite(v) for key in FINITE_LIST_FIELDS
               for v in meanline[key])


def _isfinite_number(value):
    """True for a real finite number, never for a bool."""
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _frozen_audit(frozen, raw_old):
    """Recompute fixed-field evidence and bind every row to the raw old meanline.

    Malformed or non-positive raw numbers raise ValueError; a stored summary that
    disagrees with the recomputation returns False flags so classify fails closed.
    Each row must use the raw old stage inputs, the declared floor branch, and the
    ratio implied by that branch (0.75 on the floors of the current control point).
    """
    if not isinstance(frozen, dict):
        raise ValueError("frozen: not a mapping")
    rows = frozen.get("stages")
    if not isinstance(rows, (list, tuple)) or len(rows) != STAGE_COUNT:
        raise ValueError("frozen: stages must contain exactly 3 rows")
    raw_inputs = raw_old.get("stage_inputs") if isinstance(raw_old, dict) else None
    if not isinstance(raw_inputs, (list, tuple)) or len(raw_inputs) != STAGE_COUNT:
        raise ValueError("meanline.old.stage_inputs must contain exactly 3 rows")
    expected_keys = set(STAGE_INPUT_KEYS) | {"n_rpm", "κ"}
    errors, unchanged, delta_ok, ratios_ok = [], True, True, True
    inputs_ok, outputs_ok, branch_ok, ratio_ok, all_floor = True, True, True, True, True
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"frozen stage {i}: not a mapping")
        a, b = row.get("old"), row.get("new")
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ValueError(f"frozen stage {i}: missing old/new row")
        stored_inputs, raw_row = row.get("inputs"), raw_inputs[i - 1]
        if not isinstance(stored_inputs, dict) or set(stored_inputs) != expected_keys:
            raise ValueError(f"frozen stage {i}: inputs keys do not match the stage call")
        if not isinstance(raw_row, dict) or set(raw_row) != expected_keys:
            raise ValueError(f"meanline.old stage {i}: stage_inputs keys do not match")
        for key in expected_keys:
            if not _isfinite_number(stored_inputs[key]) or not _isfinite_number(raw_row[key]):
                raise ValueError(f"frozen stage {i}: input {key} is not finite")
            if not _close(float(stored_inputs[key]), float(raw_row[key])):
                inputs_ok = False
        for label, r, rule in (("old", a, LEGACY), ("new", b, CASE)):
            for key in ("l_rotor_m", "T1_K", "T2_K", "L_u_J_kg",
                        "alpha1_deg", "alpha2_deg", "delta_m", "hz_J_kg"):
                if not _isfinite_number(r.get(key)):
                    raise ValueError(f"frozen stage {i} {label}.{key} is not finite")
            if r["l_rotor_m"] <= 0 or r["delta_m"] <= 0:
                raise ValueError(f"frozen stage {i} {label}: non-positive length or delta")
            prescribed = max(rule["delta_rel"] * r["l_rotor_m"], rule["delta_min_m"])
            if not _close(r["delta_m"], prescribed):
                delta_ok = False
        for key, raw_key in FROZEN_RAW_FIELDS.items():
            values = raw_old.get(raw_key)
            if not isinstance(values, (list, tuple)) or len(values) != STAGE_COUNT:
                raise ValueError(f"meanline.old.{raw_key} must contain exactly 3 values")
            if not _isfinite_number(values[i - 1]):
                raise ValueError(f"meanline.old.{raw_key}[{i - 1}] is not finite")
            expected_value = float(values[i - 1])
            if not _close(a[key], expected_value) or not _close(b[key], expected_value):
                outputs_ok = False
        for key in ("delta_m", "hz_J_kg"):
            values = raw_old.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != STAGE_COUNT:
                raise ValueError(f"meanline.old.{key} must contain exactly 3 values")
            if not _isfinite_number(values[i - 1]) or values[i - 1] <= 0:
                raise ValueError(f"meanline.old.{key}[{i - 1}] is not finite positive")
            if not _close(a[key], float(values[i - 1])):
                outputs_ok = False
        if a["hz_J_kg"] <= 0 or b["hz_J_kg"] <= 0:
            raise ValueError(f"frozen stage {i}: non-positive hz")
        expected = b["delta_m"] / a["delta_m"]
        ratio = b["hz_J_kg"] / a["hz_J_kg"]
        if not (math.isfinite(expected) and expected > 0
                and math.isfinite(ratio) and ratio > 0):
            raise ValueError(f"frozen stage {i}: non-finite hz or delta ratio")
        stored_ratio = row.get("ratio")
        if not _isfinite_number(stored_ratio) or not _close(stored_ratio, ratio):
            ratios_ok = False
        errors.append(abs(ratio - expected))
        same = all(a[key] == b[key] for key in
                   ("T1_K", "T2_K", "L_u_J_kg", "alpha1_deg", "alpha2_deg"))
        if row.get("local_fields_unchanged") is not same:
            unchanged = False
        unchanged = unchanged and same
        old_floor = _close(a["delta_m"], LEGACY["delta_min_m"])
        new_floor = _close(b["delta_m"], CASE["delta_min_m"])
        old_relative = _close(a["delta_m"], LEGACY["delta_rel"] * a["l_rotor_m"])
        new_relative = _close(b["delta_m"], CASE["delta_rel"] * b["l_rotor_m"])
        expected_branch = old_floor and new_floor
        if row.get("floor_branch") is not expected_branch:
            branch_ok = False
        if expected_branch:
            expected_ratio = FLOOR_RATIO
        elif old_relative and new_relative:
            expected_ratio = CASE["delta_rel"] / LEGACY["delta_rel"]
        else:
            expected_ratio = None
        all_floor = all_floor and expected_branch
        if expected_ratio is None or not _close(ratio, expected_ratio):
            ratio_ok = False
    stored_max = frozen.get("max_ratio_error")
    if not _isfinite_number(stored_max):
        raise ValueError("frozen: max_ratio_error is not finite")
    recomputed_max = max(errors)
    ratios_ok = ratios_ok and _close(stored_max, recomputed_max)
    stored_unchanged = frozen.get("all_unchanged")
    if not isinstance(stored_unchanged, bool):
        raise ValueError("frozen: all_unchanged is not a bool")
    unchanged = unchanged and stored_unchanged
    return {"max_ratio_error": recomputed_max, "delta_rule_ok": delta_ok,
            "ratios_ok": ratios_ok, "all_unchanged": unchanged,
            "inputs_ok": inputs_ok, "outputs_ok": outputs_ok,
            "branch_ok": branch_ok, "expected_ratio_ok": ratio_ok,
            "floor_ratio_ok": all_floor and ratio_ok}


def _checks(res):
    old, new, frozen, control = res["meanline"]["old"], res["meanline"]["new"], res["frozen"], res["control_point"]
    audit = _frozen_audit(frozen, old)
    rows = []

    def add(name, ok, detail, limit=""):
        rows.append({"name": name, "pass": bool(ok), "detail": detail, "limit": limit})

    add("power_old", _power_closure(old),
        f"N_rel={old['N_rel']:.3e}; from N_W={old['N_W'] / N_TARGET - 1.0:.3e}",
        f"<={TOL_POWER:g}")
    add("power_new", _power_closure(new),
        f"N_rel={new['N_rel']:.3e}; from N_W={new['N_W'] / N_TARGET - 1.0:.3e}",
        f"<={TOL_POWER:g}")
    add("frozen_hz_ratio_expected",
        audit["ratios_ok"] and audit["expected_ratio_ok"] and audit["floor_ratio_ok"]
        and audit["max_ratio_error"] <= TOL_FROZEN,
        f"max|ratio-expected|={audit['max_ratio_error']:.3e}; "
        f"expected {FLOOR_RATIO:g} on the floor branch", f"<={TOL_FROZEN:g}")
    add("frozen_delta_prescribed", audit["delta_rule_ok"] and audit["branch_ok"],
        "old 0.2 mm, new 0.15 mm at current heights; declared floor branch matches")
    add("frozen_local_fields_unchanged", audit["all_unchanged"],
        "T1, T2, L_u, alpha1, alpha2 identical in both namespaces")
    add("frozen_inputs_match_meanline", audit["inputs_ok"] and audit["outputs_ok"],
        "frozen rows reproduce the raw old stage inputs and their local fields")
    add("same_control_point_x", old["x"] == new["x"] == control["x"], "one fixed x for both cases")
    add("finite_positive_fields", _positive_fields_ok(old) and _positive_fields_ok(new),
        "all reported inputs/outputs finite and positive")
    add("protected_hashes_unchanged",
        res["protected"]["before"] == res["protected"]["after"],
        f"{len(res['protected']['before'])} protected files")
    return rows


def _constraints(res):
    rows = []

    def add(name, ok, detail, limit=""):
        rows.append({"name": name, "pass": bool(ok), "detail": detail, "limit": limit})

    for label in ("old", "new"):
        m = res["meanline"][label]
        h = m["heights_m"]
        add(f"{label}: N=110 kW", _power_closure(m), f"{m['N_W'] / 1e3:.9f} kW")
        strict_alpha2 = all(lo <= a <= hi for (lo, hi), a in zip(ALPHA2_STRICT, m["alpha2_deg"]))
        add(f"{label}: alpha2 strict", strict_alpha2,
            ", ".join(f"{a:.5f}" for a in m["alpha2_deg"]), "70..90,70..90,80..90")
        add(f"{label}: sigma<=300 MPa", m["sigma_max_MPa"] <= 300.0,
            f"{m['sigma_max_MPa']:.6f} MPa")
        add(f"{label}: q_n<=0.995", m["q_n"] <= Q_N,
            f"{m['q_n']:.9f} (linked to sigma, not independent)")
        add(f"{label}: u<=400 m/s", m["u_max_m_s"] <= 400.0, f"{m['u_max_m_s']:.6f} m/s")
        add(f"{label}: eta_t>=0.90", m["eta_t"] >= 0.90, f"{m['eta_t']:.9f}")
        add(f"{label}: l_rk1>=16 mm", h[0] >= ROTOR_HEIGHT_MIN,
            f"{h[0] * 1e3:.6f} mm")
        add(f"{label}: heights increasing", h[0] < h[1] < h[2],
            ", ".join(f"{v * 1e3:.6f}" for v in h) + " mm")
        for i, rho in enumerate(m["rho_root_stages_1_2"], 1):
            add(f"{label}: rho_root_{i}>=0.05 (approximate geometry exponent estimate)",
                rho >= 0.05, f"{rho:.6f}")
        add(f"{label}: existing eval_candidate feasible", m["eval_feasible"],
            "; ".join(m["eval_violations"]) or "no violations")
    return rows


def _stage_input_problems(rows, label):
    """Each meanline call must keep all stage-call inputs as finite numbers."""
    expected = set(STAGE_INPUT_KEYS) | {"n_rpm", "κ"}
    if not isinstance(rows, (list, tuple)) or len(rows) != STAGE_COUNT:
        return [f"schema:{label}.stage_inputs"]
    problems = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected \
                or not all(_isfinite_number(v) for v in row.values()):
            problems.append(f"schema:{label}.stage_inputs[{i}]")
    return problems


def _schema_problems(res):
    """Explicit structural validation of the raw diagnostic result."""
    mean = res.get("meanline")
    if not isinstance(mean, dict):
        return ["schema:meanline"]
    problems = []
    for label in ("old", "new"):
        m = mean.get(label)
        if not isinstance(m, dict):
            problems.append(f"schema:meanline.{label}")
            continue
        for key in SCALAR_FIELDS:
            if not _isfinite_number(m.get(key)):
                problems.append(f"schema:{label}.{key}")
        for key in LIST_FIELDS + FINITE_LIST_FIELDS:
            values = m.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != STAGE_COUNT:
                problems.append(f"schema:{label}.{key}")
            elif not all(_isfinite_number(v) for v in values):
                problems.append(f"schema:{label}.{key}")
        x = m.get("x")
        if not isinstance(x, (list, tuple)) or len(x) != CONTROL_X_LEN \
                or not all(_isfinite_number(v) for v in x):
            problems.append(f"schema:{label}.x")
        problems.extend(_stage_input_problems(m.get("stage_inputs"), label))
        strict = m.get("strict_alpha2")
        if not isinstance(strict, bool):
            problems.append(f"schema:{label}.strict_alpha2")
        alpha2 = m.get("alpha2_deg")
        if isinstance(strict, bool) and isinstance(alpha2, (list, tuple)) \
                and len(alpha2) == STAGE_COUNT and all(_isfinite_number(v) for v in alpha2):
            actual = all(lo <= a <= hi for (lo, hi), a in zip(ALPHA2_STRICT, alpha2))
            if strict is not actual:
                problems.append(f"stale:{label}.strict_alpha2")
        if not isinstance(m.get("eval_feasible"), bool):
            problems.append(f"schema:{label}.eval_feasible")
        rho = m.get("rho_root_stages_1_2")
        if not isinstance(rho, (list, tuple)) or len(rho) != 2 \
                or not all(_isfinite_number(v) for v in rho):
            problems.append(f"schema:{label}.rho_root_stages_1_2")
    control = res.get("control_point")
    if not isinstance(control, dict):
        problems.append("schema:control_point")
    else:
        cx = control.get("x")
        if not isinstance(cx, (list, tuple)) or len(cx) != CONTROL_X_LEN \
                or not all(_isfinite_number(v) for v in cx):
            problems.append("schema:control_point.x")
        else:
            for label in ("old", "new"):
                m = mean.get(label)
                if isinstance(m, dict) and m.get("x") != [float(v) for v in cx]:
                    problems.append(f"stale:control_point.x_{label}")
    return problems


def classify(res):
    """Software acceptance summary; fails closed on missing, duplicated, empty,
    non-finite or stale entries, and never promotes diagnostic checks to an
    engineering pass."""
    if not isinstance(res, dict):
        res = {}
        problems = ["schema:result"]
    else:
        problems = []
    problems.extend(_schema_problems(res))
    try:
        expected = _checks(res)  # recomputed from raw numbers, not from booleans
    except Exception as exc:  # malformed raw result cannot be verified
        expected = []
        problems.append(f"checks_unrecomputable:{type(exc).__name__}")
    want = [c["name"] for c in expected]
    supplied = res.get("checks")
    if not isinstance(supplied, list) or not supplied:
        problems.append("schema:checks")
    else:
        names = [c.get("name") if isinstance(c, dict) else None for c in supplied]
        if names != want or len(set(names)) != len(names):
            problems.append("checks_incomplete_or_duplicate")
        for c in supplied:
            if not isinstance(c, dict) or c.get("pass") is not True:
                name = c.get("name") if isinstance(c, dict) else repr(c)
                problems.append(f"check_not_true:{name}")
        if names == want:
            problems.extend(f"stale_check:{c['name']}"
                            for c, e in zip(supplied, expected) if c["pass"] != e["pass"])
    problems.extend(f"recomputed_fail:{c['name']}" for c in expected if not c["pass"])

    protected = res.get("protected")
    before = protected.get("before") if isinstance(protected, dict) else None
    after = protected.get("after") if isinstance(protected, dict) else None
    if not (isinstance(before, dict) and before and before == after):
        problems.append("protected_sources_changed")
    elif not all(isinstance(k, str) and isinstance(v, str) and SHA256_HEX.match(v)
                 for k, v in before.items()):
        problems.append("protected_sources_invalid")

    comparison = res.get("comparison")
    for key in COMPARISON_KEYS:
        value = comparison.get(key) if isinstance(comparison, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            problems.append(f"non-finite:{key}")
    try:  # comparison must agree with the raw meanline, not only be finite
        wanted_comparison = _comparison(res["meanline"]["old"], res["meanline"]["new"])
    except Exception as exc:
        wanted_comparison = None
        problems.append(f"comparison_unrecomputable:{type(exc).__name__}")
    if wanted_comparison is not None:
        if not isinstance(comparison, dict):
            problems.append("schema:comparison")
        else:
            if set(comparison) != set(wanted_comparison):
                problems.append("comparison_keys_mismatch")
            for key, expected_value in wanted_comparison.items():
                value = comparison.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value) or not _close(value, expected_value):
                    problems.append(f"stale_comparison:{key}")

    try:  # constraints and feasibility are recomputed from raw means, never trusted
        wanted_constraints = _constraints(res)
    except Exception as exc:
        wanted_constraints = None
        problems.append(f"constraints_unrecomputable:{type(exc).__name__}")
    constraints = res.get("constraints")
    if wanted_constraints is not None:
        if not isinstance(constraints, list):
            problems.append("schema:constraints")
        else:
            names = [c.get("name") if isinstance(c, dict) else None for c in constraints]
            if names != [c["name"] for c in wanted_constraints]:
                problems.append("constraints_incomplete_or_reordered")
            else:
                for c, e in zip(constraints, wanted_constraints):
                    if not isinstance(c.get("pass"), bool):
                        problems.append(f"constraint_pass_not_bool:{c['name']}")
                    elif c["pass"] is not e["pass"]:
                        problems.append(f"stale_constraint:{c['name']}")
                    if c.get("detail") != e["detail"] or c.get("limit") != e["limit"]:
                        problems.append(f"stale_constraint_evidence:{c['name']}")
    feasible = res.get("feasible")
    if not isinstance(feasible, bool):
        problems.append("schema:feasible")
    elif wanted_constraints is not None \
            and feasible != all(c["pass"] for c in wanted_constraints):
        problems.append("stale_feasible")
    problems = list(dict.fromkeys(problems))
    return {
        "software_pass": not problems,
        "problems": problems,
        "diagnostic_checks_pass": bool(expected) and all(c["pass"] for c in expected),
        "engineering_pass": False,
        "engineering_status": "EXPERIMENTAL - not accepted design evidence",
        "scope": "diagnostic project checks only; not a full Category E acceptance",
        "verified_claim": False,
    }


def run():
    """Evaluate the requested case against the legacy clearance at one fixed control point."""
    before = protected_hashes()
    cp = rc.control_point()
    x = np.asarray(cp["x"], dtype=float).copy()
    model = oc.load_model()
    ns_old, r_old, st_old = _case(model, x, LEGACY["delta_rel"], LEGACY["delta_min_m"])
    ns_new, r_new, st_new = _case(model, x, CASE["delta_rel"], CASE["delta_min_m"])
    _validate_stages(st_old, "old")
    _validate_stages(st_new, "new")
    inputs = [stage_inputs(s, float(x[0])) for s in st_old]
    frozen = compare_frozen(inputs, ns_old, ns_new)
    old = _meanline(r_old, st_old, ns_old, x)
    new = _meanline(r_new, st_new, ns_new, x)
    comparison = _comparison(old, new)
    after = protected_hashes()
    res = {
        "case": {
            "label": "EXPERIMENTAL clearance variant, explicit user-requested diagnostic",
            "delta_rel": CASE["delta_rel"], "delta_min_m": CASE["delta_min_m"],
            "legacy_delta_rel": LEGACY["delta_rel"],
            "legacy_delta_min_m": LEGACY["delta_min_m"],
            "applied_delta_rule": "max(delta_rel*l_RK, delta_min)",
            "crossover_m_relative_branch": CASE["delta_min_m"] / CASE["delta_rel"],
            "legacy_crossover_m_relative_branch": LEGACY["delta_min_m"] / LEGACY["delta_rel"],
            "explicit_user_request": True,
            "diagnostic_entry_point_only": True,
            "global_default_migrated": False,
        },
        "control_point": {
            "x": [float(v) for v in x], "n_rpm": float(x[0]),
            "D_hub_m": float(x[1]), "w1": float(x[2]), "w2": float(x[3]),
            "alpha1_stage1_deg": float(cp["alpha1_stage1"]),
            "scenario": SCENARIO, "q_n_limit": Q_N,
            "power_target_W": N_TARGET, "power_tolerance_relative": TOL_POWER,
            "fixed_controls_not_fixed_solution": True,
        },
        "model_classification": {
            "applied_gap": "SOFTWARE ASSUMPTION: user-prescribed 0.003 / 0.15 mm; not a verified manufacturing limit",
            "hz_correlation": "EMPIRICAL CORRELATION / EXPERIMENTAL: Main_Turbinis_Calc_Book points 113-115, formula unchanged; applicability and uncertainty not verified",
            "friction_loss_and_energy_coupling": "EXPERIMENTAL legacy book_approx",
            "ideal_gas_state": "ACCEPTED",
            "new_unknowns": 0, "new_equations": 0, "new_solver": False,
        },
        "frozen": frozen,
        "meanline": {"old": old, "new": new},
        "comparison": comparison,
        "protected": {"before": before, "after": after},
        "rho_root_method": ROOT_RHO_METHOD,
        "engineering": {
            "pass": False,
            "status": "EXPERIMENTAL - not accepted design evidence",
            "reasons": [
                "Category E is not satisfied at this control point: l_RK1 < 16 mm and root reaction 1 < 0.05",
                "the 0.003 / 0.15 mm gap is a user software assumption, not a thermal or mechanical approval",
                "radial closure, stage-3 root reaction and stator-exit closure are not validated",
            ],
            "standard_verifier_baseline_gate": "FAIL in this working tree (reference/ untracked); verify.py not rerun here",
            "verified_claim": False,
        },
        "not_checked": [
            "stage-3 root reaction is NOT_CHECKED (the estimate covers stages 1-2 only and is not a radial profile)",
            "stator exit closure and stator radial fields are NOT_CHECKED",
            "radial closure and mesh refinement (801-point grid) were not rerun for this case",
            "multiple starts of any radial solver were not performed",
            "material, temperature-dependent strength, creep/resource, thermal stress and cooling are NOT_CHECKED",
            "thermal and mechanical approval of the 0.15 mm minimum gap, and life/durability, are NOT_CHECKED",
            "full Category E feasibility cannot pass at this control point",
            "no optimization, geometry search or real blade profile was performed",
        ],
        "provenance": rc.provenance(),
    }
    res["checks"] = _checks(res)
    res["constraints"] = _constraints(res)
    res["feasible"] = all(c["pass"] for c in res["constraints"])
    res["classification"] = classify(res)
    return res


def _fmt(values, scale=1.0, nd=6):
    return ", ".join(f"{v * scale:.{nd}f}" for v in values)


def _markdown(res):
    c = res["case"]
    comp = res["comparison"]
    old, new = res["meanline"]["old"], res["meanline"]["new"]
    lines = [
        "# Clearance variant delta_rel = 0.003, delta_min = 0.15 mm",        "",
        "**Status: EXPERIMENTAL diagnostic. Not accepted design evidence.**",
        "",
        "## Requested case vs legacy book_approx",
        "",
        "| quantity | legacy book_approx | requested diagnostic |",
        "|---|---|---|",
        f"| delta_rel | {c['legacy_delta_rel']:g} | {c['delta_rel']:g} |",
        f"| delta_min | {c['legacy_delta_min_m'] * 1e3:g} mm | {c['delta_min_m'] * 1e3:g} mm |",
        "",
        f"- Applied relation: `{c['applied_delta_rule']}` (software assumption, "
        "user-prescribed parameters, not a verified manufacturing limit).",
        f"- Relative-branch crossover: {c['crossover_m_relative_branch'] * 1e3:g} mm "
        f"(legacy {c['legacy_crossover_m_relative_branch'] * 1e3:g} mm); the current "
        "rotor heights are below both, so both cases sit on their floors.",
        "- This is a new explicit diagnostic entry point, **not** a global default",
        "  migration: no existing default, notebook, core module or baseline was edited.",
        "- No new unknowns, equations or solver. One fixed meanline control point is",
        "  reused with the existing flow-to-power iteration.",
        "",
        "## Classification",
        "",
    ]
    for key, value in res["model_classification"].items():
        lines.append(f"- `{key}`: {value}")
    cls, eng = res["classification"], res["engineering"]
    lines += [
        "",
        "## Software checks (diagnostic, fail-closed)",
        "",
        f"Software acceptance: **{'PASS' if cls['software_pass'] else 'FAIL'}** "
        f"(problems: {cls['problems'] or 'none'}).",
        f"Scope: {cls['scope']}.",
        "",
        "## Engineering acceptance: FAIL (EXPERIMENTAL, no verified claim)",
        "",
        f"Engineering acceptance: **FAIL**; feasible = {res['feasible']}; "
        f"diagnostic project checks passed = {cls['diagnostic_checks_pass']}.",
    ]
    lines += [f"- {reason}" for reason in eng["reasons"]]
    lines += [
        f"- standard verifier baseline gate: {eng['standard_verifier_baseline_gate']}",
        "- no overall 'verified' claim is made.",
        "",
    ]
    info = res["rho_root_method"]
    lines += [
        "## Root reaction estimate: approximate geometry exponent",
        "",
        f"- method: `{info['method']}` (not a radial profile, not the baseline law)",
        f"- source: {info['source']}",
        f"- assumption: {info['assumption']}",
        f"- stages 1-2, old case: {_fmt(old['rho_root_stages_1_2'], nd=6)}; "
        f"new case: {_fmt(new['rho_root_stages_1_2'], nd=6)}",
        f"- `{info['baseline_id']}` stores {info['baseline_stored_first_two_rounded']}: "
        f"{info['note']}",
        "",
    ]
    lines += [
        "## Source hashes (before == after)",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    lines += [f"| `{path}` | `{digest}` |" for path, digest in res["protected"]["before"].items()]
    lines += [
        "",
        "## A. Frozen-field local loss comparison (first order, fixed field)",
        "",
        "Identical stage inputs are passed to the existing stage calculator in both",
        "namespaces; only delta changes. This is not a new coupled solution.",
        "",
        "| stage | l_RK mm | delta old mm | delta new mm | hz old J/kg | hz new J/kg | dhz J/kg | hz new/old | local fields |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(res["frozen"]["stages"], 1):
        a, b = row["old"], row["new"]
        lines.append(
            f"| {i} | {a['l_rotor_m'] * 1e3:.9f} | {a['delta_m'] * 1e3:.6f} | "
            f"{b['delta_m'] * 1e3:.6f} | {a['hz_J_kg']:.6f} | {b['hz_J_kg']:.6f} | "
            f"{b['hz_J_kg'] - a['hz_J_kg']:.6f} | {row['ratio']:.12f} | "
            f"{'unchanged' if row['local_fields_unchanged'] else 'CHANGED'} |"
        )
    lines += [
        "",
        f"Max |hz ratio - {FLOOR_RATIO:g}| = {res['frozen']['max_ratio_error']:.3e}; "
        f"tolerance {TOL_FROZEN:g}.",
        "",
        "## B. Recomputed whole meanline, fixed controls (not a fixed solution)",
        "",
        "The flow is refitted to 110 kW for each clearance case, so geometry and",
        "velocities may move. Fixed controls, not a fixed solution.",
        "",
        "| property | old | new |",
        "|---|---|---|",
        f"| eta_t (global total turbine efficiency) | {old['eta_t']:.12f} | {new['eta_t']:.12f} |",
        f"| N | {old['N_W'] / 1e3:.9f} kW | {new['N_W'] / 1e3:.9f} kW |",
        f"| N relative residual | {old['N_rel']:.3e} | {new['N_rel']:.3e} |",
        f"| G_air | {old['G_air_kg_s']:.9f} kg/s | {new['G_air_kg_s']:.9f} kg/s |",
        f"| G_stator stages 1-3 | {_fmt(old['G_stator_kg_s'], nd=9)} | {_fmt(new['G_stator_kg_s'], nd=9)} |",
        f"| G_rotor stages 1-3 | {_fmt(old['G_rotor_kg_s'], nd=9)} | {_fmt(new['G_rotor_kg_s'], nd=9)} |",
        f"| rotor heights | {_fmt(old['heights_m'], 1e3)} mm | {_fmt(new['heights_m'], 1e3)} mm |",
        f"| gap delta | {_fmt(old['delta_m'], 1e3)} mm | {_fmt(new['delta_m'], 1e3)} mm |",
        f"| hz | {_fmt(old['hz_J_kg'])} J/kg | {_fmt(new['hz_J_kg'])} J/kg |",
        f"| L_u | {_fmt(old['L_u_J_kg'])} J/kg | {_fmt(new['L_u_J_kg'])} J/kg |",
        f"| L_t | {_fmt(old['L_t_J_kg'])} J/kg | {_fmt(new['L_t_J_kg'])} J/kg |",
        f"| T0 | {_fmt(old['T0_K'])} K | {_fmt(new['T0_K'])} K |",
        f"| p0 | {_fmt(old['p0_Pa'])} Pa | {_fmt(new['p0_Pa'])} Pa |",
        f"| T1 | {_fmt(old['T1_K'])} K | {_fmt(new['T1_K'])} K |",
        f"| T2 | {_fmt(old['T2_K'])} K | {_fmt(new['T2_K'])} K |",
        f"| alpha1 | {_fmt(old['alpha1_deg'])} deg | {_fmt(new['alpha1_deg'])} deg |",
        f"| alpha2 | {_fmt(old['alpha2_deg'])} deg | {_fmt(new['alpha2_deg'])} deg |",
        f"| sigma max | {old['sigma_max_MPa']:.6f} MPa | {new['sigma_max_MPa']:.6f} MPa |",
        f"| q_n | {old['q_n']:.9f} | {new['q_n']:.9f} |",
        f"| u max | {old['u_max_m_s']:.6f} m/s | {new['u_max_m_s']:.6f} m/s |",
        f"| u mean | {old['u_mean_m_s']:.6f} m/s | {new['u_mean_m_s']:.6f} m/s |",
        f"| mass residual (unheated) | {old['residual_mass_unheated']:.3e} | {new['residual_mass_unheated']:.3e} |",
        f"| energy residual (unheated) | {old['residual_energy_unheated']:.3e} | {new['residual_energy_unheated']:.3e} |",
        f"| p2z residual | {old['residual_p2z']:.3e} | {new['residual_p2z']:.3e} |",
        f"| p2 static heated exit | {_fmt(old['p_static_heated_Pa'], nd=1)} Pa | {_fmt(new['p_static_heated_Pa'], nd=1)} Pa |",
        f"| T_ст static heated exit | {_fmt(old['T_static_heated_K'], nd=3)} K | {_fmt(new['T_static_heated_K'], nd=3)} K |",
        f"| rho = p2/(R*T_ст) | {_fmt(old['rho_static_heated_kg_m3'], nd=6)} | {_fmt(new['rho_static_heated_kg_m3'], nd=6)} |",
        f"| signed mass error rho*A*c2a/G_л-1 | {_fmt(old['mass_static_heated_signed_rel_error'], nd=6)} | {_fmt(new['mass_static_heated_signed_rel_error'], nd=6)} |",
        f"| T_ст_п total heated exit (reference only, never a density) | {_fmt(old['T_total_heated_K'], nd=3)} K | {_fmt(new['T_total_heated_K'], nd=3)} K |",
        f"| p_ст_п total heated exit (reference only, never a density) | {_fmt(old['p_total_heated_Pa'], nd=1)} Pa | {_fmt(new['p_total_heated_Pa'], nd=1)} Pa |",
        "",
        "Unheated residuals are the existing eval_candidate identities and are",
        "unchanged. The heated diagnostic uses the static exit state p2/T_ст, not",
        "the total state; total fields are listed only to keep the two states distinct.",
        "",
        "## Static heated exit continuity diagnostic (diagnostic identity only)",
        "",
        "`rho = p2/(R*T_ст)` [kg/m3], area `A = pi*D_л*l_л` [m2], velocities [m/s],",
        "flow [kg/s]; the reported signed error `rho*A*c2a/G_л - 1` is dimensionless",
        "(positive = the static heated density would carry more mass than delivered).",
        "With the existing identity `A = G_л*v2/(c2a*pi*D_л)` it equals `T2/T_ст - 1`.",
        "",
        "| case | stage | p2 Pa | T_ст K | rho kg/m3 | G_л kg/s | signed error | T2/T_ст-1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label in ("old", "new"):
        m = res["meanline"][label]
        for i in range(STAGE_COUNT):
            lines.append(
                f"| {label} | {i + 1} | {m['p_static_heated_Pa'][i]:.9f} | "
                f"{m['T_static_heated_K'][i]:.9f} | {m['rho_static_heated_kg_m3'][i]:.9f} | "
                f"{m['G_rotor_kg_s'][i]:.9f} | "
                f"{m['mass_static_heated_signed_rel_error'][i]:+.9e} | "
                f"{m['T2_K'][i] / m['T_static_heated_K'][i] - 1.0:+.9e} |")
    lines += [
        "",
        f"Heated fields finite and positive: **{_positive_fields_ok(old) and _positive_fields_ok(new)}**.",
        "This is a diagnostic identity at the meanline only; it is not evidence of",
        "radial validity and it adds no equation.",
        "",
        "## Efficiency difference",
        "",
        f"- eta old = {comp['eta_old']:.12f}",
        f"- eta new = {comp['eta_new']:.12f}",
        f"- delta absolute = {comp['delta_abs']:+.12f}",
        f"- delta percentage points = {comp['delta_points']:+.9f} pp",
        f"- delta relative = {comp['delta_rel_pct']:+.9f} %",
        "",
        "The frozen hz ratio is 0.75 here because both cases sit on their gap",
        "floors; that is not a 25 % total efficiency gain and no such claim is made.",
        "",
        "## Checks and tolerances",
        "",
        "| check | pass | detail | limit |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c['name']} | {c['pass']} | {c['detail']} | {c['limit']} |"
              for c in res["checks"]]
    lines += [
        "",
        "## Diagnostic project checks (not Category E)",
        "",
        "| constraint | pass | value | limit |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c['name']} | {c['pass']} | {c['detail']} | {c['limit']} |"
              for c in res["constraints"]]
    lines += ["", f"Feasible: **{res['feasible']}**", "", "## Unresolved / NOT_CHECKED", ""]
    lines += [f"- {item}" for item in res["not_checked"]]
    lines += ["", "## Provenance", ""]
    lines += [f"- `{key}`: {value}" for key, value in res["provenance"].items()]
    lines += [
        "",
        "Generated by `Optimizer/clearance_sensitivity.py` (no optimization, no",
        "nonlinear sweep, no new radial solve).",
        "",
    ]
    return "\n".join(lines)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_evidence(res, md_path=None, json_path=None):
    md_path = Path(md_path or RESULTS / "clearance_003_min015.md")
    json_path = Path(json_path or RESULTS / "clearance_003_min015.json")
    # Serialize before writing: NaN/Infinity evidence is rejected, not emitted.
    text = json.dumps(res, indent=2, ensure_ascii=False, allow_nan=False,
                      default=_json_default)
    md_path.write_text(_markdown(res), encoding="utf-8")
    json_path.write_text(text + "\n", encoding="utf-8")
    return md_path, json_path


def main():
    """Return 0 only on software PASS; engineering FAIL is reported, not a code."""
    result = run()
    md_path, json_path = write_evidence(result)
    print(json.dumps(result["comparison"], indent=2))
    print(json.dumps(result["classification"], indent=2))
    print(f"feasible: {result['feasible']}")
    print(md_path)
    print(json_path)
    return 0 if result["classification"]["software_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
