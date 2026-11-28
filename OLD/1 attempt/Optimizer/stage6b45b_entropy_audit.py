"""Stage 6B-4.5-B read-only absolute-entropy-offset audit (EXPERIMENTAL).

This script explains the signed terms behind the saved absolute cross-rotor
entropy residual ``E_s`` on the existing Stage 6B-4.5-B fields.  It does not
solve, re-optimize, add an unknown, free a boundary condition, change an
equation/constant/constraint, update a baseline, or decide existence.  The
original saved classification is untouched.

Run::

    python Optimizer/stage6b45b_entropy_audit.py

Outputs (new files only)::

    Optimizer/results/stage6b45b_entropy_audit.md
    Optimizer/results/stage6b45b_entropy_audit.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent.parent
OPTIMIZER = Path(__file__).resolve().parent
for _path in (str(OPTIMIZER), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import rotor_exit_coupled as rc  # noqa: E402
from turbine.radial import entropy_budget as eb  # noqa: E402
from turbine.radial import rotor_exit as core  # noqa: E402

RESULTS = OPTIMIZER / "results"
SAVED = RESULTS / "stage6b45b_results.json"
OUT_MD = RESULTS / "stage6b45b_entropy_audit.md"
OUT_JSON = RESULTS / "stage6b45b_entropy_audit.json"
SETUP_RTOL = 1e-12
SETUP_ATOL = 1e-12
BUDGET_ATOL = 1e-9
PROTECTED = [
    "turbine/radial/equilibrium.py",
    "turbine/radial/rotor_exit.py",
    "turbine/radial/__init__.py",
    "turbine/radial/entropy_budget.py",
    "Optimizer/rotor_exit_coupled.py",
    "Optimizer/stage6b45b_entropy_audit.py",
    "Optimizer/results/stage6b45b_report.md",
    "Optimizer/results/stage6b45b_results.json",
    "Optimizer/results/stage6b45b_runs.csv",
    "Optimizer/results/stage6b45b_stage_summary.csv",
    "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb",
    "Turbinis-Calc/mathcad/make_mathcad.py",
    "Optimizer/radial_exit.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes() -> dict:
    paths = [Path(p) for p in PROTECTED]
    paths.extend(sorted((ROOT / "reference" / "baselines").glob("*.json")))
    out = {}
    for path in paths:
        full = path if path.is_absolute() else ROOT / path
        out[str(full.relative_to(ROOT))] = (
            sha256_file(full) if full.is_file() else None
        )
    return out


def provenance() -> dict:
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_script": str(Path(__file__).resolve().relative_to(ROOT)),
        "saved_results": str(SAVED.relative_to(ROOT)),
    }


def convert_g(g: dict) -> dict:
    return {
        key: (np.asarray(value, dtype=float) if isinstance(value, list) else value)
        for key, value in g.items()
    }


def compare_setup(saved: dict, expected: dict) -> tuple[list, float]:
    """Compare one saved setup with a regenerated one at ~1e-12 relative."""
    mismatches = []
    max_rel = 0.0
    saved_keys, expected_keys = set(saved), set(expected)
    if saved_keys != expected_keys:
        mismatches.append({
            "key": "__keys__",
            "saved_only": sorted(saved_keys - expected_keys),
            "expected_only": sorted(expected_keys - saved_keys),
        })
    for key in sorted(saved_keys & expected_keys):
        left, right = saved[key], expected[key]
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left = np.asarray(left, dtype=float)
            right = np.asarray(right, dtype=float)
            if left.shape != right.shape:
                mismatches.append({"key": key, "reason": "shape",
                                   "saved": list(left.shape),
                                   "expected": list(right.shape)})
                continue
            diff = np.abs(left - right)
            rel = float(np.max(diff / np.maximum(np.abs(right), 1e-300)))
            max_rel = max(max_rel, rel)
            if not np.allclose(left, right, rtol=SETUP_RTOL, atol=SETUP_ATOL):
                mismatches.append({"key": key, "reason": "array",
                                   "max_abs": float(np.max(diff)),
                                   "max_rel": rel})
        elif isinstance(left, (int, float, np.integer, np.floating)) and isinstance(
            right, (int, float, np.integer, np.floating)
        ):
            left_f, right_f = float(left), float(right)
            rel = abs(left_f - right_f) / max(abs(right_f), 1e-300)
            max_rel = max(max_rel, rel)
            if not math.isclose(left_f, right_f, rel_tol=SETUP_RTOL,
                                abs_tol=SETUP_ATOL):
                mismatches.append({"key": key, "reason": "scalar",
                                   "saved": left_f, "expected": right_f,
                                   "max_rel": rel})
        else:
            if left != right:
                mismatches.append({"key": key, "reason": "value",
                                   "saved": left, "expected": right})
    return mismatches, max_rel


def iter_solves(out: dict):
    for stage_res in out["stages"]:
        for mesh in stage_res["meshes"]:
            yield stage_res, mesh, "mesh", f"mesh N={mesh['N']}"
        for label in ("standard", "loose"):
            solve = stage_res["tolerance"][label]
            yield stage_res, solve, "tolerance", f"tolerance/{label}"
        yield stage_res, stage_res["bc_hub"], "bc_hub", "bc_hub"


def verify_setups(out: dict, cp: dict) -> dict:
    cache = {}
    mismatches = []
    max_rel = 0.0
    checked = 0
    for stage_res, solve, case_type, case in iter_solves(out):
        stage = int(stage_res["stage"])
        key = (stage, int(solve["N"]), str(solve["bc"]))
        if key not in cache:
            st = cp["st"][stage - 1]
            expected = core.build_setup(
                st, cp["n"], stage_res["law"], cp["R"], cp["phi"],
                int(solve["N"]), bc=str(solve["bc"]),
                psi=float(solve["g"].get("psi", rc.PSI)),
            )
            expected["stage"] = stage
            cache[key] = expected
        problems, rel = compare_setup(convert_g(solve["g"]), cache[key])
        checked += 1
        max_rel = max(max_rel, rel)
        for problem in problems:
            mismatches.append({"stage": stage, "N": solve["N"],
                               "bc": solve["bc"], "case": case, **problem})
    return {
        "checked_solves": checked,
        "unique_setups": len(cache),
        "rtol": SETUP_RTOL,
        "atol": SETUP_ATOL,
        "max_relative_difference": max_rel,
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def endpoint_rows(out: dict, cp: dict) -> list[dict]:
    rows = []
    for stage_res, solve, case_type, case in iter_solves(out):
        stage = int(stage_res["stage"])
        st = cp["st"][stage - 1]
        g = convert_g(solve["g"])
        for run in solve["runs"]:
            valid = bool(run.get("valid"))
            closed = bool(rc._closed(run))
            row = {
                "stage": stage,
                "law": stage_res["law"],
                "case_type": case_type,
                "case": case,
                "N": int(solve["N"]),
                "bc": str(solve["bc"]),
                "start": run["start"],
                "start_kind": run.get("start_kind", "cold"),
                "success": bool(run.get("success")),
                "valid": valid,
                "closed": closed,
                "unfinished": bool(valid and not closed),
                "invalid": bool(not valid),
                "equation_residual_inf": run.get("equation_residual_inf"),
                "evaluated": False,
                "error": None,
            }
            y = run.get("y")
            if valid and y is not None:
                try:
                    f = core.fields(g, np.asarray(y, dtype=float))
                    budget = eb.entropy_budget(st, g, f)
                    row.update({
                        "evaluated": True,
                        "E_anchor": budget["absolute_entropy_residual"],
                        "budget_sum": budget["sum"],
                        "mismatch": budget["mismatch"],
                        "max_abs_E_s": budget["absolute"]["max_abs"],
                        "span": budget["absolute"]["span"],
                        "terms": budget["terms"],
                    })
                except Exception as exc:  # domain failure is recorded, not hidden
                    row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    evaluated = [row for row in rows if row["evaluated"]]
    max_mismatch = max((abs(row["mismatch"]) for row in evaluated), default=None)
    by_stage = {}
    for row in rows:
        item = by_stage.setdefault(row["stage"], {
            "total": 0, "valid": 0, "closed": 0, "unfinished": 0,
            "invalid": 0, "evaluated": 0,
        })
        item["total"] += 1
        item["valid"] += int(row["valid"])
        item["closed"] += int(row["closed"])
        item["unfinished"] += int(row["unfinished"])
        item["invalid"] += int(row["invalid"])
        item["evaluated"] += int(row["evaluated"])
    by_case = {}
    for row in rows:
        item = by_case.setdefault(row["case_type"], {
            "total": 0, "valid": 0, "closed": 0, "unfinished": 0,
            "invalid": 0, "evaluated": 0,
        })
        item["total"] += 1
        item["valid"] += int(row["valid"])
        item["closed"] += int(row["closed"])
        item["unfinished"] += int(row["unfinished"])
        item["invalid"] += int(row["invalid"])
        item["evaluated"] += int(row["evaluated"])
    return {
        "total_endpoints": len(rows),
        "evaluated_endpoints": len(evaluated),
        "not_evaluated_endpoints": len(rows) - len(evaluated),
        "max_abs_budget_reconstruction_error": max_mismatch,
        "budget_atol": BUDGET_ATOL,
        "within_atol": bool(max_mismatch is not None and max_mismatch <= BUDGET_ATOL),
        "by_stage": by_stage,
        "by_case_type": by_case,
    }


def stage_payload(stage_res: dict, cp: dict) -> dict:
    stage = int(stage_res["stage"])
    st = cp["st"][stage - 1]
    finest = stage_res["meshes"][-1]
    best = finest["best"]
    g = convert_g(finest["g"])
    y = np.asarray(best["y"], dtype=float)
    f = core.fields(g, y)
    budget = eb.entropy_budget(st, g, f)
    diagnostics = eb.book_meanline_diagnostics(st, R=cp["R"], psi=float(g["psi"]))
    return {
        "stage": stage,
        "law": stage_res["law"],
        "finest": {
            "N": int(finest["N"]),
            "bc": str(finest["bc"]),
            "start": best["start"],
            "start_kind": best.get("start_kind", "cold"),
            "valid": bool(best.get("valid")),
            "closed": bool(rc._closed(best)),
            "equation_residual_inf": best.get("equation_residual_inf"),
            "saved_classification": stage_res.get("classification"),
        },
        "budget": budget,
        "book_meanline": diagnostics,
    }


def _fmt(value, digits=6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "—"
        return f"{float(value):.{digits}g}"
    return str(value)


def _signed(value, digits=6) -> str:
    if value is None:
        return "—"
    return f"{float(value):+.{digits}e}"


def write_report(payload: dict, rows: list[dict], summary: dict, path: Path) -> None:
    lines = []
    lines.append("# Stage 6B-4.5-B — absolute entropy-offset attribution\n")
    lines.append("Engineering status: **EXPERIMENTAL**. This is a read-only "
                 "diagnostic of saved fields. It is not a closure, a new solver, "
                 "or a new existence decision. The original all-INCONCLUSIVE "
                 "classification is unchanged.\n")

    lines.append("## Scope and source\n")
    lines.append("The ideal-gas state entropy "
                 "`s - s_ref = cp ln(T/T_ref) - R ln(p/p_ref)` is **ACCEPTED** "
                 "in `docs/engineering/EQUATIONS.md`. The inherited rotor "
                 "energy relation and the `psi = 0.96` loss entropy are "
                 "**EXPERIMENTAL/`book_approx`**. The meanline pressure is an "
                 "inherited book approximation. No primary source-page "
                 "applicability data (Mach, Reynolds, geometry, validity range) "
                 "is available; this is a disclosed limitation.\n")
    lines.append("Book source: `Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb` "
                 "cell 12, points 90–93 and 107; "
                 "`Turbinis-Calc/mathcad/make_mathcad.py` lines 296–325. "
                 "The relations are evaluated as inherited inputs, not "
                 "re-derived or tuned.\n")
    lines.append("Equation count: unchanged. The saved coupled system still has "
                 "`N` unknowns and `N+1` compatibility conditions "
                 "(`N-1` momentum + 1 mass + 1 absolute cross-rotor entropy). "
                 "This audit adds no unknown and frees no boundary condition.\n")

    lines.append("## Units, domains, tolerances\n")
    lines.append("- SI units: `T [K]`, `p [Pa]`, `cp, R [J/(kg K)]`, "
                 "`w2 [m/s]`, entropy terms `J/(kg K)`.\n")
    lines.append("- Valid domain: finite positive `T`, `p`; `cp > 0`, `R > 0`; "
                 "`0 < psi <= 1`; `h_l >= 0`; conditional log `0 <= x < 1`, "
                 "`x = h_l/(cp T2)`.\n")
    lines.append(f"- Budget reconstruction: `|sum of terms - core "
                 f"absolute_entropy_residual(anchor)| <= {BUDGET_ATOL:.0e} "
                 "J/(kg K)` (algebraic identity, not model uncertainty).\n")
    lines.append(f"- Book identity checks: energy identity "
                 f"{eb.ENERGY_IDENTITY_ABS_TOL_K:.0e} K absolute / "
                 f"{eb.ENERGY_IDENTITY_REL_TOL:.0e} relative; "
                 f"`cp/R` identity {eb.CP_R_IDENTITY_REL_TOL:.0e} relative. "
                 "These are floating-point compatibility caps, not physical "
                 "acceptance bands.\n")

    prov = payload["provenance"]
    lines.append("## Provenance and input integrity\n")
    lines.append(f"Python {prov['python']}; NumPy {prov['numpy']}; "
                 f"SciPy {prov['scipy']}; platform {prov['platform']}.\n")
    lines.append(f"Saved results SHA-256: `{payload['input_hashes']['Optimizer/results/stage6b45b_results.json']}`.\n")
    setup = payload["setup_integrity"]
    lines.append(f"Setup regeneration check: **{_fmt(setup['passed'])}**; "
                 f"{setup['unique_setups']} unique setups and "
                 f"{setup['checked_solves']} saved solves compared at "
                 f"rtol/atol {setup['rtol']:.0e}/{setup['atol']:.0e}; "
                 f"max relative difference `{setup['max_relative_difference']:.3e}`. "
                 "No mixed-history fields were used.\n")
    if setup["mismatches"]:
        lines.append("Setup mismatches:\n")
        for mismatch in setup["mismatches"]:
            lines.append(f"- `{mismatch}`\n")

    cp = payload["control_point"]
    lines.append("## Control point\n")
    lines.append(f"n = {cp['n']:.0f} rpm; D_hub = {cp['D_hub'] * 1e3:.0f} mm; "
                 f"w1 = {cp['w1']}; w2 = {cp['w2']}; "
                 f"kappa2 = {cp['kappa2']}; kappa3 = {cp['kappa3']}; "
                 f"alpha1_stage1 = {cp['alpha1_stage1']:.2f} deg; "
                 f"R = {cp['R']} J/(kg K); phi = {cp['phi']}.\n")
    lines.append("Laws: " + ", ".join(
        f"stage {item['stage']}: {item['law']}" for item in payload["stages"]
    ) + ".\n")

    lines.append("## Signed entropy budget at the meanline anchor\n")
    lines.append("Definitions (all `J/(kg K)`):\n")
    lines.append("`E_s = cp ln(T2/T1) - R ln(p2/p1) - ds_l`.\n")
    lines.append("At anchor `a`:\n")
    lines.append("`E_s = E_book + inlet_state_shift + exit_temperature_shift "
                 "+ exit_pressure_shift + loss_shift`,\n")
    lines.append("`E_book = cp ln(T2*/T1*) - R ln(p2*/p1*) - h_l*/T2*`,\n")
    lines.append("`inlet_state_shift = -cp ln(T1_a/T1*) + R ln(p1_a/p1*)`, "
                 "`exit_temperature_shift = cp ln(T2_a/T2*)`, "
                 "`exit_pressure_shift = -R ln(p2_a/p2*)`, "
                 "`loss_shift = -(ds_l_a - h_l*/T2*)`.\n")
    lines.append("| stage | E_book | inlet | exit T | exit p | loss | sum | E_s | mismatch | span | max\\|E_s\\| |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for item in payload["stages"]:
        b = item["budget"]
        t = b["terms"]
        lines.append(
            f"| {item['stage']} | {_signed(t['book_offset'])} | "
            f"{_signed(t['inlet_state_shift'])} | "
            f"{_signed(t['exit_temperature_shift'])} | "
            f"{_signed(t['exit_pressure_shift'])} | "
            f"{_signed(t['loss_shift'])} | {_signed(b['sum'])} | "
            f"{_signed(b['absolute_entropy_residual'])} | "
            f"{_signed(b['mismatch'])} | {_signed(b['absolute']['span'])} | "
            f"{_fmt(b['absolute']['max_abs'], 8)} |"
        )
    lines.append("")

    lines.append("## Finest saved endpoint and input ratios\n")
    lines.append("| stage | N | start | kind | valid | closed | eq res max\\|f\\| |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in payload["stages"]:
        f = item["finest"]
        lines.append(
            f"| {item['stage']} | {f['N']} | {f['start']} | "
            f"{f['start_kind']} | {_fmt(f['valid'])} | {_fmt(f['closed'])} | "
            f"{_fmt(f['equation_residual_inf'], 3)} |"
        )
    lines.append("")
    lines.append("| stage | T1_a/T1* | p1_a/p1* | T2_a/T2* | p2_a/p2* | ds_l_a/(h_l*/T2*) | c2a_a/c2a* | w2_a/w2* |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for item in payload["stages"]:
        r = item["budget"]["ratios"]
        lines.append(
            f"| {item['stage']} | {_fmt(r['T1_a_over_book_T1'])} | "
            f"{_fmt(r['p1_a_over_book_p1'])} | "
            f"{_fmt(r['T2_a_over_book_T2'])} | "
            f"{_fmt(r['p2_a_over_book_p2'])} | "
            f"{_fmt(r['dsl_a_over_book_ds_l'])} | "
            f"{_fmt(r.get('c2a_a_over_book_c2a'))} | "
            f"{_fmt(r.get('w2_a_over_book_w2'))} |"
        )
    lines.append("")
    lines.append("Signed term interpretation is algebraic. `exit_pressure_shift` "
                 "is zero at the primary mean anchor because the saved boundary "
                 "condition pins `p2(mean) = p2*`; this is a software assumption, "
                 "not physical evidence. Inlet ratios identical to one at the "
                 "mean anchor are likewise a construction observation.\n")

    lines.append("## Inherited meanline identities\n")
    lines.append("`energy_resid = (T2 - T2_ad) - h_l/cp` [K]. "
                 "`conditional` uses `x = h_l/(cp T2)` and is marked "
                 "NOT_APPLICABLE when its assumptions fail; no invalid log is "
                 "evaluated.\n")
    lines.append("| stage | h_l [J/kg] | h_l key diff | energy_resid [K] | cp/R diff | p2 recon rel. | ds_state [J/(kg K)] | h_l/T2 [J/(kg K)] | conditional | diff = -cp ln(1-x)-h_l/T2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for item in payload["stages"]:
        d = item["book_meanline"]
        cond = d["conditional_log"]
        cp_r = d["cp_over_R"]
        recon = d["pressure_reconstruction"]
        lines.append(
            f"| {item['stage']} | {_fmt(d['hl']['value'], 8)} | "
            f"{_fmt(d['hl']['key_abs_difference'], 3)} | "
            f"{_fmt(d['energy_identity_residual_K'], 3)} | "
            f"{_fmt(cp_r.get('relative_difference'), 3)} | "
            f"{_fmt(recon.get('relative_residual'), 3)} | "
            f"{_fmt(d['ds_state_book'], 10)} | {_fmt(d['ds_l'], 10)} | "
            f"{cond['status']} | {_fmt(cond['difference'], 8)} |"
        )
    lines.append("")
    lines.append("`h_l key diff` is `h_l(from psi, w2) - h_l(stored key)` when "
                 "`h_л`/`h_l` is present. For the saved control point it is "
                 "floating-point negligible.\n")

    lines.append("## All saved endpoints (72)\n")
    lines.append(f"Audited: {summary['total_endpoints']} endpoints; "
                 f"evaluated: {summary['evaluated_endpoints']}; "
                 f"not evaluated: {summary['not_evaluated_endpoints']}. "
                 f"Max absolute budget reconstruction error: "
                 f"`{_signed(summary['max_abs_budget_reconstruction_error'])}` "
                 f"J/(kg K) (atol {summary['budget_atol']:.0e}).\n")
    lines.append("| stage | total | valid | closed | unfinished | invalid | evaluated |")
    lines.append("|---|---|---|---|---|---|---|")
    for stage, item in sorted(summary["by_stage"].items()):
        lines.append(
            f"| {stage} | {item['total']} | {item['valid']} | {item['closed']} | "
            f"{item['unfinished']} | {item['invalid']} | {item['evaluated']} |"
        )
    lines.append("")
    lines.append("| case type | total | valid | closed | unfinished | invalid | evaluated |")
    lines.append("|---|---|---|---|---|---|---|")
    for case, item in sorted(summary["by_case_type"].items()):
        lines.append(
            f"| {case} | {item['total']} | {item['valid']} | {item['closed']} | "
            f"{item['unfinished']} | {item['invalid']} | {item['evaluated']} |"
        )
    lines.append("")
    lines.append("`closed` means `valid` and "
                 f"`equation_residual_inf <= {rc.SOLVER_CLOSURE_TOL:.0e}`; "
                 "`unfinished` means valid but not closed; `invalid` means the "
                 "saved run was not valid. Unfinished and invalid runs are not "
                 "used as existence evidence.\n")

    for item in payload["stages"]:
        stage = item["stage"]
        lines.append(f"### Stage {stage} ({item['law']}) mesh/start/tolerance/BC table\n")
        lines.append("| case | N | bc | start | kind | valid | closed | unfinished | eq res | E_anchor | E_book | inlet | exit T | exit p | loss | mismatch |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for row in rows:
            if row["stage"] != stage:
                continue
            terms = row.get("terms") or {}
            lines.append(
                f"| {row['case']} | {row['N']} | {row['bc']} | {row['start']} | "
                f"{row['start_kind']} | {_fmt(row['valid'])} | {_fmt(row['closed'])} | "
                f"{_fmt(row['unfinished'])} | {_fmt(row['equation_residual_inf'], 3)} | "
                f"{_signed(row.get('E_anchor'))} | "
                f"{_signed(terms.get('book_offset'))} | "
                f"{_signed(terms.get('inlet_state_shift'))} | "
                f"{_signed(terms.get('exit_temperature_shift'))} | "
                f"{_signed(terms.get('exit_pressure_shift'))} | "
                f"{_signed(terms.get('loss_shift'))} | "
                f"{_signed(row.get('mismatch'))} |"
            )
        if any(row["stage"] == stage and row["error"] for row in rows):
            lines.append("")
            for row in rows:
                if row["stage"] == stage and row["error"]:
                    lines.append(f"- not evaluated: `{row['case']}` / "
                                 f"`{row['start']}`: {row['error']}\n")
        lines.append("")

    lines.append("## Interpretation\n")
    lines.append("The decomposition is diagnostic only. It does not assign a "
                 "true physical cause to the residual and it does not average "
                 "alternative closures. On the finest saved primary endpoints:\n")
    for item in payload["stages"]:
        b = item["budget"]
        t = b["terms"]
        dominant = max(t, key=lambda key: abs(t[key]))
        lines.append(
            f"- Stage {item['stage']}: `{dominant}` has the largest absolute "
            f"term ({_signed(t[dominant])} J/(kg K)); `E_book` is "
            f"{_signed(t['book_offset'])} J/(kg K) while `E_s` is "
            f"{_signed(b['absolute_entropy_residual'])} J/(kg K). "
            f"The sum of terms reproduces `E_s` to {_signed(b['mismatch'])} "
            "J/(kg K)."
        )
    lines.append("")
    lines.append("Across the three primary stages `book_offset` is positive and "
                 "small compared with the negative `E_s`; `exit_temperature_shift` "
                 "is negative and dominant; `exit_pressure_shift` is zero at the "
                 "mean anchor because of the prescribed pressure boundary "
                 "condition; `inlet_state_shift` is zero at the mean anchor; and "
                 "`loss_shift` is a negative secondary term. Therefore the "
                 "hypothesis that the book loss/energy approximation alone "
                 "explains the offset is **not sufficient** on these saved fields. "
                 "The dominant term is the difference between the radial-solution "
                 "exit temperature at the anchor and the book meanline `T2`, "
                 "which follows from the solved axial scale and radial field, not "
                 "from a new closure. This statement is an observation, not a "
                 "license to modify the meanline or radial values.\n")
    lines.append("Hub-anchor sensitivity endpoints show different offsets and are "
                 "reported in the JSON endpoint rows. They are sensitivity cases, "
                 "not selected design evidence and not an alternative closure.\n")

    lines.append("## Tests and limitations\n")
    lines.append("- `tests/optimization/test_entropy_budget.py` checks exact "
                 "synthetic budget reconstruction, each perturbed term "
                 "(including pressure), zero loss at `psi=1`, small-`x` "
                 "logarithmic comparison, domain rejection (`x>=1`, non-positive "
                 "`p/T`), conditional meanline applicability, and the saved "
                 "three-stage identity when artifacts are present.\n")
    lines.append("- No new nonlinear sweep, no mesh refinement, and no new "
                 "initial guesses were run. Saved solver closure and labels are "
                 "as recorded; no run is upgraded to existence evidence.\n")
    lines.append("- The book meanline pressure and loss are inherited "
                 "`book_approx`; their radial applicability and primary source "
                 "page are not established. The conditional log identity is "
                 "reported only when `T2 - T2_ad = h_l/cp` holds numerically.\n")
    lines.append("- Not checked: primary source-page applicability and validity "
                 "range; independent radial closure; stator-exit closure; "
                 "temperature strength, creep/resource, thermal stress, cooling, "
                 "stage-3 root reaction, and real blade profiles. No mass/angle "
                 "pass or existence claim is made.\n")
    lines.append("- The `book_offset` positive values are consistent with the "
                 "conditional identity `-cp ln(1-x) - h_l/T2 = cp x^2/2 + ...`, "
                 "which is positive; they are not evidence that the experimental "
                 "loss model is accepted.\n")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    out_dir = args.out_dir
    out_md = out_dir / "stage6b45b_entropy_audit.md"
    out_json = out_dir / "stage6b45b_entropy_audit.json"

    if not SAVED.is_file():
        raise SystemExit(f"missing saved results: {SAVED}")
    saved_text = SAVED.read_text(encoding="utf-8")
    out = json.loads(saved_text)
    cp_resolved = rc.control_point()
    cp = {
        "n": float(cp_resolved["n"]),
        "R": float(cp_resolved["R"]),
        "phi": float(cp_resolved["phi"]),
        "alpha1_stage1": float(cp_resolved["alpha1_stage1"]),
        "D_hub": 0.062,
        "w1": 0.70,
        "w2": 0.8125,
        "kappa2": 1.04375,
        "kappa3": 1.25,
    }

    setup = verify_setups(out, cp_resolved)
    if not setup["passed"]:
        raise SystemExit(
            "saved setup drift: regenerated g does not match saved g; "
            "refusing to mix histories: " + json.dumps(setup["mismatches"][:5])
        )

    rows = endpoint_rows(out, cp_resolved)
    summary = summarize_rows(rows)
    stages = [stage_payload(stage_res, cp_resolved) for stage_res in out["stages"]]

    payload = {
        "diagnostic": "stage6b45b_entropy_audit",
        "status": "EXPERIMENTAL",
        "classification_effect": "none; original saved classification unchanged",
        "existing_question_count": {
            "unknowns": "N",
            "compatibility_conditions": "N+1 (N-1 momentum + 1 mass + 1 absolute entropy)",
            "added_unknowns": 0,
            "added_equations": 0,
        },
        "source": {
            "ideal_gas_entropy": "docs/engineering/EQUATIONS.md (ACCEPTED)",
            "rotor_energy_and_loss": "EXPERIMENTAL/book_approx",
            "notebook": "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb cell 12, points 90-93 and 107",
            "mathcad_generator": "Turbinis-Calc/mathcad/make_mathcad.py lines 296-325",
            "primary_source_page_applicability": "not available",
        },
        "units": {
            "temperature": "K",
            "pressure": "Pa",
            "entropy_terms": "J/(kg K)",
            "specific_energy": "J/kg",
            "velocity": "m/s",
        },
        "tolerances": {
            "budget_reconstruction_atol": BUDGET_ATOL,
            "setup_rtol": SETUP_RTOL,
            "setup_atol": SETUP_ATOL,
            "energy_identity_abs_tol_K": eb.ENERGY_IDENTITY_ABS_TOL_K,
            "energy_identity_rel_tol": eb.ENERGY_IDENTITY_REL_TOL,
            "cp_over_R_identity_rel_tol": eb.CP_R_IDENTITY_REL_TOL,
            "solver_closure_tol": rc.SOLVER_CLOSURE_TOL,
        },
        "provenance": provenance(),
        "input_hashes": input_hashes(),
        "control_point": cp,
        "setup_integrity": setup,
        "stages": stages,
        "endpoints": {
            "summary": summary,
            "rows": rows,
        },
        "interpretation": {
            "book_approximation_alone_sufficient": False,
            "dominant_term_by_stage": {
                str(item["stage"]): max(
                    item["budget"]["terms"],
                    key=lambda key: abs(item["budget"]["terms"][key]),
                )
                for item in stages
            },
            "primary_mean_anchor_pressure_term_note": (
                "exit_pressure_shift is zero because p2(mean)=p2* is the saved "
                "software boundary condition"
            ),
            "no_new_existence_decision": True,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_report(payload, rows, summary, out_md)
    out_json.write_text(
        json.dumps(rc._jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("stage  E_s [J/(kg K)]   E_book      exitT       loss       max|mismatch|")
    for item in stages:
        b = item["budget"]
        t = b["terms"]
        print(f"{item['stage']:>5}  {b['absolute_entropy_residual']:+.10f}  "
              f"{t['book_offset']:+.6e}  {t['exit_temperature_shift']:+.6e}  "
              f"{t['loss_shift']:+.6e}  {summary['max_abs_budget_reconstruction_error']:.3e}")
    print("endpoints:", summary["total_endpoints"],
          "evaluated:", summary["evaluated_endpoints"])
    print("report:", out_md)
    print("json:  ", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
