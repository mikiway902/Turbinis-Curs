#!/usr/bin/env python3
"""Read-only cross-run analysis/plots for completed source-sized RAW stage sweeps.

Calls the independent v3 auditor and reads each RAW journal once; never reruns the
meanline or mutates RAW. Root rows are deduplicated by physical candidate id with
source memberships preserved; failures stay per source. Missing metrics are null/
NOT_CHECKED, never zero. This is FOUR_PROFILE_SCREENING_NOT_OVERALL_FEASIBILITY:
not ranking, Pareto, best-design, production or full-feasibility evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
from typing import Any, Mapping
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path: sys.path.insert(0, str(_path))
import audit_sized_stage_roots as auditor  # noqa: E402
from turbine_engine.dataset import DatasetError, read_records  # noqa: E402
ANALYSIS_VERSION, SCOPE = "sized-stage-sweep-analysis-v1", "FOUR_PROFILE_SCREENING_NOT_OVERALL_FEASIBILITY"
CONSTRAINTS = ("rho_065_upper", "prompt_pi_1_8_2_4", "alpha2_source_page_38", "sigma_project_290MPa")
DOMAIN, STATUSES = "reaction_table_lD_domain", ("SUCCESS", "NUMERICAL_FAILURE", "NON_PHYSICAL_STATE", "MODEL_ERROR")
TARGET_W, POWER_REL_TOL = 110000.0, 1e-6
OVERALL_REASON = "maps_cooling_material_radial_resources_not_validated"
NOT_CHECKED = {"compressor_maps_and_stage_split": "NOT_CHECKED", "cooling_and_bleed_semantics": "NOT_CHECKED",
    "material_allowables_metal_temperature_thermal_stress": "NOT_CHECKED", "3D_radial_equilibrium_and_secondary_flows": "NOT_CHECKED",
    "creep_and_resource_life": "NOT_CHECKED", "production_design_space_coverage": "NOT_CHECKED"}
ARCH_COLORS = {"SINGLE_SHAFT_SINGLE_CC": "#1f77b4", "SINGLE_SHAFT_DUAL_CC": "#d62728", "FREE_POWER_TURBINE": "#2ca02c"}
ROLE_MARKERS = {"main": "o", "gas_generator": "s", "free_power": "^"}
class AnalysisError(RuntimeError): pass
def _need(ok: Any, message: str) -> None:
    if not ok: raise AnalysisError(message)
def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)): return None
    value = float(value); return value if math.isfinite(value) else None
def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()
def _bump(table: dict[str, Any], key: str) -> None: table[key] = table.get(key, 0) + 1
def _status_counts(value: Any) -> dict[str, int]:
    value = value if isinstance(value, Mapping) else {}; return {key: int(value.get(key, 0)) for key in STATUSES}
def _sum_status_counts(values: list[Mapping[str, Any]]) -> dict[str, int]:
    return {key: sum(int(item.get(key, 0)) for item in values) for key in STATUSES}
def _ratios(payload: Mapping[str, Any]) -> list[float] | None:
    source = payload.get("architecture_cycle_source") if isinstance(payload.get("architecture_cycle_source"), Mapping) else {}; raw = source.get("compressor_pressure_ratios")
    values = [_num(item) for item in raw] if isinstance(raw, (list, tuple)) and raw else None
    return None if values is None or any(item is None for item in values) else values
def _stage_metrics(root: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    stages, by_role = [], {}
    for role, group in sorted((root.get("groups") or {}).items()):
        for stage in (group.get("stages") if isinstance(group, Mapping) else None) or []:
            metrics = stage.get("metrics") if isinstance(stage, Mapping) else None
            if isinstance(metrics, Mapping): stages.append(metrics); by_role.setdefault(role, []).append(metrics)
    def peak(seq, key, pick=max):
        values = [v for v in (_num(item.get(key)) for item in seq) if v is not None]; return pick(values) if values else None
    def ext(seq): return {"peak_sigma_MPa": peak(seq, "sigma_MPa"), "max_M_c1": peak(seq, "M_c1"), "max_M_w2": peak(seq, "M_w2"), "min_l_l_m": peak(seq, "l_l_m", min), "min_D_l_m": peak(seq, "D_l_m", min)}
    return ext(stages), {role: ext(seq) for role, seq in by_role.items()}
def _effective_controls(root: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for role, group in sorted((root.get("groups") or {}).items()):
        closure = group.get("closure") if isinstance(group, Mapping) else None
        raw = closure.get("controls") if isinstance(closure, Mapping) else None; raw = raw if isinstance(raw, Mapping) else {}; speed, hub = _num(raw.get("n_rpm")), _num(raw.get("D_hub_m"))
        out[role] = {"n_rpm": speed, "D_hub_m": hub, "status": "OBSERVED" if speed is not None and hub is not None else "NOT_CHECKED",
            "basis": "ACTUAL_CLOSURE_CONTROLS_NOT_SOURCE_ANCHOR"}
    return out
def _net_power(root: Mapping[str, Any]) -> dict[str, Any]:
    groups = root.get("groups") or {}; role = "free_power" if "free_power" in groups else "main"
    group = groups.get(role) if isinstance(groups.get(role), Mapping) else {}; power = group.get("power") if isinstance(group.get("power"), Mapping) else {}
    net, target = _num(power.get("net_mechanical_power_W")), _num(power.get("target_W"))
    target = TARGET_W if target is None else target
    signed, relative = _num(power.get("signed_power_residual_W")), _num(power.get("relative_power_residual"))
    if signed is None and net is not None: signed = net - target
    if relative is None and signed is not None and target: relative = abs(signed) / abs(target)
    closed = power.get("engine_power_closed")
    status = "PASS" if closed is True else "FAIL" if closed is False else "NOT_CHECKED" if net is None else ("PASS" if relative is not None and relative <= POWER_REL_TOL else "FAIL")
    return {"role": role, "target_W": target if target else None, "net_mechanical_power_W": net, "signed_residual_W": signed,
        "relative_residual": relative, "numerical_power_closed": closed if isinstance(closed, bool) else None, "status": status}
def _profile_margins(root: Mapping[str, Any]) -> dict[str, Any]:
    constraints = root.get("constraints") if isinstance(root.get("constraints"), Mapping) else {}; out = {}
    for key in CONSTRAINTS:
        item = constraints.get(key) if isinstance(constraints.get(key), Mapping) else {}
        out[key] = {"status": item.get("status") if isinstance(item.get("status"), str) else "NOT_CHECKED", "min_signed_margin": _num(item.get("min_signed_margin"))}
    domain = root.get("source_domain") if isinstance(root.get("source_domain"), Mapping) else {}
    out[DOMAIN] = {"status": domain.get("status") if isinstance(domain.get("status"), str) else "NOT_CHECKED",
        "min_source_domain_margin": _num(domain.get("min_source_domain_margin"))}
    return out
def _root_row(root: Mapping[str, Any], record: Any, run_id: str) -> dict[str, Any]:
    payload = record.payload if isinstance(record.payload, Mapping) else {}
    ratios = _ratios(payload); product = math.prod(ratios) if ratios else None
    weight = math.log(ratios[0]) / math.log(product) if ratios is not None and len(ratios) == 2 and product is not None and product > 1.0 and min(ratios) > 0.0 else None
    architecture = root.get("architecture") or record.architecture
    mass = root.get("mass_fuel_air") if isinstance(root.get("mass_fuel_air"), Mapping) else {}
    extremes, role_extremes = _stage_metrics(root); margins = _profile_margins(root); net = _net_power(root); numerical = root.get("numerical_closed")
    role_alpha = {}
    for role, group in sorted((root.get("groups") or {}).items()):
        values = [_num(((stage.get("checks") or {}).get("alpha2_source_page_38") or {}).get("margin")) for stage in (group.get("stages") or []) if isinstance(stage, Mapping)]
        observed = bool(values) and all(value is not None for value in values); role_alpha[role] = {"min_margin_deg": min(values) if observed else None, "status": "OBSERVED" if observed else "NOT_CHECKED"}
    screening = bool(all(margins[key]["status"] == "PASS" for key in CONSTRAINTS) and margins[DOMAIN]["status"] == "WITHIN_SOURCE_DOMAIN" and numerical is True)
    available = {"pi_product": product is not None, "compressorRR": ratios is not None, "air_mass_flow_kg_s": _num(mass.get("air_mass_flow_kg_s")) is not None,
        "fuel_mass_flow_kg_s": _num(mass.get("fuel_mass_flow_kg_s")) is not None, "gas_mass_flow_kg_s": _num(mass.get("gas_mass_flow_kg_s")) is not None,
        "net_mechanical_power_W": net["net_mechanical_power_W"] is not None, "peak_sigma_MPa": extremes["peak_sigma_MPa"] is not None, "max_M_c1": extremes["max_M_c1"] is not None,
        "max_M_w2": extremes["max_M_w2"] is not None, "min_l_l_m": extremes["min_l_l_m"] is not None, "source_domain": margins[DOMAIN]["status"] != "NOT_CHECKED",
        "numerical_closed": numerical is not None, "numerical_power_closed": root.get("numerical_power_closed") is not None}
    row = {"candidate_id": root.get("root_candidate_id"), "source_run_ids": [run_id], "source_cases": [{"run_id": run_id, "case": root.get("case")}],
        "architecture": architecture, "stage_counts": payload.get("stage_counts"), "pi_product": product,
        "compressorRR": {"unit_pressure_ratios": ratios, "first_over_second": ratios[0] / ratios[1] if ratios is not None and len(ratios) == 2 and ratios[1] else None, "status": "OBSERVED" if ratios else "NOT_CHECKED"},
        "logsplit_weight_A": weight if architecture == "SINGLE_SHAFT_DUAL_CC" else None,
        "logsplit_weight_A_status": ("OBSERVED" if weight is not None else "NOT_CHECKED") if architecture == "SINGLE_SHAFT_DUAL_CC" else "NOT_APPLICABLE",
        "actual_sized_flows": {"air_mass_flow_kg_s": _num(mass.get("air_mass_flow_kg_s")), "fuel_mass_flow_kg_s": _num(mass.get("fuel_mass_flow_kg_s")),
            "gas_mass_flow_kg_s": _num(mass.get("gas_mass_flow_kg_s")), "fuel_air_ratio": _num(mass.get("fuel_air_ratio")),
            "mass_balance_residual_kg_s": _num(mass.get("mass_balance_residual_kg_s")), "status": mass.get("status") if isinstance(mass.get("status"), str) else "NOT_CHECKED"},
        "net_power": net, "profile_margins": margins, "source_domain_status": margins[DOMAIN]["status"],
        "stage_extremes": extremes, "stage_extremes_by_role": role_extremes, "per_role_alpha2_margin": role_alpha, "effective_controls": _effective_controls(root),
        "numerical_closed": numerical, "numerical_power_closed": root.get("numerical_power_closed"),
        "known_constraint_violation": any(margins[key]["status"] == "FAIL" for key in CONSTRAINTS),
        "four_profile_screening_pass": screening, "screening_scope": SCOPE,
        "available_metrics": available, "metric_status": {key: "OBSERVED" if ok else "NOT_CHECKED" for key, ok in available.items()}, "not_checked_metrics": [key for key, ok in available.items() if not ok],
        "residuals": {key: _num(value) for key, value in (root.get("reported_residuals") or {}).items()} if isinstance(root.get("reported_residuals"), Mapping) else {},
        "overall_engineering_feasibility": None, "overall_engineering_feasibility_status": "NOT_CHECKED"}
    row["physical_signature_sha256"] = _digest({key: value for key, value in row.items() if key not in {"source_run_ids", "source_cases", "physical_signature_sha256"}})
    return row
def _failed_row(case: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    states = case.get("evaluated_states") if isinstance(case.get("evaluated_states"), list) else []; last = states[-1] if states and isinstance(states[-1], Mapping) else {}
    return {"source_run_id": run_id, "case": case.get("case"), "status": case.get("status"), "reason": case.get("reason"),
        "architecture": case.get("architecture"), "source_point_index": case.get("source_point_index"), "trial_count": len(states),
        "evaluated_state_ids": [state.get("candidate_id") for state in states if isinstance(state, Mapping)],
        "last_state": {"candidate_id": last.get("candidate_id"), "air_mass_flow_kg_s": _num(last.get("air_mass_flow_kg_s")), "calculation_status": last.get("calculation_status"),
            "net_mechanical_power_W": _num(last.get("net_mechanical_power_W")), "signed_power_residual_W": _num(last.get("signed_power_residual_W")), "relative_power_residual": _num(last.get("relative_power_residual"))},
        "inner_power_residual_note": "EXPECTED_FLOW_SIZING_TRACE_NOT_ENGINE_ERROR", "four_profile_screening_pass": False,
        "overall_engineering_feasibility": None, "overall_engineering_feasibility_status": "NOT_CHECKED", "interpretation": "TYPED_OUTER_FAILURE_NOT_PHYSICAL_INFEASIBILITY"}
def _merge_root(pool: dict[str, dict[str, Any]], row: dict[str, Any], run_id: str, case: Any) -> None:
    identifier = row["candidate_id"]; _need(isinstance(identifier, str) and identifier, "root candidate id is missing")
    membership = {"run_id": run_id, "case": case}
    if identifier not in pool: pool[identifier] = row; return
    existing = pool[identifier]
    _need(existing["physical_signature_sha256"] == row["physical_signature_sha256"], f"root candidate {identifier} has conflicting physical metrics across sources; refusing to merge")
    if run_id not in existing["source_run_ids"]: existing["source_run_ids"].append(run_id)
    if membership not in existing["source_cases"]: existing["source_cases"].append(membership)
def _control_values(rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    table = {}
    for row in rows:
        for role, item in row["effective_controls"].items():
            speed, hub = item["n_rpm"], item["D_hub_m"]
            if speed is not None and hub is not None: table.setdefault(role, {}).setdefault((speed, hub), []).append(row["candidate_id"])
    return {role: [{"n_rpm": speed, "D_hub_m": hub, "root_candidate_ids": sorted(ids)} for (speed, hub), ids in sorted(items.items())] for role, items in sorted(table.items())}
def _failure_minima(rows: list[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    counts, margins = {}, {}
    for row in rows:
        for key, item in row["profile_margins"].items():
            margin = item.get("min_signed_margin", item.get("min_source_domain_margin"))
            if item["status"] in ("FAIL", "OUTSIDE_SOURCE_DOMAIN"): _bump(counts.setdefault(row["architecture"], {}), key)
            if margin is not None:
                table = margins.setdefault(row["architecture"], {}); table[key] = margin if key not in table else min(table[key], margin)
    return counts, margins
def _residual_maxima(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    maxima = {}
    for row in rows:
        for key, value in row["residuals"].items():
            if value is not None: maxima[key] = max(maxima.get(key, 0.0), abs(value))
    return maxima
def _numerical_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"numerical_closed_true": sum(r["numerical_closed"] is True for r in rows), "numerical_closed_false": sum(r["numerical_closed"] is False for r in rows),
        "numerical_closed_not_checked": sum(r["numerical_closed"] is None for r in rows), "power_only_true": sum(r["numerical_power_closed"] is True for r in rows),
        "power_only_false": sum(r["numerical_power_closed"] is False for r in rows), "power_only_not_checked": sum(r["numerical_power_closed"] is None for r in rows),
        "note": "numericalClosed requires power+pressure+mass+continuity; powerOnly is power closure alone"}
def _source_report(path: Path, audit: Mapping[str, Any], records: tuple[Any, ...]) -> tuple[dict[str, Any], list[tuple[dict[str, Any], Any]], list[dict[str, Any]]]:
    by_id = {record.candidate_id: record for record in records}
    _need(len(by_id) == len(records), f"duplicate candidate ids in RAW: {path}")
    run = audit.get("run") if isinstance(audit.get("run"), Mapping) else {}; run_id = run.get("run_id")
    _need(isinstance(run_id, str) and run_id, f"audit run_id missing: {path}")
    roots = []
    for root in audit.get("roots") or []:
        identifier = root.get("root_candidate_id"); _need(identifier in by_id, f"audited root {identifier!r} is missing from RAW: {path}")
        roots.append((_root_row(root, by_id[identifier], run_id), root.get("case")))
    failed = [_failed_row(case, run_id) for case in audit.get("failed_cases") or []]
    source = {"path": str(path.resolve()), "run_id": run_id, "journal_sha256": run.get("journal_sha256"), "audit_version": audit.get("audit_version"),
        "audit_digest": audit.get("audit_digest"), "record_count": run.get("record_count"), "complete": run.get("complete") is True, "outer_case_status_counts": _status_counts(audit.get("outer_case_status_counts")),
        "inner_trial_status_counts": _status_counts(audit.get("inner_trial_status_counts")),
        "root_candidate_ids": [row["candidate_id"] for row, _ in roots], "root_cases": [{"candidate_id": row["candidate_id"], "case": case} for row, case in roots],
        "failed_cases": failed, "case_count": audit.get("case_count"), "root_count": len(roots), "failed_count": len(failed)}
    return source, roots, failed
def build_analysis(datasets: list[str | Path]) -> dict[str, Any]:
    paths, seen_paths = [], set()
    for item in datasets:
        path = Path(item); key = str(path.resolve()).casefold()
        _need(key not in seen_paths, f"duplicate dataset path: {path}"); _need(path.is_dir(), f"dataset directory does not exist: {path}")
        seen_paths.add(key); paths.append(path)
    pool, sources, unique_sources, all_failed, seen_journals = {}, [], [], [], set()
    for path in paths:
        audit = auditor.audit_dataset(path); _need(audit.get("audit_version") == auditor.AUDIT_VERSION, f"unexpected audit version: {audit.get('audit_version')!r}")
        records = read_records(path)  # exactly one journal read per dataset here
        source, roots, failed = _source_report(path, audit, records); sources.append(source)
        for row, case in roots: _merge_root(pool, row, source["run_id"], case)
        if source["journal_sha256"] not in seen_journals:
            seen_journals.add(source["journal_sha256"]); unique_sources.append(source); all_failed.extend(failed)
    rows = sorted(pool.values(), key=lambda row: row["candidate_id"]); architecture_counts, stage_counts, available_counts = {}, {}, {}
    for row in rows:
        _bump(architecture_counts, row["architecture"]); _bump(stage_counts, f"{row['architecture']}|{','.join(f'{k}={v}' for k, v in sorted((row.get('stage_counts') or {}).items()))}")
        for key, ok in row["available_metrics"].items(): available_counts[key] = available_counts.get(key, 0) + (1 if ok else 0)
    failures, minima = _failure_minima(rows)
    observed = [value for row in all_failed if (value := row["last_state"]["relative_power_residual"]) is not None]
    return {"analysis_version": ANALYSIS_VERSION, "scope": SCOPE,
        "scope_note": "four audited profiles + source l/D domain + numerical closure only; not overall feasibility", "sources": sources,
        "source_counts": {"datasets": len(sources), "unique_journals": len(unique_sources), "raw_records": sum(s.get("record_count") or 0 for s in unique_sources),
            "outer_cases": sum((s.get("case_count") or 0) for s in unique_sources), "inner_trials": sum(sum(s["inner_trial_status_counts"].values()) for s in unique_sources),
            "root_source_memberships": sum(s["root_count"] for s in unique_sources), "unique_root_candidates": len(rows), "failed_cases": len(all_failed)},
        "outer_case_status_counts": _sum_status_counts([s["outer_case_status_counts"] for s in unique_sources]),
        "inner_trial_status_counts": _sum_status_counts([s["inner_trial_status_counts"] for s in unique_sources]), "roots": rows, "root_count": len(rows),
        "architecture_counts": architecture_counts, "architecture_stage_count_counts": stage_counts, "effective_control_values": _control_values(rows), "active_known_check_failure_counts": failures,
        "min_observed_margin_by_architecture": minima, "root_residual_max_abs": _residual_maxima(rows), "numerical_closure": _numerical_summary(rows),
        "available_metric_counts": available_counts, "failed_cases": all_failed, "failed_case_count": len(all_failed),
        "failed_case_inner_power_residuals": {"last_state_observed": len(observed), "max_abs_relative": max(map(abs, observed)) if observed else None,
            "status": "OBSERVED" if observed else "NOT_CHECKED", "note": "EXPECTED_FLOW_SIZING_TRACE_NOT_AN_ENGINE_ERROR; excluded from root residual maxima"},
        "not_checked": dict(NOT_CHECKED), "overall_engineering_feasibility": None, "overall_engineering_feasibility_status": "NOT_CHECKED",
        "overall_engineering_feasibility_reason": OVERALL_REASON, "selection": None, "pareto_front": None, "production_readiness": None,
        "production_readiness_status": "NOT_CHECKED", "claim_scope": SCOPE, "plot_status": {"status": "NOT_ATTEMPTED"}}
def _screen_state(row: Mapping[str, Any]) -> str: return "SCREENING_PASS" if row["four_profile_screening_pass"] else "KNOWN_CHECK_FAIL" if row["known_constraint_violation"] or row["source_domain_status"] == "OUTSIDE_SOURCE_DOMAIN" else "NOT_SCREENED"
def _scatter_groups(ax: Any, points: list[Mapping[str, Any]], xkey: str, ykey: str) -> None:
    groups = {}
    for point in points:
        if point.get(xkey) is not None and point.get(ykey) is not None: groups.setdefault((point["architecture"], point["screen_state"]), []).append((point[xkey], point[ykey]))
    for (architecture, state), pairs in sorted(groups.items()):
        color = ARCH_COLORS.get(architecture, "#444444"); marker, fill = {"SCREENING_PASS": ("o", "full"), "KNOWN_CHECK_FAIL": ("x", "full"), "NOT_SCREENED": ("s", "none")}[state]
        kwargs = {"marker": marker, "s": 42, "linewidths": 1.2, "label": f"{architecture} {state.replace('_', ' ')} (screening scope only)"}
        kwargs.update({"color": color} if marker == "x" or fill == "full" else {"edgecolors": color, "facecolors": "none"})
        ax.scatter([x for x, _ in pairs], [y for _, y in pairs], **kwargs)
    if not groups: ax.text(0.5, 0.5, "NOT_CHECKED\nno observed values", ha="center", va="center", transform=ax.transAxes, fontsize=9)
def _control_points(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        for role, control in row["effective_controls"].items():
            speed, hub = control["n_rpm"], control["D_hub_m"]; alpha = row["per_role_alpha2_margin"].get(role, {}).get("min_margin_deg")
            if speed is not None and hub is not None:
                points.append({"architecture": row["architecture"], "role": role, "n_rpm": speed, "D_hub_m": hub,
                    "peak_sigma_MPa": row["stage_extremes_by_role"].get(role, {}).get("peak_sigma_MPa"), "alpha2_margin_deg": alpha,
                    "screen_state": _screen_state(row)})
    return points
def _draw_controls(ax: Any, points: list[Mapping[str, Any]], xkey: str, ykey: str) -> None:
    groups = {}
    for point in points:
        if point.get(xkey) is not None and point.get(ykey) is not None: groups.setdefault((point["architecture"], point["role"]), []).append((point[xkey], point[ykey]))
    for (architecture, role), pairs in sorted(groups.items()):
        ax.scatter([x for x, _ in pairs], [y for _, y in pairs], marker=ROLE_MARKERS.get(role, "o"), s=42,
            color=ARCH_COLORS.get(architecture, "#444444"), edgecolors="black", linewidths=0.5, label=f"{architecture} {role}")
    if not groups: ax.text(0.5, 0.5, "NOT_CHECKED\nno observed control values", ha="center", va="center", transform=ax.transAxes, fontsize=9)
def _make_plots(rows: list[Mapping[str, Any]], destination: Path) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": "NOT_AVAILABLE", "reason": f"matplotlib unavailable: {exc}", "files": [], "root_count": len(rows)}
    try:
        points = [{"architecture": row["architecture"], "screen_state": _screen_state(row), "pi_product": row["pi_product"],
            "air_mass_flow_kg_s": row["actual_sized_flows"]["air_mass_flow_kg_s"], "peak_sigma_MPa": row["stage_extremes"]["peak_sigma_MPa"]} for row in rows]
        files = []; fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
        _scatter_groups(axes[0], points, "pi_product", "air_mass_flow_kg_s")
        axes[0].set_xlabel("pi product (requested compressor pressure ratios)"); axes[0].set_ylabel("actual sized air mass flow, kg/s"); axes[0].set_title("pi vs actual sized air flow")
        _scatter_groups(axes[1], points, "pi_product", "peak_sigma_MPa")
        axes[1].axhline(290.0, color="0.45", linestyle="--", linewidth=1.0, label="290 MPa project assumption (not book material allowable)")
        axes[1].set_xlabel("pi product (requested compressor pressure ratios)"); axes[1].set_ylabel("peak sigma_rotor, MPa"); axes[1].set_title("pi vs peak rotor stress")
        for ax in axes: ax.grid(alpha=0.25); ax.legend(fontsize=6, loc="best")
        fig.suptitle(f"{SCOPE} | audited roots only; no interpolation, not a feasibility map", fontsize=9)
        path = destination / "sweep_pi_airflow_stress.png"; fig.savefig(path, dpi=140); plt.close(fig); files.append(path.name)
        controls = _control_points(rows)
        fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.0), constrained_layout=True)
        _draw_controls(axes[0][0], controls, "n_rpm", "peak_sigma_MPa")
        axes[0][0].set_xlabel("actual closure n_rpm"); axes[0][0].set_ylabel("peak sigma_rotor, MPa"); axes[0][0].set_title("n vs peak stress (role-specific)")
        _draw_controls(axes[0][1], controls, "D_hub_m", "peak_sigma_MPa")
        axes[0][1].set_xlabel("actual closure D_hub_m, m"); axes[0][1].set_ylabel("peak sigma_rotor, MPa"); axes[0][1].set_title("hub vs peak stress (role-specific)")
        for ax in axes[0]: ax.axhline(290.0, color="0.45", linestyle="--", linewidth=1.0, label="290 MPa project assumption (not book material allowable)")
        _draw_controls(axes[1][0], controls, "n_rpm", "alpha2_margin_deg")
        axes[1][0].set_xlabel("actual closure n_rpm"); axes[1][0].set_ylabel("min alpha2 p.38 margin, deg (role-specific)"); axes[1][0].set_title("n vs alpha2 margin (role-specific)")
        _draw_controls(axes[1][1], controls, "D_hub_m", "alpha2_margin_deg")
        axes[1][1].set_xlabel("actual closure D_hub_m, m"); axes[1][1].set_ylabel("min alpha2 p.38 margin, deg (role-specific)"); axes[1][1].set_title("hub vs alpha2 margin (role-specific)")
        for row in axes:
            for ax in row: ax.grid(alpha=0.25); ax.legend(fontsize=6, loc="best") if ax.get_legend_handles_labels()[0] else None
        fig.suptitle(f"{SCOPE} | source control families are marker-distinct; no lines or interpolation; not a feasibility map", fontsize=9)
        path = destination / "sweep_n_hub_stress_alpha.png"; fig.savefig(path, dpi=140); plt.close(fig); files.append(path.name)
        missing = [name for name in files if not (destination / name).is_file()]
        if missing: return {"status": "FAILED", "reason": f"plot files missing: {missing}", "files": files, "root_count": len(rows)}

        return {"status": "GENERATED", "files": files, "root_count": len(rows), "scope": SCOPE,
            "note": "scatter only; 290 MPa line is a project assumption, not a book allowable; screen markers are not full feasibility"}
    except Exception as exc:  # plotting must never suppress the JSON report
        return {"status": "FAILED", "reason": f"plot generation failed: {exc}", "files": [], "root_count": len(rows)}
def analyze(datasets: list[str | Path], destination: str | Path) -> Path:
    target = Path(destination)
    if target.exists(): _need(target.is_dir() and not any(target.iterdir()), f"destination is not a new empty directory: {target}")
    analysis = build_analysis(datasets); target.mkdir(parents=True, exist_ok=True)
    analysis["plot_status"] = _make_plots(analysis["roots"], target)
    output = target / "analysis.json"
    with open(output, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(analysis, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False); handle.write("\n")
    return output
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse completed source-sized RAW stage sweeps; read-only, no ranking or feasibility claim.")
    parser.add_argument("--dataset", action="append", required=True, help="completed verified RAW dataset directory (repeatable)")
    parser.add_argument("--destination", required=True, help="new empty output directory"); args = parser.parse_args(argv)
    try: analyze(args.dataset, args.destination)
    except (AnalysisError, auditor.AuditError, DatasetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 0
if __name__ == "__main__": raise SystemExit(main())
