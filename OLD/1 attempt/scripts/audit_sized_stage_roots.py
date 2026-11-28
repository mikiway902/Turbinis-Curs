#!/usr/bin/env python3
"""Post-hoc audit of the three saved flow-sized stage roots.

Reads a complete RAW journal plus its flow-sizing summary. It recomputes source 0D
compressor arithmetic from the saved input, but never reruns the meanline, changes RAW,
ranks designs, or claims physical validation. Numerical closure is separate from
engineering acceptance. rho<=0.65 and 1.8<=pi<=2.4 are user/project profiles; alpha2
is book p.38; sigma<=290 MPa is a DESIGN_ASSUMPTIONS assumption, not material validation.
l/D outside the source table is clipped by existing np.interp, not physical impossibility.
Overall feasibility is always NOT_CHECKED. Since v3 every planned case is analysed: only
successful outer cases get root reports, while typed outer/inner failures are preserved
with their actual evaluated states and never relabelled as physical infeasibility.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
from typing import Any, Mapping
ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from turbine_engine.architecture_cycle import calculate_architecture_cycle  # noqa: E402
from turbine_engine.architecture_raw import payload_to_input  # noqa: E402
from turbine_engine.dataset import DatasetError, load_manifest, read_records  # noqa: E402
from turbine_engine.raw import CalculationStatus  # noqa: E402
AUDIT_VERSION = "sized-stage-root-audit-v3"
SUMMARY_NAME, POWER_REL_TOL, PRESSURE_REL_TOL = "summary.json", 1e-6, 1e-6
FAILURE_STATUSES = ("NUMERICAL_FAILURE", "NON_PHYSICAL_STATE", "MODEL_ERROR")
CASE_STATUSES = ("SUCCESS",) + FAILURE_STATUSES
BOUNDARY_REASON = "outer_bracket_not_bracketed"
OUTER_FAILURE_REASONS = (BOUNDARY_REASON, "outer_iteration_limit", "stalled_adjacent_floats")
TRIAL_FAILURE_REASON, MISSING_METRIC_REASON = "trial_failure_at_air_flow", "success_missing_power_metric"
FAILURE_REASONS = OUTER_FAILURE_REASONS + (TRIAL_FAILURE_REASON, MISSING_METRIC_REASON)
CMP_RTOL, CMP_ATOL, RHO_LIMIT, PI_BOUNDS, DOMAIN_TOL = 1e-9, 1e-9, 0.65, (1.8, 2.4), 1e-12
SOURCE_PAGE = 38
NOT_CHECKED_ITEMS = ("compressor maps and stage split", "cooling and bleed semantics",
    "material allowables, metal temperature, thermal stress", "creep/resource life",
    "3D/radial equilibrium and secondary flows", "universal Mach/velocity acceptance limits (no supplied bounds)",
    "production design-space coverage")
ALIASES = {"pi": ("pi_stage", "\u03c0_\u0441\u0442"), "rho": ("rho", "\u03c1"), "alpha2": ("alpha2_deg", "\u03b12"),
    "sigma": ("sigma_rotor", "\u03c3_\u0440"), "l_l": ("l_l", "l_\u043b"), "D_l": ("D_l", "D_\u043b"),
    "L_t": ("L_t", "L_\u0442"), "p_total": ("p_total", "p_\u0441\u0442_\u043f"),
    "T_total": ("T_total", "T_\u0441\u0442_\u043f"), "T0": ("T0", "T0"), "p0": ("p0", "p0")}
METRICS = ("pi", "rho", "alpha2", "sigma", "l_l", "D_l", "c2", "u1", "u2", "M_c1", "M_w2", "L_t", "G_c", "p_total")
class AuditError(RuntimeError): pass

def _need(ok: Any, message: str) -> None:
    if not ok: raise AuditError(message)

def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)): return None
    value = float(value); return value if math.isfinite(value) else None

def _same(a: Any, b: Any, rtol: float = CMP_RTOL, atol: float = CMP_ATOL) -> bool:
    x, y = _num(a), _num(b)
    return x is not None and y is not None and abs(x - y) <= max(atol, rtol * max(abs(x), abs(y)))

def _match(label: str, computed: Any, saved: Any) -> None:
    if computed is None or saved is None: return
    if isinstance(computed, bool) or isinstance(saved, bool): _need(computed is saved, label)
    else: _need(_same(computed, saved, rtol=1e-12, atol=1e-9), label)

def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return _sha(text.encode("utf-8"))

def _load_book() -> tuple[Any, str, Path]:
    path = ROOT / "Turbinis-Calc" / "constraints.py"; _need(path.is_file(), f"missing book metadata file: {path}")
    digest = _sha(path.read_bytes()); spec = importlib.util.spec_from_file_location("_tc_audit_book_constraints", path)
    _need(spec is not None and spec.loader is not None, "cannot load book metadata")
    module = importlib.util.module_from_spec(spec); old, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try: spec.loader.exec_module(module)
    finally: sys.dont_write_bytecode = old
    return module, digest, path

def _book_metadata(module: Any, digest: str, path: Path) -> dict[str, Any]:
    by_key = {item["key"]: item for item in module.BOOK_CONSTRAINTS}; mid, last = by_key["alpha2_mid"], by_key["alpha2_last"]
    _need(mid["page"] == SOURCE_PAGE and last["page"] == SOURCE_PAGE, "alpha2 source page changed")
    sigma = _num(module.DESIGN_ASSUMPTIONS["sigma_allow_MPa"]["value"]); _need(sigma is not None and sigma > 0, "missing project sigma assumption")
    table = [float(item) for item in module.LD_TABLE]; _need(len(table) >= 2 and table[0] < table[-1], "invalid l/D table domain")
    stressed = by_key.get("sigma_stressed", {})
    return {"file": path.relative_to(ROOT).as_posix(), "sha256": digest, "alpha2": {"mid": {"key": "alpha2_mid", "lo": float(mid["lo"]), "hi": float(mid["hi"]), "page": SOURCE_PAGE, "status": mid["status"]},
        "last": {"key": "alpha2_last", "lo": float(last["lo"]), "hi": float(last["hi"]), "page": SOURCE_PAGE, "status": last["status"]}, "angle_convention": "ACTUAL_ALPHA2_DEG_STORED_NO_ABS_OR_FLIP"},
        "sigma_limit": {"value": sigma, "unit": "MPa", "basis": "DESIGN_ASSUMPTIONS_PROJECT_ASSUMPTION_NOT_BOOK_MATERIAL_VALIDATION", "book_reference": {"lo": stressed.get("lo"), "hi": stressed.get("hi"), "note": "book recommendation; not material validation"}},
        "reaction_lD_source_domain": {"lo": table[0], "hi": table[-1], "table": table, "basis": "BOOK_RECOMMENDATION_TABLE_PAGE_58_NOT_HARD_UNIVERSAL_LIMIT",
            "clip_note": "outside domain is clipped by existing np.interp; clipped_not_rejected, not physical impossibility"},
        "rho_profile": {"profile": "RHO065", "limit": RHO_LIMIT, "basis": "USER_ACCEPTED_ACCEPTANCE_CEILING_UPPER_ONLY", "lower_bound_applied": False}, "pi_profile": {"profile": "PROMPT_STAGE_PI_1_8_2_4", "bounds": list(PI_BOUNDS), "basis": "USER_PROMPT_ENGINEERING_PROFILE_NOT_BOOK"}}

def _field(stage: Mapping[str, Any], name: str) -> float | None:
    for key in ALIASES.get(name, (name,)):
        if key in stage:
            value = _num(stage[key])
            if value is not None: return value
    return None

def _upper(value: float | None, limit: float, basis: str, note: str) -> dict[str, Any]:
    if value is None: return {"status": "NOT_CHECKED", "value": None, "limit": limit, "margin": None, "basis": basis, "note": note}
    margin = limit - value; return {"status": "PASS" if margin >= 0 else "FAIL", "value": value, "limit": limit, "margin": margin, "basis": basis, "note": note}

def _range(value: float | None, lo: float, hi: float, basis: str, note: str) -> dict[str, Any]:
    if value is None: return {"status": "NOT_CHECKED", "value": None, "bounds": [lo, hi], "margin": None, "lower_margin": None, "upper_margin": None, "basis": basis, "note": note}
    lower, upper = value - lo, hi - value
    return {"status": "PASS" if min(lower, upper) >= 0 else "FAIL", "value": value, "bounds": [lo, hi], "margin": min(lower, upper), "lower_margin": lower, "upper_margin": upper, "basis": basis, "note": note}

def _lD_check(lD: float | None, meta: Mapping[str, Any]) -> dict[str, Any]:
    domain = meta["reaction_lD_source_domain"]; lo, hi = domain["lo"], domain["hi"]
    if lD is None: return {"status": "NOT_CHECKED", "value": None, "source_bounds": [lo, hi], "source_domain_margin": None, "clip_status": "NOT_CHECKED", "note": domain["clip_note"]}
    lower, upper = lD - lo, hi - lD; outside = min(lower, upper) < -DOMAIN_TOL
    return {"status": "OUTSIDE_SOURCE_DOMAIN" if outside else "WITHIN_SOURCE_DOMAIN", "value": lD, "source_bounds": [lo, hi],
        "source_domain_margin": min(lower, upper), "lower_margin": lower, "upper_margin": upper,
        "clip_status": "SOURCE_TABLE_CLIPPED_NOT_REJECTED" if outside else "WITHIN_TABLE", "note": domain["clip_note"]}

def _stage_report(group: str, index: int, total: int, stage: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any]:
    v = {name: _field(stage, name) for name in METRICS}; l_l, D_l = v["l_l"], v["D_l"]
    lD = None if l_l is None or D_l is None or D_l <= 0 else l_l / D_l; last = index == total
    alpha = meta["alpha2"]["last"] if last else meta["alpha2"]["mid"]; sigma = meta["sigma_limit"]
    checks = {"rho_065_upper": _upper(v["rho"], RHO_LIMIT, meta["rho_profile"]["basis"], "upper-only user acceptance ceiling; no 0.2 lower bound"), "prompt_pi_1_8_2_4": _range(v["pi"], *PI_BOUNDS, meta["pi_profile"]["basis"], "post-hoc project profile, not book fundamental"),
        "alpha2_source_page_38": _range(v["alpha2"], alpha["lo"], alpha["hi"], "BOOK_REQUIREMENT_PAGE_38", f"actual alpha2, no flip/abs; {'last' if last else 'intermediate'} stage"), "sigma_project_290MPa": _upper(v["sigma"], sigma["value"], sigma["basis"], "project assumption; material/creep/thermal validation absent"),
        "reaction_table_lD_domain": _lD_check(lD, meta), "mach_velocity_universal": {"status": "NOT_CHECKED", "reason": "no universal Mach/velocity acceptance bounds supplied", "M_c1": v["M_c1"], "M_w2": v["M_w2"], "c2": v["c2"], "u1": v["u1"], "u2": v["u2"]}}
    return {"group": group, "stage": index, "last_stage": last, "metrics": {"pi": v["pi"], "rho": v["rho"], "alpha2_deg": v["alpha2"], "M_c1": v["M_c1"], "M_w2": v["M_w2"],
        "l_l_m": l_l, "D_l_m": D_l, "lD": lD, "c2_m_s": v["c2"], "u1_m_s": v["u1"], "u2_m_s": v["u2"], "sigma_MPa": v["sigma"]}, "checks": checks}

def _rollup(stages: list[dict[str, Any]], key: str) -> dict[str, Any]:
    checks = [stage["checks"][key] for stage in stages]; statuses = [item["status"] for item in checks]
    status = "FAIL" if "FAIL" in statuses else "NOT_CHECKED" if "NOT_CHECKED" in statuses else "PASS"
    margins = [item["margin"] for item in checks if item.get("margin") is not None]
    return {"status": status, "min_signed_margin": min(margins) if margins else None,
        "not_checked_stages": [f"{stage['group']}:{stage['stage']}" for stage, item in zip(stages, checks) if item["status"] == "NOT_CHECKED"]}

def _domain_rollup(stages: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [stage["checks"]["reaction_table_lD_domain"] for stage in stages]; statuses = [item["status"] for item in checks]
    status = "OUTSIDE_SOURCE_DOMAIN" if "OUTSIDE_SOURCE_DOMAIN" in statuses else "WITHIN_SOURCE_DOMAIN" if statuses and all(x == "WITHIN_SOURCE_DOMAIN" for x in statuses) else "NOT_CHECKED"
    margins = [item["source_domain_margin"] for item in checks if item.get("source_domain_margin") is not None]
    return {"status": status, "min_source_domain_margin": min(margins) if margins else None,
        "outside_stages": [f"{stage['group']}:{stage['stage']}" for stage, item in zip(stages, checks) if item["status"] == "OUTSIDE_SOURCE_DOMAIN"]}

def _stage_gas_power(stages: list[Mapping[str, Any]]) -> float | None:
    pairs = [(_field(stage, "G_c"), _field(stage, "L_t")) for stage in stages]
    return None if any(g is None or l is None for g, l in pairs) else sum(g * l for g, l in pairs)

def _source_reference(record: Any) -> dict[str, float | None]:
    payload = record.payload.get("architecture_cycle_source")
    _need(isinstance(payload, Mapping), "root input is missing architecture_cycle_source")
    try: result = calculate_architecture_cycle(payload_to_input(payload))
    except Exception as exc: raise AuditError(f"source 0D cycle failed: {exc}") from exc
    return {"specific_work_J_kg": _num(result.compressor.specific_work), "air_mass_flow_kg_s": _num(result.air_mass_flow), "compressor_power_W": _num(result.compressor.power)}

def _compressor_check(flow: Mapping[str, Any], source: Mapping[str, Any], actual_saved: Any, reference_saved: Any,
                      actual_sized_saved: Any = None, boundary_target_saved: Any = None,
                      require_actual_sized: bool = False, require_boundary_target: bool = False) -> dict[str, Any]:
    source_flow, trial_air = _num(flow.get("source_air_mass_flow_kg_s")), _num(flow.get("trial_air_mass_flow_kg_s"))
    target, lc = _num(flow.get("actual_compressor_power_target_W")), _num(source.get("specific_work_J_kg"))
    source_air, source_power = _num(source.get("air_mass_flow_kg_s")), _num(source.get("compressor_power_W"))
    actual, reference, sized, boundary = _num(actual_saved), _num(reference_saved), _num(actual_sized_saved), _num(boundary_target_saved)
    expected_actual = lc * trial_air if lc is not None and trial_air is not None else None
    if lc is not None and source_air is not None and source_power is not None: _match("source 0D compressor power does not equal Lc*source air", source_power, lc * source_air)
    for label, x, y in (("saved reference 0D compressor power differs from source cycle", source_power, reference),
            ("source air flow differs from saved source flow", source_air, source_flow),
            ("actual compressor target differs from Lc*trial air", target, expected_actual),
            ("actual branch compressor power differs from Lc*trial air", actual, expected_actual),
            ("actual branch compressor power differs from flow target", actual, target),
            ("actual sized compressor power differs from flow target", sized, target),
            ("boundary compressor target differs from flow target", boundary, target)): _match(label, x, y)
    required = [source_flow, trial_air, target, lc, source_air, source_power, actual, reference]
    if require_actual_sized: required.append(sized)
    if require_boundary_target: required.append(boundary)
    status = "PASS" if all(value is not None for value in required) else "NOT_CHECKED"
    return {"status": status, "specific_work_J_kg": lc, "source_air_mass_flow_kg_s": source_air, "source_compressor_power_W": source_power, "saved_source_air_mass_flow_kg_s": source_flow,
        "trial_air_mass_flow_kg_s": trial_air, "flow_actual_compressor_power_target_W": target, "expected_actual_power_W": expected_actual, "saved_branch_compressor_power_W": actual,
        "saved_reference_0d_compressor_power_W": reference, "saved_actual_sized_compressor_power_W": sized, "saved_boundary_target_compressor_power_W": boundary,
        "source_minus_actual_W": None if source_power is None or actual is None else source_power - actual, "expected_actual_minus_saved_W": None if expected_actual is None or actual is None else expected_actual - actual,
        "source_and_actual_distinct": None if source_power is None or actual is None else not _same(source_power, actual, rtol=1e-12, atol=1e-9)}

def _state_continuity(gg_stages: list[Mapping[str, Any]], fpt_stages: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not gg_stages or not fpt_stages: return {"status": "NOT_CHECKED", "reason": "missing GG/FPT stage", "source": "INDEPENDENT_LEGACY_STAGE_KEYS_NOT_STORED_BOOLEANS"}
    gg, fpt = gg_stages[-1], fpt_stages[0]
    gt, gp, gg_flow = _field(gg, "T_total"), _field(gg, "p_total"), _field(gg, "G_c")
    ft, fp, fpt_flow = _field(fpt, "T0"), _field(fpt, "p0"), _field(fpt, "G_c")
    report = {"source": "INDEPENDENT_LEGACY_STAGE_KEYS_NOT_STORED_BOOLEANS",
        "gas_generator_last": {"T_total_K": gt, "p_total_Pa": gp, "G_c_kg_s": gg_flow},
        "free_power_first": {"T0_K": ft, "p0_Pa": fp, "G_c_kg_s": fpt_flow},
        "temperature_delta_K": None if gt is None or ft is None else ft - gt,
        "pressure_delta_Pa": None if gp is None or fp is None else fp - gp,
        "flow_delta_kg_s": None if gg_flow is None or fpt_flow is None else fpt_flow - gg_flow}
    values = (gt, gp, gg_flow, ft, fp, fpt_flow)
    if any(value is None for value in values):
        report.update({"status": "NOT_CHECKED", "missing": [name for name, value in zip(("GG.T_total", "GG.p_total", "GG.G_c", "FPT.T0", "FPT.p0", "FPT.G_c"), values) if value is None]})
        return report
    report["temperature_match"] = _same(gt, ft, rtol=1e-12, atol=1e-9)
    report["pressure_match"] = _same(gp, fp, rtol=1e-12, atol=1e-9)
    report["flow_match"] = _same(gg_flow, fpt_flow, rtol=1e-12, atol=1e-9)
    report["status"] = "PASS" if report["temperature_match"] and report["pressure_match"] and report["flow_match"] else "FAIL"
    return report

def _numerical_closed(base: Any, compressor_status: str, mass_status: str, continuity_status: str) -> bool | None:
    if mass_status == "FAIL" or continuity_status == "FAIL": return False
    if compressor_status != "PASS" or base is None or mass_status == "NOT_CHECKED" or continuity_status == "NOT_CHECKED": return None
    return bool(base)

def _power_component(group: str, stages: list[Mapping[str, Any]], efficiency: float | None, compressor: float | None, target: float | None, subtract: bool) -> dict[str, Any]:
    gas = _stage_gas_power(stages)
    if gas is None or efficiency is None or target is None or (group != "free_power" and compressor is None):
        return {"group": group, "status": "NOT_CHECKED", "reason": "required stage power/flow or target metric missing", "stage_gas_power_W": gas, "mechanical_efficiency": efficiency,
            "compressor_power_W": compressor, "target_W": target, "net_mechanical_power_W": None, "signed_power_residual_W": None, "relative_power_residual": None, "engine_power_closed": None}
    mechanical = efficiency * gas; net = mechanical - compressor if subtract else mechanical; residual = net - target
    relative = abs(residual) / abs(target) if target else None; closed = relative is not None and relative <= POWER_REL_TOL
    return {"group": group, "status": "PASS" if closed else "FAIL", "stage_gas_power_W": gas, "mechanical_efficiency": efficiency,
        "stage_mechanical_power_W": mechanical, "compressor_power_W": compressor, "target_W": target, "net_mechanical_power_W": net,
        "signed_power_residual_W": residual, "relative_power_residual": relative, "engine_power_closed": bool(closed)}

def _closure_report(closure: Mapping[str, Any], payload_target: Any) -> dict[str, Any]:
    target, actual = _num(closure.get("target_value")), _num(closure.get("actual_metric")); kind = closure.get("target_kind")
    saved = _num(closure.get("relative_residual")); controls = closure.get("controls") if isinstance(closure.get("controls"), Mapping) else {}
    _need(controls, "saved closure controls are missing")
    if target is None or actual is None: return {"target_kind": kind, "status": "NOT_CHECKED", "target_value": target, "actual_metric": actual, "relative_residual": None, "controls": dict(controls)}
    if payload_target is not None: _need(_same(target, payload_target), "saved closure target differs from input target")
    relative = abs(actual - target) / abs(target) if target else None; _need(saved is not None, "saved closure relative residual is missing")
    _need(_same(saved, relative, atol=1e-12), "saved closure relative residual does not reproduce")
    tol = PRESSURE_REL_TOL if kind == "total_outlet_pressure" else POWER_REL_TOL
    return {"target_kind": kind, "status": "PASS" if relative is not None and relative <= tol else "FAIL", "target_value": target,
        "actual_metric": actual, "signed_residual": actual - target, "relative_residual": relative, "saved_relative_residual": saved,
        "saved_residual_consistent": True, "controls": dict(controls)}

def _pressure_closure(closure: Mapping[str, Any], stages: list[Mapping[str, Any]]) -> dict[str, Any]:
    report = _closure_report(closure, closure.get("target_value")); report["last_stage_total_pressure_Pa"] = _field(stages[-1], "p_total") if stages else None
    report["last_stage_matches_actual"] = _same(report["last_stage_total_pressure_Pa"], report.get("actual_metric"))
    if closure.get("target_kind") != "total_outlet_pressure":
        report.update({"status": "NOT_APPLICABLE_POWER_TARGET", "pressure_closed": None}); return report
    report["pressure_closed"] = bool(report["last_stage_matches_actual"] and report.get("relative_residual") is not None and report["relative_residual"] <= PRESSURE_REL_TOL)
    report["status"] = "PASS" if report["pressure_closed"] else "FAIL"; return report

def _mass_check(flow: Mapping[str, Any], payload: Mapping[str, Any], stages: list[Mapping[str, Any]]) -> dict[str, Any]:
    air, fuel, gas = (_num(flow.get(k)) for k in ("trial_air_mass_flow_kg_s", "trial_fuel_mass_flow_kg_s", "trial_gas_mass_flow_kg_s"))
    _need(air is not None and fuel is not None and gas is not None, "flow summary is missing trial air/fuel/gas")
    _need(_same(air, payload.get("air_mass_flow_kg_s")), "trial air flow differs from record input")
    residual, saved = gas - (air + fuel), _num(flow.get("mass_residual_kg_s")); _need(saved is not None and _same(saved, residual, atol=1e-12), "mass residual does not match air/fuel/gas")
    _need(flow.get("pre_reducer") is True, "flow trial is not labelled pre-reducer")
    flows = [_field(stage, "G_c") for stage in stages]; stage_ok = None if any(v is None for v in flows) else all(_same(v, gas, rtol=1e-12, atol=1e-12) for v in flows)
    status = "NOT_CHECKED" if stage_ok is None else "PASS" if abs(residual) <= 1e-12 and stage_ok else "FAIL"
    return {"air_mass_flow_kg_s": air, "fuel_mass_flow_kg_s": fuel, "gas_mass_flow_kg_s": gas, "fuel_air_ratio": fuel / air if air else None,
        "mass_balance_residual_kg_s": residual, "pre_reducer": True, "stage_flow_consistent": stage_ok, "status": status}

def _verify_summary(case: Mapping[str, Any], record: Any, net: float | None, flow: Mapping[str, Any], target: float | None) -> None:
    root = case.get("root"); _need(isinstance(root, Mapping), f"{case.get('case')}: missing summary root")
    _need(root.get("candidate_id") == record.candidate_id, f"{case.get('case')}: summary root id does not reference the record")
    expected = {"air_mass_flow_kg_s": flow["trial_air_mass_flow_kg_s"], "fuel_mass_flow_kg_s": flow["trial_fuel_mass_flow_kg_s"], "gas_mass_flow_kg_s": flow["trial_gas_mass_flow_kg_s"]}
    if net is not None: expected["net_mechanical_power_W"] = net
    for key, value in expected.items(): _need(key in root and _same(root[key], value), f"{case.get('case')}: summary root {key} differs from record")
    if net is not None:
        _need(target is not None and _same(target, case.get("target_gearbox_input_power_W")), f"{case.get('case')}: summary/case power target differs from output")
        residual, relative = net - target, abs(net - target) / target
        _need(_same(root.get("signed_power_residual_W"), residual), f"{case.get('case')}: summary signed power residual differs")
        _need(_same(root.get("relative_power_residual"), relative), f"{case.get('case')}: summary relative power residual differs")
        _need(root.get("numerical_power_closed") is (relative <= POWER_REL_TOL), f"{case.get('case')}: summary power-closed claim is not reproducible")

def _root_report(case: Mapping[str, Any], record: Any, meta: Mapping[str, Any]) -> dict[str, Any]:
    outputs = record.outputs if isinstance(record.outputs, Mapping) else {}; boundary, power, flow = outputs.get("actual_boundary"), outputs.get("power"), outputs.get("flow_sizing")
    _need(record.calculation_status is CalculationStatus.SUCCESS, f"{case.get('case')}: root record is not SUCCESS")
    _need(isinstance(boundary, Mapping) and isinstance(power, Mapping) and isinstance(flow, Mapping), f"{case.get('case')}: missing root outputs")
    source = _source_reference(record); groups: dict[str, Any] = {}
    if isinstance(boundary.get("closure"), Mapping):
        stages = boundary["closure"].get("actual_stages"); _need(isinstance(stages, list) and stages, "AB root has no actual stages")
        target = (record.payload.get("targets") or {}).get("main_total_outlet_pressure_Pa"); closure = _pressure_closure(boundary["closure"], stages)
        _need(target is not None and _same(closure.get("target_value"), target), "AB pressure target provenance mismatch")
        reports = [_stage_report("main", i, len(stages), stage, meta) for i, stage in enumerate(stages, 1)]
        gp = _power_component("main", stages, _num(power.get("mechanical_efficiency")), _num(power.get("compressor_power_W")), _num(power.get("target_gearbox_input_power_W")), True)
        for label, computed, saved in (("stage gas power", gp.get("stage_gas_power_W"), power.get("stage_gas_power_W")), ("stage mechanical power", gp.get("stage_mechanical_power_W"), power.get("stage_mechanical_power_W")),
                ("net mechanical power", gp.get("net_mechanical_power_W"), power.get("net_mechanical_power_W")), ("signed power residual", gp.get("signed_power_residual_W"), power.get("signed_power_residual_W")),
                ("relative power residual", gp.get("relative_power_residual"), power.get("relative_power_residual")), ("engine power closed", gp.get("engine_power_closed"), power.get("engine_power_closed"))): _match(f"AB saved {label} does not reproduce", computed, saved)
        comp = _compressor_check(flow, source, power.get("compressor_power_W"), power.get("reference_0d_compressor_power_W"), power.get("actual_sized_compressor_power_W"), require_actual_sized=True)
        groups["main"] = {"stages": reports, "closure": closure, "power": gp, "compressor_power": comp}; net = gp["net_mechanical_power_W"]
        base_numerical = gp["engine_power_closed"] and closure["pressure_closed"]; num_power = gp["engine_power_closed"]; raw_stages = stages
        continuity = {"status": "NOT_APPLICABLE", "reason": "AB has one turbine group"}
    elif isinstance(boundary.get("gas_generator"), Mapping) and isinstance(boundary.get("free_power"), Mapping):
        gg, fpt = boundary["gas_generator"], boundary["free_power"]; gg_stages, fpt_stages = gg["closure"].get("actual_stages"), fpt["closure"].get("actual_stages")
        _need(isinstance(gg_stages, list) and gg_stages and isinstance(fpt_stages, list) and fpt_stages, "C root has no actual stages")
        targets = record.payload.get("targets") or {}; gps, fps = power.get("gas_generator") or {}, power.get("free_power") or {}
        gg_closure, fpt_closure = _closure_report(gg["closure"], None), _pressure_closure(fpt["closure"], fpt_stages)
        _need(_same(fpt_closure.get("target_value"), targets.get("free_power_outlet_total_pressure_Pa")), "C FPT pressure target provenance mismatch")
        gg_reports = [_stage_report("gas_generator", i, len(gg_stages), stage, meta) for i, stage in enumerate(gg_stages, 1)]
        fpt_reports = [_stage_report("free_power", i, len(fpt_stages), stage, meta) for i, stage in enumerate(fpt_stages, 1)]
        gg_power = _power_component("gas_generator", gg_stages, _num(gps.get("mechanical_efficiency")), _num(gps.get("compressor_power_W")), _num(gg_closure.get("target_value")), False)
        fpt_power = _power_component("free_power", fpt_stages, _num(fps.get("mechanical_efficiency")), None, _num(power.get("target_gearbox_input_power_W")), False)
        for label, computed, saved in (("GG closure target", gg_closure.get("target_value"), gps.get("compressor_power_W")), ("GG boundary target", gg_closure.get("target_value"), gg.get("target_compressor_power_W")),
                ("GG stage gas power", gg_power.get("stage_gas_power_W"), gps.get("stage_gas_power_W")), ("GG mechanical power", gg_power.get("stage_mechanical_power_W"), gps.get("mechanical_power_W")),
                ("GG balance residual", gg_power.get("signed_power_residual_W"), gps.get("balance_residual_W")), ("GG boundary residual", gg_power.get("signed_power_residual_W"), gg.get("gas_generator_balance_residual_W")),
                ("GG closure actual", gg_closure.get("actual_metric"), gg_power.get("stage_mechanical_power_W")), ("FPT stage gas power", fpt_power.get("stage_gas_power_W"), fps.get("stage_gas_power_W")),
                ("FPT mechanical power", fpt_power.get("net_mechanical_power_W"), fps.get("mechanical_power_W")), ("FPT signed power residual", fpt_power.get("signed_power_residual_W"), power.get("signed_power_residual_W")),
                ("FPT relative power residual", fpt_power.get("relative_power_residual"), power.get("relative_power_residual")), ("FPT engine power closed", fpt_power.get("engine_power_closed"), power.get("engine_power_closed"))): _match(f"C saved {label} does not reproduce", computed, saved)
        comp = _compressor_check(flow, source, gps.get("compressor_power_W"), gps.get("reference_0d_compressor_power_W"), boundary_target_saved=gg.get("target_compressor_power_W"), require_boundary_target=True)
        groups["gas_generator"] = {"stages": gg_reports, "closure": gg_closure, "power": gg_power, "compressor_power": comp}; groups["free_power"] = {"stages": fpt_reports, "closure": fpt_closure, "power": fpt_power}
        net = fpt_power["net_mechanical_power_W"]; base_numerical = gg_power["engine_power_closed"] and fpt_power["engine_power_closed"] and fpt_closure["pressure_closed"]
        num_power = fpt_power["engine_power_closed"] if gg_power["engine_power_closed"] is not None and fpt_power["engine_power_closed"] is not None else None; raw_stages = gg_stages + fpt_stages
        continuity = _state_continuity(gg_stages, fpt_stages)
    else: raise AuditError(f"{case.get('case')}: unsupported actual_boundary layout")
    all_stages = [stage for group in groups.values() for stage in group["stages"]]; mass = _mass_check(flow, record.payload, raw_stages)
    power_target = groups["free_power"]["power"]["target_W"] if "free_power" in groups else groups["main"]["power"]["target_W"]
    numerical = _numerical_closed(base_numerical, comp["status"], mass["status"], continuity["status"])
    numerical_power = None if comp["status"] != "PASS" or num_power is None else bool(num_power)
    _verify_summary(case, record, net, flow, power_target)
    keys = ("rho_065_upper", "prompt_pi_1_8_2_4", "alpha2_source_page_38", "sigma_project_290MPa"); constraints = {key: _rollup(all_stages, key) for key in keys}; domain = _domain_rollup(all_stages)
    residuals = {"mass_balance_residual_kg_s": mass["mass_balance_residual_kg_s"], "compressor_source_minus_actual_W": comp.get("source_minus_actual_W"), "compressor_expected_actual_minus_saved_W": comp.get("expected_actual_minus_saved_W"),
        "continuity_temperature_delta_K": continuity.get("temperature_delta_K"), "continuity_pressure_delta_Pa": continuity.get("pressure_delta_Pa"), "continuity_flow_delta_kg_s": continuity.get("flow_delta_kg_s")}
    for name, group in groups.items():
        residuals[f"{name}_power_signed_residual_W"] = group["power"].get("signed_power_residual_W"); residuals[f"{name}_closure_signed_residual"] = group["closure"].get("signed_residual")
    return {"case": case.get("case"), "root_candidate_id": record.candidate_id, "architecture": record.architecture, "record_status": record.calculation_status.value, "groups": groups,
        "compressor_power": comp, "state_continuity": continuity, "constraints": constraints, "source_domain": domain, "mass_fuel_air": mass, "numerical_closed": numerical, "numerical_power_closed": numerical_power,
        "reported_residuals": residuals, "positive_residuals": {key: value for key, value in residuals.items() if value is not None and value > 0},
        "known_constraint_violation": any(item["status"] == "FAIL" for item in constraints.values()), "source_domain_violation": domain["status"] == "OUTSIDE_SOURCE_DOMAIN",
        "has_not_checked": any(item["status"] == "NOT_CHECKED" for item in constraints.values()) or domain["status"] == "NOT_CHECKED" or comp["status"] != "PASS" or continuity["status"] == "NOT_CHECKED", "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED", "overall_engineering_feasibility_reason": "maps_cooling_material_thermal_resource_3d_not_validated"}

def _counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values: counts[value] = counts.get(value, 0) + 1
    return counts

def _raw_net_power(record: Any) -> float | None:
    if record.calculation_status is not CalculationStatus.SUCCESS: return None
    outputs = record.outputs if isinstance(record.outputs, Mapping) else {}
    power = outputs.get("power")
    return _num(power.get("net_mechanical_power_W")) if isinstance(power, Mapping) else None

def _has_partial_gas_generator(record: Any) -> bool:
    if record.calculation_status is CalculationStatus.SUCCESS: return False
    outputs = record.outputs if isinstance(record.outputs, Mapping) else {}
    boundary, power = outputs.get("actual_boundary"), outputs.get("power")
    return (isinstance(outputs.get("gas_generator"), Mapping)
        or (isinstance(boundary, Mapping) and isinstance(boundary.get("gas_generator"), Mapping))
        or (isinstance(power, Mapping) and isinstance(power.get("gas_generator"), Mapping)))

def _check_state(case: Mapping[str, Any], state: Any, by_id: Mapping[str, Any], target: float) -> Any:
    label = case.get("case"); identifier = state.get("candidate_id") if isinstance(state, Mapping) else None
    _need(identifier in by_id, f"{label}: evaluated state reference not found: {identifier!r}")
    record = by_id[identifier]
    _need(record.architecture == case.get("architecture"), f"{label}: state architecture differs from case")
    _need(record.payload.get("point_index") == case.get("source_point_index"), f"{label}: state source point index differs from case")
    _need(state.get("calculation_status") == record.calculation_status.value, f"{label}: state status differs from RAW")
    _need(state.get("failure_reason") == record.failure_reason, f"{label}: state failure reason differs from RAW")
    _need(state.get("failure_stage") == record.failure_stage, f"{label}: state failure stage differs from RAW")
    air = _num(record.payload.get("air_mass_flow_kg_s"))
    _need(air is not None and _same(state.get("air_mass_flow_kg_s"), air), f"{label}: state air flow differs from RAW")
    metric = _raw_net_power(record); fields = ("net_mechanical_power_W", "signed_power_residual_W", "relative_power_residual")
    if metric is None:
        _need(all(state.get(key) is None for key in fields), f"{label}: state without a RAW power metric fabricates power metrics")
    else:
        residual = metric - target
        _need(_same(state.get("net_mechanical_power_W"), metric), f"{label}: state net power differs from RAW")
        _need(_same(state.get("signed_power_residual_W"), residual), f"{label}: state signed residual differs from RAW")
        _need(_same(state.get("relative_power_residual"), abs(residual) / target), f"{label}: state relative residual differs from RAW")
    return record

def _check_states(case: Mapping[str, Any], by_id: Mapping[str, Any], target: float) -> tuple[list[Any], list[Any]]:
    states = case.get("evaluated_states")
    _need(isinstance(states, list) and states, f"{case.get('case')}: evaluated_states are missing or empty")
    _need(case.get("trial_count") == len(states), f"{case.get('case')}: trial_count differs from evaluated_states")
    _need(isinstance(case.get("last_evaluated"), Mapping) and case["last_evaluated"] == states[-1], f"{case.get('case')}: last_evaluated differs from the last state")
    records = [_check_state(case, state, by_id, target) for state in states]
    return states, records

def _failed_case_report(case: Mapping[str, Any], states: list[Any], records: list[Any], target: float) -> dict[str, Any]:
    label, reason, status = case.get("case"), case.get("reason"), case.get("status")
    _need(isinstance(reason, str) and reason, f"{label}: failure case has no reason")
    _need(reason in FAILURE_REASONS, f"{label}: unknown failure reason {reason!r}")
    _need(case.get("root") is None, f"{label}: non-SUCCESS case has a root; ambiguous failure/root")
    _need(case.get("numerical_power_closed") is None, f"{label}: failure case fabricates numerical_power_closed")
    _need(case.get("numerical_closed") is None, f"{label}: failure case fabricates numerical_closed")
    _need(case.get("overall_engineering_feasibility") is None, f"{label}: failure case fabricates engineering feasibility")
    _need(case.get("overall_engineering_feasibility_status") == "NOT_CHECKED", f"{label}: failure case must keep overall status NOT_CHECKED")
    residuals = [None if (metric := _raw_net_power(record)) is None else metric - target for record in records]
    same_sign = None if residuals[0] is None or residuals[-1] is None else ((residuals[0] > 0 and residuals[-1] > 0) or (residuals[0] < 0 and residuals[-1] < 0))
    if reason in OUTER_FAILURE_REASONS:
        _need(status == "NUMERICAL_FAILURE", f"{label}: outer reason {reason} requires NUMERICAL_FAILURE")
    elif reason == TRIAL_FAILURE_REASON:
        final = records[-1]
        _need(final.calculation_status is not CalculationStatus.SUCCESS, f"{label}: trial_failure_at_air_flow but the final RAW record is SUCCESS")
        _need(status == final.calculation_status.value, f"{label}: case status differs from the final failed RAW status")
    elif reason == MISSING_METRIC_REASON:
        _need(status == "MODEL_ERROR", f"{label}: success_missing_power_metric requires MODEL_ERROR")
        _need(records[-1].calculation_status is CalculationStatus.SUCCESS and _raw_net_power(records[-1]) is None,
            f"{label}: success_missing_power_metric requires a SUCCESS final RAW record with no power metric")
    if reason == BOUNDARY_REASON:
        _need(len(states) >= 2, f"{label}: not-bracketed claim requires at least two evaluated states")
        _need(residuals[0] is not None and residuals[-1] is not None, f"{label}: not-bracketed claim lacks actual endpoint power residuals")
        _need(residuals[0] != 0 and residuals[-1] != 0 and same_sign is True,
            f"{label}: {BOUNDARY_REASON} claimed but endpoint power residuals are zero or have different signs")
    outer = {"reason": reason, "first_signed_power_residual_W": residuals[0], "last_signed_power_residual_W": residuals[-1],
        "endpoint_same_sign": same_sign, "basis": "ACTUAL_RAW_POWER_METRICS_NO_CAUSE_INFERENCE"}
    partial = [state.get("candidate_id") for state, record in zip(states, records) if _has_partial_gas_generator(record)]
    return {"case": label, "status": status, "reason": reason, "architecture": case.get("architecture"),
        "source_point_index": case.get("source_point_index"), "target_gearbox_input_power_W": target, "trial_count": len(states),
        "evaluated_states": states, "last_evaluated": states[-1], "root": None, "numerical_closed": None, "numerical_power_closed": None,
        "overall_engineering_feasibility": None, "overall_engineering_feasibility_status": "NOT_CHECKED",
        "outer_residual_check": outer, "partial_gas_generator_candidate_ids": partial,
        "partial_gas_generator_basis": "FAILED_RAW_RECORD_OUTPUTS_CONTAIN_GAS_GENERATOR_METADATA; OUTER_CASE_HAS_NO_ROOT",
        "failure_interpretation": "TYPED_FAILURE_PRESERVED_NOT_PHYSICAL_INFEASIBILITY"}

def audit_dataset(dataset: str | Path) -> dict[str, Any]:
    path = Path(dataset); _need(path.is_dir(), f"dataset directory does not exist: {path}"); manifest = load_manifest(path); _need(manifest.get("complete") is True, "dataset manifest is not complete")
    records = read_records(path); _need(len(records) == manifest["record_count"], "RAW record count differs from the verified manifest")
    summary_path = path / SUMMARY_NAME; _need(summary_path.is_file(), f"missing flow summary: {summary_path}"); summary_bytes = summary_path.read_bytes()
    try: summary = json.loads(summary_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc: raise AuditError(f"cannot read flow summary: {exc}") from exc
    _need(isinstance(summary, Mapping), "flow summary must be an object"); run = summary.get("run") or {}
    _need(run.get("run_id") == manifest["run_id"], "summary run_id does not match the verified manifest"); _need(run.get("journal_sha256") == manifest["journal_sha256"], "summary journal hash does not match the verified manifest")
    _need(run.get("record_count") == manifest["record_count"], "summary record count does not match the verified manifest"); _need(run.get("complete") is True, "summary run is not complete")
    cases = summary.get("cases"); _need(isinstance(cases, list) and cases, "flow summary has no cases")
    _need(len({case.get("case") for case in cases if isinstance(case, Mapping)}) == len(cases), "duplicate or malformed case names in summary")
    by_id = {record.candidate_id: record for record in records}; module, digest, book_path = _load_book(); meta = _book_metadata(module, digest, book_path)
    roots, failed_cases, root_ids, outer_statuses, inner_statuses, state_ids = [], [], [], [], [], []
    for case in cases:
        _need(isinstance(case, Mapping), "summary case must be an object")
        _need(isinstance(case.get("case"), str) and case.get("case"), "summary case has no name")
        status = case.get("status"); status = "SUCCESS" if status is None and isinstance(case.get("root"), Mapping) else status
        _need(status in CASE_STATUSES, f"case {case.get('case')}: unknown case status {status!r}"); outer_statuses.append(status)
        root = case.get("root")
        if status != "SUCCESS": _need(root is None, f"{case.get('case')}: non-SUCCESS case must not have a root; ambiguous failure/root")
        else: _need(isinstance(root, Mapping), f"case {case.get('case')}: missing root")
        target = _num(case.get("target_gearbox_input_power_W")); _need(target is not None and target > 0, f"{case.get('case')}: missing positive target power")
        states, state_records = _check_states(case, by_id, target); inner_statuses.extend(state.get("calculation_status") for state in states); state_ids.extend(state.get("candidate_id") for state in states)
        if status != "SUCCESS":
            failed_cases.append(_failed_case_report(case, states, state_records, target)); continue
        identifier = root.get("candidate_id"); _need(identifier in by_id, f"root reference not found: {identifier!r}")
        _need(identifier in {state.get("candidate_id") for state in states}, f"{case.get('case')}: root is not one of its evaluated states")
        root_ids.append(identifier); roots.append(_root_report(case, by_id[identifier], meta))
    _need(len(set(root_ids)) == len(root_ids), "duplicate root ids in summary")
    accounted = len(state_ids) == len(set(state_ids)) and set(state_ids) == set(by_id)
    _need(accounted, "evaluated states must reference every verified RAW candidate exactly once")
    identity = {"audit_version": AUDIT_VERSION, "source_metadata_sha256": digest, "journal_sha256": manifest["journal_sha256"], "run_id": manifest["run_id"], "root_ids": root_ids, "case_statuses": dict(zip((case["case"] for case in cases), outer_statuses)), "summary_sha256": _sha(summary_bytes)}
    return {"audit_version": AUDIT_VERSION, "dataset": str(path.resolve()), "source_metadata": meta,
        "run": {"run_id": manifest["run_id"], "journal_sha256": manifest["journal_sha256"], "record_count": manifest["record_count"], "complete": True, "summary_sha256": identity["summary_sha256"], "root_ids": root_ids},
        "audit_digest": _digest(identity), "roots": roots, "failed_cases": failed_cases, "case_count": len(cases), "root_count": len(roots), "failed_count": len(failed_cases),
        "evaluated_state_count": len(state_ids), "all_raw_records_accounted_for": accounted,
        "outer_case_status_counts": _counts(outer_statuses), "inner_trial_status_counts": _counts(inner_statuses),
        "case_analysis": "ALL_PLANNED_CASES; ROOT_REPORTS_ONLY_FOR_SUCCESS; TYPED_FAILURES_PRESERVED", "claim_scope": "NUMERICAL_CLOSURE_AND_POSTHOC_ENGINEERING_MARGINS_ONLY", "physical_validation": "NOT_PERFORMED",
        "not_checked": list(NOT_CHECKED_ITEMS), "overall_engineering_feasibility": None, "overall_engineering_feasibility_status": "NOT_CHECKED", "best_feasible_recommendation": None}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit saved flow-sized roots; recomputes source 0D compressor arithmetic, never reruns the meanline or changes RAW.")
    parser.add_argument("--dataset", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(argv); output = Path(args.output)
    try:
        if output.exists(): raise AuditError(f"output already exists; refusing to overwrite: {output}")
        report = audit_dataset(args.dataset); output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False); handle.write("\n")
    except (AuditError, DatasetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
