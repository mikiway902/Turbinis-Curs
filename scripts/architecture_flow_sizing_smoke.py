#!/usr/bin/env python3
"""Bounded explicit-air-flow sizing of the three illustrative coupled cases.

The source cycle book (printed p.22 section 3.2) sizes flow from specific work:
``N_specific = N/G_air`` and ``N = (L_t*G_gas*eta_m - L_c*G_air)*eta_reducer``.
At the user-confirmed 110 kW pre-reducer plane the downstream ``eta_reducer``
is omitted.  The saved 0D cycle flow is therefore only a preliminary value, not
a sized design flow.

By default this smoke keeps Tg, compressor pressure ratios, fuel/air ratio,
speed and hub controls fixed from the coupled fixture and solves one unknown:
air mass flow.  An explicit case may instead carry source compressor pressure
ratios built by ``with_compressor_pressure_ratios``; its source flow metadata,
targets and brackets are then recomputed from that actual 0D variant, never
fitted.  ``source_point_index`` identifies the stage-count template point, not a
fixed pressure reference.
Each outer trial inserts the explicit ``air_mass_flow_kg_s`` into the existing
coupled evaluator, which rebuilds and re-solves the actual meanline stages plus
the GG/FPT closures at that flow.  Stored stage powers are never rescaled from
the 0D result.  Every trial is persisted as a normal RAW record.

The outer search is a deterministic bracketed bisection on the actual net
mechanical power.  Brackets are explicit and are never widened.  A missing
bracket or an iteration limit is a typed NUMERICAL_FAILURE with the evaluated
states, not engine infeasibility.  A typed inner trial failure stops that case
and is preserved.  Cases are independent.  Maps, cooling, strength and overall
engineering feasibility remain NOT_CHECKED; nothing here is physical validation.
"""
from __future__ import annotations

import argparse, hashlib, json, math, sys, time
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "scripts", ROOT / "Optimizer"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import architecture_coupled_stage_smoke as coupled  # noqa: E402
import architecture_stage_smoke as stage_smoke  # noqa: E402
import optimize_extended as oe  # noqa: E402
from turbine_engine.architecture_cycle import calculate_architecture_cycle  # noqa: E402
from turbine_engine.architecture_raw import payload_to_input  # noqa: E402
from turbine_engine.dataset import (  # noqa: E402
    DatasetError, DatasetWriter, load_manifest, read_records)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus, canonical_hash, copy_json_value, make_candidate_id)

EXPERIMENT_VERSION = "architecture-flow-sizing-v1"
CONFIG_VERSION = "architecture-flow-sizing-config-v1"
SUMMARY_VERSION = "architecture-flow-sizing-summary-v1"
EVALUATOR_VERSION = "architecture-flow-sizing-smoke-v1"
MODEL_VARIANT = "architecture_flow_sizing_smoke_v1"
PLAN_NAME, SUMMARY_NAME = "three-fixed-cases-explicit-air-flow", "summary.json"
NEW_SMOKE_FILE = "scripts/architecture_flow_sizing_smoke.py"
MODEL_SOURCE_FILES = tuple(coupled.MODEL_SOURCE_FILES) + (NEW_SMOKE_FILE,)
TARGET = float(coupled.TARGET)
BRACKET_FACTORS = (0.9, 1.4)
OUTER_RELATIVE_TOL = 1e-6
OUTER_MAX_ITER = 30
BOUNDARY_CLOSURE_RELATIVE_TOL = coupled.POWER_CLOSURE_RELATIVE_TOL
CASES = (("A_MAIN_3", 2), ("B_MAIN_3", 5), ("C_GG3_FPT1", 12))
SIZING_BASIS = "CYCLE_BOOK_PRINTED_P22_SPECIFIC_POWER_PRE_REDUCER"
CASE_KEYS = frozenset({
    "case", "source_point_index", "assembly", "coupling", "architecture",
    "source_payload", "source_payload_sha256", "source_air_mass_flow_kg_s",
    "source_gas_mass_flow_kg_s", "source_fuel_mass_flow_kg_s",
    "air_flow_bracket_kg_s", "target_gearbox_input_power_W",
    "outer_relative_tolerance", "outer_max_iterations"})


class FlowSizingSmokeError(RuntimeError):
    pass


def _need(condition, message):
    if not condition:
        raise FlowSizingSmokeError(message)


def _digest(relatives):
    digest = hashlib.sha256()
    for relative in relatives:
        digest.update(relative.encode("utf-8") + b"\0")
        with open(ROOT / relative, "rb") as handle:
            file_hash = hashlib.sha256()
            for block in iter(lambda: handle.read(1 << 20), b""):
                file_hash.update(block)
            digest.update(file_hash.digest())
        digest.update(b"\0")
    return digest.hexdigest()


def plan_cases():
    """Return the fixed three default cases with explicit source flow/bracket.

    ``source_point_index`` is the stage-count template point; the default
    payload keeps the coupled fixture's source compressor pressure ratios.
    """
    source_payloads = coupled.plan_payloads()
    cases = []
    for name, index in CASES:
        payload = copy_json_value(source_payloads[index])
        result = calculate_architecture_cycle(
            payload_to_input(payload["architecture_cycle_source"]))
        air = float(result.air_mass_flow)
        cases.append({
            "case": name, "source_point_index": index,
            "assembly": payload["assembly"], "coupling": payload["coupling"],
            "architecture": payload["architecture"],
            "source_payload": payload,
            "source_payload_sha256": canonical_hash(payload),
            "source_air_mass_flow_kg_s": air,
            "source_gas_mass_flow_kg_s": float(result.gas_mass_flow),
            "source_fuel_mass_flow_kg_s": float(result.fuel_mass_flow),
            "air_flow_bracket_kg_s": [BRACKET_FACTORS[0] * air,
                                       BRACKET_FACTORS[1] * air],
            "target_gearbox_input_power_W": TARGET,
            "outer_relative_tolerance": OUTER_RELATIVE_TOL,
            "outer_max_iterations": OUTER_MAX_ITER,
        })
    return cases


def _same_physical_base(payload, base_payload):
    """Compare to the expected base/ratio variant, allowing only explicit controls."""
    if set(payload) != set(base_payload):
        return False
    for key in set(base_payload) - {"stage_controls", "brackets"}:
        if payload[key] != base_payload[key]:
            return False
    if set(payload["brackets"]) != set(base_payload["brackets"]):
        return False
    controls, base_controls = (dict(payload["stage_controls"]),
                               dict(base_payload["stage_controls"]))
    for key in ("operating_points", "design_overrides",
                "reaction_table_policy"):
        controls.pop(key, None)
        base_controls.pop(key, None)
    return controls == base_controls


def _validate_cases(cases):
    """Bind each case to the coupled base or an explicit source-ratio variant."""
    _need(isinstance(cases, (list, tuple)) and bool(cases),
          "cases must be a non-empty sequence")
    base = coupled.plan_payloads()
    names, sources = set(), set()
    for case in cases:
        _need(isinstance(case, Mapping) and set(case) == CASE_KEYS,
              "case keys changed")
        name = case["case"]
        _need(isinstance(name, str) and name, "case name must be non-empty")
        _need(name not in names, f"duplicate case name {name!r}")
        names.add(name)
        index = case["source_point_index"]
        _need(type(index) is int and 0 <= index < len(base),
              "source_point_index is not an actual base point")
        payload = case["source_payload"]
        _need(isinstance(payload, Mapping), "source_payload must be an object")
        try:
            point = coupled._validate_payload(payload)
        except (coupled.ArchitectureCoupledSmokeError, ValueError, TypeError,
                KeyError) as exc:
            raise FlowSizingSmokeError(f"source_payload invalid: {exc}") from exc
        _need(point.index == index,
              "source_point_index differs from payload point_index")
        _need(point.architecture.value == case["architecture"]
              and point.assembly == case["assembly"]
              and point.coupling == case["coupling"],
              "case architecture/assembly differs from source_payload")
        try:
            expected = coupled.with_compressor_pressure_ratios(
                base[index],
                payload["architecture_cycle_source"][
                    "compressor_pressure_ratios"])
        except (coupled.ArchitectureCoupledSmokeError, ValueError, TypeError,
                KeyError) as exc:
            raise FlowSizingSmokeError(
                f"source_payload pressure ratios invalid: {exc}") from exc
        _need(_same_physical_base(payload, expected),
              "source_payload differs from the actual base or pressure variant")
        source_hash = canonical_hash(payload)
        _need(source_hash not in sources,
              "duplicate source_payload across explicit cases")
        sources.add(source_hash)
        _need(case["source_payload_sha256"] == source_hash,
              "source_payload_sha256 does not match source_payload")
        result = calculate_architecture_cycle(
            payload_to_input(payload["architecture_cycle_source"]))
        for key, value in (
                ("source_air_mass_flow_kg_s", result.air_mass_flow),
                ("source_gas_mass_flow_kg_s", result.gas_mass_flow),
                ("source_fuel_mass_flow_kg_s", result.fuel_mass_flow)):
            _need(not isinstance(case[key], bool)
                  and isinstance(case[key], (int, float))
                  and math.isfinite(float(case[key]))
                  and float(case[key]) == float(value),
                  f"{key} differs from the recomputed source cycle")
        _need(not isinstance(case["target_gearbox_input_power_W"], bool)
              and case["target_gearbox_input_power_W"] == TARGET,
              "target power must equal the fixed 110000 W target")
        pair = case["air_flow_bracket_kg_s"]
        _need(isinstance(pair, (list, tuple)) and len(pair) == 2,
              "air_flow_bracket_kg_s must be a pair")
        _need(all(not isinstance(item, bool)
                  and isinstance(item, (int, float))
                  and math.isfinite(float(item)) and float(item) > 0.0
                  for item in pair) and float(pair[0]) < float(pair[1]),
              "air_flow_bracket_kg_s must be finite, positive and ordered")
        _need(case["outer_relative_tolerance"] == OUTER_RELATIVE_TOL
              and case["outer_max_iterations"] == OUTER_MAX_ITER,
              "outer solver controls changed")
    return list(cases)


def make_run_config(cases, constants, model_digest):
    locked, base = [], coupled.plan_payloads()
    for case in cases:
        points = case["source_payload"]["stage_controls"].get(
            "operating_points")
        item = {
            "case": case["case"],
            "source_point_index": case["source_point_index"],
            "architecture": case["architecture"], "assembly": case["assembly"],
            "coupling": case["coupling"],
            "source_payload_sha256": case["source_payload_sha256"],
            "source_air_mass_flow_kg_s": case["source_air_mass_flow_kg_s"],
            "source_gas_mass_flow_kg_s": case["source_gas_mass_flow_kg_s"],
            "source_fuel_mass_flow_kg_s": case["source_fuel_mass_flow_kg_s"],
            "air_flow_bracket_kg_s": case["air_flow_bracket_kg_s"],
            "target_gearbox_input_power_W":
                case["target_gearbox_input_power_W"],
            "outer_relative_tolerance": case["outer_relative_tolerance"],
            "outer_max_iterations": case["outer_max_iterations"]}
        source_cycle = case["source_payload"]["architecture_cycle_source"]
        ratios = list(source_cycle["compressor_pressure_ratios"])
        index = case["source_point_index"]
        if ratios != list(base[index]["architecture_cycle_source"][
                "compressor_pressure_ratios"]):
            item["compressor_pressure_ratios"] = ratios
            item["compressor_pressure_ratio_product"] = math.prod(ratios)
            item["source_cycle_pressure_provenance"] = (
                "EXPLICIT_SOURCE_COMPRESSOR_PRESSURE_RATIOS")
        design = case["source_payload"]["stage_controls"].get(
            "design_overrides")
        policy = case["source_payload"]["stage_controls"].get(
            "reaction_table_policy")
        if points is not None or design is not None or policy is not None:
            item["stage_controls_sha256"] = canonical_hash(
                case["source_payload"]["stage_controls"])
            if points is not None:
                item["operating_points"] = copy_json_value(points)
            if design is not None:
                item["design_overrides"] = copy_json_value(design)
            if policy is not None:
                item["reaction_table_policy"] = policy
        locked.append(item)
    return {
        "experiment_version": EXPERIMENT_VERSION, "config_version": CONFIG_VERSION,
        "plan_name": PLAN_NAME, "case_count": len(cases),
        "plan_sha256": canonical_hash(locked), "cases": locked,
        "model_variant": MODEL_VARIANT, "evaluator_version": EVALUATOR_VERSION,
        "model_digest": model_digest,
        "model_digest_sources": list(MODEL_SOURCE_FILES),
        "boundary_closure_relative_tolerance": BOUNDARY_CLOSURE_RELATIVE_TOL,
        "sizing_basis": SIZING_BASIS,
        "source_book": "cycle textbook printed p.22 section 3.2",
        "source_equation": "N_specific=N/G_air; "
                           "N=(L_t*G_gas*eta_m-L_c*G_air)*eta_reducer",
        "pre_reducer": True, "eta_reducer_applied": False,
        "source_flow_role": "PRELIMINARY_0D_NOT_A_SIZED_DESIGN_FLOW",
        "stage_constants": json.loads(json.dumps(constants, allow_nan=False)),
        "stage_count_automatic_selection": "NONE",
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
    }


def _metric(record):
    if record.calculation_status is not CalculationStatus.SUCCESS:
        return None
    power = (record.outputs or {}).get("power")
    if not isinstance(power, Mapping):
        return None
    value = power.get("net_mechanical_power_W")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _state(air, record):
    metric = _metric(record)
    return {
        "air_mass_flow_kg_s": float(air), "candidate_id": record.candidate_id,
        "calculation_status": record.calculation_status.value,
        "failure_stage": record.failure_stage,
        "failure_reason": record.failure_reason,
        "net_mechanical_power_W": metric,
        "signed_power_residual_W": None if metric is None else metric - TARGET,
        "relative_power_residual": None if metric is None
        else abs(metric - TARGET) / TARGET,
    }


def _case_result(case, status, reason, states, root_record=None):
    root = None
    power = {}
    if root_record is not None:
        power = (root_record.outputs or {}).get("power") or {}
        flow = (root_record.outputs or {}).get("flow_sizing") or {}
        root = {
            "candidate_id": root_record.candidate_id,
            "air_mass_flow_kg_s": float(root_record.payload["air_mass_flow_kg_s"]),
            "gas_mass_flow_kg_s": flow.get("trial_gas_mass_flow_kg_s"),
            "fuel_mass_flow_kg_s": flow.get("trial_fuel_mass_flow_kg_s"),
            "net_mechanical_power_W": power.get("net_mechanical_power_W"),
            "signed_power_residual_W": power.get("signed_power_residual_W"),
            "relative_power_residual": power.get("relative_power_residual"),
            "numerical_power_closed": power.get("engine_power_closed"),
            "numerical_power_label": power.get("engine_power_label"),
        }
    return {
        "case": case["case"], "source_point_index": case["source_point_index"],
        "architecture": case["architecture"], "assembly": case["assembly"],
        "coupling": case["coupling"], "status": status, "reason": reason,
        "target_gearbox_input_power_W": TARGET,
        "air_flow_bracket_kg_s": list(case["air_flow_bracket_kg_s"]),
        "numerical_power_closed": None if root is None
        else root["numerical_power_closed"],
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "trial_count": len(states), "evaluated_states": states,
        "last_evaluated": states[-1] if states else None, "root": root,
    }


def _solve_case(case, trial, *, max_iter=None, relative_tol=None):
    """Deterministic bisection using only the trial record's actual outcome."""
    max_iter = OUTER_MAX_ITER if max_iter is None else int(max_iter)
    relative_tol = OUTER_RELATIVE_TOL if relative_tol is None else float(relative_tol)
    lo, hi = (float(value) for value in case["air_flow_bracket_kg_s"])
    tolerance = relative_tol * TARGET
    states = []

    def evaluate(air):
        record = trial(air)
        state = _state(air, record)
        states.append(state)
        return record, _metric(record)

    def failure(reason):
        return _case_result(case, "NUMERICAL_FAILURE", reason, states)

    def failed(record):
        if record.calculation_status is CalculationStatus.SUCCESS:
            return _case_result(case, "MODEL_ERROR",
                                "success_missing_power_metric", states)
        return _case_result(case, record.calculation_status.value,
                            "trial_failure_at_air_flow", states)

    record_lo, metric_lo = evaluate(lo)
    if metric_lo is None:
        return failed(record_lo)
    if abs(metric_lo - TARGET) <= tolerance:
        return _case_result(case, "SUCCESS", "root_at_bracket_low", states,
                            record_lo)
    record_hi, metric_hi = evaluate(hi)
    if metric_hi is None:
        return failed(record_hi)
    if abs(metric_hi - TARGET) <= tolerance:
        return _case_result(case, "SUCCESS", "root_at_bracket_high", states,
                            record_hi)
    if (metric_lo < TARGET) == (metric_hi < TARGET):
        return failure("outer_bracket_not_bracketed")
    for _ in range(max_iter):
        mid = lo + 0.5 * (hi - lo)
        if not (lo < mid < hi):
            return failure("stalled_adjacent_floats")
        record, metric = evaluate(mid)
        if metric is None:
            return failed(record)
        if abs(metric - TARGET) <= tolerance:
            return _case_result(case, "SUCCESS", "root_found", states, record)
        if (metric_lo < TARGET) == (metric < TARGET):
            lo, metric_lo = mid, metric
        else:
            hi = mid
    return failure("outer_iteration_limit")


def _open_writer(path, resume, config):
    if path.exists() and any(path.iterdir()):
        if not resume:
            raise FlowSizingSmokeError("destination is not empty")
        return DatasetWriter.open(path, expected_run_config=config,
                                  expected_evaluator_version=EVALUATOR_VERSION)
    if resume:
        raise FlowSizingSmokeError("cannot resume an empty/missing dataset")
    return DatasetWriter.create(path, run_config=config)


def _trial(case, air, writer, existing, evaluate):
    payload = copy_json_value(case["source_payload"])
    payload["air_mass_flow_kg_s"] = float(air)
    identifier = make_candidate_id(case["architecture"], MODEL_VARIANT, payload)
    if identifier in existing:
        return existing[identifier]
    record = writer.evaluate_and_append(
        evaluate, payload, architecture=case["architecture"],
        sampling_method="explicit_air_flow_outer_bisection",
        sampling_metadata={
            "plan": PLAN_NAME, "case": case["case"],
            "source_point_index": case["source_point_index"],
            "air_mass_flow_kg_s": float(air),
            "outer_target_power_W": TARGET,
            "outer_relative_tolerance": OUTER_RELATIVE_TOL,
            "sizing_basis": SIZING_BASIS})
    existing[identifier] = record
    return record


def _build_summary(cases, case_results, manifest, appended, elapsed, reproduce,
                   model_digest, plan_sha256):
    return {
        "report_version": SUMMARY_VERSION, "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "plan_name": PLAN_NAME, "sizing_basis": SIZING_BASIS,
        "source_book": "cycle textbook printed p.22 section 3.2",
        "pre_reducer": True,
        "experiment": {
            "plan_name": PLAN_NAME, "case_count": len(cases),
            "plan_sha256": plan_sha256, "model_digest": model_digest},
        "run": {"run_id": manifest["run_id"],
                "record_count": manifest["record_count"],
                "complete": manifest["complete"],
                "journal_sha256": manifest["journal_sha256"],
                "records_appended_this_invocation": appended,
                "elapsed_seconds": round(elapsed, 6)},
        "cases": case_results,
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "reproduce": copy_json_value(reproduce),
    }


def run_plan(destination, *, resume=False, evaluator=None, model=None,
             constants=None, model_digest=None, reproduce=True, cases=None):
    """Solve the explicit-air-flow cases and persist every trial.

    ``cases=None`` keeps the original three-case plan.  An explicit
    ``cases`` sequence is validated against the real coupled base payloads and
    locked by hash; it does not choose an optimum or adapt brackets.
    """
    cases = _validate_cases(plan_cases() if cases is None else cases)
    if constants is None:
        if model is None:
            model = coupled._get_model()
        template, _ = oe.build_ns(
            model, gas_property_basis=coupled.TEXTBOOK,
            cycle_gas_property_basis=coupled.TEXTBOOK)
        constants = stage_smoke._stage_constants(template)
    model_digest = model_digest or _digest(MODEL_SOURCE_FILES)
    if evaluator is None:
        if model is None:
            model = coupled._get_model()
        evaluate = lambda item: coupled.evaluate_coupled_payload(
            item, model=model)
    else:
        evaluate = evaluator
    config = make_run_config(cases, constants, model_digest)
    path = Path(destination)
    writer = _open_writer(path, resume=resume, config=config)
    existing = ({record.candidate_id: record for record in read_records(path)}
                if resume else {})
    results, appended, started, completed = [], 0, time.perf_counter(), False
    try:
        for case in cases:
            before = writer.record_count
            results.append(_solve_case(
                case, lambda air, c=case: _trial(
                    c, air, writer, existing, evaluate)))
            appended += writer.record_count - before
        completed = True
    finally:
        writer.close(finalize=completed)
    elapsed = time.perf_counter() - started
    records = read_records(path)
    _need(len(records) == sum(item["trial_count"] for item in results),
          "journal records differ from replayed trials")
    manifest = load_manifest(path)
    reproduce_result = (stage_smoke._reproduce_all(records, evaluate)
                        if reproduce else {
                            "checked": 0, "matches": None, "differences": [],
                            "rtol": 1e-12, "atol": 1e-9})
    summary = _build_summary(cases, results, manifest, appended, elapsed,
                             reproduce_result, model_digest,
                             config["plan_sha256"])
    stage_smoke._write_summary_atomic(path / SUMMARY_NAME, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Bounded explicit-air-flow sizing of three illustrative coupled cases; "
        "not a sweep, design selection, or physical validation."))
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_plan(args.destination, resume=args.resume)
    except (FlowSizingSmokeError, DatasetError,
            stage_smoke.ArchitectureStageSmokeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False))
    return 3 if summary["reproduce"]["matches"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
