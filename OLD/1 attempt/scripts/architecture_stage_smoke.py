#!/usr/bin/env python3
"""Bounded source-backed stage-count / const-hub meanline smoke.
Methodological fixture only: A MAIN (dual-CC), B MAIN (single-CC), C GG and
C FPT at explicit counts 1/2/3.  Inlet state/flow come from textbook 0D
architecture cycles; each attempt builds a fresh textbook namespace and calls
``распределить_π_const_hub`` directly (never ``eval_candidate``).  Aggregate
outlet total is translated to static with fixture ``M_t=0.3379`` (not solved by
the meanline); stage total pressure is reported against it without forcing
closure.  All attempts/typed failures are stored; plan hash, source digest,
constants and assumptions lock the run config.  Interrupted runs leave the
manifest incomplete and resume appends zero duplicates.  The C GG->FPT 0D
The item-54 estimate is unrounded and never selects a count.

The fixed plan defaults to the explicit ``raw_unbounded`` pressure-ratio policy
(no engineering pi clip, p-expansion domain ``pi > 1`` only).  The CLI can also
run the unmodified ``legacy_clipped`` reference policy for comparison.  The
post-hoc pi 1.8..2.4 vector is named ``PROMPT_STAGE_PI_1_8_2_4``, is an
engineering profile rather than a book fundamental, and its profile ``pass``
depends only on the per-stage 1.8..2.4 range checks.  Overall engineering
feasibility remains a separate ``None`` / ``NOT_CHECKED`` result because the
strength verification is missing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "scripts", ROOT / "Optimizer"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import optimize_extended as oe  # noqa: E402
import optimizer_core as oc  # noqa: E402
from architecture_smoke import GEARBOX, P_T_OUT, TARGET, request_for  # noqa: E402
from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS,
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK as TEXTBOOK,
    ArchitectureCycleError, ArchitectureCyclePhysicalError, calculate_architecture_cycle)
from turbine_engine.architecture_raw import input_to_payload, payload_to_input  # noqa: E402
from turbine_engine.baseline import estimate_stage_count  # noqa: E402
from turbine_engine.cycle import CycleSolverError  # noqa: E402
from turbine_engine.dataset import (  # noqa: E402
    DatasetError, DatasetWriter, compare_candidates, load_manifest, read_records)
from turbine_engine.diagnostics import (  # noqa: E402
    NumericalConvergenceError, SolverDiagnosticsError)
from turbine_engine.orchestration import (  # noqa: E402
    PRESSURE_RATIO_LEGACY_CLIPPED, PRESSURE_RATIO_RAW_UNBOUNDED,
    build_const_hub_orchestration, validate_pressure_ratio_policy)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus, EvaluationOutcome, PayloadValidationError, canonical_hash,
    copy_json_value, evaluate_candidate, make_candidate_id)

EXPERIMENT_VERSION = "architecture-stage-meanline-v1"
CONFIG_VERSION = "architecture-stage-meanline-config-v1"
SUMMARY_VERSION = "architecture-stage-meanline-summary-v1"
EVALUATOR_VERSION = "architecture-stage-meanline-smoke-v1"
MODEL_VARIANT = "architecture_stage_meanline_smoke_v1"
PLAN_NAME = "four-assemblies-three-stage-counts"
SUMMARY_NAME = "summary.json"
BACKEND = "reference_python_float64"
INPUT_SCHEMA_VERSION = "architecture-stage-meanline-input-v1"
OUTPUT_SCHEMA_VERSION = "architecture-stage-meanline-output-v1"
COUPLING_LABEL = "NOT_COUPLED_ARCHITECTURE_STAGE"
MAX_PLAN_POINTS, STAGE_COUNTS = 12, (1, 2, 3)
SPEED_RPM, HUB_DIAMETER_M, REFERENCE_EXIT_MACH = 84_000.0, 0.065, 0.3379
ALPHA1_ANCHOR, KAPPA_ANCHOR = (11.0, 11.0, 19.0), (1.38, 1.12, 1.15)
PI_WEIGHTS_ANCHOR = (0.75, 0.8, 1.0)
DEFAULT_PRESSURE_RATIO_POLICY = PRESSURE_RATIO_RAW_UNBOUNDED
PRESSURE_RATIO_POLICY_CHOICES = (
    PRESSURE_RATIO_RAW_UNBOUNDED, PRESSURE_RATIO_LEGACY_CLIPPED)
PI_ACCEPTANCE_PROFILE = "PROMPT_STAGE_PI_1_8_2_4"
PI_ACCEPTANCE_BOUNDS = (1.8, 2.4)
ASSEMBLY_ORDER = ("A_MAIN", "B_MAIN", "C_GG", "C_FPT")

ASSEMBLY_SPECS = {
    "A_MAIN": (Architecture.SINGLE_SHAFT_DUAL_CC, "main", 0),
    "B_MAIN": (Architecture.SINGLE_SHAFT_SINGLE_CC, "main", 0),
    "C_GG": (Architecture.FREE_POWER_TURBINE, "gas_generator", 0),
    "C_FPT": (Architecture.FREE_POWER_TURBINE, "free_power", 1),
}

MODEL_SOURCE_FILES = (
    "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb", "Turbinis-Cycle/Turbinis-Cycle.ipynb", "Turbinis-Calc/cycle_link.py", "Optimizer/optimizer_core.py",
    "Optimizer/optimize_extended.py", "scripts/architecture_smoke.py", "scripts/architecture_stage_smoke.py", "turbine_engine/architecture.py",
    "turbine_engine/architecture_cycle.py", "turbine_engine/architecture_raw.py", "turbine_engine/baseline.py", "turbine_engine/compressor.py",
    "Turbinis-Calc/constraints.py", "Turbinis-Calc/sizing.py", "turbine_engine/constraints.py", "turbine_engine/cycle.py", "turbine_engine/dataset.py", "turbine_engine/diagnostics.py",
    "turbine_engine/expansion.py", "turbine_engine/gas.py", "turbine_engine/geometry.py", "turbine_engine/orchestration.py",
    "turbine_engine/performance.py", "turbine_engine/raw.py", "turbine_engine/reaction.py", "turbine_engine/shaft.py",
    "turbine_engine/stage.py", "turbine_engine/turbine.py",
)

ASSUMPTIONS = {
    "fixture_source": "architecture_smoke.request_for(... textbook basis)",
    "inlet": [288.0, 101_300.0], "combustor": ["legacy", 1250.0, 0.96],
    "target_gearbox_input_power_W": TARGET, "gearbox_efficiency": GEARBOX,
    "compressor_polytropic_efficiency": 0.85,
    "turbine_polytropic_efficiency": 0.90,
    "turbine_mechanical_efficiency": 0.99,
    "turbine_outlet_total_pressure_Pa": P_T_OUT,
    "source_pressure_ratios": {
        "A_MAIN": [math.sqrt(11.0)] * 2, "B_MAIN": [11.0],
        "C_GG": [8.0], "C_FPT": [8.0],
    },
    "fixed_rotational_speed_rpm": SPEED_RPM,
    "fixed_hub_diameter_m": HUB_DIAMETER_M,
    "speed_basis": "fixed regression anchor, not a validated/selected optimum; "
                   "C GG uses the same compressor/GG caller assumption and C FPT "
                   "receives the same value solely as a fixture",
    "stage_arrays": {
        "1": [[11.0], [1.38], [1.0]],
        "2": [[11.0, 11.0], [1.38, 1.12], [0.75, 1.0]],
        "3": [[11.0, 11.0, 19.0], [1.38, 1.12, 1.15], [0.75, 0.8, 1.0]],
    },
    "boundary": {
        "method": "static from aggregate total outlet and fixture Mach",
        "reference_exit_mach": REFERENCE_EXIT_MACH,
        "mach_source": "CycleConfig.M_t fixture assumption",
        "mach_solved_by_meanline": False,
        "formula": "p_total/(1+0.5*(k-1)*M**2)**(k/(k-1))",
    },
    "stage_count_estimate": {
        "formula": "(c0**2/v**2-1)/(u_mean**2/(v**2*y0**2)-1)",
        "v": "mu*mean(actual stage c2)",
        "u_mean": "0.5*(first_stage_u1+last_stage_u2)",
        "automatic_selection": "NONE",
        "c0_basis": "PRELIMINARY_0D_END_STATE_TOTAL_TO_STATIC_ASSUMPTION",
    },
    "coupling_label": COUPLING_LABEL,
    "c_gg_to_fpt_0d_state_transfer": "CORRECT_0D_GG_OUTLET_TO_FPT_INLET",
    "meanline_coupling": "NOT_COUPLED_ARCHITECTURE_STAGE",
    "physically_validated": False, "illustrative_only": True,
}


_MODEL: Any | None = None


class ArchitectureStageSmokeError(RuntimeError):
    pass


class StageNoSolution(Exception):

    def __init__(self, reason, last_valid_state=None):
        self.reason, self.last_valid_state = reason, dict(last_valid_state or {})
        super().__init__(reason)


class StageNumericalFailure(Exception):

    def __init__(self, path, value):
        self.path, self.value, self.partial = path, value, {}
        super().__init__(f"non-finite stage value at {path}: {value!r}")


@dataclass(frozen=True)
class PlanPoint:
    index: int
    assembly: str
    stage_count: int
    architecture: Architecture
    turbine_role: str
    turbine_index: int


def _get_model():
    global _MODEL
    _MODEL = _MODEL or oc.load_model()
    return _MODEL


def _stage_controls(count):
    if isinstance(count, bool) or count not in STAGE_COUNTS:
        raise ArchitectureStageSmokeError(f"bad stage_count {count!r}")
    return {
        "alpha1_deg": list(ALPHA1_ANCHOR[:count]),
        "kappa": list(KAPPA_ANCHOR[:count]),
        "pi_weights": [*PI_WEIGHTS_ANCHOR[: count - 1], 1.0],
    }


def plan_points() -> tuple[PlanPoint, ...]:
    points = []
    for assembly in ASSEMBLY_ORDER:
        architecture, role, index = ASSEMBLY_SPECS[assembly]
        for count in STAGE_COUNTS:
            points.append(PlanPoint(len(points), assembly, count, architecture, role, index))
    if len(points) != MAX_PLAN_POINTS:
        raise ArchitectureStageSmokeError("internal plan size changed")
    return tuple(points)


def _payload_policy(payload):
    return payload.get("stage_controls", {}).get(
        "pressure_ratio_policy", DEFAULT_PRESSURE_RATIO_POLICY)


def plan_payloads(
    pressure_ratio_policy: str = DEFAULT_PRESSURE_RATIO_POLICY,
) -> list[dict[str, Any]]:
    pressure_ratio_policy = validate_pressure_ratio_policy(pressure_ratio_policy)
    payloads = []
    for point in plan_points():
        controls = _stage_controls(point.stage_count)
        request = replace(request_for(point.architecture), gas_property_basis=TEXTBOOK)
        payloads.append({
            "schema_version": INPUT_SCHEMA_VERSION, "assembly": point.assembly,
            "stage_count": point.stage_count, "architecture": point.architecture.value,
            "turbine_role": point.turbine_role, "turbine_index": point.turbine_index,
            "architecture_cycle": input_to_payload(request),
            "stage_controls": {
                "speed_rpm": SPEED_RPM, "hub_diameter_m": HUB_DIAMETER_M,
                "alpha1_deg": controls["alpha1_deg"], "kappa": controls["kappa"],
                "pi_weights": controls["pi_weights"],
                "reference_exit_mach": REFERENCE_EXIT_MACH, "boundary_is_static": True,
                "pressure_ratio_policy": pressure_ratio_policy}})
    return payloads


def _model_digest() -> str:
    digest = hashlib.sha256()
    for relative in MODEL_SOURCE_FILES:
        digest.update(relative.encode("utf-8") + b"\0")
        with open(ROOT / relative, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _constant(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _stage_constants(namespace):
    keys = ("R", "φ", "ψ", "η_м", "μ", "δ_отн", "δ_мин", "k_n", "σ_л",
            "y0", "π_ст_макс", "q_n", "l_т_D_т")
    constants = {key: _constant(namespace[key]) for key in keys}
    constants.update({
        "reaction_table": _constant(namespace["ρ_таблица"]),
        "source": "source-built textbook namespace (oe.build_ns)",
        "stage_gas_property_basis": namespace["_gas_property_basis"],
        "cycle_gas_property_basis": namespace["_cycle_gas_property_basis"],
    })
    return constants


def make_run_config(payloads, constants, model_digest, pressure_ratio_policy):
    pressure_ratio_policy = validate_pressure_ratio_policy(pressure_ratio_policy)
    return {
        "experiment_version": EXPERIMENT_VERSION, "config_version": CONFIG_VERSION,
        "plan_name": PLAN_NAME, "plan_order": "A_MAIN, B_MAIN, C_GG, C_FPT; counts 1, 2, 3",
        "payload_plan_count": len(payloads),
        "payload_plan_sha256": canonical_hash(list(payloads)),
        "model_variant": MODEL_VARIANT, "evaluator_version": EVALUATOR_VERSION,
        "backend": BACKEND, "model_digest": model_digest,
        "model_digest_sources": list(MODEL_SOURCE_FILES),
        "pressure_ratio_policy": pressure_ratio_policy,
        "stage_constants": _constant(constants),
        "assumptions": copy_json_value(ASSUMPTIONS),
        "limitations": list(ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS),
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "stage_count_automatic_selection": "NONE", "coupling_label": COUPLING_LABEL,
    }


def _static_boundary(total_pressure, k, mach):
    return total_pressure / (1.0 + 0.5 * (k - 1.0) * mach**2) ** (k / (k - 1.0))


def _json_value(value, path):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise StageNumericalFailure(path, value)
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError(f"non-string key at {path}")
        return {key: _json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist(), path)
    if hasattr(value, "item"):
        return _json_value(value.item(), path)
    raise TypeError(f"unsupported value at {path}: {type(value).__name__}")


def _build_namespace(model, result, turbine, count, controls, p_static,
                     pressure_ratio_policy=DEFAULT_PRESSURE_RATIO_POLICY):
    namespace, _ = oe.build_ns(
        model, gas_property_basis=TEXTBOOK, cycle_gas_property_basis=TEXTBOOK)
    flow = float(result.gas_mass_flow)
    namespace.update({
        "T_г_полн": float(turbine.inlet.T_total),
        "p_г_полн": float(turbine.inlet.p_total),
        "G_в": flow, "G_c1": flow, "G_л_ост": flow, "G_base": (flow, flow, flow),
        "p_т": float(p_static), "z": count,
        "α1_ст": list(controls["alpha1_deg"]), "κ_ст": list(controls["kappa"]),
        "π_веса": list(controls["pi_weights"]), "R": float(result.gas_model.R),
        "α_г": float(result.combustor.alpha),
        "N_зад": float(result.target_gearbox_input_power),
        "L_к": float(result.compressor.specific_work)})
    # Reinstall the same extracted siblings after the fixture overrides so the
    # selected policy is live for this fresh namespace without re-evaluating the
    # textbook gas/stage adapters or hard-coded three-stage evaluation.
    orchestration = build_const_hub_orchestration(
        namespace, interp=namespace.get("_линейно"),
        pressure_ratio_policy=pressure_ratio_policy)
    orchestration.install()
    return namespace


def _stage_count_diagnostic(raw):
    info = {
        "status": "COMPUTED", "requested_stage_count": raw["stage_count"],
        "automatic_selection": "NONE", "mach_solved_by_meanline": False,
        "c0_basis": "PRELIMINARY_0D_END_STATE_TOTAL_TO_STATIC_ASSUMPTION",
    }
    try:
        ns, turbine, stages = raw["namespace"], raw["turbine"], raw["stages"]
        k_out, mach = float(turbine.outlet.k), float(raw["mach"])
        t_static = float(turbine.outlet.T_total) / (
            1.0 + 0.5 * (k_out - 1.0) * mach**2)
        t_ref = 0.5 * (float(turbine.inlet.T_total) + t_static)
        cp_ref, k_ref = ns["газ"](t_ref)
        p_in, p_out = float(turbine.inlet.p_total), float(raw["p_out_static"])
        h_drop = cp_ref * float(turbine.inlet.T_total) * (
            1.0 - (p_out / p_in) ** ((k_ref - 1.0) / k_ref))
        c0 = math.sqrt(2.0 * h_drop)
        u_mean = 0.5 * (stages[0]["u1"] + stages[-1]["u2"])
        mean_c2 = sum(stage["c2"] for stage in stages) / len(stages)
        recovered, y0 = float(ns["μ"]) * mean_c2, float(ns["y0"])
        info.update({
            "c0_total_m_s": c0, "reference_mean_temperature_K": t_ref,
            "cp_reference_J_kgK": cp_ref, "k_reference": k_ref,
            "total_to_static_pressure_ratio": p_in / p_out,
            "total_to_static_drop_J_kg": h_drop, "u_mean_m_s": u_mean,
            "mean_c2_m_s": mean_c2, "recovered_exit_velocity_m_s": recovered,
            "y0": y0, "z_estimate": estimate_stage_count(c0, u_mean, y0, recovered),
            "estimate_rounded": False, "mach_assumption": mach,
            "end_state_assumption": "aggregate 0D outlet total state with fixture Mach; preliminary only"})
    except (ValueError, TypeError, ZeroDivisionError, OverflowError,
            NumericalConvergenceError) as exc:
        info.update({"status": "NOT_CHECKED", "reason": str(exc) or type(exc).__name__,
                     "error_type": type(exc).__name__})
    return info


def _stage_metric(index, stage, basis):
    c0 = math.sqrt(2.0 * stage["H0"])
    local_total = stage["p2"] * (stage["T_ст_п"] / stage["T_ст"]) ** (
        stage["k"] / (stage["k"] - 1.0))
    angles = {key: stage[key] for key in ("α1", "β1", "α2", "β2") if key in stage}
    mach = {key: stage[key] for key in ("M_c1", "M_w2") if key in stage}
    diameters = {key: stage[key] for key in ("D_c", "D_л") if key in stage}
    heights = {key: stage[key] for key in ("l_c", "l_л") if key in stage}
    return {
        "index": index, "pi_stage": stage["π_ст"], "rho": stage["ρ"],
        "specific_work_J_kg": stage["L_т"], "power_W": stage["G_c"] * stage["L_т"],
        "c0_stage_m_s": c0, "u_over_c0": 0.5 * (stage["u1"] + stage["u2"]) / c0,
        "angles_deg": angles or {"status": "NOT_CHECKED"},
        "mach": mach or {"status": "NOT_CHECKED"},
        "diameters_m": diameters or {"status": "NOT_CHECKED"},
        "heights_m": heights or {"status": "NOT_CHECKED"},
        "rotor_stress_MPa": stage.get("σ_р", {"status": "NOT_CHECKED"}),
        "cp_J_kgK": stage["cp"], "k": stage["k"], "property_basis": basis,
        "gas_mass_flow_kg_s": stage["G_c"],
        "local_total_pressure": {
            "stage_p_st_p": stage["p_ст_п"],
            "reconstructed_from_p2_Tstp_Tst": local_total,
            "reconstruction_error_Pa": local_total - stage["p_ст_п"]},
    }


def _power_comparison(raw):
    result, turbine, stages = raw["result"], raw["turbine"], raw["stages"]
    mech = float(result.shaft.mechanical_efficiencies[raw["turbine_index"]])
    stage_gas = sum(stage["G_c"] * stage["L_т"] for stage in stages)
    stage_mech, agg_mech = mech * stage_gas, mech * float(turbine.power)
    comparison = {
        "mechanical_efficiency": mech, "stage_gas_power_W": stage_gas,
        "aggregate_gas_power_W": turbine.power, "stage_mechanical_power_W": stage_mech,
        "aggregate_mechanical_power_W": agg_mech,
        "difference_stage_minus_aggregate_mechanical_W": stage_mech - agg_mech,
        "uncoupled_label": COUPLING_LABEL,
        "target_gearbox_input_power_W": result.target_gearbox_input_power}
    if raw["turbine_role"] == "main":
        net = stage_mech - float(result.air_mass_flow) * float(
            result.compressor.specific_work)
        comparison["main_net_stage_minus_source_Lc_W"] = net
        comparison["main_target_residual_W"] = net - float(result.target_gearbox_input_power)
    elif raw["turbine_role"] == "gas_generator":
        comparison["gg_stage_minus_compressor_power_W"] = stage_mech - float(result.compressor.power)
    else:
        comparison["fpt_output_stage_W"] = stage_mech
        comparison["fpt_target_residual_W"] = stage_mech - float(result.target_gearbox_input_power)
    return comparison


def _posthoc_pi_acceptance(stages):
    lower, upper = PI_ACCEPTANCE_BOUNDS
    vector = []
    for index, stage in enumerate(stages, 1):
        actual = stage["π_ст"]
        within = bool(lower <= actual <= upper)
        vector.append({
            "stage": index,
            "actual": actual, "min": lower, "max": upper,
            "actual_pi_stage": actual,
            "min_pi_stage": lower, "max_pi_stage": upper,
            "lower_margin": actual - lower, "upper_margin": upper - actual,
            "status": "PASS" if within else "FAIL"})
    return {
        "profile": PI_ACCEPTANCE_PROFILE,
        "basis": "USER_PROMPT_ENGINEERING_PROFILE_NOT_BOOK_FUNDAMENTAL",
        "bounds": [lower, upper], "stages": vector,
        "pass": all(item["status"] == "PASS" for item in vector),
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "overall_engineering_feasibility_reason": "strength_verification_missing",
        "note": "post-hoc engineering profile only; overall feasibility NOT_CHECKED"}


def _build_outputs(raw):
    stages, turbine = raw["stages"], raw["turbine"]
    converted = []
    for index, stage in enumerate(stages):
        try:
            converted.append({key: _json_value(value, f"stage.{key}") for key, value in stage.items()})
        except StageNumericalFailure as exc:
            exc.partial = {"stages": converted, "stage_index": index}
            raise
    basis, comparison = raw["stage_basis"], _power_comparison(raw)
    static_error = stages[-1]["p2"] - raw["p_out_static"]
    total_mismatch = stages[-1]["p_ст_п"] - turbine.outlet.p_total
    result = raw["result"]
    outputs = {
        "schema_version": OUTPUT_SCHEMA_VERSION, "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "coupling_label": COUPLING_LABEL, "architecture": raw["architecture"],
        "assembly": raw["assembly"], "turbine_role": raw["turbine_role"],
        "pressure_ratio_policy": raw["pressure_ratio_policy"],
        "posthoc_pi_acceptance": _posthoc_pi_acceptance(stages),
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "overall_engineering_feasibility_reason": "strength_verification_missing",
        "stage_count_requested": raw["stage_count"], "stage_count_selection": "NONE",
        "stage_count_estimate": raw["stage_count_estimate"],
        "source_cycle": {
            "architecture": result.architecture.value,
            "gas_property_basis": result.gas_property_basis,
            "R_J_kgK": result.gas_model.R,
            "inlet_total_temperature_K": turbine.inlet.T_total,
            "inlet_total_pressure_Pa": turbine.inlet.p_total,
            "outlet_total_temperature_K": turbine.outlet.T_total,
            "outlet_total_pressure_Pa": turbine.outlet.p_total, "outlet_k": turbine.outlet.k,
            "air_mass_flow_kg_s": result.air_mass_flow,
            "gas_mass_flow_kg_s": result.gas_mass_flow,
            "fuel_mass_flow_kg_s": result.fuel_mass_flow, "combustor_alpha": result.combustor.alpha,
            "compressor_power_W": result.compressor.power,
            "compressor_specific_work_J_kg": result.compressor.specific_work,
            "turbine_power_W": turbine.power,
            "turbine_specific_work_J_kg": turbine.specific_work,
            "target_gearbox_input_power_W": result.target_gearbox_input_power},
        "boundary": {
            "reference_exit_mach": raw["mach"],
            "mach_source": "CycleConfig.M_t fixture assumption",
            "mach_solved_by_meanline": False,
            "aggregate_outlet_total_pressure_Pa": turbine.outlet.p_total,
            "derived_static_outlet_pressure_Pa": raw["p_out_static"],
            "formula": "p_total/(1+0.5*(k-1)*M**2)**(k/(k-1))",
            "last_stage_static_outlet_error_Pa": static_error,
            "last_stage_total_pressure_Pa": stages[-1]["p_ст_п"],
            "last_stage_total_minus_aggregate_total_Pa": total_mismatch,
            "last_stage_total_relative_mismatch": total_mismatch / turbine.outlet.p_total,
            "boundary_is_static": True},
        "property_basis": {
            "stage_namespace_marker": raw["namespace"]["_gas_property_basis"],
            "stage_adapter": basis, "cycle": result.gas_property_basis},
        "stage_solution": {
            "stage_count": len(converted), "seed_pressure_ratio": raw["seed"],
            "solved_pressure_ratio": raw["solved_ratio"], "stages": converted},
        "stage_metrics": [_stage_metric(i, stage, basis) for i, stage in enumerate(stages, 1)],
        "power_comparison": comparison}
    try:
        return _json_value(outputs, "outputs")
    except StageNumericalFailure as exc:
        exc.partial = {"stages": converted, "path": exc.path}
        raise


def _run_stage_meanline(payload, model=None):
    if model is None:
        model = _get_model()
    count, assembly = payload["stage_count"], payload["assembly"]
    architecture, role, turbine_index = ASSEMBLY_SPECS[assembly]
    if payload["architecture"] != architecture.value:
        raise ArchitectureStageSmokeError("assembly architecture mismatch")
    result = calculate_architecture_cycle(payload_to_input(payload["architecture_cycle"]))
    turbine = result.turbines[turbine_index]
    controls = payload["stage_controls"]
    pressure_ratio_policy = _payload_policy(payload)
    mach = float(controls["reference_exit_mach"])
    p_static = _static_boundary(float(turbine.outlet.p_total), float(turbine.outlet.k), mach)
    namespace = _build_namespace(
        model, result, turbine, count, controls, p_static, pressure_ratio_policy)
    seed = (float(turbine.inlet.p_total) / p_static) ** (1.0 / count)
    try:
        solved_ratio, stages = namespace["распределить_π_const_hub"](
            float(controls["speed_rpm"]), float(controls["hub_diameter_m"]), seed)
    except namespace["НетРешения"] as exc:
        raise StageNoSolution(str(exc) or "no stage geometry solution",
                              {"assembly": assembly, "seed": seed}) from exc
    if not stages:
        raise StageNoSolution("orchestration returned no stages", {"seed": seed})
    raw = {
        "assembly": assembly, "stage_count": count,
        "architecture": architecture.value, "turbine_role": role,
        "turbine_index": turbine_index, "result": result, "turbine": turbine,
        "namespace": namespace,
        "stage_basis": getattr(namespace["ступень"], oc.GAS_PROPERTY_BASIS_ATTR, None),
        "p_out_static": p_static, "mach": mach, "seed": seed,
        "solved_ratio": solved_ratio, "stages": list(stages),
        "pressure_ratio_policy": pressure_ratio_policy,
    }
    raw["stage_count_estimate"] = _stage_count_diagnostic(raw)
    return raw


def evaluate_stage_payload(payload, *, model=None):
    """Evaluate one fixed plan point; preserve typed failures and partials."""
    input_payload = copy_json_value(payload, path="$.payload")
    policy = _payload_policy(payload)
    try:
        outputs = _build_outputs(_run_stage_meanline(payload, model=model))
    except StageNoSolution as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NON_PHYSICAL_STATE, "stage_geometry_domain", exc.reason,
            last_valid_state={"input": input_payload, **exc.last_valid_state},
            diagnostics={"error_type": type(exc).__name__,
                         "pressure_ratio_policy": policy})
    except StageNumericalFailure as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NUMERICAL_FAILURE, "stage_numerical", "non_finite_stage_output",
            last_valid_state={"input": input_payload, "path": exc.path},
            diagnostics={"error_type": type(exc).__name__, "path": exc.path,
                         "pressure_ratio_policy": policy},
            outputs=exc.partial or None)
    except SolverDiagnosticsError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus(exc.status), exc.stage or "stage_solver",
            exc.reason or type(exc).__name__, last_valid_state={
                "input": input_payload, "solver_last_valid_state": exc.last_valid_state},
            diagnostics={"error_type": type(exc).__name__, "solver": exc.as_record(),
                         "pressure_ratio_policy": policy})
    except ArchitectureCyclePhysicalError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NON_PHYSICAL_STATE, "architecture_cycle_physical",
            str(exc) or type(exc).__name__, last_valid_state={"input": input_payload},
            diagnostics={"error_type": type(exc).__name__,
                         "pressure_ratio_policy": policy})
    except CycleSolverError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.NUMERICAL_FAILURE, "cycle_solver",
            str(exc) or type(exc).__name__, last_valid_state={"input": input_payload},
            diagnostics={"error_type": type(exc).__name__,
                         "pressure_ratio_policy": policy})
    except ArchitectureCycleError as exc:
        return EvaluationOutcome.failure(
            CalculationStatus.MODEL_ERROR, "architecture_cycle_config",
            str(exc) or type(exc).__name__, last_valid_state={"input": input_payload},
            diagnostics={"error_type": type(exc).__name__,
                         "pressure_ratio_policy": policy})
    return EvaluationOutcome.success(outputs=outputs, diagnostics={
        "schema_version": OUTPUT_SCHEMA_VERSION, "architecture": outputs["architecture"],
        "assembly": outputs["assembly"], "stage_count_requested": outputs["stage_count_requested"],
        "coupling_label": COUPLING_LABEL, "pressure_ratio_policy": policy,
        "validation_status": "NOT_PHYSICALLY_VALIDATED"})


def _validate_plan(payloads):
    if len(payloads) != MAX_PLAN_POINTS:
        raise ArchitectureStageSmokeError("plan must contain 12 points")
    policies = {_payload_policy(p) for p in payloads}
    if len(policies) != 1:
        raise ArchitectureStageSmokeError("plan mixes pressure-ratio policies")
    ids = [make_candidate_id(p["architecture"], MODEL_VARIANT, p) for p in payloads]
    if len(set(ids)) != len(ids):
        raise ArchitectureStageSmokeError("duplicate plan candidate inputs")


def _open_writer(destination, *, resume, config):
    if (destination / SUMMARY_NAME).exists() and not resume:
        raise ArchitectureStageSmokeError("summary exists; use --resume")
    if destination.exists():
        if not destination.is_dir():
            raise ArchitectureStageSmokeError("destination is not a directory")
        if any(destination.iterdir()):
            if not resume:
                raise ArchitectureStageSmokeError("destination is not empty")
            return DatasetWriter.open(
                destination, expected_run_config=config,
                expected_evaluator_version=EVALUATOR_VERSION)
        if resume:
            raise ArchitectureStageSmokeError("cannot resume an empty destination")
    elif resume:
        raise ArchitectureStageSmokeError("cannot resume a missing dataset")
    return DatasetWriter.create(destination, run_config=config)


def _ordered_records(points, payloads, records):
    if len(records) != len(points):
        raise ArchitectureStageSmokeError("journal length differs from plan")
    expected = [make_candidate_id(p.architecture.value, MODEL_VARIANT, payload)
                for p, payload in zip(points, payloads)]
    by_id = {record.candidate_id: record for record in records}
    if set(by_id) != set(expected):
        raise ArchitectureStageSmokeError("journal records differ from plan")
    return [by_id[identifier] for identifier in expected]


def _reproduce_all(records, evaluator):
    differences = []
    for expected in records:
        actual = evaluate_candidate(
            evaluator, expected.payload, architecture=expected.architecture,
            model_variant=expected.model_variant, run_id=expected.run_id,
            sampling_method=expected.sampling_method,
            sampling_metadata=expected.sampling)
        diff = compare_candidates(expected, actual, rtol=1e-12, atol=1e-9)
        if diff:
            differences.append({"candidate_id": expected.candidate_id,
                                "differences": list(diff)})
    return {"checked": len(records), "matches": not differences,
            "differences": differences, "rtol": 1e-12, "atol": 1e-9}


def _point_entry(point, record):
    entry = {
        "index": point.index, "candidate_id": record.candidate_id,
        "assembly": point.assembly, "stage_count_requested": point.stage_count,
        "architecture": point.architecture.value, "turbine_role": point.turbine_role,
        "calculation_status": record.calculation_status.value,
        "failure_stage": record.failure_stage, "failure_reason": record.failure_reason,
        "pressure_ratio_policy": _payload_policy(record.payload),
        "coupling_label": COUPLING_LABEL, "stage_count_selection": "NONE"}
    if record.calculation_status is not CalculationStatus.SUCCESS:
        entry["metrics"] = {
            "status": "NOT_CHECKED", "reason": "attempt_not_success",
            "posthoc_pi_acceptance": {
                "profile": PI_ACCEPTANCE_PROFILE, "status": "NOT_CHECKED",
                "pass": None},
            "overall_engineering_feasibility": None,
            "overall_engineering_feasibility_status": "NOT_CHECKED",
            "overall_engineering_feasibility_reason": "strength_verification_missing"}
        return entry
    boundary, comparison = record.outputs["boundary"], record.outputs["power_comparison"]
    acceptance = record.outputs.get("posthoc_pi_acceptance")
    entry["metrics"] = {
        "posthoc_pi_acceptance": acceptance if acceptance else {
            "profile": PI_ACCEPTANCE_PROFILE, "status": "NOT_CHECKED",
            "pass": None},
        "stage_count_estimate": record.outputs["stage_count_estimate"],
        "last_stage_static_outlet_error_Pa": boundary["last_stage_static_outlet_error_Pa"],
        "last_stage_total_minus_aggregate_total_Pa": boundary["last_stage_total_minus_aggregate_total_Pa"],
        "stage_minus_aggregate_mechanical_power_W": comparison["difference_stage_minus_aggregate_mechanical_W"],
        "main_target_residual_W": comparison.get("main_target_residual_W", {"status": "NOT_CHECKED"}),
        "gg_stage_minus_compressor_power_W": comparison.get("gg_stage_minus_compressor_power_W", {"status": "NOT_CHECKED"}),
        "fpt_target_residual_W": comparison.get("fpt_target_residual_W", {"status": "NOT_CHECKED"})}
    return entry


def _build_summary(destination, points, payloads, records, manifest, appended,
                   elapsed, reproduce, model_digest, pressure_ratio_policy):
    statuses = Counter(record.calculation_status.value for record in records)
    by_assembly = {}
    for assembly in ASSEMBLY_ORDER:
        by_count = {}
        for count in STAGE_COUNTS:
            rows = [r for p, r in zip(points, records)
                    if p.assembly == assembly and p.stage_count == count]
            found = Counter(r.calculation_status.value for r in rows)
            by_count[str(count)] = {
                **{s.value: found.get(s.value, 0) for s in CalculationStatus},
                "total": len(rows)}
        by_assembly[assembly] = by_count
    return {
        "report_version": SUMMARY_VERSION, "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED", "illustrative_only": True,
        "coupling_label": COUPLING_LABEL,
        "experiment": {
            "plan_name": PLAN_NAME, "payload_plan_count": len(payloads),
            "payload_plan_sha256": canonical_hash(list(payloads)),
            "model_digest": model_digest,
            "pressure_ratio_policy": pressure_ratio_policy},
        "run": {
            "run_id": manifest["run_id"], "run_config_hash": manifest["run_config_hash"],
            "model_variant": manifest["run_config"]["model_variant"],
            "record_count": manifest["record_count"], "complete": manifest["complete"],
            "journal_sha256": manifest["journal_sha256"],
            "points_attempted": len(points),
            "records_appended_this_invocation": appended,
            "elapsed_seconds": round(elapsed, 6)},
        "status_counts": dict(sorted(statuses.items())),
        "status_by_assembly_and_count": by_assembly,
        "points": [_point_entry(point, record) for point, record in zip(points, records)],
        "reproduce": copy_json_value(reproduce)}


def _write_summary_atomic(path, summary):
    try:
        strict = copy_json_value(summary, path="$.summary")
    except PayloadValidationError as exc:
        raise ArchitectureStageSmokeError(f"summary is not strict JSON: {exc}") from exc
    text = json.dumps(strict, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def run_plan(destination, *, resume=False, evaluator=None, model=None,
             constants=None, model_digest=None, reproduce=True,
             pressure_ratio_policy=DEFAULT_PRESSURE_RATIO_POLICY):
    """Evaluate the fixed 12-point plan, store all attempts, and summarise."""
    pressure_ratio_policy = validate_pressure_ratio_policy(pressure_ratio_policy)
    if struct.calcsize("d") != 8 or sys.float_info.mant_dig != 53:
        raise ArchitectureStageSmokeError("float64 reference backend required")
    if constants is None:
        if model is None:
            model = _get_model()
        template, _ = oe.build_ns(
            model, gas_property_basis=TEXTBOOK, cycle_gas_property_basis=TEXTBOOK)
        constants = _stage_constants(template)
    if model_digest is None:
        model_digest = _model_digest()
    if evaluator is None:
        if model is None:
            model = _get_model()
        evaluate = lambda item: evaluate_stage_payload(item, model=model)
    else:
        evaluate = evaluator
    points = plan_points()
    payloads = plan_payloads(pressure_ratio_policy)
    _validate_plan(payloads)
    config = make_run_config(
        payloads, constants, model_digest, pressure_ratio_policy)
    path = Path(destination)
    writer = _open_writer(path, resume=resume, config=config)
    existing = {r.candidate_id for r in read_records(path)} if resume else set()
    appended, started, completed = 0, time.perf_counter(), False
    try:
        for index, (point, payload) in enumerate(zip(points, payloads)):
            candidate_id = make_candidate_id(point.architecture.value, MODEL_VARIANT, payload)
            if candidate_id in existing:
                continue
            before = writer.record_count
            writer.evaluate_and_append(
                evaluate, payload, architecture=point.architecture.value,
                sampling_method="stage_meanline_fixture",
                sampling_metadata={"plan": PLAN_NAME, "index": index, "assembly": point.assembly,
                                   "stage_count": point.stage_count,
                                   "architecture": point.architecture.value,
                                   "turbine_role": point.turbine_role})
            appended += writer.record_count > before
        completed = True
    finally:
        writer.close(finalize=completed)
    elapsed = time.perf_counter() - started
    records = _ordered_records(points, payloads, read_records(path))
    manifest = load_manifest(path)
    reproduce_result = (_reproduce_all(records, evaluate) if reproduce else {
        "checked": 0, "matches": None, "differences": [], "rtol": 1e-12, "atol": 1e-9})
    summary = _build_summary(path, points, payloads, records, manifest, appended,
                             elapsed, reproduce_result, model_digest,
                             pressure_ratio_policy)
    _write_summary_atomic(path / SUMMARY_NAME, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=("Bounded 12-attempt stage-count / const-hub meanline smoke. "
                     "Not an architecture-stage coupling model, not physical validation."))
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--pressure-ratio-policy", choices=PRESSURE_RATIO_POLICY_CHOICES,
        default=DEFAULT_PRESSURE_RATIO_POLICY,
        help="raw_unbounded (default) or legacy_clipped reference comparison")
    args = parser.parse_args(argv)
    try:
        summary = run_plan(
            args.destination, resume=args.resume,
            pressure_ratio_policy=args.pressure_ratio_policy)
    except (ArchitectureStageSmokeError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False))
    return 3 if summary["reproduce"]["matches"] is False else 0

if __name__ == "__main__":
    raise SystemExit(main())
