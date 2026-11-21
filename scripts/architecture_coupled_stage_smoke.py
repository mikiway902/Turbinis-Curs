#!/usr/bin/env python3
"""Bounded actual-boundary A/B/C stage-coupling RAW smoke (fixed 15 attempts).

A/B counts 1/2/3 close actual last-stage total outlet pressure to the unchanged
0D boundary in the explicit (120000, 150000) Pa bracket; net work subtracts the
source compressor power.  Nine C GG x FPT pairs are attempted independently:
close actual GG mechanical power to source compressor power, rebuild a fresh FPT
namespace, overwrite only its inlet total T/p with the actual GG endpoint, and
close the actual FPT outlet total pressure to the original 0D boundary.

GG bracket = 0.8/1.2 * preliminary 0D total-to-static conversion (source k,
M_t=0.3379, bracket provenance only).  Brackets are never searched or widened.
A bracket without the root is NUMERICAL_FAILURE, not physical infeasibility.
``НетРешения`` is NON_PHYSICAL_STATE with location; unexpected exceptions stay
the RAW MODEL_ERROR boundary; a successful GG closure survives an FPT failure.
Calculations are SUCCESS regardless of power deficit or pi profile (overall
feasibility NOT_CHECKED).  No flow, speed, hub, stage count, or bracket is tuned.

Optional ``stage_controls.reaction_table_policy`` (absent = legacy clipped):

* ``legacy_clipped`` keeps the source ``np.interp`` clip and unchanged outputs.
* ``require_final_source_domain`` rejects a final actual closure whose stage
  ``l_l/D_l`` lies outside the actual source ``ρ_таблица`` input bounds.  The
  inner fixed point may still use the existing clip for numerical
  initialization; only the accepted final closure is checked.  The rejection is
  a typed source-domain MODEL_ERROR, not engineering infeasibility.
"""
from __future__ import annotations

import argparse, hashlib, inspect, json, math, os, struct, sys, time
from collections import Counter, namedtuple
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT, ROOT / "scripts", ROOT / "Optimizer"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import optimize_extended as oe  # noqa: E402
import optimizer_core as oc  # noqa: E402
import architecture_stage_smoke as stage_smoke  # noqa: E402
from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK as TEXTBOOK,
    calculate_architecture_cycle)
from turbine_engine.architecture_raw import (  # noqa: E402
    input_to_payload, payload_to_input)
from turbine_engine.dataset import (  # noqa: E402
    DatasetError, DatasetWriter, load_manifest, read_records)
from turbine_engine.diagnostics import SolverDiagnosticsError  # noqa: E402
from turbine_engine.orchestration import (  # noqa: E402
    PRESSURE_RATIO_RAW_UNBOUNDED, ConstHubOrchestration)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus, EvaluationOutcome, PayloadValidationError, canonical_hash,
    copy_json_value, make_candidate_id)
from turbine_engine.stage import StageResult  # noqa: E402

EXPERIMENT_VERSION = "architecture-coupled-stage-v2"
CONFIG_VERSION = "architecture-coupled-stage-config-v2"
SUMMARY_VERSION = "architecture-coupled-stage-summary-v2"
EVALUATOR_VERSION = "architecture-coupled-stage-smoke-v2"
MODEL_VARIANT = "architecture_coupled_stage_smoke_v2"
PLAN_NAME, SUMMARY_NAME = "ab-actual-pressure-c-gg-x-fpt", "summary.json"
BACKEND = "reference_python_float64"
INPUT_SCHEMA_VERSION = "architecture-coupled-stage-input-v1"
OUTPUT_SCHEMA_VERSION = "architecture-coupled-stage-output-v1"
COUPLING_LABEL = "ACTUAL_BOUNDARY_STAGE_COUPLED"
POWER_CLOSED_LABEL = "ACTUAL_BOUNDARY_LINK_POWER_CLOSED"
POWER_NOT_CLOSED_LABEL = "ACTUAL_BOUNDARY_LINK_NOT_POWER_CLOSED"
STAGE_COUNTS = (1, 2, 3)
AB_ASSEMBLIES, C_ASSEMBLY = ("A_MAIN", "B_MAIN"), "C_COUPLED"
SPEED_RPM = stage_smoke.SPEED_RPM
HUB_DIAMETER_M = stage_smoke.HUB_DIAMETER_M
REFERENCE_EXIT_MACH = stage_smoke.REFERENCE_EXIT_MACH
TARGET = stage_smoke.TARGET
SPEED_BASIS = "fixed regression anchor copied from the stage fixture; no selection"
AB_BRACKET, C_FPT_BRACKET = (120_000.0, 150_000.0), (120_000.0, 150_000.0)
GG_BRACKET_FACTORS = (0.8, 1.2)
POWER_CLOSURE_RELATIVE_TOL = 1e-6
REACTION_TABLE_POLICY_LEGACY_CLIPPED = "legacy_clipped"
REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN = "require_final_source_domain"
REACTION_TABLE_POLICIES = (
    REACTION_TABLE_POLICY_LEGACY_CLIPPED,
    REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN,
)
REACTION_TABLE_DOMAIN_TOL = 1e-12
NEW_SMOKE_FILE = "scripts/architecture_coupled_stage_smoke.py"
MODEL_SOURCE_FILES = tuple(stage_smoke.MODEL_SOURCE_FILES) + (NEW_SMOKE_FILE,)

_MODEL: Any | None = None


class ArchitectureCoupledSmokeError(RuntimeError):
    pass


class SourceTableDomainError(Exception):
    """Final actual closure outside the source reaction-table input domain.

    This is a source-support error, not a physical-impossibility or engineering
    acceptance result.  The attached closure is the complete actual final
    closure; ``partial`` is populated by the evaluator with every stage already
    calculated.
    """

    def __init__(self, role, stage, table_bounds, actuals, margins, closure,
                 flow_metadata=None):
        self.role = role
        self.stage = stage
        self.reason = "reaction_table_source_domain"
        self.table_bounds = list(table_bounds)
        self.actuals = list(actuals)
        self.signed_margins = list(margins)
        self.closure = closure
        self.flow_metadata = flow_metadata
        self.partial: dict[str, Any] = {}
        super().__init__(
            f"{role} final stage {stage} l_l/D_l outside source reaction table "
            f"[{self.table_bounds[0]!r}, {self.table_bounds[1]!r}]")

    def as_record(self):
        record = {
            "role": self.role, "stage": self.stage, "reason": self.reason,
            "table_bounds": list(self.table_bounds),
            "actual_l_over_D": list(self.actuals),
            "signed_margins": list(self.signed_margins),
            "actual_closure": self.closure}
        if self.flow_metadata is not None:
            record["flow_metadata"] = self.flow_metadata
        return record


class CoupledNoSolution(Exception):
    """Namespace-reported invalid evaluated state, with its stage location."""

    def __init__(self, reason, location, last_valid_state=None):
        self.reason, self.location = reason, location
        self.last_valid_state = dict(last_valid_state or {})
        self.partial: dict[str, Any] = {}
        super().__init__(reason)


PlanPoint = namedtuple("PlanPoint", (
    "index coupling assembly main_stage_count gas_generator_stage_count "
    "free_power_stage_count architecture"))


def _get_model():
    global _MODEL
    _MODEL = _MODEL or oc.load_model()
    return _MODEL


def plan_points() -> tuple[PlanPoint, ...]:
    points = [PlanPoint(i, "AB", assembly, count, None, None,
                        stage_smoke.ASSEMBLY_SPECS[assembly][0])
              for i, (assembly, count) in enumerate(
                  (a, c) for a in AB_ASSEMBLIES for c in STAGE_COUNTS)]
    start = len(points)
    points += [PlanPoint(start + i, "C", C_ASSEMBLY, None, gg, fpt,
                         Architecture.FREE_POWER_TURBINE)
               for i, (gg, fpt) in enumerate(
                   (gg, fpt) for gg in STAGE_COUNTS for fpt in STAGE_COUNTS)]
    return tuple(points)


def _source_index():
    return {(point.assembly, point.stage_count): payload for point, payload
            in zip(stage_smoke.plan_points(), stage_smoke.plan_payloads())}


DESIGN_CONTROL_KEYS = ("alpha1_deg", "kappa", "pi_weights")


def _design_overrides(payload):
    return payload["stage_controls"].get("design_overrides")


def _controls(payload, role):
    """Return effective stage arrays: explicit design override, else anchors."""
    overrides = _design_overrides(payload)
    block = (overrides[role] if overrides is not None
             else payload["stage_controls"][role])
    return {key: list(block[key]) for key in DESIGN_CONTROL_KEYS}


def _design_output(payload):
    """Record source anchor arrays and effective arrays; absent by default."""
    overrides = _design_overrides(payload)
    if overrides is None:
        return None
    return {
        "override_source": "EXPLICIT_STAGE_CONTROLS_DESIGN_OVERRIDES",
        "roles": {role: {
            "source": stage_smoke._stage_controls(
                payload["stage_counts"][role]),
            "effective": {key: list(values)
                          for key, values in block.items()}}
            for role, block in overrides.items()}}


def _operating(payload, role):
    """Return the explicit effective ``(speed_rpm, hub_diameter_m, overridden)``."""
    points = payload["stage_controls"].get("operating_points")
    if points is None:
        return SPEED_RPM, HUB_DIAMETER_M, False
    point = points[role]
    return float(point["speed_rpm"]), float(point["hub_diameter_m"]), True


def _operating_output(effective):
    """Record explicit source-vs-effective controls; absent by default."""
    if not any(item[2] for item in effective.values()):
        return None
    return {role: {
        "source_speed_rpm": SPEED_RPM,
        "source_hub_diameter_m": HUB_DIAMETER_M,
        "effective_speed_rpm": n,
        "effective_hub_diameter_m": hub,
        "override_source": "EXPLICIT_STAGE_CONTROLS_OPERATING_POINTS",
        "compressor_speed_compatibility": "NOT_CHECKED",
        "implicit_powertrain_map": "NOT_APPLIED"}
        for role, (n, hub, _) in effective.items()}


def _reaction_table_policy(payload):
    return payload["stage_controls"].get(
        "reaction_table_policy", REACTION_TABLE_POLICY_LEGACY_CLIPPED)


def _reaction_table_bounds(namespace):
    """Return the actual source reaction-table l/D bounds, never hard-coded."""
    try:
        table = namespace["ρ_таблица"]
        bounds = table[0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ArchitectureCoupledSmokeError(
            "actual namespace reaction table is missing") from exc
    try:
        values = [float(value) for value in bounds]
    except (TypeError, ValueError) as exc:
        raise ArchitectureCoupledSmokeError(
            "actual namespace reaction table bounds are not numeric") from exc
    _need(bool(values) and all(math.isfinite(value) for value in values)
          and all(left < right for left, right in zip(values, values[1:])),
          "actual namespace reaction table bounds must be finite and strictly "
          "increasing")
    return values[0], values[-1]


def _reaction_table_actuals(closure):
    """Return final stage l_l/D_l values through the typed StageResult."""
    actuals = []
    for index, item in enumerate(closure["actual_stages"], 1):
        stage = StageResult.from_legacy_dict(item)
        _need(math.isfinite(stage.D_l) and stage.D_l > 0.0,
              f"actual stage {index} has non-positive rotor diameter")
        actuals.append(float(stage.l_l) / float(stage.D_l))
    _need(bool(actuals), "actual closure has no stage results")
    return actuals


def _reaction_table_domain_record(role, closure, namespace, *, policy,
                                  flow_metadata=None):
    """Check only the accepted final closure against the actual table bounds."""
    lo, hi = _reaction_table_bounds(namespace)
    actuals = _reaction_table_actuals(closure)
    margins = [min(actual - lo, hi - actual) for actual in actuals]
    outside = [index for index, margin in enumerate(margins, 1)
               if margin < -REACTION_TABLE_DOMAIN_TOL]
    record = {
        "role": role, "policy": policy,
        "status": "FINAL_SOURCE_DOMAIN_VERIFIED" if not outside
        else "OUTSIDE_SOURCE_DOMAIN",
        "final_domain_verified": not outside,
        "table_bounds": [lo, hi],
        "source_domain_tolerance": REACTION_TABLE_DOMAIN_TOL,
        "actual_l_over_D": actuals, "signed_margins": margins,
        "stages": [{
            "stage": index, "actual_l_over_D": actual,
            "lower_margin": actual - lo, "upper_margin": hi - actual,
            "signed_margin": min(actual - lo, hi - actual),
            "status": "WITHIN_SOURCE_DOMAIN" if index not in outside
            else "OUTSIDE_SOURCE_DOMAIN"}
            for index, actual in enumerate(actuals, 1)],
        "intermediate_clipping":
            "NUMERICAL_INITIALIZATION_ONLY_NOT_VALIDATED_STATES",
        "note": "only the accepted final actual closure is checked; the inner "
                "reaction fixed point retains the existing source clip for "
                "numerical initialization and those intermediate states are "
                "not validated outputs"}
    if flow_metadata is not None:
        record["flow_metadata"] = flow_metadata
    return record, outside


def _enforce_reaction_table_domain(role, closure, namespace, *, policy,
                                   flow_metadata=None):
    record, outside = _reaction_table_domain_record(
        role, closure, namespace, policy=policy, flow_metadata=flow_metadata)
    if outside:
        raise SourceTableDomainError(
            role=role, stage=outside[0], table_bounds=record["table_bounds"],
            actuals=record["actual_l_over_D"],
            margins=record["signed_margins"], closure=closure,
            flow_metadata=flow_metadata)
    return record


def _preliminary_static(total_pressure, k):
    return stage_smoke._static_boundary(
        float(total_pressure), float(k), REFERENCE_EXIT_MACH)


def _common_controls(**extra):
    return {"speed_rpm": SPEED_RPM, "hub_diameter_m": HUB_DIAMETER_M,
            "reference_exit_mach": REFERENCE_EXIT_MACH,
            "boundary_is_static": True, "speed_basis": SPEED_BASIS,
            "pressure_ratio_policy": PRESSURE_RATIO_RAW_UNBOUNDED, **extra}


def plan_payloads():
    """Return the fixed 15-point plan with explicit targets and brackets."""
    source, ab_results, c_result, payloads = _source_index(), {}, None, []
    for point in plan_points():
        if point.coupling == "AB":
            if point.assembly not in ab_results:
                ab_results[point.assembly] = calculate_architecture_cycle(
                    payload_to_input(source[(point.assembly, 1)]["architecture_cycle"]))
            result = ab_results[point.assembly]
            turbine = result.turbines[0]
            payloads.append({
                "schema_version": INPUT_SCHEMA_VERSION, "coupling": "AB",
                "point_index": point.index, "assembly": point.assembly,
                "architecture": point.architecture.value, "turbine_role": "main",
                "stage_counts": {"main": point.main_stage_count},
                "architecture_cycle_source": copy_json_value(
                    source[(point.assembly, point.main_stage_count)]["architecture_cycle"]),
                "stage_controls": _common_controls(
                    main=stage_smoke._stage_controls(point.main_stage_count)),
                "targets": {"main_total_outlet_pressure_Pa": float(turbine.outlet.p_total),
                            "main_compressor_power_W": float(result.compressor.power),
                            "target_gearbox_input_power_W": TARGET},
                "brackets": {"main": list(AB_BRACKET),
                             "provenance": "explicit 120000/150000 Pa static bracket"}})
            continue
        if c_result is None:
            c_result = calculate_architecture_cycle(
                payload_to_input(source[("C_GG", 1)]["architecture_cycle"]))
        gas_generator, free_power = c_result.turbines
        static = _preliminary_static(gas_generator.outlet.p_total,
                                     gas_generator.outlet.k)
        payloads.append({
            "schema_version": INPUT_SCHEMA_VERSION, "coupling": "C",
            "point_index": point.index, "assembly": C_ASSEMBLY,
            "architecture": point.architecture.value,
            "turbine_role": "gas_generator_free_power",
            "stage_counts": {"gas_generator": point.gas_generator_stage_count,
                             "free_power": point.free_power_stage_count},
            "architecture_cycle_source": copy_json_value(
                source[("C_GG", point.gas_generator_stage_count)]["architecture_cycle"]),
            "stage_controls": _common_controls(
                gas_generator=stage_smoke._stage_controls(
                    point.gas_generator_stage_count),
                free_power=stage_smoke._stage_controls(point.free_power_stage_count)),
            "targets": {"gas_generator_compressor_power_W": float(
                            c_result.compressor.power),
                        "free_power_outlet_total_pressure_Pa": float(
                            free_power.outlet.p_total),
                        "target_gearbox_input_power_W": TARGET},
            "brackets": {"gas_generator": [GG_BRACKET_FACTORS[0] * static,
                                           GG_BRACKET_FACTORS[1] * static],
                         "free_power": list(C_FPT_BRACKET),
                         "provenance": "gas_generator: 0.8/1.2 * preliminary 0D static "
                                       "conversion (source k, M_t=0.3379); free_power: "
                                       "explicit 120000/150000 Pa"}})
    return payloads


def _pressure_ratios(ratios):
    """Validate explicit source compressor ratios without an engineering filter."""
    _need(isinstance(ratios, (list, tuple))
          and not isinstance(ratios, (str, bytes)) and bool(ratios),
          "compressor_pressure_ratios must be a non-empty ordered sequence")
    values = []
    for index, value in enumerate(ratios):
        _need(not isinstance(value, bool) and isinstance(value, (int, float))
              and math.isfinite(float(value)) and float(value) > 1.0,
              f"compressor_pressure_ratios[{index}] must be a finite number > 1")
        values.append(float(value))
    return tuple(values)


def with_compressor_pressure_ratios(base_payload, ratios):
    """Copy a plan payload with only explicit source compressor ratios changed.

    The source :class:`ArchitectureCycleInput` is read from
    ``base_payload['architecture_cycle_source']``; only its
    ``compressor_pressure_ratios`` are replaced, then the existing 0D cycle is
    recalculated and serialized with ``input_to_payload``.  Stage counts, role,
    control arrays and the case point are preserved.  Targets are the new exact
    source compressor power and outlet pressure; the 110 kW target is unchanged.
    The C GG static bracket is the same declared reference-Mach conversion at
    the new source k and GG outlet state.  The AB explicit bracket is unchanged.
    This is a pressure-variant payload, not flow/efficiency fitting and not a
    second preliminary meanline.  Requesting the source ratios returns an
    unchanged deep copy so old payload hashes stay bit-identical.
    """
    ratio_tuple = _pressure_ratios(ratios)
    payload = copy_json_value(base_payload)
    request = payload_to_input(payload["architecture_cycle_source"])
    if ratio_tuple == tuple(request.compressor_pressure_ratios):
        return payload
    try:
        variant = replace(request, compressor_pressure_ratios=ratio_tuple)
    except (TypeError, ValueError) as exc:
        raise ArchitectureCoupledSmokeError(str(exc) or type(exc).__name__) from exc
    result = calculate_architecture_cycle(variant)
    payload["architecture_cycle_source"] = input_to_payload(variant)
    note = ("explicit source compressor pressure-ratio variant; targets and "
            "brackets recomputed from the 0D cycle at the requested ratios; "
            "no flow or efficiency fitting and no additional meanline")
    payload["targets"]["provenance"] = note
    if payload["coupling"] == "AB":
        turbine = result.turbines[0]
        payload["targets"]["main_total_outlet_pressure_Pa"] = float(
            turbine.outlet.p_total)
        payload["targets"]["main_compressor_power_W"] = float(
            result.compressor.power)
        payload["brackets"]["provenance"] = (
            "explicit 120000/150000 Pa static bracket; " + note)
    else:
        gas_generator, free_power = result.turbines
        payload["targets"]["gas_generator_compressor_power_W"] = float(
            result.compressor.power)
        payload["targets"]["free_power_outlet_total_pressure_Pa"] = float(
            free_power.outlet.p_total)
        static = _preliminary_static(gas_generator.outlet.p_total,
                                     gas_generator.outlet.k)
        payload["brackets"]["gas_generator"] = [
            GG_BRACKET_FACTORS[0] * static, GG_BRACKET_FACTORS[1] * static]
        payload["brackets"]["provenance"] = (
            "gas_generator: 0.8/1.2 * preliminary 0D static conversion at the "
            f"variant source k (M_t={REFERENCE_EXIT_MACH}); free_power: "
            "explicit 120000/150000 Pa; " + note)
    return payload


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


def make_run_config(payloads, constants, model_digest):
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "config_version": CONFIG_VERSION, "plan_name": PLAN_NAME,
        "payload_plan_count": len(payloads),
        "payload_plan_sha256": canonical_hash(list(payloads)),
        "source_stage_fixture_sha256": canonical_hash(stage_smoke.plan_payloads()),
        "model_variant": MODEL_VARIANT, "evaluator_version": EVALUATOR_VERSION,
        "backend": BACKEND, "model_digest": model_digest,
        "model_digest_sources": list(MODEL_SOURCE_FILES),
        "pressure_ratio_policy": PRESSURE_RATIO_RAW_UNBOUNDED,
        "power_closure_relative_tolerance": POWER_CLOSURE_RELATIVE_TOL,
        "stage_constants": json.loads(json.dumps(constants, allow_nan=False)),
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "stage_count_automatic_selection": "NONE", "coupling_label": COUPLING_LABEL,
    }


def _need(condition, message):
    if not condition:
        raise ArchitectureCoupledSmokeError(message)


def _finite(value, name):
    _need(not isinstance(value, bool) and isinstance(value, (int, float)),
          f"{name} must be a real number")
    value = float(value)
    _need(math.isfinite(value) and value > 0.0, f"{name} must be finite and > 0")
    return value


def _count(value):
    _need(type(value) is int and value in STAGE_COUNTS, "invalid stage count")
    return value


def _stage_block(block, count):
    expected = stage_smoke._stage_controls(count)
    _need(isinstance(block, Mapping) and set(block) == set(expected),
          "stage controls block changed")
    for key, values in expected.items():
        _need(list(block[key]) == list(values), f"stage control {key} was changed")


def _design_block(block, count, role):
    """Validate complete mathematical override arrays for one active role."""
    _need(isinstance(block, Mapping) and set(block) == set(DESIGN_CONTROL_KEYS),
          f"design_overrides.{role} keys changed")
    arrays = {}
    for key in DESIGN_CONTROL_KEYS:
        values = block[key]
        _need(isinstance(values, (list, tuple)) and len(values) == count,
              f"design_overrides.{role}.{key} must have {count} entries")
        _need(all(not isinstance(value, bool)
                  and isinstance(value, (int, float))
                  and math.isfinite(float(value)) for value in values),
              f"design_overrides.{role}.{key} must be finite numbers")
        values = [float(value) for value in values]
        if key == "alpha1_deg":
            _need(all(0.0 < value < 180.0 for value in values),
                  f"design_overrides.{role}.alpha1_deg must be in (0, 180)")
        else:
            _need(all(value > 0.0 for value in values),
                  f"design_overrides.{role}.{key} must be > 0")
        arrays[key] = values
    _need(arrays["pi_weights"][-1] == 1.0,
          f"design_overrides.{role}.pi_weights[-1] must be 1.0")
    return arrays


def _validate_payload(payload) -> PlanPoint:
    top = {"schema_version", "point_index", "coupling", "assembly", "architecture",
           "turbine_role", "stage_counts", "architecture_cycle_source",
           "stage_controls", "targets", "brackets"}
    allowed = top | {"air_mass_flow_kg_s"}
    _need(isinstance(payload, Mapping) and top <= set(payload) <= allowed,
          "payload keys or schema changed")
    if "air_mass_flow_kg_s" in payload:
        _finite(payload["air_mass_flow_kg_s"], "air_mass_flow_kg_s")
    _need(payload["schema_version"] == INPUT_SCHEMA_VERSION, "schema version changed")
    coupling, assembly = payload["coupling"], payload["assembly"]
    if coupling == "AB":
        _need(assembly in AB_ASSEMBLIES, "AB assembly mismatch")
        counts = {"main": _count(payload["stage_counts"]["main"])}
    elif coupling == "C":
        _need(assembly == C_ASSEMBLY, "C assembly mismatch")
        counts = {key: _count(payload["stage_counts"][key])
                  for key in ("gas_generator", "free_power")}
    else:
        raise ArchitectureCoupledSmokeError("coupling must be 'AB' or 'C'")
    architecture = Architecture(payload["architecture"])
    controls = payload["stage_controls"]
    _need(isinstance(controls, Mapping) and set(controls) == {
        "speed_rpm", "hub_diameter_m", "reference_exit_mach",
        "boundary_is_static", "pressure_ratio_policy", "speed_basis"} | set(counts)
        | ({"operating_points"} if "operating_points" in controls else set())
        | ({"design_overrides"} if "design_overrides" in controls else set())
        | ({"reaction_table_policy"} if "reaction_table_policy" in controls
           else set()),
        "stage_controls keys changed")
    for key, value in (("speed_rpm", SPEED_RPM), ("hub_diameter_m", HUB_DIAMETER_M),
                       ("reference_exit_mach", REFERENCE_EXIT_MACH),
                       ("boundary_is_static", True), ("speed_basis", SPEED_BASIS),
                       ("pressure_ratio_policy", PRESSURE_RATIO_RAW_UNBOUNDED)):
        _need(controls[key] == value, f"stage_controls.{key} was changed")
    if "reaction_table_policy" in controls:
        _need(controls["reaction_table_policy"] in REACTION_TABLE_POLICIES,
              "stage_controls.reaction_table_policy must be one of "
              f"{REACTION_TABLE_POLICIES!r}")
    for key, count in counts.items():
        _stage_block(controls[key], count)
    if "design_overrides" in controls:
        overrides = controls["design_overrides"]
        _need(isinstance(overrides, Mapping) and set(overrides) == set(counts),
              "design_overrides must cover exactly the active turbine roles")
        for role, count in counts.items():
            _design_block(overrides[role], count, role)
    if "operating_points" in controls:
        points = controls["operating_points"]
        _need(isinstance(points, Mapping) and set(points) == set(counts),
              "operating_points must cover exactly the active turbine roles")
        for role, point in points.items():
            _need(isinstance(point, Mapping)
                  and set(point) == {"speed_rpm", "hub_diameter_m"},
                  f"operating_points.{role} keys changed")
            _finite(point["speed_rpm"], f"operating_points.{role}.speed_rpm")
            _finite(point["hub_diameter_m"],
                    f"operating_points.{role}.hub_diameter_m")
    brackets = payload["brackets"]
    _need(isinstance(brackets, Mapping), "brackets must be an object")
    for key in counts:
        pair = brackets[key]
        _need(isinstance(pair, (list, tuple)) and len(pair) == 2,
              f"brackets.{key} must be a pair")
        _need(_finite(pair[0], "bracket low") < _finite(pair[1], "bracket high"),
              f"brackets.{key} is not ordered")
    request = payload_to_input(payload["architecture_cycle_source"])
    _need(request.gas_property_basis == TEXTBOOK and request.architecture is architecture,
          "source cycle basis or architecture changed")
    return PlanPoint(payload["point_index"], coupling, assembly,
                     counts.get("main"), counts.get("gas_generator"),
                     counts.get("free_power"), architecture)


def _validate_plan(payloads):
    points = plan_points()
    _need(len(payloads) == len(points), "plan must contain 15 points")
    for payload in payloads:
        _validate_payload(payload)
    identifiers = [make_candidate_id(point.architecture.value, MODEL_VARIANT, payload)
                   for point, payload in zip(points, payloads)]
    _need(len(set(identifiers)) == len(identifiers), "duplicate plan inputs")


def _bound_orchestration(namespace):
    for value in namespace.values():
        owner = getattr(value, "__self__", None)
        if inspect.ismethod(value) and isinstance(owner, ConstHubOrchestration):
            return owner
    raise ArchitectureCoupledSmokeError("fresh namespace has no bound orchestration")


def _fresh_namespace(model, result, turbine, count, controls, p_static,
                     gas_flow=None):
    namespace = stage_smoke._build_namespace(
        model, result, turbine, count, controls, p_static,
        PRESSURE_RATIO_RAW_UNBOUNDED)
    if gas_flow is not None:
        namespace.update({"G_в": float(gas_flow), "G_c1": float(gas_flow),
                          "G_л_ост": float(gas_flow),
                          "G_base": (float(gas_flow),) * 3})
    return namespace, _bound_orchestration(namespace)


def _call_closure(namespace, location, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except namespace["НетРешения"] as exc:
        raise CoupledNoSolution(str(exc) or "no stage geometry solution", location,
                                {"location": location}) from exc


def _power(kind, net, extra):
    residual = net - TARGET
    relative = abs(residual) / TARGET
    closed = relative <= POWER_CLOSURE_RELATIVE_TOL
    return {"kind": kind, **extra, "net_mechanical_power_W": net,
            "target_gearbox_input_power_W": TARGET,
            "signed_power_residual_W": residual,
            "relative_power_residual": relative, "engine_power_closed": closed,
            "engine_power_label": POWER_CLOSED_LABEL if closed
            else POWER_NOT_CLOSED_LABEL}


def _base_output(payload, result, posthoc):
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION, "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "coupling_label": COUPLING_LABEL,
        "assembly": payload["assembly"], "stage_counts": payload["stage_counts"],
        "stage_count_selection": "NONE", "pressure_ratio_policy": PRESSURE_RATIO_RAW_UNBOUNDED,
        "source_input_sha256": canonical_hash(payload["architecture_cycle_source"]),
        "source_gas_basis": result.gas_property_basis,
        "posthoc_pi_acceptance": posthoc, "rho_065_posthoc": {"status": "NOT_APPLIED"},
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "overall_engineering_feasibility_reason": "strength_verification_missing",
    }


def _sizing(payload, result):
    """Return explicit unknown-air-flow trial values, or None when absent."""
    raw = payload.get("air_mass_flow_kg_s")
    if raw is None:
        return None
    air = _finite(raw, "air_mass_flow_kg_s")
    source_air = float(result.air_mass_flow)
    source_gas = float(result.gas_mass_flow)
    source_fuel = float(result.fuel_mass_flow)
    ratio = air / source_air
    gas, fuel = air * (source_gas / source_air), source_fuel * ratio
    return {"air": air, "gas": gas, "fuel": fuel, "ratio": ratio,
            "source_air": source_air, "source_gas": source_gas,
            "source_fuel": source_fuel,
            "compressor_power": float(result.compressor.specific_work) * air,
            "mass_residual": gas - (air + fuel)}


def _flow_output(sizing, result):
    """Explicit design-flow provenance; never claims the 0D flow was sized."""
    return {
        "kind": "EXPLICIT_UNKNOWN_AIR_FLOW_DESIGN_TRIAL",
        "sizing_basis": "CYCLE_BOOK_PRINTED_P22_SPECIFIC_POWER_PRE_REDUCER",
        "source_book": "cycle textbook printed p.22 section 3.2",
        "source_equation": "N_specific=N/G_air; "
                           "N=(L_t*G_gas*eta_m-L_c*G_air)*eta_reducer; "
                           "eta_reducer omitted at the pre-reducer 110 kW plane",
        "pre_reducer": True,
        "source_flow_role": "PRELIMINARY_0D_NOT_A_SIZED_DESIGN_FLOW",
        "source_air_mass_flow_kg_s": sizing["source_air"],
        "source_gas_mass_flow_kg_s": sizing["source_gas"],
        "source_fuel_mass_flow_kg_s": sizing["source_fuel"],
        "trial_air_mass_flow_kg_s": sizing["air"],
        "trial_gas_mass_flow_kg_s": sizing["gas"],
        "trial_fuel_mass_flow_kg_s": sizing["fuel"],
        "air_flow_ratio_to_source_0d": sizing["ratio"],
        "combustor_alpha": float(result.combustor.alpha),
        "alpha_unchanged": True, "mass_residual_kg_s": sizing["mass_residual"],
        "gas_property_basis": result.gas_property_basis,
        "actual_compressor_power_target_W": sizing["compressor_power"],
        "compressor_power_basis":
            "actual compressor specific work times explicit trial air flow",
        "note": "every trial re-evaluates the actual meanline and GG/FPT "
                "closures at the trial air flow; stored stage powers are not "
                "rescaled from the 0D result",
    }


def _evaluate_ab(payload, model):
    result = calculate_architecture_cycle(
        payload_to_input(payload["architecture_cycle_source"]))
    turbine = result.turbines[0]
    target = float(payload["targets"]["main_total_outlet_pressure_Pa"])
    sizing = _sizing(payload, result)
    compressor_power = (sizing["compressor_power"] if sizing is not None
                        else float(result.compressor.power))
    _need(target == float(turbine.outlet.p_total)
          and float(payload["targets"]["main_compressor_power_W"])
          == float(result.compressor.power), "AB target was changed")
    namespace, orchestration = _fresh_namespace(
        model, result, turbine, payload["stage_counts"]["main"],
        _controls(payload, "main"),
        _preliminary_static(turbine.outlet.p_total, turbine.outlet.k),
        None if sizing is None else sizing["gas"])
    main_operating = _operating(payload, "main")
    n, hub = main_operating[:2]
    closure = _call_closure(
        namespace, "AB_MAIN.solve_total_outlet_pressure",
        orchestration.solve_total_outlet_pressure, n, hub,
        target, static_pressure_bracket=payload["brackets"]["main"])
    stages = closure["actual_stages"]
    efficiency = float(result.shaft.mechanical_efficiencies[0])
    stage_gas_power = sum(stage.G_c * stage.L_t for stage
                          in (StageResult.from_legacy_dict(item) for item in stages))
    mechanical = efficiency * stage_gas_power
    boundary = {
        "target_kind": "total_outlet_pressure", "source_0d_outlet_total_pressure_Pa": float(turbine.outlet.p_total),
        "target_outlet_total_pressure_Pa": target, "actual_outlet_total_pressure_Pa": float(closure["actual_metric"]),
        "signed_total_pressure_residual_Pa": float(closure["signed_residual"]), "relative_total_pressure_residual": float(closure["relative_residual"]),
        "source_target_differs_from_actual": float(closure["actual_metric"]) != target,
        "closure": stage_smoke._json_value(closure, "ab.closure")}
    outputs = _base_output(
        payload, result, {"main": stage_smoke._posthoc_pi_acceptance(stages)})
    operating = _operating_output({"main": main_operating})
    if operating is not None:
        outputs["operating_points"] = operating
    design = _design_output(payload)
    if design is not None:
        outputs["design_controls"] = design
    outputs["actual_boundary"] = boundary
    power_extra = {
        "mechanical_efficiency": efficiency, "stage_gas_power_W": stage_gas_power,
        "stage_mechanical_power_W": mechanical, "compressor_power_W": compressor_power,
        "compressor_power_basis": "exactly source result.compressor.power"}
    if sizing is not None:
        power_extra.update({
            "reference_0d_compressor_power_W": float(result.compressor.power),
            "actual_sized_compressor_power_W": compressor_power,
            "sized_air_mass_flow_kg_s": sizing["air"],
            "compressor_power_basis": "actual specific work times explicit "
                                      "trial air flow"})
        outputs["flow_sizing"] = _flow_output(sizing, result)
    outputs["power"] = _power(
        "AB_NET_STAGE_MECHANICAL_MINUS_SOURCE_COMPRESSOR_POWER",
        mechanical - compressor_power, power_extra)
    policy = _reaction_table_policy(payload)
    if policy == REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN:
        try:
            outputs["reaction_table_source_domain"] = (
                _enforce_reaction_table_domain(
                    "main", closure, namespace, policy=policy,
                    flow_metadata=(None if sizing is None
                                   else _flow_output(sizing, result))))
        except SourceTableDomainError as exc:
            try:
                exc.partial = stage_smoke._json_value(outputs, "ab.outputs")
            except stage_smoke.StageNumericalFailure:
                exc.partial = {"actual_boundary": boundary}
            raise
    try:
        return stage_smoke._json_value(outputs, "ab.outputs")
    except stage_smoke.StageNumericalFailure as exc:
        exc.partial = {"actual_boundary": boundary}
        raise


def _evaluate_c(payload, model):
    result = calculate_architecture_cycle(
        payload_to_input(payload["architecture_cycle_source"]))
    gas_generator, free_power = result.turbines
    sizing = _sizing(payload, result)
    reference_compressor_power = float(result.compressor.power)
    compressor_power = (sizing["compressor_power"] if sizing is not None
                        else reference_compressor_power)
    gg_target = float(payload["targets"]["gas_generator_compressor_power_W"])
    fpt_target = float(payload["targets"]["free_power_outlet_total_pressure_Pa"])
    policy = _reaction_table_policy(payload)
    flow_metadata = None if sizing is None else _flow_output(sizing, result)
    _need(gg_target == reference_compressor_power
          and fpt_target == float(free_power.outlet.p_total), "C target was changed")
    gg_eff, fpt_eff = (float(item) for item
                       in result.shaft.mechanical_efficiencies[:2])
    gas_flow = None if sizing is None else sizing["gas"]
    gg_operating = _operating(payload, "gas_generator")
    fpt_operating = _operating(payload, "free_power")
    gg_namespace, gg_orchestration = _fresh_namespace(
        model, result, gas_generator, payload["stage_counts"]["gas_generator"],
        _controls(payload, "gas_generator"),
        _preliminary_static(gas_generator.outlet.p_total, gas_generator.outlet.k),
        gas_flow)
    gg_closure = _call_closure(
        gg_namespace, "C_GAS_GENERATOR.solve_prescribed_mechanical_power",
        gg_orchestration.solve_prescribed_mechanical_power, gg_operating[0],
        gg_operating[1], compressor_power, gg_eff,
        static_pressure_bracket=payload["brackets"]["gas_generator"])
    gg_typed = [StageResult.from_legacy_dict(item)
                for item in gg_closure["actual_stages"]]
    gg_last, gg_gas = gg_typed[-1], sum(item.G_c * item.L_t for item in gg_typed)
    gg_mech = gg_eff * gg_gas
    gg_record = {
        "target_compressor_power_W": compressor_power,
        "actual_mechanical_power_W": float(gg_closure["actual_metric"]),
        "gas_generator_balance_residual_W": gg_mech - compressor_power,
        "closure": stage_smoke._json_value(gg_closure, "c.gas_generator")}
    if sizing is not None:
        gg_record["reference_0d_compressor_power_W"] = reference_compressor_power
        gg_record["target_compressor_power_basis"] = (
            "actual specific work times explicit trial air flow")
    partial = {"gas_generator": gg_record}
    gg_domain = None
    if policy == REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN:
        try:
            gg_domain = _enforce_reaction_table_domain(
                "gas_generator", gg_closure, gg_namespace, policy=policy,
                flow_metadata=flow_metadata)
        except SourceTableDomainError as exc:
            exc.partial = dict(partial)
            if flow_metadata is not None:
                exc.partial["flow_sizing"] = flow_metadata
            raise
    fpt_namespace, fpt_orchestration = _fresh_namespace(
        model, result, free_power, payload["stage_counts"]["free_power"],
        _controls(payload, "free_power"),
        _preliminary_static(free_power.outlet.p_total, free_power.outlet.k),
        gas_flow)
    flow_keys = ("G_в", "G_c1", "G_л_ост")
    flow_before = {key: float(fpt_namespace[key]) for key in flow_keys}
    R_before, alpha_before = (float(fpt_namespace["R"]),
                              float(fpt_namespace["α_г"]))
    fpt_namespace["T_г_полн"] = float(gg_last.T_total)
    fpt_namespace["p_г_полн"] = float(gg_last.p_total)
    unchanged = ({key: float(fpt_namespace[key]) for key in flow_keys} == flow_before
                 and float(fpt_namespace["R"]) == R_before
                 and float(fpt_namespace["α_г"]) == alpha_before)
    _need(unchanged, "FPT flow/R/alpha changed during GG state transfer")
    try:
        fpt_closure = _call_closure(
            fpt_namespace, "C_FREE_POWER.solve_total_outlet_pressure",
            fpt_orchestration.solve_total_outlet_pressure, fpt_operating[0],
            fpt_operating[1], fpt_target,
            static_pressure_bracket=payload["brackets"]["free_power"])
    except CoupledNoSolution as exc:
        exc.partial = partial
        raise
    except SolverDiagnosticsError as exc:
        exc.partial_outputs = partial
        raise
    fpt_typed = [StageResult.from_legacy_dict(item)
                 for item in fpt_closure["actual_stages"]]
    fpt_first = fpt_typed[0]
    source_flow = float(result.gas_mass_flow)
    sized_flow = source_flow if gas_flow is None else float(gas_flow)
    continuity = {
        "gas_generator_last_stage_T_total_K": float(gg_last.T_total), "gas_generator_last_stage_p_total_Pa": float(gg_last.p_total),
        "free_power_first_stage_T0_K": float(fpt_first.T0), "free_power_first_stage_p0_Pa": float(fpt_first.p0),
        "temperature_exact": fpt_first.T0 == gg_last.T_total, "pressure_exact": fpt_first.p0 == gg_last.p_total,
        "flow_exact": fpt_first.G_c == gg_last.G_c, "source_flow_kg_s": source_flow,
        "flow_matches_source": fpt_first.G_c == source_flow,
        "flow_R_alpha_unchanged": unchanged}
    if sizing is not None:
        continuity.update({
            "sized_gas_flow_kg_s": sized_flow,
            "flow_matches_sized": fpt_first.G_c == sized_flow,
            "flow_check_target": "ACTUAL_SIZED_FLOW"})
    _need(continuity["temperature_exact"] and continuity["pressure_exact"]
          and continuity["flow_exact"]
          and fpt_first.G_c == sized_flow,
          "GG to FPT total state continuity is not exact")
    fpt_gas = sum(item.G_c * item.L_t for item in fpt_typed)
    fpt_mech = fpt_eff * fpt_gas
    fpt_record = {
        "target_outlet_total_pressure_Pa": fpt_target,
        "actual_outlet_total_pressure_Pa": float(fpt_closure["actual_metric"]),
        "mechanical_efficiency": fpt_eff, "stage_gas_power_W": fpt_gas,
        "closure": stage_smoke._json_value(fpt_closure, "c.free_power")}
    fpt_domain = None
    if policy == REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN:
        try:
            fpt_domain = _enforce_reaction_table_domain(
                "free_power", fpt_closure, fpt_namespace, policy=policy,
                flow_metadata=flow_metadata)
        except SourceTableDomainError as exc:
            exc.partial = {**partial, "free_power": fpt_record}
            if flow_metadata is not None:
                exc.partial["flow_sizing"] = flow_metadata
            raise
    outputs = _base_output(payload, result, {
        "gas_generator": stage_smoke._posthoc_pi_acceptance(
            gg_closure["actual_stages"]),
        "free_power": stage_smoke._posthoc_pi_acceptance(
            fpt_closure["actual_stages"])})
    operating = _operating_output({
        "gas_generator": gg_operating, "free_power": fpt_operating})
    if operating is not None:
        outputs["operating_points"] = operating
    design = _design_output(payload)
    if design is not None:
        outputs["design_controls"] = design
    outputs["actual_boundary"] = {
        "gas_generator": gg_record, "free_power": fpt_record,
        "gas_generator_to_free_power_continuity": continuity}
    if policy == REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN:
        outputs["reaction_table_source_domain"] = {
            "policy": policy, "status": "FINAL_SOURCE_DOMAIN_VERIFIED",
            "final_domain_verified": True,
            "table_bounds": gg_domain["table_bounds"],
            "source_domain_tolerance": REACTION_TABLE_DOMAIN_TOL,
            "roles": {"gas_generator": gg_domain, "free_power": fpt_domain},
            "intermediate_clipping":
                "NUMERICAL_INITIALIZATION_ONLY_NOT_VALIDATED_STATES",
            "note": "only the accepted final actual GG and FPT closures are "
                    "checked; intermediate reaction clipping is numerical "
                    "initialization and is not a validated state"}
    if flow_metadata is not None:
        outputs["flow_sizing"] = flow_metadata
    gg_power = {
        "mechanical_efficiency": gg_eff, "stage_gas_power_W": gg_gas,
        "mechanical_power_W": gg_mech, "compressor_power_W": compressor_power,
        "balance_residual_W": gg_mech - compressor_power}
    fpt_power = {
        "mechanical_efficiency": fpt_eff, "stage_gas_power_W": fpt_gas,
        "mechanical_power_W": fpt_mech}
    if sizing is not None:
        gg_power["reference_0d_compressor_power_W"] = reference_compressor_power
        fpt_power["sized_air_mass_flow_kg_s"] = sizing["air"]
    outputs["power"] = _power(
        "C_FPT_MECHANICAL_POWER_GG_BALANCED_AGAINST_COMPRESSOR", fpt_mech, {
            "gas_generator": gg_power, "free_power": fpt_power})
    try:
        return stage_smoke._json_value(outputs, "c.outputs")
    except stage_smoke.StageNumericalFailure as exc:
        exc.partial = partial
        raise


def _failure(status, stage, reason, input_payload, diagnostics, outputs=None,
             last=None):
    return EvaluationOutcome.failure(
        status, stage, reason, last_valid_state=last or {"input": input_payload},
        diagnostics=diagnostics, outputs=outputs)


def evaluate_coupled_payload(payload, *, model=None):
    """Evaluate one plan point, preserving typed failures and partial outputs."""
    try:
        input_payload = copy_json_value(payload, path="$.payload")
    except PayloadValidationError as exc:
        return _failure(CalculationStatus.MODEL_ERROR, "payload_validation",
                        str(exc), {}, {"error_type": type(exc).__name__})
    try:
        point = _validate_payload(payload)
    except (ArchitectureCoupledSmokeError, ValueError, TypeError, KeyError) as exc:
        return _failure(CalculationStatus.MODEL_ERROR, "payload_validation",
                        str(exc) or type(exc).__name__, input_payload,
                        {"error_type": type(exc).__name__})
    if model is None:
        model = _get_model()
    diag = {"point_index": point.index, "coupling": point.coupling}
    try:
        outputs = (_evaluate_ab(payload, model) if point.coupling == "AB"
                   else _evaluate_c(payload, model))
    except CoupledNoSolution as exc:
        return _failure(CalculationStatus.NON_PHYSICAL_STATE,
                        "stage_geometry_domain", exc.reason, input_payload,
                        {**diag, "location": exc.location}, exc.partial or None,
                        {"input": input_payload, "location": exc.location})
    except SourceTableDomainError as exc:
        return _failure(CalculationStatus.MODEL_ERROR,
                        "reaction_table_source_domain", exc.reason,
                        input_payload, {**diag, "source_domain": exc.as_record()},
                        exc.partial or None,
                        {"input": input_payload, "role": exc.role,
                         "stage": exc.stage,
                         "table_bounds": list(exc.table_bounds)})
    except stage_smoke.StageNumericalFailure as exc:
        return _failure(CalculationStatus.NUMERICAL_FAILURE, "stage_numerical",
                        "non_finite_coupled_output", input_payload,
                        {**diag, "path": exc.path}, exc.partial or None,
                        {"input": input_payload, "path": exc.path})
    except SolverDiagnosticsError as exc:
        return _failure(CalculationStatus(exc.status),
                        exc.stage or "coupled_stage_solver",
                        exc.reason or type(exc).__name__, input_payload,
                        {**diag, "solver": exc.as_record()},
                        getattr(exc, "partial_outputs", None),
                        {"input": input_payload,
                         "solver_last_valid_state": exc.last_valid_state})
    except ArchitectureCoupledSmokeError as exc:
        return _failure(CalculationStatus.MODEL_ERROR, "coupled_control_error",
                        str(exc) or type(exc).__name__, input_payload,
                        {**diag, "error_type": type(exc).__name__})
    return EvaluationOutcome.success(outputs=outputs, diagnostics=diag)


def _open_writer(path, resume, config):
    if path.exists() and any(path.iterdir()):
        if not resume:
            raise ArchitectureCoupledSmokeError("destination is not empty")
        return DatasetWriter.open(path, expected_run_config=config,
                                  expected_evaluator_version=EVALUATOR_VERSION)
    if resume:
        raise ArchitectureCoupledSmokeError("cannot resume an empty/missing dataset")
    return DatasetWriter.create(path, run_config=config)


def _point_entry(point, record):
    success = record.calculation_status is CalculationStatus.SUCCESS
    power = (record.outputs or {}).get("power") or {}
    return {
        "candidate_id": record.candidate_id,
        "calculation_status": record.calculation_status.value,
        "failure_stage": record.failure_stage,
        "failure_reason": record.failure_reason,
        "net_mechanical_power_W": power.get("net_mechanical_power_W") if success else None,
        "signed_power_residual_W": power.get("signed_power_residual_W") if success else None,
        "relative_power_residual": power.get("relative_power_residual") if success else None,
        "engine_power_closed": power.get("engine_power_closed") if success else None,
        "engine_power_label": power.get("engine_power_label") if success else None,
        "overall_engineering_feasibility_status": "NOT_CHECKED" if success else None,
    }


def _build_summary(points, payloads, records, manifest, appended, elapsed,
                   reproduce, model_digest):
    statuses = Counter(record.calculation_status.value for record in records)
    return {
        "report_version": SUMMARY_VERSION, "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "coupling_label": COUPLING_LABEL,
        "experiment": {"plan_name": PLAN_NAME, "payload_plan_count": len(payloads),
                       "payload_plan_sha256": canonical_hash(list(payloads)),
                       "model_digest": model_digest},
        "run": {"run_id": manifest["run_id"],
                "record_count": manifest["record_count"],
                "complete": manifest["complete"],
                "journal_sha256": manifest["journal_sha256"],
                "points_attempted": len(points),
                "records_appended_this_invocation": appended,
                "elapsed_seconds": round(elapsed, 6)},
        "status_counts": dict(sorted(statuses.items())),
        "points": [_point_entry(point, record)
                   for point, record in zip(points, records)],
        "reproduce": copy_json_value(reproduce),
    }


def run_plan(destination, *, resume=False, evaluator=None, model=None,
             constants=None, model_digest=None, reproduce=True):
    """Evaluate the fixed 15-point plan, store all attempts, and summarise."""
    if constants is None:
        if model is None:
            model = _get_model()
        template, _ = oe.build_ns(
            model, gas_property_basis=TEXTBOOK, cycle_gas_property_basis=TEXTBOOK)
        constants = stage_smoke._stage_constants(template)
    model_digest = model_digest or _digest(MODEL_SOURCE_FILES)
    if evaluator is None:
        if model is None:
            model = _get_model()
        evaluate = lambda item: evaluate_coupled_payload(item, model=model)
    else:
        evaluate = evaluator
    points, payloads = plan_points(), plan_payloads()
    _validate_plan(payloads)
    config = make_run_config(payloads, constants, model_digest)
    path = Path(destination)
    writer = _open_writer(path, resume=resume, config=config)
    existing = ({record.candidate_id for record in read_records(path)}
                if resume else set())
    appended, started, completed = 0, time.perf_counter(), False
    try:
        for index, (point, payload) in enumerate(zip(points, payloads)):
            candidate_id = make_candidate_id(
                point.architecture.value, MODEL_VARIANT, payload)
            if candidate_id in existing:
                continue
            before = writer.record_count
            writer.evaluate_and_append(
                evaluate, payload, architecture=point.architecture.value,
                sampling_method="actual_boundary_stage_fixture",
                sampling_metadata={"plan": PLAN_NAME, "index": index})
            appended += writer.record_count > before
        completed = True
    finally:
        writer.close(finalize=completed)
    elapsed = time.perf_counter() - started
    records = sorted(read_records(path), key=lambda record: record.payload["point_index"])
    _need([record.payload["point_index"] for record in records] == list(range(15)),
          "journal records differ from plan")
    manifest = load_manifest(path)
    reproduce_result = (stage_smoke._reproduce_all(records, evaluate)
                        if reproduce else {
                            "checked": 0, "matches": None, "differences": [],
                            "rtol": 1e-12, "atol": 1e-9})
    summary = _build_summary(points, payloads, records, manifest, appended,
                             elapsed, reproduce_result, model_digest)
    stage_smoke._write_summary_atomic(path / SUMMARY_NAME, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        "Bounded 15-attempt actual-boundary stage coupling smoke; not a sweep, "
        "design selection, or physical validation."))
    parser.add_argument("--destination", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_plan(args.destination, resume=args.resume)
    except (ArchitectureCoupledSmokeError, DatasetError,
            stage_smoke.ArchitectureStageSmokeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False))
    return 3 if summary["reproduce"]["matches"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
