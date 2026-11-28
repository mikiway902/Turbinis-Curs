"""Focused tests for the explicit-air-flow coupled sizing smoke."""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "Optimizer", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import architecture_coupled_stage_smoke as coupled  # noqa: E402
import architecture_flow_sizing_smoke as smoke  # noqa: E402
from turbine_engine.dataset import (  # noqa: E402
    DatasetError, load_manifest, read_records)
from turbine_engine.diagnostics import NumericalConvergenceError  # noqa: E402
from turbine_engine.orchestration import ConstHubOrchestration  # noqa: E402
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus, EvaluationOutcome, make_candidate_id)
from turbine_engine.sampling import split_pressure_ratio_log  # noqa: E402


def _outputs(metric):
    residual = metric - smoke.TARGET
    return {
        "power": {
            "net_mechanical_power_W": metric,
            "signed_power_residual_W": residual,
            "relative_power_residual": abs(residual) / smoke.TARGET,
            "engine_power_closed": abs(residual) <= 1e-6 * smoke.TARGET,
        },
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
    }


class _Record:
    def __init__(self, air, metric=None, *, status=CalculationStatus.SUCCESS):
        self.payload = {"air_mass_flow_kg_s": air}
        self.calculation_status = status
        self.outputs = None if metric is None else _outputs(metric)
        self.candidate_id = f"cand_{air!r}"
        self.failure_stage = None if metric is not None else "unit.flow"
        self.failure_reason = None if metric is not None else "unit_failure"


def _case(bracket=(1.0, 2.0)):
    return {"case": "UNIT", "source_point_index": 0,
            "architecture": "SINGLE_SHAFT_SINGLE_CC", "assembly": "B_MAIN",
            "coupling": "AB", "air_flow_bracket_kg_s": list(bracket)}


def _case_entry(index, name, bracket, points=None, design=None, ratios=None):
    payload = copy.deepcopy(coupled.plan_payloads()[index])
    if ratios is not None:
        payload = coupled.with_compressor_pressure_ratios(payload, ratios)
    if points is not None:
        payload["stage_controls"]["operating_points"] = copy.deepcopy(points)
    if design is not None:
        payload["stage_controls"]["design_overrides"] = copy.deepcopy(design)
    result = coupled.calculate_architecture_cycle(
        coupled.payload_to_input(payload["architecture_cycle_source"]))
    return {
        "case": name, "source_point_index": index,
        "assembly": payload["assembly"], "coupling": payload["coupling"],
        "architecture": payload["architecture"], "source_payload": payload,
        "source_payload_sha256": coupled.canonical_hash(payload),
        "source_air_mass_flow_kg_s": float(result.air_mass_flow),
        "source_gas_mass_flow_kg_s": float(result.gas_mass_flow),
        "source_fuel_mass_flow_kg_s": float(result.fuel_mass_flow),
        "air_flow_bracket_kg_s": list(bracket),
        "target_gearbox_input_power_W": smoke.TARGET,
        "outer_relative_tolerance": smoke.OUTER_RELATIVE_TOL,
        "outer_max_iterations": smoke.OUTER_MAX_ITER}


def _design_c():
    return {
        "gas_generator": {
            "alpha1_deg": [11.0, 11.0], "kappa": [1.38, 1.12],
            "pi_weights": [1.0, 1.0]},
        "free_power": {
            "alpha1_deg": [11.0], "kappa": [1.38],
            "pi_weights": [1.0]}}


def _points_c():
    return {
        "gas_generator": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065},
        "free_power": {"speed_rpm": 61_000.0, "hub_diameter_m": 0.08}}


def _case_evaluator(payload):
    root = 0.75 if payload["point_index"] == 2 else 0.70
    return EvaluationOutcome.success(outputs=_outputs(
        smoke.TARGET + 400_000.0 * (payload["air_mass_flow_kg_s"] - root)))


def _synthetic_evaluator(payload):
    point = payload["point_index"]
    air = payload["air_mass_flow_kg_s"]
    if point == 2:
        if 0.75 < air < 0.82:
            return EvaluationOutcome.failure(
                CalculationStatus.NUMERICAL_FAILURE, "unit.flow",
                "unit_trial_failure")
        metric = smoke.TARGET + 300_000.0 * (air - 0.75)
    elif point == 5:
        metric = smoke.TARGET - 500_000.0 * (air - 0.70)
    else:
        metric = smoke.TARGET + 400_000.0 * (air - 0.73)
    return EvaluationOutcome.success(outputs=_outputs(metric))


class ArchitectureFlowSizingSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="architecture-flow-sizing-")
        cls.destination = Path(cls.temporary.name) / "dataset"
        cls.summary = smoke.run_plan(cls.destination)
        cls.records = read_records(cls.destination)
        cls.manifest = load_manifest(cls.destination)
        cls.constants = cls.manifest["run_config"]["stage_constants"]
        cls.model_digest = cls.manifest["run_config"]["model_digest"]

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temporary.name, ignore_errors=True)

    def _case_summary(self, name):
        for item in self.summary["cases"]:
            if item["case"] == name:
                return item
        self.fail(f"missing case {name}")

    def test_fixed_three_cases_and_explicit_source_brackets(self):
        cases = smoke.plan_cases()
        self.assertEqual([case["case"] for case in cases],
                         ["A_MAIN_3", "B_MAIN_3", "C_GG3_FPT1"])
        self.assertEqual([case["source_point_index"] for case in cases],
                         [2, 5, 12])
        for case in cases:
            air = case["source_air_mass_flow_kg_s"]
            self.assertGreater(air, 0.0)
            self.assertEqual(case["air_flow_bracket_kg_s"],
                             [0.9 * air, 1.4 * air])
            self.assertEqual(case["target_gearbox_input_power_W"], 110_000.0)
            self.assertEqual(case["outer_relative_tolerance"], 1e-6)
            self.assertEqual(case["outer_max_iterations"], 30)
            self.assertNotIn("air_mass_flow_kg_s", case["source_payload"])
        config = self.manifest["run_config"]
        self.assertEqual(config["sizing_basis"], smoke.SIZING_BASIS)
        self.assertTrue(config["pre_reducer"])
        self.assertFalse(config["eta_reducer_applied"])
        self.assertEqual(config["boundary_closure_relative_tolerance"], 1e-6)
        self.assertIn(smoke.NEW_SMOKE_FILE, config["model_digest_sources"])
        self.assertIn("scripts/architecture_coupled_stage_smoke.py",
                      config["model_digest_sources"])
        self.assertNotIn(
            "scripts/test_architecture_flow_sizing_smoke.py",
            config["model_digest_sources"])
        for item in config["cases"]:
            self.assertNotIn("stage_controls_sha256", item)
            self.assertNotIn("design_overrides", item)
            self.assertNotIn("operating_points", item)

    def test_bisection_endpoint_root(self):
        result = smoke._solve_case(
            _case(), lambda air: _Record(air, smoke.TARGET))
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["reason"], "root_at_bracket_low")
        self.assertEqual(result["trial_count"], 1)
        self.assertEqual(result["root"]["candidate_id"],
                         result["evaluated_states"][0]["candidate_id"])

    def test_bracket_not_root_is_typed_numerical_failure(self):
        result = smoke._solve_case(
            _case(), lambda air: _Record(air, smoke.TARGET + 5.0))
        self.assertEqual(result["status"], "NUMERICAL_FAILURE")
        self.assertEqual(result["reason"], "outer_bracket_not_bracketed")
        self.assertIsNone(result["root"])
        self.assertEqual(result["trial_count"], 2)
        self.assertIsNone(result["overall_engineering_feasibility"])
        self.assertEqual(
            result["overall_engineering_feasibility_status"], "NOT_CHECKED")

    def test_iteration_limit_keeps_last_evaluated_state(self):
        def trial(air):
            return _Record(air, smoke.TARGET + 100_000.0 * (air - 1.6))

        result = smoke._solve_case(_case(), trial, max_iter=1,
                                   relative_tol=1e-9)
        self.assertEqual(result["status"], "NUMERICAL_FAILURE")
        self.assertEqual(result["reason"], "outer_iteration_limit")
        self.assertEqual(result["trial_count"], 3)
        self.assertEqual(result["last_evaluated"]["air_mass_flow_kg_s"], 1.5)
        self.assertIsNone(result["root"])

    def test_typed_trial_failure_stops_the_case_and_is_preserved(self):
        def failing(air):
            return _Record(air, None,
                           status=CalculationStatus.NUMERICAL_FAILURE)

        low = smoke._solve_case(_case(), failing)
        self.assertEqual(low["status"], "NUMERICAL_FAILURE")
        self.assertEqual(low["reason"], "trial_failure_at_air_flow")
        self.assertEqual(low["trial_count"], 1)
        self.assertEqual(low["evaluated_states"][0]["calculation_status"],
                         "NUMERICAL_FAILURE")

        def high_fails(air):
            return _Record(air, smoke.TARGET - 1.0 if air == 1.0 else None,
                           status=(CalculationStatus.SUCCESS if air == 1.0
                                   else CalculationStatus.MODEL_ERROR))

        high = smoke._solve_case(_case(), high_fails)
        self.assertEqual(high["status"], "MODEL_ERROR")
        self.assertEqual(high["reason"], "trial_failure_at_air_flow")
        self.assertEqual(high["trial_count"], 2)

    def test_explicit_flow_fpt_failure_keeps_partial_gas_generator(self):
        case = next(item for item in smoke.plan_cases()
                    if item["case"] == "C_GG3_FPT1")
        payload = copy.deepcopy(case["source_payload"])
        payload["air_mass_flow_kg_s"] = case["source_air_mass_flow_kg_s"] * 1.05
        expected_gas = payload["air_mass_flow_kg_s"] * (
            case["source_gas_mass_flow_kg_s"]
            / case["source_air_mass_flow_kg_s"])
        failure = NumericalConvergenceError(
            "test FPT stall", stage="unit.free_power",
            reason="unit_test_fpt_stall", last_valid_state={"y": 2.0})
        with mock.patch.object(
                ConstHubOrchestration, "solve_total_outlet_pressure",
                side_effect=failure):
            outcome = coupled.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(outcome.failure_stage, "unit.free_power")
        gas_generator = outcome.outputs["gas_generator"]
        self.assertEqual(len(gas_generator["closure"]["actual_stages"]), 3)
        self.assertNotIn("free_power", outcome.outputs)
        for stage in gas_generator["closure"]["actual_stages"]:
            self.assertEqual(stage["G_c"], expected_gas)
        source = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(payload["architecture_cycle_source"]))
        self.assertEqual(
            gas_generator["target_compressor_power_W"],
            float(source.compressor.specific_work)
            * payload["air_mass_flow_kg_s"])
        self.assertEqual(gas_generator["reference_0d_compressor_power_W"],
                         float(source.compressor.power))

    def test_synthetic_two_direction_run_continues_other_cases(self):
        temporary = tempfile.TemporaryDirectory(prefix="flow-synthetic-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        summary = smoke.run_plan(
            destination, evaluator=_synthetic_evaluator, constants={},
            model_digest="0" * 64, reproduce=False)
        statuses = {item["case"]: (item["status"], item["reason"])
                    for item in summary["cases"]}
        self.assertEqual(statuses["A_MAIN_3"],
                         ("NUMERICAL_FAILURE", "trial_failure_at_air_flow"))
        self.assertEqual(statuses["B_MAIN_3"][0], "SUCCESS")
        self.assertEqual(statuses["C_GG3_FPT1"][0], "SUCCESS")
        for name in ("B_MAIN_3", "C_GG3_FPT1"):
            root = next(item for item in summary["cases"]
                        if item["case"] == name)
            self.assertLessEqual(
                abs(root["root"]["relative_power_residual"]), 1e-6)
        self.assertEqual(summary["overall_engineering_feasibility"], None)
        self.assertEqual(
            summary["overall_engineering_feasibility_status"], "NOT_CHECKED")

    def test_interrupt_then_resume_reproduces_and_appends_only_missing(self):
        temporary = tempfile.TemporaryDirectory(prefix="flow-interrupt-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        calls = {"count": 0}

        def interrupting(payload):
            calls["count"] += 1
            if calls["count"] == 5:
                raise KeyboardInterrupt()
            return _synthetic_evaluator(payload)

        with self.assertRaises(KeyboardInterrupt):
            smoke.run_plan(destination, evaluator=interrupting, constants={},
                           model_digest="0" * 64, reproduce=False)
        incomplete = load_manifest(destination)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["record_count"], 4)
        self.assertFalse((destination / smoke.SUMMARY_NAME).exists())

        summary = smoke.run_plan(
            destination, resume=True, evaluator=_synthetic_evaluator,
            constants={}, model_digest="0" * 64, reproduce=False)
        complete = load_manifest(destination)
        self.assertTrue(complete["complete"])
        self.assertEqual(summary["run"]["records_appended_this_invocation"],
                         complete["record_count"] - 4)
        self.assertEqual(len(read_records(destination)), complete["record_count"])

    def test_strict_explicit_flow_values(self):
        base = smoke.plan_cases()[0]["source_payload"]
        for bad in (0, -1.0, float("inf"), float("nan"), True, "1"):
            payload = copy.deepcopy(base)
            payload["air_mass_flow_kg_s"] = bad
            with self.subTest(value=bad):
                with self.assertRaises((coupled.ArchitectureCoupledSmokeError,
                                        ValueError, TypeError)):
                    coupled._validate_payload(payload)
        payload = copy.deepcopy(base)
        payload["air_mass_flow_kg_s"] = 0.7
        point = coupled._validate_payload(payload)
        self.assertEqual(point.index, base["point_index"])

    def test_real_three_case_roots_and_summary(self):
        self.assertEqual(self.manifest["record_count"], len(self.records))
        self.assertTrue(self.summary["reproduce"]["matches"])
        self.assertEqual(self.summary["reproduce"]["checked"],
                         len(self.records))
        for name in ("A_MAIN_3", "B_MAIN_3", "C_GG3_FPT1"):
            case = self._case_summary(name)
            self.assertEqual(case["status"], "SUCCESS", name)
            self.assertEqual(case["reason"], "root_found", name)
            root = case["root"]
            self.assertLessEqual(abs(root["relative_power_residual"]), 1e-6)
            self.assertTrue(root["numerical_power_closed"])
            self.assertAlmostEqual(root["gas_mass_flow_kg_s"],
                                   root["air_mass_flow_kg_s"]
                                   + root["fuel_mass_flow_kg_s"], places=12)
            self.assertEqual(case["overall_engineering_feasibility"], None)
            self.assertEqual(
                case["overall_engineering_feasibility_status"], "NOT_CHECKED")
            self.assertGreaterEqual(case["trial_count"], 2)
            self.assertLessEqual(case["trial_count"], 2 + smoke.OUTER_MAX_ITER)
        self.assertEqual(self.summary["run"]["complete"], True)
        self.assertNotIn("actual_stages", json.dumps(self.summary["cases"]))

    def test_real_trials_use_sized_flow_and_actual_stage_geometry(self):
        by_id = {record.candidate_id: record for record in self.records}
        for record in self.records:
            payload = record.payload
            self.assertIn("air_mass_flow_kg_s", payload)
            if record.calculation_status is not CalculationStatus.SUCCESS:
                continue
            flow = record.outputs["flow_sizing"]
            gas = flow["trial_gas_mass_flow_kg_s"]
            self.assertEqual(flow["trial_air_mass_flow_kg_s"],
                             payload["air_mass_flow_kg_s"])
            self.assertEqual(flow["source_flow_role"],
                             "PRELIMINARY_0D_NOT_A_SIZED_DESIGN_FLOW")
            self.assertTrue(flow["alpha_unchanged"])
            self.assertAlmostEqual(flow["mass_residual_kg_s"], 0.0, places=12)
            boundary = record.outputs["actual_boundary"]
            if payload["coupling"] == "AB":
                stages = boundary["closure"]["actual_stages"]
                self.assertEqual(record.outputs["power"]["compressor_power_W"],
                                 flow["actual_compressor_power_target_W"])
            else:
                stages = boundary["gas_generator"]["closure"]["actual_stages"]
                self.assertTrue(
                    boundary["gas_generator_to_free_power_continuity"]
                    ["flow_matches_sized"])
            for stage in stages:
                self.assertEqual(stage["G_c"], gas)

        cases = smoke.plan_cases()
        fixed = coupled.evaluate_coupled_payload(
            copy.deepcopy(cases[2]["source_payload"]))
        self.assertIs(fixed.status, CalculationStatus.SUCCESS)
        root = self._case_summary("C_GG3_FPT1")["root"]
        sized = by_id[root["candidate_id"]]
        fixed_stage = fixed.outputs["actual_boundary"]["gas_generator"][
            "closure"]["actual_stages"][0]
        sized_stage = sized.outputs["actual_boundary"]["gas_generator"][
            "closure"]["actual_stages"][0]
        self.assertNotEqual(fixed_stage["G_c"], sized_stage["G_c"])
        self.assertTrue(
            fixed_stage.get("D_c") != sized_stage.get("D_c")
            or fixed_stage.get("l_c") != sized_stage.get("l_c")
            or fixed_stage.get("D_л") != sized_stage.get("D_л"))

    def test_strict_design_override_mathematical_validation(self):
        payload = copy.deepcopy(coupled.plan_payloads()[9])
        payload["stage_controls"]["design_overrides"] = _design_c()
        self.assertEqual(coupled._validate_payload(payload).index, 9)
        near = copy.deepcopy(payload)
        near["stage_controls"]["design_overrides"]["gas_generator"][
            "alpha1_deg"] = [1.0, 179.5]
        self.assertEqual(coupled._validate_payload(near).index, 9)
        ab = copy.deepcopy(coupled.plan_payloads()[2])
        ab["stage_controls"]["design_overrides"] = {
            "main": {"alpha1_deg": [1.0, 2.0, 3.0],
                     "kappa": [1.1, 1.2, 1.3],
                     "pi_weights": [0.8, 0.9, 1.0]}}
        self.assertEqual(coupled._validate_payload(ab).index, 2)

        def reject(mutator):
            candidate = copy.deepcopy(payload)
            mutator(candidate["stage_controls"]["design_overrides"])
            with self.assertRaises(coupled.ArchitectureCoupledSmokeError):
                coupled._validate_payload(candidate)

        reject(lambda d: d.pop("free_power"))
        reject(lambda d: d.update(main={"alpha1_deg": [11.0],
                                        "kappa": [1.38],
                                        "pi_weights": [1.0]}))
        reject(lambda d: d["gas_generator"].update(extra=1))
        reject(lambda d: d["gas_generator"].pop("kappa"))
        reject(lambda d: d["gas_generator"].update(alpha1_deg=[11.0]))
        reject(lambda d: d["gas_generator"].update(kappa=[1.38]))
        reject(lambda d: d["free_power"].update(pi_weights=[1.0, 1.0]))
        for key in ("alpha1_deg", "kappa", "pi_weights"):
            for bad in (True, "11", float("nan"), float("inf")):
                reject(lambda d, k=key, v=bad: d["gas_generator"][k].__setitem__(0, v))
        for bad in (0.0, -1.0, 180.0):
            reject(lambda d, v=bad: d["gas_generator"]["alpha1_deg"].__setitem__(0, v))
        for key in ("kappa", "pi_weights"):
            for bad in (0.0, -1.0):
                reject(lambda d, k=key, v=bad: d["gas_generator"][k].__setitem__(0, v))
        reject(lambda d: d["gas_generator"]["pi_weights"].__setitem__(-1, 0.99))
        reject(lambda d: d["free_power"]["pi_weights"].__setitem__(0, 0.5))

        bad = copy.deepcopy(payload)
        bad["stage_controls"]["design_overrides"] = []
        outcome = coupled.evaluate_coupled_payload(bad)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "payload_validation")
        self.assertEqual(
            outcome.last_valid_state["input"]["stage_controls"][
                "design_overrides"], [])

        bad = copy.deepcopy(payload)
        bad["stage_controls"]["design_overrides"].pop("free_power")
        outcome = coupled.evaluate_coupled_payload(bad)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "payload_validation")
        self.assertEqual(
            outcome.last_valid_state["input"]["stage_controls"][
                "design_overrides"],
            bad["stage_controls"]["design_overrides"])

    def test_real_c_design_override_reaches_actual_stages_and_stays_success(self):
        base = copy.deepcopy(coupled.plan_payloads()[9])
        self.assertNotIn("air_mass_flow_kg_s", base)
        payload = copy.deepcopy(base)
        payload["stage_controls"]["design_overrides"] = _design_c()
        payload["stage_controls"]["operating_points"] = _points_c()
        seen = []
        original = coupled.stage_smoke._build_namespace

        def spy(model, result, turbine, count, controls, p_static, *args,
                **kwargs):
            seen.append(copy.deepcopy(controls))
            return original(model, result, turbine, count, controls, p_static,
                            *args, **kwargs)

        with mock.patch.object(
                coupled.stage_smoke, "_build_namespace", side_effect=spy):
            outcome = coupled.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.SUCCESS)
        self.assertEqual(seen, [_design_c()["gas_generator"],
                                _design_c()["free_power"]])
        output = outcome.outputs
        design = output["design_controls"]
        self.assertEqual(design["roles"]["gas_generator"]["source"],
                         coupled.stage_smoke._stage_controls(2))
        self.assertEqual(design["roles"]["free_power"]["source"],
                         coupled.stage_smoke._stage_controls(1))
        self.assertEqual(design["roles"]["gas_generator"]["effective"],
                         _design_c()["gas_generator"])
        boundary = output["actual_boundary"]
        gg = boundary["gas_generator"]["closure"]["actual_stages"]
        fpt = boundary["free_power"]["closure"]["actual_stages"]
        self.assertEqual(boundary["gas_generator"]["closure"]["controls"]["n_rpm"],
                         72_000.0)
        self.assertEqual(boundary["free_power"]["closure"]["controls"]["D_hub_m"],
                         0.08)

        reference = coupled.evaluate_coupled_payload(base)
        self.assertIs(reference.status, CalculationStatus.SUCCESS)
        borrowed = reference.outputs["actual_boundary"]["gas_generator"][
            "closure"]["actual_stages"]
        self.assertNotEqual([stage["π_ст"] for stage in gg],
                            [stage["π_ст"] for stage in borrowed])
        self.assertNotEqual([stage["L_т"] for stage in gg],
                            [stage["L_т"] for stage in borrowed])
        self.assertNotEqual([(stage["D_c"], stage["l_c"]) for stage in gg],
                            [(stage["D_c"], stage["l_c"]) for stage in borrowed])
        continuity = boundary["gas_generator_to_free_power_continuity"]
        self.assertTrue(continuity["temperature_exact"])
        self.assertTrue(continuity["pressure_exact"])
        self.assertTrue(continuity["flow_exact"])
        self.assertEqual(continuity["free_power_first_stage_T0_K"],
                         gg[-1]["T_ст_п"])
        self.assertEqual(continuity["free_power_first_stage_p0_Pa"],
                         gg[-1]["p_ст_п"])
        self.assertEqual(fpt[0]["G_c"], gg[-1]["G_c"])
        power = output["power"]
        self.assertFalse(power["engine_power_closed"])
        self.assertEqual(power["engine_power_label"],
                         coupled.POWER_NOT_CLOSED_LABEL)
        self.assertLess(power["signed_power_residual_W"], 0.0)
        acceptance = output["posthoc_pi_acceptance"]["gas_generator"]
        self.assertFalse(acceptance["pass"])
        self.assertTrue(any(stage["status"] == "FAIL"
                            for stage in acceptance["stages"]))
        self.assertIsNone(output["overall_engineering_feasibility"])
        self.assertEqual(output["overall_engineering_feasibility_status"],
                         "NOT_CHECKED")

    def test_explicit_cases_design_hash_and_changed_design_resume_rejected(self):
        from turbine_engine.dataset import DatasetError
        case = _case_entry(9, "DESIGN_C", [0.5, 0.9], _points_c(), _design_c())
        self.assertEqual(len(smoke._validate_cases([case])), 1)
        temporary = tempfile.TemporaryDirectory(prefix="flow-design-hash-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        summary = smoke.run_plan(
            destination, cases=[case], evaluator=_case_evaluator, constants={},
            model_digest="0" * 64)
        self.assertTrue(summary["reproduce"]["matches"])
        item = load_manifest(destination)["run_config"]["cases"][0]
        self.assertEqual(item["design_overrides"], _design_c())
        self.assertEqual(item["operating_points"], _points_c())
        self.assertEqual(item["source_payload_sha256"],
                         case["source_payload_sha256"])
        self.assertEqual(item["stage_controls_sha256"],
                         coupled.canonical_hash(
                             case["source_payload"]["stage_controls"]))
        resumed = smoke.run_plan(
            destination, resume=True, cases=[case], evaluator=_case_evaluator,
            constants={}, model_digest="0" * 64, reproduce=False)
        self.assertEqual(
            resumed["run"]["records_appended_this_invocation"], 0)
        changed = copy.deepcopy(case)
        changed["source_payload"]["stage_controls"]["design_overrides"][
            "gas_generator"]["pi_weights"] = [0.9, 1.0]
        changed["source_payload_sha256"] = coupled.canonical_hash(
            changed["source_payload"])
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                destination, resume=True, cases=[changed],
                evaluator=_case_evaluator, constants={},
                model_digest="0" * 64, reproduce=False)

    def test_no_bracket_control_or_flow_rescaling_tuning(self):
        cases = {case["case"]: case for case in smoke.plan_cases()}
        for record in self.records:
            payload = record.payload
            source = cases[{
                2: "A_MAIN_3", 5: "B_MAIN_3", 12: "C_GG3_FPT1"
            }[payload["point_index"]]]
            self.assertEqual(payload["brackets"], source["source_payload"]["brackets"])
            self.assertEqual(payload["stage_controls"],
                             source["source_payload"]["stage_controls"])
            self.assertEqual(payload["stage_counts"],
                             source["source_payload"]["stage_counts"])
            self.assertEqual(payload["targets"],
                             source["source_payload"]["targets"])
            self.assertGreaterEqual(payload["air_mass_flow_kg_s"],
                                    source["air_flow_bracket_kg_s"][0])
            self.assertLessEqual(payload["air_mass_flow_kg_s"],
                                 source["air_flow_bracket_kg_s"][1])

    def test_strict_operating_point_roles_and_values(self):
        ab = copy.deepcopy(coupled.plan_payloads()[2])
        good_ab = copy.deepcopy(ab)
        good_ab["stage_controls"]["operating_points"] = {
            "main": {"speed_rpm": 90_000.0, "hub_diameter_m": 0.07}}
        self.assertEqual(coupled._validate_payload(good_ab).index, 2)
        c = copy.deepcopy(coupled.plan_payloads()[12])
        good_c = copy.deepcopy(c)
        good_c["stage_controls"]["operating_points"] = {
            "gas_generator": {"speed_rpm": 84_000.0, "hub_diameter_m": 0.065},
            "free_power": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065}}
        self.assertEqual(coupled._validate_payload(good_c).index, 12)

        bad_ab = [
            {"gas_generator": {"speed_rpm": 90_000.0, "hub_diameter_m": 0.07}},
            {},
            {"main": {"speed_rpm": 90_000.0}},
            {"main": {"speed_rpm": 90_000.0, "hub_diameter_m": 0.07,
                      "extra": 1}},
            {"main": {"speed_rpm": 0.0, "hub_diameter_m": 0.07}},
            {"main": {"speed_rpm": float("inf"), "hub_diameter_m": 0.07}},
            {"main": {"speed_rpm": True, "hub_diameter_m": 0.07}},
            {"main": {"speed_rpm": 90_000.0, "hub_diameter_m": -0.07}},
            {"main": {"speed_rpm": "90000", "hub_diameter_m": 0.07}},
            [],
        ]
        for points in bad_ab:
            payload = copy.deepcopy(ab)
            payload["stage_controls"]["operating_points"] = points
            with self.subTest(points=points):
                with self.assertRaises(coupled.ArchitectureCoupledSmokeError):
                    coupled._validate_payload(payload)
        bad_c = [
            {"gas_generator": {"speed_rpm": 84_000.0, "hub_diameter_m": 0.065}},
            {"gas_generator": {"speed_rpm": 84_000.0, "hub_diameter_m": 0.065},
             "free_power": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065},
             "main": {"speed_rpm": 84_000.0, "hub_diameter_m": 0.065}},
            {"gas_generator": {"speed_rpm": float("nan"), "hub_diameter_m": 0.065},
             "free_power": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065}},
        ]
        for points in bad_c:
            payload = copy.deepcopy(c)
            payload["stage_controls"]["operating_points"] = points
            with self.subTest(points=points):
                with self.assertRaises(coupled.ArchitectureCoupledSmokeError):
                    coupled._validate_payload(payload)
        payload = copy.deepcopy(ab)
        payload["stage_controls"]["operating_points"] = {
            "main": {"speed_rpm": 0.0, "hub_diameter_m": 0.07}}
        outcome = coupled.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_stage, "payload_validation")

    def test_ab_operating_point_is_passed_to_the_actual_closure(self):
        payload = copy.deepcopy(coupled.plan_payloads()[2])
        payload["stage_controls"]["operating_points"] = {
            "main": {"speed_rpm": 90_000.0, "hub_diameter_m": 0.07}}
        calls = []

        def spy(self, n, D_hub, target, **kwargs):
            calls.append((n, D_hub))
            raise NumericalConvergenceError(
                "unit mock", stage="unit.operating", reason="unit_mock",
                last_valid_state={})

        with mock.patch.object(
                ConstHubOrchestration, "solve_total_outlet_pressure", new=spy):
            outcome = coupled.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(calls, [(90_000.0, 0.07)])
        self.assertEqual(payload["stage_controls"]["speed_rpm"], 84_000.0)
        self.assertEqual(payload["stage_controls"]["hub_diameter_m"], 0.065)

    def test_c_operating_points_use_independent_gg_and_fpt_speeds(self):
        reference = self._case_summary("C_GG3_FPT1")["root"]
        reference_record = next(record for record in self.records
                                if record.candidate_id == reference["candidate_id"])
        payload = copy.deepcopy(reference_record.payload)
        payload["stage_controls"]["operating_points"] = {
            "gas_generator": {"speed_rpm": 84_000.0, "hub_diameter_m": 0.065},
            "free_power": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065}}
        outcome = coupled.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.SUCCESS)
        output = outcome.outputs
        boundary = output["actual_boundary"]
        gg_closure = boundary["gas_generator"]["closure"]
        fpt_closure = boundary["free_power"]["closure"]
        self.assertEqual(gg_closure["controls"]["n_rpm"], 84_000.0)
        self.assertEqual(gg_closure["controls"]["D_hub_m"], 0.065)
        self.assertEqual(fpt_closure["controls"]["n_rpm"], 72_000.0)
        self.assertEqual(fpt_closure["controls"]["D_hub_m"], 0.065)
        operating = output["operating_points"]
        self.assertEqual(set(operating), {"gas_generator", "free_power"})
        for role, closure in (("gas_generator", gg_closure),
                              ("free_power", fpt_closure)):
            self.assertEqual(operating[role]["source_speed_rpm"], 84_000.0)
            self.assertEqual(operating[role]["source_hub_diameter_m"], 0.065)
            self.assertEqual(operating[role]["effective_speed_rpm"],
                             closure["controls"]["n_rpm"])
            self.assertEqual(operating[role]["effective_hub_diameter_m"],
                             closure["controls"]["D_hub_m"])
            self.assertEqual(operating[role]["compressor_speed_compatibility"],
                             "NOT_CHECKED")
            self.assertEqual(operating[role]["implicit_powertrain_map"],
                             "NOT_APPLIED")
        self.assertEqual(operating["free_power"]["effective_speed_rpm"],
                         72_000.0)
        continuity = boundary["gas_generator_to_free_power_continuity"]
        self.assertTrue(continuity["temperature_exact"])
        self.assertTrue(continuity["pressure_exact"])
        self.assertTrue(continuity["flow_exact"])
        reference_fpt = reference_record.outputs["actual_boundary"][
            "free_power"]["closure"]["actual_stages"][0]
        controlled_fpt = fpt_closure["actual_stages"][0]
        self.assertNotEqual(controlled_fpt["D_c"], reference_fpt["D_c"])
        self.assertNotEqual(controlled_fpt["l_c"], reference_fpt["l_c"])
        self.assertNotEqual(output["power"]["net_mechanical_power_W"],
                            reference_record.outputs["power"][
                                "net_mechanical_power_W"])

    def test_default_records_have_no_operating_point_metadata(self):
        for record in self.records:
            self.assertNotIn("operating_points",
                             record.payload["stage_controls"])
            if record.calculation_status is CalculationStatus.SUCCESS:
                self.assertNotIn("operating_points", record.outputs)
        defaults = coupled.plan_payloads()[12]
        self.assertIsNone(coupled._operating_output({
            "gas_generator": coupled._operating(defaults, "gas_generator"),
            "free_power": coupled._operating(defaults, "free_power")}))

    def test_explicit_cases_api_dynamic_count_reproduce_and_resume(self):
        cases = [
            _case_entry(2, "CASE_A", [0.5, 0.9],
                        {"main": {"speed_rpm": 90_000.0,
                                  "hub_diameter_m": 0.07}}),
            _case_entry(5, "CASE_B", [0.5, 0.9],
                        {"main": {"speed_rpm": 86_000.0,
                                  "hub_diameter_m": 0.066}})]
        self.assertEqual(len(smoke._validate_cases(cases)), 2)
        temporary = tempfile.TemporaryDirectory(prefix="flow-cases-api-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        summary = smoke.run_plan(
            destination, cases=cases, evaluator=_case_evaluator,
            constants={}, model_digest="0" * 64)
        self.assertEqual(summary["experiment"]["case_count"], 2)
        self.assertEqual([item["case"] for item in summary["cases"]],
                         ["CASE_A", "CASE_B"])
        self.assertTrue(all(item["status"] == "SUCCESS"
                            for item in summary["cases"]))
        self.assertTrue(summary["reproduce"]["matches"])
        self.assertEqual(summary["reproduce"]["checked"],
                         len(read_records(destination)))
        config = load_manifest(destination)["run_config"]
        self.assertEqual(config["case_count"], 2)
        self.assertEqual(len(config["cases"]), 2)
        self.assertEqual(config["cases"][0]["operating_points"]["main"]
                         ["speed_rpm"], 90_000.0)
        self.assertEqual(len(config["cases"][0]["stage_controls_sha256"]), 64)

        before = load_manifest(destination)
        resumed = smoke.run_plan(
            destination, resume=True, cases=cases, evaluator=_case_evaluator,
            constants={}, model_digest="0" * 64, reproduce=False)
        after = load_manifest(destination)
        self.assertEqual(resumed["run"]["records_appended_this_invocation"], 0)
        self.assertEqual(after["record_count"], before["record_count"])
        self.assertEqual(after["journal_sha256"], before["journal_sha256"])

        changed = copy.deepcopy(cases)
        changed[0]["source_payload"]["stage_controls"]["operating_points"][
            "main"]["speed_rpm"] = 91_000.0
        changed[0]["source_payload_sha256"] = coupled.canonical_hash(
            changed[0]["source_payload"])
        from turbine_engine.dataset import DatasetError
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                destination, resume=True, cases=changed,
                evaluator=_case_evaluator, constants={},
                model_digest="0" * 64, reproduce=False)

    def test_explicit_cases_api_rejects_inconsistent_plans(self):
        base = _case_entry(2, "CASE_A", [0.5, 0.9])
        valid = copy.deepcopy(base)

        def bad(mutator):
            candidate = copy.deepcopy(valid)
            mutator(candidate)
            with self.assertRaises(smoke.FlowSizingSmokeError):
                smoke._validate_cases([candidate])

        bad(lambda c: c.update(case=""))
        bad(lambda c: c.update(source_point_index=3))
        bad(lambda c: c.update(source_payload_sha256="0" * 64))
        bad(lambda c: c.update(architecture="SINGLE_SHAFT_SINGLE_CC"))
        bad(lambda c: c.update(assembly="B_MAIN"))
        bad(lambda c: c.update(source_air_mass_flow_kg_s=(
            base["source_air_mass_flow_kg_s"] * 0.5)))
        bad(lambda c: c.update(target_gearbox_input_power_W=100_000.0))
        bad(lambda c: c.update(air_flow_bracket_kg_s=[0.9, 0.5]))
        bad(lambda c: c.update(air_flow_bracket_kg_s=[0.0, 0.9]))
        bad(lambda c: c.update(outer_relative_tolerance=1e-5))
        bad(lambda c: c.update(outer_max_iterations=31))

        def bad_payload(mutator):
            candidate = copy.deepcopy(valid)
            mutator(candidate["source_payload"])
            candidate["source_payload_sha256"] = coupled.canonical_hash(
                candidate["source_payload"])
            with self.assertRaises(smoke.FlowSizingSmokeError):
                smoke._validate_cases([candidate])

        bad_payload(lambda p: p["stage_controls"].update(
            operating_points={"gas_generator": {"speed_rpm": 84_000.0,
                                                "hub_diameter_m": 0.065}}))
        bad_payload(lambda p: p["stage_controls"].pop("main"))
        with self.assertRaises(smoke.FlowSizingSmokeError):
            smoke._validate_cases([base, copy.deepcopy(base)])

    def test_explicit_cases_require_full_base_payload(self):
        valid_ab = _case_entry(2, "VALID_AB", [0.5, 0.9])
        valid_c = _case_entry(9, "VALID_GG2", [0.5, 0.9])
        self.assertEqual(len(smoke._validate_cases([valid_ab, valid_c])), 2)

        def reject(case, mutator):
            candidate = copy.deepcopy(case)
            mutator(candidate["source_payload"])
            candidate["source_payload_sha256"] = coupled.canonical_hash(
                candidate["source_payload"])
            with self.assertRaises(smoke.FlowSizingSmokeError):
                smoke._validate_cases([candidate])

        reject(valid_ab, lambda p: p.update(assembly="B_MAIN"))
        reject(valid_ab, lambda p: p.update(turbine_role="free_power"))
        reject(valid_ab, lambda p: p["targets"].update(
            target_gearbox_input_power_W=120_000.0))
        reject(valid_ab, lambda p: p.update(schema_version="other"))
        reject(valid_ab, lambda p: p.update(point_index=3))

        def gg3(payload):
            payload["stage_counts"]["gas_generator"] = 3
            payload["stage_controls"]["gas_generator"] = (
                coupled.stage_smoke._stage_controls(3))

        reject(valid_c, gg3)

    def test_pressure_ratio_helper_recomputes_c_source_and_gg_bracket(self):
        base = copy.deepcopy(coupled.plan_payloads()[9])
        same = coupled.with_compressor_pressure_ratios(
            base, base["architecture_cycle_source"]["compressor_pressure_ratios"])
        self.assertEqual(same, base)
        self.assertEqual(coupled.canonical_hash(same), coupled.canonical_hash(base))
        self.assertIsNot(same, base)
        variant = coupled.with_compressor_pressure_ratios(base, [11.0])
        self.assertEqual(base, coupled.plan_payloads()[9])
        self.assertEqual(variant["architecture_cycle_source"]
                         ["compressor_pressure_ratios"], [11.0])
        for key in ("point_index", "assembly", "coupling", "architecture",
                    "turbine_role", "stage_counts", "stage_controls"):
            self.assertEqual(variant[key], base[key])
        self.assertEqual(variant["targets"]["target_gearbox_input_power_W"],
                         smoke.TARGET)
        source = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(variant["architecture_cycle_source"]))
        reference = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(base["architecture_cycle_source"]))
        self.assertEqual(variant["targets"]["gas_generator_compressor_power_W"],
                         float(source.compressor.power))
        self.assertNotEqual(float(source.compressor.power),
                            float(reference.compressor.power))
        self.assertEqual(variant["targets"]["free_power_outlet_total_pressure_Pa"],
                         float(source.turbines[1].outlet.p_total))
        self.assertEqual(variant["targets"]["free_power_outlet_total_pressure_Pa"],
                         base["targets"]["free_power_outlet_total_pressure_Pa"])
        static = coupled._preliminary_static(source.turbines[0].outlet.p_total,
                                             source.turbines[0].outlet.k)
        self.assertEqual(variant["brackets"]["gas_generator"],
                         [0.8 * static, 1.2 * static])
        self.assertNotEqual(variant["brackets"]["gas_generator"],
                            base["brackets"]["gas_generator"])
        self.assertEqual(variant["brackets"]["free_power"],
                         base["brackets"]["free_power"])
        self.assertIn("pressure-ratio variant",
                      variant["brackets"]["provenance"])
        self.assertNotEqual(float(source.fuel_mass_flow),
                            float(reference.fuel_mass_flow))
        self.assertNotEqual(float(source.combustor.alpha),
                            float(reference.combustor.alpha))
        self.assertNotEqual(float(source.turbines[0].outlet.p_total),
                            float(reference.turbines[0].outlet.p_total))

    def test_pressure_ratio_helper_dual_log_split_and_strict_validation(self):
        base = copy.deepcopy(coupled.plan_payloads()[0])
        ratios = split_pressure_ratio_log(11.0, [0.4, 0.6])
        self.assertAlmostEqual(math.prod(ratios), 11.0, places=12)
        variant = coupled.with_compressor_pressure_ratios(base, ratios)
        self.assertEqual(variant["architecture_cycle_source"]
                         ["compressor_pressure_ratios"], list(ratios))
        source = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(variant["architecture_cycle_source"]))
        self.assertEqual(variant["targets"]["main_compressor_power_W"],
                         float(source.compressor.power))
        self.assertEqual(variant["brackets"]["main"], [120_000.0, 150_000.0])
        self.assertIn("pressure-ratio variant",
                      variant["brackets"]["provenance"])
        for bad in ([11.0], [True], ["11"], [1.0], [float("nan")],
                    [float("inf")], 11.0, "11", []):
            with self.subTest(ratios=bad):
                with self.assertRaises(coupled.ArchitectureCoupledSmokeError):
                    coupled.with_compressor_pressure_ratios(base, bad)
        single = coupled.plan_payloads()[9]
        with self.assertRaises(coupled.ArchitectureCoupledSmokeError):
            coupled.with_compressor_pressure_ratios(single, [11.0, 11.0])
        for ratio in (3.0, 13.0):
            accepted = coupled.with_compressor_pressure_ratios(single, [ratio])
            self.assertEqual(accepted["architecture_cycle_source"]
                             ["compressor_pressure_ratios"], [ratio])

    def test_explicit_variant_case_uses_actual_source_flows_and_rejects_edits(self):
        case = _case_entry(9, "C_GG2_PI11", [0.5, 0.9], _points_c(),
                           _design_c(), ratios=[11.0])
        self.assertEqual(len(smoke._validate_cases([case])), 1)
        payload = case["source_payload"]
        self.assertEqual(payload["architecture_cycle_source"]
                         ["compressor_pressure_ratios"], [11.0])
        source = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(payload["architecture_cycle_source"]))
        self.assertEqual(case["source_air_mass_flow_kg_s"],
                         float(source.air_mass_flow))
        self.assertEqual(case["source_gas_mass_flow_kg_s"],
                         float(source.gas_mass_flow))
        self.assertEqual(case["source_fuel_mass_flow_kg_s"],
                         float(source.fuel_mass_flow))
        temporary = tempfile.TemporaryDirectory(prefix="flow-variant-case-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        summary = smoke.run_plan(
            destination, cases=[case], evaluator=_case_evaluator, constants={},
            model_digest="0" * 64)
        self.assertTrue(summary["reproduce"]["matches"])
        item = load_manifest(destination)["run_config"]["cases"][0]
        self.assertEqual(item["compressor_pressure_ratios"], [11.0])
        self.assertEqual(item["compressor_pressure_ratio_product"], 11.0)
        self.assertEqual(item["source_cycle_pressure_provenance"],
                         "EXPLICIT_SOURCE_COMPRESSOR_PRESSURE_RATIOS")

        def reject(mutator):
            candidate = copy.deepcopy(case)
            mutator(candidate["source_payload"])
            candidate["source_payload_sha256"] = coupled.canonical_hash(
                candidate["source_payload"])
            with self.assertRaises(smoke.FlowSizingSmokeError):
                smoke._validate_cases([candidate])

        reject(lambda p: p["architecture_cycle_source"]["combustor"].update(
            outlet_total_temperature=1251.0))
        reject(lambda p: p["architecture_cycle_source"].update(
            compressor_polytropic_efficiency=0.889))
        reject(lambda p: p["architecture_cycle_source"]["gas_model"].update(
            R=288.0))
        reject(lambda p: p["architecture_cycle_source"]["combustor"].update(
            total_pressure_recovery=0.979))
        reject(lambda p: p["targets"].update(
            gas_generator_compressor_power_W=(
                p["targets"]["gas_generator_compressor_power_W"] * 1.001)))
        reject(lambda p: p["stage_controls"].update(speed_rpm=85_000.0))

        def reject_ratios(bad):
            candidate = copy.deepcopy(case)
            candidate["source_payload"]["architecture_cycle_source"][
                "compressor_pressure_ratios"] = bad
            candidate["source_payload_sha256"] = coupled.canonical_hash(
                candidate["source_payload"])
            with self.assertRaises(smoke.FlowSizingSmokeError):
                smoke._validate_cases([candidate])

        for bad in ([11.0, 11.0], [True], [1.0], ["11"]):
            with self.subTest(ratios=bad):
                reject_ratios(bad)

    def test_synthetic_resume_changed_pressure_variant_is_rejected(self):
        first = _case_entry(9, "PI11", [0.5, 0.9], _points_c(),
                            ratios=[11.0])
        self.assertEqual(len(smoke._validate_cases([first])), 1)
        temporary = tempfile.TemporaryDirectory(prefix="flow-pressure-resume-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        smoke.run_plan(destination, cases=[first], evaluator=_case_evaluator,
                       constants={}, model_digest="0" * 64, reproduce=False)
        before = load_manifest(destination)
        resumed = smoke.run_plan(
            destination, resume=True, cases=[first], evaluator=_case_evaluator,
            constants={}, model_digest="0" * 64, reproduce=False)
        after = load_manifest(destination)
        self.assertEqual(resumed["run"]["records_appended_this_invocation"], 0)
        self.assertEqual(after["journal_sha256"], before["journal_sha256"])
        changed = _case_entry(9, "PI12", [0.5, 0.9], _points_c(),
                              ratios=[12.0])
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                destination, resume=True, cases=[changed],
                evaluator=_case_evaluator, constants={},
                model_digest="0" * 64, reproduce=False)

    def test_real_c_gg2_fpt1_pi11_balanced_trial(self):
        base = copy.deepcopy(coupled.plan_payloads()[9])
        payload = coupled.with_compressor_pressure_ratios(base, [11.0])
        payload["stage_controls"]["design_overrides"] = {
            "gas_generator": {
                "alpha1_deg": [11.0, 11.0], "kappa": [1.38, 1.12],
                "pi_weights": [1.0, 1.0]},
            "free_power": {
                "alpha1_deg": [11.0], "kappa": [1.38],
                "pi_weights": [1.0]}}
        payload["stage_controls"]["operating_points"] = _points_c()
        self.assertEqual(coupled._validate_payload(payload).index, 9)
        source = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(payload["architecture_cycle_source"]))
        reference = coupled.calculate_architecture_cycle(
            coupled.payload_to_input(base["architecture_cycle_source"]))
        outcome = coupled.evaluate_coupled_payload(payload)
        if outcome.status is not CalculationStatus.SUCCESS:
            self.assertIn(outcome.status, (CalculationStatus.NON_PHYSICAL_STATE,
                                           CalculationStatus.NUMERICAL_FAILURE))
            self.assertIsNotNone(outcome.failure_stage)
            return
        outputs = outcome.outputs
        boundary = outputs["actual_boundary"]
        gg, fpt = boundary["gas_generator"], boundary["free_power"]
        self.assertEqual(gg["target_compressor_power_W"],
                         float(source.compressor.power))
        self.assertNotEqual(gg["target_compressor_power_W"],
                            float(reference.compressor.power))
        gg_stages = gg["closure"]["actual_stages"]
        fpt_stages = fpt["closure"]["actual_stages"]
        self.assertEqual(gg_stages[0]["G_c"], float(source.gas_mass_flow))
        self.assertNotEqual(gg_stages[0]["G_c"],
                            float(reference.gas_mass_flow))
        self.assertEqual(fpt_stages[0]["G_c"], gg_stages[-1]["G_c"])
        continuity = boundary["gas_generator_to_free_power_continuity"]
        self.assertTrue(continuity["temperature_exact"])
        self.assertTrue(continuity["pressure_exact"])
        self.assertTrue(continuity["flow_exact"])
        self.assertEqual(fpt_stages[0]["T0"], gg_stages[-1]["T_ст_п"])
        self.assertEqual(fpt_stages[0]["p0"], gg_stages[-1]["p_ст_п"])
        self.assertNotEqual(outputs["power"]["net_mechanical_power_W"],
                            smoke.TARGET)

    def test_duplicate_physical_source_payloads_rejected_before_writer(self):
        base = _case_entry(9, "DUPLICATE_BASE", [0.5, 0.9])
        first = copy.deepcopy(base)
        first["case"] = "DUPLICATE_1"
        second = copy.deepcopy(base)
        second["case"] = "DUPLICATE_2"
        with self.assertRaises(smoke.FlowSizingSmokeError):
            smoke._validate_cases([first, second])
        changed_bracket = copy.deepcopy(base)
        changed_bracket["case"] = "DUPLICATE_3"
        changed_bracket["air_flow_bracket_kg_s"] = [0.6, 1.0]
        with self.assertRaises(smoke.FlowSizingSmokeError):
            smoke._validate_cases([first, changed_bracket])

        low = {"gas_generator": {"speed_rpm": 84_000.0,
                                 "hub_diameter_m": 0.065},
               "free_power": {"speed_rpm": 72_000.0, "hub_diameter_m": 0.065}}
        high = copy.deepcopy(low)
        high["free_power"]["speed_rpm"] = 70_000.0
        distinct = [_case_entry(9, "INDEPENDENT_1", [0.5, 0.9], low),
                    _case_entry(9, "INDEPENDENT_2", [0.5, 0.9], high)]
        self.assertEqual(len(smoke._validate_cases(distinct)), 2)
        identifiers = [make_candidate_id(
            case["architecture"], smoke.MODEL_VARIANT,
            {**case["source_payload"], "air_mass_flow_kg_s": 0.7})
            for case in distinct]
        self.assertEqual(len(set(identifiers)), 2)

        temporary = tempfile.TemporaryDirectory(prefix="flow-duplicate-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        with self.assertRaises(smoke.FlowSizingSmokeError):
            smoke.run_plan(
                destination, cases=[first, second],
                evaluator=_case_evaluator, constants={},
                model_digest="0" * 64)
        self.assertFalse(destination.exists())

    def test_resume_appends_zero_and_preserves_journal(self):
        before = load_manifest(self.destination)
        summary = smoke.run_plan(
            self.destination, resume=True, reproduce=False,
            constants=self.constants, model_digest=self.model_digest)
        after = load_manifest(self.destination)
        self.assertEqual(summary["run"]["records_appended_this_invocation"], 0)
        self.assertEqual(after["record_count"], before["record_count"])
        self.assertEqual(after["journal_sha256"], before["journal_sha256"])
        self.assertTrue(after["complete"])

    def test_changed_digest_is_rejected_on_resume(self):
        from turbine_engine.dataset import DatasetError
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                self.destination, resume=True, reproduce=False,
                constants=self.constants, model_digest="0" * 64)


if __name__ == "__main__":
    unittest.main()
