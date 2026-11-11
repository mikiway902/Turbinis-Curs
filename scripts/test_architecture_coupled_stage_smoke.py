"""Focused tests for the bounded actual-boundary A/B/C stage-coupling smoke."""

from __future__ import annotations

import copy
import json
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

import architecture_coupled_stage_smoke as smoke  # noqa: E402
import architecture_flow_sizing_smoke as flow_smoke  # noqa: E402
import architecture_stage_smoke as stage_smoke  # noqa: E402
from turbine_engine.dataset import (  # noqa: E402
    DatasetError, compare_candidates, load_manifest, read_records)
from turbine_engine.diagnostics import NumericalConvergenceError  # noqa: E402
from turbine_engine.orchestration import ConstHubOrchestration  # noqa: E402
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus, EvaluationOutcome, evaluate_candidate)


def _fake_success_outputs() -> dict:
    return {
        "power": {
            "net_mechanical_power_W": 1.0,
            "target_gearbox_input_power_W": smoke.TARGET,
            "signed_power_residual_W": 1.0 - smoke.TARGET,
            "relative_power_residual": 1.0,
            "engine_power_closed": False,
            "engine_power_label": smoke.POWER_NOT_CLOSED_LABEL,
        },
        "posthoc_pi_acceptance": {
            "main": {"profile": stage_smoke.PI_ACCEPTANCE_PROFILE, "pass": None}
        },
        "overall_engineering_feasibility": None,
        "overall_engineering_feasibility_status": "NOT_CHECKED",
        "overall_engineering_feasibility_reason": "strength_verification_missing",
    }


PI4_A3_AIR = 1.1069144083442255
PI4_A3_CANDIDATE = (
    "cand_8557c187058710328e826efefbfaa3766603400f4cc23bfaa80fb980626068b5")
C2_CANDIDATE = (
    "cand_282a1d7353cce3a6f894d652e2f7e351846d211e12616f52974063a8c88f190b")
C2_AIR = 0.6892386929451052
REQUIRE_DOMAIN = smoke.REACTION_TABLE_POLICY_REQUIRE_FINAL_SOURCE_DOMAIN
LEGACY_CLIPPED = smoke.REACTION_TABLE_POLICY_LEGACY_CLIPPED


def _saved_payload(relative: str, candidate_id: str):
    """Return an optional retained payload without making the fixture required."""
    path = ROOT / relative / "dataset.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("candidate_id") == candidate_id:
                return record["payload"]
    return None


def _pi4_a3_payload() -> dict:
    """Real source-limited A3 point: pi=4 split, 78k/0.07, explicit air."""
    payload = smoke.with_compressor_pressure_ratios(
        smoke.plan_payloads()[2], [2.0, 2.0])
    payload["stage_controls"]["operating_points"] = {
        "main": {"speed_rpm": 78_000.0, "hub_diameter_m": 0.07}}
    payload["air_mass_flow_kg_s"] = PI4_A3_AIR
    return payload


def _c2_payload() -> dict:
    """Retained C GG2/FPT1 refinement point with all final l/D inside."""
    payload = smoke.with_compressor_pressure_ratios(
        smoke.plan_payloads()[9], [10.0])
    payload["stage_controls"]["design_overrides"] = {
        "gas_generator": {
            "alpha1_deg": [11.0, 11.0], "kappa": [1.38, 1.12],
            "pi_weights": [1.0, 1.0]},
        "free_power": {
            "alpha1_deg": [11.0], "kappa": [1.38], "pi_weights": [1.0]}}
    payload["stage_controls"]["operating_points"] = {
        "gas_generator": {"speed_rpm": 76_000.0, "hub_diameter_m": 0.073},
        "free_power": {"speed_rpm": 61_000.0, "hub_diameter_m": 0.08}}
    payload["air_mass_flow_kg_s"] = C2_AIR
    return payload


def _with_policy(payload: dict) -> dict:
    changed = copy.deepcopy(payload)
    changed["stage_controls"]["reaction_table_policy"] = REQUIRE_DOMAIN
    return changed


class ArchitectureCoupledStageSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="architecture-coupled-smoke-")
        cls.destination = Path(cls.temporary.name) / "dataset"
        cls.summary = smoke.run_plan(cls.destination)
        cls.records = read_records(cls.destination)
        cls.manifest = load_manifest(cls.destination)
        cls.constants = cls.manifest["run_config"]["stage_constants"]
        cls.model_digest = cls.manifest["run_config"]["model_digest"]

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temporary.name, ignore_errors=True)

    def _record(self, index: int):
        for record in self.records:
            if record.payload["point_index"] == index:
                return record
        self.fail(f"missing record {index}")

    def _table_bounds(self):
        return smoke._reaction_table_bounds(
            {"ρ_таблица": self.constants["reaction_table"]})

    def test_fixed_plan_has_fifteen_unique_explicit_points(self):
        points, payloads = smoke.plan_points(), smoke.plan_payloads()
        self.assertEqual(len(points), 15)
        self.assertEqual(len(payloads), 15)
        self.assertEqual([point.index for point in points], list(range(15)))
        ab = [point for point in points if point.coupling == "AB"]
        self.assertEqual(
            [(point.assembly, point.main_stage_count) for point in ab],
            [("A_MAIN", 1), ("A_MAIN", 2), ("A_MAIN", 3),
             ("B_MAIN", 1), ("B_MAIN", 2), ("B_MAIN", 3)])
        pairs = [(point.gas_generator_stage_count, point.free_power_stage_count)
                 for point in points if point.coupling == "C"]
        self.assertEqual(pairs, [(g, f) for g in (1, 2, 3) for f in (1, 2, 3)])
        identifiers = [record.candidate_id for record in self.records]
        self.assertEqual(len(identifiers), 15)
        self.assertEqual(len(set(identifiers)), 15)
        smoke._validate_plan(payloads)
        for payload in payloads:
            controls = payload["stage_controls"]
            self.assertEqual(controls["speed_rpm"], 84_000.0)
            self.assertEqual(controls["hub_diameter_m"], 0.065)
            self.assertEqual(controls["reference_exit_mach"], 0.3379)
            self.assertEqual(controls["pressure_ratio_policy"], "raw_unbounded")
            self.assertEqual(payload["targets"]["target_gearbox_input_power_W"],
                             110_000.0)
            self.assertTrue(payload["brackets"]["provenance"])
            if payload["coupling"] == "AB":
                count = payload["stage_counts"]["main"]
                for key in ("alpha1_deg", "kappa", "pi_weights"):
                    self.assertEqual(controls["main"][key],
                                     stage_smoke._stage_controls(count)[key])
                self.assertEqual(payload["brackets"]["main"],
                                 [120_000.0, 150_000.0])
            else:
                for role in ("gas_generator", "free_power"):
                    count = payload["stage_counts"][role]
                    for key in ("alpha1_deg", "kappa", "pi_weights"):
                        self.assertEqual(
                            controls[role][key],
                            stage_smoke._stage_controls(count)[key])
                self.assertEqual(payload["brackets"]["free_power"],
                                 [120_000.0, 150_000.0])

    def test_manifest_config_records_plan_constants_and_model_digest(self):
        config = self.manifest["run_config"]
        self.assertEqual(config["payload_plan_count"], 15)
        self.assertEqual(config["pressure_ratio_policy"], "raw_unbounded")
        self.assertEqual(config["stage_count_automatic_selection"], "NONE")
        self.assertEqual(config["coupling_label"], smoke.COUPLING_LABEL)
        self.assertEqual(config["power_closure_relative_tolerance"], 1e-6)
        for key in ("payload_plan_sha256", "source_stage_fixture_sha256",
                    "model_digest"):
            self.assertEqual(len(config[key]), 64)
            int(config[key], 16)
        sources = config["model_digest_sources"]
        self.assertIn(smoke.NEW_SMOKE_FILE, sources)
        self.assertNotIn("scripts/test_architecture_coupled_stage_smoke.py", sources)
        self.assertNotIn("test_digest", config)
        for relative in ("turbine_engine/orchestration.py",
                         "scripts/architecture_stage_smoke.py"):
            self.assertIn(relative, sources)
        self.assertEqual(config["stage_constants"],
                         stage_smoke._stage_constants(
                             smoke.oe.build_ns(
                                 smoke._get_model(),
                                 gas_property_basis=smoke.TEXTBOOK,
                                 cycle_gas_property_basis=smoke.TEXTBOOK)[0]))
        self.assertTrue(self.manifest["complete"])
        self.assertEqual(self.manifest["record_count"], 15)

    def test_all_attempts_stored_with_typed_failures_no_implicit_selection(self):
        self.assertEqual(self.summary["status_counts"],
                         {"NUMERICAL_FAILURE": 2, "SUCCESS": 13})
        by_coupling = {
            coupling: [record.calculation_status.value for record in self.records
                       if record.payload["coupling"] == coupling]
            for coupling in ("AB", "C")}
        self.assertEqual(by_coupling["AB"].count("SUCCESS"), 4)
        self.assertEqual(by_coupling["AB"].count("NUMERICAL_FAILURE"), 2)
        self.assertEqual(by_coupling["C"].count("SUCCESS"), 9)
        for index in (0, 3):
            record = self._record(index)
            self.assertIs(record.calculation_status,
                          CalculationStatus.NUMERICAL_FAILURE)
            self.assertEqual(record.failure_stage,
                             "orchestration.solve_total_outlet_pressure")
            self.assertEqual(record.failure_reason, "boundary_not_bracketed")
            self.assertEqual(
                record.last_valid_state["solver_last_valid_state"][
                    "evaluated_static_pressure_Pa"], [120_000.0, 150_000.0])
            self.assertEqual(record.payload["brackets"]["main"],
                             [120_000.0, 150_000.0])
        for record in self.records:
            payload = record.payload
            self.assertNotIn("stage_count_selection", payload)
            if record.calculation_status is CalculationStatus.SUCCESS:
                self.assertEqual(record.outputs["stage_count_selection"], "NONE")
                self.assertEqual(record.outputs["pressure_ratio_policy"],
                                 "raw_unbounded")
            json.dumps(record.outputs, ensure_ascii=False, allow_nan=False)

    def test_actual_ab_pressure_closure_and_source_power_subtraction(self):
        ab_successes = [record for record in self.records
                        if record.payload["coupling"] == "AB"
                        and record.calculation_status is CalculationStatus.SUCCESS]
        self.assertEqual(len(ab_successes), 4)
        for record in ab_successes:
            boundary = record.outputs["actual_boundary"]
            self.assertEqual(boundary["target_kind"], "total_outlet_pressure")
            self.assertEqual(
                boundary["target_outlet_total_pressure_Pa"], 150_000.0)
            self.assertEqual(boundary["closure"]["actual_metric"],
                             boundary["actual_outlet_total_pressure_Pa"])
            self.assertGreaterEqual(len(boundary["closure"]["actual_stages"]), 1)
            self.assertEqual(
                boundary["closure"]["controls"]["static_pressure_bracket_Pa"],
                [120_000.0, 150_000.0])
        record = self._record(2)
        self.assertIs(record.calculation_status, CalculationStatus.SUCCESS)
        output = record.outputs
        boundary = output["actual_boundary"]
        self.assertEqual(boundary["source_0d_outlet_total_pressure_Pa"],
                         150_000.0)
        self.assertEqual(len(boundary["closure"]["actual_stages"]), 3)
        self.assertAlmostEqual(
            boundary["actual_outlet_total_pressure_Pa"], 150_000.03650198862,
            delta=1e-6 * 150_000.0)
        self.assertTrue(boundary["source_target_differs_from_actual"])
        power = output["power"]
        self.assertEqual(power["compressor_power_W"],
                         record.payload["targets"]["main_compressor_power_W"])
        self.assertEqual(power["compressor_power_basis"],
                         "exactly source result.compressor.power")
        stage_gas = sum(
            stage["G_c"] * stage["L_т"]
            for stage in boundary["closure"]["actual_stages"])
        self.assertAlmostEqual(power["stage_gas_power_W"], stage_gas, places=9)
        expected = power["mechanical_efficiency"] * stage_gas
        self.assertAlmostEqual(power["stage_mechanical_power_W"], expected,
                               places=6)
        self.assertAlmostEqual(
            power["net_mechanical_power_W"],
            expected - record.payload["targets"]["main_compressor_power_W"],
            places=6)
        self.assertEqual(power["engine_power_closed"], False)
        self.assertEqual(power["engine_power_label"],
                         smoke.POWER_NOT_CLOSED_LABEL)
        self.assertIsNone(output["overall_engineering_feasibility"])
        self.assertEqual(output["overall_engineering_feasibility_status"],
                         "NOT_CHECKED")

    def test_explicit_lower_static_bracket_ab_single_stage_diagnostic(self):
        diagnostic_bracket = [60_000.0, 120_000.0]
        provenance = (
            "Explicit lower-static-pressure diagnostic; no automatic bracket "
            "widening; no flow/geometry/speed adjustment")
        expected_net = {0: 87_508.02147234694, 3: 87_598.49500288698}
        plan = smoke.plan_payloads()
        for index in (0, 3):
            with self.subTest(point_index=index):
                source_payload = plan[index]
                source_cycle = smoke.calculate_architecture_cycle(
                    smoke.payload_to_input(
                        source_payload["architecture_cycle_source"]))
                original = copy.deepcopy(source_payload)
                payload = copy.deepcopy(source_payload)
                payload["brackets"]["main"] = list(diagnostic_bracket)
                payload["brackets"]["provenance"] = provenance
                expected_payload = copy.deepcopy(original)
                expected_payload["brackets"]["main"] = list(diagnostic_bracket)
                expected_payload["brackets"]["provenance"] = provenance
                self.assertEqual(payload, expected_payload)
                self.assertEqual(plan[index], original)

                outcome = smoke.evaluate_coupled_payload(payload)
                self.assertIs(outcome.status, CalculationStatus.SUCCESS)
                self.assertIsNone(outcome.failure_stage)
                self.assertIsNone(outcome.failure_reason)
                output = outcome.outputs
                boundary = output["actual_boundary"]
                closure = boundary["closure"]
                power = output["power"]

                self.assertEqual(
                    boundary["target_outlet_total_pressure_Pa"], 150_000.0)
                self.assertEqual(
                    boundary["source_0d_outlet_total_pressure_Pa"], 150_000.0)
                actual_outlet = boundary["actual_outlet_total_pressure_Pa"]
                self.assertEqual(actual_outlet, closure["actual_metric"])
                self.assertLessEqual(
                    abs(actual_outlet - 150_000.0) / 150_000.0, 1e-6)
                self.assertEqual(closure["controls"]["static_pressure_bracket_Pa"],
                                 diagnostic_bracket)
                solved_static = closure["solved_static_pressure_Pa"]
                self.assertGreater(solved_static, diagnostic_bracket[0])
                self.assertLess(solved_static, diagnostic_bracket[1])
                self.assertLess(solved_static, 120_000.0)

                stages = closure["actual_stages"]
                self.assertEqual(len(stages), 1)
                self.assertEqual(closure["source_point"]["stage_count"],
                                 payload["stage_counts"]["main"])
                self.assertEqual(closure["controls"]["n_rpm"], 84_000.0)
                self.assertEqual(closure["controls"]["D_hub_m"], 0.065)
                self.assertEqual(payload["stage_controls"]["speed_rpm"], 84_000.0)
                self.assertEqual(payload["stage_controls"]["hub_diameter_m"], 0.065)
                self.assertEqual(payload["stage_controls"]["reference_exit_mach"],
                                 0.3379)
                source_flow = float(source_cycle.gas_mass_flow)
                for stage in stages:
                    self.assertEqual(stage["G_c"], source_flow)
                self.assertEqual(
                    output["source_input_sha256"],
                    smoke.canonical_hash(
                        payload["architecture_cycle_source"]))

                stage_gas = sum(stage["G_c"] * stage["L_т"]
                                for stage in stages)
                efficiency = float(
                    source_cycle.shaft.mechanical_efficiencies[0])
                mechanical = efficiency * stage_gas
                compressor = float(source_cycle.compressor.power)
                recomputed_net = mechanical - compressor
                self.assertAlmostEqual(power["stage_gas_power_W"], stage_gas,
                                       places=9)
                self.assertEqual(power["mechanical_efficiency"], efficiency)
                self.assertEqual(power["compressor_power_W"], compressor)
                self.assertAlmostEqual(
                    power["net_mechanical_power_W"], recomputed_net, places=6)
                self.assertGreater(recomputed_net, 0.0)
                self.assertLess(recomputed_net, 110_000.0)
                self.assertAlmostEqual(recomputed_net, expected_net[index],
                                       delta=1e-6 * expected_net[index])
                self.assertFalse(power["engine_power_closed"])
                self.assertEqual(power["engine_power_label"],
                                 smoke.POWER_NOT_CLOSED_LABEL)
                self.assertIsNone(output["overall_engineering_feasibility"])
                self.assertEqual(
                    output["overall_engineering_feasibility_status"],
                    "NOT_CHECKED")

        self.assertEqual(len(self.records), 15)
        self.assertEqual(self.summary["status_counts"],
                         {"NUMERICAL_FAILURE": 2, "SUCCESS": 13})
        for index in (0, 3):
            record = self._record(index)
            self.assertIs(record.calculation_status,
                          CalculationStatus.NUMERICAL_FAILURE)
            self.assertEqual(record.failure_reason, "boundary_not_bracketed")
            self.assertEqual(record.payload["brackets"]["main"],
                             [120_000.0, 150_000.0])

    def test_c_gg3_fpt1_actual_output_regression_from_stage_metrics(self):
        record = self._record(12)
        self.assertIs(record.calculation_status, CalculationStatus.SUCCESS)
        self.assertEqual(record.payload["stage_counts"],
                         {"gas_generator": 3, "free_power": 1})
        output = record.outputs
        power = output["power"]
        expected_output = 106_095.68575825705
        self.assertAlmostEqual(power["net_mechanical_power_W"], expected_output,
                               delta=1e-6 * abs(expected_output))
        self.assertAlmostEqual(power["signed_power_residual_W"],
                               expected_output - 110_000.0,
                               delta=1e-6 * 3_904.314241742948)
        boundary = output["actual_boundary"]
        gg_closure = boundary["gas_generator"]["closure"]
        fpt_closure = boundary["free_power"]["closure"]
        self.assertEqual(len(gg_closure["actual_stages"]), 3)
        self.assertEqual(len(fpt_closure["actual_stages"]), 1)
        self.assertEqual(boundary["gas_generator"]["target_compressor_power_W"],
                         record.payload["targets"]["gas_generator_compressor_power_W"])
        self.assertAlmostEqual(
            boundary["gas_generator"]["actual_mechanical_power_W"],
            power["gas_generator"]["mechanical_power_W"], places=6)
        recomputed_gg = power["gas_generator"]["mechanical_efficiency"] * sum(
            stage["G_c"] * stage["L_т"]
            for stage in gg_closure["actual_stages"])
        self.assertAlmostEqual(
            recomputed_gg, power["gas_generator"]["mechanical_power_W"],
            places=6)
        recomputed_fpt = power["free_power"]["mechanical_efficiency"] * sum(
            stage["G_c"] * stage["L_т"]
            for stage in fpt_closure["actual_stages"])
        self.assertAlmostEqual(
            recomputed_fpt, power["net_mechanical_power_W"], places=6)
        continuity = boundary["gas_generator_to_free_power_continuity"]
        self.assertTrue(continuity["temperature_exact"])
        self.assertTrue(continuity["pressure_exact"])
        self.assertTrue(continuity["flow_exact"])
        self.assertTrue(continuity["flow_matches_source"])
        self.assertEqual(continuity["source_flow_kg_s"],
                         gg_closure["actual_stages"][-1]["G_c"])
        self.assertEqual(continuity["free_power_first_stage_T0_K"],
                         continuity["gas_generator_last_stage_T_total_K"])
        self.assertEqual(continuity["free_power_first_stage_p0_Pa"],
                         continuity["gas_generator_last_stage_p_total_Pa"])
        self.assertEqual(
            fpt_closure["actual_stages"][0]["T0"],
            gg_closure["actual_stages"][-1]["T_ст_п"])
        self.assertEqual(power["engine_power_closed"], False)
        self.assertEqual(power["engine_power_label"],
                         smoke.POWER_NOT_CLOSED_LABEL)
        self.assertEqual(output["source_gas_basis"], "textbook_true_cp")
        self.assertIsNone(output["overall_engineering_feasibility"])
        self.assertEqual(output["overall_engineering_feasibility_status"],
                         "NOT_CHECKED")

    def test_pi_outside_profile_is_kept_success_not_filtered(self):
        record = self._record(12)
        self.assertIs(record.calculation_status, CalculationStatus.SUCCESS)
        acceptance = record.outputs["posthoc_pi_acceptance"]
        gas_generator = acceptance["gas_generator"]
        self.assertEqual(gas_generator["profile"], "PROMPT_STAGE_PI_1_8_2_4")
        self.assertFalse(gas_generator["pass"])
        for stage in gas_generator["stages"]:
            self.assertLess(stage["actual_pi_stage"], 1.8)
            self.assertEqual(stage["status"], "FAIL")
        self.assertIsNone(record.outputs["overall_engineering_feasibility"])
        self.assertEqual(record.outputs["rho_065_posthoc"]["status"],
                         "NOT_APPLIED")

    def test_pi4_a3_legacy_success_outside_table_optin_source_error(self):
        payload = _pi4_a3_payload()
        lo, hi = self._table_bounds()
        saved = _saved_payload(
            "reference/architecture_stage_smoke/"
            "2026-11-11-pi3-13-sized-coarse", PI4_A3_CANDIDATE)
        if saved is not None:
            self.assertEqual(smoke.canonical_hash(saved),
                             smoke.canonical_hash(payload))
            payload = copy.deepcopy(saved)
        legacy = smoke.evaluate_coupled_payload(payload)
        self.assertIs(legacy.status, CalculationStatus.SUCCESS)
        self.assertNotIn("reaction_table_source_domain", legacy.outputs)
        self.assertEqual(legacy.outputs["rho_065_posthoc"],
                         {"status": "NOT_APPLIED"})
        stages = legacy.outputs["actual_boundary"]["closure"]["actual_stages"]
        actuals = [stage["l_л"] / stage["D_л"] for stage in stages]
        self.assertEqual(len(actuals), 3)
        self.assertTrue(all(value > hi for value in actuals))
        self.assertEqual(legacy.outputs["power"]["sized_air_mass_flow_kg_s"],
                         PI4_A3_AIR)

        outcome = smoke.evaluate_coupled_payload(_with_policy(payload))
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertIsNot(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertIsNot(outcome.status, CalculationStatus.NON_PHYSICAL_STATE)
        self.assertEqual(outcome.failure_stage, "reaction_table_source_domain")
        self.assertEqual(outcome.failure_reason, "reaction_table_source_domain")
        self.assertEqual(
            outcome.outputs["actual_boundary"]["closure"]["actual_stages"],
            stages)
        self.assertEqual(outcome.outputs["power"], legacy.outputs["power"])
        self.assertEqual(
            outcome.outputs["flow_sizing"]["trial_air_mass_flow_kg_s"],
            PI4_A3_AIR)
        domain = outcome.diagnostics["source_domain"]
        self.assertEqual(domain["role"], "main")
        self.assertEqual(domain["table_bounds"], [lo, hi])
        self.assertEqual(domain["actual_l_over_D"], actuals)
        self.assertTrue(all(margin < 0.0 for margin in domain["signed_margins"]))
        self.assertEqual(domain["actual_closure"]["actual_stages"], stages)
        self.assertEqual(
            domain["flow_metadata"]["trial_air_mass_flow_kg_s"], PI4_A3_AIR)
        self.assertEqual(outcome.last_valid_state["role"], "main")
        self.assertEqual(
            outcome.last_valid_state["input"]["stage_controls"]
            ["reaction_table_policy"], REQUIRE_DOMAIN)
        self.assertNotIn("reaction_table_policy",
                         payload["stage_controls"])

        self.assertEqual(self.summary["status_counts"],
                         {"NUMERICAL_FAILURE": 2, "SUCCESS": 13})
        for default in smoke.plan_payloads():
            self.assertNotIn("reaction_table_policy", default["stage_controls"])

    def test_c2_refined_optin_success_matches_legacy_and_verifies_table(self):
        payload = _c2_payload()
        lo, hi = self._table_bounds()
        saved = _saved_payload(
            "reference/architecture_stage_smoke/"
            "2026-11-11-c-independent-refinement", C2_CANDIDATE)
        if saved is not None:
            self.assertEqual(smoke.canonical_hash(saved),
                             smoke.canonical_hash(payload))
            payload = copy.deepcopy(saved)
        legacy = smoke.evaluate_coupled_payload(payload)
        optin = smoke.evaluate_coupled_payload(_with_policy(payload))
        self.assertIs(legacy.status, CalculationStatus.SUCCESS)
        self.assertIs(optin.status, CalculationStatus.SUCCESS)
        self.assertNotIn("reaction_table_source_domain", legacy.outputs)
        self.assertEqual(optin.outputs["power"], legacy.outputs["power"])
        self.assertEqual(optin.outputs["actual_boundary"],
                         legacy.outputs["actual_boundary"])
        self.assertEqual(optin.outputs["flow_sizing"],
                         legacy.outputs["flow_sizing"])
        continuity = optin.outputs["actual_boundary"][
            "gas_generator_to_free_power_continuity"]
        self.assertTrue(continuity["flow_exact"])
        self.assertTrue(continuity["flow_matches_sized"])
        self.assertEqual(
            continuity["sized_gas_flow_kg_s"],
            optin.outputs["flow_sizing"]["trial_gas_mass_flow_kg_s"])
        domain = optin.outputs["reaction_table_source_domain"]
        self.assertTrue(domain["final_domain_verified"])
        self.assertEqual(domain["status"], "FINAL_SOURCE_DOMAIN_VERIFIED")
        self.assertEqual(domain["table_bounds"], [lo, hi])
        self.assertEqual(domain["intermediate_clipping"],
                         "NUMERICAL_INITIALIZATION_ONLY_NOT_VALIDATED_STATES")
        self.assertEqual(set(domain["roles"]),
                         {"gas_generator", "free_power"})
        for role in domain["roles"].values():
            self.assertTrue(role["final_domain_verified"])
            self.assertEqual(role["table_bounds"], [lo, hi])
            for actual in role["actual_l_over_D"]:
                self.assertGreaterEqual(
                    actual, lo - smoke.REACTION_TABLE_DOMAIN_TOL)
                self.assertLessEqual(
                    actual, hi + smoke.REACTION_TABLE_DOMAIN_TOL)

    def test_synthetic_c_gg_domain_failure_skips_fpt_and_keeps_gg(self):
        payload = _with_policy(smoke.plan_payloads()[12])
        closure = copy.deepcopy(
            self._record(12).outputs["actual_boundary"]["gas_generator"]
            ["closure"])
        for stage in closure["actual_stages"]:
            stage["l_л"] = 0.5 * stage["D_л"]
        with mock.patch.object(
                ConstHubOrchestration, "solve_prescribed_mechanical_power",
                return_value=closure) as gg_call, mock.patch.object(
                    ConstHubOrchestration, "solve_total_outlet_pressure",
                    side_effect=AssertionError("FPT must not be computed")):
            outcome = smoke.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_reason, "reaction_table_source_domain")
        self.assertEqual(gg_call.call_count, 1)
        self.assertIn("gas_generator", outcome.outputs)
        self.assertNotIn("free_power", outcome.outputs)
        self.assertEqual(
            outcome.outputs["gas_generator"]["closure"]["actual_stages"],
            closure["actual_stages"])

    def test_synthetic_c_fpt_domain_failure_preserves_both_closures(self):
        payload = _with_policy(smoke.plan_payloads()[12])
        boundary = self._record(12).outputs["actual_boundary"]
        gg_closure = copy.deepcopy(boundary["gas_generator"]["closure"])
        fpt_closure = copy.deepcopy(boundary["free_power"]["closure"])
        with mock.patch.object(
                ConstHubOrchestration, "solve_prescribed_mechanical_power",
                return_value=gg_closure), mock.patch.object(
                    ConstHubOrchestration, "solve_total_outlet_pressure",
                    return_value=fpt_closure):
            outcome = smoke.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
        self.assertEqual(outcome.failure_reason, "reaction_table_source_domain")
        self.assertEqual(outcome.diagnostics["source_domain"]["role"],
                         "free_power")
        self.assertEqual(
            outcome.outputs["gas_generator"]["closure"]["actual_stages"],
            gg_closure["actual_stages"])
        self.assertEqual(
            outcome.outputs["free_power"]["closure"]["actual_stages"],
            fpt_closure["actual_stages"])

    def test_invalid_reaction_table_policy_enum_is_rejected(self):
        base = smoke.plan_payloads()[2]
        for bad in ("legacy", "", None, 1, True, ["require_final_source_domain"]):
            payload = copy.deepcopy(base)
            payload["stage_controls"]["reaction_table_policy"] = bad
            with self.subTest(policy=bad):
                outcome = smoke.evaluate_coupled_payload(payload)
                self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
                self.assertEqual(outcome.failure_stage, "payload_validation")
                with self.assertRaises(
                        (smoke.ArchitectureCoupledSmokeError,
                         ValueError, TypeError, KeyError)):
                    smoke._validate_plan([payload] * 15)
        for good in (LEGACY_CLIPPED, REQUIRE_DOMAIN):
            payload = copy.deepcopy(base)
            payload["stage_controls"]["reaction_table_policy"] = good
            self.assertEqual(smoke._validate_payload(payload).index, 2)

    def test_engineering_pi_outside_table_inside_stays_success(self):
        payload = _with_policy(smoke.plan_payloads()[12])
        boundary = self._record(12).outputs["actual_boundary"]
        gg_closure = copy.deepcopy(boundary["gas_generator"]["closure"])
        fpt_closure = copy.deepcopy(boundary["free_power"]["closure"])
        for stage in fpt_closure["actual_stages"]:
            stage["l_л"] = 0.2 * stage["D_л"]
        with mock.patch.object(
                ConstHubOrchestration, "solve_prescribed_mechanical_power",
                return_value=gg_closure), mock.patch.object(
                    ConstHubOrchestration, "solve_total_outlet_pressure",
                    return_value=fpt_closure):
            outcome = smoke.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.SUCCESS)
        acceptance = outcome.outputs["posthoc_pi_acceptance"]["gas_generator"]
        self.assertFalse(acceptance["pass"])
        self.assertTrue(any(stage["status"] == "FAIL"
                            for stage in acceptance["stages"]))
        self.assertEqual(outcome.outputs["rho_065_posthoc"],
                         {"status": "NOT_APPLIED"})
        domain = outcome.outputs["reaction_table_source_domain"]
        self.assertTrue(domain["final_domain_verified"])
        for role in domain["roles"].values():
            for actual in role["actual_l_over_D"]:
                self.assertGreaterEqual(
                    actual, domain["table_bounds"][0]
                    - smoke.REACTION_TABLE_DOMAIN_TOL)
                self.assertLessEqual(
                    actual, domain["table_bounds"][1]
                    + smoke.REACTION_TABLE_DOMAIN_TOL)

    def test_reaction_table_bounds_use_actual_namespace_not_hardcoded(self):
        custom = {"ρ_таблица": ([0.2, 0.4], [0.3, 0.5])}
        self.assertEqual(smoke._reaction_table_bounds(custom), (0.2, 0.4))
        stage = copy.deepcopy(
            self._record(12).outputs["actual_boundary"]["gas_generator"]
            ["closure"]["actual_stages"][0])
        stage["l_л"] = 0.35 * stage["D_л"]
        closure = {"actual_stages": [stage]}
        record, outside = smoke._reaction_table_domain_record(
            "main", closure, custom, policy=REQUIRE_DOMAIN)
        self.assertEqual(outside, [])
        self.assertEqual(record["table_bounds"], [0.2, 0.4])
        self.assertTrue(record["final_domain_verified"])
        actual, actual_outside = smoke._reaction_table_domain_record(
            "main", closure,
            {"ρ_таблица": self.constants["reaction_table"]},
            policy=REQUIRE_DOMAIN)
        self.assertEqual(actual_outside, [1])
        self.assertEqual(actual["table_bounds"], list(self._table_bounds()))
        for bad in (([0.4, 0.2], [0.3, 0.5]),
                    ([0.2, float("inf")], [0.3, 0.5]), ([], [])):
            with self.subTest(table=bad):
                with self.assertRaises(smoke.ArchitectureCoupledSmokeError):
                    smoke._reaction_table_bounds({"ρ_таблица": bad})

    def test_no_flow_speed_hub_or_stage_count_tuning(self):
        seen = set()
        for record in self.records:
            payload = record.payload
            seen.add((payload["assembly"],
                      tuple(sorted(payload["stage_counts"].items()))))
            self.assertEqual(payload["stage_controls"]["speed_rpm"], 84_000.0)
            self.assertEqual(payload["stage_controls"]["hub_diameter_m"], 0.065)
            self.assertEqual(payload["stage_controls"]["reference_exit_mach"],
                             0.3379)
            if record.calculation_status is not CalculationStatus.SUCCESS:
                continue
            output = record.outputs
            if payload["coupling"] == "AB":
                stages = output["actual_boundary"]["closure"]["actual_stages"]
                source_flow = stages[0]["G_c"]
                for stage in stages:
                    self.assertEqual(stage["G_c"], source_flow)
                closures = [output["actual_boundary"]["closure"]]
            else:
                continuity = output["actual_boundary"][
                    "gas_generator_to_free_power_continuity"]
                source_flow = continuity["source_flow_kg_s"]
                self.assertTrue(continuity["flow_exact"])
                self.assertTrue(continuity["flow_matches_source"])
                closures = [
                    output["actual_boundary"]["gas_generator"]["closure"],
                    output["actual_boundary"]["free_power"]["closure"]]
            for closure in closures:
                self.assertEqual(closure["controls"]["n_rpm"], 84_000.0)
                self.assertEqual(closure["controls"]["D_hub_m"], 0.065)
                self.assertEqual(closure["pressure_ratio_policy"],
                                 "raw_unbounded")
                self.assertEqual(closure["stage_count_selection"], "NONE")
                self.assertGreater(
                    closure["source_point"]["inlet_total_pressure_Pa"], 0.0)
            if payload["coupling"] == "AB":
                self.assertEqual(closures[0]["source_point"]["stage_count"],
                                 payload["stage_counts"]["main"])
            else:
                self.assertEqual(closures[0]["source_point"]["stage_count"],
                                 payload["stage_counts"]["gas_generator"])
                self.assertEqual(closures[1]["source_point"]["stage_count"],
                                 payload["stage_counts"]["free_power"])
        self.assertEqual(len(seen), 15)

    def test_injected_fpt_failure_keeps_successful_gg_partial(self):
        payload = smoke.plan_payloads()[12]
        failure = NumericalConvergenceError(
            "test FPT stall", stage="unit.free_power",
            reason="unit_test_fpt_stall", last_valid_state={"y": 2.0})
        with mock.patch.object(
                ConstHubOrchestration, "solve_total_outlet_pressure",
                side_effect=failure):
            outcome = smoke.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(outcome.failure_stage, "unit.free_power")
        self.assertEqual(outcome.failure_reason, "unit_test_fpt_stall")
        self.assertEqual(
            outcome.last_valid_state["solver_last_valid_state"]["y"], 2.0)
        gas_generator = outcome.outputs["gas_generator"]
        self.assertEqual(len(gas_generator["closure"]["actual_stages"]), 3)
        self.assertNotIn("free_power", outcome.outputs)
        self.assertEqual(
            outcome.diagnostics["solver"]["reason"], "unit_test_fpt_stall")

    def test_typed_numerical_failure_is_not_physical_infeasibility(self):
        payload = smoke.plan_payloads()[0]
        outcome = smoke.evaluate_coupled_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertIsNot(outcome.status, CalculationStatus.NON_PHYSICAL_STATE)
        self.assertEqual(outcome.failure_reason, "boundary_not_bracketed")
        residuals = outcome.last_valid_state["solver_last_valid_state"][
            "signed_residuals"]
        self.assertEqual(len(residuals), 2)
        self.assertAlmostEqual(residuals[0], 44247.791979642905, places=9)
        self.assertAlmostEqual(residuals[1], 71879.31534821744, places=9)
        self.assertIsNone(outcome.outputs)

    def test_malformed_payload_controls_are_not_silently_ignored(self):
        base = smoke.plan_payloads()[2]

        def evaluate(mutator):
            payload = copy.deepcopy(base)
            mutator(payload)
            outcome = smoke.evaluate_coupled_payload(payload)
            self.assertIs(outcome.status, CalculationStatus.MODEL_ERROR)
            self.assertIn(outcome.failure_stage,
                          {"payload_validation", "coupled_control_error"})
            with self.assertRaises((smoke.ArchitectureCoupledSmokeError,
                                    KeyError, ValueError, TypeError)):
                smoke._validate_plan([payload] * 15)

        evaluate(lambda p: p["stage_controls"].update(speed_rpm=85_000.0))
        evaluate(lambda p: p["stage_controls"]["main"].update(alpha1_deg=[1.0]))
        evaluate(lambda p: p["brackets"].pop("main"))
        evaluate(lambda p: p.update(architecture="SINGLE_SHAFT_SINGLE_CC"))
        evaluate(lambda p: p["stage_counts"].update(main=4))
        evaluate(lambda p: p.update(extra_control=True))
        evaluate(lambda p: p["stage_controls"].update(speed_basis="tuned"))

    def test_reproduce_all_including_failures_and_successes(self):
        self.assertEqual(self.summary["reproduce"]["checked"], 15)
        self.assertTrue(self.summary["reproduce"]["matches"])
        self.assertEqual(self.summary["reproduce"]["differences"], [])
        for index in (0, 12):
            expected = self._record(index)
            actual = evaluate_candidate(
                lambda item: smoke.evaluate_coupled_payload(item),
                expected.payload, architecture=expected.architecture,
                model_variant=expected.model_variant, run_id=expected.run_id,
                sampling_method=expected.sampling_method,
                sampling_metadata=expected.sampling)
            self.assertEqual(
                compare_candidates(expected, actual, rtol=1e-12, atol=1e-9),
                ())
        residuals = [point for point in self.summary["points"]
                     if point["calculation_status"] == "SUCCESS"]
        self.assertEqual(len(residuals), 13)
        self.assertEqual(len({item["candidate_id"] for item in residuals}), 13)
        self.assertNotIn("actual_stages", json.dumps(self.summary["points"]))

    def test_resume_appends_zero_duplicates_and_keeps_journal(self):
        before = load_manifest(self.destination)
        summary = smoke.run_plan(
            self.destination, resume=True, reproduce=False,
            constants=self.constants, model_digest=self.model_digest)
        self.assertEqual(summary["run"]["records_appended_this_invocation"], 0)
        after = load_manifest(self.destination)
        self.assertEqual(after["record_count"], before["record_count"])
        self.assertEqual(after["journal_sha256"], before["journal_sha256"])
        self.assertTrue(after["complete"])

    def test_changed_model_digest_or_source_fixture_are_rejected_on_resume(self):
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                self.destination, resume=True, reproduce=False,
                constants=self.constants, model_digest="0" * 64)
        altered = copy.deepcopy(stage_smoke.plan_payloads())
        for payload in altered:
            payload["architecture_cycle"]["inlet_total_pressure"] += 1.0
        with mock.patch.object(
                stage_smoke, "plan_payloads", return_value=altered):
            with self.assertRaises(DatasetError):
                smoke.run_plan(
                    self.destination, resume=True, reproduce=False,
                    constants=self.constants, model_digest=self.model_digest)

    def test_interrupt_then_resume_without_duplicates(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="architecture-coupled-interrupt-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        calls = {"count": 0}

        def interrupting(payload):
            calls["count"] += 1
            if calls["count"] == 3:
                raise KeyboardInterrupt()
            return EvaluationOutcome.success(
                outputs=_fake_success_outputs())

        with self.assertRaises(KeyboardInterrupt):
            smoke.run_plan(
                destination, evaluator=interrupting,
                constants=self.constants, model_digest=self.model_digest,
                reproduce=False)
        incomplete = load_manifest(destination)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["record_count"], 2)
        self.assertFalse((destination / smoke.SUMMARY_NAME).exists())

        summary = smoke.run_plan(
            destination, resume=True,
            evaluator=lambda payload: EvaluationOutcome.success(
                outputs=_fake_success_outputs()),
            constants=self.constants, model_digest=self.model_digest,
            reproduce=False)
        complete = load_manifest(destination)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["record_count"], 15)
        self.assertEqual(summary["run"]["records_appended_this_invocation"], 13)
        records = read_records(destination)
        self.assertEqual(len(records), 15)
        self.assertEqual(len({record.candidate_id for record in records}), 15)

    def test_c_pairs_are_independently_attempted(self):
        c_records = [record for record in self.records
                     if record.payload["coupling"] == "C"]
        self.assertEqual(len(c_records), 9)
        keys = [(record.payload["stage_counts"]["gas_generator"],
                 record.payload["stage_counts"]["free_power"])
                for record in c_records]
        self.assertEqual(keys, [(g, f) for g in (1, 2, 3) for f in (1, 2, 3)])
        # Each pair is a distinct stored candidate payload; the same GG count
        # recurs with an independently rebuilt namespace and closure.
        payload_hashes = {smoke.canonical_hash(record.payload)
                          for record in c_records}
        self.assertEqual(len(payload_hashes), 9)
        for gas_generator in (1, 2, 3):
            rows = [record for record in c_records
                    if record.payload["stage_counts"]["gas_generator"]
                    == gas_generator]
            self.assertEqual(len(rows), 3)
            gg_values = {json.dumps(
                record.outputs["actual_boundary"]["gas_generator"]["closure"],
                sort_keys=True) for record in rows}
            self.assertEqual(len(gg_values), 1)
            fpt_values = {json.dumps(
                record.outputs["actual_boundary"]["free_power"]["closure"],
                sort_keys=True) for record in rows}
            self.assertEqual(len(fpt_values), 3)


class ReactionTablePolicyFlowConfigTests(unittest.TestCase):
    """Synthetic flow-config checks; no expensive real flow fixture is run."""

    def _case(self, policy):
        payload = copy.deepcopy(smoke.plan_payloads()[2])
        payload["stage_controls"]["reaction_table_policy"] = policy
        result = smoke.calculate_architecture_cycle(
            smoke.payload_to_input(payload["architecture_cycle_source"]))
        return {
            "case": f"POLICY_{policy}", "source_point_index": 2,
            "assembly": payload["assembly"], "coupling": payload["coupling"],
            "architecture": payload["architecture"],
            "source_payload": payload,
            "source_payload_sha256": smoke.canonical_hash(payload),
            "source_air_mass_flow_kg_s": float(result.air_mass_flow),
            "source_gas_mass_flow_kg_s": float(result.gas_mass_flow),
            "source_fuel_mass_flow_kg_s": float(result.fuel_mass_flow),
            "air_flow_bracket_kg_s": [0.5, 0.9],
            "target_gearbox_input_power_W": flow_smoke.TARGET,
            "outer_relative_tolerance": flow_smoke.OUTER_RELATIVE_TOL,
            "outer_max_iterations": flow_smoke.OUTER_MAX_ITER}

    def test_policy_is_locked_and_resume_rejects_changed_policy(self):
        required = self._case(REQUIRE_DOMAIN)
        legacy = self._case(LEGACY_CLIPPED)
        expected = smoke.with_compressor_pressure_ratios(
            smoke.plan_payloads()[2],
            required["source_payload"]["architecture_cycle_source"][
                "compressor_pressure_ratios"])
        self.assertTrue(flow_smoke._same_physical_base(
            required["source_payload"], expected))
        config = flow_smoke.make_run_config([required], {}, "0" * 64)
        item = config["cases"][0]
        self.assertEqual(item["reaction_table_policy"], REQUIRE_DOMAIN)
        self.assertIn("stage_controls_sha256", item)
        other = flow_smoke.make_run_config([legacy], {}, "0" * 64)
        self.assertNotEqual(config["plan_sha256"], other["plan_sha256"])

        absent = copy.deepcopy(required)
        del absent["source_payload"]["stage_controls"]["reaction_table_policy"]
        absent["source_payload_sha256"] = smoke.canonical_hash(
            absent["source_payload"])
        absent_config = flow_smoke.make_run_config([absent], {}, "0" * 64)
        self.assertNotIn("reaction_table_policy",
                         absent_config["cases"][0])
        self.assertNotIn("stage_controls_sha256",
                         absent_config["cases"][0])

        temporary = tempfile.TemporaryDirectory(
            prefix="reaction-table-policy-config-")
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        evaluator = lambda payload: EvaluationOutcome.success(
            outputs=_fake_success_outputs())
        flow_smoke.run_plan(
            destination, cases=[required], evaluator=evaluator,
            constants={}, model_digest="0" * 64, reproduce=False)
        with self.assertRaises(DatasetError):
            flow_smoke.run_plan(
                destination, resume=True, cases=[legacy], evaluator=evaluator,
                constants={}, model_digest="0" * 64, reproduce=False)


if __name__ == "__main__":
    unittest.main()
