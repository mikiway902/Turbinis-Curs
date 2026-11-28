"""Focused tests for the post-hoc sized-root audit (no model re-solve)."""

from __future__ import annotations

import contextlib, hashlib, io, json, sys, tempfile, unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import audit_sized_stage_roots as audit  # noqa: E402
from turbine_engine.dataset import DatasetWriter, load_manifest  # noqa: E402
from turbine_engine.raw import CalculationStatus, EvaluationOutcome  # noqa: E402

CONTROLS = {"n_rpm": 84000.0, "D_hub_m": 0.065, "static_pressure_bracket_Pa": [120000.0, 150000.0]}


def _stage(**updates):
    stage = {"pi_stage": 2.0, "rho": 0.5, "alpha2_deg": 85.0, "sigma_rotor": 250.0,
             "l_l": 0.02, "D_l": 0.1, "c2": 150.0, "u1": 300.0, "u2": 310.0,
             "M_c1": 0.7, "M_w2": 0.6, "L_t": 100000.0, "G_c": 1.02, "p_total": 150000.0,
             "T0": 1000.0, "T_total": 1000.0, "p0": 300000.0}
    stage.update(updates)
    return stage


def _flow(compressor):
    return {"trial_air_mass_flow_kg_s": 1.0, "trial_fuel_mass_flow_kg_s": 0.02,
            "trial_gas_mass_flow_kg_s": 1.02, "mass_residual_kg_s": 0.0,
            "pre_reducer": True, "source_air_mass_flow_kg_s": 0.5,
            "actual_compressor_power_target_W": compressor}


def _source_from(outputs):
    flow, power = outputs["flow_sizing"], outputs["power"]
    lc = flow["actual_compressor_power_target_W"] / flow["trial_air_mass_flow_kg_s"]
    return {"specific_work_J_kg": lc, "air_mass_flow_kg_s": flow["source_air_mass_flow_kg_s"],
            "compressor_power_W": lc * flow["source_air_mass_flow_kg_s"]}


def _outputs(stages=None):
    stages = stages or [_stage(p_total=160000.0), _stage()]
    gas_power = sum(stage["G_c"] * stage["L_t"] for stage in stages)
    mechanical = 0.99 * gas_power
    compressor = mechanical - 110000.0
    closure = {"actual_metric": stages[-1]["p_total"], "target_value": 150000.0,
               "target_kind": "total_outlet_pressure",
               "relative_residual": abs(stages[-1]["p_total"] - 150000.0) / 150000.0,
               "signed_residual": stages[-1]["p_total"] - 150000.0,
               "actual_stages": stages, "controls": dict(CONTROLS)}
    flow = _flow(compressor)
    power = {"mechanical_efficiency": 0.99, "stage_gas_power_W": gas_power,
             "stage_mechanical_power_W": mechanical, "compressor_power_W": compressor,
             "reference_0d_compressor_power_W": compressor * 0.5,
             "actual_sized_compressor_power_W": compressor,
             "net_mechanical_power_W": 110000.0, "target_gearbox_input_power_W": 110000.0,
             "signed_power_residual_W": 0.0, "relative_power_residual": 0.0,
             "engine_power_closed": True}
    return {"actual_boundary": {"closure": closure}, "power": power, "flow_sizing": flow}


def _c_outputs(gg_stages=None, fpt_stages=None):
    gg_stages = gg_stages or [_stage(G_c=1.02, L_t=100000.0, p_total=300000.0)]
    fpt_stages = fpt_stages or [_stage(G_c=1.02, L_t=110000.0 / (0.99 * 1.02))]
    gg_gas = sum(stage["G_c"] * stage["L_t"] for stage in gg_stages)
    gg_mech = 0.99 * gg_gas
    fpt_gas = sum(stage["G_c"] * stage["L_t"] for stage in fpt_stages)
    fpt_mech = 0.99 * fpt_gas
    compressor = gg_mech
    gg_closure = {"actual_metric": gg_mech, "target_value": compressor,
                  "target_kind": "mechanical_power", "relative_residual": 0.0,
                  "signed_residual": 0.0, "actual_stages": gg_stages,
                  "controls": dict(CONTROLS)}
    fpt_closure = {"actual_metric": fpt_stages[-1]["p_total"], "target_value": 150000.0,
                   "target_kind": "total_outlet_pressure",
                   "relative_residual": abs(fpt_stages[-1]["p_total"] - 150000.0) / 150000.0,
                   "signed_residual": fpt_stages[-1]["p_total"] - 150000.0,
                   "actual_stages": fpt_stages, "controls": dict(CONTROLS)}
    flow = _flow(compressor)
    power = {"gas_generator": {"mechanical_efficiency": 0.99, "stage_gas_power_W": gg_gas,
                               "mechanical_power_W": gg_mech, "compressor_power_W": compressor,
                               "balance_residual_W": 0.0,
                               "reference_0d_compressor_power_W": compressor * 0.5,
                               "target_compressor_power_W": compressor},
             "free_power": {"mechanical_efficiency": 0.99, "stage_gas_power_W": fpt_gas,
                            "mechanical_power_W": fpt_mech},
             "net_mechanical_power_W": fpt_mech, "target_gearbox_input_power_W": 110000.0,
             "signed_power_residual_W": fpt_mech - 110000.0,
             "relative_power_residual": abs(fpt_mech - 110000.0) / 110000.0,
             "engine_power_closed": abs(fpt_mech - 110000.0) <= 1e-6 * 110000.0}
    boundary = {"gas_generator": {"closure": gg_closure,
                                  "target_compressor_power_W": compressor,
                                  "gas_generator_balance_residual_W": 0.0},
                "free_power": {"closure": fpt_closure}}
    return {"actual_boundary": boundary, "power": power, "flow_sizing": flow}


def _root(record, outputs):
    power, flow = outputs["power"], outputs["flow_sizing"]
    return {"candidate_id": record.candidate_id,
            "air_mass_flow_kg_s": flow["trial_air_mass_flow_kg_s"],
            "fuel_mass_flow_kg_s": flow["trial_fuel_mass_flow_kg_s"],
            "gas_mass_flow_kg_s": flow["trial_gas_mass_flow_kg_s"],
            "net_mechanical_power_W": power["net_mechanical_power_W"],
            "signed_power_residual_W": power["signed_power_residual_W"],
            "relative_power_residual": power["relative_power_residual"],
            "numerical_power_closed": power["engine_power_closed"]}


def _summary_state(record, target=110000.0):
    outputs = record.outputs if isinstance(record.outputs, dict) else {}
    power = outputs.get("power") if isinstance(outputs, dict) else None
    metric = None
    if record.calculation_status is CalculationStatus.SUCCESS and isinstance(power, dict):
        metric = audit._num(power.get("net_mechanical_power_W"))
    return {"air_mass_flow_kg_s": record.payload["air_mass_flow_kg_s"],
            "candidate_id": record.candidate_id,
            "calculation_status": record.calculation_status.value,
            "failure_stage": record.failure_stage, "failure_reason": record.failure_reason,
            "net_mechanical_power_W": metric,
            "signed_power_residual_W": None if metric is None else metric - target,
            "relative_power_residual": None if metric is None else abs(metric - target) / target}


def _below_target_outputs(net=90000.0):
    outputs = _outputs()
    outputs["power"].update({"net_mechanical_power_W": net,
                             "signed_power_residual_W": net - 110000.0,
                             "relative_power_residual": abs(net - 110000.0) / 110000.0,
                             "engine_power_closed": False})
    return outputs


class AuditSizedStageRootsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sized-root-audit-")
        self.base = Path(self.temporary.name)
        self._source = None

    def tearDown(self):
        self.temporary.cleanup()

    def _write_cases(self, specs):
        dataset = self.base / "dataset"
        config = {"model_variant": "audit-test", "evaluator_version": "audit-test-v1"}
        built = []
        with DatasetWriter.create(dataset, run_config=config) as writer:
            for spec in specs:
                records = [writer.evaluate_and_append(
                    lambda payload, result=state["result"]: result,
                    {"air_mass_flow_kg_s": state["air"], "point_index": spec["point_index"],
                     "targets": spec.get("targets") or {"main_total_outlet_pressure_Pa": 150000.0}},
                    architecture=spec["architecture"], sampling_method="synthetic")
                    for state in spec["states"]]
                built.append((spec, records))
        manifest = load_manifest(dataset)
        cases = []
        for spec, records in built:
            target = spec.get("target", 110000.0)
            states = [_summary_state(record, target) for record in records]
            root_index = spec.get("root_index")
            root = (_root(records[root_index], spec["states"][root_index]["result"])
                    if root_index is not None else None)
            cases.append({"case": spec["case"], "architecture": spec["architecture"],
                          "source_point_index": spec["point_index"],
                          "target_gearbox_input_power_W": target,
                          "status": spec.get("status", "SUCCESS"),
                          "reason": spec.get("reason", "root_found"),
                          "root": root,
                          "numerical_power_closed": None if root is None else root["numerical_power_closed"],
                          "overall_engineering_feasibility": None,
                          "overall_engineering_feasibility_status": "NOT_CHECKED",
                          "trial_count": len(states), "evaluated_states": states,
                          "last_evaluated": states[-1]})
        summary = {"run": {"run_id": manifest["run_id"],
                           "journal_sha256": manifest["journal_sha256"],
                           "record_count": manifest["record_count"], "complete": True},
                   "cases": cases, "overall_engineering_feasibility": None,
                   "overall_engineering_feasibility_status": "NOT_CHECKED"}
        (dataset / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return dataset

    def _dataset_outputs(self, outputs, architecture="A_MAIN", targets=None, source=None):
        self._source = source or _source_from(outputs)
        return self._write_cases([{"case": "UNIT", "architecture": architecture,
            "point_index": 2, "targets": targets, "root_index": 0,
            "states": [{"air": 1.0, "result": outputs}]}])

    def _dataset(self, stages=None):
        return self._dataset_outputs(_outputs(stages))

    def _outer_failure_dataset(self, second=None, status="NUMERICAL_FAILURE",
                               reason="outer_bracket_not_bracketed"):
        second = _below_target_outputs(95000.0) if second is None else second
        return self._write_cases([{"case": "FAIL_OUTER", "architecture": "A_MAIN",
            "point_index": 2, "status": status, "reason": reason,
            "states": [{"air": 1.0, "result": _below_target_outputs(90000.0)},
                       {"air": 2.0, "result": second}]}])

    def _report(self, dataset):
        with mock.patch.object(audit, "_source_reference", return_value=self._source):
            return audit.audit_dataset(dataset)

    def _summary(self, dataset):
        path = dataset / "summary.json"
        return path, json.loads(path.read_text(encoding="utf-8"))

    def test_pass_margins_and_source_labels(self):
        report = self._report(self._dataset())
        root = report["roots"][0]
        for key in ("rho_065_upper", "prompt_pi_1_8_2_4",
                    "alpha2_source_page_38", "sigma_project_290MPa"):
            self.assertEqual(root["constraints"][key]["status"], "PASS")
        self.assertFalse(root["known_constraint_violation"])
        self.assertFalse(root["source_domain_violation"])
        self.assertTrue(root["numerical_closed"])
        self.assertEqual(root["compressor_power"]["status"], "PASS")
        self.assertTrue(root["compressor_power"]["source_and_actual_distinct"])
        self.assertIsNone(root["overall_engineering_feasibility"])
        self.assertEqual(root["overall_engineering_feasibility_status"], "NOT_CHECKED")
        self.assertIsNone(report["best_feasible_recommendation"])
        meta = report["source_metadata"]
        self.assertEqual(meta["alpha2"]["mid"]["page"], 38)
        self.assertEqual(meta["sigma_limit"]["value"], 290.0)
        self.assertIn("PROJECT_ASSUMPTION", meta["sigma_limit"]["basis"])
        self.assertIn("NOT_BOOK_MATERIAL_VALIDATION", meta["sigma_limit"]["basis"])
        self.assertFalse(meta["rho_profile"]["lower_bound_applied"])
        self.assertEqual(len(report["audit_digest"]), 64)

    def test_fail_margins_are_signed(self):
        stages = [_stage(rho=0.7, pi_stage=2.5, alpha2_deg=95.0, sigma_rotor=300.0)]
        root = self._report(self._dataset(stages))["roots"][0]
        for key in ("rho_065_upper", "prompt_pi_1_8_2_4",
                    "alpha2_source_page_38", "sigma_project_290MPa"):
            self.assertEqual(root["constraints"][key]["status"], "FAIL")
            self.assertLess(root["constraints"][key]["min_signed_margin"], 0.0)
        self.assertTrue(root["known_constraint_violation"])
        self.assertIsNone(root["overall_engineering_feasibility"])

    def test_missing_metric_is_not_checked(self):
        stages = [_stage(), _stage()]
        del stages[1]["alpha2_deg"]
        root = self._report(self._dataset(stages))["roots"][0]
        self.assertEqual(root["constraints"]["alpha2_source_page_38"]["status"], "NOT_CHECKED")
        self.assertFalse(root["known_constraint_violation"])
        self.assertTrue(root["has_not_checked"])
        self.assertIsNone(root["overall_engineering_feasibility"])
        self.assertEqual(root["groups"]["main"]["stages"][1]["metrics"]["alpha2_deg"], None)

    def test_source_domain_outside_is_not_a_physical_rejection(self):
        root = self._report(self._dataset([_stage(), _stage(l_l=0.04, D_l=0.1)]))["roots"][0]
        check = root["groups"]["main"]["stages"][1]["checks"]["reaction_table_lD_domain"]
        self.assertEqual(root["source_domain"]["status"], "OUTSIDE_SOURCE_DOMAIN")
        self.assertTrue(root["source_domain_violation"])
        self.assertLess(check["source_domain_margin"], 0.0)
        self.assertEqual(check["clip_status"], "SOURCE_TABLE_CLIPPED_NOT_REJECTED")
        self.assertIn("np.interp", check["note"])
        self.assertIn("not physical", check["note"])
        self.assertFalse(root["known_constraint_violation"])

    def test_sigma_290_is_project_not_book(self):
        meta = self._report(self._dataset())["source_metadata"]["sigma_limit"]
        self.assertEqual(meta["value"], 290.0)
        self.assertIn("DESIGN_ASSUMPTIONS", meta["basis"])
        self.assertNotIn("BOOK_REQUIREMENT", meta["basis"])

    def test_bad_root_reference_fails(self):
        dataset = self._dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["root"]["candidate_id"] = "cand_missing"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_mismatched_summary_metric_fails(self):
        dataset = self._dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["root"]["net_mechanical_power_W"] += 1.0
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_no_overwrite_and_raw_unchanged(self):
        dataset = self._dataset()
        names = ("dataset.jsonl", "manifest.json", "checkpoint.json", "summary.json")
        before = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
                  for name in names}
        output = self.base / "audit.json"
        with mock.patch.object(audit, "_source_reference", return_value=self._source):
            self.assertEqual(audit.main(["--dataset", str(dataset), "--output", str(output)]), 0)
            first = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(audit.main(["--dataset", str(dataset), "--output", str(output)]), 2)
        self.assertEqual(output.read_bytes(), first)
        after = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest()
                 for name in names}
        self.assertEqual(before, after)

    def test_mass_failure_stops_numerical_closure(self):
        outputs = _outputs(); gas_flow = 1.03
        for stage in outputs["actual_boundary"]["closure"]["actual_stages"]: stage["G_c"] = gas_flow
        gas_power = sum(stage["G_c"] * stage["L_t"] for stage in outputs["actual_boundary"]["closure"]["actual_stages"])
        mechanical = 0.99 * gas_power; compressor = mechanical - 110000.0
        outputs["flow_sizing"].update({"trial_gas_mass_flow_kg_s": gas_flow, "mass_residual_kg_s": gas_flow - 1.02,
                                       "actual_compressor_power_target_W": compressor})
        outputs["power"].update({"stage_gas_power_W": gas_power, "stage_mechanical_power_W": mechanical,
                                 "compressor_power_W": compressor, "reference_0d_compressor_power_W": compressor * 0.5,
                                 "actual_sized_compressor_power_W": compressor})
        root = self._report(self._dataset_outputs(outputs))["roots"][0]
        self.assertEqual(root["mass_fuel_air"]["status"], "FAIL")
        self.assertTrue(root["numerical_power_closed"])
        self.assertFalse(root["numerical_closed"])

    def test_c_state_mismatch_is_independent_and_stops_numerical_closure(self):
        outputs = _c_outputs()
        outputs["actual_boundary"]["gas_generator_to_free_power_continuity"] = {"temperature_exact": True, "pressure_exact": True, "flow_exact": True}
        outputs["actual_boundary"]["free_power"]["closure"]["actual_stages"][0]["T0"] += 5.0
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["state_continuity"]["status"], "FAIL")
        self.assertFalse(root["numerical_closed"])
        self.assertTrue(root["numerical_power_closed"])
        self.assertGreater(root["state_continuity"]["temperature_delta_K"], 0.0)

    def test_c_missing_state_metrics_is_not_checked(self):
        outputs = _c_outputs()
        del outputs["actual_boundary"]["gas_generator"]["closure"]["actual_stages"][-1]["T_total"]
        del outputs["actual_boundary"]["free_power"]["closure"]["actual_stages"][0]["p0"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["state_continuity"]["status"], "NOT_CHECKED")
        self.assertIn("GG.T_total", root["state_continuity"]["missing"])
        self.assertIn("FPT.p0", root["state_continuity"]["missing"])
        self.assertIsNone(root["state_continuity"]["gas_generator_last"]["T_total_K"])
        self.assertIsNone(root["numerical_closed"])

    def test_missing_efficiency_is_not_checked_ab(self):
        outputs = _outputs()
        del outputs["power"]["mechanical_efficiency"]
        root = self._report(self._dataset_outputs(outputs))["roots"][0]
        self.assertEqual(root["groups"]["main"]["power"]["status"], "NOT_CHECKED")
        self.assertIsNotNone(root["groups"]["main"]["power"]["stage_gas_power_W"])
        self.assertEqual(root["compressor_power"]["status"], "PASS")
        self.assertIsNone(root["numerical_closed"])
        self.assertIsNone(root["numerical_power_closed"])

    def test_missing_compressor_is_not_checked_ab(self):
        outputs = _outputs()
        del outputs["power"]["compressor_power_W"]
        root = self._report(self._dataset_outputs(outputs))["roots"][0]
        self.assertEqual(root["groups"]["main"]["power"]["status"], "NOT_CHECKED")
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_target_is_not_checked_ab(self):
        outputs = _outputs()
        del outputs["power"]["target_gearbox_input_power_W"]
        root = self._report(self._dataset_outputs(outputs))["roots"][0]
        self.assertEqual(root["groups"]["main"]["power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_flow_target_is_not_checked_ab(self):
        outputs = _outputs()
        source = _source_from(outputs)
        del outputs["flow_sizing"]["actual_compressor_power_target_W"]
        root = self._report(self._dataset_outputs(outputs, source=source))["roots"][0]
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_actual_sized_metadata_is_not_checked_ab(self):
        outputs = _outputs()
        del outputs["power"]["actual_sized_compressor_power_W"]
        root = self._report(self._dataset_outputs(outputs))["roots"][0]
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_wrong_preliminary_source_power_fails_ab(self):
        outputs = _outputs()
        outputs["power"]["reference_0d_compressor_power_W"] += 1.0
        with self.assertRaises(audit.AuditError):
            self._report(self._dataset_outputs(outputs))

    def test_missing_efficiency_is_not_checked_c(self):
        outputs = _c_outputs()
        del outputs["power"]["gas_generator"]["mechanical_efficiency"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["groups"]["gas_generator"]["power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])
        self.assertIsNone(root["numerical_power_closed"])

    def test_missing_compressor_is_not_checked_c(self):
        outputs = _c_outputs()
        del outputs["power"]["gas_generator"]["compressor_power_W"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["groups"]["gas_generator"]["power"]["status"], "NOT_CHECKED")
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_target_is_not_checked_c(self):
        outputs = _c_outputs()
        del outputs["power"]["target_gearbox_input_power_W"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["groups"]["free_power"]["power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_flow_target_is_not_checked_c(self):
        outputs = _c_outputs()
        source = _source_from(outputs)
        del outputs["flow_sizing"]["actual_compressor_power_target_W"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets, source=source))["roots"][0]
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_missing_boundary_compressor_target_is_not_checked_c(self):
        outputs = _c_outputs()
        del outputs["actual_boundary"]["gas_generator"]["target_compressor_power_W"]
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        root = self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))["roots"][0]
        self.assertEqual(root["compressor_power"]["status"], "NOT_CHECKED")
        self.assertIsNone(root["numerical_closed"])

    def test_wrong_preliminary_source_power_fails_c(self):
        outputs = _c_outputs()
        outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"] += 1.0
        targets = {"gas_generator_compressor_power_W": outputs["power"]["gas_generator"]["reference_0d_compressor_power_W"],
                   "free_power_outlet_total_pressure_Pa": 150000.0}
        with self.assertRaises(audit.AuditError):
            self._report(self._dataset_outputs(outputs, "C_COUPLED", targets))

    def test_failed_outer_case_preserves_inner_success_states(self):
        report = self._report(self._outer_failure_dataset())
        self.assertEqual(report["roots"], [])
        self.assertEqual((report["case_count"], report["root_count"], report["failed_count"]), (1, 0, 1))
        self.assertEqual(report["outer_case_status_counts"], {"NUMERICAL_FAILURE": 1})
        self.assertEqual(report["inner_trial_status_counts"], {"SUCCESS": 2})
        failed = report["failed_cases"][0]
        self.assertEqual(failed["status"], "NUMERICAL_FAILURE")
        self.assertEqual(failed["reason"], "outer_bracket_not_bracketed")
        self.assertIsNone(failed["root"])
        self.assertIsNone(failed["numerical_closed"])
        self.assertIsNone(failed["numerical_power_closed"])
        self.assertIsNone(failed["overall_engineering_feasibility"])
        self.assertEqual(failed["overall_engineering_feasibility_status"], "NOT_CHECKED")
        self.assertEqual(failed["trial_count"], 2)
        self.assertEqual(len(failed["evaluated_states"]), 2)
        self.assertEqual(failed["outer_residual_check"]["endpoint_same_sign"], True)
        self.assertLess(failed["outer_residual_check"]["first_signed_power_residual_W"], 0.0)

    def test_mixed_success_and_failure_counts_include_both(self):
        success, below = _outputs(), _below_target_outputs(90000.0)
        self._source = _source_from(success)
        dataset = self._write_cases([
            {"case": "UNIT_OK", "architecture": "A_MAIN", "point_index": 2, "root_index": 0,
             "states": [{"air": 1.0, "result": success}]},
            {"case": "UNIT_FAIL", "architecture": "A_MAIN", "point_index": 3,
             "status": "NUMERICAL_FAILURE", "reason": "outer_bracket_not_bracketed",
             "states": [{"air": 1.0, "result": below},
                        {"air": 2.0, "result": _below_target_outputs(95000.0)}]}])
        report = self._report(dataset)
        self.assertEqual([root["case"] for root in report["roots"]], ["UNIT_OK"])
        self.assertEqual([item["case"] for item in report["failed_cases"]], ["UNIT_FAIL"])
        self.assertEqual((report["case_count"], report["root_count"], report["failed_count"]), (2, 1, 1))
        self.assertEqual(report["outer_case_status_counts"], {"SUCCESS": 1, "NUMERICAL_FAILURE": 1})
        self.assertEqual(report["inner_trial_status_counts"], {"SUCCESS": 3})

    def test_non_success_inner_state_keeps_partial_gg_id(self):
        partial = EvaluationOutcome.failure(
            CalculationStatus.NON_PHYSICAL_STATE, "free_power", "non_positive_total_pressure",
            outputs={"actual_boundary": {"gas_generator": {"closure": {"actual_stages": []}}},
                     "power": {"gas_generator": {"mechanical_power_W": 1.0}}})
        report = self._report(self._outer_failure_dataset(
            partial, status="NON_PHYSICAL_STATE", reason="trial_failure_at_air_flow"))
        failed = report["failed_cases"][0]
        last = failed["evaluated_states"][-1]
        self.assertEqual(last["calculation_status"], "NON_PHYSICAL_STATE")
        self.assertIsNone(last["net_mechanical_power_W"])
        self.assertIsNone(last["signed_power_residual_W"])
        self.assertEqual(failed["partial_gas_generator_candidate_ids"], [last["candidate_id"]])
        self.assertEqual(report["inner_trial_status_counts"],
                         {"SUCCESS": 1, "NON_PHYSICAL_STATE": 1})
        self.assertIsNone(failed["overall_engineering_feasibility"])
        self.assertEqual(failed["failure_interpretation"],
                         "TYPED_FAILURE_PRESERVED_NOT_PHYSICAL_INFEASIBILITY")

    def test_not_bracketed_claim_with_opposite_endpoint_signs_is_rejected(self):
        dataset = self._outer_failure_dataset(second=_below_target_outputs(130000.0))
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_dropped_raw_attempt_is_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        del summary["cases"][0]["evaluated_states"][0]
        summary["cases"][0]["trial_count"] = 1
        summary["cases"][0]["last_evaluated"] = summary["cases"][0]["evaluated_states"][0]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_failed_case_with_root_is_rejected_as_ambiguous(self):
        dataset = self._dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["status"] = "NUMERICAL_FAILURE"
        summary["cases"][0]["reason"] = "outer_bracket_not_bracketed"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_duplicate_state_replacing_omitted_record_is_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        states = summary["cases"][0]["evaluated_states"]
        states[1] = dict(states[0])
        summary["cases"][0]["last_evaluated"] = states[-1]
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_outer_not_bracketed_requires_numerical_failure_status(self):
        dataset = self._outer_failure_dataset(status="NON_PHYSICAL_STATE")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_trial_failure_reason_requires_failed_final_record(self):
        dataset = self._outer_failure_dataset(status="NUMERICAL_FAILURE",
                                               reason="trial_failure_at_air_flow")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_trial_failure_status_must_match_final_raw_status(self):
        partial = EvaluationOutcome.failure(
            CalculationStatus.NON_PHYSICAL_STATE, "free_power", "non_positive_total_pressure")
        dataset = self._outer_failure_dataset(partial, status="NUMERICAL_FAILURE",
                                               reason="trial_failure_at_air_flow")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_success_missing_power_metric_is_preserved_as_model_error(self):
        outputs = _outputs()
        del outputs["power"]["net_mechanical_power_W"]
        dataset = self._write_cases([{"case": "MISSING_METRIC", "architecture": "A_MAIN",
            "point_index": 2, "status": "MODEL_ERROR", "reason": "success_missing_power_metric",
            "states": [{"air": 1.0, "result": outputs}]}])
        failed = self._report(dataset)["failed_cases"][0]
        self.assertEqual(failed["status"], "MODEL_ERROR")
        self.assertEqual(failed["reason"], "success_missing_power_metric")
        self.assertIsNone(failed["evaluated_states"][0]["net_mechanical_power_W"])
        self.assertEqual(failed["failure_interpretation"],
                         "TYPED_FAILURE_PRESERVED_NOT_PHYSICAL_INFEASIBILITY")

    def test_unknown_failure_reason_is_rejected(self):
        dataset = self._outer_failure_dataset(reason="not_a_real_failure_reason")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_unknown_case_failure_status_is_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["status"] = "MYSTERY"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_mismatched_failed_state_metric_is_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["evaluated_states"][0]["net_mechanical_power_W"] += 1.0
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_stale_failed_state_reason_is_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["evaluated_states"][0]["failure_reason"] = "stale_reason"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_failed_state_reference_and_bracket_claim_are_rejected(self):
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["evaluated_states"][1]["candidate_id"] = "cand_missing"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)
        self.base = self.base / "second"
        dataset = self._outer_failure_dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["evaluated_states"][1]["net_mechanical_power_W"] += 20000.0
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_success_case_without_root_still_fails(self):
        dataset = self._dataset()
        path, summary = self._summary(dataset)
        summary["cases"][0]["root"] = None
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaises(audit.AuditError):
            self._report(dataset)

    def test_reference_pi_anchor_split_is_failure_aware(self):
        dataset = ROOT / "reference" / "architecture_stage_smoke" / "2026-11-11-pi-anchor-split"
        if not dataset.is_dir():
            self.skipTest("saved pi-anchor-split reference dataset is absent")
        report = audit.audit_dataset(dataset)
        self.assertEqual(report["audit_version"], "sized-stage-root-audit-v3")
        self.assertEqual((report["case_count"], report["root_count"], report["failed_count"]), (11, 8, 3))
        self.assertEqual(report["outer_case_status_counts"], {"SUCCESS": 8, "NUMERICAL_FAILURE": 3})
        self.assertEqual(report["inner_trial_status_counts"], {"SUCCESS": 159})
        self.assertEqual(report["evaluated_state_count"], 159)
        self.assertTrue(report["all_raw_records_accounted_for"])
        self.assertEqual(len(report["roots"]), 8)
        self.assertEqual({root["root_candidate_id"] for root in report["roots"]},
                         set(report["run"]["root_ids"]))
        failed = {item["case"]: item for item in report["failed_cases"]}
        for name, point in (("A_PI3_W0.5", 2), ("B_PI3_W0.5", 5), ("C_PI3_W0.5", 9)):
            item = failed[name]
            self.assertEqual(item["status"], "NUMERICAL_FAILURE")
            self.assertEqual(item["reason"], "outer_bracket_not_bracketed")
            self.assertEqual(item["source_point_index"], point)
            self.assertIsNone(item["root"])
            self.assertIsNone(item["numerical_closed"])
            self.assertIsNone(item["numerical_power_closed"])
            self.assertIsNone(item["overall_engineering_feasibility"])
            self.assertEqual(item["overall_engineering_feasibility_status"], "NOT_CHECKED")
            self.assertTrue(item["outer_residual_check"]["endpoint_same_sign"])
            self.assertNotEqual(item["status"], "NON_PHYSICAL_STATE")
        self.assertEqual(failed["C_PI3_W0.5"]["partial_gas_generator_candidate_ids"], [])

    def test_reference_flow_sized_roots(self):
        dataset = ROOT / "reference" / "architecture_stage_smoke" / "2026-11-11-flow-sized"
        if not dataset.is_dir():
            self.skipTest("saved flow-sized reference dataset is absent")
        report = audit.audit_dataset(dataset)
        by_case = {root["case"]: root for root in report["roots"]}
        for name in ("A_MAIN_3", "B_MAIN_3"):
            root = by_case[name]
            self.assertTrue(root["known_constraint_violation"])
            self.assertTrue(any(stage["checks"]["sigma_project_290MPa"]["status"] == "FAIL"
                                for stage in root["groups"]["main"]["stages"]))
            self.assertFalse(root["source_domain_violation"])
            self.assertEqual(root["compressor_power"]["status"], "PASS")
            self.assertIsNone(root["overall_engineering_feasibility"])
        c = by_case["C_GG3_FPT1"]
        fpt = c["groups"]["free_power"]["stages"][0]
        self.assertTrue(c["known_constraint_violation"])
        self.assertTrue(c["source_domain_violation"])
        self.assertEqual(fpt["checks"]["alpha2_source_page_38"]["status"], "FAIL")
        self.assertEqual(fpt["checks"]["sigma_project_290MPa"]["status"], "FAIL")
        self.assertEqual(fpt["checks"]["reaction_table_lD_domain"]["status"], "OUTSIDE_SOURCE_DOMAIN")
        self.assertGreater(fpt["metrics"]["alpha2_deg"], 90.0)
        self.assertGreater(fpt["metrics"]["sigma_MPa"], 290.0)
        self.assertGreater(fpt["metrics"]["lD"], 1.0 / 3.0)
        self.assertEqual(c["compressor_power"]["status"], "PASS")
        self.assertEqual(c["state_continuity"]["status"], "PASS")
        self.assertEqual(c["numerical_closed"], True)
        self.assertEqual(report["overall_engineering_feasibility_status"], "NOT_CHECKED")


if __name__ == "__main__":
    unittest.main()
