"""Tests for the independent architecture energy/property audit.

These are software and numerical-diagnostic tests, not a physical validation.
They use analytical hand cases for the transcribed gas properties, check the
saved dataset's contract, and exercise the CLI output/overwrite behaviour.  The
production cycle/gas/compressor/turbine equations are not imported as an oracle.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import audit_architecture_energy as audit  # noqa: E402

from turbine_engine.dataset import read_records  # noqa: E402


DATASET = ROOT / "reference" / "architecture_smoke" / "2026-11-11-grid"


class GasPropertyTests(unittest.TestCase):
    def test_mean_cp_matches_analytical_hand_cases(self):
        # The saved compressor inlet at 288 K, dry air, is a known project value.
        self.assertAlmostEqual(
            audit.mean_cp(288.0, math.inf), 999.085192, places=6
        )

        expected_low = (
            (1.2 + 2.25 / 4.0) / 1e5 * (600.0 - 70.0) + 0.236
        ) * 4187.0
        self.assertAlmostEqual(
            audit.mean_cp(600.0, 4.0), expected_low, places=9
        )

        expected_high = (
            (2.2 + 1.25 / 4.0) / 1e5 * (1000.0 + 450.0) + 0.218
        ) * 4187.0
        self.assertAlmostEqual(
            audit.mean_cp(1000.0, 4.0), expected_high, places=9
        )

    def test_local_cp_uses_analytical_derivative_and_zero_conventions(self):
        T = 900.0
        alpha = 4.0
        cpbar = audit.mean_cp(T, alpha)
        derivative = audit.mean_cp_derivative(T, alpha)

        self.assertAlmostEqual(
            audit.local_cp(T, alpha, 0.0), cpbar + T * derivative, places=12
        )
        self.assertAlmostEqual(
            audit.local_cp(T, alpha, 273.15),
            cpbar + (T - 273.15) * derivative,
            places=12,
        )

    def test_enthalpy_is_cpbar_times_temperature_offset(self):
        T = 650.0
        alpha = 3.5
        cpbar = audit.mean_cp(T, alpha)
        self.assertAlmostEqual(
            audit.enthalpy(T, alpha, 0.0), cpbar * T, places=9
        )
        self.assertAlmostEqual(
            audit.enthalpy(T, alpha, 273.15),
            cpbar * (T - 273.15),
            places=9,
        )

    def test_700K_branch_is_strictly_one_sided(self):
        for alpha in (1.0, 4.0, math.inf):
            with self.subTest(alpha=audit._alpha_json(alpha)):
                left = audit.mean_cp_one_sided(700.0, alpha, "left")
                right = audit.mean_cp_one_sided(700.0, alpha, "right")
                self.assertNotEqual(left, right)
                # Values just below/at 700 follow the source branch rule.
                self.assertAlmostEqual(
                    audit.mean_cp(700.0 - 1e-9, alpha), left, places=9
                )
                self.assertAlmostEqual(audit.mean_cp(700.0, alpha), right, places=9)

                diagnostic = audit.branch_diagnostics(alpha)
                self.assertAlmostEqual(
                    diagnostic["mean_cp_jump_J_per_kg_K"], right - left, places=9
                )
                left_derivative = audit.mean_cp_derivative(699.999, alpha)
                right_derivative = audit.mean_cp_derivative(700.0, alpha)
                for label, zero in audit.CP_ZERO_CONVENTIONS:
                    local = diagnostic["local_cp"][label]
                    expected_left_local = left + (700.0 - zero) * left_derivative
                    expected_right_local = right + (700.0 - zero) * right_derivative
                    self.assertAlmostEqual(
                        local["left_local_cp_J_per_kg_K"],
                        expected_left_local,
                        places=9,
                    )
                    self.assertAlmostEqual(
                        local["right_local_cp_J_per_kg_K"],
                        expected_right_local,
                        places=9,
                    )
                    self.assertAlmostEqual(
                        local["local_cp_jump_J_per_kg_K"],
                        expected_right_local - expected_left_local,
                        places=9,
                    )
                    enthalpy = diagnostic["enthalpy_H_z"][label]
                    self.assertAlmostEqual(
                        enthalpy["enthalpy_jump_J_per_kg"],
                        (700.0 - zero) * (right - left),
                        places=9,
                    )


class FuelRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_records(DATASET)

    def test_saved_legacy_fuel_relation_closes_exactly_at_zero_offset(self):
        entry = audit.audit_legacy_fuel(
            self.records[0].payload, self.records[0].outputs
        )
        self.assertTrue(entry["available"])
        zero = entry["z_conventions"]["zero_0K"]
        self.assertAlmostEqual(
            zero["retained_residual_J_per_kg_air"], 0.0, places=6
        )

        alternative = entry["z_conventions"]["zero_273_15K"]
        self.assertIsNotNone(alternative["alternative_f"])
        self.assertLess(alternative["relative_change_vs_stored"], 0.0)

    def test_altered_saved_fuel_ratio_triggers_nonzero_residual(self):
        outputs = copy.deepcopy(self.records[0].outputs)
        outputs["combustor"]["fuel_air_ratio"] *= 1.01
        entry = audit.audit_legacy_fuel(self.records[0].payload, outputs)
        residual = entry["z_conventions"]["zero_0K"][
            "retained_residual_J_per_kg_air"
        ]
        self.assertGreater(abs(residual), 1000.0)

    def test_alpha_and_heat_identity_are_reported_independently(self):
        entry = audit.audit_legacy_fuel(
            self.records[0].payload, self.records[0].outputs
        )
        alpha = audit.audit_alpha(
            self.records[0].payload, self.records[0].outputs
        )
        heat = audit.audit_heat(
            self.records[0].payload, self.records[0].outputs
        )
        self.assertTrue(entry["available"])
        self.assertTrue(alpha["available"])
        self.assertTrue(heat["available"])
        self.assertGreater(alpha["alpha_fuel_based_1_over_f_l0"], 0.0)
        self.assertGreater(alpha["alpha_empirical_polynomial"], 0.0)
        self.assertLess(abs(heat["relative_residual"]), 1e-12)

    def test_combustor_residual_reports_both_zero_and_alpha_bases(self):
        entry = audit.audit_combustor_energy_residual(
            self.records[0].payload, self.records[0].outputs
        )
        self.assertTrue(entry["available"])
        self.assertEqual(len(entry["cases"]), 4)
        bases = {(case["zero_convention"], case["alpha_basis"]) for case in entry["cases"]}
        self.assertEqual(len(bases), 4)


class ComponentWorkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_records(DATASET)

    def test_turbine_positive_drop_constant_cp_sign_regression(self):
        # Legacy free-power point: two turbines, same inherited inlet cp/k.
        record = self.records[8]
        outputs = record.outputs
        alpha = outputs["combustor"]["alpha"]
        basis = outputs["combustor"]["alpha_basis"]
        turbines = audit.audit_turbine_components(
            record.payload, outputs, alpha, basis
        )
        self.assertEqual(len(turbines), 2)

        for turbine in turbines:
            cp_in = turbine["inlet_cp_J_per_kg_K"]
            work = turbine["specific_work_stored_J_per_kg"]
            with mock.patch.object(audit, "mean_cp", return_value=cp_in):
                patched = audit.audit_turbine_components(
                    record.payload, outputs, alpha, basis
                )
            entry = patched[turbine["index"]]
            self.assertIn("H_in - H_out", entry["comparison_direction"])
            for label, _zero in audit.CP_ZERO_CONVENTIONS:
                block = entry["endpoint_sensible_enthalpy_drop"][label]
                drop = block["endpoint_sensible_enthalpy_drop_J_per_kg"]
                residual = block["stored_work_minus_endpoint_drop_J_per_kg"]
                # For constant cp, stored positive work equals the positive
                # H_in - H_out drop; the reverse sign would not close.
                self.assertGreater(drop, 0.0)
                self.assertAlmostEqual(drop, work, places=6)
                self.assertAlmostEqual(residual, 0.0, places=6)


class ExternalAndFailureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = read_records(DATASET)

    def test_external_unknowns_are_preserved_as_null_with_reason(self):
        external = [
            record
            for record in self.records
            if audit._closure_kind(record.payload) == "external"
        ]
        self.assertEqual(len(external), 3)
        record = external[0]
        entry = audit.audit_record(record, 0)
        heat = entry["heat_identity"]
        alpha = entry["alpha_analysis"]
        combustor = entry["combustor_energy_residual"]
        self.assertFalse(heat["available"])
        self.assertIsNone(heat["stored_heat_input_W"])
        self.assertIsNone(heat["gross_chemical_power_W"])
        self.assertIn("not inferred", heat["reason"])
        self.assertFalse(alpha["available"])
        self.assertIsNone(alpha["stored_alpha"])
        self.assertIn("not inferred", alpha["reason"])
        self.assertFalse(combustor["available"])
        self.assertEqual(combustor["cases"], [])

    def test_failed_point_is_retained_with_skip_reason(self):
        failures = [
            record
            for record in self.records
            if record.calculation_status.value != "SUCCESS"
        ]
        self.assertEqual(len(failures), 1)
        entry = audit.audit_record(failures[0], 12)
        self.assertEqual(entry["calculation_status"], "NON_PHYSICAL_STATE")
        self.assertEqual(entry["failure"]["stage"], "architecture_cycle_physical")
        self.assertIn("not SUCCESS", entry["skipped"]["reason"])
        self.assertTrue(entry["failure"]["reason"])
        self.assertIn("input_payload", entry)

    def test_external_fpt_inherited_origin_is_reported_but_cpbar_is_null(self):
        record = [
            item
            for item in self.records
            if item.architecture == "FREE_POWER_TURBINE"
            and audit._closure_kind(item.payload) == "external"
        ][0]
        entry = audit.audit_record(record, 11)
        fpt = entry["fpt_inherited_properties"]
        self.assertTrue(fpt["available"])
        self.assertIsNotNone(fpt["properties_originated_at"])
        self.assertGreater(
            fpt["properties_originated_at"]["T_K"], fpt["fpt_inlet_T_K"]
        )
        self.assertIsNone(fpt["independent_cpbar_at_actual_inlet_J_per_kg_K"])
        self.assertIn(
            "not inferred", fpt["relative_cp_discrepancy_unavailable_reason"]
        )


class SavedDatasetReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit.build_report(DATASET)

    def test_counts_and_retention(self):
        summary = self.report["summary"]
        self.assertEqual(summary["record_count"], 13)
        self.assertEqual(summary["counts"]["by_status"]["SUCCESS"], 12)
        self.assertEqual(
            summary["counts"]["by_status"]["NON_PHYSICAL_STATE"], 1
        )
        self.assertEqual(summary["counts"]["by_closure"]["legacy"], 10)
        self.assertEqual(summary["counts"]["by_closure"]["external"], 3)
        self.assertEqual(len(self.report["records"]), 13)
        self.assertEqual(
            [entry["index"] for entry in self.report["records"]],
            list(range(13)),
        )

    def test_fpt_inherited_difference_is_nonzero_for_legacy_records(self):
        fpt_entries = [
            entry["fpt_inherited_properties"]
            for entry in self.report["records"]
            if entry["architecture"] == "FREE_POWER_TURBINE"
            and entry["combustor_closure"] == "legacy"
        ]
        self.assertEqual(len(fpt_entries), 3)
        for entry in fpt_entries:
            self.assertTrue(entry["available"])
            origin = entry["properties_originated_at"]
            self.assertIsNotNone(origin)
            self.assertGreater(origin["T_K"], entry["fpt_inlet_T_K"])
            self.assertGreater(
                entry["relative_cp_discrepancy_inherited_minus_independent"],
                0.0,
            )
            self.assertGreater(
                entry["independent_cpbar_at_actual_inlet_J_per_kg_K"],
                0.0,
            )

    def test_report_is_strict_finite_json_and_declares_limitations(self):
        text = json.dumps(self.report, allow_nan=False, sort_keys=True)
        self.assertIn('"NOT_PHYSICALLY_VALIDATED"', text)
        self.assertEqual(
            self.report["validation_status"], "NOT_PHYSICALLY_VALIDATED"
        )
        self.assertIn("does not scan", self.report["scan_note"])
        paths = {item["path"] for item in self.report["source_references"]}
        self.assertIn("turbine_engine/cycle.py", paths)
        self.assertIn("turbine_engine/gas.py", paths)
        self.assertIn("turbine_engine/architecture_cycle.py", paths)
        self.assertEqual(self.report["non_finite_replacements"], [])

    def test_branch_diagnostics_are_present_and_one_sided(self):
        branches = self.report["gas_property_diagnostics"]["branch_700K"]
        self.assertGreaterEqual(len(branches), 2)
        for branch in branches:
            self.assertNotEqual(
                branch["mean_cp_left_J_per_kg_K"],
                branch["mean_cp_right_J_per_kg_K"],
            )
            for label in ("zero_0K", "zero_273_15K"):
                local = branch["local_cp"][label]
                self.assertNotEqual(
                    local["left_local_cp_J_per_kg_K"],
                    local["right_local_cp_J_per_kg_K"],
                )

    def test_retained_algebraic_residuals_are_finite(self):
        residuals = self.report["summary"]["retained_algebraic_residuals"]
        for name, stats in residuals.items():
            with self.subTest(name=name):
                self.assertGreater(stats["count"], 0)
                self.assertIsNotNone(stats["max_abs"])
                self.assertTrue(math.isfinite(stats["max_abs"]))
                self.assertIn("unit", stats)

    def test_summary_has_no_mixed_unit_global_maximum(self):
        summary = self.report["summary"]
        self.assertNotIn("max_abs_algebraic_residual", summary)
        residuals = summary["retained_algebraic_residuals"]
        units = {stats["unit"] for stats in residuals.values()}
        self.assertGreaterEqual(len(units), 2)
        for name, stats in residuals.items():
            with self.subTest(name=name):
                self.assertIn("unit", stats)
                self.assertNotIn("units", stats)

    def test_turbine_endpoint_drops_are_positive_where_alpha_is_known(self):
        for entry in self.report["records"]:
            if entry.get("skipped") is not None:
                continue
            for turbine in entry["components"]["turbines"]:
                for label, _zero in audit.CP_ZERO_CONVENTIONS:
                    block = turbine["endpoint_sensible_enthalpy_drop"][label]
                    drop = block["endpoint_sensible_enthalpy_drop_J_per_kg"]
                    if drop is not None:
                        self.assertGreater(drop, 0.0)


class CliTests(unittest.TestCase):
    def test_cli_writes_new_file_and_refuses_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            first = audit.main(
                [
                    "--dataset",
                    str(DATASET),
                    "--output",
                    str(output),
                    "--quiet",
                ]
            )
            self.assertEqual(first, 0)
            self.assertTrue(output.is_file())
            original = output.read_text(encoding="utf-8")
            report = json.loads(original)
            self.assertEqual(report["summary"]["record_count"], 13)

            second = audit.main(
                [
                    "--dataset",
                    str(DATASET),
                    "--output",
                    str(output),
                    "--quiet",
                ]
            )
            self.assertEqual(second, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_cli_missing_dataset_is_nonzero_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"
            result = audit.main(
                [
                    "--dataset",
                    str(Path(directory) / "missing-dataset"),
                    "--output",
                    str(output),
                    "--quiet",
                ]
            )
            self.assertEqual(result, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
