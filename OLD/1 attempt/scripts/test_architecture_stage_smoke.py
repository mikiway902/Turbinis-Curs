"""Focused tests for the bounded architecture-stage meanline smoke."""

from __future__ import annotations

import io
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

import architecture_stage_smoke as smoke  # noqa: E402
from turbine_engine.baseline import estimate_stage_count  # noqa: E402
from turbine_engine.dataset import (  # noqa: E402
    DatasetError,
    load_manifest,
    read_records,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NUMERICAL_FAILURE,
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
)
from turbine_engine.raw import CalculationStatus, EvaluationOutcome  # noqa: E402


def _fake_success_outputs() -> dict:
    return {
        "boundary": {
            "last_stage_static_outlet_error_Pa": 0.0,
            "last_stage_total_minus_aggregate_total_Pa": 0.0,
        },
        "power_comparison": {
            "difference_stage_minus_aggregate_mechanical_W": 0.0,
        },
        "stage_count_estimate": {
            "status": "NOT_CHECKED",
            "reason": "interrupt_resume_fixture",
            "automatic_selection": "NONE",
        },
    }


class ArchitectureStageSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="architecture-stage-smoke-"
        )
        cls.destination = Path(cls.temporary.name) / "dataset"
        cls.summary = smoke.run_plan(cls.destination)
        cls.records = read_records(cls.destination)
        cls.manifest = load_manifest(cls.destination)
        cls.constants = cls.manifest["run_config"]["stage_constants"]
        cls.model_digest = cls.manifest["run_config"]["model_digest"]
        cls.model = smoke._get_model()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temporary.name, ignore_errors=True)

    def _record(self, assembly: str, count: int):
        for record in self.records:
            payload = record.payload
            if (
                payload["assembly"] == assembly
                and payload["stage_count"] == count
            ):
                return record
        self.fail(f"missing record {assembly}/{count}")

    def test_fixed_plan_has_four_assemblies_and_three_explicit_counts(self):
        points = smoke.plan_points()
        self.assertEqual(len(points), 12)
        for assembly in smoke.ASSEMBLY_ORDER:
            selected = [point for point in points if point.assembly == assembly]
            self.assertEqual([p.stage_count for p in selected], [1, 2, 3])
            architecture, role, index = smoke.ASSEMBLY_SPECS[assembly]
            for point in selected:
                self.assertIs(point.architecture, architecture)
                self.assertEqual(point.turbine_role, role)
                self.assertEqual(point.turbine_index, index)

        for payload in smoke.plan_payloads():
            count = payload["stage_count"]
            controls = payload["stage_controls"]
            self.assertEqual(len(controls["alpha1_deg"]), count)
            self.assertEqual(len(controls["kappa"]), count)
            self.assertEqual(len(controls["pi_weights"]), count)
            self.assertEqual(controls["pi_weights"][-1], 1.0)
            self.assertTrue(controls["boundary_is_static"])
            self.assertEqual(
                payload["architecture_cycle"]["gas_property_basis"],
                "textbook_true_cp",
            )

        expected_ratios = {
            "A_MAIN": [math.sqrt(11.0), math.sqrt(11.0)],
            "B_MAIN": [11.0],
            "C_GG": [8.0],
            "C_FPT": [8.0],
        }
        for payload in smoke.plan_payloads():
            self.assertEqual(
                payload["architecture_cycle"]["compressor_pressure_ratios"],
                expected_ratios[payload["assembly"]],
            )

    def test_static_boundary_and_method_labels(self):
        record = self._record("A_MAIN", 3)
        self.assertIs(record.calculation_status, CalculationStatus.SUCCESS)
        outputs = record.outputs
        boundary = outputs["boundary"]
        self.assertTrue(boundary["boundary_is_static"])
        self.assertFalse(boundary["mach_solved_by_meanline"])
        self.assertEqual(boundary["reference_exit_mach"], 0.3379)
        self.assertLess(
            boundary["derived_static_outlet_pressure_Pa"],
            boundary["aggregate_outlet_total_pressure_Pa"],
        )
        self.assertAlmostEqual(
            boundary["last_stage_static_outlet_error_Pa"], 0.0, delta=1e-6
        )
        self.assertEqual(outputs["coupling_label"], smoke.COUPLING_LABEL)
        self.assertEqual(outputs["stage_count_selection"], "NONE")
        self.assertEqual(
            self.manifest["run_config"]["assumptions"]["boundary"][
                "mach_solved_by_meanline"
            ],
            False,
        )

    def test_all_attempts_stored_and_three_stage_success_retained(self):
        self.assertEqual(self.manifest["record_count"], 12)
        self.assertTrue(self.manifest["complete"])
        self.assertEqual(self.summary["run"]["points_attempted"], 12)
        statuses = {
            record.calculation_status.value for record in self.records
        }
        self.assertTrue(statuses.issubset({s.value for s in CalculationStatus}))
        # Observed raw-unbounded result after this policy change: all 12
        # attempts solve.  The legacy reference run below keeps 5 typed stalls.
        self.assertEqual(statuses, {CalculationStatus.SUCCESS.value})
        three_stage_success = [
            record
            for record in self.records
            if record.payload["stage_count"] == 3
            and record.calculation_status is CalculationStatus.SUCCESS
        ]
        self.assertEqual(len(three_stage_success), 4)
        for record in self.records:
            json.dumps(record.outputs, ensure_ascii=False, allow_nan=False)
            if record.calculation_status is not CalculationStatus.SUCCESS:
                self.assertTrue(record.failure_stage)
                self.assertTrue(record.failure_reason)

    def test_payload_config_summary_and_records_carry_policy(self):
        for payload in smoke.plan_payloads():
            self.assertEqual(
                payload["stage_controls"]["pressure_ratio_policy"],
                "raw_unbounded",
            )
        self.assertEqual(
            self.manifest["run_config"]["pressure_ratio_policy"],
            "raw_unbounded",
        )
        self.assertEqual(
            self.summary["experiment"]["pressure_ratio_policy"],
            "raw_unbounded",
        )
        for record in self.records:
            self.assertEqual(
                record.payload["stage_controls"]["pressure_ratio_policy"],
                "raw_unbounded",
            )
            self.assertEqual(
                record.diagnostics["pressure_ratio_policy"], "raw_unbounded"
            )

    def test_posthoc_pi_acceptance_profile_alone_decides_pass(self):
        high = self._record("A_MAIN", 1)
        self.assertIs(high.calculation_status, CalculationStatus.SUCCESS)
        acceptance = high.outputs["posthoc_pi_acceptance"]
        self.assertEqual(acceptance["profile"], "PROMPT_STAGE_PI_1_8_2_4")
        self.assertIn("NOT_BOOK_FUNDAMENTAL", acceptance["basis"])
        self.assertEqual(acceptance["bounds"], [1.8, 2.4])
        self.assertFalse(acceptance["pass"])
        stage = acceptance["stages"][0]
        self.assertEqual(stage["min"], 1.8)
        self.assertEqual(stage["max"], 2.4)
        self.assertEqual(stage["min_pi_stage"], 1.8)
        self.assertEqual(stage["max_pi_stage"], 2.4)
        self.assertEqual(stage["actual"], stage["actual_pi_stage"])
        self.assertGreater(stage["actual"], 2.4)
        self.assertEqual(stage["status"], "FAIL")
        self.assertAlmostEqual(stage["lower_margin"], stage["actual_pi_stage"] - 1.8)
        self.assertAlmostEqual(stage["upper_margin"], 2.4 - stage["actual_pi_stage"])
        self.assertLess(stage["upper_margin"], 0.0)
        self.assertIsNone(acceptance["overall_engineering_feasibility"])
        self.assertEqual(
            acceptance["overall_engineering_feasibility_status"],
            "NOT_CHECKED",
        )
        self.assertEqual(
            acceptance["overall_engineering_feasibility_reason"],
            "strength_verification_missing",
        )
        self.assertIsNone(high.outputs["overall_engineering_feasibility"])
        self.assertEqual(
            high.outputs["overall_engineering_feasibility_status"], "NOT_CHECKED"
        )
        summary_entry = next(
            point
            for point in self.summary["points"]
            if point["assembly"] == "A_MAIN"
            and point["stage_count_requested"] == 1
        )
        summary_acceptance = summary_entry["metrics"]["posthoc_pi_acceptance"]
        self.assertFalse(summary_acceptance["pass"])
        summary_stage = summary_acceptance["stages"][0]
        self.assertEqual(
            summary_stage["actual_pi_stage"], stage["actual_pi_stage"]
        )
        self.assertEqual(
            summary_stage["lower_margin"], stage["lower_margin"]
        )
        self.assertEqual(
            summary_stage["upper_margin"], stage["upper_margin"]
        )
        self.assertEqual(summary_stage["status"], "FAIL")

        within = self._record("A_MAIN", 3).outputs["posthoc_pi_acceptance"]
        self.assertTrue(within["pass"])
        for stage in within["stages"]:
            self.assertEqual(stage["min"], 1.8)
            self.assertEqual(stage["max"], 2.4)
            self.assertEqual(stage["min_pi_stage"], 1.8)
            self.assertEqual(stage["max_pi_stage"], 2.4)
            self.assertEqual(stage["actual"], stage["actual_pi_stage"])
            self.assertGreaterEqual(stage["lower_margin"], 0.0)
            self.assertGreaterEqual(stage["upper_margin"], 0.0)
            self.assertEqual(stage["status"], "PASS")
        self.assertIsNone(within["overall_engineering_feasibility"])
        self.assertEqual(
            within["overall_engineering_feasibility_status"], "NOT_CHECKED"
        )

    def test_legacy_clipped_reference_policy_keeps_five_typed_stalls(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="architecture-stage-legacy-policy-"
        )
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        summary = smoke.run_plan(
            destination,
            pressure_ratio_policy="legacy_clipped",
            reproduce=False,
            constants=self.constants,
            model_digest=self.model_digest,
        )
        records = read_records(destination)
        self.assertEqual(
            summary["status_counts"],
            {"NUMERICAL_FAILURE": 5, "SUCCESS": 7},
        )
        self.assertEqual(
            summary["experiment"]["pressure_ratio_policy"], "legacy_clipped"
        )
        failures = [
            record
            for record in records
            if record.calculation_status is not CalculationStatus.SUCCESS
        ]
        self.assertEqual(len(failures), 5)
        for record in failures:
            self.assertEqual(record.failure_reason, "secant_denominator_stalled")
            self.assertEqual(
                record.diagnostics["pressure_ratio_policy"], "legacy_clipped"
            )
        failed_entries = [
            point
            for point in summary["points"]
            if point["calculation_status"] != "SUCCESS"
        ]
        self.assertEqual(len(failed_entries), 5)
        for point in failed_entries:
            acceptance = point["metrics"]["posthoc_pi_acceptance"]
            self.assertEqual(acceptance["status"], "NOT_CHECKED")
            self.assertIsNone(acceptance["pass"])
            self.assertIsNone(
                point["metrics"]["overall_engineering_feasibility"]
            )
            self.assertEqual(
                point["metrics"]["overall_engineering_feasibility_status"],
                "NOT_CHECKED",
            )

    def test_resume_rejects_cross_policy_with_same_model_digest(self):
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                self.destination,
                resume=True,
                reproduce=False,
                pressure_ratio_policy="legacy_clipped",
                constants=self.constants,
                model_digest=self.model_digest,
                evaluator=lambda payload: EvaluationOutcome.success(
                    outputs=_fake_success_outputs()
                ),
            )

    def test_stage_count_estimate_matches_shared_formula(self):
        record = self._record("A_MAIN", 3)
        estimate = record.outputs["stage_count_estimate"]
        self.assertEqual(estimate["status"], "COMPUTED")
        self.assertEqual(estimate["requested_stage_count"], 3)
        self.assertEqual(estimate["automatic_selection"], "NONE")
        self.assertFalse(estimate["estimate_rounded"])
        expected = estimate_stage_count(
            estimate["c0_total_m_s"],
            estimate["u_mean_m_s"],
            estimate["y0"],
            estimate["recovered_exit_velocity_m_s"],
        )
        self.assertEqual(estimate["z_estimate"], expected)

    def test_reproduce_all_stored_attempts(self):
        self.assertEqual(self.summary["reproduce"]["checked"], 12)
        self.assertTrue(self.summary["reproduce"]["matches"])
        self.assertEqual(self.summary["reproduce"]["differences"], [])
        self.assertEqual(
            self.manifest["run_config"]["model_digest"], self.model_digest
        )
        self.assertEqual(len(self.model_digest), 64)
        int(self.model_digest, 16)

    def test_resume_appends_zero_duplicates_and_keeps_journal(self):
        before_hash = self.manifest["journal_sha256"]
        before_count = self.manifest["record_count"]
        summary = smoke.run_plan(
            self.destination,
            resume=True,
            reproduce=False,
            constants=self.constants,
            model_digest=self.model_digest,
            evaluator=None,
            model=self.model,
        )
        self.assertEqual(summary["run"]["records_appended_this_invocation"], 0)
        manifest = load_manifest(self.destination)
        self.assertEqual(manifest["record_count"], before_count)
        self.assertEqual(manifest["journal_sha256"], before_hash)
        self.assertTrue(manifest["complete"])

    def test_resume_rejects_changed_model_digest(self):
        with self.assertRaises(DatasetError):
            smoke.run_plan(
                self.destination,
                resume=True,
                reproduce=False,
                constants=self.constants,
                model_digest="0" * 64,
                evaluator=lambda payload: EvaluationOutcome.success(
                    outputs=_fake_success_outputs()
                ),
            )

    def test_model_digest_covers_previously_missing_core_dependencies(self):
        baseline = smoke._model_digest()
        sources = self.manifest["run_config"]["model_digest_sources"]
        for relative in (
            "turbine_engine/compressor.py",
            "turbine_engine/turbine.py",
            "turbine_engine/shaft.py",
            "turbine_engine/reaction.py",
        ):
            with self.subTest(dependency=relative):
                self.assertIn(relative, smoke.MODEL_SOURCE_FILES)
                self.assertIn(relative, sources)
                target = (smoke.ROOT / relative).resolve()
                real_open = open

                def patched(file, *args, **kwargs):
                    if Path(file).resolve() == target:
                        return io.BytesIO(b"correction-cycle-2-mock")
                    return real_open(file, *args, **kwargs)

                with mock.patch("builtins.open", side_effect=patched):
                    altered = smoke._model_digest()
                self.assertNotEqual(altered, baseline)
                with self.assertRaises(DatasetError):
                    smoke.run_plan(
                        self.destination,
                        resume=True,
                        reproduce=False,
                        constants=self.constants,
                        model_digest=altered,
                        evaluator=lambda payload: EvaluationOutcome.success(
                            outputs=_fake_success_outputs()
                        ),
                    )

    def test_typed_numerical_and_physical_failures_are_preserved(self):
        payload = smoke.plan_payloads()[0]
        numerical = NumericalConvergenceError(
            "test stall",
            stage="orchestration.test",
            reason="test_stall",
            last_valid_state={"y": 2.0},
        )
        with mock.patch.object(
            smoke, "_run_stage_meanline", side_effect=numerical
        ):
            outcome = smoke.evaluate_stage_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(outcome.failure_stage, "orchestration.test")
        self.assertEqual(outcome.failure_reason, "test_stall")
        self.assertEqual(
            outcome.last_valid_state["solver_last_valid_state"]["y"], 2.0
        )
        self.assertEqual(
            outcome.diagnostics["pressure_ratio_policy"], "raw_unbounded"
        )

        with mock.patch.object(
            smoke,
            "_run_stage_meanline",
            side_effect=smoke.StageNoSolution("geometry domain"),
        ):
            physical = smoke.evaluate_stage_payload(payload)
        self.assertIs(physical.status, CalculationStatus.NON_PHYSICAL_STATE)
        self.assertEqual(physical.failure_stage, "stage_geometry_domain")
        self.assertEqual(physical.failure_reason, "geometry domain")

    def test_nonfinite_stage_output_maps_to_numerical_partial(self):
        payload = smoke.plan_payloads()[0]
        exc = smoke.StageNumericalFailure("stage.L_t", math.nan)
        exc.partial = {"stages": [{"partial": True}]}
        with mock.patch.object(
            smoke, "_run_stage_meanline", return_value={}
        ), mock.patch.object(smoke, "_build_outputs", side_effect=exc):
            outcome = smoke.evaluate_stage_payload(payload)
        self.assertIs(outcome.status, CalculationStatus.NUMERICAL_FAILURE)
        self.assertEqual(outcome.failure_stage, "stage_numerical")
        self.assertEqual(
            outcome.outputs["stages"], [{"partial": True}]
        )

    def test_interrupt_then_resume_uses_fixed_plan_without_duplicates(self):
        temporary = tempfile.TemporaryDirectory(
            prefix="architecture-stage-interrupt-"
        )
        self.addCleanup(temporary.cleanup)
        destination = Path(temporary.name) / "dataset"
        calls = {"count": 0}

        def interrupting(payload):
            calls["count"] += 1
            if calls["count"] == 3:
                raise KeyboardInterrupt()
            return EvaluationOutcome.success(
                outputs=_fake_success_outputs()
            )

        with self.assertRaises(KeyboardInterrupt):
            smoke.run_plan(
                destination,
                evaluator=interrupting,
                constants=self.constants,
                model_digest=self.model_digest,
                reproduce=False,
            )
        incomplete = load_manifest(destination)
        self.assertFalse(incomplete["complete"])
        self.assertEqual(incomplete["record_count"], 2)
        self.assertFalse((destination / smoke.SUMMARY_NAME).exists())

        def succeeding(payload):
            return EvaluationOutcome.success(outputs=_fake_success_outputs())

        summary = smoke.run_plan(
            destination,
            resume=True,
            evaluator=succeeding,
            constants=self.constants,
            model_digest=self.model_digest,
            reproduce=False,
        )
        complete = load_manifest(destination)
        self.assertTrue(complete["complete"])
        self.assertEqual(complete["record_count"], 12)
        self.assertEqual(summary["run"]["records_appended_this_invocation"], 10)
        records = read_records(destination)
        self.assertEqual(len(records), 12)
        self.assertEqual(len({record.candidate_id for record in records}), 12)


if __name__ == "__main__":
    unittest.main()
