"""Tests for the bounded coarse ``pi_k`` experiment.

These are software/dataset tests, not a physical validation.  They exercise the
fixed plan products and coverage, deterministic config integrity, no-overwrite
and metadata-only resume behaviour, an interrupted/resumed dataset, retention
of failed attempts, post-hoc ``NOT_CHECKED`` rho, strict reproduction, and
adjacent-pi slope breaks.  No optimizer, GPU, calibration, or production sweep
is invoked.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(ROOT))

import architecture_coarse as coarse  # noqa: E402
import architecture_smoke as smoke  # noqa: E402

from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_raw import (  # noqa: E402
    ARCHITECTURE_RAW_MODEL_VARIANT,
)
from turbine_engine.dataset import (  # noqa: E402
    DatasetError,
    JOURNAL_NAME,
    load_manifest,
    read_records,
)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus,
    EvaluationOutcome,
    canonical_hash,
)


def assert_finite_json(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value), value)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(test, item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_json(test, item)


class PlanTests(unittest.TestCase):
    def test_fixed_plan_has_55_points_and_expected_coverage(self):
        points = coarse.plan_points()
        self.assertEqual(len(points), 55)
        self.assertEqual(coarse.PLAN_POINT_COUNT, 55)
        self.assertEqual([point.index for point in points], list(range(55)))

        counts = Counter(point.architecture for point in points)
        self.assertEqual(counts[Architecture.SINGLE_SHAFT_SINGLE_CC], 11)
        self.assertEqual(counts[Architecture.FREE_POWER_TURBINE], 11)
        self.assertEqual(counts[Architecture.SINGLE_SHAFT_DUAL_CC], 33)

        for pi in range(3, 14):
            for architecture in (
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                Architecture.FREE_POWER_TURBINE,
            ):
                matching = [
                    point
                    for point in points
                    if point.architecture is architecture and point.pi == pi
                ]
                self.assertEqual(len(matching), 1)
                self.assertIsNone(matching[0].split_weight)
                self.assertEqual(matching[0].pressure_ratios, (float(pi),))

            dual = [
                point
                for point in points
                if point.architecture is Architecture.SINGLE_SHAFT_DUAL_CC
                and point.pi == pi
            ]
            self.assertEqual(
                {point.split_weight for point in dual},
                set(coarse.SPLIT_WEIGHTS),
            )
            for point in dual:
                self.assertEqual(len(point.pressure_ratios), 2)
                first, second = point.pressure_ratios
                self.assertGreater(first, 0.0)
                self.assertGreater(second, 0.0)
                self.assertAlmostEqual(
                    first, pi ** point.split_weight, places=12
                )
                self.assertAlmostEqual(
                    second,
                    pi ** (1.0 - point.split_weight),
                    places=12,
                )
                self.assertAlmostEqual(first * second, pi, places=12)

    def test_plan_payloads_are_strict_and_plan_size_is_capped(self):
        payloads = coarse.plan_payloads()
        self.assertEqual(len(payloads), 55)
        self.assertEqual(
            [payload["architecture"] for payload in payloads].count(
                Architecture.SINGLE_SHAFT_SINGLE_CC.value
            ),
            11,
        )
        for payload in payloads:
            self.assertEqual(
                payload["schema_version"], "architecture-cycle-input-v1"
            )
            self.assertEqual(
                payload["combustor"]["kind"], "legacy"
            )
            self.assertEqual(
                payload["combustor"]["outlet_total_temperature"],
                1250.0,
            )
            self.assertEqual(
                payload["turbine_outlet_total_pressure"], 150_000.0
            )
            json.dumps(payload, allow_nan=False)

        coarse.validate_plan_size(payloads)
        with self.assertRaises(coarse.ArchitectureCoarseError):
            coarse.validate_plan_size([])
        with self.assertRaises(coarse.ArchitectureCoarseError):
            coarse.validate_plan_size(
                [{}] * (coarse.MAX_PLAN_POINTS + 1)
            )

    def test_run_config_is_deterministic_and_has_static_keys_only(self):
        payloads = coarse.plan_payloads()
        config = coarse.make_run_config(payloads)
        again = coarse.make_run_config(payloads)
        self.assertEqual(config, again)
        self.assertEqual(
            config["payload_plan_sha256"], canonical_hash(payloads)
        )
        self.assertEqual(config["payload_plan_count"], 55)
        self.assertEqual(
            config["model_variant"], ARCHITECTURE_RAW_MODEL_VARIANT
        )
        self.assertEqual(
            config["evaluator_version"], coarse.EVALUATOR_VERSION
        )
        self.assertEqual(config["backend"], "reference_python_float64")
        self.assertEqual(config["config_version"], coarse.CONFIG_VERSION)
        self.assertEqual(
            config["assumptions"]["inlet_total_temperature_K"], 288.0
        )
        self.assertEqual(
            config["assumptions"]["inlet_total_pressure_Pa"], 101_300.0
        )
        self.assertEqual(
            config["assumptions"]["target_gearbox_input_power_W"],
            110_000.0,
        )
        self.assertEqual(
            config["model_variants"]["architecture_raw"],
            ARCHITECTURE_RAW_MODEL_VARIANT,
        )
        for forbidden in (
            "generated_at",
            "generated_at_utc",
            "timestamp",
            "git_commit",
            "git_dirty",
            "created_at",
            "host",
        ):
            self.assertNotIn(forbidden, config)

    def test_smoke_fixture_points_are_reused_unchanged(self):
        points = coarse.plan_points()
        single = next(
            point
            for point in points
            if point.architecture is Architecture.SINGLE_SHAFT_SINGLE_CC
            and point.pi == 11
        )
        free_power = next(
            point
            for point in points
            if point.architecture is Architecture.FREE_POWER_TURBINE
            and point.pi == 8
        )
        self.assertEqual(
            coarse.payload_for(single),
            smoke.input_to_payload(
                smoke.request_for(
                    Architecture.SINGLE_SHAFT_SINGLE_CC,
                    ratios=(11.0,),
                )
            ),
        )
        self.assertEqual(
            coarse.payload_for(free_power),
            smoke.input_to_payload(
                smoke.request_for(
                    Architecture.FREE_POWER_TURBINE,
                    ratios=(8.0,),
                )
            ),
        )


class DatasetRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_no_silent_overwrite_and_metadata_only_resume(self):
        path = self.root / "dataset"
        first = coarse.run_coarse(path)
        self.assertEqual(first["run"]["record_count"], 55)
        self.assertTrue(first["run"]["complete"])
        self.assertTrue((path / coarse.SUMMARY_NAME).is_file())
        journal = path / JOURNAL_NAME
        before = journal.read_bytes()

        with self.assertRaises(coarse.ArchitectureCoarseError):
            coarse.run_coarse(path)
        self.assertEqual(journal.read_bytes(), before)

        summary_path = path / coarse.SUMMARY_NAME
        changed = json.loads(summary_path.read_text(encoding="utf-8"))
        changed["generated_at_utc"] = "2000-01-01T00:00:00+00:00"
        changed["provenance"] = {
            "git": {
                "commit": "deadbeef",
                "dirty": True,
                "source": "git",
            },
            "runtime": {},
        }
        summary_path.write_text(
            json.dumps(changed), encoding="utf-8"
        )

        resumed = coarse.run_coarse(path, resume=True)
        self.assertEqual(resumed["run"]["record_count"], 55)
        self.assertEqual(
            resumed["run"]["records_appended_this_invocation"], 0
        )
        self.assertEqual(journal.read_bytes(), before)
        self.assertEqual(len(read_records(path)), 55)
        self.assertTrue(load_manifest(path)["complete"])

    def test_resume_with_changed_plan_is_rejected_without_duplicates(self):
        path = self.root / "dataset"
        coarse.run_coarse(path)
        journal = path / JOURNAL_NAME
        before = journal.read_bytes()

        points = list(coarse.plan_points())
        points[0] = replace(
            points[0], pressure_ratios=(3.25,)
        )
        with mock.patch.object(
            coarse, "plan_points", return_value=tuple(points)
        ):
            with self.assertRaises(DatasetError):
                coarse.run_coarse(path, resume=True)
        self.assertEqual(journal.read_bytes(), before)
        self.assertEqual(len(read_records(path)), 55)

    def test_interrupt_after_three_appends_and_resume_matches(self):
        uninterrupted = self.root / "uninterrupted"
        expected_summary = coarse.run_coarse(uninterrupted)
        expected_bytes = (
            uninterrupted / JOURNAL_NAME
        ).read_bytes()

        interrupted = self.root / "interrupted"
        original_append = coarse.DatasetWriter.evaluate_and_append
        calls = {"count": 0}

        def fail_fourth(self, *args, **kwargs):
            if calls["count"] >= 3:
                raise RuntimeError("simulated interruption")
            calls["count"] += 1
            return original_append(self, *args, **kwargs)

        with mock.patch.object(
            coarse.DatasetWriter,
            "evaluate_and_append",
            new=fail_fourth,
        ):
            with self.assertRaises(RuntimeError):
                coarse.run_coarse(interrupted)

        manifest = load_manifest(interrupted)
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["record_count"], 3)
        self.assertFalse((interrupted / coarse.SUMMARY_NAME).exists())

        resumed = coarse.run_coarse(interrupted, resume=True)
        self.assertEqual(resumed["run"]["record_count"], 55)
        self.assertEqual(
            resumed["run"]["records_appended_this_invocation"], 52
        )
        self.assertTrue(load_manifest(interrupted)["complete"])
        self.assertEqual(
            (interrupted / JOURNAL_NAME).read_bytes(), expected_bytes
        )
        self.assertEqual(
            [record.candidate_id for record in read_records(uninterrupted)],
            [record.candidate_id for record in read_records(interrupted)],
        )
        self.assertEqual(
            resumed["run"]["run_config_hash"],
            expected_summary["run"]["run_config_hash"],
        )


class SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def test_all_55_reproduce_and_summary_is_strict_finite_json(self):
        path = self.root / "dataset"
        summary = coarse.run_coarse(path)
        self.assertEqual(summary["reproduce"]["checked"], 55)
        self.assertTrue(summary["reproduce"]["matches"])
        self.assertEqual(summary["reproduce"]["differences"], [])
        self.assertEqual(summary["reproduce"]["rtol"], 1e-12)
        self.assertEqual(summary["reproduce"]["atol"], 1e-9)
        self.assertEqual(len(read_records(path)), 55)
        self.assertFalse(summary["physically_validated"])
        self.assertTrue(summary["illustrative_only"])
        for entry in summary["points"]:
            self.assertTrue(entry["illustrative_only"])
            self.assertFalse(entry["physically_validated"])
            self.assertEqual(
                entry["compressor_viability_gate"], "NOT_APPLIED"
            )

        text = (path / coarse.SUMMARY_NAME).read_text(encoding="utf-8")
        data = json.loads(
            text,
            parse_constant=lambda token: self.fail(
                f"non-finite JSON constant {token!r}"
            ),
        )
        assert_finite_json(self, data)

        for entry in summary["controlled_slices"]["dual_split_ranges"]:
            self.assertEqual(len(entry["splits"]), 3)
            for key in (
                "gas_mass_flow_spread",
                "fuel_mass_flow_spread",
            ):
                spread = entry[key]
                self.assertEqual(spread["count"], 3)
                self.assertTrue(math.isfinite(spread["range"]))
                self.assertAlmostEqual(
                    spread["range"], spread["max"] - spread["min"]
                )

        for series in summary["controlled_slices"][
            "adjacent_pi_slopes"
        ]:
            self.assertEqual(len(series["pairs"]), 10)
            for pair in series["pairs"]:
                self.assertEqual(
                    pair["pi_to"] - pair["pi_from"], 1
                )
                self.assertEqual(pair["status"], "computed")
                self.assertTrue(
                    math.isfinite(pair["d_gas_mass_flow_d_pi"])
                )
                self.assertTrue(
                    math.isfinite(pair["d_fuel_mass_flow_d_pi"])
                )

    def test_failures_are_retained_and_rho_stays_not_checked(self):
        path = self.root / "dataset"
        real_evaluator = coarse.evaluate_architecture_payload

        def forced_failure(payload):
            if (
                payload["architecture"]
                == Architecture.FREE_POWER_TURBINE.value
                and payload["compressor_pressure_ratios"] == [13.0]
            ):
                return EvaluationOutcome.failure(
                    CalculationStatus.NON_PHYSICAL_STATE,
                    "forced_test_failure",
                    "synthetic failure retained by the test",
                )
            return real_evaluator(payload)

        with mock.patch.object(
            coarse,
            "evaluate_architecture_payload",
            new=forced_failure,
        ):
            summary = coarse.run_coarse(path)

        records = read_records(path)
        self.assertEqual(len(records), 55)
        failures = [
            record
            for record in records
            if record.calculation_status is not CalculationStatus.SUCCESS
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0].calculation_status,
            CalculationStatus.NON_PHYSICAL_STATE,
        )
        self.assertEqual(
            failures[0].failure_stage, "forced_test_failure"
        )
        self.assertTrue(failures[0].failure_reason)

        failed_entry = next(
            entry
            for entry in summary["points"]
            if entry["candidate_id"] == failures[0].candidate_id
        )
        self.assertIsNone(failed_entry["values"])
        self.assertEqual(
            failed_entry["calculation_status"],
            CalculationStatus.NON_PHYSICAL_STATE.value,
        )
        self.assertEqual(
            failed_entry["rho"]["status"],
            "NOT_CHECKED",
        )
        self.assertIsNone(failed_entry["rho"]["feasible"])
        self.assertEqual(
            failed_entry["stage_strength"]["status"], "NOT_CHECKED"
        )
        self.assertEqual(
            summary["status_counts"][
                CalculationStatus.NON_PHYSICAL_STATE.value
            ],
            1,
        )
        self.assertEqual(
            summary["status_by_architecture"][
                Architecture.FREE_POWER_TURBINE.value
            ][CalculationStatus.NON_PHYSICAL_STATE.value],
            1,
        )
        self.assertEqual(
            summary["status_by_architecture"][
                Architecture.FREE_POWER_TURBINE.value
            ]["total"],
            11,
        )
        self.assertEqual(
            summary["posthoc_stage_rho"]["status"], "NOT_CHECKED"
        )
        self.assertEqual(
            summary["posthoc_stage_strength"]["status"], "NOT_CHECKED"
        )
        for forbidden in (
            "feasible_count",
            "feasible_points",
            "passed_count",
            "best_engine",
        ):
            self.assertNotIn(forbidden, summary)

    def test_adjacent_slopes_break_at_failure_without_bridging(self):
        path = self.root / "dataset"
        real_evaluator = coarse.evaluate_architecture_payload

        def forced_failure(payload):
            if (
                payload["architecture"]
                == Architecture.FREE_POWER_TURBINE.value
                and payload["compressor_pressure_ratios"] == [13.0]
            ):
                return EvaluationOutcome.failure(
                    CalculationStatus.NON_PHYSICAL_STATE,
                    "forced_test_failure",
                    "synthetic failure retained by the test",
                )
            return real_evaluator(payload)

        with mock.patch.object(
            coarse,
            "evaluate_architecture_payload",
            new=forced_failure,
        ):
            summary = coarse.run_coarse(path)

        series = next(
            item
            for item in summary["controlled_slices"][
                "adjacent_pi_slopes"
            ]
            if item["architecture"]
            == Architecture.FREE_POWER_TURBINE.value
            and item["split_weight"] is None
        )
        self.assertEqual(len(series["pairs"]), 10)
        for pair in series["pairs"]:
            self.assertEqual(pair["pi_to"] - pair["pi_from"], 1)
            if pair["status"] == "computed":
                self.assertTrue(
                    math.isfinite(pair["d_gas_mass_flow_d_pi"])
                )
                self.assertTrue(
                    math.isfinite(pair["d_fuel_mass_flow_d_pi"])
                )
            else:
                self.assertEqual(pair["status"], "break")
                self.assertIsNone(pair["d_gas_mass_flow_d_pi"])
                self.assertIsNone(pair["d_fuel_mass_flow_d_pi"])

        breaking = [
            pair
            for pair in series["pairs"]
            if pair["status"] == "break"
        ]
        self.assertEqual(len(breaking), 1)
        self.assertEqual(breaking[0]["pi_from"], 12)
        self.assertEqual(breaking[0]["pi_to"], 13)
        self.assertEqual(
            breaking[0]["pi_to_status"],
            CalculationStatus.NON_PHYSICAL_STATE.value,
        )
        # No pair may bridge a failed adjacency by skipping a pi value.
        for pair in series["pairs"]:
            self.assertLessEqual(
                pair["pi_to"] - pair["pi_from"], 1
            )


if __name__ == "__main__":
    unittest.main()
