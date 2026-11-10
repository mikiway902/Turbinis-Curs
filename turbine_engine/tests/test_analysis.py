"""Focused tests for pure post-calculation analysis.

These tests use synthetic annotated rows only.  They check explicit schema and
constraint semantics, feasibility levels, grouped denominators, independently
known Pareto fronts, near-feasibility ranking, and controlled paired
sensitivity.  They do not run a cycle, sweep, or physical model and do not claim
that any synthetic row is a real engine design.
"""

from __future__ import annotations

import array
import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.analysis import (  # noqa: E402
    AnalysisError,
    CalculationStatus,
    CandidateRow,
    ConstraintEvaluation,
    ConstraintStatus,
    Direction,
    FeasibilityStatus,
    Objective,
    SensitivityError,
    VerificationStatus,
    paired_sensitivity,
    pareto_front,
    rank_near_feasible,
    summarize_design_groups,
)


def constraint(
    status: str,
    *,
    actual: float | None = None,
    lower: float | None = None,
    upper: float | None = None,
    scale: float | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {"status": status}
    if actual is not None:
        value["actual"] = actual
    if lower is not None:
        value["lower"] = lower
    if upper is not None:
        value["upper"] = upper
    if scale is not None:
        value["scale"] = scale
    return value


def raw_row(
    candidate_id: str,
    *,
    architecture: str = "A",
    calculation_status: str = "SUCCESS",
    constraints: dict[str, object] | None = None,
    metrics: dict[str, object] | None = None,
    inputs: dict[str, object] | None = None,
    pi_compressor: float | None = None,
    model_variant: str | None = None,
    verification_status: str | None = None,
    failure_stage: str | None = None,
    failure_reason: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_id": candidate_id,
        "architecture": architecture,
        "calculation_status": calculation_status,
        "constraints": constraints if constraints is not None else {},
        "metrics": metrics if metrics is not None else {},
        "inputs": inputs if inputs is not None else {},
        "pi_compressor": pi_compressor,
        "model_variant": model_variant,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
    }
    if verification_status is not None:
        row["verification_status"] = verification_status
    return row


def analyzed(
    candidate_id: str,
    *,
    architecture: str = "A",
    metrics: dict[str, float] | None = None,
    constraints: dict[str, object] | None = None,
    pi_compressor: float | None = None,
    model_variant: str | None = None,
) -> CandidateRow:
    return CandidateRow.from_mapping(
        raw_row(
            candidate_id,
            architecture=architecture,
            metrics=metrics,
            constraints=constraints
            if constraints is not None
            else {"l1": constraint("PASS", actual=15.0, lower=14.0)},
            pi_compressor=pi_compressor,
            model_variant=model_variant,
        )
    )


class AdapterAndConstraintTests(unittest.TestCase):
    def test_required_keys_are_declared_and_enforced(self) -> None:
        row = raw_row(
            "r1",
            constraints={"l1": constraint("PASS", actual=15.0, lower=14.0)},
        )
        missing = dict(row)
        del missing["metrics"]
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(missing)

        unknown = dict(row)
        unknown["not_in_schema"] = 1
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(unknown)

    def test_status_is_explicit_and_truthiness_is_not_accepted(self) -> None:
        base = raw_row("r1", constraints={"c": {"status": True}})
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(base)

        lowercase = raw_row("r1", constraints={"c": {"status": "pass"}})
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(lowercase)

        missing_status = raw_row(
            "r1", constraints={"c": {"actual": 1.0, "lower": 0.0}}
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(missing_status)

    def test_pass_and_fail_must_agree_with_bounds(self) -> None:
        false_pass = raw_row(
            "r1",
            constraints={"c": constraint("PASS", actual=10.0, lower=14.0)},
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(false_pass)

        false_fail = raw_row(
            "r1",
            constraints={"c": constraint("FAIL", actual=15.0, lower=14.0)},
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(false_fail)

        no_actual = raw_row("r1", constraints={"c": {"status": "PASS", "lower": 1.0}})
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(no_actual)

        no_bound = raw_row("r1", constraints={"c": {"status": "PASS", "actual": 1.0}})
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(no_bound)

    def test_margins_and_normalization_are_derived(self) -> None:
        row = CandidateRow.from_mapping(
            raw_row(
                "r1",
                constraints={
                    "lower_fail": constraint(
                        "FAIL", actual=10.0, lower=14.0, scale=2.0
                    ),
                    "upper_fail_no_scale": constraint(
                        "FAIL", actual=21.0, upper=20.0
                    ),
                    "pass": constraint(
                        "PASS", actual=15.0, lower=14.0, upper=16.0, scale=2.0
                    ),
                },
            )
        )

        lower_fail = row.constraints["lower_fail"]
        self.assertEqual(lower_fail.lower_margin, -4.0)
        self.assertIsNone(lower_fail.upper_margin)
        self.assertEqual(lower_fail.violation, 4.0)
        self.assertEqual(lower_fail.normalized_violation, 2.0)

        upper_fail = row.constraints["upper_fail_no_scale"]
        self.assertEqual(upper_fail.violation, 1.0)
        self.assertIsNone(upper_fail.normalized_violation)

        passed = row.constraints["pass"]
        self.assertEqual(passed.lower_margin, 1.0)
        self.assertEqual(passed.upper_margin, 1.0)
        self.assertEqual(passed.violation, 0.0)
        self.assertEqual(passed.normalized_violation, 0.0)

    def test_not_checked_never_becomes_pass_or_zero_violation(self) -> None:
        row = CandidateRow.from_mapping(
            raw_row(
                "r1",
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0),
                    "material_life": constraint(
                        "NOT_CHECKED", actual=10.0, lower=14.0
                    ),
                },
            )
        )

        unknown = row.constraints["material_life"]
        self.assertEqual(unknown.status, ConstraintStatus.NOT_CHECKED)
        self.assertEqual(unknown.lower_margin, -4.0)
        self.assertIsNone(unknown.violation)
        self.assertIsNone(unknown.normalized_violation)
        self.assertFalse(row.screening_feasible)
        self.assertEqual(row.feasibility_status, FeasibilityStatus.UNVERIFIED)

    def test_feasibility_levels_are_separate(self) -> None:
        all_pass = analyzed("pass")
        self.assertTrue(all_pass.calculation_succeeded)
        self.assertTrue(all_pass.screening_feasible)
        self.assertFalse(all_pass.fully_verified)
        self.assertEqual(
            all_pass.feasibility_status, FeasibilityStatus.SCREENED_FEASIBLE
        )

        verified = CandidateRow.from_mapping(
            raw_row(
                "verified",
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
                verification_status="VERIFIED",
            )
        )
        self.assertTrue(verified.fully_verified)

        failed = CandidateRow.from_mapping(
            raw_row(
                "calc_failed",
                calculation_status="NUMERICAL_FAILURE",
                constraints={},
            )
        )
        self.assertFalse(failed.calculation_succeeded)
        self.assertFalse(failed.screening_feasible)
        self.assertFalse(failed.fully_verified)
        self.assertEqual(
            failed.feasibility_status,
            FeasibilityStatus.CALCULATION_FAILED,
        )

        no_constraints = CandidateRow.from_mapping(raw_row("empty", constraints={}))
        self.assertFalse(no_constraints.screening_feasible)
        self.assertEqual(
            no_constraints.feasibility_status,
            FeasibilityStatus.UNVERIFIED,
        )

        infeasible = CandidateRow.from_mapping(
            raw_row(
                "infeasible",
                constraints={
                    "l1": constraint("FAIL", actual=10.0, lower=14.0)
                },
            )
        )
        self.assertFalse(infeasible.screening_feasible)
        self.assertFalse(infeasible.fully_verified)
        self.assertEqual(
            infeasible.feasibility_status,
            FeasibilityStatus.ENGINEERING_INFEASIBLE,
        )

    def test_explicit_verification_does_not_override_unchecked(self) -> None:
        row = CandidateRow.from_mapping(
            raw_row(
                "r1",
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0),
                    "life": constraint("NOT_CHECKED"),
                },
                verification_status="VERIFIED",
            )
        )
        self.assertEqual(row.verification_status, VerificationStatus.VERIFIED)
        self.assertFalse(row.screening_feasible)
        self.assertFalse(row.fully_verified)

    def test_row_is_immutable_and_copies_inputs(self) -> None:
        constraints = {"c": constraint("PASS", actual=15.0, lower=14.0)}
        metrics = {"eta": 0.8}
        inputs = {"pi_k": 7.0, "geometry": ["a", "b"]}
        row = CandidateRow.from_mapping(
            raw_row(
                "r1",
                constraints=constraints,
                metrics=metrics,
                inputs=inputs,
            )
        )

        constraints["c"] = constraint("FAIL", actual=1.0, lower=14.0)
        metrics["eta"] = 0.9
        inputs["pi_k"] = 8.0
        inputs["geometry"].append("c")

        self.assertEqual(row.metrics["eta"], 0.8)
        self.assertEqual(row.inputs["pi_k"], 7.0)
        self.assertEqual(row.inputs["geometry"], ("a", "b"))
        self.assertEqual(row.constraints["c"].status, ConstraintStatus.PASS)

        with self.assertRaises(FrozenInstanceError):
            row.candidate_id = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            row.metrics["eta"] = 0.9  # type: ignore[index]

    def test_inputs_reject_opaque_mutable_aliases(self) -> None:
        base = raw_row(
            "r1",
            constraints={"c": constraint("PASS", actual=1.0, lower=0.0)},
            inputs={"opaque": bytearray(b"x")},
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(base)

        array_row = raw_row(
            "r2",
            constraints={"c": constraint("PASS", actual=1.0, lower=0.0)},
            inputs={"opaque": array.array("d", [1.0])},
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(array_row)

        numpy_row = raw_row(
            "r3",
            constraints={"c": constraint("PASS", actual=1.0, lower=0.0)},
            inputs={"opaque": np.array([1.0])},
        )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(numpy_row)

        valid = CandidateRow.from_mapping(
            raw_row(
                "r4",
                constraints={"c": constraint("PASS", actual=1.0, lower=0.0)},
                inputs={
                    "number": 1,
                    "flag": True,
                    "nested": {"items": [1.0, "x"]},
                },
            )
        )
        self.assertEqual(valid.inputs["number"], 1)
        self.assertEqual(valid.inputs["nested"]["items"], (1.0, "x"))

    def test_lossy_numeric_coercion_and_numeric_strings_are_rejected(self) -> None:
        with self.assertRaises(AnalysisError):
            ConstraintEvaluation(
                name="c",
                status="FAIL",
                actual=9007199254740993,
                upper=9007199254740992,
            )
        with self.assertRaises(AnalysisError):
            ConstraintEvaluation(
                name="c", status="PASS", actual="15", lower="14"
            )
        with self.assertRaises(AnalysisError):
            CandidateRow.from_mapping(
                raw_row(
                    "r1",
                    metrics={"objective": 9007199254740993},
                    constraints={"c": constraint("PASS", actual=1.0, lower=0.0)},
                )
            )

        exact = ConstraintEvaluation(
            name="c",
            status="PASS",
            actual=9007199254740992,
            upper=9007199254740992,
        )
        self.assertEqual(exact.upper_margin, 0.0)

    def test_constraint_overflow_is_explicit(self) -> None:
        with self.assertRaises(AnalysisError):
            ConstraintEvaluation(
                name="c",
                status="FAIL",
                actual=1e308,
                upper=-1e308,
                scale=1e308,
            )

    def test_normalized_overflow_and_underflow_are_not_silent(self) -> None:
        overflow = ConstraintEvaluation(
            name="c",
            status="FAIL",
            actual=1e308,
            upper=0.0,
            scale=1e-308,
        )
        self.assertEqual(overflow.violation, 1e308)
        self.assertIsNone(overflow.normalized_violation)

        underflow = ConstraintEvaluation(
            name="c",
            status="FAIL",
            actual=1e-308,
            upper=0.0,
            scale=1e308,
        )
        self.assertEqual(underflow.violation, 1e-308)
        self.assertIsNone(underflow.normalized_violation)

        row = CandidateRow.from_mapping(
            raw_row(
                "underflow",
                constraints={
                    "c": {
                        "status": "FAIL",
                        "actual": 1e-308,
                        "upper": 0.0,
                        "scale": 1e308,
                    }
                },
            )
        )
        report = rank_near_feasible([row])
        entry = report.entries[0]
        self.assertIsNone(entry.rank)
        self.assertEqual(entry.missing_normalization, ("c",))
        self.assertIn("underflowing", entry.unranked_reason or "")

    def test_constraint_with_bounds_alias_and_immutability(self) -> None:
        evaluation = ConstraintEvaluation(
            name="c",
            status=ConstraintStatus.FAIL,
            actual=9.0,
            lower=10.0,
            upper=20.0,
        )
        self.assertEqual(evaluation.violation, 1.0)
        with self.assertRaises(FrozenInstanceError):
            evaluation.actual = 11.0  # type: ignore[misc]

        row = CandidateRow.from_mapping(
            raw_row(
                "r1",
                constraints={"c": {"status": "FAIL", "actual": 9, "bounds": (10, 20)}},
            )
        )
        self.assertEqual(row.constraints["c"].lower, 10.0)
        self.assertEqual(row.constraints["c"].upper, 20.0)


class GroupingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            raw_row(
                "a7_pass",
                architecture="A",
                pi_compressor=7.0,
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0),
                    "stress": constraint("PASS", actual=250.0, upper=300.0),
                },
            ),
            raw_row(
                "a7_fail",
                architecture="A",
                pi_compressor=7.0,
                constraints={
                    "l1": constraint("FAIL", actual=13.0, lower=14.0),
                    "stress": constraint("NOT_CHECKED"),
                },
            ),
            raw_row(
                "a7_calc_fail",
                architecture="A",
                pi_compressor=7.0,
                calculation_status="NUMERICAL_FAILURE",
                failure_stage="compressor",
                failure_reason="no convergence",
                constraints={},
            ),
            raw_row(
                "a8_pass",
                architecture="A",
                pi_compressor=8.0,
                constraints={
                    "l1": constraint("PASS", actual=16.0, lower=14.0)
                },
                verification_status="VERIFIED",
            ),
            raw_row(
                "b_none_calc_fail",
                architecture="B",
                calculation_status="NON_PHYSICAL_STATE",
                failure_stage="turbine",
                failure_reason="negative pressure",
                constraints={"stress": constraint("NOT_CHECKED")},
            ),
        ]

    def test_groups_are_ordered_and_denominators_are_explicit(self) -> None:
        groups = summarize_design_groups(self.rows)

        self.assertEqual(
            [(item.architecture, item.pi_compressor) for item in groups],
            [("A", 7.0), ("A", 8.0), ("B", None)],
        )

        a7 = groups[0]
        self.assertEqual(a7.total_rows, 3)
        self.assertEqual(a7.calculation_success, 2)
        self.assertEqual(a7.calculation_failures, 1)
        self.assertEqual(
            a7.feasibility_counts[FeasibilityStatus.SCREENED_FEASIBLE], 1
        )
        self.assertEqual(
            a7.feasibility_counts[FeasibilityStatus.ENGINEERING_INFEASIBLE], 1
        )
        self.assertEqual(
            a7.feasibility_counts[FeasibilityStatus.CALCULATION_FAILED], 1
        )
        self.assertEqual(
            a7.feasibility_counts[FeasibilityStatus.UNVERIFIED], 0
        )
        self.assertEqual(a7.failure_reason_counts, {"no convergence": 1})
        self.assertEqual(a7.failure_stage_counts, {"compressor": 1})

        l1 = a7.constraint_statistics["l1"]
        self.assertEqual(l1.denominator_rows, 3)
        self.assertEqual(l1.present_rows, 2)
        self.assertEqual(l1.absent_rows, 1)
        self.assertEqual(l1.pass_count, 1)
        self.assertEqual(l1.fail_count, 1)
        self.assertEqual(l1.not_checked_count, 0)
        self.assertEqual(l1.checked_rows, 2)
        self.assertEqual(l1.unchecked_or_absent_rows, 1)
        self.assertEqual(l1.pass_fraction_of_checked, 0.5)
        self.assertEqual(l1.fail_fraction_of_checked, 0.5)

        stress = a7.constraint_statistics["stress"]
        self.assertEqual(stress.denominator_rows, 3)
        self.assertEqual(stress.present_rows, 2)
        self.assertEqual(stress.absent_rows, 1)
        self.assertEqual(stress.pass_count, 1)
        self.assertEqual(stress.fail_count, 0)
        self.assertEqual(stress.not_checked_count, 1)
        self.assertEqual(stress.checked_rows, 1)
        self.assertEqual(stress.unchecked_or_absent_rows, 2)
        self.assertEqual(stress.pass_fraction_of_checked, 1.0)

    def test_absent_constraint_is_not_counted_as_checked(self) -> None:
        a8 = summarize_design_groups(self.rows)[1]
        stress = a8.constraint_statistics["stress"]
        self.assertEqual(stress.denominator_rows, 1)
        self.assertEqual(stress.present_rows, 0)
        self.assertEqual(stress.absent_rows, 1)
        self.assertEqual(stress.checked_rows, 0)
        self.assertIsNone(stress.pass_fraction_of_checked)
        self.assertIsNone(stress.fail_fraction_of_checked)

    def test_failure_reason_counts_use_failure_denominator(self) -> None:
        groups = summarize_design_groups(self.rows)
        a7 = groups[0]
        self.assertEqual(sum(a7.failure_reason_counts.values()), a7.calculation_failures)
        b = groups[2]
        self.assertEqual(b.calculation_failures, 1)
        self.assertEqual(b.failure_reason_counts, {"negative pressure": 1})
        self.assertEqual(b.failure_stage_counts, {"turbine": 1})

    def test_missing_failure_reason_is_labeled_not_dropped(self) -> None:
        groups = summarize_design_groups(
            [
                raw_row(
                    "failed",
                    calculation_status="NUMERICAL_FAILURE",
                    constraints={},
                )
            ]
        )
        self.assertEqual(groups[0].failure_reason_counts, {"<not_recorded>": 1})
        self.assertEqual(groups[0].failure_stage_counts, {"<not_recorded>": 1})

    def test_declared_required_constraint_changes_feasibility_status(self) -> None:
        groups = summarize_design_groups(
            [
                raw_row(
                    "only_l1",
                    architecture="A",
                    pi_compressor=7.0,
                    constraints={
                        "l1": constraint("PASS", actual=15.0, lower=14.0)
                    },
                ),
                raw_row(
                    "l1_fail",
                    architecture="A",
                    pi_compressor=7.0,
                    constraints={
                        "l1": constraint("FAIL", actual=10.0, lower=14.0)
                    },
                ),
            ],
            required_constraints=("life",),
        )
        group = groups[0]
        life = group.constraint_statistics["life"]
        self.assertEqual(life.denominator_rows, 2)
        self.assertEqual(life.present_rows, 0)
        self.assertEqual(life.absent_rows, 2)
        self.assertEqual(life.checked_rows, 0)
        self.assertEqual(life.unchecked_or_absent_rows, 2)

        # A missing required criterion is UNVERIFIED, but a known failed
        # constraint remains ENGINEERING_INFEASIBLE.
        self.assertEqual(
            group.feasibility_counts[FeasibilityStatus.UNVERIFIED], 1
        )
        self.assertEqual(
            group.feasibility_counts[FeasibilityStatus.ENGINEERING_INFEASIBLE],
            1,
        )
        self.assertEqual(
            group.feasibility_counts[FeasibilityStatus.SCREENED_FEASIBLE], 0
        )


class ParetoTests(unittest.TestCase):
    def test_known_minimization_front_and_duplicate_preservation(self) -> None:
        points = {
            "A": (1.0, 4.0),
            "B": (2.0, 3.0),
            "C": (3.0, 1.0),
            "D": (2.0, 2.0),
            "E": (2.0, 2.0),
            "F": (4.0, 4.0),
            "G": (4.0, 5.0),
        }
        rows = [
            analyzed(
                name,
                metrics={"size": value[0], "stress": value[1]},
            )
            for name, value in points.items()
        ]

        result = pareto_front(
            rows,
            [
                Objective("size", Direction.MINIMIZE),
                Objective("stress", Direction.MINIMIZE),
            ],
        )

        self.assertEqual(
            [item.candidate_id for item in result.front],
            ["A", "C", "D", "E"],
        )
        self.assertEqual(
            [item.candidate_id for item in result.front].count("D"), 1
        )
        self.assertEqual(
            [item.candidate_id for item in result.front].count("E"), 1
        )
        self.assertEqual(result.excluded, ())

    def test_maximize_and_minimize_directions(self) -> None:
        rows = [
            analyzed(
                "A", metrics={"efficiency": 0.80, "stress": 300.0}
            ),
            analyzed(
                "B", metrics={"efficiency": 0.90, "stress": 320.0}
            ),
            analyzed(
                "C", metrics={"efficiency": 0.85, "stress": 280.0}
            ),
            analyzed(
                "D", metrics={"efficiency": 0.70, "stress": 250.0}
            ),
        ]

        result = pareto_front(
            rows,
            [
                Objective("efficiency", Direction.MAXIMIZE),
                Objective("stress", Direction.MINIMIZE),
            ],
        )

        self.assertEqual(
            [item.candidate_id for item in result.front],
            ["B", "C", "D"],
        )

    def test_default_policy_excludes_infeasible_unchecked_and_failed(self) -> None:
        feasible = analyzed("feasible", metrics={"objective": 1.0})
        infeasible = CandidateRow.from_mapping(
            raw_row(
                "infeasible",
                metrics={"objective": 2.0},
                constraints={
                    "l1": constraint("FAIL", actual=10.0, lower=14.0)
                },
            )
        )
        unchecked = CandidateRow.from_mapping(
            raw_row(
                "unchecked",
                metrics={"objective": 0.0},
                constraints={"life": constraint("NOT_CHECKED")},
            )
        )
        no_constraints = CandidateRow.from_mapping(
            raw_row("no_constraints", metrics={"objective": -1.0})
        )
        calc_failed = CandidateRow.from_mapping(
            raw_row(
                "calc_failed",
                metrics={"objective": -2.0},
                calculation_status="NUMERICAL_FAILURE",
                constraints={},
            )
        )

        default = pareto_front(
            [feasible, infeasible, unchecked, no_constraints, calc_failed],
            [Objective("objective", Direction.MINIMIZE)],
        )
        self.assertEqual(
            [item.candidate_id for item in default.front], ["feasible"]
        )
        excluded_ids = {
            item.candidate_id for item in default.excluded
        }
        self.assertEqual(
            excluded_ids,
            {"infeasible", "unchecked", "no_constraints", "calc_failed"},
        )

        include_infeasible = pareto_front(
            [feasible, infeasible, unchecked, no_constraints, calc_failed],
            [Objective("objective", Direction.MINIMIZE)],
            include_infeasible=True,
        )
        self.assertNotIn(
            "infeasible",
            {item.candidate_id for item in include_infeasible.excluded},
        )
        self.assertIn(
            "unchecked",
            {item.candidate_id for item in include_infeasible.excluded},
        )

        include_all = pareto_front(
            [feasible, infeasible, unchecked, no_constraints, calc_failed],
            [Objective("objective", Direction.MINIMIZE)],
            include_infeasible=True,
            include_unchecked=True,
        )
        self.assertEqual(
            [item.candidate_id for item in include_all.front],
            ["no_constraints"],
        )
        self.assertEqual(
            {item.candidate_id for item in include_all.excluded},
            {"calc_failed"},
        )

    def test_missing_or_nonfinite_objectives_are_excluded_not_zeroed(self) -> None:
        rows = [
            analyzed("missing", metrics={"other": 1.0}),
            analyzed("nan", metrics={"objective": math.nan}),
            analyzed("inf", metrics={"objective": math.inf}),
            analyzed("valid", metrics={"objective": 5.0}),
        ]

        result = pareto_front(
            rows, [Objective("objective", Direction.MINIMIZE)]
        )

        self.assertEqual(
            [item.candidate_id for item in result.front], ["valid"]
        )
        self.assertEqual(
            {item.candidate_id for item in result.excluded},
            {"missing", "nan", "inf"},
        )
        reasons = {item.candidate_id: item.reason for item in result.excluded}
        self.assertIn("missing", reasons["missing"])
        self.assertIn("non-finite", reasons["nan"])
        self.assertIn("non-finite", reasons["inf"])

    def test_required_constraint_absent_from_row_is_unchecked(self) -> None:
        row = analyzed("only_l1", metrics={"objective": 1.0})

        default = pareto_front(
            [row],
            [Objective("objective", Direction.MINIMIZE)],
            required_constraints=("life",),
        )
        self.assertEqual(default.front, ())
        self.assertEqual(len(default.excluded), 1)
        self.assertIn("missing required constraints", default.excluded[0].reason)

        explicit = pareto_front(
            [row],
            [Objective("objective", Direction.MINIMIZE)],
            required_constraints=("life",),
            include_unchecked=True,
        )
        self.assertEqual(
            [item.candidate_id for item in explicit.front], ["only_l1"]
        )

    def test_policy_flags_require_real_booleans(self) -> None:
        row = analyzed("A", metrics={"x": 1.0})
        objectives = [Objective("x", Direction.MINIMIZE)]
        with self.assertRaises(AnalysisError):
            pareto_front([row], objectives, include_infeasible="False")  # type: ignore[arg-type]
        with self.assertRaises(AnalysisError):
            pareto_front([row], objectives, include_unchecked="false")  # type: ignore[arg-type]
        with self.assertRaises(AnalysisError):
            pareto_front([row], objectives, include_calculation_failures=1)  # type: ignore[arg-type]

    def test_objective_validation(self) -> None:
        row = analyzed("A", metrics={"x": 1.0})
        with self.assertRaises(AnalysisError):
            pareto_front([row], [])
        with self.assertRaises(AnalysisError):
            pareto_front(
                [row],
                [
                    Objective("x", Direction.MINIMIZE),
                    Objective("x", Direction.MAXIMIZE),
                ],
            )
        with self.assertRaises(AnalysisError):
            Objective("x", "MIN")
        with self.assertRaises(AnalysisError):
            Objective("", Direction.MINIMIZE)


class NearFeasibilityTests(unittest.TestCase):
    def _rows(self) -> list[CandidateRow]:
        return [
            CandidateRow.from_mapping(
                raw_row(
                    "X_worst",
                    metrics={"eta": 0.8},
                    constraints={
                        "l1": constraint(
                            "FAIL", actual=10.0, lower=14.0, scale=2.0
                        ),
                        "stress": constraint(
                            "PASS", actual=250.0, upper=300.0
                        ),
                    },
                )
            ),
            CandidateRow.from_mapping(
                raw_row(
                    "Y_better",
                    metrics={"eta": 0.81},
                    constraints={
                        "l1": constraint(
                            "FAIL", actual=13.0, lower=14.0, scale=2.0
                        ),
                        "stress": constraint(
                            "PASS", actual=250.0, upper=300.0
                        ),
                    },
                )
            ),
            CandidateRow.from_mapping(
                raw_row(
                    "Z_unknown",
                    constraints={
                        "l1": constraint(
                            "NOT_CHECKED", actual=10.0, lower=14.0
                        ),
                        "stress": constraint(
                            "PASS", actual=250.0, upper=300.0
                        ),
                    },
                )
            ),
            CandidateRow.from_mapping(
                raw_row(
                    "W_no_scale",
                    constraints={
                        "l1": constraint("FAIL", actual=10.0, lower=14.0),
                        "stress": constraint(
                            "PASS", actual=250.0, upper=300.0
                        ),
                    },
                )
            ),
            CandidateRow.from_mapping(
                raw_row(
                    "V_failed",
                    calculation_status="NUMERICAL_FAILURE",
                    constraints={},
                )
            ),
            CandidateRow.from_mapping(
                raw_row(
                    "F_pass",
                    constraints={
                        "l1": constraint(
                            "PASS", actual=15.0, lower=14.0, scale=2.0
                        ),
                        "stress": constraint(
                            "PASS", actual=250.0, upper=300.0
                        ),
                    },
                )
            ),
        ]

    def test_full_vectors_and_deterministic_rank(self) -> None:
        report = rank_near_feasible(self._rows())
        ranks = {
            entry.candidate.candidate_id: entry.rank
            for entry in report.entries
        }

        self.assertEqual(ranks["F_pass"], 1)
        self.assertEqual(ranks["Y_better"], 2)
        self.assertEqual(ranks["X_worst"], 3)
        self.assertIsNone(ranks["Z_unknown"])
        self.assertIsNone(ranks["W_no_scale"])
        self.assertNotIn("V_failed", ranks)

        x = next(
            entry
            for entry in report.entries
            if entry.candidate.candidate_id == "X_worst"
        )
        self.assertEqual(x.violation_vector, {"l1": 4.0, "stress": 0.0})
        self.assertEqual(
            x.normalized_violation_vector, {"l1": 2.0, "stress": 0.0}
        )

        z = next(
            entry
            for entry in report.entries
            if entry.candidate.candidate_id == "Z_unknown"
        )
        self.assertEqual(z.violation_vector, {"l1": None, "stress": 0.0})
        self.assertEqual(z.unknown_constraints, ("l1",))
        self.assertIn("unchecked", z.unranked_reason or "")

        w = next(
            entry
            for entry in report.entries
            if entry.candidate.candidate_id == "W_no_scale"
        )
        self.assertEqual(w.missing_normalization, ("l1",))
        self.assertIn("scale", w.unranked_reason or "")

    def test_include_flag_requires_real_boolean(self) -> None:
        with self.assertRaises(AnalysisError):
            rank_near_feasible(
                self._rows(), include_calculation_failures="False"
            )  # type: ignore[arg-type]

    def test_calculation_failures_excluded_by_default_and_flag(self) -> None:
        default = rank_near_feasible(self._rows())
        self.assertNotIn(
            "V_failed",
            {entry.candidate.candidate_id for entry in default.entries},
        )

        included = rank_near_feasible(
            self._rows(), include_calculation_failures=True
        )
        failed = next(
            entry
            for entry in included.entries
            if entry.candidate.candidate_id == "V_failed"
        )
        self.assertIsNone(failed.rank)
        self.assertIn("NUMERICAL_FAILURE", failed.unranked_reason or "")

    def test_missing_declared_constraint_is_unknown_not_zero(self) -> None:
        row = CandidateRow.from_mapping(
            raw_row(
                "only_l1",
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )
        report = rank_near_feasible(
            [row], required_constraints=("life",)
        )
        entry = report.entries[0]
        self.assertIsNone(entry.rank)
        self.assertEqual(entry.violation_vector, {"l1": 0.0, "life": None})
        self.assertEqual(entry.unknown_constraints, ("life",))
        self.assertIn("missing required", entry.unranked_reason or "")

    def test_unknown_never_contributes_zero(self) -> None:
        report = rank_near_feasible(self._rows())
        z = next(
            entry
            for entry in report.entries
            if entry.candidate.candidate_id == "Z_unknown"
        )
        self.assertIsNone(z.rank)
        self.assertIsNone(z.normalized_violation_vector["l1"])
        self.assertFalse(
            z.candidate.screening_feasible,
        )


class SensitivityTests(unittest.TestCase):
    def _row(
        self,
        candidate_id: str,
        *,
        pi_k: float = 7.0,
        metric_eta: object = 0.80,
        architecture: str = "A",
        model_variant: str = "v1",
        calculation_status: str = "SUCCESS",
        extra_inputs: dict[str, object] | None = None,
    ) -> CandidateRow:
        inputs: dict[str, object] = {
            "pi_k": pi_k,
            "T_g": 1250.0,
            "geometry": ("a", "b"),
        }
        if extra_inputs:
            inputs.update(extra_inputs)
        return CandidateRow.from_mapping(
            raw_row(
                candidate_id,
                architecture=architecture,
                model_variant=model_variant,
                calculation_status=calculation_status,
                inputs=inputs,
                metrics={
                    "eta": metric_eta,
                    "l1": 0.010 + (pi_k - 7.0) * 0.001,
                    "zero_baseline": 0.0,
                },
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )

    def test_paired_difference_and_slope(self) -> None:
        baseline = self._row("base", pi_k=7.0, metric_eta=0.80)
        perturbed = self._row("pert", pi_k=7.7, metric_eta=0.82)

        result = paired_sensitivity(
            baseline,
            perturbed,
            parameter="pi_k",
            metric_names=("eta", "l1"),
            relative=True,
        )

        self.assertAlmostEqual(result.dx, 0.7, places=12)
        self.assertEqual(result.baseline_parameter, 7.0)
        self.assertEqual(result.perturbed_parameter, 7.7)
        eta = result.metrics["eta"]
        self.assertAlmostEqual(eta.difference or 0.0, 0.02, places=12)
        self.assertAlmostEqual(
            eta.derivative or 0.0, 0.02 / 0.7, places=12
        )
        self.assertAlmostEqual(
            eta.relative_change or 0.0, 0.02 / 0.8, places=12
        )
        l1 = result.metrics["l1"]
        self.assertAlmostEqual(l1.difference or 0.0, 0.0007, places=12)

    def test_zero_baseline_relative_change_is_none(self) -> None:
        baseline = self._row("base", pi_k=7.0)
        perturbed = self._row("pert", pi_k=7.7)
        result = paired_sensitivity(
            baseline,
            perturbed,
            parameter="pi_k",
            metric_names=("zero_baseline",),
            relative=True,
        )
        metric = result.metrics["zero_baseline"]
        self.assertEqual(metric.difference, 0.0)
        self.assertEqual(metric.derivative, 0.0)
        self.assertIsNone(metric.relative_change)

    def test_unpaired_inputs_are_refused_without_tolerance(self) -> None:
        baseline = self._row("base", pi_k=7.0)
        different_tg = CandidateRow.from_mapping(
            raw_row(
                "different",
                model_variant="v1",
                inputs={
                    "pi_k": 7.7,
                    "T_g": 1250.0000001,
                    "geometry": ("a", "b"),
                },
                metrics={"eta": 0.82},
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )
        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, different_tg, parameter="pi_k")

        extra_key = CandidateRow.from_mapping(
            raw_row(
                "extra",
                model_variant="v1",
                inputs={
                    "pi_k": 7.7,
                    "T_g": 1250.0,
                    "geometry": ("a", "b"),
                    "extra": 1.0,
                },
                metrics={"eta": 0.82},
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )
        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, extra_key, parameter="pi_k")

        missing_key = CandidateRow.from_mapping(
            raw_row(
                "missing",
                model_variant="v1",
                inputs={"pi_k": 7.7, "T_g": 1250.0},
                metrics={"eta": 0.82},
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )
        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, missing_key, parameter="pi_k")

    def test_architecture_variant_and_status_must_match(self) -> None:
        baseline = self._row("base", pi_k=7.0)

        with self.assertRaises(SensitivityError):
            paired_sensitivity(
                baseline,
                self._row("arch", pi_k=7.7, architecture="B"),
                parameter="pi_k",
            )
        with self.assertRaises(SensitivityError):
            paired_sensitivity(
                baseline,
                self._row("variant", pi_k=7.7, model_variant="v2"),
                parameter="pi_k",
            )
        with self.assertRaises(SensitivityError):
            paired_sensitivity(
                baseline,
                self._row(
                    "failed",
                    pi_k=7.7,
                    calculation_status="NUMERICAL_FAILURE",
                ),
                parameter="pi_k",
            )

    def test_zero_dx_and_non_numeric_parameter_are_invalid(self) -> None:
        baseline = self._row("base", pi_k=7.0)
        same = self._row("same", pi_k=7.0)
        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, same, parameter="pi_k")

        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, same, parameter="geometry")

        with self.assertRaises(SensitivityError):
            paired_sensitivity(baseline, same, parameter="missing")

    def test_missing_or_nonfinite_metric_is_not_zero(self) -> None:
        baseline = self._row("base", pi_k=7.0)
        perturbed = CandidateRow.from_mapping(
            raw_row(
                "pert",
                model_variant="v1",
                inputs={
                    "pi_k": 7.7,
                    "T_g": 1250.0,
                    "geometry": ("a", "b"),
                },
                metrics={"eta": math.nan},
                constraints={
                    "l1": constraint("PASS", actual=15.0, lower=14.0)
                },
            )
        )
        result = paired_sensitivity(
            baseline,
            perturbed,
            parameter="pi_k",
            metric_names=("eta",),
        )
        eta = result.metrics["eta"]
        self.assertIsNone(eta.difference)
        self.assertIsNone(eta.derivative)
        self.assertIsNone(eta.relative_change)
        self.assertIsNotNone(eta.unavailable_reason)


if __name__ == "__main__":
    unittest.main()
