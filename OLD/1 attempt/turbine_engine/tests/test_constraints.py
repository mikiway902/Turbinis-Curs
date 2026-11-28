"""Focused tests for post-calculation constraint annotation.

Acceptance is deliberately separate from RAW persistence: changing a profile
must not mutate a RAW record or call an evaluator, and missing values must never
silently pass.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.constraints import (  # noqa: E402
    RHO_LE_0_65_PROFILE_NAME,
    ConstraintProfile,
    ConstraintRule,
    annotate_candidate,
    annotate_records,
    rho_065_profile,
)
from turbine_engine.raw import (  # noqa: E402
    CalculationStatus,
    ConstraintStatus,
    EvaluationOutcome,
    NumericalFailure,
    evaluate_candidate,
    make_run_id,
)


RUN_ID = make_run_id(
    {"model_variant": "synthetic-v1", "evaluator_version": "eval-test"}
)


def make_record(outputs, *, evaluator=None):
    if evaluator is None:
        def evaluator(payload):
            return outputs

    return evaluate_candidate(
        evaluator,
        {"x": 1, "denominator": 2},
        architecture="SYNTHETIC",
        model_variant="synthetic-v1",
        run_id=RUN_ID,
        sampling_method="unit_test",
    )


class ConstraintAnnotationTests(unittest.TestCase):
    def test_rho_profile_is_opt_in_and_annotates_only(self):
        self.assertEqual(rho_065_profile().name, RHO_LE_0_65_PROFILE_NAME)
        record = make_record({"rho": 0.70, "work": 12.0})
        profile = rho_065_profile()
        annotated = annotate_candidate(record, profile)

        self.assertIs(annotated.constraint_status, ConstraintStatus.FAIL)
        self.assertIs(annotated.feasible, False)
        check = annotated.checks[0]
        self.assertAlmostEqual(check.actual, 0.70)
        self.assertAlmostEqual(check.maximum, 0.65)
        self.assertAlmostEqual(check.upper_margin, -0.05)
        self.assertAlmostEqual(check.normalized_violation, 0.05 / 0.65)
        self.assertTrue(check.checked)
        self.assertIs(check.passed, False)

        # RAW stayed untouched, including its calculation/constraint status.
        self.assertIs(record.constraint_status, ConstraintStatus.NOT_EVALUATED)
        self.assertEqual(annotated.raw.to_dict(), record.to_dict())

        passing = annotate_candidate(make_record({"rho": 0.60}), profile)
        self.assertIs(passing.constraint_status, ConstraintStatus.PASS)
        self.assertIs(passing.feasible, True)

    def test_changing_acceptance_does_not_mutate_raw_or_rerun_evaluator(self):
        calls = []

        def evaluator(payload):
            calls.append(payload)
            return {"rho": 0.70, "work": 12.0}

        record = make_record(None, evaluator=evaluator)
        before = record.to_dict()
        strict = annotate_candidate(record, rho_065_profile())
        relaxed = annotate_candidate(
            record,
            ConstraintProfile(
                "RELAXED_UNIT_TEST",
                (ConstraintRule("outputs.rho", name="rho", maximum=0.80),),
            ),
        )
        self.assertIs(strict.constraint_status, ConstraintStatus.FAIL)
        self.assertIs(relaxed.constraint_status, ConstraintStatus.PASS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(record.to_dict(), before)
        self.assertEqual(strict.raw.record_hash, relaxed.raw.record_hash)

    def test_missing_value_is_not_checked_not_pass(self):
        record = make_record({"work": 12.0})
        annotated = annotate_candidate(record, rho_065_profile())
        self.assertIs(annotated.constraint_status, ConstraintStatus.NOT_CHECKED)
        self.assertIsNone(annotated.feasible)
        self.assertFalse(annotated.checks[0].checked)
        self.assertIsNone(annotated.checks[0].passed)
        self.assertEqual(annotated.checks[0].reason, "missing")
        self.assertIsNone(annotated.checks[0].actual)

    def test_non_numeric_and_null_are_not_checked(self):
        for value, reason in ((None, "null"), (True, "non_numeric"), ("bad", "non_numeric")):
            with self.subTest(value=value):
                record = make_record({"rho": value})
                annotated = annotate_candidate(record, rho_065_profile())
                self.assertIs(annotated.constraint_status, ConstraintStatus.NOT_CHECKED)
                self.assertFalse(annotated.checks[0].checked)
                self.assertEqual(annotated.checks[0].reason, reason)

    def test_non_success_record_is_never_marked_feasible(self):
        def evaluator(payload):
            raise NumericalFailure(
                "solver",
                "did not converge",
                outputs={"rho": 0.10},
            )

        record = make_record(None, evaluator=evaluator)
        self.assertIs(record.calculation_status, CalculationStatus.NUMERICAL_FAILURE)
        annotated = annotate_candidate(record, rho_065_profile())
        self.assertIs(annotated.constraint_status, ConstraintStatus.NOT_CHECKED)
        self.assertIsNone(annotated.feasible)
        self.assertTrue(annotated.checks[0].checked)
        self.assertIs(annotated.checks[0].passed, True)

    def test_minimum_maximum_margins_and_violation(self):
        record = make_record({"work": 12.0})
        inside = annotate_candidate(
            record,
            ConstraintProfile(
                "WINDOW",
                (
                    ConstraintRule(
                        "outputs.work",
                        name="work",
                        minimum=10.0,
                        maximum=20.0,
                        scale=5.0,
                    ),
                ),
            ),
        )
        check = inside.checks[0]
        self.assertAlmostEqual(check.lower_margin, 2.0)
        self.assertAlmostEqual(check.upper_margin, 8.0)
        self.assertAlmostEqual(check.normalized_violation, 0.0)
        self.assertIs(inside.constraint_status, ConstraintStatus.PASS)

        outside_record = make_record({"work": 25.0})
        outside = annotate_candidate(
            outside_record,
            ConstraintProfile(
                "WINDOW",
                (
                    ConstraintRule(
                        "outputs.work",
                        name="work",
                        minimum=10.0,
                        maximum=20.0,
                        scale=5.0,
                    ),
                ),
            ),
        )
        check = outside.checks[0]
        self.assertAlmostEqual(check.upper_margin, -5.0)
        self.assertAlmostEqual(check.normalized_violation, 1.0)
        self.assertIs(outside.constraint_status, ConstraintStatus.FAIL)

    def test_multiple_rules_fail_dominates_and_no_filtering(self):
        record = make_record({"rho": 0.70, "work": 12.0})
        profile = ConstraintProfile(
            "TWO_RULES",
            (
                ConstraintRule("outputs.rho", name="rho", maximum=0.65),
                ConstraintRule("outputs.work", name="work", maximum=20.0),
            ),
        )
        annotated = annotate_candidate(record, profile)
        self.assertIs(annotated.constraint_status, ConstraintStatus.FAIL)
        self.assertIs(annotated.feasible, False)
        annotations = annotate_records([record], profile)
        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0].raw.record_hash, record.record_hash)

    def test_profile_from_mapping_and_duplicate_rule_names(self):
        profile = ConstraintProfile.from_mapping(
            "CUSTOM_UNIT",
            {
                "rho": {"path": "outputs.rho", "maximum": 0.65, "scale": 0.65},
                "work": {"path": "outputs.work", "minimum": 0.0},
            },
        )
        self.assertEqual([rule.name for rule in profile.rules], ["rho", "work"])
        with self.assertRaises(ValueError):
            ConstraintProfile(
                "DUPLICATE",
                (
                    ConstraintRule("outputs.a", name="same", maximum=1.0),
                    ConstraintRule("outputs.b", name="same", maximum=1.0),
                ),
            )
        with self.assertRaises(ValueError):
            ConstraintRule("outputs.value")
        with self.assertRaises(ValueError):
            ConstraintProfile("EMPTY", ())

    def test_annotation_dict_does_not_include_raw_unless_requested(self):
        record = make_record({"rho": 0.60})
        annotated = annotate_candidate(record, rho_065_profile())
        compact = annotated.to_dict()
        self.assertNotIn("raw", compact)
        self.assertEqual(compact["constraint_status"], ConstraintStatus.PASS.value)
        self.assertEqual(compact["checks"][0]["max"], 0.65)
        extended = annotated.to_dict(include_raw=True)
        self.assertEqual(extended["raw"], record.to_dict())

    def test_success_outcome_with_intermediates_can_be_annotated(self):
        def evaluator(payload):
            return EvaluationOutcome.success(
                outputs={"rho": 0.4},
                intermediates={"actual": 1.0},
            )

        record = make_record(None, evaluator=evaluator)
        profile = ConstraintProfile(
            "INTERMEDIATE",
            (ConstraintRule("intermediates.actual", name="actual", maximum=2.0),),
        )
        annotated = annotate_candidate(record, profile)
        self.assertIs(annotated.constraint_status, ConstraintStatus.PASS)


if __name__ == "__main__":
    unittest.main()
