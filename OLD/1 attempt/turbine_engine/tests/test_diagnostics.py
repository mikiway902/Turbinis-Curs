"""Tests for the shared numerical-failure diagnostic record.

The record is a calculation outcome, not an engineering acceptance filter.
These tests only check typing, status separation and JSON sanitization.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.diagnostics import (  # noqa: E402
    NON_PHYSICAL_STATE,
    NUMERICAL_FAILURE,
    NumericalConvergenceError,
    SolverDiagnosticsError,
    sanitize_state,
)


class SanitizeStateTests(unittest.TestCase):
    def test_non_finite_numbers_become_none(self) -> None:
        state = {
            "finite": 1.5,
            "nan": float("nan"),
            "pos_inf": math.inf,
            "neg_inf": -math.inf,
            "integer": 3,
            "flag": True,
            "text": "ok",
            "nested": [1.0, float("inf"), {"x": math.nan}],
        }
        sanitized = sanitize_state(state)
        self.assertEqual(sanitized["finite"], 1.5)
        self.assertIsNone(sanitized["nan"])
        self.assertIsNone(sanitized["pos_inf"])
        self.assertIsNone(sanitized["neg_inf"])
        self.assertEqual(sanitized["integer"], 3)
        self.assertIs(sanitized["flag"], True)
        self.assertEqual(sanitized["text"], "ok")
        self.assertEqual(sanitized["nested"], [1.0, None, {"x": None}])
        json.dumps(sanitized, allow_nan=False)

    def test_unknown_object_is_omitted(self) -> None:
        self.assertIsNone(sanitize_state(object()))
        self.assertEqual(sanitize_state({1: 2.0}), {"1": 2.0})


class DiagnosticRecordTests(unittest.TestCase):
    def test_record_is_json_friendly_and_preserves_stage_reason_status(self) -> None:
        error = NumericalConvergenceError(
            "did not close",
            stage="test.stage",
            reason="iteration_exhausted",
            last_valid_state={"x": 1.0, "bad": math.nan, "items": [2, 3]},
        )
        self.assertIsInstance(error, SolverDiagnosticsError)
        self.assertEqual(error.stage, "test.stage")
        self.assertEqual(error.reason, "iteration_exhausted")
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(
            error.last_valid_state,
            {"x": 1.0, "bad": None, "items": [2, 3]},
        )
        record = error.as_record()
        self.assertEqual(record["error"], "NumericalConvergenceError")
        self.assertEqual(record["stage"], "test.stage")
        self.assertEqual(record["reason"], "iteration_exhausted")
        self.assertEqual(record["status"], NUMERICAL_FAILURE)
        json.dumps(record, allow_nan=False)
        self.assertIn("stage=test.stage", str(error))
        self.assertIn("reason=iteration_exhausted", str(error))
        self.assertIn(NUMERICAL_FAILURE, str(error))

    def test_statuses_stay_distinct(self) -> None:
        self.assertNotEqual(NUMERICAL_FAILURE, NON_PHYSICAL_STATE)
        error = NumericalConvergenceError(
            "non-physical state",
            stage="test.stage",
            reason="invalid_state",
            status=NON_PHYSICAL_STATE,
        )
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertEqual(error.as_record()["status"], NON_PHYSICAL_STATE)

    def test_unknown_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NumericalConvergenceError(
                "bad status",
                stage="test.stage",
                reason="invalid_status",
                status="ENGINEERING_REJECTION",
            )


if __name__ == "__main__":
    unittest.main()
