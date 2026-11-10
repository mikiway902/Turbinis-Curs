"""Focused tests for candidate-attempt provenance and optional diagnostics.

These tests do not run an optimizer, sweep, or differential evolution. They
exercise the exception boundaries directly and one sizing metadata path with a
fake physical result. Software-drift diagnostics only, not physical validation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

import optimizer_core as oc  # noqa: E402


class _StructuredFailure(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.status = "failed"
        self.stage = "stage-2"
        self.reason = "negative_sqrt"
        self.last_valid_state = {"rho": 0.3}


class AttemptArchiveTests(unittest.TestCase):
    def test_calculate_failure_retains_inputs_and_structured_error(self):
        cfg = {"mode": "B", "limits": {"a1_min": 11.0}, "nested": [1, 2, 3]}
        x = np.array([1.0, 2.0, 3.0])
        archive = oc.Archive()
        exc = _StructuredFailure("stage failed")

        with patch.object(oc, "evaluate", side_effect=exc):
            score = oc._score_x(x, object(), cfg, True, archive)

        self.assertEqual(score, 1e9)
        self.assertEqual(len(archive.rows), 1)
        row = archive.rows[0]
        self.assertIsNone(row["res"])
        self.assertEqual(row["checks"], [])
        self.assertEqual(row["phase"], "calculate")
        self.assertEqual(row["score"], 1e9)
        self.assertFalse(row["feasible"])
        self.assertFalse(row["strict"])
        self.assertTrue(row["attempt"])
        self.assertEqual(row["full"], True)
        self.assertEqual(row["error"]["type"], "_StructuredFailure")
        self.assertEqual(row["error"]["status"], "failed")
        self.assertEqual(row["error"]["stage"], "stage-2")
        self.assertEqual(row["error"]["reason"], "negative_sqrt")
        self.assertEqual(row["error"]["last_valid_state"], {"rho": 0.3})
        self.assertIsNot(row["x"], x)
        self.assertIsNot(row["cfg"], cfg)
        np.testing.assert_array_equal(row["x"], x)
        self.assertEqual(row["cfg"], cfg)

        # The attempt owns a copy: later mutation of the caller's input and
        # config must not rewrite the archived record.
        x[0] = 99.0
        cfg["nested"][0] = 99
        np.testing.assert_array_equal(row["x"], [1.0, 2.0, 3.0])
        self.assertEqual(row["cfg"]["nested"], [1, 2, 3])

    def test_validate_failure_retains_computed_result(self):
        res = {"marker": "computed", "stages": []}
        cfg = {"mode": "A"}
        archive = oc.Archive()

        with patch.object(oc, "evaluate", return_value=res), \
             patch.object(oc, "validate", side_effect=ValueError("bad check")):
            score = oc._score_x(np.array([1.0]), object(), cfg, False, archive)

        self.assertEqual(score, 1e9)
        row = archive.rows[0]
        self.assertIs(row["res"], res)
        self.assertEqual(row["phase"], "validate")
        self.assertEqual(row["error"]["type"], "ValueError")
        self.assertEqual(row["error"]["message"], "bad check")
        self.assertFalse(row["feasible"])

    def test_penalty_failure_retains_result_and_checks(self):
        res = {"marker": "computed", "stages": []}
        checks = [dict(name="hard", kind="hard", ok=False, amount=1.0)]
        archive = oc.Archive()

        with patch.object(oc, "evaluate", return_value=res), \
             patch.object(oc, "validate", return_value=checks), \
             patch.object(oc, "penalty", side_effect=RuntimeError("score bug")):
            score = oc._score_x(np.array([1.0]), object(), {}, True, archive)

        self.assertEqual(score, 1e9)
        row = archive.rows[0]
        self.assertIs(row["res"], res)
        self.assertIs(row["checks"], checks)
        self.assertEqual(row["phase"], "penalty")
        self.assertEqual(row["error"]["message"], "score bug")
        self.assertFalse(row["feasible"])

    def test_keyboard_interrupt_is_not_converted_to_a_record(self):
        archive = oc.Archive()
        with patch.object(oc, "evaluate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                oc._score_x(np.array([1.0]), object(), {}, False, archive)
        self.assertEqual(archive.rows, [])

    def test_success_keeps_old_fields_and_adds_optional_provenance(self):
        res = {"marker": "ok", "stages": []}
        checks = [dict(name="ok", kind="hard", ok=True, amount=0.0)]
        archive = oc.Archive()
        x = np.array([1.0, 2.0])

        with patch.object(oc, "evaluate", return_value=res), \
             patch.object(oc, "validate", return_value=checks), \
             patch.object(oc, "penalty", return_value=0.25):
            score = oc._score_x(x, object(), {"mode": "A"}, False, archive)

        self.assertEqual(score, 0.25)
        row = archive.rows[0]
        self.assertIs(row["res"], res)
        self.assertIs(row["checks"], checks)
        self.assertEqual(row["score"], 0.25)
        self.assertTrue(row["feasible"])
        self.assertTrue(row["strict"])
        self.assertEqual(row["phase"], "score")
        np.testing.assert_array_equal(row["x"], x)

    def test_empty_checks_are_not_treated_as_feasible(self):
        # This guards the old all([]) == True trap on failure/partial records.
        self.assertFalse(oc.is_feasible([]))
        self.assertFalse(oc.is_strict([]))
        archive = oc.Archive()
        archive.add(None, [], 1e9)
        self.assertFalse(archive.rows[0]["feasible"])
        self.assertFalse(archive.rows[0]["strict"])


class SizingGraphDiagnosticTests(unittest.TestCase):
    def test_out_of_graph_uc0_opt_becomes_not_checked_without_losing_physics(self):
        stages = [
            dict(D_л=0.100, l_л=0.020, u1=100.0, u2=110.0, H0=10_000.0, ρ=0.80)
            for _ in range(3)
        ]
        fake = {"stages": stages}
        cfg = {}

        with patch.object(oc, "evaluate", return_value=fake), \
             patch.object(
                 oc.cs, "uc0_opt_from_rho",
                 side_effect=ValueError("rho (рис. 18) вне диапазона"),
             ):
            result = oc.evaluate_sizing(None, cfg, full=False, max_iter=0)

        sizing = result["sizing"]
        self.assertEqual(sizing["u_c0_opt"], [None, None, None])
        self.assertEqual(sizing["u_c0_opt_status"], ["NOT_CHECKED"] * 3)
        self.assertTrue(all("рис. 18" in reason for reason in sizing["u_c0_opt_reason"]))
        # Physical sizing fields are still computed and finite.
        self.assertEqual(len(sizing["u_c0"]), 3)
        self.assertTrue(all(np.isfinite(v) for v in sizing["u_c0"]))
        self.assertEqual(len(sizing["D_hub"]), 3)


if __name__ == "__main__":
    unittest.main()
