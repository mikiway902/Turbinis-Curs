"""Focused tests for scripts/profile_engine_paths.py.

These tests do not load the optimizer model, execute a candidate, run an
optimizer, or write a real profile.  They exercise the pure timing/summary and
parity helpers, CLI validation, refusal to overwrite, and the lazy-import
boundary.  The parent task executes the real bounded profile separately.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import profile_engine_paths as profile  # noqa: E402


class RepeatsValidationTests(unittest.TestCase):
    def test_accepts_bounds(self):
        self.assertEqual(profile.positive_int_max10("1"), 1)
        self.assertEqual(profile.positive_int_max10("10"), 10)

    def test_rejects_zero_above_max_and_non_integers(self):
        for value in ("0", "11", "-1", "3.5", "abc", ""):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    profile.positive_int_max10(value)

    def test_cli_default_repeats_is_three(self):
        args = profile._build_parser().parse_args(["--output", "profile.json"])
        self.assertEqual(args.repeats, 3)


class TimingHelperTests(unittest.TestCase):
    def test_measure_repeated_uses_timer_for_each_repeat(self):
        ticks = iter([10.0, 10.5, 20.0, 20.25])
        timer = mock.Mock(side_effect=lambda: next(ticks))

        values, durations = profile.measure_repeated(
            lambda: "value", 2, timer=timer
        )

        self.assertEqual(values, ["value", "value"])
        self.assertEqual(durations, [0.5, 0.25])
        self.assertEqual(timer.call_count, 4)

    def test_measure_repeated_rejects_non_positive_repeats(self):
        with self.assertRaises(ValueError):
            profile.measure_repeated(lambda: None, 0)

    def test_summarize_timings_reports_median_min_max(self):
        summary = profile.summarize_timings([0.3, 0.1, 0.2])

        self.assertEqual(summary["repeats"], 3)
        self.assertEqual(summary["seconds"], [0.3, 0.1, 0.2])
        self.assertAlmostEqual(summary["median_seconds"], 0.2)
        self.assertEqual(summary["min_seconds"], 0.1)
        self.assertEqual(summary["max_seconds"], 0.3)

    def test_summarize_timings_handles_empty_and_rejects_non_finite(self):
        empty = profile.summarize_timings([])
        self.assertEqual(empty["repeats"], 0)
        self.assertIsNone(empty["median_seconds"])

        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    profile.summarize_timings([value])


class SignatureTests(unittest.TestCase):
    def test_extract_signature_keeps_only_documented_scalars(self):
        result = {field: float(index + 1) for index, field in enumerate(
            profile.SIGNATURE_FIELDS
        )}
        result["N"] = 110e3
        result["sigma"] = math.nan
        result["a2"] = [1.0] * 100_000
        result["stages"] = [{"unused": [0.0] * 100_000}]

        signature = profile.extract_signature(result)

        self.assertEqual(set(signature), set(profile.SIGNATURE_FIELDS))
        self.assertEqual(signature["N"], 110e3)
        self.assertIsNone(signature["sigma"])
        self.assertNotIn("a2", signature)
        self.assertNotIn("stages", signature)

    def test_compare_signatures_passes_at_tight_tolerance(self):
        base = {field: 1.0 for field in profile.SIGNATURE_FIELDS}
        almost = dict(base)
        almost["N"] = 1.0 + 5e-13

        comparison = profile.compare_signatures(base, almost)

        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(comparison["failed_fields"], [])
        self.assertLessEqual(comparison["max_abs_diff"], 1e-12)
        self.assertLessEqual(comparison["max_rel_diff"], 1e-12)

    def test_compare_signatures_reports_mismatch_and_missing_values(self):
        base = {field: 1.0 for field in profile.SIGNATURE_FIELDS}
        changed = dict(base)
        changed["lc1"] = 1.0 + 1e-6
        changed["p2z"] = None

        comparison = profile.compare_signatures(base, changed)

        self.assertEqual(comparison["status"], "mismatch")
        self.assertIn("lc1", comparison["failed_fields"])
        self.assertIn("p2z", comparison["failed_fields"])
        self.assertFalse(comparison["fields"]["lc1"]["passed"])
        self.assertEqual(
            comparison["fields"]["p2z"]["reason"],
            "missing or non-finite value",
        )

    def test_compare_signatures_zero_values_pass(self):
        base = {field: 0.0 for field in profile.SIGNATURE_FIELDS}
        comparison = profile.compare_signatures(base, base)
        self.assertEqual(comparison["status"], "pass")
        self.assertEqual(comparison["max_rel_diff"], 0.0)


class PathHarnessTests(unittest.TestCase):
    def test_build_ns_tuple_is_unpacked_before_evaluation(self):
        seen_namespaces = []

        def fake_build_ns(model, *, q_n_limit, orchestration_core):
            self.assertEqual(q_n_limit, profile.Q_N_LIMIT)
            self.assertTrue(orchestration_core)
            return {"namespace": True}, {"cycle": True}

        def fake_eval_candidate(x, namespace, scenario, q_n_limit):
            self.assertIsInstance(namespace, dict)
            seen_namespaces.append(namespace)
            return {
                "status": "ok",
                "err": "",
                **{field: 1.0 for field in profile.SIGNATURE_FIELDS},
            }

        fake_oe = mock.Mock(
            build_ns=fake_build_ns,
            eval_candidate=fake_eval_candidate,
        )

        stage = profile._profile_one_path(
            fake_oe,
            model=object(),
            repeats=2,
            path_name="extracted_const_hub",
            orchestration_core=True,
        )

        self.assertEqual(stage["setup"]["status"], "ok")
        self.assertEqual(stage["warmup"]["status"], "ok")
        self.assertEqual(stage["measured"]["status"], "ok")
        self.assertEqual(stage["measured"]["repeats"], 2)
        self.assertEqual(stage["cprofile"]["status"], "ok")
        self.assertEqual(stage["cprofile"]["profiled_calls"], 2)
        self.assertEqual(stage["result_status"], "ok")
        self.assertTrue(stage["parity_ready"])
        self.assertEqual(len(seen_namespaces), 5)


class ProfileFormattingTests(unittest.TestCase):
    def test_profile_entries_sorts_by_cumulative_time_and_relativizes_files(self):
        slow_file = str(ROOT / "Optimizer" / "a.py")
        fast_file = str(ROOT / "Optimizer" / "b.py")
        stats = {
            (fast_file, 20, "fast"): (1, 3, 0.2, 0.5, {}),
            (slow_file, 10, "slow"): (2, 4, 0.1, 1.0, {}),
        }

        rows = profile.profile_entries(stats, root=ROOT, limit=20)

        self.assertEqual([row["function"] for row in rows], ["slow", "fast"])
        self.assertEqual(rows[0]["file"], "Optimizer/a.py")
        self.assertEqual(rows[0]["calls"], 4)
        self.assertEqual(rows[0]["primitive_calls"], 2)
        self.assertAlmostEqual(rows[0]["self_seconds"], 0.1)
        self.assertAlmostEqual(rows[0]["cumulative_seconds"], 1.0)

    def test_profile_entries_respects_limit(self):
        stats = {
            (str(ROOT / f"m{index}.py"), index, f"f{index}"): (
                1,
                1,
                0.0,
                float(index),
                {},
            )
            for index in range(5)
        }

        rows = profile.profile_entries(stats, root=ROOT, limit=3)

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["function"], "f4")

    def test_sanitize_replaces_non_finite_values(self):
        notes: list[dict[str, str]] = []
        cleaned = profile._sanitize(
            {"ok": 1.0, "bad": math.nan, "items": [math.inf]}, "$", notes
        )

        self.assertEqual(cleaned["ok"], 1.0)
        self.assertIsNone(cleaned["bad"])
        self.assertIsNone(cleaned["items"][0])
        self.assertEqual(len(notes), 2)


class CliTests(unittest.TestCase):
    def test_help_exits_zero_and_documents_required_arguments(self):
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "profile_engine_paths.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output", completed.stdout)
        self.assertIn("--repeats", completed.stdout)
        self.assertIn("required", completed.stdout)

    def test_existing_output_is_refused_and_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile.json"
            output.write_text("keep-me", encoding="utf-8")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                code = profile.main(["--output", str(output)])

            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep-me")
            self.assertIn("will not be overwritten", stderr.getvalue())

    def test_invalid_repeats_is_rejected_before_the_model_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "profile.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    profile.main(
                        ["--output", str(output), "--repeats", "11"]
                    )

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("repeats", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_missing_output_is_rejected(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                profile.main([])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--output", stderr.getvalue())

    def test_module_import_does_not_import_numpy_or_scipy(self):
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(SCRIPT_DIR)!r})\n"
            "import profile_engine_paths\n"
            "assert 'numpy' not in sys.modules, 'numpy imported eagerly'\n"
            "assert 'scipy' not in sys.modules, 'scipy imported eagerly'\n"
            "print('ok')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
