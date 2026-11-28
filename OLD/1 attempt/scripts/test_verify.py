"""Standard-library tests for scripts/verify.py."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import verify  # noqa: E402


def valid_notebook(cells: list[dict] | None = None) -> dict:
    return {
        "cells": cells
        if cells is not None
        else [
            {
                "cell_type": "markdown",
                "id": "intro",
                "metadata": {},
                "source": ["# Test notebook"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "code-1",
                "metadata": {},
                "outputs": [],
                "source": ["print('ok')"],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def populate_required(root: Path, corrupt: str | None = None) -> None:
    notebook_paths = {relative for relative, _ in verify.CORE_NOTEBOOKS}
    for relative, _description in verify.REQUIRED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if corrupt is not None and relative == corrupt:
            path.write_text("{this is not valid json", encoding="utf-8")
        elif relative in notebook_paths:
            path.write_text(json.dumps(valid_notebook()), encoding="utf-8")
        else:
            path.write_text("placeholder\n", encoding="utf-8")


class NotebookValidationTests(unittest.TestCase):
    def test_valid_notebook_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "book.ipynb"
            path.write_text(json.dumps(valid_notebook()), encoding="utf-8")

            check = verify.check_notebook(path)

            self.assertEqual(check.status, verify.PASS)
            self.assertIn("2 cells", check.detail)

    def test_malformed_notebook_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "book.ipynb"
            path.write_text("{not valid json", encoding="utf-8")

            check = verify.check_notebook(path)

            self.assertEqual(check.status, verify.FAIL)
            self.assertIn("invalid notebook JSON", check.detail)

    def test_notebook_without_cells_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            notebook = valid_notebook()
            del notebook["cells"]
            path = Path(temporary) / "book.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")

            check = verify.check_notebook(path)

            self.assertEqual(check.status, verify.FAIL)
            self.assertIn("cells", check.detail)

    def test_code_cell_without_outputs_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            notebook = valid_notebook()
            del notebook["cells"][1]["outputs"]
            path = Path(temporary) / "book.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")

            check = verify.check_notebook(path)

            self.assertEqual(check.status, verify.FAIL)
            self.assertIn("outputs", check.detail)

    def test_duplicate_cell_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            notebook = valid_notebook()
            notebook["cells"][1]["id"] = notebook["cells"][0]["id"]
            path = Path(temporary) / "book.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")

            check = verify.check_notebook(path)

            self.assertEqual(check.status, verify.FAIL)
            self.assertIn("duplicates id", check.detail)

    def test_non_string_cell_type_fails_without_crashing(self) -> None:
        for invalid_type in ([], {}):
            with self.subTest(cell_type=invalid_type), tempfile.TemporaryDirectory() as temporary:
                notebook = valid_notebook()
                notebook["cells"][0]["cell_type"] = invalid_type
                path = Path(temporary) / "book.ipynb"
                path.write_text(json.dumps(notebook), encoding="utf-8")

                check = verify.check_notebook(path)

                self.assertEqual(check.status, verify.FAIL)
                self.assertIn("invalid cell_type", check.detail)

    def test_check_notebook_does_not_modify_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "book.ipynb"
            payload = json.dumps(valid_notebook()).encode("utf-8")
            path.write_bytes(payload)

            verify.check_notebook(path)

            self.assertEqual(path.read_bytes(), payload)


class ReportStatusTests(unittest.TestCase):
    def test_empty_report_warns(self) -> None:
        report = verify.Report(Path("."))

        self.assertEqual(report.status, verify.WARN)
        self.assertEqual(report.exit_code(), 0)
        self.assertEqual(report.exit_code(strict=True), 2)

    def test_fail_dominates_warnings(self) -> None:
        report = verify.Report(Path("."))
        report.add(verify.Check("a", verify.PASS))
        report.add(verify.Check("b", verify.WARN))
        report.add(verify.Check("c", verify.FAIL))

        self.assertEqual(report.status, verify.FAIL)
        self.assertEqual(report.exit_code(), 1)
        self.assertEqual(report.exit_code(strict=True), 1)

    def test_warning_exit_depends_on_strict(self) -> None:
        report = verify.Report(Path("."))
        report.add(verify.Check("a", verify.WARN))

        self.assertEqual(report.status, verify.WARN)
        self.assertEqual(report.exit_code(), 0)
        self.assertEqual(report.exit_code(strict=True), 2)

    def test_all_pass_exits_zero_even_in_strict_mode(self) -> None:
        report = verify.Report(Path("."))
        report.add(verify.Check("a", verify.PASS))

        self.assertEqual(report.status, verify.PASS)
        self.assertEqual(report.exit_code(), 0)
        self.assertEqual(report.exit_code(strict=True), 0)


class RepositoryCheckTests(unittest.TestCase):
    def test_missing_required_paths_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            report = verify.run_verification(root)

            self.assertEqual(report.status, verify.FAIL)
            self.assertEqual(report.exit_code(), 1)

    def test_complete_required_files_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_required(root)

            report = verify.run_verification(root)

            required = next(
                check for check in report.checks if check.name == "required files"
            )
            self.assertEqual(required.status, verify.PASS)
            self.assertEqual(report.count(verify.FAIL), 0)

    def test_malformed_core_notebook_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_required(
                root, corrupt="Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb"
            )

            report = verify.run_verification(root)

            self.assertEqual(report.status, verify.FAIL)
            self.assertEqual(report.exit_code(), 1)
            notebook_failures = [
                check
                for check in report.checks
                if check.status == verify.FAIL and check.name.startswith("notebook:")
            ]
            self.assertEqual(len(notebook_failures), 1)

    def test_missing_authority_paths_warn_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_required(root)

            report = verify.run_verification(root)

            authority_checks = [
                check
                for check in report.checks
                if check.name.startswith("authority:")
            ]
            self.assertTrue(authority_checks)
            self.assertTrue(
                all(check.status == verify.WARN for check in authority_checks)
            )
            self.assertEqual(report.count(verify.FAIL), 0)

    def test_present_authority_path_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "docs" / "engineering" / "README.md"
            path.parent.mkdir(parents=True)
            path.write_text("authority\n", encoding="utf-8")

            checks = verify._check_authority_paths(root)

            readme = next(
                check
                for check in checks
                if check.name == "authority: docs/engineering/README.md"
            )
            self.assertEqual(readme.status, verify.PASS)


class BaselineCheckTests(unittest.TestCase):
    def _write_baseline(self, directory: Path, payload: dict | None = None) -> Path:
        path = directory / "BOOK-APPROX-001.json"
        path.write_text(
            json.dumps(payload or {"id": "BOOK-APPROX-001", "values": [1, 2]}),
            encoding="utf-8",
        )
        return path

    def test_missing_baseline_directory_warns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checks = verify.check_baselines(Path(temporary))

            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0].status, verify.WARN)

    def test_baseline_is_parsed_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / verify.BASELINE_DIR
            directory.mkdir(parents=True)
            path = self._write_baseline(directory)
            before = path.read_bytes()

            checks = verify.check_baselines(root)

            self.assertEqual(path.read_bytes(), before)
            baseline_check = next(
                check for check in checks if check.name.startswith("baseline:")
            )
            self.assertEqual(baseline_check.status, verify.PASS)

    def test_malformed_baseline_json_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / verify.BASELINE_DIR
            directory.mkdir(parents=True)
            (directory / "broken.json").write_text("{", encoding="utf-8")

            checks = verify.check_baselines(root)

            self.assertTrue(any(check.status == verify.FAIL for check in checks))

    def test_manifest_mismatch_fails_unless_changes_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / verify.BASELINE_DIR
            directory.mkdir(parents=True)
            path = self._write_baseline(directory)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  {path.name}\n", encoding="utf-8"
            )

            matching = verify.check_baselines(root)
            self.assertTrue(any(check.status == verify.PASS for check in matching))

            path.write_text(
                json.dumps({"id": "BOOK-APPROX-001", "values": [9, 9]}),
                encoding="utf-8",
            )
            changed = verify.check_baselines(root)
            self.assertTrue(any(check.status == verify.FAIL for check in changed))

            allowed = verify.check_baselines(root, allow_changes=True)
            self.assertFalse(any(check.status == verify.FAIL for check in allowed))
            self.assertTrue(any(check.status == verify.WARN for check in allowed))

    def test_manifest_detects_when_all_baselines_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / verify.BASELINE_DIR
            directory.mkdir(parents=True)
            baseline = self._write_baseline(directory)
            digest = hashlib.sha256(baseline.read_bytes()).hexdigest()
            (directory / "SHA256SUMS").write_text(
                f"{digest}  {baseline.name}\n", encoding="utf-8"
            )
            baseline.unlink()

            checks = verify.check_baselines(root)

            integrity = next(check for check in checks if check.name == "baseline integrity")
            self.assertEqual(integrity.status, verify.FAIL)
            self.assertIn("listed in manifest but missing", integrity.detail)


class CommandLineTests(unittest.TestCase):
    def test_help_succeeds(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "verify.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("Static repository verifier", completed.stdout)
        self.assertIn("--allow-baseline-changes", completed.stdout)

    def test_cli_malformed_notebook_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_required(
                root, corrupt="Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb"
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "verify.py"),
                    "--root",
                    str(root),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("RESULT: FAIL", completed.stdout)
            self.assertIn("Physics: NOT VERIFIED", completed.stdout)

    def test_cli_strict_turns_warnings_into_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            populate_required(root)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "verify.py"),
                    "--root",
                    str(root),
                    "--strict",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("RESULT: WARN", completed.stdout)


if __name__ == "__main__":
    unittest.main()
