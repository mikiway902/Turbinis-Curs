#!/usr/bin/env python3
"""Deterministic project verification entry point."""
from __future__ import annotations

import argparse
import compileall
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def tests(path: str) -> Check:
    if importlib.util.find_spec("pytest"):
        command = [sys.executable, "-m", "pytest", path, "-q"]
        runner = "pytest"
    else:
        command = [sys.executable, "-m", "unittest", "discover", "-s", path, "-v"]
        runner = "unittest fallback (pytest not installed)"
    result = run(command)
    if result.returncode:
        return Check(path, "FAIL", (result.stdout + result.stderr).strip())
    return Check(path, "PASS", runner)


def baselines_changed() -> bool:
    result = run(["git", "status", "--porcelain", "--", "reference/baselines"])
    return bool(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-baseline-changes",
        action="store_true",
        help="setup/review only; never use this to approve changed physics",
    )
    args = parser.parse_args()
    checks: list[Check] = []

    compiled = all(
        compileall.compile_dir(ROOT / path, quiet=1)
        for path in ("turbine", "tests", "scripts")
    )
    checks.append(Check("Syntax", "PASS" if compiled else "FAIL"))

    imported = run([sys.executable, "-c", "import turbine; import turbine.radial"])
    checks.append(Check("Import", "PASS" if imported.returncode == 0 else "FAIL", imported.stderr.strip()))
    checks.append(tests("tests/unit"))
    checks.append(tests("tests/physics"))
    checks.append(tests("tests/regression"))

    if importlib.util.find_spec("scipy"):
        checks.append(tests("tests/optimization"))
    else:
        checks.append(Check("tests/optimization", "SKIP", "SciPy is not installed in this interpreter"))

    reproduced = run([sys.executable, "scripts/reproduce_baseline.py"])
    checks.append(
        Check(
            "Reproducibility",
            "PASS" if reproduced.returncode == 0 else "FAIL",
            (reproduced.stdout + reproduced.stderr).strip(),
        )
    )
    changed = baselines_changed()
    checks.append(
        Check(
            "Changed baselines",
            "PASS" if not changed or args.allow_baseline_changes else "FAIL",
            "allowed for this setup run" if changed and args.allow_baseline_changes else ("yes" if changed else "0"),
        )
    )

    print("TURBINIS VALIDATION")
    print("===================")
    for check in checks:
        print(f"{check.name:22} {check.status}")
        if check.status == "FAIL" and check.detail:
            print(check.detail)

    failed = any(check.status == "FAIL" for check in checks)
    skipped = any(check.status == "SKIP" for check in checks)
    software_status = "FAIL" if failed else ("PARTIAL" if skipped else "PASS")
    print(f"\nSOFTWARE VALIDATION: {software_status}")
    print("ENGINEERING MODEL: EXPERIMENTAL (stage 3 and stator radial closure open)")
    return 1 if failed else 2 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
