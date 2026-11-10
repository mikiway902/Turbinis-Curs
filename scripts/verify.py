#!/usr/bin/env python3
"""Static repository verifier for Turbinis-Curs.

This command checks repository structure and core-notebook JSON integrity only.
It does not execute notebooks, formulae, optimizers, sweeps, or validation
cases, and it does not verify physics.

Exit codes:
  0  all checks passed (warnings may be present unless --strict is used)
  1  one or more checks failed
  2  warnings were present and --strict was requested
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

_STATUS_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Files that are present in this checkout and are needed to navigate/operate it.
# Authority documents and baseline data are checked separately because they may
# legitimately be absent from a partial checkout.
REQUIRED_PATHS: tuple[tuple[str, str], ...] = (
    ("README.md", "project overview and verifier contract"),
    ("docs/README.md", "documentation index"),
    ("reference/README.md", "baseline and published-case policy"),
    (
        "reference/published_cases/README.md",
        "published validation-case policy",
    ),
    (
        "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb",
        "legacy book_approx meanline notebook",
    ),
    (
        "Turbinis-Cycle/Turbinis-Cycle.ipynb",
        "cycle boundary-condition notebook",
    ),
    ("Optimizer/optimizer_core.py", "shared optimizer core"),
    ("Turbinis-Calc/constraints.py", "constraint source module"),
)

CORE_NOTEBOOKS: tuple[tuple[str, str], ...] = (
    (
        "Turbinis-Calc/Main_Turbinis_Calc_Book.ipynb",
        "legacy book_approx meanline notebook",
    ),
    (
        "Turbinis-Cycle/Turbinis-Cycle.ipynb",
        "cycle boundary-condition notebook",
    ),
)

# Paths named by README.md or docs/README.md as authority or canonical
# calculation sources. Absence is reported as a warning, never as physical
# validation and never as a reason to invent content.
AUTHORITY_PATHS: tuple[tuple[str, str], ...] = (
    ("docs/engineering/README.md", "equation-status authority"),
    ("docs/engineering/METHOD.md", "calculation-layer and change-gate authority"),
    ("docs/engineering/CONSTRAINTS.md", "hard-limit authority"),
    ("docs/engineering/EQUATIONS.md", "accepted-equation authority"),
    (
        "docs/engineering/VALIDATION_CASES.md",
        "required solver-evidence authority",
    ),
    ("turbine_engine/", "new extracted calculation package"),
    (
        "turbine_engine/radial/equilibrium.py",
        "accepted static-pressure radial equilibrium",
    ),
)

BASELINE_DIR = Path("reference") / "baselines"
BASELINE_MANIFEST_NAMES = ("SHA256SUMS", "sha256sums.txt", "baselines.sha256")

VALID_CELL_TYPES = frozenset({"code", "markdown", "raw"})


@dataclass(frozen=True)
class Check:
    """One independently reported verification result."""

    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    """Collection of checks with aggregate PASS/WARN/FAIL semantics."""

    root: Path
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    def extend(self, checks: Iterable[Check]) -> None:
        self.checks.extend(checks)

    @property
    def status(self) -> str:
        if not self.checks:
            return WARN
        return max(
            (check.status for check in self.checks),
            key=_STATUS_RANK.__getitem__,
        )

    def count(self, status: str) -> int:
        return sum(1 for check in self.checks if check.status == status)

    def exit_code(self, strict: bool = False) -> int:
        status = self.status
        if status == FAIL:
            return 1
        if status == WARN and strict:
            return 2
        return 0


def repository_root() -> Path:
    """Return the repository root relative to this script, never a CWD guess."""

    return Path(__file__).resolve().parent.parent


def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _check_required_paths(root: Path) -> Check:
    if not root.is_dir():
        return Check(
            "required files",
            FAIL,
            f"repository root is not a directory: {root}",
        )

    missing = [
        f"{relative} ({description})"
        for relative, description in REQUIRED_PATHS
        if not (root / relative).is_file()
    ]
    if missing:
        return Check(
            "required files",
            FAIL,
            "missing required path(s): " + "; ".join(missing),
        )
    return Check(
        "required files",
        PASS,
        f"{len(REQUIRED_PATHS)}/{len(REQUIRED_PATHS)} required path(s) present",
    )


def _cell_source_is_valid(source: object) -> bool:
    if isinstance(source, str):
        return True
    if isinstance(source, list):
        return all(isinstance(part, str) for part in source)
    return False


def check_notebook(path: Path, root: Path | None = None) -> Check:
    """Validate one notebook as read-only JSON using only the standard library."""

    display_root = root if root is not None else path.parent
    name = f"notebook: {_relative(display_root, path)}"

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return Check(name, FAIL, f"cannot read notebook: {exc}")

    try:
        notebook = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return Check(name, FAIL, f"invalid notebook JSON: {exc}")

    problems: list[str] = []
    if not isinstance(notebook, dict):
        return Check(name, FAIL, "notebook JSON must be a top-level object")

    for key in ("cells", "metadata", "nbformat", "nbformat_minor"):
        if key not in notebook:
            problems.append(f"missing top-level key {key!r}")

    nbformat = notebook.get("nbformat")
    nbformat_minor = notebook.get("nbformat_minor")
    if nbformat != 4:
        problems.append(f"unsupported nbformat {nbformat!r} (expected 4)")
    if not isinstance(nbformat_minor, int) or nbformat_minor < 0:
        problems.append("nbformat_minor must be a non-negative integer")

    metadata = notebook.get("metadata")
    if not isinstance(metadata, dict):
        problems.append("metadata must be an object")
    else:
        kernelspec = metadata.get("kernelspec")
        if not isinstance(kernelspec, dict) or not isinstance(
            kernelspec.get("name"), str
        ):
            problems.append("metadata.kernelspec.name must be a string")
        language_info = metadata.get("language_info")
        if not isinstance(language_info, dict):
            problems.append("metadata.language_info must be an object")
        elif language_info.get("name") != "python":
            problems.append("metadata.language_info.name must be 'python'")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        problems.append("cells must be a non-empty list")
        cells = []

    code_cells = 0
    markdown_cells = 0
    cell_ids: dict[str, int] = {}

    for index, cell in enumerate(cells):
        prefix = f"cell {index}"
        if not isinstance(cell, dict):
            problems.append(f"{prefix} must be an object")
            continue

        cell_type = cell.get("cell_type")
        if not isinstance(cell_type, str) or cell_type not in VALID_CELL_TYPES:
            problems.append(f"{prefix} has invalid cell_type {cell_type!r}")
            continue
        if cell_type == "code":
            code_cells += 1
        elif cell_type == "markdown":
            markdown_cells += 1

        if not _cell_source_is_valid(cell.get("source")):
            problems.append(f"{prefix} source must be a string or list of strings")

        if cell_type == "code":
            if not isinstance(cell.get("outputs"), list):
                problems.append(f"{prefix} code cell outputs must be a list")
            execution_count = cell.get("execution_count")
            if execution_count is not None and not isinstance(execution_count, int):
                problems.append(
                    f"{prefix} execution_count must be an integer or null"
                )

        cell_id = cell.get("id")
        if isinstance(nbformat_minor, int) and nbformat_minor >= 5:
            if not isinstance(cell_id, str) or not cell_id:
                problems.append(f"{prefix} id must be a non-empty string")
            else:
                if cell_id in cell_ids:
                    problems.append(
                        f"{prefix} duplicates id {cell_id!r} "
                        f"from cell {cell_ids[cell_id]}"
                    )
                else:
                    cell_ids[cell_id] = index

    if problems:
        preview = "; ".join(problems[:5])
        suffix = "" if len(problems) <= 5 else f"; and {len(problems) - 5} more"
        return Check(name, FAIL, preview + suffix)

    return Check(
        name,
        PASS,
        f"nbformat {nbformat}.{nbformat_minor}, {len(cells)} cells "
        f"({code_cells} code, {markdown_cells} markdown)",
    )


def _check_authority_paths(root: Path) -> list[Check]:
    checks: list[Check] = []
    for relative, description in AUTHORITY_PATHS:
        path = root / relative
        if path.exists():
            checks.append(
                Check(
                    f"authority: {relative}",
                    PASS,
                    f"{description} present (not executed)",
                )
            )
        else:
            checks.append(
                Check(
                    f"authority: {relative}",
                    WARN,
                    f"{description} absent from this checkout; no content was "
                    "created or inferred and no physics was verified",
                )
            )
    return checks


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _check_runtime_dependencies() -> list[Check]:
    checks = [
        Check(
            "runtime: python",
            PASS,
            f"{sys.version_info.major}.{sys.version_info.minor}."
            f"{sys.version_info.micro}",
        )
    ]
    if _module_available("numpy"):
        checks.append(
            Check(
                "runtime: numpy",
                PASS,
                "importable (not imported or exercised by this verifier)",
            )
        )
    else:
        checks.append(
            Check(
                "runtime: numpy",
                WARN,
                "not importable; calculation and optimizer paths cannot run. "
                "Only static checks were performed.",
            )
        )
    return checks


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _read_hash_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"{path.name}:{line_number}: expected '<sha256>  <filename>'"
            )
        digest, filename = parts
        filename = filename.lstrip("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise ValueError(
                f"{path.name}:{line_number}: invalid SHA-256 digest {digest!r}"
            )
        entries[filename] = digest.lower()
    return entries


def _check_baseline_manifest(
    directory: Path, json_files: Sequence[Path], allow_changes: bool
) -> list[Check]:
    manifest = next(
        (
            directory / name
            for name in BASELINE_MANIFEST_NAMES
            if (directory / name).is_file()
        ),
        None,
    )
    if manifest is None:
        return [
            Check(
                "baseline integrity",
                WARN,
                "no SHA256SUMS manifest found; baseline files were parsed "
                "read-only but value drift is not detectable",
            )
        ]

    try:
        entries = _read_hash_manifest(manifest)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [
            Check(
                "baseline integrity",
                FAIL,
                f"cannot parse {manifest.name}: {exc}",
            )
        ]

    problems: list[str] = []
    for filename, expected_digest in sorted(entries.items()):
        candidate = directory / filename
        if not _is_within(candidate, directory):
            problems.append(f"{filename}: path escapes the baseline directory")
            continue
        if not candidate.is_file():
            problems.append(f"{filename}: listed in manifest but missing")
            continue
        try:
            actual_digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        except OSError as exc:
            problems.append(f"{filename}: cannot read: {exc}")
            continue
        if actual_digest != expected_digest:
            problems.append(
                f"{filename}: sha256 {actual_digest[:16]}... does not match "
                f"manifest {expected_digest[:16]}..."
            )

    listed = {Path(filename).as_posix() for filename in entries}
    for path in json_files:
        relative = path.relative_to(directory).as_posix()
        if relative not in listed:
            problems.append(f"{relative}: present but not listed in manifest")

    if not problems:
        return [
            Check(
                "baseline integrity",
                PASS,
                f"{len(entries)} baseline file(s) match {manifest.name}",
            )
        ]

    status = WARN if allow_changes else FAIL
    detail = "; ".join(problems)
    if allow_changes:
        detail += (
            "; --allow-baseline-changes is for setup/review only and is not "
            "approval of changed physics"
        )
    return [Check("baseline integrity", status, detail)]


def check_baselines(root: Path, allow_changes: bool = False) -> list[Check]:
    """Parse frozen baselines read-only, if present, without modifying them."""

    directory = root / BASELINE_DIR
    if not directory.is_dir():
        return [
            Check(
                "reference/baselines",
                WARN,
                "optional frozen-baseline directory absent; no baseline values "
                "were created or inferred",
            )
        ]

    json_files = sorted(
        path for path in directory.glob("*.json") if path.is_file()
    )
    if not json_files:
        checks = [
            Check(
                "reference/baselines",
                WARN,
                "baseline directory exists but contains no *.json files",
            )
        ]
        # An existing manifest remains authoritative even if all listed files
        # have disappeared; don't let an empty directory bypass integrity checks.
        if any((directory / name).is_file() for name in BASELINE_MANIFEST_NAMES):
            checks.extend(_check_baseline_manifest(directory, json_files, allow_changes))
        return checks

    checks: list[Check] = []
    for path in json_files:
        name = f"baseline: {_relative(root, path)}"
        try:
            raw = path.read_bytes()
            baseline = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            checks.append(
                Check(name, FAIL, f"cannot read/parse baseline JSON: {exc}")
            )
            continue
        if not isinstance(baseline, dict):
            checks.append(
                Check(name, FAIL, "baseline JSON must be a top-level object")
            )
            continue

        digest = hashlib.sha256(raw).hexdigest()
        identifier = (
            baseline.get("id")
            or baseline.get("case_id")
            or baseline.get("name")
        )
        label = f"id={identifier!r}, " if isinstance(identifier, str) else ""
        checks.append(
            Check(
                name,
                PASS,
                f"read-only JSON object, {label}{len(baseline)} top-level key(s), "
                f"sha256 {digest[:16]}...",
            )
        )

    checks.extend(_check_baseline_manifest(directory, json_files, allow_changes))
    return checks


def run_verification(root: Path, allow_baseline_changes: bool = False) -> Report:
    """Run all static checks and return a report without printing."""

    report = Report(root=root)
    report.add(_check_required_paths(root))

    for relative, description in CORE_NOTEBOOKS:
        path = root / relative
        if path.is_file():
            report.add(check_notebook(path, root))
        else:
            report.add(
                Check(
                    f"notebook: {relative}",
                    FAIL,
                    f"core notebook missing ({description}); cannot validate structure",
                )
            )

    report.extend(_check_authority_paths(root))
    report.extend(_check_runtime_dependencies())
    report.extend(check_baselines(root, allow_changes=allow_baseline_changes))
    return report


def print_report(report: Report, strict: bool = False) -> None:
    print("Turbinis-Curs static verifier")
    print(f"Repository: {report.root}")
    print(
        "Scope: static repository structure only. No formula, model, optimizer, "
        "sweep, or notebook was executed."
    )
    print("Physics: NOT VERIFIED by this command.")
    print()

    for check in report.checks:
        print(f"[{check.status}] {check.name}: {check.detail}")

    counts = ", ".join(
        f"{report.count(status)} {status}" for status in (PASS, WARN, FAIL)
    )
    print()
    print(f"Summary: {counts}")

    status = report.status
    if status == FAIL:
        print("RESULT: FAIL (structural checks did not pass; physics NOT verified)")
        print("Exit status: 1")
    elif status == WARN:
        print(
            "RESULT: WARN (structural checks completed with warnings; "
            "physics NOT verified)"
        )
        if strict:
            print("Exit status: 2 (--strict treats warnings as failure)")
        else:
            print(
                "Exit status: 0 (warnings do not fail by default; "
                "use --strict for exit 2)"
            )
    else:
        print("RESULT: PASS (structural checks only; physics NOT verified)")
        print("Exit status: 0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify.py",
        description=(
            "Static repository verifier for Turbinis-Curs. It checks required "
            "files, core-notebook JSON structure, optional authority paths, "
            "runtime dependency availability, and frozen baselines if present. "
            "It does not execute or validate physics."
        ),
        epilog=(
            "Exit codes: 0 = all checks passed (warnings allowed unless "
            "--strict); 1 = one or more FAIL checks; 2 = warnings present with "
            "--strict. Frozen baselines are always treated as read-only inputs."
        ),
    )
    parser.add_argument(
        "--allow-baseline-changes",
        action="store_true",
        help=(
            "downgrade baseline integrity mismatches to warnings for "
            "setup/review only; never an approval of changed physics"
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 2 when warnings are present",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root (defaults to the parent directory of scripts/)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = (args.root if args.root is not None else repository_root()).resolve()
    report = run_verification(
        root, allow_baseline_changes=args.allow_baseline_changes
    )
    print_report(report, strict=args.strict)
    return report.exit_code(strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
