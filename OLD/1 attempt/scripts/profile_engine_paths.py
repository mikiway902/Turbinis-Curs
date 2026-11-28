#!/usr/bin/env python3
"""Bounded current-host profile of the active optimizer evaluation path.

SCOPE
=====
This script closes the M1 "current-host profile" evidence gap.  It profiles one
fixed anchor candidate (the point already recorded by
``scripts/test_optimizer_regression.py``) on the current host.  It does **not**
run an optimizer, differential evolution, a parameter sweep, or a physical
validation.

Two paths are measured with exactly the same candidate and the same
``optimize_extended.eval_candidate`` entry point:

* ``extracted_const_hub``: ``build_ns(..., orchestration_core=True)``.  The
  extracted ``turbine_engine.orchestration`` adapters replace the three
  const-hub notebook functions, matching the default optimizer path.
* ``reference_notebook``: ``build_ns(..., orchestration_core=False)``.  The
  notebook's own const-hub functions remain installed as the independent
  software reference, matching the existing parity test.

Both paths use ``fast=True`` because that is the default used by the existing
parity regression test.  The reference path is a software-drift reference, not
a claim that the notebook is an unoptimized production baseline.

TIMING METHOD
=============
Model load, per-path setup (``build_ns``), and warmup are timed separately.
Measured repeats are warm ``eval_candidate`` calls; the reported summary is the
median of ``--repeats`` calls (1..10, default 3).  A separate ``cProfile`` run
records the top functions by cumulative time, with self time and call counts.
Profiling overhead is not mixed into the measured wall times.

The JSON report is strict: non-finite values are replaced with ``null`` and
listed under ``non_finite_replacements``; ``json.dumps(..., allow_nan=False)``
is enforced.  The output file is never silently overwritten.

USAGE
=====
    python scripts/profile_engine_paths.py --output profile.json
    python scripts/profile_engine_paths.py --output profile.json --repeats 5

Dependencies (NumPy/SciPy and the project modules) are imported only after the
CLI arguments and the output path have been validated.  No dependency is
installed by this script.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import platform
import pstats
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
OPTIMIZER_DIR = ROOT / "Optimizer"
CALC_DIR = ROOT / "Turbinis-Calc"

PROFILE_SCHEMA_VERSION = "engine-path-profile-v1"

# Fixed anchor from scripts/test_optimizer_regression.py.  This is a single
# software-drift point, deliberately engineering-infeasible; it is not an
# optimizer result and not an acceptable design.
CANDIDATE = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)
SCENARIO = "B87_relaxed"
Q_N_LIMIT = 0.995

DEFAULT_REPEATS = 3
MAX_REPEATS = 10
WARMUPS = 1
TOP_FUNCTION_COUNT = 20

SIGNATURE_FIELDS = ("N", "lc1", "sigma", "eta_t", "cont", "energy", "p2z")

# The existing orchestration parity test uses a 1e-12 relative tolerance.  The
# task's 1e-9 target is a ceiling; keep the already-established stricter value
# instead of loosening it.
PARITY_RTOL = 1e-12
PARITY_ATOL = 1e-12

LIMITATIONS = (
    "Bounded current-host wall-clock profile; not a benchmark suite.",
    "One fixed anchor candidate is repeated. No optimizer search, differential "
    "evolution, parameter sweep or production design-space run is performed.",
    "The anchor is deliberately engineering-infeasible. This is software-drift "
    "and timing evidence, not physical validation or an acceptable design.",
    "Timings depend on host load, CPU frequency/power state and Python "
    "interpreter state; the median of a small bounded repeat count is not a "
    "portable throughput guarantee.",
    "cProfile instrumentation overhead is separate from the measured wall "
    "times and is used only to identify hot functions.",
    "The reference path keeps notebook const-hub functions but uses the same "
    "fast=True lru_cache/interpolant setup as the existing parity test; it is "
    "not a completely unoptimized notebook execution.",
    "No JAX/GPU path, memory profiling, allocation accounting or IO profiling "
    "is included.",
    "Parity is a software-drift check on selected scalar outputs, not an "
    "independent physics validation.",
    "The CPU name is the safe platform/Windows processor string only; no "
    "credentials, machine name or hardware sensor data are collected.",
)


def positive_int_max10(value: str) -> int:
    """Argparse type for ``--repeats``: int in [1, 10]."""

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"repeats must be an integer in [1, {MAX_REPEATS}], got {value!r}"
        ) from exc
    if parsed < 1 or parsed > MAX_REPEATS:
        raise argparse.ArgumentTypeError(
            f"repeats must be in [1, {MAX_REPEATS}], got {parsed}"
        )
    return parsed


def measure_repeated(
    func: Callable[[], Any],
    repeats: int,
    timer: Callable[[], float] | None = None,
) -> tuple[list[Any], list[float]]:
    """Call ``func`` ``repeats`` times and return values plus wall durations."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    clock = timer or time.perf_counter
    values: list[Any] = []
    durations: list[float] = []
    for _ in range(repeats):
        start = clock()
        values.append(func())
        durations.append(clock() - start)
    return values, durations


def summarize_timings(durations: Iterable[float]) -> dict[str, Any]:
    """Return a finite median/min/max summary for measured wall times."""

    seconds = [float(value) for value in durations]
    if any(not math.isfinite(value) for value in seconds):
        raise ValueError("measured durations must be finite")
    if not seconds:
        return {
            "repeats": 0,
            "seconds": [],
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "repeats": len(seconds),
        "seconds": seconds,
        "median_seconds": float(statistics.median(seconds)),
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
    }


def _finite_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def extract_signature(result: Any) -> dict[str, float | None]:
    """Extract only the documented scalar comparison fields (no arrays)."""

    if not isinstance(result, Mapping):
        return {field: None for field in SIGNATURE_FIELDS}
    return {field: _finite_or_none(result.get(field)) for field in SIGNATURE_FIELDS}


def compare_signatures(
    extracted: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    rtol: float = PARITY_RTOL,
    atol: float = PARITY_ATOL,
) -> dict[str, Any]:
    """Compare selected outputs with ``|a-b| <= atol + rtol*max(|a|,|b|)``."""

    fields: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    max_abs = 0.0
    max_rel = 0.0

    for field in SIGNATURE_FIELDS:
        left = _finite_or_none(extracted.get(field))
        right = _finite_or_none(reference.get(field))
        if left is None or right is None:
            fields[field] = {
                "extracted": left,
                "reference": right,
                "abs_diff": None,
                "rel_diff": None,
                "passed": False,
                "reason": "missing or non-finite value",
            }
            failed.append(field)
            continue

        difference = abs(left - right)
        scale = max(abs(left), abs(right))
        relative = difference / scale if scale else 0.0
        passed = difference <= atol + rtol * scale
        max_abs = max(max_abs, difference)
        max_rel = max(max_rel, relative)
        fields[field] = {
            "extracted": left,
            "reference": right,
            "abs_diff": float(difference),
            "rel_diff": float(relative),
            "passed": bool(passed),
            "reason": "" if passed else "tolerance exceeded",
        }
        if not passed:
            failed.append(field)

    return {
        "status": "pass" if not failed else "mismatch",
        "rtol": float(rtol),
        "atol": float(atol),
        "fields": fields,
        "failed_fields": failed,
        "max_abs_diff": float(max_abs),
        "max_rel_diff": float(max_rel),
    }


def _repo_relative(filename: str, root: Path = ROOT) -> str:
    try:
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return Path(filename).as_posix()


def profile_entries(
    stats: Mapping[tuple[str, int, str], Sequence[Any]],
    *,
    root: Path = ROOT,
    limit: int = TOP_FUNCTION_COUNT,
) -> list[dict[str, Any]]:
    """Format pstats rows as top functions sorted by cumulative time."""

    rows: list[dict[str, Any]] = []
    for key, payload in stats.items():
        if len(payload) < 4:
            continue
        primitive_calls, calls, self_seconds, cumulative_seconds = payload[:4]
        filename, line, function = key
        rows.append(
            {
                "function": str(function),
                "file": _repo_relative(str(filename), root),
                "line": int(line),
                "cumulative_seconds": float(cumulative_seconds),
                "self_seconds": float(self_seconds),
                "calls": int(calls),
                "primitive_calls": int(primitive_calls),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["cumulative_seconds"],
            -row["self_seconds"],
            row["function"],
            row["file"],
            row["line"],
        )
    )
    return rows[: max(0, int(limit))]


def _profile_top(profile: cProfile.Profile) -> list[dict[str, Any]]:
    return profile_entries(pstats.Stats(profile).stats)


def _exc_text(exc: BaseException) -> str:
    message = str(exc).replace("\r", " ").replace("\n", " ")
    return f"{type(exc).__name__}: {message}"


def _timed(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    start = time.perf_counter()
    value = func(*args, **kwargs)
    return value, time.perf_counter() - start


def _git_info() -> dict[str, Any]:
    """Read commit id and dirty boolean only; never include diff/status text."""

    try:
        commit_run = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
        status_run = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "dirty": None}

    available = commit_run.returncode == 0 and status_run.returncode == 0
    commit = commit_run.stdout.strip() or None
    dirty = bool(status_run.stdout.strip()) if status_run.returncode == 0 else None
    return {
        "available": bool(available),
        "commit": commit if available else None,
        "dirty": dirty,
    }


def _host_info(np: Any, scipy: Any) -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        "machine": platform.machine(),
        "cpu_processor": platform.processor() or None,
        "windows_processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER") or None,
        "cpu_count": os.cpu_count(),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
    }


def _new_stage(path_name: str, orchestration_core: bool) -> dict[str, Any]:
    if orchestration_core:
        backend = (
            "CPython + NumPy CPU (no JAX/GPU); extracted const-hub "
            "orchestration adapters"
        )
    else:
        backend = (
            "CPython + NumPy CPU (no JAX/GPU); notebook reference const-hub "
            "functions"
        )
    return {
        "path": path_name,
        "orchestration_core": bool(orchestration_core),
        "build_ns_fast": True,
        "backend": backend,
        "setup": {"status": "not_run", "seconds": None, "error": ""},
        "warmup": {
            "status": "not_run",
            "warmups": WARMUPS,
            "seconds": [],
            "total_seconds": None,
            "result_statuses": [],
            "error": "",
        },
        "measured": {
            "status": "not_run",
            "repeats": 0,
            "seconds": [],
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "error": "",
        },
        "cprofile": {
            "status": "not_run",
            "profiled_calls": 0,
            "wall_seconds": None,
            "top": [],
            "error": "",
        },
        "signature": {field: None for field in SIGNATURE_FIELDS},
        "result_status": "not_run",
        "result_error": "",
        "parity_ready": False,
    }


def _profile_one_path(
    oe: Any,
    model: Any,
    repeats: int,
    path_name: str,
    orchestration_core: bool,
) -> dict[str, Any]:
    stage = _new_stage(path_name, orchestration_core)

    try:
        (namespace, _cycle), setup_seconds = _timed(
            oe.build_ns,
            model,
            q_n_limit=Q_N_LIMIT,
            orchestration_core=orchestration_core,
        )
    except Exception as exc:  # noqa: BLE001 - path failure is report data
        stage["setup"] = {"status": "error", "seconds": None, "error": _exc_text(exc)}
        stage["result_status"] = "setup_error"
        stage["result_error"] = _exc_text(exc)
        return stage

    stage["setup"] = {
        "status": "ok",
        "seconds": _finite_or_none(setup_seconds),
        "error": "",
    }

    def evaluate() -> Mapping[str, Any]:
        return oe.eval_candidate(CANDIDATE, namespace, SCENARIO, Q_N_LIMIT)

    warm_results: list[Any] = []
    warm_seconds: list[float] = []
    try:
        for _ in range(WARMUPS):
            result, seconds = _timed(evaluate)
            warm_results.append(result)
            warm_seconds.append(_finite_or_none(seconds))
        stage["warmup"] = {
            "status": "ok",
            "warmups": WARMUPS,
            "seconds": warm_seconds,
            "total_seconds": _finite_or_none(math.fsum(warm_seconds)),
            "result_statuses": [
                str(result.get("status", "missing"))
                if isinstance(result, Mapping)
                else "invalid_result"
                for result in warm_results
            ],
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - path failure is report data
        stage["warmup"] = {
            "status": "error",
            "warmups": WARMUPS,
            "seconds": warm_seconds,
            "total_seconds": None,
            "result_statuses": [],
            "error": _exc_text(exc),
        }
        stage["result_status"] = "warmup_error"
        stage["result_error"] = _exc_text(exc)
        return stage

    try:
        measured_results, durations = measure_repeated(evaluate, repeats)
    except Exception as exc:  # noqa: BLE001 - path failure is report data
        stage["measured"] = {
            "status": "error",
            "repeats": 0,
            "seconds": [],
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "error": _exc_text(exc),
        }
        stage["result_status"] = "measured_error"
        stage["result_error"] = _exc_text(exc)
        return stage

    measured = summarize_timings(durations)
    measured["status"] = "ok"
    measured["error"] = ""
    stage["measured"] = measured

    last_result = measured_results[-1]
    if isinstance(last_result, Mapping):
        stage["result_status"] = str(last_result.get("status", "missing"))
        stage["result_error"] = str(last_result.get("err", ""))
    else:
        stage["result_status"] = "invalid_result"
        stage["result_error"] = "eval_candidate did not return a mapping"
    stage["signature"] = extract_signature(last_result)
    stage["parity_ready"] = stage["result_status"] == "ok" and all(
        value is not None for value in stage["signature"].values()
    )

    profiler = cProfile.Profile()
    profile_results: list[Any] = []
    profile_error = ""
    profile_start = time.perf_counter()
    try:
        profiler.enable()
        for _ in range(repeats):
            profile_results.append(evaluate())
        profiler.disable()
    except Exception as exc:  # noqa: BLE001 - path failure is report data
        profile_error = _exc_text(exc)
        try:
            profiler.disable()
        except Exception:  # pragma: no cover - defensive
            pass
    profile_wall = time.perf_counter() - profile_start

    if profile_error:
        stage["cprofile"] = {
            "status": "error",
            "profiled_calls": len(profile_results),
            "wall_seconds": _finite_or_none(profile_wall),
            "top": [],
            "error": profile_error,
        }
    else:
        try:
            top = _profile_top(profiler)
            stage["cprofile"] = {
                "status": "ok",
                "profiled_calls": len(profile_results),
                "wall_seconds": _finite_or_none(profile_wall),
                "top": top,
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - path failure is report data
            stage["cprofile"] = {
                "status": "error",
                "profiled_calls": len(profile_results),
                "wall_seconds": _finite_or_none(profile_wall),
                "top": [],
                "error": _exc_text(exc),
            }

    return stage


def _sanitize(value: Any, path: str, notes: list[dict[str, str]]) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            notes.append(
                {
                    "path": path,
                    "reason": "non-finite value replaced with null for strict "
                    "finite JSON",
                }
            )
            return None
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize(child, f"{path}.{key}", notes)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize(child, f"{path}[{index}]", notes)
            for index, child in enumerate(value)
        ]
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _finalize(report: dict[str, Any]) -> dict[str, Any]:
    notes: list[dict[str, str]] = []
    cleaned = _sanitize(report, "$", notes)
    cleaned["non_finite_replacements"] = notes
    return cleaned


def _parity_not_compared(reason: str) -> dict[str, Any]:
    return {
        "status": "not_compared",
        "reason": reason,
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
        "fields": {},
        "failed_fields": [],
        "max_abs_diff": None,
        "max_rel_diff": None,
    }


def _timing_comparison(paths: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    extracted_median = paths["extracted_const_hub"]["measured"].get("median_seconds")
    reference_median = paths["reference_notebook"]["measured"].get("median_seconds")
    ratio = None
    if (
        isinstance(extracted_median, (int, float))
        and isinstance(reference_median, (int, float))
        and math.isfinite(float(extracted_median))
        and math.isfinite(float(reference_median))
        and float(extracted_median) > 0.0
    ):
        ratio = float(reference_median) / float(extracted_median)
    return {
        "metric": "median wall seconds over measured warm repeats",
        "extracted_median_seconds": _finite_or_none(extracted_median),
        "reference_median_seconds": _finite_or_none(reference_median),
        "reference_over_extracted_ratio": _finite_or_none(ratio),
        "interpretation": (
            "wall-clock ratio only; not a speedup or acceleration claim"
        ),
    }


def build_report(repeats: int = DEFAULT_REPEATS) -> dict[str, Any]:
    """Profile both paths and return the JSON-ready report."""

    if repeats < 1 or repeats > MAX_REPEATS:
        raise ValueError(f"repeats must be in [1, {MAX_REPEATS}], got {repeats}")

    # Dependency imports are intentionally delayed until the CLI and output
    # path have been validated by main().
    if str(OPTIMIZER_DIR) not in sys.path:
        sys.path.insert(0, str(OPTIMIZER_DIR))
    if str(CALC_DIR) not in sys.path:
        sys.path.insert(0, str(CALC_DIR))

    import numpy as np  # noqa: PLC0415
    import scipy  # noqa: PLC0415
    import optimizer_core as oc  # noqa: PLC0415
    import optimize_extended as oe  # noqa: PLC0415

    report: dict[str, Any] = {
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete",
        "profile_complete": False,
        "physically_validated": False,
        "validation_status": "NOT_PHYSICALLY_VALIDATED",
        "host": _host_info(np, scipy),
        "git": _git_info(),
        "anchor": {
            "source": "scripts/test_optimizer_regression.py",
            "scenario": SCENARIO,
            "q_n_limit": Q_N_LIMIT,
            "x": list(CANDIDATE),
            "evaluation_kind": (
                "single fixed candidate repeated; no optimizer search or sweep"
            ),
        },
        "method": {
            "timer": "time.perf_counter",
            "statistic": "median of measured warm repeats",
            "repeats": repeats,
            "warmups": WARMUPS,
            "cprofile_top_n": TOP_FUNCTION_COUNT,
            "cprofile_sort": "cumulative_seconds descending",
            "compute_backend": (
                "CPython + NumPy CPU; SciPy imported but no optimizer solver "
                "invoked; no JAX/GPU"
            ),
            "paths": {
                "extracted_const_hub": (
                    "optimize_extended.build_ns(orchestration_core=True, fast=True) "
                    "+ optimize_extended.eval_candidate"
                ),
                "reference_notebook": (
                    "optimize_extended.build_ns(orchestration_core=False, fast=True) "
                    "+ optimize_extended.eval_candidate"
                ),
            },
            "parity_metric": (
                "|extracted-reference| <= atol + rtol*max(|extracted|,|reference|) "
                f"with rtol={PARITY_RTOL:g}, atol={PARITY_ATOL:g}"
            ),
        },
        "stages": {
            "model_load": {"status": "not_run", "seconds": None, "error": ""},
            "paths": {},
        },
        "checks": {
            "model_loaded": False,
            "both_paths_status_ok": False,
            "timings_available": False,
            "cprofile_available": False,
            "parity_pass": False,
        },
        "parity": _parity_not_compared("not run"),
        "timing_comparison": {
            "metric": "median wall seconds over measured warm repeats",
            "extracted_median_seconds": None,
            "reference_median_seconds": None,
            "reference_over_extracted_ratio": None,
            "interpretation": (
                "wall-clock ratio only; not a speedup or acceleration claim"
            ),
        },
        "limitations": list(LIMITATIONS),
    }

    try:
        model, load_seconds = _timed(oc.load_model)
    except Exception as exc:  # noqa: BLE001 - report the environment/path failure
        report["stages"]["model_load"] = {
            "status": "error",
            "seconds": None,
            "error": _exc_text(exc),
        }
        report["status"] = "model_load_error"
        return _finalize(report)

    report["stages"]["model_load"] = {
        "status": "ok",
        "seconds": _finite_or_none(load_seconds),
        "error": "",
    }
    report["checks"]["model_loaded"] = True

    paths = {
        "extracted_const_hub": _profile_one_path(
            oe, model, repeats, "extracted_const_hub", orchestration_core=True
        ),
        "reference_notebook": _profile_one_path(
            oe, model, repeats, "reference_notebook", orchestration_core=False
        ),
    }
    report["stages"]["paths"] = paths

    extracted = paths["extracted_const_hub"]
    reference = paths["reference_notebook"]
    both_ok = (
        extracted["result_status"] == "ok" and reference["result_status"] == "ok"
    )
    timings_available = (
        extracted["measured"]["status"] == "ok"
        and reference["measured"]["status"] == "ok"
        and extracted["measured"]["median_seconds"] is not None
        and reference["measured"]["median_seconds"] is not None
    )
    cprofile_available = (
        extracted["cprofile"]["status"] == "ok"
        and reference["cprofile"]["status"] == "ok"
    )

    report["checks"]["both_paths_status_ok"] = both_ok
    report["checks"]["timings_available"] = timings_available
    report["checks"]["cprofile_available"] = cprofile_available

    if both_ok:
        parity = compare_signatures(extracted["signature"], reference["signature"])
    else:
        parity = _parity_not_compared(
            "one or both paths did not return result status 'ok'"
        )
    report["parity"] = parity
    report["checks"]["parity_pass"] = parity["status"] == "pass"
    report["timing_comparison"] = _timing_comparison(paths)

    if not both_ok:
        report["status"] = "path_error"
    elif not report["checks"]["parity_pass"]:
        report["status"] = "parity_mismatch"
    elif not timings_available or not cprofile_available:
        report["status"] = "incomplete"
    else:
        report["status"] = "ok"

    report["profile_complete"] = report["status"] == "ok"
    return _finalize(report)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded current-host profile of one fixed optimizer anchor "
            "candidate through the extracted const-hub and notebook reference "
            "paths. This is timing/software-drift evidence, not physical "
            "validation, and it never runs a search or sweep."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="required JSON output file; an existing file is refused",
    )
    parser.add_argument(
        "--repeats",
        type=positive_int_max10,
        default=DEFAULT_REPEATS,
        help=(
            "measured warm eval_candidate repeats in [1, 10] "
            f"(default {DEFAULT_REPEATS})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    output = Path(args.output)

    if output.exists():
        print(
            f"error: output already exists and will not be overwritten: {output}",
            file=sys.stderr,
        )
        return 2

    try:
        report = build_report(args.repeats)
        text = json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            ensure_ascii=True,
        )
        if output.parent != Path("."):
            output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.write("\n")
    except FileExistsError:
        print(
            f"error: output already exists and will not be overwritten: {output}",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, TypeError, ImportError) as exc:
        print(f"error: {_exc_text(exc)}", file=sys.stderr)
        return 2

    print(f"profile written: {output}")
    if report.get("status") == "ok":
        return 0
    print(f"profile status: {report.get('status')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
