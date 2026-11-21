#!/usr/bin/env python3
"""Validate and benchmark the optional gas cp/k numerical backends.

This command is deliberately bounded.  It exercises one existing scalar kernel,
``turbine_engine.gas.calc_gas_cp_k`` (the ``gas(T)`` helper built on
``cycle.calc_cp_kerosene``), through:

* the existing scalar Python reference;
* a vectorized NumPy float64 port;
* an optional jitted JAX float64 kernel on an explicitly selected CPU device;
* the same JAX kernel on an explicitly selected GPU device, when available.

It also measures first-call compilation/warmup and steady kernel execution with
``block_until_ready`` synchronization.  It does **not** run the cycle, the
stage model, an engine-design sweep, or any optimizer, and it makes no
full-cycle or engine speedup claim.

GPU is optional.  When ``jax.devices("gpu")`` is unavailable or the optional
CUDA backend cannot run, the exact error is recorded in the JSON report and the
CPU checks continue.  A GPU result is labelled ``gpu`` only after that explicit
device selection; there is no silent CPU fallback.

Example::

    .\\.venv\\Scripts\\python.exe scripts\\validate_backends.py
    ~/.venvs/turbinis-gpu/bin/python scripts/validate_backends.py

The report is written to ``docs/backend_validation.json`` by default.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from turbine_engine import backends  # noqa: E402
from turbine_engine.cycle import DEFAULT_CONFIG  # noqa: E402
from turbine_engine.gas import (  # noqa: E402
    GAS_CELSIUS_OFFSET_K,
    GAS_FINITE_DIFFERENCE_STEP_K,
)

DEFAULT_OUTPUT = Path("docs") / "backend_validation.json"
DEFAULT_RTOL = 1e-12
DEFAULT_ATOL = 1e-12
GAS_CONSTANT = DEFAULT_CONFIG.R
XLA_PREALLOCATE_VARIABLE = "XLA_PYTHON_CLIENT_PREALLOCATE"


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat(
        timespec="seconds"
    )


def _parse_devices(text: str) -> list[str]:
    devices = [part.strip().lower() for part in text.split(",") if part.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    invalid = [name for name in devices if name not in {"cpu", "gpu"}]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown device(s): " + ", ".join(repr(name) for name in invalid)
        )
    # Preserve first-seen order while removing duplicates.
    return list(dict.fromkeys(devices))


def _validation_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Representative physical points plus the exact 700 K branch boundary."""

    temperatures = np.array(
        [288.0, 650.0, 699.999, 700.0, 700.001, 850.0, 1100.0, 1250.0],
        dtype=np.float64,
    )
    alphas = np.array(
        [1.0, 1.8, 3.0, 3.973413639490531, math.inf],
        dtype=np.float64,
    )
    t = np.repeat(temperatures, alphas.size)
    alpha = np.tile(alphas, temperatures.size)
    r = np.full(t.shape, GAS_CONSTANT, dtype=np.float64)
    return t, alpha, r


def _reference_arrays(
    temperatures: np.ndarray, alpha: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cp = np.empty(temperatures.shape, dtype=np.float64)
    k = np.empty(temperatures.shape, dtype=np.float64)
    for index in range(temperatures.size):
        cp[index], k[index] = backends.reference_gas_cp_k(
            float(temperatures[index]), float(alpha[index]), float(r[index])
        )
    return cp, k


def _comparison_dict(comparison: backends.ArrayComparison) -> dict[str, Any]:
    return {
        "element_count": comparison.element_count,
        "max_abs_error": comparison.max_abs_error,
        "max_rel_error": comparison.max_rel_error,
        "allclose": comparison.allclose,
        "rtol": comparison.rtol,
        "atol": comparison.atol,
    }


def _kernel_comparison(
    reference_cp: np.ndarray,
    reference_k: np.ndarray,
    actual_cp: np.ndarray,
    actual_k: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    cp_comparison = backends.compare_arrays(
        reference_cp, actual_cp, rtol=rtol, atol=atol
    )
    k_comparison = backends.compare_arrays(
        reference_k, actual_k, rtol=rtol, atol=atol
    )
    return {
        "cp": _comparison_dict(cp_comparison),
        "k": _comparison_dict(k_comparison),
        "allclose": bool(cp_comparison.allclose and k_comparison.allclose),
    }


def _benchmark_dict(
    result: backends.BenchmarkResult,
    *,
    first_call_includes_jax_compilation: bool,
) -> dict[str, Any]:
    return {
        "first_call_seconds": result.first_call_seconds,
        "first_call_includes_jax_compilation": (
            first_call_includes_jax_compilation
        ),
        "warmup_seconds": list(result.warmup_seconds),
        "sample_seconds": list(result.sample_seconds),
        "samples": len(result.sample_seconds),
        "median_seconds": result.median_seconds,
        "min_seconds": result.min_seconds,
        "elements_per_call": result.elements_per_call,
        "throughput_elements_per_second": (
            result.throughput_elements_per_second
        ),
    }


def _benchmark_batch(batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temperatures = np.linspace(288.0, 1250.0, batch_size, dtype=np.float64)
    index = np.arange(batch_size, dtype=np.float64)
    alpha = 1.0 + 3.0 * (index % 97.0) / 96.0
    r = np.full(batch_size, GAS_CONSTANT, dtype=np.float64)
    return temperatures, alpha, r


def _jax_versions() -> dict[str, Any]:
    try:
        import jax
        import jaxlib
    except ModuleNotFoundError:
        return {"installed": False}

    return {
        "installed": True,
        "jax": getattr(jax, "__version__", "unknown"),
        "jaxlib": getattr(jaxlib, "__version__", "unknown"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _gpu_error_report(exc: backends.BackendUnavailableError) -> dict[str, Any]:
    cause = exc.__cause__
    return {
        "requested": True,
        "selection": "jax.devices('gpu')",
        "status": "unavailable",
        "error_type": (
            type(cause).__name__ if cause is not None else type(exc).__name__
        ),
        "error": str(exc),
    }


def _gpu_exception_report(exc: BaseException) -> dict[str, Any]:
    return {
        "requested": True,
        "selection": "jax.devices('gpu')",
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _run_gpu_validation(
    temperatures: np.ndarray,
    alpha: np.ndarray,
    r: np.ndarray,
    reference_cp: np.ndarray,
    reference_k: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> tuple[dict[str, Any], Any | None]:
    try:
        device = backends.select_jax_device("gpu")
    except backends.BackendUnavailableError as exc:
        return _gpu_error_report(exc), None
    except Exception as exc:  # Optional backend boundary: record and fall back.
        return _gpu_exception_report(exc), None

    report: dict[str, Any] = {
        "requested": True,
        "selection": "jax.devices('gpu')",
        "status": "selected",
        "device": backends.describe_device(device).__dict__,
    }
    try:
        import jax

        report["devices"] = [
            backends.describe_device(item).__dict__
            for item in jax.devices("gpu")
        ]
        kernel = backends.make_jax_gas_cp_k(device)
        actual_cp, actual_k = backends.run_jax_gas_cp_k(
            kernel, temperatures, alpha, r
        )
        comparison = _kernel_comparison(
            reference_cp,
            reference_k,
            actual_cp,
            actual_k,
            rtol=rtol,
            atol=atol,
        )
        report.update(
            {
                "status": "ok" if comparison["allclose"] else "failed",
                "dtype": kernel.dtype,
                "comparison": comparison,
            }
        )
        # Return the selected device, not the JaxKernel wrapper: the benchmark
        # helper builds a fresh kernel and must receive a real JAX device.
        return report, device
    except Exception as exc:  # Optional backend boundary: record and fall back.
        report.update(_gpu_exception_report(exc))
        return report, None


def _benchmark_jax(
    device: Any,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    warmup_calls: int,
    samples: int,
) -> dict[str, Any]:
    kernel = backends.make_jax_gas_cp_k(device)
    inputs = backends.prepare_jax_inputs(kernel, *batch)
    result = backends.benchmark_callable(
        lambda: kernel.function(*inputs),
        warmup_calls=warmup_calls,
        samples=samples,
        elements_per_call=int(batch[0].size),
        synchronize=backends.block_until_ready,
    )
    entry = _benchmark_dict(
        result, first_call_includes_jax_compilation=True
    )
    entry["device"] = backends.describe_device(device).__dict__
    entry["dtype"] = kernel.dtype
    return entry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validate_backends.py",
        description=(
            "Validate the existing gas cp/k scalar kernel against vectorized "
            "NumPy and optional JAX CPU/GPU float64 ports, then benchmark the "
            "kernel only. GPU selection is explicit via jax.devices('gpu')."
        ),
    )
    parser.add_argument(
        "--devices",
        type=_parse_devices,
        default=["cpu", "gpu"],
        help=(
            "comma-separated devices to attempt: cpu,gpu (default). GPU "
            "unavailability is recorded and CPU continues."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=(
            "JSON report path (relative paths are resolved from the repository "
            "root); use '-' to skip writing a file"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="kernel benchmark elements per call (default: 65536)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="steady-state benchmark samples (default: 10)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="warmup calls after the separately timed first call (default: 3)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
        help=f"relative tolerance for numeric parity (default: {DEFAULT_RTOL})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
        help=f"absolute tolerance for numeric parity (default: {DEFAULT_ATOL})",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="return exit status 1 unless the explicit GPU backend validates",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Process-local allocator setting for an 8 GB consumer GPU: this bounded
    # kernel does not need XLA to reserve most of the device.  An externally
    # set value is preserved so the report records the actual configuration.
    os.environ.setdefault(XLA_PREALLOCATE_VARIABLE, "false")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be >= 0")

    print("Turbinis-Curs optional numerical backend validation")
    print("Scope: one gas cp/k kernel only; not the cycle or engine performance.")
    print(
        f"Host: Python {platform.python_version()}, NumPy {np.__version__}, "
        f"{platform.system()} {platform.machine()}"
    )

    temperatures, alpha, r = _validation_grid()
    reference_cp, reference_k = _reference_arrays(temperatures, alpha, r)

    numpy_cp, numpy_k = backends.numpy_gas_cp_k(temperatures, alpha, r)
    numpy_comparison = _kernel_comparison(
        reference_cp,
        reference_k,
        numpy_cp,
        numpy_k,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        "NumPy vs reference: "
        f"{'PASS' if numpy_comparison['allclose'] else 'FAIL'} "
        f"(max abs cp={numpy_comparison['cp']['max_abs_error']:.3e}, "
        f"k={numpy_comparison['k']['max_abs_error']:.3e})"
    )

    jax_cpu_report: dict[str, Any]
    cpu_device: Any | None = None
    if "cpu" in args.devices:
        try:
            cpu_device = backends.select_jax_device("cpu")
            cpu_kernel = backends.make_jax_gas_cp_k(cpu_device)
            cpu_cp, cpu_k = backends.run_jax_gas_cp_k(
                cpu_kernel, temperatures, alpha, r
            )
            cpu_comparison = _kernel_comparison(
                reference_cp,
                reference_k,
                cpu_cp,
                cpu_k,
                rtol=args.rtol,
                atol=args.atol,
            )
            jax_cpu_report = {
                "requested": True,
                "selection": "jax.devices('cpu')",
                "status": "ok" if cpu_comparison["allclose"] else "failed",
                "device": backends.describe_device(cpu_device).__dict__,
                "dtype": cpu_kernel.dtype,
                "comparison": cpu_comparison,
            }
            print(
                "JAX CPU vs reference: "
                f"{'PASS' if cpu_comparison['allclose'] else 'FAIL'} "
                f"(device={jax_cpu_report['device']['kind']}, "
                f"dtype={cpu_kernel.dtype})"
            )
        except backends.BackendUnavailableError as exc:
            cpu_device = None
            jax_cpu_report = {
                "requested": True,
                "selection": "jax.devices('cpu')",
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"JAX CPU: UNAVAILABLE ({exc})")
    else:
        jax_cpu_report = {
            "requested": False,
            "status": "not_requested",
        }

    gpu_report: dict[str, Any]
    gpu_device: Any | None = None
    if "gpu" in args.devices:
        gpu_report, gpu_device = _run_gpu_validation(
            temperatures,
            alpha,
            r,
            reference_cp,
            reference_k,
            rtol=args.rtol,
            atol=args.atol,
        )
        if gpu_report.get("status") == "ok":
            print(
                "JAX GPU vs reference: PASS "
                f"(device={gpu_report['device']['kind']}, "
                f"dtype={gpu_report.get('dtype')})"
            )
        elif gpu_report.get("status") == "failed":
            print("JAX GPU vs reference: FAIL (numeric parity)")
        else:
            print(
                "JAX GPU: "
                f"{str(gpu_report.get('status')).upper()} "
                f"({gpu_report.get('error_type', 'error')}: "
                f"{gpu_report.get('error', 'no GPU device selected')})"
            )
    else:
        gpu_report = {"requested": False, "status": "not_requested"}

    batch = _benchmark_batch(args.batch_size)
    benchmark: dict[str, Any] = {
        "scope": (
            "kernel throughput only; not a full-cycle, stage, or engine "
            "speedup measurement"
        ),
        "batch_size": args.batch_size,
        "warmup_calls": args.warmup,
        "samples": args.samples,
        "numpy": _benchmark_dict(
            backends.benchmark_callable(
                lambda: backends.numpy_gas_cp_k(*batch),
                warmup_calls=args.warmup,
                samples=args.samples,
                elements_per_call=args.batch_size,
            ),
            first_call_includes_jax_compilation=False,
        ),
    }
    print(
        "NumPy kernel median: "
        f"{benchmark['numpy']['median_seconds'] * 1e3:.3f} ms "
        f"({benchmark['numpy']['throughput_elements_per_second'] / 1e6:.2f} "
        "M elements/s)"
    )

    if cpu_device is not None:
        try:
            benchmark["jax_cpu"] = _benchmark_jax(
                cpu_device,
                batch,
                warmup_calls=args.warmup,
                samples=args.samples,
            )
            print(
                "JAX CPU kernel median: "
                f"{benchmark['jax_cpu']['median_seconds'] * 1e3:.3f} ms "
                "(first call "
                f"{benchmark['jax_cpu']['first_call_seconds'] * 1e3:.1f} ms, "
                "includes compilation)"
            )
        except Exception as exc:
            benchmark["jax_cpu"] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"JAX CPU benchmark: ERROR ({type(exc).__name__}: {exc})")

    if gpu_device is not None and gpu_report.get("status") == "ok":
        try:
            benchmark["jax_gpu"] = _benchmark_jax(
                gpu_device,
                batch,
                warmup_calls=args.warmup,
                samples=args.samples,
            )
            print(
                "JAX GPU kernel median: "
                f"{benchmark['jax_gpu']['median_seconds'] * 1e3:.3f} ms "
                "(first call "
                f"{benchmark['jax_gpu']['first_call_seconds'] * 1e3:.1f} ms, "
                "includes compilation)"
            )
        except Exception as exc:
            benchmark["jax_gpu"] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"JAX GPU benchmark: ERROR ({type(exc).__name__}: {exc})")

    comparison: dict[str, Any] = {
        "definition": (
            "steady_time_ratio = JAX median seconds / NumPy median seconds; "
            "values > 1 mean the JAX kernel was slower on this host and batch"
        ),
        "jax_cpu_steady_time_ratio": None,
        "jax_gpu_steady_time_ratio": None,
        "interpretation": (
            "Measured single-kernel throughput only. No full-cycle or engine "
            "speedup is claimed; a GPU result on another batch or host may "
            "differ."
        ),
    }
    numpy_median = benchmark["numpy"]["median_seconds"]
    if numpy_median > 0.0:
        if "jax_cpu" in benchmark and "median_seconds" in benchmark["jax_cpu"]:
            comparison["jax_cpu_steady_time_ratio"] = (
                benchmark["jax_cpu"]["median_seconds"] / numpy_median
            )
        if "jax_gpu" in benchmark and "median_seconds" in benchmark["jax_gpu"]:
            comparison["jax_gpu_steady_time_ratio"] = (
                benchmark["jax_gpu"]["median_seconds"] / numpy_median
            )
    benchmark["comparison"] = comparison

    jax_cpu_failed = jax_cpu_report.get("status") == "failed"
    gpu_failed = gpu_report.get("status") == "failed"
    cpu_benchmark_failed = benchmark.get("jax_cpu", {}).get("status") == "error"
    gpu_benchmark_failed = benchmark.get("jax_gpu", {}).get("status") == "error"
    gpu_ready = gpu_report.get("status") == "ok"
    jax_cpu_ready = jax_cpu_report.get("status") == "ok"
    numeric_ok = bool(numpy_comparison["allclose"]) and not (
        jax_cpu_failed or gpu_failed or cpu_benchmark_failed
    )
    warning = bool(
        ("cpu" in args.devices and not jax_cpu_ready)
        or ("gpu" in args.devices and not gpu_ready)
        or gpu_benchmark_failed
    )
    if not numeric_ok:
        result_status = "FAIL"
    elif warning:
        result_status = "WARN"
    else:
        result_status = "PASS"

    exit_code = 0
    if not numeric_ok:
        exit_code = 1
    if args.require_gpu and not gpu_ready:
        exit_code = 1
    if gpu_benchmark_failed and args.require_gpu:
        exit_code = 1

    report: dict[str, Any] = {
        "schema": "turbinis.backend-validation.v1",
        "generated_at_utc": _utc_now(),
        "scope": {
            "purpose": "bounded optional JAX CPU/GPU numerical validation",
            "kernel_only": True,
            "full_cycle": False,
            "engine_design_sweep": False,
            "physical_validation": False,
            "speedup_claim": "none",
        },
        "cli": {
            "default_command": "python scripts/validate_backends.py",
            "devices": list(args.devices),
            "batch_size": args.batch_size,
            "samples": args.samples,
            "warmup": args.warmup,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "kernel": {
            "name": backends.KERNEL_NAME,
            "reference": "turbine_engine.gas.calc_gas_cp_k",
            "formula_source": "turbine_engine.cycle.calc_cp_kerosene",
            "dtype": backends.KERNEL_DTYPE,
            "branch_temperature_k": backends.CP_BRANCH_TEMPERATURE_K,
            "finite_difference_step_k": GAS_FINITE_DIFFERENCE_STEP_K,
            "celsius_offset_k": GAS_CELSIUS_OFFSET_K,
            "calorie_to_joule": backends.CP_CALORIE_TO_JOULE,
            "equation": (
                "cp(T<700)=((1.2+2.25/alpha)/1e5*(T-70)+0.236)*4187; "
                "cp(T>=700)=((2.2+1.25/alpha)/1e5*(T+450)+0.218)*4187; "
                "dcp/dT=(cp(T+1)-cp(T-1))/2; "
                "cp_gas=cp(T)+(T-273.15)*dcp/dT; k=cp_gas/(cp_gas-R)"
            ),
        },
        "host": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jax": _jax_versions(),
            "xla_python_client_preallocate": os.environ.get(
                XLA_PREALLOCATE_VARIABLE
            ),
            "xla_setting_scope": "this process only",
        },
        "validation_inputs": {
            "temperatures_k": [float(value) for value in np.unique(
                temperatures
            )],
            "alphas": [
                "inf" if math.isinf(float(value)) else float(value)
                for value in np.unique(alpha)
            ],
            "gas_constant_j_per_kg_k": GAS_CONSTANT,
            "point_count": int(temperatures.size),
            "boundary_note": (
                "Includes T=699.999, 700.0 and 700.001 K; exactly 700 K uses "
                "the T >= 700 source branch."
            ),
        },
        "backends": {
            "reference": {
                "status": "ok",
                "device": "host CPU (Python scalar)",
            },
            "numpy": {
                "status": "ok" if numpy_comparison["allclose"] else "failed",
                "dtype": "float64",
                "comparison": numpy_comparison,
            },
            "jax_cpu": jax_cpu_report,
            "jax_gpu": gpu_report,
        },
        "benchmark": benchmark,
        "result": {
            "status": result_status,
            "numeric_ok": numeric_ok,
            "gpu_ready": gpu_ready,
            "gpu_benchmark_ok": not gpu_benchmark_failed,
            "jax_cpu_ready": jax_cpu_ready,
            "note": (
                "GPU unavailability/error is recorded, not labelled as a GPU "
                "pass; CPU validation still runs."
            ),
        },
    }

    output_path: Path | None
    if args.output == "-":
        output_path = None
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        try:
            shown_path = output_path.relative_to(ROOT).as_posix()
        except ValueError:
            shown_path = output_path.name
        print(f"Wrote {shown_path}")

    if result_status == "PASS":
        if gpu_ready:
            print("Result: PASS (explicit JAX GPU kernel validated; CPU too)")
        else:
            print("Result: PASS (CPU kernel validated; GPU not requested)")
    elif result_status == "WARN":
        print(
            "Result: WARN (CPU numerical validation completed; optional "
            f"backend status: CPU={jax_cpu_report.get('status')}, "
            f"GPU={gpu_report.get('status')})"
        )
    else:
        print("Result: FAIL (see numeric parity or --require-gpu status above)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
