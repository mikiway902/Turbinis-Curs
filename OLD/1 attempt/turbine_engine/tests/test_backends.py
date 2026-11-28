"""Focused tests for the optional gas cp/k numerical backends.

PROVENANCE / STATUS: software-drift and numerical-port checks only.

The reference is the existing scalar :func:`turbine_engine.gas.calc_gas_cp_k`
built on :func:`turbine_engine.cycle.calc_cp_kerosene`.  These tests cover the
literal 700 K branch boundary, float64 parity of the NumPy and optional JAX
ports, explicit CPU/GPU device selection, bounded OOM chunk retry, dtype
preservation, and lazy JAX import.  They do not run the cycle or the stage and
do not claim physical validation or engine speedup.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Process-local only: keep the small test kernel from asking XLA to reserve
# most of a consumer GPU.  An externally set value is preserved.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine import backends  # noqa: E402

RTOL = 1e-12
ATOL = 1e-12
GAS_CONSTANT = 287.4


def validation_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def reference_arrays(
    t: np.ndarray, alpha: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    cp = np.empty(t.shape, dtype=np.float64)
    k = np.empty(t.shape, dtype=np.float64)
    for index in range(t.size):
        cp[index], k[index] = backends.reference_gas_cp_k(
            float(t[index]), float(alpha[index]), float(r[index])
        )
    return cp, k


class NumPyKernelTests(unittest.TestCase):
    def test_numpy_matches_scalar_reference_including_700k_boundary(self) -> None:
        t, alpha, r = validation_points()
        expected_cp, expected_k = reference_arrays(t, alpha, r)
        actual_cp, actual_k = backends.numpy_gas_cp_k(t, alpha, r)

        cp_comparison = backends.compare_arrays(
            expected_cp, actual_cp, rtol=RTOL, atol=ATOL
        )
        k_comparison = backends.compare_arrays(
            expected_k, actual_k, rtol=RTOL, atol=ATOL
        )
        self.assertTrue(cp_comparison.allclose, cp_comparison)
        self.assertTrue(k_comparison.allclose, k_comparison)
        self.assertEqual(actual_cp.dtype, np.dtype(np.float64))
        self.assertEqual(actual_k.dtype, np.dtype(np.float64))

    def test_boundary_points_are_present_and_use_source_side(self) -> None:
        t, alpha, r = validation_points()
        for temperature in (699.999, 700.0, 700.001):
            with self.subTest(temperature=temperature):
                mask = t == temperature
                self.assertTrue(np.any(mask))
                actual_cp, actual_k = backends.numpy_gas_cp_k(
                    t[mask], alpha[mask], r[mask]
                )
                expected_cp, expected_k = reference_arrays(
                    t[mask], alpha[mask], r[mask]
                )
                np.testing.assert_allclose(actual_cp, expected_cp, rtol=RTOL, atol=ATOL)
                np.testing.assert_allclose(actual_k, expected_k, rtol=RTOL, atol=ATOL)

    def test_invalid_inputs_fail_explicitly(self) -> None:
        cases = (
            (0.0, 1.0, GAS_CONSTANT),
            (-1.0, 1.0, GAS_CONSTANT),
            (math.nan, 1.0, GAS_CONSTANT),
            (math.inf, 1.0, GAS_CONSTANT),
            (288.0, 0.0, GAS_CONSTANT),
            (288.0, -1.0, GAS_CONSTANT),
            (288.0, math.nan, GAS_CONSTANT),
            (288.0, 1.0, 0.0),
            (288.0, 1.0, math.nan),
            (288.0, 1.0, 1e9),
        )
        for temperature, alpha, r in cases:
            with self.subTest(T=temperature, alpha=alpha, R=r):
                with self.assertRaises(ValueError):
                    backends.numpy_gas_cp_k(temperature, alpha, r)

    def test_chunked_helper_preserves_dtype_and_values(self) -> None:
        values = np.arange(10, dtype=np.float32)

        def kernel(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return chunk * np.float32(2.0), chunk + np.float32(1.0)

        doubled, shifted = backends.run_chunked(
            kernel, (values,), chunk_size=3
        )
        self.assertEqual(doubled.dtype, np.dtype(np.float32))
        self.assertEqual(shifted.dtype, np.dtype(np.float32))
        np.testing.assert_array_equal(doubled, values * np.float32(2.0))
        np.testing.assert_array_equal(shifted, values + np.float32(1.0))

    def test_chunked_helper_halves_only_on_recognized_oom(self) -> None:
        calls: list[int] = []

        def flaky(chunk: np.ndarray) -> tuple[np.ndarray, ...]:
            calls.append(int(chunk.size))
            if chunk.size > 2:
                raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: simulated")
            return (chunk * 2.0,)

        result = backends.run_chunked(
            flaky, (np.arange(5, dtype=np.float64),), chunk_size=5
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0], np.arange(5) * 2.0)
        self.assertEqual(calls, [5, 2, 2, 1])

        def non_oom(chunk: np.ndarray) -> tuple[np.ndarray, ...]:
            raise ValueError("kernel bug")

        with self.assertRaises(ValueError):
            backends.run_chunked(
                non_oom,
                (np.arange(5, dtype=np.float64),),
                chunk_size=5,
            )

    def test_jax_is_not_imported_by_backends_import(self) -> None:
        code = (
            "import sys\n"
            "import turbine_engine.backends\n"
            "assert 'jax' not in sys.modules, sorted(sys.modules)\n"
            "assert 'jaxlib' not in sys.modules, sorted(sys.modules)\n"
            "print('lazy-import-ok')\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
        )
        self.assertIn("lazy-import-ok", completed.stdout)


class JaxCpuKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import jax

            cls.jax = jax
            cls.device = backends.select_jax_device("cpu")
            cls.kernel = backends.make_jax_gas_cp_k(cls.device)
        except backends.BackendUnavailableError as exc:
            raise unittest.SkipTest(
                f"optional JAX CPU backend unavailable: {exc}"
            ) from exc

    def test_explicit_cpu_device_and_float64(self) -> None:
        self.assertEqual(self.kernel.info().platform, "cpu")
        self.assertEqual(self.kernel.dtype, "float64")
        self.assertTrue(self.jax.config.jax_enable_x64)

    def test_jax_cpu_matches_reference(self) -> None:
        t, alpha, r = validation_points()
        expected_cp, expected_k = reference_arrays(t, alpha, r)
        actual_cp, actual_k = backends.run_jax_gas_cp_k(
            self.kernel, t, alpha, r
        )
        self.assertEqual(actual_cp.dtype, np.dtype(np.float64))
        self.assertEqual(actual_k.dtype, np.dtype(np.float64))
        np.testing.assert_allclose(actual_cp, expected_cp, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual_k, expected_k, rtol=RTOL, atol=ATOL)

    def test_gpu_request_does_not_silently_fall_back(self) -> None:
        try:
            device = backends.select_jax_device("gpu")
        except backends.BackendUnavailableError as exc:
            self.assertIn("gpu", str(exc))
        else:
            self.assertEqual(backends.describe_device(device).platform, "gpu")


class JaxGpuKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.device = backends.select_jax_device("gpu")
            cls.kernel = backends.make_jax_gas_cp_k(cls.device)
        except backends.BackendUnavailableError as exc:
            raise unittest.SkipTest(
                f"optional JAX GPU backend unavailable: {exc}"
            ) from exc

    def test_gpu_is_explicit_and_float64(self) -> None:
        self.assertEqual(self.kernel.info().platform, "gpu")
        self.assertEqual(self.kernel.dtype, "float64")

    def test_jax_gpu_matches_reference(self) -> None:
        t, alpha, r = validation_points()
        expected_cp, expected_k = reference_arrays(t, alpha, r)
        actual_cp, actual_k = backends.run_jax_gas_cp_k(
            self.kernel, t, alpha, r
        )
        np.testing.assert_allclose(actual_cp, expected_cp, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(actual_k, expected_k, rtol=RTOL, atol=ATOL)


class ValidationCliTests(unittest.TestCase):
    def test_cpu_cli_writes_valid_float64_report(self) -> None:
        if importlib.util.find_spec("jax") is None:
            raise unittest.SkipTest(
                "optional JAX backend unavailable; CLI CPU JAX check skipped"
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "backend_validation.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "validate_backends.py"),
                    "--devices",
                    "cpu",
                    "--output",
                    str(output),
                    "--batch-size",
                    "512",
                    "--samples",
                    "2",
                    "--warmup",
                    "1",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"stdout={completed.stdout!r}\nstderr={completed.stderr!r}",
            )
            text = output.read_text(encoding="utf-8")

            def reject_constant(value: str) -> None:
                raise AssertionError(f"non-finite JSON constant {value!r}")

            report = json.loads(text, parse_constant=reject_constant)
            self.assertEqual(report["schema"], "turbinis.backend-validation.v1")
            self.assertTrue(report["result"]["numeric_ok"])
            self.assertEqual(report["backends"]["jax_cpu"]["status"], "ok")
            self.assertEqual(report["backends"]["jax_cpu"]["dtype"], "float64")
            self.assertEqual(
                report["backends"]["jax_cpu"]["device"]["platform"], "cpu"
            )
            self.assertEqual(
                report["benchmark"]["jax_cpu"]["dtype"], "float64"
            )
            self.assertTrue(
                report["benchmark"]["jax_cpu"][
                    "first_call_includes_jax_compilation"
                ]
            )


if __name__ == "__main__":
    unittest.main()
