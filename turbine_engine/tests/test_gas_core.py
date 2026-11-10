"""Tests for the extracted stage gas-property helper.

PROVENANCE / STATUS: software-drift diagnostic only.

The parity tests below dynamically build the *current* notebook namespace once,
compute one fixed three-stage candidate with the legacy stage distribution, and
compare :mod:`turbine_engine.gas` with the live notebook ``газ`` and
``cp_k_ступени`` functions.  They are software-drift checks, not physical
validation or an acceptable-design claim.

No optimizer search, no parameter sweep, and no notebook execution from
``turbine_engine.gas`` are involved.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.cycle import calc_cp_kerosene  # noqa: E402
from turbine_engine.gas import (  # noqa: E402
    GasPropertyError,
    calc_gas_cp_k,
    calc_h0_for_pressure,
    calc_stage_cp_k,
)


class FormulaAndValidationTests(unittest.TestCase):
    """Check the source expression directly, without loading a notebook."""

    RELATIVE_TOLERANCE = 1e-12

    def assertRelative(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        if expected == 0.0:
            self.assertEqual(actual, 0.0, context)
            return
        self.assertLessEqual(
            abs(actual / expected - 1.0),
            self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def source_gas(self, T: float, alpha: float, R: float) -> tuple[float, float]:
        # Literal transcription of notebook cell 7 for this local check.
        dT = 1.0
        c_pm = float(calc_cp_kerosene(T, alpha))
        dc_pm = (
            float(calc_cp_kerosene(T + dT, alpha))
            - float(calc_cp_kerosene(T - dT, alpha))
        ) / (2 * dT)
        c_p = c_pm + (T - 273.15) * dc_pm
        return c_p, c_p / (c_p - R)

    def source_h0_for_pressure(
        self, T0: float, p0: float, p2: float, alpha: float, R: float, phi: float
    ) -> float:
        # Literal transcription of notebook cell 7 ``H0_по_p``.
        cp, k = self.source_gas(T0, alpha, R)
        H0 = 0.0
        for _ in range(50):
            H0 = cp * T0 * (1 - (p2 / p0) ** ((k - 1) / k))
            cp_new, k_new = self.source_gas(T0 - H0 * phi**2 / (2 * cp), alpha, R)
            if abs(cp_new - cp) < 1e-9:
                break
            cp, k = cp_new, k_new
        return H0

    def test_gas_expression_matches_source_transcription(self) -> None:
        for T in (288.0, 699.999, 700.0, 850.0, 1250.0):
            for alpha in (1.0, 1.8, 3.0, math.inf):
                with self.subTest(T=T, alpha=alpha):
                    expected_cp, expected_k = self.source_gas(T, alpha, 287.4)
                    actual_cp, actual_k = calc_gas_cp_k(T, alpha, 287.4)
                    self.assertRelative(actual_cp, expected_cp, "cp")
                    self.assertRelative(actual_k, expected_k, "k")

    def test_stage_iteration_matches_source_transcription(self) -> None:
        T0, H0, alpha, R, phi = 1250.0, 256493.20980295012, 3.973413639490531, 287.4, 0.97
        expected_cp, expected_k = self.source_gas(T0, alpha, R)
        for _ in range(50):
            cp_new, k_new = self.source_gas(
                T0 - H0 * phi**2 / (2 * expected_cp), alpha, R
            )
            if abs(cp_new - expected_cp) < 1e-9:
                break
            expected_cp, expected_k = cp_new, k_new

        actual_cp, actual_k = calc_stage_cp_k(T0, H0, alpha, R, phi)
        self.assertRelative(actual_cp, expected_cp, "cp")
        self.assertRelative(actual_k, expected_k, "k")

    def test_h0_pressure_iteration_matches_source_transcription(self) -> None:
        for T0, p0, p2 in (
            (288.0, 101325.0, 60795.0),
            (700.0, 1.0e6, 5.0e5),
            (850.0, 1.5e6, 8.333333333333333e5),
            (1250.0, 2.0e6, 8.0e5),
        ):
            for alpha in (1.0, 1.8, 3.973413639490531, math.inf):
                for phi in (0.95, 0.97):
                    with self.subTest(T0=T0, p0=p0, p2=p2, alpha=alpha, phi=phi):
                        expected = self.source_h0_for_pressure(
                            T0, p0, p2, alpha, 287.4, phi
                        )
                        actual = calc_h0_for_pressure(
                            T0, p0, p2, alpha, 287.4, phi
                        )
                        self.assertRelative(actual, expected, "H0")

    def test_invalid_gas_inputs_fail_explicitly(self) -> None:
        for kwargs in (
            {"T": 0.0, "alpha": 1.0, "R": 287.4},
            {"T": -1.0, "alpha": 1.0, "R": 287.4},
            {"T": float("nan"), "alpha": 1.0, "R": 287.4},
            {"T": float("inf"), "alpha": 1.0, "R": 287.4},
            {"T": 288.0, "alpha": 0.0, "R": 287.4},
            {"T": 288.0, "alpha": -1.0, "R": 287.4},
            {"T": 288.0, "alpha": float("nan"), "R": 287.4},
            {"T": 288.0, "alpha": 1.0, "R": 0.0},
            {"T": 288.0, "alpha": 1.0, "R": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calc_gas_cp_k(**kwargs)

        with self.assertRaises(GasPropertyError):
            calc_gas_cp_k(288.0, 1.0, 1e9)

    def test_invalid_stage_inputs_fail_explicitly(self) -> None:
        good = {
            "T0": 1250.0,
            "H0": 256493.20980295012,
            "alpha": 3.973413639490531,
            "R": 287.4,
            "phi": 0.97,
        }
        for name, value in (
            ("T0", 0.0),
            ("T0", float("nan")),
            ("H0", -1.0),
            ("H0", float("inf")),
            ("alpha", 0.0),
            ("alpha", float("nan")),
            ("R", 0.0),
            ("R", float("inf")),
            ("phi", 0.0),
            ("phi", 1.01),
            ("phi", float("nan")),
        ):
            kwargs = dict(good)
            kwargs[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    calc_stage_cp_k(**kwargs)

    def test_invalid_h0_pressure_inputs_fail_explicitly(self) -> None:
        good = {
            "T0": 1250.0,
            "p0": 2.0e6,
            "p2": 8.0e5,
            "alpha": 3.973413639490531,
            "R": 287.4,
            "phi": 0.97,
        }
        for name, value in (
            ("T0", 0.0),
            ("T0", float("nan")),
            ("p0", 0.0),
            ("p0", -1.0),
            ("p2", 0.0),
            ("p2", -1.0),
            ("p2", float("inf")),
            ("alpha", 0.0),
            ("alpha", float("nan")),
            ("R", 0.0),
            ("R", float("inf")),
            ("phi", 0.0),
            ("phi", 1.01),
            ("phi", float("nan")),
        ):
            kwargs = dict(good)
            kwargs[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    calc_h0_for_pressure(**kwargs)

    def test_standalone_import_does_not_load_notebook_or_numpy(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.gas import (\n"
            "    calc_gas_cp_k, calc_h0_for_pressure, calc_stage_cp_k,\n"
            ")\n"
            "cp, k = calc_gas_cp_k(1250.0, 3.973413639490531, 287.4)\n"
            "assert abs(cp / 1194.600623988671 - 1.0) < 1e-12, cp\n"
            "assert abs(k / 1.3167987238990138 - 1.0) < 1e-12, k\n"
            "cp, k = calc_stage_cp_k(\n"
            "    1250.0, 256493.20980295012, 3.973413639490531, 287.4, 0.97\n"
            ")\n"
            "assert abs(cp / 1172.93782044023 - 1.0) < 1e-12, cp\n"
            "assert abs(k / 1.3245485323903206 - 1.0) < 1e-12, k\n"
            "H0 = calc_h0_for_pressure(\n"
            "    1250.0, 2.0e6, 8.0e5, 3.973413639490531, 287.4, 0.97\n"
            ")\n"
            "assert abs(H0 / 294751.88457137416 - 1.0) < 1e-12, H0\n"
            "for name in ('numpy', 'scipy', 'cycle_link', 'optimizer_core'):\n"
            "    assert name not in sys.modules, (name, sorted(sys.modules))\n"
            "print(cp)\n"
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


class LegacyGasParityTests(unittest.TestCase):
    """Compare the extracted helper with the live notebook implementation."""

    CANDIDATE_N = 84000.0
    CANDIDATE_D_HUB = 0.065
    RELATIVE_TOLERANCE = 1e-12
    TEMPERATURES = (288.0, 699.999, 700.0, 850.0, 1250.0)
    ALPHAS = (1.0, 1.8, 3.0, math.inf)
    GAS_CONSTANTS = (287.4, 300.0)

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(ROOT / "Optimizer"))
        sys.path.insert(0, str(ROOT / "Turbinis-Calc"))
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "legacy gas comparison needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        # orchestration_core=False keeps the notebook gas/H0 functions as the
        # independent reference; fast=False keeps them uncached, so the
        # alpha/R/phi grid below cannot read a result cached for different
        # namespace values.
        cls.namespace, _ = oe.build_ns(
            cls.model, q_n_limit=0.995, fast=False, orchestration_core=False
        )

        pi0 = cls.namespace["π_т"] ** (1.0 / cls.namespace["z"])
        _, cls.stages = cls.namespace["распределить_π_const_hub"](
            cls.CANDIDATE_N, cls.CANDIDATE_D_HUB, pi0
        )

    def assertRelative(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        if expected == 0.0:
            self.assertEqual(actual, 0.0, context)
            return
        self.assertLessEqual(
            abs(actual / expected - 1.0),
            self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def test_gas_matches_legacy_on_temperature_alpha_and_r_grid(self) -> None:
        alpha_before = self.namespace["α_г"]
        R_before = self.namespace["R"]
        try:
            for R in self.GAS_CONSTANTS:
                self.namespace["R"] = R
                for alpha in self.ALPHAS:
                    self.namespace["α_г"] = alpha
                    for T in self.TEMPERATURES:
                        with self.subTest(T=T, alpha=alpha, R=R):
                            expected_cp, expected_k = self.namespace["газ"](T)
                            actual_cp, actual_k = calc_gas_cp_k(T, alpha, R)
                            self.assertRelative(actual_cp, expected_cp, "cp")
                            self.assertRelative(actual_k, expected_k, "k")
        finally:
            self.namespace["α_г"] = alpha_before
            self.namespace["R"] = R_before

    def test_stage_cp_k_matches_legacy_for_all_three_fixed_stages(self) -> None:
        self.assertEqual(len(self.stages), 3, "expected the fixed three-stage model")
        alpha = self.namespace["α_г"]
        R = self.namespace["R"]
        phi = self.namespace["φ"]

        for index, stage in enumerate(self.stages, start=1):
            with self.subTest(stage=index, T0=stage["T0"], H0=stage["H0"]):
                expected_cp, expected_k = self.namespace["cp_k_ступени"](
                    stage["T0"], stage["H0"]
                )
                actual_cp, actual_k = calc_stage_cp_k(
                    stage["T0"], stage["H0"], alpha, R, phi
                )
                self.assertRelative(actual_cp, expected_cp, "cp")
                self.assertRelative(actual_k, expected_k, "k")

    def test_h0_pressure_matches_legacy_on_temperature_and_ratio_grid(self) -> None:
        alpha = self.namespace["α_г"]
        R = self.namespace["R"]
        phi = self.namespace["φ"]
        p0 = 1.5e6
        for T0 in (288.0, 700.0, 850.0, 1250.0):
            for ratio in (0.9, 0.7, 0.5, 1.0 / 1.8, 1.0 / 2.4):
                p2 = p0 * ratio
                with self.subTest(T0=T0, ratio=ratio):
                    expected = self.namespace["H0_по_p"](T0, p0, p2)
                    actual = calc_h0_for_pressure(
                        T0, p0, p2, alpha, R, phi
                    )
                    self.assertRelative(actual, expected, "H0")

    def test_h0_pressure_matches_legacy_for_all_three_fixed_stages(self) -> None:
        self.assertEqual(len(self.stages), 3, "expected the fixed three-stage model")
        alpha = self.namespace["α_г"]
        R = self.namespace["R"]
        phi = self.namespace["φ"]

        for index, stage in enumerate(self.stages, start=1):
            with self.subTest(
                stage=index, T0=stage["T0"], p0=stage["p0"], p2=stage["p2"]
            ):
                expected = self.namespace["H0_по_p"](
                    stage["T0"], stage["p0"], stage["p2"]
                )
                actual = calc_h0_for_pressure(
                    stage["T0"], stage["p0"], stage["p2"], alpha, R, phi
                )
                self.assertRelative(actual, expected, "H0")


if __name__ == "__main__":
    unittest.main()
