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
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine.gas as gas_module  # noqa: E402
from turbine_engine.cycle import calc_cp_kerosene  # noqa: E402
from turbine_engine.diagnostics import (  # noqa: E402
    NUMERICAL_FAILURE,
    NumericalConvergenceError,
    SolverDiagnosticsError,
)
from turbine_engine.gas import (  # noqa: E402
    STAGE_CP_K_MAX_ITER,
    STAGE_CP_K_TOL,
    GasPropertyError,
    calc_gas_cp_k,
    calc_h0_for_pressure,
    calc_stage_cp_k,
    calc_textbook_gas_cp_k,
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


class ExplicitNonconvergenceTests(unittest.TestCase):
    """Exhausted loops must report a typed failure, not a silent last iterate.

    ``calc_gas_cp_k`` is replaced by a deterministic drifting sequence so the
    real 50-iteration limits are exercised without inventing a physical input
    on which the source formula would fail to converge.
    """

    T0 = 1250.0
    H0 = 256493.20980295012
    P0 = 2.0e6
    P2 = 8.0e5
    ALPHA = 3.973413639490531
    R = 287.4
    PHI = 0.97

    @staticmethod
    def drifting_gas(calls: list[float]):
        def gas(T: float, alpha: float, R: float) -> tuple[float, float]:
            # Every call returns a different cp, so abs(cp_new - cp) never
            # drops below 1e-9 and the source iteration limit is reached.
            index = len(calls)
            calls.append(T)
            return 1000.0 + index, 1.4 + 0.001 * index

        return gas

    def test_stage_cp_k_exhaustion_raises_coherent_typed_error(self) -> None:
        calls: list[float] = []
        with mock.patch.object(
            gas_module, "calc_gas_cp_k", side_effect=self.drifting_gas(calls)
        ):
            with self.assertRaises(NumericalConvergenceError) as raised:
                calc_stage_cp_k(self.T0, self.H0, self.ALPHA, self.R, self.PHI)

        error = raised.exception
        self.assertIsInstance(error, SolverDiagnosticsError)
        self.assertEqual(len(calls), STAGE_CP_K_MAX_ITER + 1)
        self.assertEqual(STAGE_CP_K_MAX_ITER, 50)
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "gas.calc_stage_cp_k")
        self.assertEqual(error.reason, "cp_k_iteration_exhausted")

        record = error.last_valid_state
        self.assertEqual(record["iterations"], STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["iteration_limit"], STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["tolerance"], STAGE_CP_K_TOL)
        self.assertEqual(record["cp"], 1000.0 + STAGE_CP_K_MAX_ITER - 1)
        self.assertEqual(record["k"], 1.4 + 0.001 * (STAGE_CP_K_MAX_ITER - 1))
        self.assertEqual(record["cp_new"], 1000.0 + STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["k_new"], 1.4 + 0.001 * STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["residual"], record["cp_new"] - record["cp"])
        self.assertEqual(record["residual"], 1.0)
        self.assertEqual(
            record["T_mean"],
            self.T0 - self.H0 * self.PHI**2 / (2 * record["cp"]),
        )
        self.assertEqual(record["H0"], self.H0)
        # The last evaluated state and the proposed update are separate but
        # coherent: the final gas call really received the recorded T_mean.
        self.assertEqual(calls[-1], record["T_mean"])
        self.assertEqual(error.as_record()["last_valid_state"], record)

    def test_h0_pressure_exhaustion_raises_coherent_typed_error(self) -> None:
        calls: list[float] = []
        with mock.patch.object(
            gas_module, "calc_gas_cp_k", side_effect=self.drifting_gas(calls)
        ):
            with self.assertRaises(NumericalConvergenceError) as raised:
                calc_h0_for_pressure(
                    self.T0, self.P0, self.P2, self.ALPHA, self.R, self.PHI
                )

        error = raised.exception
        self.assertEqual(len(calls), STAGE_CP_K_MAX_ITER + 1)
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "gas.calc_h0_for_pressure")
        self.assertEqual(error.reason, "h0_pressure_iteration_exhausted")

        record = error.last_valid_state
        self.assertEqual(record["iterations"], STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["iteration_limit"], STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["tolerance"], STAGE_CP_K_TOL)
        self.assertEqual(record["cp"], 1000.0 + STAGE_CP_K_MAX_ITER - 1)
        self.assertEqual(record["k"], 1.4 + 0.001 * (STAGE_CP_K_MAX_ITER - 1))
        self.assertEqual(record["cp_new"], 1000.0 + STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["k_new"], 1.4 + 0.001 * STAGE_CP_K_MAX_ITER)
        self.assertEqual(record["residual"], record["cp_new"] - record["cp"])
        self.assertEqual(record["residual"], 1.0)
        self.assertEqual(
            record["H0"],
            record["cp"]
            * self.T0
            * (1 - (self.P2 / self.P0) ** ((record["k"] - 1) / record["k"])),
        )
        self.assertEqual(
            record["T_mean"],
            self.T0 - record["H0"] * self.PHI**2 / (2 * record["cp"]),
        )
        self.assertEqual(calls[-1], record["T_mean"])
        self.assertEqual(error.as_record()["last_valid_state"], record)

    def test_stage_zero_cp_remains_domain_error(self) -> None:
        with mock.patch.object(
            gas_module, "calc_gas_cp_k", return_value=(0.0, 1.4)
        ):
            with self.assertRaises(GasPropertyError) as raised:
                calc_stage_cp_k(self.T0, self.H0, self.ALPHA, self.R, self.PHI)
        self.assertNotIsInstance(raised.exception, NumericalConvergenceError)

    def test_interrupts_propagate_from_exhaustion_loops(self) -> None:
        calls = (
            (
                calc_stage_cp_k,
                (self.T0, self.H0, self.ALPHA, self.R, self.PHI),
            ),
            (
                calc_h0_for_pressure,
                (self.T0, self.P0, self.P2, self.ALPHA, self.R, self.PHI),
            ),
        )
        for function, arguments in calls:
            received: list[float] = []

            def interrupted_gas(
                T: float, alpha: float, R: float, _received=received
            ) -> tuple[float, float]:
                _received.append(T)
                if len(_received) == 1:
                    return 1000.0, 1.4
                raise KeyboardInterrupt

            with self.subTest(function=function.__name__):
                with mock.patch.object(
                    gas_module, "calc_gas_cp_k", side_effect=interrupted_gas
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        function(*arguments)


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


class TextbookGasPropertyTests(unittest.TestCase):
    """Opt-in published true-cp fit; source-drift check, not validation."""

    R = 287.4
    RELATIVE_TOLERANCE = 1e-12

    @staticmethod
    def source_textbook_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
        # Independent literal transcription of the published Appendix 6 fit.
        if T < 750.0:
            cp_kcal = (
                (0.0174 / alpha + 0.2407)
                + (0.0093 / alpha + 0.0193) * (0.0025 * T - 0.875)
                + (0.002 - 0.001056 / (alpha - 0.2))
                * (0.000025 * T * T - 0.0275 * T + 6.5625)
            )
        else:
            cp_kcal = (
                (0.0267 / alpha + 0.26)
                + (0.0133 / alpha + 0.032) * (0.001176 * T - 0.88235)
                - (0.00374 + 0.0094 / (alpha * alpha + 10.0))
                * (0.0000055556 * T * T - 0.013056 * T + 6.67)
            )
        cp = cp_kcal * 4187.0
        return cp, cp / (cp - R)

    def assertRelative(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        self.assertLessEqual(
            abs(actual / expected - 1.0),
            self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def test_source_hand_values_and_branch_selection(self) -> None:
        # Values evaluated independently from the printed correlation.
        cases = (
            (300.0, 1.0, 1067.2977025, 1.3685098687670514),
            (749.999, 1.0, 1200.412572157971, 1.314782083800565),
            (750.0, 1.0, 1200.2883220955, 1.3148249276979296),
            (1250.0, math.inf, 1182.5355090875, 1.3210687064497924),
        )
        for T, alpha, expected_cp, expected_k in cases:
            with self.subTest(T=T, alpha=alpha):
                actual_cp, actual_k = calc_textbook_gas_cp_k(T, alpha, self.R)
                self.assertRelative(actual_cp, expected_cp, "cp")
                self.assertRelative(actual_k, expected_k, "k")

        for T in (288.0, 300.0, 749.999, 750.0, 1250.0):
            for alpha in (1.0, 4.0, math.inf):
                with self.subTest(T=T, alpha=alpha):
                    expected_cp, expected_k = self.source_textbook_cp_k(
                        T, alpha, self.R
                    )
                    actual_cp, actual_k = calc_textbook_gas_cp_k(T, alpha, self.R)
                    self.assertRelative(actual_cp, expected_cp, "cp")
                    self.assertRelative(actual_k, expected_k, "k")

        # The published finite jump at 750 K is preserved, not smoothed.
        lower = calc_textbook_gas_cp_k(749.999, 1.0, self.R)[0]
        upper = calc_textbook_gas_cp_k(750.0, 1.0, self.R)[0]
        self.assertGreater(abs(upper - lower), 0.1)

    def test_air_infinite_alpha_uses_analytic_reciprocal_limit(self) -> None:
        for T in (288.0, 300.0, 749.999, 750.0, 1250.0):
            if T < 750.0:
                cp_kcal = (
                    0.2407
                    + 0.0193 * (0.0025 * T - 0.875)
                    + 0.002 * (0.000025 * T * T - 0.0275 * T + 6.5625)
                )
            else:
                cp_kcal = (
                    0.26
                    + 0.032 * (0.001176 * T - 0.88235)
                    - 0.00374
                    * (0.0000055556 * T * T - 0.013056 * T + 6.67)
                )
            expected_cp = cp_kcal * 4187.0
            expected_k = expected_cp / (expected_cp - self.R)
            actual_cp, actual_k = calc_textbook_gas_cp_k(T, math.inf, self.R)
            with self.subTest(T=T):
                self.assertRelative(actual_cp, expected_cp, "air cp")
                self.assertRelative(actual_k, expected_k, "air k")

    def test_invalid_inputs_fail_explicitly_without_clamping(self) -> None:
        for kwargs in (
            {"T": True, "alpha": 1.0, "R": self.R},
            {"T": "300", "alpha": 1.0, "R": self.R},
            {"T": 300.0, "alpha": True, "R": self.R},
            {"T": 300.0, "alpha": "1", "R": self.R},
            {"T": 300.0, "alpha": 1.0, "R": False},
            {"T": 300.0, "alpha": 1.0, "R": "287.4"},
            {"T": 0.0, "alpha": 1.0, "R": self.R},
            {"T": -1.0, "alpha": 1.0, "R": self.R},
            {"T": float("nan"), "alpha": 1.0, "R": self.R},
            {"T": float("inf"), "alpha": 1.0, "R": self.R},
            {"T": 300.0, "alpha": float("nan"), "R": self.R},
            {"T": 300.0, "alpha": 1.0, "R": 0.0},
            {"T": 300.0, "alpha": 1.0, "R": -1.0},
            {"T": 300.0, "alpha": 1.0, "R": float("nan")},
            {"T": 300.0, "alpha": 1.0, "R": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    calc_textbook_gas_cp_k(**kwargs)

        for alpha in (0.0, -1.0, 0.999, -math.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaises(GasPropertyError):
                    calc_textbook_gas_cp_k(300.0, alpha, self.R)

        # cp <= R and finite-input overflow are domain errors, not clamps.
        with self.assertRaises(GasPropertyError):
            calc_textbook_gas_cp_k(300.0, 1.0, 1e9)
        with self.assertRaises(GasPropertyError):
            calc_textbook_gas_cp_k(1e308, 1.0, self.R)

    def test_invalid_selector_type_or_value_is_rejected(self) -> None:
        calls = (
            (
                calc_stage_cp_k,
                (1250.0, 256493.20980295012, 1.0, self.R, 0.97),
            ),
            (
                calc_h0_for_pressure,
                (1250.0, 2.0e6, 8.0e5, 1.0, self.R, 0.97),
            ),
        )
        for function, arguments in calls:
            for selector in ("bogus", "", None, 1, True):
                with self.subTest(function=function.__name__, selector=selector):
                    with self.assertRaises(ValueError):
                        function(*arguments, gas_property_basis=selector)

    def test_opt_in_stage_basis_matches_independent_fixed_point(self) -> None:
        T0, H0, alpha, phi = 1250.0, 256493.20980295012, 1.0, 0.97
        base = (T0, H0, alpha, self.R, phi)

        expected_cp, expected_k = self.source_textbook_cp_k(T0, alpha, self.R)
        for _ in range(50):
            cp_new, k_new = self.source_textbook_cp_k(
                T0 - H0 * phi**2 / (2 * expected_cp), alpha, self.R
            )
            if abs(cp_new - expected_cp) < 1e-9:
                break
            expected_cp, expected_k = cp_new, k_new
        else:
            self.fail("independent textbook stage fixed point did not converge")

        with mock.patch.object(
            gas_module,
            "calc_gas_cp_k",
            side_effect=AssertionError("legacy basis called for textbook request"),
        ):
            actual_cp, actual_k = calc_stage_cp_k(
                *base, gas_property_basis="textbook_true_cp"
            )
        self.assertRelative(actual_cp, expected_cp, "textbook stage cp")
        self.assertRelative(actual_k, expected_k, "textbook stage k")
        next_cp, _ = self.source_textbook_cp_k(
            T0 - H0 * phi**2 / (2 * actual_cp), alpha, self.R
        )
        self.assertLess(abs(next_cp - actual_cp), 1e-9)

        legacy_cp, legacy_k = calc_stage_cp_k(*base)
        explicit_cp, explicit_k = calc_stage_cp_k(
            *base, gas_property_basis="legacy_derivative"
        )
        self.assertEqual((legacy_cp, legacy_k), (explicit_cp, explicit_k))
        self.assertGreater(abs(actual_cp - legacy_cp), 1.0)
        self.assertGreater(abs(actual_k - legacy_k), 1e-3)

    def test_opt_in_pressure_basis_matches_independent_fixed_point(self) -> None:
        T0, p0, p2, alpha, phi = 1250.0, 2.0e6, 8.0e5, 1.0, 0.97
        arguments = (T0, p0, p2, alpha, self.R, phi)

        cp, k = self.source_textbook_cp_k(T0, alpha, self.R)
        expected_h0 = 0.0
        for _ in range(50):
            expected_h0 = cp * T0 * (1 - (p2 / p0) ** ((k - 1) / k))
            cp_new, k_new = self.source_textbook_cp_k(
                T0 - expected_h0 * phi**2 / (2 * cp), alpha, self.R
            )
            if abs(cp_new - cp) < 1e-9:
                break
            cp, k = cp_new, k_new
        else:
            self.fail("independent textbook pressure fixed point did not converge")

        with mock.patch.object(
            gas_module,
            "calc_gas_cp_k",
            side_effect=AssertionError("legacy basis called for textbook request"),
        ):
            actual_h0 = calc_h0_for_pressure(
                *arguments, gas_property_basis="textbook_true_cp"
            )
        self.assertRelative(actual_h0, expected_h0, "textbook H0")

        legacy_h0 = calc_h0_for_pressure(*arguments)
        explicit_h0 = calc_h0_for_pressure(
            *arguments, gas_property_basis="legacy_derivative"
        )
        self.assertEqual(legacy_h0, explicit_h0)
        self.assertGreater(abs(actual_h0 - legacy_h0), 100.0)

    def test_selected_basis_diagnostics_and_interrupts(self) -> None:
        def drifting(calls: list[float]):
            def gas(T: float, alpha: float, R: float) -> tuple[float, float]:
                calls.append(T)
                index = len(calls) - 1
                return 1000.0 + index, 1.4 + 0.001 * index

            return gas

        stage_arguments = (1250.0, 256493.20980295012, 1.0, self.R, 0.97)
        pressure_arguments = (1250.0, 2.0e6, 8.0e5, 1.0, self.R, 0.97)
        for function, arguments in (
            (calc_stage_cp_k, stage_arguments),
            (calc_h0_for_pressure, pressure_arguments),
        ):
            for target, selector in (
                ("calc_gas_cp_k", "legacy_derivative"),
                ("calc_textbook_gas_cp_k", "textbook_true_cp"),
            ):
                calls: list[float] = []
                with self.subTest(
                    function=function.__name__, selector=selector
                ):
                    with mock.patch.object(
                        gas_module, target, side_effect=drifting(calls)
                    ):
                        with self.assertRaises(NumericalConvergenceError) as raised:
                            function(
                                *arguments, gas_property_basis=selector
                            )
                    self.assertEqual(len(calls), STAGE_CP_K_MAX_ITER + 1)
                    self.assertEqual(
                        raised.exception.last_valid_state["gas_property_basis"],
                        selector,
                    )

        received: list[float] = []

        def interrupted(T: float, alpha: float, R: float) -> tuple[float, float]:
            received.append(T)
            if len(received) == 1:
                return 1000.0, 1.4
            raise KeyboardInterrupt

        with mock.patch.object(
            gas_module, "calc_textbook_gas_cp_k", side_effect=interrupted
        ):
            with self.assertRaises(KeyboardInterrupt):
                calc_stage_cp_k(
                    *stage_arguments, gas_property_basis="textbook_true_cp"
                )


if __name__ == "__main__":
    unittest.main()
