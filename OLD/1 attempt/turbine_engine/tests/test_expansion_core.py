"""Parity and wiring tests for the cell-9 expansion extraction.

PROVENANCE / STATUS: software-drift diagnostic only.

``Optimizer/optimizer_core._evaluate(stage_core=True)`` detects the notebook
cell whose numerical block starts with ``π_т = p_г_полн / p_т`` and replaces it
with :func:`turbine_engine.expansion.calculate_turbine_expansion`.  With
``stage_core=False`` the original cell 9, including its pandas table/display
code, is still executed as the independent comparison path.

These tests use one fixed namespace (the model's own initial operating point)
and do not run a scan, sweep, warm start, or optimizer search.  The candidate is
not a validated physical baseline or an acceptable design.
"""

from __future__ import annotations

import ast
import inspect
import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

from turbine_engine.diagnostics import (  # noqa: E402
    NUMERICAL_FAILURE,
    NumericalConvergenceError,
)
from turbine_engine.expansion import (  # noqa: E402
    EXPANSION_ITER_MAX,
    EXPANSION_TEMPERATURE_TOL,
    TurbineExpansion,
    calculate_turbine_expansion,
)

LEGACY_FIELDS = (
    "π_т",
    "T_т",
    "H_т",
    "cp0",
    "k0",
    "v_т",
    "A_т0",
    "n_max0",
    "n_раб0",
    "c0_т",
    "u_η",
    "D_η",
    "l_η",
    "D_т0",
)


def _relative_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual / expected - 1.0)


class FormulaTests(unittest.TestCase):
    """Check the extracted arithmetic against a literal source transcription."""

    RELATIVE_TOLERANCE = 1e-12

    def _call(self, gas, **overrides):
        kwargs = dict(
            p_g_total=2.0e6,
            p_t=2.0e5,
            T_g_total=1250.0,
            eta_l_expected=0.90,
            gas=gas,
            R=287.4,
            G_l_ost=1.25,
            c_ta=190.0,
            sigma_l=290.0,
            k_n=6.8e-6,
            q_n=0.995,
            y0=0.54,
            l_t_D_t=0.26,
        )
        kwargs.update(overrides)
        return calculate_turbine_expansion(**kwargs)

    @staticmethod
    def _legacy(pi_args, gas):
        p_g_total, p_t, T_g_total, eta_l_expected = pi_args
        R, G_l_ost, c_ta, sigma_l, k_n, q_n, y0, l_t_D_t = (
            287.4, 1.25, 190.0, 290.0, 6.8e-6, 0.995, 0.54, 0.26
        )
        # Literal transcription of the source block.
        π_т = p_g_total / p_t
        T_т = T_g_total * π_т**-0.25
        for _ in range(50):
            cp0, k0 = gas(0.5 * (T_g_total + T_т))
            H_т = cp0 * T_g_total * (1 - π_т ** ((1 - k0) / k0))
            T_т_н = T_g_total - H_т * eta_l_expected / cp0
            if abs(T_т_н - T_т) < 1e-6:
                break
            T_т = T_т_н
        T_т = T_т_н
        v_т = R * T_т / p_t
        A_т0 = G_l_ost * v_т / c_ta
        n_max0 = math.sqrt(sigma_l / (k_n * A_т0))
        n_раб0 = q_n * n_max0
        c0_т = math.sqrt(2 * H_т)
        u_η = y0 * c0_т
        D_η = 60 * u_η / (math.pi * n_раб0)
        l_η = A_т0 / (math.pi * D_η)
        D_т0 = math.sqrt(A_т0 / (math.pi * l_t_D_t))
        return {
            "π_т": π_т,
            "T_т": T_т,
            "H_т": H_т,
            "cp0": cp0,
            "k0": k0,
            "v_т": v_т,
            "A_т0": A_т0,
            "n_max0": n_max0,
            "n_раб0": n_раб0,
            "c0_т": c0_т,
            "u_η": u_η,
            "D_η": D_η,
            "l_η": l_η,
            "D_т0": D_т0,
        }

    def test_literal_source_transcription_matches_every_field(self) -> None:
        def gas(T: float) -> tuple[float, float]:
            return 1140.0 - 0.03 * (T - 900.0), 1.335

        args = (2.0e6, 2.0e5, 1250.0, 0.90)
        expected = self._legacy(args, gas)
        actual = self._call(gas).legacy_values()

        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        for name in LEGACY_FIELDS:
            with self.subTest(field=name):
                self.assertLessEqual(
                    _relative_error(actual[name], expected[name]),
                    self.RELATIVE_TOLERANCE,
                    name,
                )

    def test_exact_iteration_limit_raises_typed_nonconvergence(self) -> None:
        calls: list[tuple[float, float]] = []

        def gas(T: float) -> tuple[float, float]:
            # k changes on every call, so the fixed-point temperature never
            # settles within 1e-6 and the source's 50-iteration limit is hit.
            cp = 1000.0 + 0.5 * len(calls)
            k = 1.30 + 1e-4 * len(calls)
            calls.append((cp, k))
            return cp, k

        T_g = 1250.0
        pi_t = 2.0e6 / 2.0e5

        def iterate(cp: float, k: float, T_in: float) -> float:
            H = cp * T_g * (1 - pi_t ** ((1 - k) / k))
            return T_g - H * 0.90 / cp

        cp_last = 1000.0 + 0.5 * (EXPANSION_ITER_MAX - 1)
        k_last = 1.30 + 1e-4 * (EXPANSION_ITER_MAX - 1)
        cp_prev = 1000.0 + 0.5 * (EXPANSION_ITER_MAX - 2)
        k_prev = 1.30 + 1e-4 * (EXPANSION_ITER_MAX - 2)
        T_initial = T_g * pi_t**-0.25
        T_prev = iterate(cp_prev, k_prev, T_initial)
        T_last = iterate(cp_last, k_last, T_prev)

        with self.assertRaises(NumericalConvergenceError) as raised:
            self._call(gas)

        self.assertEqual(len(calls), EXPANSION_ITER_MAX)
        self.assertEqual(EXPANSION_ITER_MAX, 50)
        self.assertEqual(EXPANSION_TEMPERATURE_TOL, 1e-6)
        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "expansion.calculate_turbine_expansion")
        self.assertEqual(error.reason, "temperature_iteration_exhausted")
        self.assertEqual(error.last_valid_state["iterations"], EXPANSION_ITER_MAX)
        self.assertLessEqual(
            _relative_error(error.last_valid_state["T_t_previous"], T_prev),
            self.RELATIVE_TOLERANCE * 10,
        )
        self.assertLessEqual(
            _relative_error(error.last_valid_state["T_t"], T_last),
            self.RELATIVE_TOLERANCE * 10,
        )
        self.assertLessEqual(
            _relative_error(error.last_valid_state["T_t_next"], T_last),
            self.RELATIVE_TOLERANCE * 10,
        )
        self.assertLessEqual(
            _relative_error(
                error.last_valid_state["residual"], T_last - T_prev
            ),
            self.RELATIVE_TOLERANCE * 10,
        )
        self.assertEqual(
            error.last_valid_state["iteration_limit"], EXPANSION_ITER_MAX
        )
        self.assertEqual(
            error.last_valid_state["tolerance"], EXPANSION_TEMPERATURE_TOL
        )
        self.assertIsNotNone(error.last_valid_state["residual"])

    def test_strict_temperature_tolerance_changes_iteration_count(self) -> None:
        T_g = 1250.0
        pi_t = 2.0e6 / 2.0e5
        eta = 0.90
        T_initial = T_g * pi_t**-0.25

        def constant_gas_for_delta(delta: float):
            target = T_initial + delta
            q = 1.0 - (1.0 - target / T_g) / eta
            k = math.log(pi_t) / (math.log(q) + math.log(pi_t))
            calls: list[float] = []

            def gas(T: float) -> tuple[float, float]:
                calls.append(T)
                return 1000.0, k

            return gas, calls

        below_gas, below_calls = constant_gas_for_delta(1e-7)
        below = self._call(below_gas)
        self.assertEqual(len(below_calls), 1, "a difference below 1e-6 must break")
        self.assertLessEqual(
            _relative_error(below.T_t, T_initial + 1e-7), self.RELATIVE_TOLERANCE
        )

        above_gas, above_calls = constant_gas_for_delta(1e-5)
        above = self._call(above_gas)
        self.assertEqual(
            len(above_calls), 2, "a difference above 1e-6 must iterate again"
        )
        self.assertLessEqual(
            _relative_error(above.T_t, T_initial + 1e-5), self.RELATIVE_TOLERANCE
        )

    def test_result_is_immutable_and_legacy_keys_are_exact(self) -> None:
        result = self._call(lambda T: (1000.0, 1.33))
        self.assertIsInstance(result, TurbineExpansion)
        self.assertEqual(set(result.legacy_values()), set(LEGACY_FIELDS))
        with self.assertRaises(AttributeError):
            result.T_t = 0.0  # type: ignore[misc]


class StandaloneImportTests(unittest.TestCase):
    def test_import_does_not_load_notebook_or_numpy(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.expansion import (\n"
            "    calculate_turbine_expansion, TurbineExpansion,\n"
            ")\n"
            "r = calculate_turbine_expansion(\n"
            "    1.0e6, 1.0e5, 1250.0, 0.90,\n"
            "    gas=lambda T: (1000.0, 1.33),\n"
            "    R=287.4, G_l_ost=1.0, c_ta=180.0, sigma_l=290.0,\n"
            "    k_n=6.8e-6, q_n=0.995, y0=0.54, l_t_D_t=0.25,\n"
            ")\n"
            "assert isinstance(r, TurbineExpansion)\n"
            "assert isinstance(r.legacy_values()['T_т'], float)\n"
            "for name in ('numpy', 'scipy', 'pandas', 'cycle_link', "
            "'optimizer_core'):\n"
            "    assert name not in sys.modules, (name, sorted(sys.modules))\n"
            "print('ok')\n"
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

    def test_module_has_no_numeric_or_presentation_imports(self) -> None:
        source = Path(inspect.getfile(calculate_turbine_expansion)).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertEqual(imported & {"numpy", "scipy", "pandas"}, set())
        self.assertNotIn("DataFrame(", source)
        self.assertNotIn("pd.", source)


class NotebookCell9ParityTests(unittest.TestCase):
    RELATIVE_TOLERANCE = 1e-12
    Q_N_LIMIT = 0.995

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "cell-9 parity needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        # Legacy orchestration keeps the notebook path independent; in
        # particular ``газ`` inside this namespace is the notebook function.
        cls.namespace, _ = oe.build_ns(
            cls.model, q_n_limit=cls.Q_N_LIMIT, fast=False,
            orchestration_core=False,
        )
        cls.expansion_cell = next(
            i
            for i in cls.model.core_idx
            if "π_т = p_г_полн / p_т" in cls.model.code[i]
            and "табл_п1_31" in cls.model.code[i]
        )
        cls.cell9_source = cls.model.code[cls.expansion_cell]

    def _legacy_namespace(self) -> dict:
        namespace = dict(self.namespace)
        displays: list[int] = []

        def display(*_args, **_kwargs):
            displays.append(1)

        namespace["display"] = display
        exec(compile(self.cell9_source, "ячейка 9", "exec"), namespace)
        namespace["_display_calls"] = displays
        return namespace

    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        self.assertLessEqual(
            _relative_error(actual, expected), self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def test_every_field_matches_actual_legacy_cell9_execution(self) -> None:
        namespace = self._legacy_namespace()
        result = calculate_turbine_expansion(
            namespace["p_г_полн"],
            namespace["p_т"],
            namespace["T_г_полн"],
            namespace["η_л_ожид"],
            gas=namespace["газ"],
            R=namespace["R"],
            G_l_ost=namespace["G_л_ост"],
            c_ta=namespace["c_та"],
            sigma_l=namespace["σ_л"],
            k_n=namespace["k_n"],
            q_n=namespace["q_n"],
            y0=namespace["y0"],
            l_t_D_t=namespace["l_т_D_т"],
        )
        actual = result.legacy_values()

        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        for name in LEGACY_FIELDS:
            with self.subTest(field=name):
                self.assertFiniteNear(actual[name], namespace[name], name)

    def test_legacy_cell9_builds_presentation_table(self) -> None:
        import pandas as pd

        namespace = self._legacy_namespace()
        self.assertIsInstance(namespace["табл_п1_31"], pd.DataFrame)
        # Cell 9's presentation is a trailing Jupyter expression; ``exec`` does
        # not call the namespace ``display`` helper.  The wired path skips the
        # whole cell (covered below), so no table is built there either.
        self.assertEqual(namespace["_display_calls"], [])

    def test_detection_marker_is_the_expected_cell(self) -> None:
        self.assertIn("π_т = p_г_полн / p_т", self.cell9_source)
        self.assertIn("табл_п1_31", self.cell9_source)
        module_source = inspect.getsource(calculate_turbine_expansion)
        self.assertNotIn("DataFrame", module_source)
        self.assertNotIn("display", module_source)


class EvaluateWiringTests(unittest.TestCase):
    RELATIVE_TOLERANCE = 1e-12
    CFG = {
        "q_n": 0.995,
        "seal_type": "honeycomb",
        "phi": 0.965,
        "psi": 0.95,
    }

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "expansion wiring needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.model = oc.load_model()
        cls.expansion_cell = next(
            i
            for i in cls.model.core_idx
            if "π_т = p_г_полн / p_т" in cls.model.code[i]
            and "табл_п1_31" in cls.model.code[i]
        )
        cls.original_cell_source = cls.model.code[cls.expansion_cell]
        cls.x = oc.warm_start(cls.CFG)

    def tearDown(self) -> None:
        self.model.code[self.expansion_cell] = self.original_cell_source

    def assertValueNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        self.assertLessEqual(
            _relative_error(actual, expected), self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def test_wired_path_uses_module_and_skips_cell9_execution(self) -> None:
        calls: list[int] = []
        original = self.oc.calculate_turbine_expansion

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        self.oc.calculate_turbine_expansion = counted
        try:
            # Keep the detection markers but make execution fail loudly.
            self.model.code[self.expansion_cell] = (
                "π_т = p_г_полн / p_т\n"
                "# табл_п1_31\n"
                "raise AssertionError('cell 9 must not be executed')"
            )
            result = self.oc._evaluate(
                self.x, self.model, self.CFG, stage_core=True
            )
        finally:
            self.oc.calculate_turbine_expansion = original
            self.model.code[self.expansion_cell] = self.original_cell_source

        self.assertGreater(len(calls), 0, "wired path did not call the module")
        self.assertTrue(math.isfinite(result["N_val"]))
        self.assertTrue(math.isfinite(result["eta"]))

    def test_legacy_path_still_executes_cell9(self) -> None:
        self.model.code[self.expansion_cell] = (
            "raise AssertionError('cell 9 executed')"
        )
        with self.assertRaisesRegex(AssertionError, "cell 9 executed"):
            self.oc._evaluate(self.x, self.model, self.CFG, stage_core=False)

    def test_missing_cell9_fails_loudly(self) -> None:
        original = self.model.core_idx
        self.model.core_idx = [
            i for i in original if i != self.expansion_cell
        ]
        try:
            with self.assertRaisesRegex(RuntimeError, "expansion cell"):
                self.oc._evaluate(self.x, self.model, self.CFG, stage_core=True)
        finally:
            self.model.core_idx = original

    def test_wired_and_legacy_evaluate_results_stay_identical(self) -> None:
        wired = self.oc._evaluate(self.x, self.model, self.CFG, stage_core=True)
        legacy = self.oc._evaluate(self.x, self.model, self.CFG, stage_core=False)

        self.assertEqual(set(wired), set(legacy), "result keys changed")
        for name in ("N_val", "eta", "sigma_max", "lc1", "D_t", "u_t", "n_t"):
            with self.subTest(field=name):
                self.assertValueNear(wired[name], legacy[name], name)
        self.assertEqual(len(wired["stages"]), len(legacy["stages"]))
        for index, (actual, expected) in enumerate(
            zip(wired["stages"], legacy["stages"]), start=1
        ):
            self.assertEqual(set(actual), set(expected), f"stage {index} keys")
            for field, expected_value in expected.items():
                with self.subTest(stage=index, field=field):
                    self.assertValueNear(
                        float(actual[field]), float(expected_value), field
                    )


if __name__ == "__main__":
    unittest.main()
