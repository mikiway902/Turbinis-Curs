"""Parity and wiring tests for the cell-20 turbine performance extraction.

PROVENANCE / STATUS: software-drift diagnostic only.

``Optimizer/optimizer_core._evaluate(stage_core=True)`` detects the notebook
cell whose numerical aggregation starts with
``g_i = [s["G_c"] / G_c1`` and replaces it with
:func:`turbine_engine.performance.calculate_turbine_performance`.  With
``stage_core=False`` the original cell 20, including its pandas table/display
code, is still executed as the independent comparison path.

These tests use one fixed candidate (``N = 84e3``, ``D_hub = 0.065 m`` and the
model's own initial pressure ratio).  They do not run a scan, sweep, warm
start, or optimizer search, and the candidate is not a validated physical
baseline or an acceptable design.
"""

from __future__ import annotations

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

from turbine_engine.performance import (  # noqa: E402
    TurbinePerformance,
    calculate_turbine_performance,
)

LEGACY_FIELDS = (
    "g_i",
    "L_тΣ",
    "η_т_турб",
    "h_вz",
    "η_л_турб",
    "p_т_полн",
    "cp_т",
    "k_т",
    "H_т_полн",
    "η_т_полн",
    "N_т",
    "N_вал",
)


def _relative_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual / expected - 1.0)


class FormulaTests(unittest.TestCase):
    """Check the extracted arithmetic against a literal source transcription."""

    RELATIVE_TOLERANCE = 1e-12

    def test_literal_source_transcription_matches_every_field(self) -> None:
        stages = [
            {"G_c": 1.5, "L_т": 120000.0, "c2": 180.0, "p_ст_п": 300000.0, "T_ст_п": 1100.0},
            {"G_c": 2.0, "L_т": 140000.0, "c2": 200.0, "p_ст_п": 250000.0, "T_ст_п": 1050.0},
            {"G_c": 2.5, "L_т": 160000.0, "c2": 220.0, "p_ст_п": 200000.0, "T_ст_п": 1000.0},
        ]
        G_c1 = 1.5
        H_t = 450000.0
        T_g_total = 1250.0
        p_g_total = 1000000.0
        eta_m = 0.99
        G_v = 1.4
        L_k = 250000.0
        eta_r = 0.98

        def gas(T: float) -> tuple[float, float]:
            return 1100.0, 1.33

        # Literal transcription of the source block (pre-reducer shaft power).
        g_i = [s["G_c"] / G_c1 for s in stages]
        L_t_sum = sum(g * s["L_т"] for g, s in zip(g_i, stages))
        eta_t_turb = L_t_sum / H_t
        h_v_exit = stages[-1]["c2"] ** 2 / 2
        eta_l_turb = (L_t_sum + h_v_exit) / H_t
        p_t_total = stages[-1]["p_ст_п"]
        cp_t, k_t = gas(0.5 * (T_g_total + stages[-1]["T_ст_п"]))
        H_t_total = cp_t * T_g_total * (
            1 - (p_t_total / p_g_total) ** ((k_t - 1) / k_t)
        )
        eta_t_total = L_t_sum / H_t_total
        N_turbine = L_t_sum * G_c1 * eta_m
        N_shaft = N_turbine - G_v * L_k
        N_shaft_reducer_out = N_shaft * eta_r
        expected = {
            "g_i": g_i,
            "L_тΣ": L_t_sum,
            "η_т_турб": eta_t_turb,
            "h_вz": h_v_exit,
            "η_л_турб": eta_l_turb,
            "p_т_полн": p_t_total,
            "cp_т": cp_t,
            "k_т": k_t,
            "H_т_полн": H_t_total,
            "η_т_полн": eta_t_total,
            "N_т": N_turbine,
            "N_вал": N_shaft,
        }

        result = calculate_turbine_performance(
            stages,
            G_c1=G_c1,
            H_t=H_t,
            T_g_total=T_g_total,
            p_g_total=p_g_total,
            eta_m=eta_m,
            G_v=G_v,
            L_k=L_k,
            eta_r=eta_r,
            gas=gas,
        )
        actual = result.legacy_values()

        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        self.assertEqual(len(actual["g_i"]), len(expected["g_i"]))
        for index, (a, e) in enumerate(zip(actual["g_i"], expected["g_i"])):
            self.assertLessEqual(
                _relative_error(a, e), self.RELATIVE_TOLERANCE, f"g_i[{index}]"
            )
        for name in LEGACY_FIELDS[1:]:
            with self.subTest(field=name):
                self.assertLessEqual(
                    _relative_error(actual[name], expected[name]),
                    self.RELATIVE_TOLERANCE,
                    name,
                )
        self.assertAlmostEqual(
            result.N_shaft_reducer_out / N_shaft_reducer_out, 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            result.N_shaft_reducer_out,
            result.N_shaft * eta_r,
            delta=1e-9,
        )
        self.assertLess(result.N_shaft_reducer_out, result.N_shaft)

    def test_result_is_immutable_and_gas_is_called_once(self) -> None:
        calls: list[float] = []

        def gas(T: float) -> tuple[float, float]:
            calls.append(T)
            return 1000.0, 1.3

        result = calculate_turbine_performance(
            [{"G_c": 1.0, "L_т": 1.0, "c2": 1.0, "p_ст_п": 1.0, "T_ст_п": 1.0}],
            G_c1=1.0,
            H_t=1.0,
            T_g_total=2.0,
            p_g_total=2.0,
            eta_m=1.0,
            G_v=0.0,
            L_k=0.0,
            eta_r=1.0,
            gas=gas,
        )
        self.assertIsInstance(result, TurbinePerformance)
        self.assertEqual(calls, [1.5])
        with self.assertRaises(AttributeError):
            result.L_t_sum = 0.0  # type: ignore[misc]


class StandaloneImportTests(unittest.TestCase):
    def test_import_does_not_load_notebook_or_numpy(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.performance import "
            "calculate_turbine_performance, TurbinePerformance\n"
            "assert 'numpy' not in sys.modules, sorted(sys.modules)\n"
            "assert 'scipy' not in sys.modules, sorted(sys.modules)\n"
            "assert 'pandas' not in sys.modules, sorted(sys.modules)\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "r = calculate_turbine_performance(\n"
            "    [{'G_c': 1.0, 'L_т': 1.0, 'c2': 1.0, "
            "'p_ст_п': 1.0, 'T_ст_п': 1.0}],\n"
            "    G_c1=1.0, H_t=1.0, T_g_total=2.0, p_g_total=2.0,\n"
            "    eta_m=1.0, G_v=0.0, L_k=0.0, eta_r=1.0,\n"
            "    gas=lambda T: (1.0, 1.3),\n"
            ")\n"
            "assert isinstance(r, TurbinePerformance)\n"
            "assert r.N_shaft == 1.0\n"
            "assert r.N_shaft_reducer_out == 1.0\n"
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


class NotebookCell20ParityTests(unittest.TestCase):
    RELATIVE_TOLERANCE = 1e-12
    CANDIDATE_N = 84000.0
    CANDIDATE_D_HUB = 0.065
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
                    "cell-20 parity needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        # The legacy distribution keeps the notebook orchestrations, so the
        # comparison namespace is the source path rather than the extracted one.
        cls.namespace, _ = oe.build_ns(
            cls.model, q_n_limit=cls.Q_N_LIMIT, orchestration_core=False
        )
        cls.pi0 = cls.namespace["π_т"] ** (1.0 / cls.namespace["z"])
        cls.scale_pi, cls.stages = cls.namespace["распределить_π_const_hub"](
            cls.CANDIDATE_N, cls.CANDIDATE_D_HUB, cls.pi0
        )
        cls.performance_cell = next(
            i
            for i in cls.model.core_idx
            if 'g_i = [s["G_c"] / G_c1' in cls.model.code[i]
            and "N_вал = (" in cls.model.code[i]
        )
        cls.cell20_source = cls.model.code[cls.performance_cell]

    def _legacy_namespace(self) -> dict:
        namespace = dict(self.namespace)
        namespace["ступени"] = list(self.stages)
        displays: list[int] = []

        def display(*_args, **_kwargs):
            displays.append(1)

        namespace["display"] = display
        exec(compile(self.cell20_source, "ячейка 20", "exec"), namespace)
        namespace["_display_calls"] = displays
        return namespace

    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
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

    def test_every_field_matches_actual_legacy_cell20_execution(self) -> None:
        namespace = self._legacy_namespace()
        result = calculate_turbine_performance(
            self.stages,
            G_c1=namespace["G_c1"],
            H_t=namespace["H_т"],
            T_g_total=namespace["T_г_полн"],
            p_g_total=namespace["p_г_полн"],
            eta_m=namespace["η_м"],
            G_v=namespace["G_в"],
            L_k=namespace["L_к"],
            eta_r=namespace["η_р"],
            gas=namespace["газ"],
        )
        actual = result.legacy_values()

        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        self.assertEqual(set(actual), set(LEGACY_FIELDS), "legacy aliases changed")
        self.assertEqual(len(actual["g_i"]), len(namespace["g_i"]))
        for index, (a, e) in enumerate(zip(actual["g_i"], namespace["g_i"])):
            self.assertFiniteNear(a, e, f"g_i[{index}]")
        for name in LEGACY_FIELDS[1:]:
            with self.subTest(field=name):
                self.assertFiniteNear(actual[name], namespace[name], name)
        self.assertAlmostEqual(
            result.N_shaft_reducer_out,
            result.N_shaft * namespace["η_р"],
            delta=1e-9,
        )

    def test_legacy_cell20_builds_dataframes_and_display(self) -> None:
        import pandas as pd

        namespace = self._legacy_namespace()
        self.assertIsInstance(namespace["табл_турб"], pd.DataFrame)
        self.assertIsInstance(namespace["сверка"], pd.DataFrame)
        self.assertEqual(len(namespace["_display_calls"]), 1)
        self.assertEqual(namespace["сверка"].shape[0], 6)

    def test_detection_marker_is_the_expected_cell(self) -> None:
        self.assertIn('g_i = [s["G_c"] / G_c1', self.cell20_source)
        self.assertIn("N_вал = (", self.cell20_source)
        # The presentation-only code must not be part of the pure module.
        module_source = inspect.getsource(calculate_turbine_performance)
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
                    "evaluate wiring needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.model = oc.load_model()
        cls.performance_cell = next(
            i
            for i in cls.model.core_idx
            if 'g_i = [s["G_c"] / G_c1' in cls.model.code[i]
            and "N_вал = (" in cls.model.code[i]
        )
        cls.original_cell_source = cls.model.code[cls.performance_cell]
        cls.x = oc.warm_start(cls.CFG)

    def tearDown(self) -> None:
        self.model.code[self.performance_cell] = self.original_cell_source

    def assertValueNear(self, actual: float, expected: float, context: str) -> None:
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

    def test_wired_path_uses_module_and_skips_cell20_execution(self) -> None:
        calls: list[int] = []
        original = self.oc.calculate_turbine_performance

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        self.oc.calculate_turbine_performance = counted
        try:
            # Keep the detection marker but make execution fail loudly.
            self.model.code[self.performance_cell] = (
                'g_i = [s["G_c"] / G_c1\n'
                "# N_вал = (\n"
                "raise AssertionError('cell 20 must not be executed')"
            )
            result = self.oc._evaluate(
                self.x, self.model, self.CFG, stage_core=True
            )
        finally:
            self.oc.calculate_turbine_performance = original
            self.model.code[self.performance_cell] = self.original_cell_source

        self.assertGreater(len(calls), 0, "wired path did not call the module")
        self.assertTrue(math.isfinite(result["N_val"]))
        self.assertTrue(math.isfinite(result["eta"]))

    def test_legacy_path_still_executes_cell20(self) -> None:
        self.model.code[self.performance_cell] = (
            "raise AssertionError('cell 20 executed')"
        )
        with self.assertRaisesRegex(AssertionError, "cell 20 executed"):
            self.oc._evaluate(self.x, self.model, self.CFG, stage_core=False)

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
