"""Parity and wiring tests for the cell-22 meridional geometry extraction.

PROVENANCE / STATUS: software-drift diagnostic only.

``Optimizer/optimizer_core._evaluate(stage_core=True, full=True)`` detects the
notebook cell whose numerical block starts with ``def γ_участка(`` /
``контур, геом, x = [], [], 0.0`` and replaces it with
:func:`turbine_engine.geometry.calculate_meridional_geometry`.  With
``stage_core=False`` the original cell 22, including its pandas display
objects, is still executed as the independent comparison path.  With
``full=False`` the cell is not executed at all.

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

from turbine_engine.geometry import (  # noqa: E402
    WIDTH_ITER_MAX,
    BladeWidths,
    ContourPoint,
    MeridionalGeometry,
    MeridionalGeometryConfig,
    calculate_meridional_geometry,
)

LEGACY_FIELDS = ("контур", "геом", "γ_max", "dx_т", "γ_турб")


def _relative_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual / expected - 1.0)


def _stages() -> list[dict[str, float]]:
    return [
        {"D_c": 0.78, "l_c": 0.030, "D_л": 0.80, "l_л": 0.032},
        {"D_c": 0.84, "l_c": 0.045, "D_л": 0.86, "l_л": 0.050},
        {"D_c": 0.90, "l_c": 0.060, "D_л": 0.92, "l_л": 0.068},
    ]


def _config() -> MeridionalGeometryConfig:
    return MeridionalGeometryConfig(
        b_a_min=5e-3,
        lb_c=2.25,
        lb_l=2.75,
        lb_min_c=1.0,
        lb_min_l=1.5,
        delta_a_ratio=0.22,
        gamma_target=12.0,
        gamma_limit=25.0,
    )


class FormulaTests(unittest.TestCase):
    """Check the extracted arithmetic against a literal source transcription."""

    RELATIVE_TOLERANCE = 1e-12

    @staticmethod
    def _legacy(stages, config):
        def gamma_section(r_n1, r_v1, r_n2, r_v2, dx):
            return math.degrees(
                math.atan((r_n2 - r_n1) / dx)
                + math.atan((r_v1 - r_v2) / dx)
            )

        def width(l, lb, lb_min, delta_do, r1, r2):
            b0 = max(l / lb, config.b_a_min)
            b_max = max(l / lb_min, b0)
            gamma_at = lambda b: gamma_section(*r1, *r2, delta_do + b)
            if gamma_at(b0) <= config.gamma_target:
                return b0
            if gamma_at(b_max) > config.gamma_target:
                return b_max
            lo, hi = b0, b_max
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                lo, hi = (
                    (mid, hi)
                    if gamma_at(mid) > config.gamma_target
                    else (lo, mid)
                )
            return hi

        def radii(D, l):
            return (D / 2 + l / 2, D / 2 - l / 2)

        contour, geom, x = [], [], 0.0
        b_c = max(stages[0]["l_c"] / config.lb_c, config.b_a_min)
        r_prev = radii(stages[0]["D_c"], stages[0]["l_c"])
        contour.append((x, *r_prev, "СА1 выход"))
        for i, s in enumerate(stages):
            delta_a = config.delta_a_ratio * b_c
            r_l = radii(s["D_л"], s["l_л"])
            b_l = width(s["l_л"], config.lb_l, config.lb_min_l, delta_a, r_prev, r_l)
            x += delta_a + b_l
            contour.append((x, *r_l, f"РК{i + 1} выход"))
            geom.append(dict(b_c=b_c, b_л=b_l))
            r_prev = r_l
            if i + 1 < len(stages):
                next_stage = stages[i + 1]
                r_c = radii(next_stage["D_c"], next_stage["l_c"])
                delta_a_l = config.delta_a_ratio * b_l
                b_c = width(
                    next_stage["l_c"],
                    config.lb_c,
                    config.lb_min_c,
                    delta_a_l,
                    r_prev,
                    r_c,
                )
                x += max(
                    delta_a_l + b_c,
                    max(abs(r_c[0] - r_prev[0]), abs(r_c[1] - r_prev[1]))
                    / math.tan(math.radians(config.gamma_limit)),
                )
                contour.append((x, *r_c, f"СА{i + 2} выход"))
                r_prev = r_c

        gamma_n = [
            math.degrees(math.atan((b[1] - a[1]) / (b[0] - a[0])))
            for a, b in zip(contour, contour[1:])
        ]
        gamma_v = [
            math.degrees(math.atan(-(b[2] - a[2]) / (b[0] - a[0])))
            for a, b in zip(contour, contour[1:])
        ]
        gamma_max = max(n + v for n, v in zip(gamma_n, gamma_v))
        dx_t = contour[-1][0]
        gamma_turb = gamma_section(
            contour[0][1], contour[0][2], contour[-1][1], contour[-1][2], dx_t
        )
        return {
            "контур": contour,
            "геом": geom,
            "γ_max": gamma_max,
            "dx_т": dx_t,
            "γ_турб": gamma_turb,
        }

    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        self.assertLessEqual(
            _relative_error(actual, expected),
            self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def test_literal_source_transcription_matches_every_field(self) -> None:
        expected = self._legacy(_stages(), _config())
        actual = calculate_meridional_geometry(_stages(), _config()).legacy_values()

        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        self.assertEqual(len(actual["контур"]), len(expected["контур"]))
        for index, (actual_point, expected_point) in enumerate(
            zip(actual["контур"], expected["контур"]), start=1
        ):
            self.assertEqual(actual_point[3], expected_point[3], f"point {index}")
            for field in range(3):
                with self.subTest(point=index, field=field):
                    self.assertFiniteNear(
                        float(actual_point[field]),
                        float(expected_point[field]),
                        f"point {index}[{field}]",
                    )
        self.assertEqual(len(actual["геом"]), len(expected["геом"]))
        for index, (actual_widths, expected_widths) in enumerate(
            zip(actual["геом"], expected["геом"]), start=1
        ):
            self.assertEqual(set(actual_widths), set(expected_widths))
            for field, expected_value in expected_widths.items():
                with self.subTest(section=index, field=field):
                    self.assertFiniteNear(
                        float(actual_widths[field]),
                        float(expected_value),
                        f"geom {index} {field}",
                    )
        for name in ("γ_max", "dx_т", "γ_турб"):
            with self.subTest(field=name):
                self.assertFiniteNear(actual[name], expected[name], name)

    def test_result_is_immutable_and_legacy_keys_are_exact(self) -> None:
        result = calculate_meridional_geometry(_stages(), _config())
        self.assertIsInstance(result, MeridionalGeometry)
        self.assertIsInstance(result.contour[0], ContourPoint)
        self.assertIsInstance(result.geom[0], BladeWidths)
        self.assertEqual(set(result.legacy_values()), set(LEGACY_FIELDS))
        with self.assertRaises(AttributeError):
            result.gamma_max = 0.0  # type: ignore[misc]

    def test_width_iteration_limit_is_the_source_limit(self) -> None:
        self.assertEqual(WIDTH_ITER_MAX, 50)


class StandaloneImportTests(unittest.TestCase):
    def test_import_does_not_load_notebook_or_numpy(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.geometry import (\n"
            "    MeridionalGeometryConfig,\n"
            "    calculate_meridional_geometry,\n"
            ")\n"
            "cfg = MeridionalGeometryConfig(\n"
            "    5e-3, 2.25, 2.75, 1.0, 1.5, 0.22, 12.0, 25.0,\n"
            ")\n"
            "stages = [\n"
            "    {'D_c': 0.78, 'l_c': 0.030, 'D_л': 0.80, 'l_л': 0.032},\n"
            "    {'D_c': 0.84, 'l_c': 0.045, 'D_л': 0.86, 'l_л': 0.050},\n"
            "]\n"
            "r = calculate_meridional_geometry(stages, cfg)\n"
            "assert r.legacy_values()['γ_max'] > 0\n"
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
        source = Path(inspect.getfile(calculate_meridional_geometry)).read_text(
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


class NotebookCell22ParityTests(unittest.TestCase):
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
                    "cell-22 parity needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        # Legacy orchestration keeps the notebook path independent; in
        # particular cells 16/20 are the notebook calculations.
        cls.namespace, _ = oe.build_ns(
            cls.model, q_n_limit=cls.Q_N_LIMIT, fast=False,
            orchestration_core=False,
        )
        cls.geometry_cell = next(
            i
            for i in cls.model.full_idx
            if "def γ_участка(" in cls.model.code[i]
            and "def ширина(" in cls.model.code[i]
            and "def r_нв(" in cls.model.code[i]
            and "контур, геом, x = [], [], 0.0" in cls.model.code[i]
        )
        cls.cell22_source = cls.model.code[cls.geometry_cell]

    def _legacy_namespace(self) -> dict:
        namespace = dict(self.namespace)
        displays: list[int] = []

        def display(*_args, **_kwargs):
            displays.append(1)

        namespace["display"] = display
        exec(compile(self.cell22_source, "ячейка 22", "exec"), namespace)
        namespace["_display_calls"] = displays
        return namespace

    def _config_from(self, namespace) -> MeridionalGeometryConfig:
        return MeridionalGeometryConfig(
            b_a_min=namespace["b_a_мин"],
            lb_c=namespace["lb_c"],
            lb_l=namespace["lb_л"],
            lb_min_c=namespace["lb_мин_c"],
            lb_min_l=namespace["lb_мин_л"],
            delta_a_ratio=namespace["δа_к_ba"],
            gamma_target=namespace["γ_цель"],
            gamma_limit=namespace["γ_макс"],
        )

    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        if expected == 0.0:
            self.assertEqual(actual, 0.0, context)
            return
        self.assertLessEqual(
            _relative_error(actual, expected), self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def _assert_matches_legacy(self, actual: dict, namespace: dict) -> None:
        self.assertEqual(set(actual), set(LEGACY_FIELDS))
        self.assertEqual(len(actual["контур"]), len(namespace["контур"]))
        for index, (actual_point, expected_point) in enumerate(
            zip(actual["контур"], namespace["контур"]), start=1
        ):
            self.assertEqual(actual_point[3], expected_point[3], f"point {index}")
            for field in range(3):
                with self.subTest(point=index, field=field):
                    self.assertFiniteNear(
                        float(actual_point[field]),
                        float(expected_point[field]),
                        f"point {index}[{field}]",
                    )
        self.assertEqual(len(actual["геом"]), len(namespace["геом"]))
        for index, (actual_widths, expected_widths) in enumerate(
            zip(actual["геом"], namespace["геом"]), start=1
        ):
            self.assertEqual(set(actual_widths), set(expected_widths))
            for field, expected_value in expected_widths.items():
                with self.subTest(section=index, field=field):
                    self.assertFiniteNear(
                        float(actual_widths[field]),
                        float(expected_value),
                        f"geom {index} {field}",
                    )
        for name in ("γ_max", "dx_т", "γ_турб"):
            with self.subTest(field=name):
                self.assertFiniteNear(actual[name], namespace[name], name)

    def test_every_field_matches_actual_legacy_cell22_execution(self) -> None:
        namespace = self._legacy_namespace()
        result = calculate_meridional_geometry(
            namespace["ступени"], self._config_from(namespace)
        )
        self._assert_matches_legacy(result.legacy_values(), namespace)

    def test_non_default_constants_are_read_from_the_namespace(self) -> None:
        namespace = dict(self.namespace)
        namespace.update(
            {
                "b_a_мин": 6e-3,
                "lb_c": 2.0,
                "lb_л": 2.5,
                "lb_мин_c": 0.9,
                "lb_мин_л": 1.4,
                "δа_к_ba": 0.20,
                "γ_цель": 10.0,
                "γ_макс": 20.0,
            }
        )
        exec(compile(self.cell22_source, "ячейка 22", "exec"), namespace)
        result = calculate_meridional_geometry(
            namespace["ступени"], self._config_from(namespace)
        )
        self._assert_matches_legacy(result.legacy_values(), namespace)

        default = calculate_meridional_geometry(
            self.namespace["ступени"], self._config_from(self.namespace)
        )
        self.assertNotEqual(result.gamma_turbine, default.gamma_turbine)


    def test_legacy_cell22_builds_presentation_objects(self) -> None:
        import pandas as pd

        namespace = self._legacy_namespace()
        for name in ("контур_df", "участки", "табл_мер", "ширины_df"):
            self.assertIsInstance(namespace[name], pd.DataFrame, name)
        self.assertEqual(len(namespace["_display_calls"]), 2)

    def test_detection_marker_is_the_expected_cell(self) -> None:
        self.assertIn("def γ_участка(", self.cell22_source)
        self.assertIn("def ширина(", self.cell22_source)
        self.assertIn("def r_нв(", self.cell22_source)
        self.assertIn("контур, геом, x = [], [], 0.0", self.cell22_source)
        self.assertIn("контур_df = pd.DataFrame", self.cell22_source)


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
                    "geometry wiring needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.model = oc.load_model()
        cls.geometry_cell = next(
            i
            for i in cls.model.full_idx
            if "def γ_участка(" in cls.model.code[i]
            and "def ширина(" in cls.model.code[i]
            and "def r_нв(" in cls.model.code[i]
            and "контур, геом, x = [], [], 0.0" in cls.model.code[i]
        )
        cls.original_cell_source = cls.model.code[cls.geometry_cell]
        cls.x = oc.warm_start(cls.CFG)

    def tearDown(self) -> None:
        self.model.code[self.geometry_cell] = self.original_cell_source

    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite value, got {actual!r}"
        )
        if expected == 0.0:
            self.assertEqual(actual, 0.0, context)
            return
        self.assertLessEqual(
            _relative_error(actual, expected), self.RELATIVE_TOLERANCE,
            f"{context}: expected ~{expected!r}, got {actual!r}",
        )

    def assertValueMatches(self, actual, expected, context: str) -> None:
        if isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected), context)
            for index, (actual_item, expected_item) in enumerate(
                zip(actual, expected)
            ):
                self.assertFiniteNear(
                    float(actual_item), float(expected_item), f"{context}[{index}]"
                )
            return
        self.assertFiniteNear(float(actual), float(expected), context)

    def test_wired_full_path_uses_module_and_skips_cell22_execution(self) -> None:
        calls: list[int] = []
        original = self.oc.calculate_meridional_geometry

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        self.oc.calculate_meridional_geometry = counted
        try:
            # Keep the detection markers but make execution fail loudly.
            self.model.code[self.geometry_cell] = (
                "def γ_участка(*args):\n"
                "    return 0.0\n"
                "def ширина(*args, **kwargs):\n"
                "    return 0.0\n"
                "def r_нв(*args):\n"
                "    return (0.0, 0.0)\n"
                "контур, геом, x = [], [], 0.0\n"
                "raise AssertionError('cell 22 must not be executed')\n"
            )
            result = self.oc._evaluate(
                self.x, self.model, self.CFG, full=True, stage_core=True
            )
        finally:
            self.oc.calculate_meridional_geometry = original
            self.model.code[self.geometry_cell] = self.original_cell_source

        self.assertGreater(len(calls), 0, "wired path did not call the module")
        self.assertTrue(math.isfinite(result["gamma_max"]))
        self.assertEqual(len(result["geom"]), 3)

    def test_legacy_full_path_still_executes_cell22(self) -> None:
        self.model.code[self.geometry_cell] = (
            "raise AssertionError('cell 22 executed')"
        )
        with self.assertRaisesRegex(AssertionError, "cell 22 executed"):
            self.oc._evaluate(
                self.x, self.model, self.CFG, full=True, stage_core=False
            )

    def test_default_full_false_does_not_execute_cell22(self) -> None:
        self.model.code[self.geometry_cell] = (
            "raise AssertionError('cell 22 executed')"
        )
        result = self.oc._evaluate(
            self.x, self.model, self.CFG, full=False, stage_core=True
        )
        self.assertNotIn("gamma_max", result)
        self.assertNotIn("geom", result)

    def test_missing_cell22_fails_loudly(self) -> None:
        original = self.model.full_idx
        self.model.full_idx = [
            i for i in original if i != self.geometry_cell
        ]
        try:
            with self.assertRaisesRegex(RuntimeError, "meridional geometry cell"):
                self.oc._evaluate(
                    self.x, self.model, self.CFG, full=True, stage_core=True
                )
        finally:
            self.model.full_idx = original

    def test_wired_and_legacy_full_results_and_validation_stay_identical(self) -> None:
        wired = self.oc._evaluate(
            self.x, self.model, self.CFG, full=True, stage_core=True
        )
        legacy = self.oc._evaluate(
            self.x, self.model, self.CFG, full=True, stage_core=False
        )

        self.assertEqual(set(wired), set(legacy), "full result keys changed")
        for field in (
            "gamma_max", "rho_root", "lambda_c1", "alpha1_root",
            "phi_ref", "psi_ref", "N_val", "eta", "sigma_max", "lc1",
            "D_t", "u_t", "n_t",
        ):
            with self.subTest(field=field):
                self.assertValueMatches(wired[field], legacy[field], field)

        self.assertEqual(len(wired["geom"]), len(legacy["geom"]))
        for index, (actual, expected) in enumerate(
            zip(wired["geom"], legacy["geom"]), start=1
        ):
            self.assertEqual(set(actual), set(expected), f"geom {index} keys")
            for field, expected_value in expected.items():
                with self.subTest(geom=index, field=field):
                    self.assertFiniteNear(
                        float(actual[field]),
                        float(expected_value),
                        f"geom {index} field {field}",
                    )

        self.assertEqual(len(wired["stages"]), len(legacy["stages"]))
        for index, (actual, expected) in enumerate(
            zip(wired["stages"], legacy["stages"]), start=1
        ):
            self.assertEqual(set(actual), set(expected), f"stage {index} keys")
            for field, expected_value in expected.items():
                with self.subTest(stage=index, field=field):
                    self.assertFiniteNear(
                        float(actual[field]),
                        float(expected_value),
                        f"stage {index} field {field}",
                    )

        wired_checks = self.oc.validate(wired, self.CFG, full=True)
        legacy_checks = self.oc.validate(legacy, self.CFG, full=True)
        self.assertEqual(len(wired_checks), len(legacy_checks))
        for actual, expected in zip(wired_checks, legacy_checks):
            self.assertEqual(actual["name"], expected["name"])
            self.assertEqual(actual["kind"], expected["kind"])
            self.assertEqual(actual["ok"], expected["ok"])
            with self.subTest(check=expected["name"]):
                self.assertFiniteNear(
                    float(actual["amount"]),
                    float(expected["amount"]),
                    f"check {expected['name']} amount",
                )


if __name__ == "__main__":
    unittest.main()
