"""Parity and wiring tests for the conventional multi-stage cell-16 extraction.

PROVENANCE / STATUS: software-drift diagnostic only.

``Optimizer/optimizer_core._evaluate(stage_core=True)`` detects notebook cell 16
by its semantic markers (``турбина``, ``распределить_π``, ``решить``,
``N_на_валу`` and the patched ``_N_of(G)`` wrapper), skips the cell, and calls
:func:`turbine_engine.baseline.calculate_baseline_turbine`.  With
``stage_core=False`` the patched notebook cell is still executed in full,
including its presentation tables, as the independent comparison path.

The tests use fixed warm-start candidates only.  They do not run a scan, sweep,
warm start, differential evolution, or any optimizer search.  The candidates
are software-drift anchors, not a validated physical baseline or an acceptable
design.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

from turbine_engine.baseline import (  # noqa: E402
    BaselineTurbine,
    BaselineTurbineResult,
    calculate_baseline_turbine,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NUMERICAL_FAILURE,
    NumericalConvergenceError,
)

BASELINE_FIELDS = (
    "G_в_цикл",
    "π0",
    "G_в",
    "G_c1",
    "G_л_ост",
    "A_т0",
    "G_a",
    "G_b",
    "N_a",
    "N_b",
    "A_т",
    "A_факт",
    "n_т",
    "D_т",
    "u_т",
    "z_расч",
    "ступени",
    "σ_л_факт",
)
RESULT_FIELDS = (
    "G_v_cycle",
    "pi0",
    "G_v",
    "G_c1",
    "G_l_ost",
    "A_t0",
    "G_a",
    "G_b",
    "N_a",
    "N_b",
    "A_t",
    "A_fact",
    "n_t",
    "D_t",
    "u_t",
    "z_calc",
    "stages",
    "iterations",
    "sigma_l_fact",
)

CFGS = (
    {"q_n": 0.995, "seal_type": "honeycomb", "phi": 0.965, "psi": 0.95},
    {"q_n": 0.99, "seal_type": "abradable", "phi": 0.94, "psi": 0.93},
)
RELATIVE_TOLERANCE = 1e-12


def _relative_error(actual: float, expected: float) -> float:
    if expected == 0.0:
        return 0.0 if actual == 0.0 else math.inf
    return abs(actual / expected - 1.0)


def _baseline_cell(oc, model) -> int:
    cell = next(
        (
            i
            for i in model.core_idx
            if "def турбина(" in model.code[i]
            and "def распределить_π(" in model.code[i]
            and "def решить(" in model.code[i]
            and "def N_на_валу(" in model.code[i]
            and "def _N_of(G)" in model.code[i]
        ),
        None,
    )
    if cell is None:
        raise AssertionError("baseline cell 16 not found")
    return cell


def _build_pre_namespace(oc, model, cfg):
    """Reproduce ``_evaluate`` namespace preparation and run cells before 16.

    This is intentionally a test-side transcription of the setup block in
    ``_evaluate``.  It runs the notebook cells raw (no adapters) so the module
    can be compared with the actual legacy cell-16 execution, and it leaves the
    original flow state untouched for both paths.
    """

    x = oc.warm_start(cfg)
    p = oc.декодировать(x, cfg)
    if cfg.get("D_override"):
        p["D"] = [float(v) for v in cfg["D_override"]]
    if cfg.get("lD_override") is not None:
        p["lD"] = float(cfg["lD_override"])
    c = oc.cycle_point(model, p["pi"], p["Tg"], p["N"])
    ns = dict(model.base)
    ns["T_г_полн"] = p["Tg"]
    ns["охлаждение_лопаток"] = c["охлаждение"]
    ns["g_охл"] = c["g_ohl"]
    ns["g_воз"] = c["g_voz"]
    exec(compile(model.code[model.param_idx], "параметры", "exec"), ns)
    ns["q_n"] = float(cfg.get("q_n", 0.995))
    seal = str(cfg.get("seal_type", "honeycomb"))
    if seal == "honeycomb":
        ns["δ_отн"], ns["δ_мин"] = 0.005, 0.2e-3
    else:
        ns["δ_отн"], ns["δ_мин"] = 0.0085, 0.5e-3
    ns.update({
        "π_к": c["pi"], "p_г_полн": c["p_g"], "p_т": c["p_t"],
        "G_в": c["G"], "G_c1": c["G"] * c["g_t"],
        "G_л_ост": c["G"] * (c["g_t"] + c["g_voz"]),
        "L_к": c["L_k"], "c_та": c["c_t"], "N_зад": c["N"],
        "g_т": c["g_t"], "α_г": c["alpha_g"],
        "η_т_полн_цикл": c["eta_t_cycle"],
        "N_т_цикл": (
            c["G"]
            * c["L_t"]
            * (c["g_t"] + c["g_voz"])
            * float(model.cycle["η_м_тур"])
        ),
        "р": c["d"],
    })
    cycle_copy = dict(model.cycle)
    cycle_copy["c_т"] = c["c_t"]
    cycle_copy["T_г_полн"] = p["Tg"]
    ns["Ц"] = cycle_copy
    ns.update({
        "D_ст": p["D"], "l_т_D_т": p["lD"], "α1_ст": p["a1"],
        "π_веса": p["w"], "κ_ст": p["k"], "σ_л": p["sigma"],
        "φ": p["phi"], "ψ": p["psi"],
    })
    baseline_cell = _baseline_cell(oc, model)
    for i in model.core_idx:
        if i == baseline_cell:
            break
        exec(compile(model.code[i], f"ядро {i}", "exec"), ns)
    return x, ns


def _run_legacy_cell16(model, namespace):
    ns = dict(namespace)
    displays: list[int] = []

    def display(*_args, **_kwargs):
        displays.append(1)

    ns["display"] = display
    exec(compile(model.code[_baseline_cell_local(model)], "ячейка 16", "exec"), ns)
    return ns, displays


def _baseline_cell_local(model) -> int:
    return next(
        i
        for i in model.core_idx
        if "def турбина(" in model.code[i]
        and "def _N_of(G)" in model.code[i]
    )


class _НетРешения(Exception):
    pass


def _minimal_namespace(**overrides):
    """Small dependency-injected namespace for failure-semantics tests."""

    def H0_по_p(T0, p0, p2):
        return 100000.0 * (1.0 - p2 / p0)

    def ступень_авто(T0, p0, H0, D_c, D_л, n, G_c, G_л, a1, k, law):
        return {
            "π_ст": p0 / 2.0e5,
            "p2": 2.0e5,
            "D_л": D_л,
            "l_л": 0.05,
            "G_c": G_c,
            "L_т": 100000.0,
            "c2": 200.0,
            "cp": 1000.0,
            "T_ст": 1000.0,
            "T_ст_п": 1020.0,
            "k": 1.33,
        }

    ns = {
        "G_в": 1.0, "G_c1": 1.0, "G_л_ост": 1.0, "A_т0": 0.1,
        "π_т": 4.0, "z": 1, "D_ст": [1.0],
        "T_г_полн": 1250.0, "p_г_полн": 8.0e5, "μ": 1.0,
        "p_т": 2.0e5, "π_веса": [1.0], "α1_ст": [15.0],
        "κ_ст": [1.1], "закон_ст": ["Γ"],
        "ступень_авто": ступень_авто, "H0_по_p": H0_по_p,
        "НетРешения": _НетРешения, "π_ст_макс": 5.0,
        "c_та": 200.0, "c0_т": 200.0, "σ_л": 290.0, "k_n": 6.8e-6,
        "q_n": 0.995, "l_т_D_т": 0.25, "y0": 0.54,
        "η_м": 1.0, "L_к": 0.0, "η_р": 1.0, "N_зад": 100000.0,
    }
    ns.update(overrides)
    return ns


class _ParityAssertions(unittest.TestCase):
    def assertFiniteNear(self, actual: float, expected: float, context: str) -> None:
        self.assertTrue(
            math.isfinite(actual), f"{context}: expected finite, got {actual!r}"
        )
        self.assertLessEqual(
            _relative_error(actual, expected),
            RELATIVE_TOLERANCE,
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

    def assertStageDictsEqual(self, actual, expected) -> None:
        self.assertEqual(len(actual), len(expected), "stage count changed")
        for index, (actual_stage, expected_stage) in enumerate(
            zip(actual, expected), start=1
        ):
            self.assertEqual(
                set(actual_stage), set(expected_stage),
                f"stage {index} keys changed",
            )
            for field, expected_value in expected_stage.items():
                with self.subTest(stage=index, field=field):
                    self.assertFiniteNear(
                        float(actual_stage[field]),
                        float(expected_value),
                        f"stage {index} field {field}",
                    )


class FormulaTests(_ParityAssertions):
    """Compare the module with a literal source transcription."""

    def test_per_stage_sequence_and_iteration_log_match_legacy(self) -> None:
        try:
            import numpy as np
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    f"cell-16 parity needs runtime dependency '{dependency}'"
                ) from exc
            raise

        model = oc.load_model()
        _, pre = _build_pre_namespace(oc, model, CFGS[0])
        legacy, displays = _run_legacy_cell16(model, pre)
        result = calculate_baseline_turbine(pre, clip=np.clip, mean=np.mean)

        self.assertIsInstance(result, BaselineTurbineResult)
        self.assertEqual(set(result.legacy_values()), set(BASELINE_FIELDS))
        for name in BASELINE_FIELDS:
            if name == "ступени":
                continue
            with self.subTest(field=name):
                self.assertFiniteNear(
                    float(result.legacy_values()[name]),
                    float(legacy[name]),
                    name,
                )

        self.assertStageDictsEqual(result.stages, legacy["ступени"])
        self.assertEqual(len(result.iterations), len(legacy["_r"]["итерации"]))
        for index, (actual, expected) in enumerate(
            zip(result.iterations, legacy["_r"]["итерации"])
        ):
            self.assertEqual(len(actual), len(expected), f"iteration {index} shape")
            for field, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            ):
                self.assertFiniteNear(
                    float(actual_value),
                    float(expected_value),
                    f"iteration {index} field {field}",
                )
        self.assertEqual(len(displays), 1, "legacy cell 16 display call changed")

    def test_pressure_closure_and_stage_geometry_sequence(self) -> None:
        try:
            import numpy as np
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    f"cell-16 closure test needs runtime dependency '{dependency}'"
                ) from exc
            raise

        model = oc.load_model()
        _, pre = _build_pre_namespace(oc, model, CFGS[0])
        result = calculate_baseline_turbine(pre, clip=np.clip, mean=np.mean)
        stages = result.stages
        self.assertEqual(len(stages), pre["z"])
        for index, stage in enumerate(stages, start=1):
            with self.subTest(stage=index):
                expected_diameter = pre["D_ст"][index - 1] * result.D_t
                self.assertEqual(stage["D_c"], expected_diameter)
                self.assertEqual(stage["D_л"], expected_diameter)
                self.assertFiniteNear(
                    stage["π_ст"], stage["p0"] / stage["p2"], "π_ст=p0/p2"
                )
                if index > 1:
                    previous = stages[index - 2]
                    reconstructed = previous["p2"] * (
                        1
                        + (pre["μ"] * previous["c2"]) ** 2
                        / (2 * previous["cp"] * previous["T_ст"])
                    ) ** (previous["k"] / (previous["k"] - 1))
                    self.assertFiniteNear(
                        stage["p0"], reconstructed, "interstage total pressure"
                    )
        self.assertLessEqual(
            abs(stages[-1]["p2"] / pre["p_т"] - 1.0),
            1e-6,
            "last-stage outlet pressure closure",
        )

    def test_repeated_calls_are_deterministic_and_leave_namespace_unchanged(
        self,
    ) -> None:
        try:
            import numpy as np
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    f"cell-16 determinism test needs runtime dependency '{dependency}'"
                ) from exc
            raise

        model = oc.load_model()
        _, pre = _build_pre_namespace(oc, model, CFGS[0])
        flow_keys = ("G_в", "G_c1", "G_л_ост", "A_т0")
        before = {key: pre[key] for key in flow_keys}

        first = calculate_baseline_turbine(pre, clip=np.clip, mean=np.mean)
        second = calculate_baseline_turbine(pre, clip=np.clip, mean=np.mean)

        self.assertEqual({key: pre[key] for key in flow_keys}, before)
        for field in RESULT_FIELDS:
            first_value = getattr(first, field)
            second_value = getattr(second, field)
            if field in ("stages", "iterations"):
                self.assertEqual(len(first_value), len(second_value), field)
                continue
            self.assertEqual(first_value, second_value, field)
        self.assertStageDictsEqual(first.stages, second.stages)
        self.assertEqual(first.iterations, second.iterations)

        # A direct N_на_валу call mutates only the object, using the same
        # G/G_base scaling as the source; a following calculate() resets it.
        turbine = BaselineTurbine(pre, clip=np.clip, mean=np.mean)
        trial = 1.03 * pre["G_в"]
        pi0 = pre["π_т"] ** (1 / pre["z"])
        power, solution = turbine.N_на_валу(trial, pi0)
        self.assertTrue(math.isfinite(power))
        self.assertFiniteNear(turbine.G_v, trial, "trial G state")
        self.assertFiniteNear(
            turbine.A_t0,
            pre["A_т0"] * (trial / pre["G_в"]),
            "trial A_t0 scaling",
        )
        reset = turbine.calculate()
        for field in RESULT_FIELDS:
            if field in ("stages", "iterations"):
                continue
            with self.subTest(field=field):
                self.assertEqual(getattr(reset, field), getattr(first, field))
        self.assertStageDictsEqual(reset.stages, first.stages)


class FailureSemanticsTests(unittest.TestCase):
    def test_last_stage_pressure_guard_uses_namespace_exception(self) -> None:
        ns = _minimal_namespace(p_г_полн=2.0e5)
        turbine = BaselineTurbine(ns)
        with self.assertRaises(_НетРешения) as raised:
            turbine.турбина(10000.0, 0.3, 4.0)
        self.assertIn(
            "давление перед последней ступенью не выше p_т",
            str(raised.exception),
        )

    def test_initial_pressure_solve_failure_matches_source(self) -> None:
        def failing_stage(*_args, **_kwargs):
            raise _НетРешения("no stage")

        ns = _minimal_namespace(ступень_авто=failing_stage)
        turbine = BaselineTurbine(ns)
        with self.assertRaises(_НетРешения) as raised:
            turbine.распределить_π(10000.0, 0.3, 4.0)
        self.assertIn(
            "нет решения при начальной π ступени", str(raised.exception)
        )

    def test_missing_dependencies_fail_before_calculation(self) -> None:
        with self.assertRaises(KeyError) as raised:
            calculate_baseline_turbine({})
        self.assertIn("G_в", str(raised.exception))

    def test_shaft_power_target_is_independent_of_reducer_efficiency(self) -> None:
        ns_default = _minimal_namespace(η_р=0.98)
        ns_low = _minimal_namespace(η_р=0.50)
        pi0 = ns_default["π_т"] ** (1.0 / ns_default["z"])
        power_default, _ = BaselineTurbine(ns_default).N_на_валу(1.0, pi0)
        power_low, _ = BaselineTurbine(ns_low).N_на_валу(1.0, pi0)
        self.assertEqual(
            power_default,
            power_low,
            "baseline shaft power must target the pre-reducer plane and not "
            "depend on eta_reducer",
        )


class ExplicitDiagnosticsTests(unittest.TestCase):
    """Final-failure and exhaustion branches must not leak None/TypeError."""

    def test_failed_final_pressure_evaluation_raises_typed_error(self) -> None:
        base = _minimal_namespace()
        calls: list[int] = []

        def stage_only_once(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise _НетРешения("stage failed after the initial evaluation")
            return base["ступень_авто"](*args, **kwargs)

        turbine = BaselineTurbine(
            _minimal_namespace(ступень_авто=stage_only_once)
        )
        with self.assertRaises(NumericalConvergenceError) as raised:
            turbine.распределить_π(10000.0, 0.3, 4.0)

        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "baseline.распределить_π")
        self.assertEqual(error.reason, "no_valid_final_evaluation")
        self.assertEqual(
            error.last_valid_state["iteration_limit"],
            30,
        )
        self.assertTrue(math.isfinite(error.last_valid_state["residual"]))
        self.assertTrue(math.isfinite(error.last_valid_state["y"]))
        json.dumps(error.as_record(), allow_nan=False)

    def test_решить_propagates_typed_failure_instead_of_typeerror(self) -> None:
        base = _minimal_namespace()
        calls: list[int] = []

        def stage_only_once(*args, **kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise _НетРешения("stage failed after the initial evaluation")
            return base["ступень_авто"](*args, **kwargs)

        turbine = BaselineTurbine(
            _minimal_namespace(ступень_авто=stage_only_once)
        )
        with self.assertRaises(NumericalConvergenceError) as raised:
            turbine.N_на_валу(1.0, 4.0)

        error = raised.exception
        self.assertEqual(error.stage, "baseline.распределить_π")
        self.assertEqual(error.reason, "no_valid_final_evaluation")
        self.assertNotIsInstance(error, TypeError)

    def test_area_iteration_exhaustion_raises_typed_error(self) -> None:
        namespace = _minimal_namespace()
        turbine = BaselineTurbine(namespace)

        def always_growing_area(n_t: float, D_t: float, pi0: float):
            area = math.pi * namespace["l_т_D_т"] * D_t**2
            blade_length = 2.0 * area / (math.pi * 1.0)
            return pi0, [
                {
                    "D_л": 1.0,
                    "l_л": blade_length,
                    "c2": 200.0,
                    "π_ст": pi0,
                }
            ]

        turbine.распределить_π = always_growing_area  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            turbine.решить(4.0)

        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "baseline.решить")
        self.assertEqual(error.reason, "area_iteration_exhausted")
        state = error.last_valid_state
        self.assertEqual(state["iteration_limit"], 40)
        self.assertEqual(state["relative_tolerance"], 1e-4)

        # The recorded state must be the state actually evaluated in the last
        # iteration, not the post-relaxation proposal.
        A_t = state["A_t"]
        expected_n = namespace["q_n"] * math.sqrt(
            namespace["σ_л"] / (namespace["k_n"] * A_t)
        )
        expected_D = math.sqrt(
            A_t / (math.pi * namespace["l_т_D_т"])
        )
        self.assertTrue(math.isclose(state["n_t"], expected_n, rel_tol=1e-12))
        self.assertTrue(math.isclose(state["D_t"], expected_D, rel_tol=1e-12))
        self.assertTrue(
            math.isclose(
                state["relative_residual"],
                abs(state["A_fact"] - A_t) / A_t,
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                state["A_t_next"],
                0.5 * (A_t + state["A_fact"]),
                rel_tol=1e-12,
            )
        )
        self.assertNotEqual(state["A_t_next"], A_t)
        json.dumps(error.as_record(), allow_nan=False)

    def test_outer_flow_secant_exhaustion_raises_typed_error(self) -> None:
        turbine = BaselineTurbine(_minimal_namespace())
        solution = {
            "A_т": 0.1,
            "A_факт": 0.1,
            "n_т": 10000.0,
            "D_т": 0.3,
            "u_т": 100.0,
            "z_расч": 1.0,
            "ступени": [],
            "итерации": [],
        }
        turbine.N_на_валу = lambda G, pi0: (0.0, solution)  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            turbine.calculate()

        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "baseline.calculate")
        self.assertEqual(error.reason, "outer_flow_iteration_exhausted")
        state = error.last_valid_state
        self.assertEqual(state["iteration_limit"], 20)
        self.assertEqual(state["N_b"], 0.0)
        self.assertEqual(state["relative_tolerance"], 1e-5)
        self.assertEqual(
            state["residual"],
            abs(state["N_b"] - state["N_target"]),
        )
        self.assertEqual(
            state["relative_residual"],
            state["residual"] / abs(state["N_target"]),
        )
        json.dumps(error.as_record(), allow_nan=False)


class LastStagePressureBoundaryTests(unittest.TestCase):
    """Boundary regression for the last-stage positive-drop guard.

    Source: turbine book part 1, printed p. 92, items 138-140.  The prescribed
    last-stage outlet is ``p2z = p_t`` and the drop is
    ``cp*T0z*[1-(p2z/p0z*)^((k-1)/k)]``, so only ``p0 <= p_t`` has no positive
    drop.  The checked text has no 2 % margin.  These tests use recording stubs:
    passing the guard means the stage function is reached, not that the stage is
    physically feasible or converged.
    """

    RATIOS_WITH_POSITIVE_DROP = (1.000001, 1.01, 1.02, 1.020001)
    RATIOS_WITHOUT_POSITIVE_DROP = (1.0, 0.999999)

    def _recording_namespace(self, p_gas: float):
        h0_calls: list[tuple[float, float, float]] = []
        stage_calls: list[tuple[tuple, dict]] = []
        base = _minimal_namespace(p_г_полн=p_gas)

        def H0_по_p(T0: float, p0: float, p2: float) -> float:
            h0_calls.append((T0, p0, p2))
            return base["H0_по_p"](T0, p0, p2)

        def ступень_авто(*args, **kwargs):
            stage_calls.append((args, kwargs))
            return base["ступень_авто"](*args, **kwargs)

        namespace = dict(base, H0_по_p=H0_по_p, ступень_авто=ступень_авто)
        return namespace, h0_calls, stage_calls

    def test_any_positive_drop_reaches_h0_and_stage_calculations(self) -> None:
        p_t = 2.0e5
        for ratio in self.RATIOS_WITH_POSITIVE_DROP:
            with self.subTest(ratio=ratio):
                namespace, h0_calls, stage_calls = self._recording_namespace(
                    p_t * ratio
                )
                result = BaselineTurbine(namespace).турбина(10000.0, 0.3, 4.0)

                self.assertEqual(len(result), 1)
                self.assertEqual(len(h0_calls), 1)
                self.assertEqual(len(stage_calls), 1)
                _, p0, p2 = h0_calls[0]
                self.assertEqual(p2, p_t)
                self.assertGreater(p0, p_t)
                h0 = stage_calls[0][0][2]
                self.assertGreater(h0, 0.0)

    def test_exact_and_below_p_t_are_rejected_before_h0_or_stage(self) -> None:
        p_t = 2.0e5
        for ratio in self.RATIOS_WITHOUT_POSITIVE_DROP:
            with self.subTest(ratio=ratio):
                namespace, h0_calls, stage_calls = self._recording_namespace(
                    p_t * ratio
                )
                with self.assertRaises(_НетРешения) as raised:
                    BaselineTurbine(namespace).турбина(10000.0, 0.3, 4.0)
                self.assertIn(
                    "давление перед последней ступенью не выше p_т",
                    str(raised.exception),
                )
                self.assertIn("нет положительного перепада", str(raised.exception))
                self.assertEqual(h0_calls, [])
                self.assertEqual(stage_calls, [])


class SourceBoundaryParityTests(unittest.TestCase):
    """The active module and active notebook must share the same boundary."""

    def test_active_notebook_cell16_has_no_percentage_margin(self) -> None:
        notebook_path = ROOT / "Turbinis-Calc" / "Main_Turbinis_Calc_Book.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cell = next(
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
            and "def турбина(" in "".join(cell.get("source", []))
            and "def _N_of(G)" in "".join(cell.get("source", []))
        )
        source = "".join(cell["source"])
        self.assertIn("if p0 <= p_т:", source)
        self.assertNotIn("p_т * 1.02", source)
        self.assertIn("нет положительного перепада", source)
        self.assertEqual(cell.get("outputs", []), [])
        self.assertIsNone(cell.get("execution_count"))

    def test_extracted_module_has_no_percentage_margin(self) -> None:
        source = Path(inspect.getfile(BaselineTurbine)).read_text(encoding="utf-8")
        self.assertIn("if p0 <= p_t:", source)
        self.assertNotIn("p_t * 1.02", source)
        self.assertIn("нет положительного перепада", source)


class LegacyTableTests(_ParityAssertions):
    def test_legacy_cell16_builds_both_presentation_tables(self) -> None:
        try:
            import pandas as pd
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    f"cell-16 table test needs runtime dependency '{dependency}'"
                ) from exc
            raise

        model = oc.load_model()
        _, pre = _build_pre_namespace(oc, model, CFGS[0])
        legacy, displays = _run_legacy_cell16(model, pre)

        self.assertIsInstance(legacy["табл_итер"], pd.DataFrame)
        self.assertIsInstance(legacy["табл_мн"], pd.DataFrame)
        self.assertEqual(list(legacy["табл_итер"].columns), [
            "итерация", "A_т, см²", "n_т, об/мин", "D_т, мм", "u_т, м/с",
            "z по п. 54", "π посл. ступени", "A факт., см²",
        ])
        self.assertEqual(len(legacy["табл_итер"]), len(legacy["_r"]["итерации"]))
        self.assertEqual(len(legacy["табл_мн"]), 8)
        self.assertEqual(len(displays), 1)


class StandaloneImportTests(unittest.TestCase):
    def test_import_and_calculation_do_not_load_numpy_or_pandas(self) -> None:
        code = textwrap.dedent("""
            import sys

            class НетРешения(Exception):
                pass

            def H0_по_p(T0, p0, p2):
                return 100000.0 * (1.0 - p2 / p0)

            p_t = 2.0e5

            def ступень_авто(T0, p0, H0, D_c, D_л, n, G_c, G_л, a1, k, law):
                return {
                    "π_ст": p0 / p_t,
                    "p2": p_t,
                    "D_л": D_л,
                    "l_л": 0.05,
                    "G_c": G_c,
                    "L_т": 100000.0,
                    "c2": 200.0,
                    "cp": 1000.0,
                    "T_ст": 1000.0,
                    "T_ст_п": 1020.0,
                    "k": 1.33,
                }

            ns = {
                "G_в": 1.0, "G_c1": 1.0, "G_л_ост": 1.0, "A_т0": 0.1,
                "π_т": 4.0, "z": 1, "D_ст": [1.0],
                "T_г_полн": 1250.0, "p_г_полн": 8.0e5, "μ": 1.0,
                "p_т": p_t, "π_веса": [1.0], "α1_ст": [15.0],
                "κ_ст": [1.1], "закон_ст": ["Γ"],
                "ступень_авто": ступень_авто, "H0_по_p": H0_по_p,
                "НетРешения": НетРешения, "π_ст_макс": 5.0,
                "c_та": 200.0, "σ_л": 290.0, "k_n": 6.8e-6,
                "q_n": 0.995, "l_т_D_т": 0.25, "c0_т": 200.0, "y0": 0.54,
                "η_м": 1.0, "L_к": 0.0, "η_р": 1.0, "N_зад": 100000.0,
            }

            from turbine_engine.baseline import (
                BaselineTurbineResult, calculate_baseline_turbine,
            )
            result = calculate_baseline_turbine(ns)
            assert isinstance(result, BaselineTurbineResult)
            assert len(result.stages) == 1
            assert abs(result.G_v - 1.0) < 1e-9
            assert isinstance(result.legacy_values()["ступени"], list)
            for name in ("numpy", "scipy", "pandas"):
                assert name not in sys.modules, (name, sorted(sys.modules))
            print("ok")
        """)
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
        source = Path(inspect.getfile(BaselineTurbine)).read_text(encoding="utf-8")
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
        self.assertNotIn("display(", source)


class EvaluateWiringTests(_ParityAssertions):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import optimizer_core as oc
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    f"cell-16 wiring needs runtime dependency '{dependency}'"
                ) from exc
            raise

        cls.oc = oc
        cls.model = oc.load_model()
        cls.baseline_cell = _baseline_cell(oc, cls.model)
        cls.original_cell_source = cls.model.code[cls.baseline_cell]

    def tearDown(self) -> None:
        self.model.code[self.baseline_cell] = self.original_cell_source

    def test_wired_path_calls_module_and_skips_cell16_execution(self) -> None:
        x = self.oc.warm_start(CFGS[0])
        calls: list[int] = []
        original = self.oc.calculate_baseline_turbine

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        self.oc.calculate_baseline_turbine = counted
        try:
            # Keep every detection marker but make execution fail loudly.
            self.model.code[self.baseline_cell] = (
                "def турбина(n, D_т, π_ст_):\n"
                "    raise AssertionError('cell 16 must not be executed')\n"
                "def распределить_π(n, D_т, π0):\n"
                "    raise AssertionError('cell 16 must not be executed')\n"
                "def решить(π0):\n"
                "    raise AssertionError('cell 16 must not be executed')\n"
                "def N_на_валу(G, π0):\n"
                "    raise AssertionError('cell 16 must not be executed')\n"
                "def _N_of(G):\n"
                "    raise AssertionError('cell 16 must not be executed')\n"
            )
            result = self.oc._evaluate(
                x, self.model, CFGS[0], stage_core=True
            )
        finally:
            self.oc.calculate_baseline_turbine = original
            self.model.code[self.baseline_cell] = self.original_cell_source

        self.assertGreater(len(calls), 0, "wired path did not call the module")
        self.assertTrue(math.isfinite(result["N_val"]))
        self.assertTrue(math.isfinite(result["eta"]))
        self.assertNotIn("табл_итер", result)

    def test_legacy_path_still_executes_cell16(self) -> None:
        x = self.oc.warm_start(CFGS[0])
        self.model.code[self.baseline_cell] = (
            "raise AssertionError('cell 16 executed')"
        )
        with self.assertRaisesRegex(AssertionError, "cell 16 executed"):
            self.oc._evaluate(
                x, self.model, CFGS[0], stage_core=False
            )

    def test_missing_cell16_fails_loudly(self) -> None:
        x = self.oc.warm_start(CFGS[0])
        original_idx = self.model.core_idx
        self.model.core_idx = [
            i for i in original_idx if i != self.baseline_cell
        ]
        try:
            with self.assertRaisesRegex(RuntimeError, "baseline turbine cell"):
                self.oc._evaluate(
                    x, self.model, CFGS[0], stage_core=True
                )
        finally:
            self.model.core_idx = original_idx

    def test_wired_and_legacy_evaluate_results_stay_identical(self) -> None:
        for cfg in CFGS:
            x = self.oc.warm_start(cfg)
            with self.subTest(cfg=cfg):
                wired = self.oc._evaluate(
                    x, self.model, cfg, stage_core=True
                )
                legacy = self.oc._evaluate(
                    x, self.model, cfg, stage_core=False
                )
                self.assertEqual(set(wired), set(legacy), "result keys changed")
                self.assertEqual(wired["p"], legacy["p"])
                for field in (
                    "N_val", "eta", "loss", "sigma_max", "lc1", "lc", "ll",
                    "a2", "pi", "sigma", "D_t", "u_t", "n_t",
                ):
                    with self.subTest(cfg=cfg, field=field):
                        self.assertValueMatches(
                            wired[field], legacy[field], field
                        )
                self.assertStageDictsEqual(wired["stages"], legacy["stages"])

    def test_full_wired_and_legacy_results_stay_identical(self) -> None:
        cfg = CFGS[0]
        x = self.oc.warm_start(cfg)
        wired = self.oc._evaluate(
            x, self.model, cfg, stage_core=True, full=True
        )
        legacy = self.oc._evaluate(
            x, self.model, cfg, stage_core=False, full=True
        )
        self.assertEqual(set(wired), set(legacy), "full result keys changed")
        for field in (
            "gamma_max", "rho_root", "lambda_c1", "alpha1_root",
            "phi_ref", "psi_ref",
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
        self.assertStageDictsEqual(wired["stages"], legacy["stages"])


if __name__ == "__main__":
    unittest.main()
