"""Parity and wiring tests for the const-hub orchestration extraction.

PROVENANCE / STATUS: software-drift diagnostic only.

``Optimizer/optimize_extended.build_ns`` still executes the notebook cells.
With the default ``orchestration_core=True`` it replaces ``газ``/``H0_по_p``,
``ступень`` and ``ρ_корня`` immediately after their source cells with the
adapters from ``optimizer_core``, then replaces exactly the three functions
``ступень_const_hub``, ``турбина_const_hub`` and ``распределить_π_const_hub``
with :mod:`turbine_engine.orchestration`.  The cell ``const-hub-scan`` is still
executed so its ``ρ_таблица`` and legacy definitions remain available.
``orchestration_core=False`` keeps every notebook function unchanged as the
independent reference.

These tests use one fixed candidate (the existing anchor ``N = 84e3``,
``D_hub = 0.065 m`` and the model's own initial pressure ratio).  They do not
run a scan, sweep, warm start, or optimizer search, and the candidate is not a
validated physical baseline or an acceptable design.
"""

from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
import types
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
from turbine_engine.orchestration import (  # noqa: E402
    ConstHubOrchestration,
    build_const_hub_orchestration,
    clip_scalar,
    linear_interp,
)


class HelperTests(unittest.TestCase):
    def test_default_interpolant_matches_source_fast_path(self) -> None:
        xp = [0.1111111111111111, 0.25, 1.0]
        fp = [0.2, 0.4, 0.5]
        self.assertEqual(linear_interp(0.05, xp, fp), fp[0])
        self.assertEqual(linear_interp(2.0, xp, fp), fp[-1])
        expected = (fp[1] - fp[0]) / (xp[1] - xp[0]) * (0.2 - xp[0]) + fp[0]
        self.assertEqual(linear_interp(0.2, xp, fp), expected)

    def test_default_clip_matches_scalar_numpy_semantics(self) -> None:
        self.assertEqual(clip_scalar(1.0, 1.05, 2.4), 1.05)
        self.assertEqual(clip_scalar(3.0, 1.05, 2.4), 2.4)
        self.assertEqual(clip_scalar(2.0, 1.05, 2.4), 2.0)
        self.assertTrue(math.isnan(clip_scalar(float("nan"), 1.05, 2.4)))

    def test_factory_reports_missing_dependencies(self) -> None:
        with self.assertRaises(KeyError):
            build_const_hub_orchestration({})
        orchestration = build_const_hub_orchestration({}, validate=False)
        self.assertIn("ступень", orchestration.missing_namespace_keys())

    def test_standalone_import_does_not_load_notebook_or_numpy(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.orchestration import "
            "build_const_hub_orchestration, clip_scalar, linear_interp\n"
            "assert 'numpy' not in sys.modules, sorted(sys.modules)\n"
            "assert 'scipy' not in sys.modules, sorted(sys.modules)\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "assert linear_interp(0.5, [0.0, 1.0], [1.0, 3.0]) == 2.0\n"
            "assert clip_scalar(0.0, 1.0, 2.0) == 1.0\n"
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


class ConstHubParityTests(unittest.TestCase):
    RELATIVE_TOLERANCE = 1e-12
    CANDIDATE_N = 84000.0
    CANDIDATE_D_HUB = 0.065
    Q_N_LIMIT = 0.995
    STAGE_FIELD_COUNT = 70

    @classmethod
    def setUpClass(cls) -> None:
        # Importing the legacy model requires the declared runtime dependencies.
        # Skip only when one is genuinely unavailable; all other failures must
        # surface instead of being hidden as a skip.
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "const-hub parity needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        cls.core_ns, _ = oe.build_ns(
            cls.model, q_n_limit=cls.Q_N_LIMIT, orchestration_core=True
        )
        cls.legacy_ns, _ = oe.build_ns(
            cls.model, q_n_limit=cls.Q_N_LIMIT, orchestration_core=False
        )

        cls.pi0 = cls.legacy_ns["π_т"] ** (1.0 / cls.legacy_ns["z"])
        (
            cls.legacy_scale_pi,
            cls.legacy_stages,
        ) = cls.legacy_ns["распределить_π_const_hub"](
            cls.CANDIDATE_N, cls.CANDIDATE_D_HUB, cls.pi0
        )
        cls.core_scale_pi, cls.core_stages = cls.core_ns[
            "распределить_π_const_hub"
        ](cls.CANDIDATE_N, cls.CANDIDATE_D_HUB, cls.pi0)

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

    def assertStageListsEqual(self, actual, expected, context: str) -> None:
        self.assertEqual(len(actual), len(expected), f"{context}: stage count")
        for index, (actual_stage, expected_stage) in enumerate(
            zip(actual, expected), start=1
        ):
            self.assertEqual(
                set(actual_stage),
                set(expected_stage),
                f"{context}: stage {index} keys changed",
            )
            self.assertEqual(
                len(expected_stage),
                self.STAGE_FIELD_COUNT,
                f"{context}: stage {index} field count",
            )
            for field, expected_value in expected_stage.items():
                with self.subTest(context=context, stage=index, field=field):
                    self.assertFiniteNear(
                        float(actual_stage[field]),
                        float(expected_value),
                        f"{context}: stage {index} field {field}",
                    )

    def test_default_build_installs_extracted_functions(self) -> None:
        for name in (
            "ступень_const_hub",
            "турбина_const_hub",
            "распределить_π_const_hub",
        ):
            with self.subTest(name=name):
                function = self.core_ns[name]
                self.assertTrue(inspect.ismethod(function))
                self.assertIsInstance(function.__self__, ConstHubOrchestration)
                self.assertEqual(
                    function.__module__, "turbine_engine.orchestration"
                )

    def test_legacy_build_keeps_notebook_functions(self) -> None:
        for name in (
            "ступень_const_hub",
            "турбина_const_hub",
            "распределить_π_const_hub",
        ):
            with self.subTest(name=name):
                function = self.legacy_ns[name]
                self.assertTrue(inspect.isfunction(function))
                self.assertEqual(function.__code__.co_filename, "const_hub")
                self.assertFalse(hasattr(function, "__self__"))

    def test_default_build_installs_extracted_physical_helpers(self) -> None:
        adapters = {
            "газ": "_газ_ядро",
            "H0_по_p": "_h0_по_p_ядро",
            "ступень": "_ступень_ядро",
            "ρ_корня": "_ρ_корня_ядро",
        }
        for name, adapter in adapters.items():
            with self.subTest(name=name):
                function = inspect.unwrap(self.core_ns[name])
                self.assertEqual(function.__module__, "optimizer_core")
                self.assertIn(adapter, function.__qualname__)

    def test_legacy_build_keeps_notebook_physical_helpers(self) -> None:
        filenames = {
            "газ": "ядро 7",
            "H0_по_p": "ядро 7",
            "ступень": "ядро 12",
            "ρ_корня": "ядро 14",
        }
        for name, filename in filenames.items():
            with self.subTest(name=name):
                function = inspect.unwrap(self.legacy_ns[name])
                self.assertIsInstance(function, types.FunctionType)
                self.assertEqual(function.__code__.co_filename, filename)
                self.assertNotEqual(function.__module__, "optimizer_core")

    def test_default_build_wires_helpers_immediately_after_source_cells(self) -> None:
        helper_names = ("газ", "H0_по_p", "ступень", "ρ_корня")
        adapters = (
            ("газ", "_газ_ядро"),
            ("H0_по_p", "_h0_по_p_ядро"),
            ("ступень", "_ступень_ядро"),
            ("ρ_корня", "_ρ_корня_ядро"),
        )
        calls: list[str] = []
        snapshots: list[dict[str, object]] = []
        originals = {
            adapter: getattr(self.oc, adapter) for _, adapter in adapters
        }

        def make_wrapper(adapter: str):
            original = originals[adapter]

            def wrapper(namespace):
                snapshots.append({
                    name: (
                        inspect.unwrap(namespace[name])
                        if name in namespace
                        else None
                    )
                    for name in helper_names
                })
                calls.append(adapter)
                return original(namespace)

            return wrapper

        build_calls = {
            "calculate_stage": 0,
            "calc_h0_for_pressure": 0,
            "calc_gas_cp_k": 0,
        }
        build_originals = {name: getattr(self.oc, name) for name in build_calls}

        def counting(name: str):
            original = build_originals[name]

            def wrapper(*args, **kwargs):
                build_calls[name] += 1
                return original(*args, **kwargs)

            return wrapper

        for _, adapter in adapters:
            setattr(self.oc, adapter, make_wrapper(adapter))
        for name in build_calls:
            setattr(self.oc, name, counting(name))
        try:
            ns, _ = self.oe.build_ns(
                self.model, q_n_limit=self.Q_N_LIMIT, orchestration_core=True
            )
        finally:
            for adapter, original in originals.items():
                setattr(self.oc, adapter, original)
            for name, original in build_originals.items():
                setattr(self.oc, name, original)

        self.assertEqual(
            calls,
            ["_газ_ядро", "_h0_по_p_ядро", "_ступень_ядро", "_ρ_корня_ядро"],
            "physical adapters were not installed in source order",
        )
        self.assertEqual(len(snapshots), len(adapters))
        self.assertTrue(
            all(
                inspect.unwrap(ns[name]).__module__ == "optimizer_core"
                for name in helper_names
            )
        )
        # Cells 16 and 20 run after the replacements, so the build itself must
        # already have called the extracted stage, H0 and gas cores.
        for name, count in build_calls.items():
            with self.subTest(build_call=name):
                self.assertGreater(
                    count, 0,
                    f"later notebook cells did not observe {name} replacement",
                )

        # Each factory sees exactly the preceding replacements and still sees
        # the notebook definition it is about to replace.  Sources that have
        # not executed yet are simply absent.
        gas_seen, h0_seen, stage_seen, reaction_seen = snapshots
        self.assertNotEqual(gas_seen["газ"].__module__, "optimizer_core")
        self.assertNotEqual(gas_seen["H0_по_p"].__module__, "optimizer_core")
        # газ and H0_по_p are installed as one unit, so the H0 factory already
        # sees the extracted газ while its own notebook definition is current.
        self.assertEqual(h0_seen["газ"].__module__, "optimizer_core")
        self.assertNotEqual(h0_seen["H0_по_p"].__module__, "optimizer_core")
        for seen in (gas_seen, h0_seen):
            self.assertIsNone(seen["ступень"])
            self.assertIsNone(seen["ρ_корня"])
        for name in ("газ", "H0_по_p"):
            self.assertEqual(stage_seen[name].__module__, "optimizer_core")
            self.assertEqual(reaction_seen[name].__module__, "optimizer_core")
        self.assertNotEqual(stage_seen["ступень"].__module__, "optimizer_core")
        self.assertIsNone(stage_seen["ρ_корня"])
        self.assertNotEqual(reaction_seen["ρ_корня"].__module__, "optimizer_core")
        self.assertEqual(
            reaction_seen["ступень"].__module__, "optimizer_core"
        )

    def test_default_fixed_point_routes_physical_helpers_to_core(self) -> None:
        calls = {
            "calculate_stage": 0,
            "calc_stage_cp_k": 0,
            "calc_h0_for_pressure": 0,
            "calc_gas_cp_k": 0,
            "calculate_root_reaction": 0,
        }
        originals = {name: getattr(self.oc, name) for name in calls}

        def counting(name):
            original = originals[name]

            def wrapper(*args, **kwargs):
                calls[name] += 1
                return original(*args, **kwargs)

            return wrapper

        gas_temperature = 1234.567
        try:
            for name in calls:
                setattr(self.oc, name, counting(name))
            # setUpClass already evaluated the same fixed point; clear the
            # fast-cache wrappers so this test observes the actual calls.
            for name in ("газ", "H0_по_p"):
                cached = self.core_ns[name]
                if hasattr(cached, "cache_clear"):
                    cached.cache_clear()
            _, _ = self.core_ns["распределить_π_const_hub"](
                self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.pi0
            )
            core_gas = self.core_ns["газ"](gas_temperature)
            core_root = self.core_ns["ρ_корня"](
                self.core_ns["ступени"][0], "Γ"
            )
            core_calls = dict(calls)

            for name in calls:
                calls[name] = 0
            _, _ = self.legacy_ns["распределить_π_const_hub"](
                self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.pi0
            )
            legacy_gas = self.legacy_ns["газ"](gas_temperature)
            legacy_root = self.legacy_ns["ρ_корня"](
                self.legacy_ns["ступени"][0], "Γ"
            )
            legacy_calls = dict(calls)
        finally:
            for name, original in originals.items():
                setattr(self.oc, name, original)

        for name in ("calculate_stage", "calc_stage_cp_k", "calc_h0_for_pressure",
                     "calc_gas_cp_k", "calculate_root_reaction"):
            with self.subTest(name=name):
                self.assertGreater(
                    core_calls[name], 0,
                    f"default build path did not use turbine_engine {name}",
                )
                self.assertEqual(
                    legacy_calls[name], 0,
                    f"legacy reference path used turbine_engine {name}",
                )

        for index, (actual, expected) in enumerate(
            zip(core_gas, legacy_gas)
        ):
            self.assertFiniteNear(actual, expected, f"fixed-point газ[{index}]")
        self.assertFiniteNear(core_root, legacy_root, "fixed-point ρ_корня")

    def test_stage_const_hub_matches_notebook_at_fixed_point(self) -> None:
        namespace = self.legacy_ns
        weights = namespace["π_веса"]
        H0 = namespace["H0_по_p"](
            namespace["T_г_полн"],
            namespace["p_г_полн"],
            namespace["p_г_полн"]
            / self.legacy_scale_pi ** (weights[0] / weights[-1]),
        )
        args = (
            namespace["T_г_полн"],
            namespace["p_г_полн"],
            H0,
            namespace["G_c1"],
            namespace["G_c1"],
            self.CANDIDATE_N,
            namespace["α1_ст"][0],
            namespace["κ_ст"][0],
            self.CANDIDATE_D_HUB,
            self.CANDIDATE_D_HUB + 0.02,
        )
        expected = namespace["ступень_const_hub"](*args)
        actual = self.core_ns["ступень_const_hub"](*args)
        self.assertStageListsEqual([actual], [expected], "ступень_const_hub")

    def test_turbine_const_hub_matches_notebook_at_fixed_point(self) -> None:
        expected = self.legacy_ns["турбина_const_hub"](
            self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.legacy_scale_pi
        )
        actual = self.core_ns["турбина_const_hub"](
            self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.legacy_scale_pi
        )
        self.assertStageListsEqual(actual, expected, "турбина_const_hub")

    def test_pressure_distribution_matches_notebook_at_fixed_point(self) -> None:
        self.assertFiniteNear(
            self.core_scale_pi,
            self.legacy_scale_pi,
            "распределить_π_const_hub: scale_pi",
        )
        self.assertStageListsEqual(
            self.core_stages, self.legacy_stages, "распределить_π_const_hub"
        )

    def test_core_without_fast_cache_matches_legacy_numpy_path(self) -> None:
        core_ns, _ = self.oe.build_ns(
            self.model,
            q_n_limit=self.Q_N_LIMIT,
            fast=False,
            orchestration_core=True,
        )
        legacy_ns, _ = self.oe.build_ns(
            self.model,
            q_n_limit=self.Q_N_LIMIT,
            fast=False,
            orchestration_core=False,
        )
        pi0 = legacy_ns["π_т"] ** (1.0 / legacy_ns["z"])
        legacy_scale, legacy_stages = legacy_ns["распределить_π_const_hub"](
            self.CANDIDATE_N, self.CANDIDATE_D_HUB, pi0
        )
        core_scale, core_stages = core_ns["распределить_π_const_hub"](
            self.CANDIDATE_N, self.CANDIDATE_D_HUB, pi0
        )
        self.assertFiniteNear(core_scale, legacy_scale, "fast=False: scale_pi")
        self.assertStageListsEqual(
            core_stages, legacy_stages, "fast=False distribution"
        )

    def test_core_reads_live_stage_and_h0_bindings(self) -> None:
        stage_calls: list[int] = []
        h0_calls: list[int] = []
        original_stage = self.core_ns["ступень"]
        original_h0 = self.core_ns["H0_по_p"]

        def counted_stage(*args, **kwargs):
            stage_calls.append(1)
            return original_stage(*args, **kwargs)

        def counted_h0(*args, **kwargs):
            h0_calls.append(1)
            return original_h0(*args, **kwargs)

        self.core_ns["ступень"] = counted_stage
        self.core_ns["H0_по_p"] = counted_h0
        try:
            namespace = self.legacy_ns
            weights = namespace["π_веса"]
            H0 = original_h0(
                namespace["T_г_полн"],
                namespace["p_г_полн"],
                namespace["p_г_полн"]
                / self.legacy_scale_pi ** (weights[0] / weights[-1]),
            )
            self.core_ns["ступень_const_hub"](
                namespace["T_г_полн"],
                namespace["p_г_полн"],
                H0,
                namespace["G_c1"],
                namespace["G_c1"],
                self.CANDIDATE_N,
                namespace["α1_ст"][0],
                namespace["κ_ст"][0],
                self.CANDIDATE_D_HUB,
                self.CANDIDATE_D_HUB + 0.02,
            )
            self.assertGreater(len(stage_calls), 0, "ступень was resolved at build time")

            stage_calls.clear()
            self.core_ns["турбина_const_hub"](
                self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.legacy_scale_pi
            )
            self.assertGreater(len(h0_calls), 0, "H0_по_p was resolved at build time")
            self.assertGreater(
                len(stage_calls), 0, "stage binding was not live"
            )
        finally:
            self.core_ns["ступень"] = original_stage
            self.core_ns["H0_по_p"] = original_h0

    def test_core_reads_live_sibling_functions_after_install(self) -> None:
        original_turbine = self.core_ns["турбина_const_hub"]
        turbine_calls: list[int] = []

        def counted_turbine(*args, **kwargs):
            turbine_calls.append(1)
            return original_turbine(*args, **kwargs)

        self.core_ns["турбина_const_hub"] = counted_turbine
        try:
            _, stages = self.core_ns["распределить_π_const_hub"](
                self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.pi0
            )
        finally:
            self.core_ns["турбина_const_hub"] = original_turbine
        self.assertGreater(
            len(turbine_calls), 0, "distribution cached its turbine sibling"
        )
        self.assertStageListsEqual(
            stages, self.legacy_stages, "live sibling турбина_const_hub"
        )

        original_stage = self.core_ns["ступень_const_hub"]
        stage_calls: list[int] = []

        def counted_stage(*args, **kwargs):
            stage_calls.append(1)
            return original_stage(*args, **kwargs)

        self.core_ns["ступень_const_hub"] = counted_stage
        try:
            stages = self.core_ns["турбина_const_hub"](
                self.CANDIDATE_N, self.CANDIDATE_D_HUB, self.legacy_scale_pi
            )
        finally:
            self.core_ns["ступень_const_hub"] = original_stage
        self.assertGreater(
            len(stage_calls), 0, "turbine cached its stage sibling"
        )
        self.assertStageListsEqual(
            stages, self.legacy_stages, "live sibling ступень_const_hub"
        )

    def test_default_eval_candidate_uses_extracted_distribution(self) -> None:
        x = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)
        scenario = "B87_relaxed"
        core_ns, _ = self.oe.build_ns(
            self.model, q_n_limit=self.Q_N_LIMIT, orchestration_core=True
        )
        legacy_ns, _ = self.oe.build_ns(
            self.model, q_n_limit=self.Q_N_LIMIT, orchestration_core=False
        )
        distribution_calls: list[int] = []
        core_distribution = core_ns["распределить_π_const_hub"]

        def counted_distribution(*args, **kwargs):
            distribution_calls.append(1)
            return core_distribution(*args, **kwargs)

        core_ns["распределить_π_const_hub"] = counted_distribution
        actual = self.oe.eval_candidate(
            x, core_ns, scenario, self.Q_N_LIMIT
        )
        expected = self.oe.eval_candidate(
            x, legacy_ns, scenario, self.Q_N_LIMIT
        )

        self.assertGreater(
            len(distribution_calls),
            0,
            "eval_candidate did not use extracted orchestration",
        )
        self.assertEqual(actual["status"], "ok", actual.get("err", ""))
        self.assertEqual(actual["status"], expected["status"])
        self.assertEqual(actual["feasible"], expected["feasible"])
        self.assertEqual(actual["eta_ok"], expected["eta_ok"])
        self.assertEqual(actual["violation_count"], expected["violation_count"])
        self.assertEqual(
            [violation["name"] for violation in actual["violations"]],
            [violation["name"] for violation in expected["violations"]],
        )
        for field in (
            "lc1", "a2", "sigma", "u", "lD", "qn", "N", "cont", "energy",
            "p2z", "eta_t", "hard", "rho", "rho_root", "uc0", "uc0_opt",
        ):
            with self.subTest(field=field):
                actual_value = actual[field]
                expected_value = expected[field]
                if isinstance(expected_value, (list, tuple)):
                    self.assertEqual(len(actual_value), len(expected_value))
                    for index, (a, e) in enumerate(zip(actual_value, expected_value)):
                        self.assertFiniteNear(
                            float(a), float(e), f"{field}[{index}]"
                        )
                else:
                    self.assertFiniteNear(
                        float(actual_value), float(expected_value), field
                    )
        self.assertStageListsEqual(
            core_ns["ступени"], legacy_ns["ступени"], "eval_candidate"
        )


class _НетРешения(Exception):
    pass


def _failure_namespace(**overrides):
    namespace = {
        "НетРешения": _НетРешения,
        "π_ст_макс": 5.0,
        "ρ_таблица": ([0.0, 1.0], [0.3, 0.3]),
    }
    namespace.update(overrides)
    return namespace


class ExplicitDiagnosticsTests(unittest.TestCase):
    """Silent secant/fixed-point exhaustion must become typed diagnostics."""

    def test_initial_invalid_evaluation_raises_namespace_exception(self) -> None:
        orchestration = ConstHubOrchestration(_failure_namespace())

        def failing_turbine(*_args, **_kwargs):
            raise _НетРешения("no stage")

        orchestration.турбина_const_hub = failing_turbine  # type: ignore[method-assign]
        with self.assertRaises(_НетРешения) as raised:
            orchestration.распределить_π_const_hub(84000.0, 0.065, 4.0)
        self.assertIn("начальной π", str(raised.exception))

    def test_final_invalid_evaluation_raises_typed_error(self) -> None:
        orchestration = ConstHubOrchestration(_failure_namespace())
        calls: list[int] = []

        def stage_only_once(_n, _D_hub, y):
            calls.append(1)
            if len(calls) > 1:
                raise _НетРешения("stage failed after the initial evaluation")
            return [{"π_ст": y}]

        orchestration.турбина_const_hub = stage_only_once  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.распределить_π_const_hub(84000.0, 0.065, 4.0)

        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(
            error.stage, "orchestration.распределить_π_const_hub"
        )
        self.assertEqual(error.reason, "no_valid_final_evaluation")
        self.assertEqual(error.last_valid_state["iteration_limit"], 30)
        self.assertTrue(math.isfinite(error.last_valid_state["residual"]))

    def test_stalled_secant_is_not_treated_as_convergence(self) -> None:
        orchestration = ConstHubOrchestration(_failure_namespace())

        def flat_residual(_n, _D_hub, y):
            return [{"π_ст": y + 0.1}]

        orchestration.турбина_const_hub = flat_residual  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.распределить_π_const_hub(84000.0, 0.065, 4.0)

        error = raised.exception
        self.assertEqual(error.reason, "secant_denominator_stalled")
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertAlmostEqual(
            error.last_valid_state["residual"], 0.1, places=12
        )
        self.assertGreater(
            abs(error.last_valid_state["residual"]), 1e-5
        )

    def test_small_residual_still_converges_before_denominator_test(self) -> None:
        orchestration = ConstHubOrchestration(_failure_namespace())

        def almost_closed(_n, _D_hub, y):
            return [{"π_ст": y + 5e-6}]

        orchestration.турбина_const_hub = almost_closed  # type: ignore[method-assign]
        y, stages = orchestration.распределить_π_const_hub(
            84000.0, 0.065, 4.0
        )
        self.assertGreater(y, 0.0)
        self.assertEqual(len(stages), 1)

    def test_secant_iteration_limit_raises_typed_error(self) -> None:
        orchestration = ConstHubOrchestration(_failure_namespace())
        calls: list[int] = []

        def slowly_diverging(_n, _D_hub, y):
            calls.append(1)
            return [{"π_ст": y + 0.2 + 0.01 * len(calls)}]

        orchestration.турбина_const_hub = slowly_diverging  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.распределить_π_const_hub(84000.0, 0.065, 4.0)

        error = raised.exception
        self.assertEqual(error.reason, "pi_secant_iteration_exhausted")
        self.assertEqual(error.last_valid_state["iteration_limit"], 30)
        self.assertGreater(error.last_valid_state["residual"], 1e-5)

    def test_stage_reaction_fixed_point_exhaustion_raises(self) -> None:
        calls: list[int] = []

        def never_settles(T0, p0, H0, rho, D_c, D_l, n, G_c, G_l, a1, k):
            calls.append(1)
            scale = 1.0 + 0.5 * len(calls)
            return {
                "v1": scale,
                "c1a": 1.0,
                "v2": scale,
                "c2a": 1.0,
            }

        namespace = _failure_namespace(ступень=never_settles)
        orchestration = ConstHubOrchestration(namespace)
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.ступень_const_hub(
                1250.0,
                8.0e5,
                1.0e5,
                1.0,
                1.0,
                84000.0,
                15.0,
                1.1,
                0.065,
                0.085,
            )

        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "orchestration.ступень_const_hub")
        self.assertEqual(error.reason, "reaction_fixed_point_exhausted")
        state = error.last_valid_state
        self.assertEqual(state["iteration_limit"], 60)
        self.assertEqual(state["tolerance"], 1e-7)
        self.assertEqual(len(calls), 60)

        # Reconstruct the final evaluated state, its original residuals and the
        # proposed half-relaxed next iterate from the same test stage law.
        D_c = D_l = 0.085
        rho = 0.3
        for index in range(1, 61):
            scale = 1.0 + 0.5 * index
            l = (
                math.sqrt(0.065**2 + 4.0 * scale / math.pi) - 0.065
            ) / 2
            target = 0.065 + l
            if index == 60:
                evaluated = (D_c, D_l, rho)
                targets = (target, target, 0.3)
            D_c = 0.5 * (D_c + target)
            D_l = 0.5 * (D_l + target)
            rho = 0.5 * (rho + 0.3)

        self.assertTrue(
            math.isclose(state["D_c"], evaluated[0], rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(state["D_l"], evaluated[1], rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(state["rho"], evaluated[2], rel_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(
                state["D_c_delta"], targets[0] - evaluated[0], rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                state["D_l_delta"], targets[1] - evaluated[1], rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                state["rho_delta"], targets[2] - evaluated[2], rel_tol=1e-12
            )
        )
        self.assertTrue(
            math.isclose(
                state["D_c_next"],
                0.5 * (evaluated[0] + targets[0]),
                rel_tol=1e-12,
            )
        )
        self.assertTrue(
            math.isclose(
                state["D_c_delta"],
                2.0 * (state["D_c_next"] - state["D_c"]),
                rel_tol=1e-12,
            )
        )
        json.dumps(error.as_record(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
