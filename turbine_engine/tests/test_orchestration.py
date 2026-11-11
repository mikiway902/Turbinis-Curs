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

import copy
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
    PRESSURE_RATIO_LEGACY_CLIPPED,
    PRESSURE_RATIO_RAW_UNBOUNDED,
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

    def test_pressure_ratio_policy_validation_and_default(self) -> None:
        default = build_const_hub_orchestration({}, validate=False)
        self.assertEqual(
            default.pressure_ratio_policy, PRESSURE_RATIO_LEGACY_CLIPPED
        )
        raw = build_const_hub_orchestration(
            {}, validate=False, pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED
        )
        self.assertEqual(raw.pressure_ratio_policy, PRESSURE_RATIO_RAW_UNBOUNDED)
        self.assertNotIn("π_ст_макс", raw.missing_namespace_keys())
        self.assertIn("π_ст_макс", default.missing_namespace_keys())
        with self.assertRaises(ValueError):
            build_const_hub_orchestration(
                {}, validate=False, pressure_ratio_policy="engineering_guess"
            )
        with self.assertRaises(ValueError):
            ConstHubOrchestration({}, pressure_ratio_policy="engineering_guess")

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

    def test_nonfinite_initial_residual_raises_before_second_evaluation(
        self,
    ) -> None:
        for policy in (
            PRESSURE_RATIO_LEGACY_CLIPPED,
            PRESSURE_RATIO_RAW_UNBOUNDED,
        ):
            for bad in (math.nan, math.inf, -math.inf):
                with self.subTest(policy=policy, value=bad):
                    calls: list[float] = []

                    def first_bad(_n, _D_hub, y):
                        calls.append(y)
                        if len(calls) == 1:
                            return [{"π_ст": bad + y}]
                        return [{"π_ст": y}]

                    orchestration = ConstHubOrchestration(
                        _failure_namespace(), pressure_ratio_policy=policy
                    )
                    orchestration.турбина_const_hub = first_bad  # type: ignore[method-assign]
                    with self.assertRaises(NumericalConvergenceError) as raised:
                        orchestration.распределить_π_const_hub(
                            84000.0, 0.065, 4.0
                        )

                    error = raised.exception
                    self.assertEqual(
                        error.reason, "non_finite_initial_residual"
                    )
                    self.assertEqual(
                        error.stage,
                        "orchestration.распределить_π_const_hub",
                    )
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(error.last_valid_state["input_y"], 4.0)
                    self.assertIsNone(
                        error.last_valid_state["initial_residual"]
                    )
                    self.assertNotIn("y", error.last_valid_state)
                    if policy == PRESSURE_RATIO_RAW_UNBOUNDED:
                        self.assertEqual(
                            error.last_valid_state["pressure_ratio_policy"],
                            policy,
                        )

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


class PressureRatioPolicyTests(unittest.TestCase):
    """Raw policy removes only the engineering clip, not numerical checks."""

    N = 84000.0
    D_HUB = 0.065

    @staticmethod
    def _constant_root(root):
        def turbine(_n, _D_hub, _y):
            return [{"π_ст": root}]

        return turbine

    def test_legacy_clip_stalls_where_raw_reaches_root_five(self) -> None:
        legacy = ConstHubOrchestration(_failure_namespace(π_ст_макс=2.0))
        legacy.турбина_const_hub = self._constant_root(5.0)  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            legacy.распределить_π_const_hub(self.N, self.D_HUB, 4.0)
        error = raised.exception
        self.assertEqual(error.reason, "secant_denominator_stalled")
        self.assertNotIn("pressure_ratio_policy", error.last_valid_state)

        raw = ConstHubOrchestration(
            _failure_namespace(π_ст_макс=2.0),
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        raw.турбина_const_hub = self._constant_root(5.0)  # type: ignore[method-assign]
        y, stages = raw.распределить_π_const_hub(self.N, self.D_HUB, 4.0)
        self.assertEqual(y, 5.0)
        self.assertEqual(len(stages), 1)

    def test_raw_never_calls_clip_or_reads_pi_max(self) -> None:
        class Guarded(dict):
            def __getitem__(self, key):
                if key == "π_ст_макс":
                    raise AssertionError("raw policy read π_ст_макс")
                return super().__getitem__(key)

        def forbidden(*_args, **_kwargs):
            raise AssertionError("raw policy called the clip function")

        orchestration = ConstHubOrchestration(
            Guarded(_failure_namespace()),
            clip=forbidden,
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        orchestration.турбина_const_hub = self._constant_root(5.0)  # type: ignore[method-assign]
        y, stages = orchestration.распределить_π_const_hub(
            self.N, self.D_HUB, 4.0
        )
        self.assertEqual(y, 5.0)
        self.assertEqual(len(stages), 1)

    def test_raw_safeguards_proposals_at_or_below_domain(self) -> None:
        orchestration = ConstHubOrchestration(
            _failure_namespace(),
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        seen: list[float] = []

        def low_root(_n, _D_hub, y):
            seen.append(y)
            return [{"π_ст": 0.5}]

        orchestration.турбина_const_hub = low_root  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.распределить_π_const_hub(self.N, self.D_HUB, 4.0)

        self.assertTrue(all(value > 1.0 for value in seen))
        self.assertAlmostEqual(seen[2], 0.5 * (4.08 + 1.0), places=12)
        error = raised.exception
        self.assertEqual(error.reason, "pi_secant_iteration_exhausted")
        self.assertEqual(
            error.last_valid_state["pressure_ratio_policy"],
            PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        self.assertGreater(error.last_valid_state["y"], 1.0)

    def test_raw_nonfinite_proposal_and_lost_domain_are_typed(self) -> None:
        orchestration = ConstHubOrchestration(
            _failure_namespace(),
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration._raw_pi_candidate(
                float("nan"), {"y": 2.0, "residual": 0.5}
            )
        error = raised.exception
        self.assertEqual(error.reason, "non_finite_proposed_pressure_ratio")
        self.assertEqual(
            error.last_valid_state["pressure_ratio_policy"],
            PRESSURE_RATIO_RAW_UNBOUNDED,
        )

        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration._raw_pi_candidate(
                0.5, {"y": 0.5, "residual": 1.0}
            )
        self.assertEqual(raised.exception.reason, "pi_expansion_domain_lost")

        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.распределить_π_const_hub(self.N, self.D_HUB, 0.9)
        self.assertEqual(raised.exception.reason, "pi_expansion_domain_lost")
        self.assertEqual(
            raised.exception.last_valid_state["pressure_ratio_policy"],
            PRESSURE_RATIO_RAW_UNBOUNDED,
        )


def _closure_stage_payloads(
    gas_power: float, total_pressure: float, z: int
) -> list[dict[str, float]]:
    """Minimal actual-stage payload with the metric and reporting keys."""

    per_stage_work = gas_power / (10.0 * z)
    stages = []
    for _ in range(z):
        stages.append({
            "G_c": 10.0,
            "L_т": per_stage_work,
            "p_ст_п": total_pressure,
            "T_ст_п": 900.0,
            "p2": total_pressure / 1.02,
            "T_ст": 950.0,
            "π_ст": 1.5,
            "ρ": 0.3,
            "H0": 100_000.0,
            "u1": 300.0,
            "u2": 300.0,
            "c2": 150.0,
            "cp": 1100.0,
            "k": 1.33,
            "α1": 11.0,
            "β1": 20.0,
            "α2": 30.0,
            "β2": 40.0,
            "M_c1": 0.5,
            "M_w2": 0.6,
            "D_c": 0.1,
            "D_л": 0.1,
            "l_c": 0.02,
            "l_л": 0.02,
            "σ_р": 100.0,
        })
    return stages


class SyntheticStaticPressureClosureTests(unittest.TestCase):
    """Bracketed closures evaluated against a synthetic actual-stage solver."""

    INLET_TOTAL = 1_000_000.0
    ORIGINAL_P_T = 500_000.0

    def _namespace(self, *, z: int = 2) -> dict[str, object]:
        return {
            "НетРешения": _НетРешения,
            "p_г_полн": self.INLET_TOTAL,
            "z": z,
            "p_т": self.ORIGINAL_P_T,
        }

    def _make(self, *, total=None, gas=None, z: int = 2):
        namespace = self._namespace(z=z)
        orchestration = ConstHubOrchestration(
            namespace,
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        calls: list[float] = []

        def resolver(_n, _D_hub, _seed):
            p_static = namespace["p_т"]
            calls.append(p_static)
            total_p = total(p_static) if total is not None else 0.0
            work = gas(p_static) if gas is not None else 0.0
            return 1.0, _closure_stage_payloads(work, total_p, z)

        orchestration.распределить_π_const_hub = resolver  # type: ignore[method-assign]
        return orchestration, namespace, calls

    def test_increasing_total_pressure_fixture_closes(self) -> None:
        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        record = orchestration.solve_total_outlet_pressure(
            84_000.0,
            0.065,
            160_000.0,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        self.assertEqual(record["target_kind"], "total_outlet_pressure")
        self.assertEqual(record["target_value"], 160_000.0)
        self.assertLessEqual(record["relative_residual"], 1e-6)
        self.assertEqual(
            record["actual_metric"],
            record["actual_outlet_total_pressure_Pa"],
        )
        self.assertEqual(
            record["actual_metric"],
            record["actual_stages"][-1]["p_ст_п"],
        )
        self.assertAlmostEqual(
            record["solved_static_pressure_Pa"],
            (160_000.0 - 10_000.0) / 1.1,
            delta=1.0,
        )
        self.assertGreater(record["iterations"], 0)
        self.assertEqual(
            record["pressure_ratio_policy"], PRESSURE_RATIO_RAW_UNBOUNDED
        )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_decreasing_mechanical_power_fixture_closes(self) -> None:
        orchestration, namespace, _ = self._make(
            gas=lambda p: 3_000_000.0 - p
        )
        efficiency = 0.99
        target = efficiency * (3_000_000.0 - 125_000.0)
        record = orchestration.solve_prescribed_mechanical_power(
            84_000.0,
            0.065,
            target,
            efficiency,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        stages = record["actual_stages"]
        recomputed = efficiency * sum(
            stage["G_c"] * stage["L_т"] for stage in stages
        )
        self.assertEqual(record["target_kind"], "mechanical_power")
        self.assertEqual(record["mechanical_power_W"], record["actual_metric"])
        self.assertAlmostEqual(record["actual_metric"], recomputed, places=9)
        self.assertLessEqual(record["relative_residual"], 1e-6)
        # The stop test is the relative metric residual, not the pressure
        # width, so the pressure tolerance follows the fixture slope.
        pressure_tolerance = (
            1e-6 * abs(target) / efficiency + 1e-9
        )
        self.assertAlmostEqual(
            record["solved_static_pressure_Pa"],
            125_000.0,
            delta=pressure_tolerance,
        )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_endpoint_root_is_accepted_in_both_directions(self) -> None:
        orchestration, _, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        for boundary, target in (
            (120_000.0, 142_000.0),
            (150_000.0, 175_000.0),
        ):
            with self.subTest(boundary=boundary):
                record = orchestration.solve_total_outlet_pressure(
                    84_000.0,
                    0.065,
                    target,
                    static_pressure_bracket=(120_000.0, 150_000.0),
                )
                self.assertEqual(
                    record["solved_static_pressure_Pa"], boundary
                )
                self.assertEqual(record["iterations"], 0)
                self.assertEqual(record["actual_metric"], target)

    def test_not_bracketed_raises_typed_boundary_error(self) -> None:
        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                500_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        error = raised.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.reason, "boundary_not_bracketed")
        state = error.last_valid_state
        self.assertEqual(state["target"], 500_000.0)
        self.assertEqual(
            state["evaluated_static_pressure_Pa"], [120_000.0, 150_000.0]
        )
        self.assertEqual(
            state["evaluated_metrics"], [142_000.0, 175_000.0]
        )
        self.assertEqual(
            state["signed_residuals"], [-358_000.0, -325_000.0]
        )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_nonfinite_initial_and_trial_metrics_are_typed(self) -> None:
        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )

        def initial_nonfinite(_n, _D_hub, _seed):
            return 1.0, _closure_stage_payloads(1e6, math.nan, 2)

        orchestration.распределить_π_const_hub = initial_nonfinite  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertEqual(
            raised.exception.reason, "non_finite_initial_metric"
        )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

        def trial_nonfinite(_n, _D_hub, _seed):
            p_static = namespace["p_т"]
            total = (
                math.nan
                if p_static == 135_000.0
                else 1.1 * p_static + 10_000.0
            )
            return 1.0, _closure_stage_payloads(1e6, total, 2)

        orchestration.распределить_π_const_hub = trial_nonfinite  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        error = raised.exception
        self.assertEqual(error.reason, "non_finite_metric")
        self.assertEqual(
            error.last_valid_state["static_pressure_Pa"], 135_000.0
        )
        self.assertIsNone(error.last_valid_state["metric"])
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_stalled_adjacent_floats_and_iteration_limit_are_typed(self) -> None:
        namespace = {
            "НетРешения": _НетРешения,
            "p_г_полн": 10.0,
            "z": 1,
            "p_т": 1.0,
        }
        orchestration = ConstHubOrchestration(
            namespace,
            pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
        )
        lo, hi = 1.0, math.nextafter(1.0, 2.0)

        def discontinuous(_n, _D_hub, _seed):
            total = 1_000_000.0 if namespace["p_т"] == lo else 0.0
            return 1.0, _closure_stage_payloads(0.0, total, 1)

        orchestration.распределить_π_const_hub = discontinuous  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                500_000.0,
                static_pressure_bracket=(lo, hi),
            )
        error = raised.exception
        self.assertEqual(error.reason, "stalled_adjacent_floats")
        self.assertEqual(error.last_valid_state["iterations"], 1)
        self.assertEqual(
            error.last_valid_state["last_evaluated"][
                "static_pressure_Pa"
            ],
            hi,
        )
        self.assertEqual(namespace["p_т"], 1.0)

        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
                max_iter=1,
            )
        error = raised.exception
        self.assertEqual(error.reason, "iteration_limit")
        self.assertEqual(error.last_valid_state["iterations"], 1)
        last_evaluated = error.last_valid_state["last_evaluated"]
        self.assertEqual(
            last_evaluated["static_pressure_Pa"], 135_000.0
        )
        self.assertEqual(last_evaluated["metric"], 158_500.0)
        self.assertEqual(last_evaluated["residual"], -1_500.0)
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_configuration_validation_is_strict(self) -> None:
        namespace = self._namespace()
        legacy = ConstHubOrchestration(namespace)
        with self.assertRaises(ValueError):
            legacy.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        with self.assertRaises(ValueError):
            legacy.solve_prescribed_mechanical_power(
                84_000.0,
                0.065,
                2_800_000.0,
                0.99,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )

        def call_total(**overrides):
            kwargs = {
                "n": 84_000.0,
                "D_hub": 0.065,
                "target_total_pressure": 160_000.0,
                "static_pressure_bracket": (120_000.0, 150_000.0),
            }
            kwargs.update(overrides)
            return orchestration.solve_total_outlet_pressure(**kwargs)

        def call_power(**overrides):
            kwargs = {
                "n": 84_000.0,
                "D_hub": 0.065,
                "target_power": 2_800_000.0,
                "mechanical_efficiency": 0.99,
                "static_pressure_bracket": (120_000.0, 150_000.0),
            }
            kwargs.update(overrides)
            return orchestration.solve_prescribed_mechanical_power(**kwargs)

        bad_scalars = (0.0, -1.0, math.inf, math.nan, True, "1.0")
        for name in ("n", "D_hub", "target_total_pressure"):
            for value in bad_scalars:
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError):
                        call_total(**{name: value})
        for name, bad_values in (
            ("target_power", bad_scalars),
            ("mechanical_efficiency", (0.0, -0.1, 1.0001, math.inf,
                                       math.nan, True, "0.99")),
        ):
            for value in bad_values:
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError):
                        call_power(**{name: value})
        for name, bad_values in (
            ("relative_tol", (0.0, -1e-6, math.inf, math.nan, True, "1e-6")),
            ("max_iter", (0, -1, 1.5, True, "40")),
        ):
            for value in bad_values:
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError):
                        call_total(**{name: value})
                    with self.assertRaises(ValueError):
                        call_power(**{name: value})
        bad_brackets = (
            None,
            (120_000.0,),
            (120_000.0, 150_000.0, 200_000.0),
            (150_000.0, 120_000.0),
            (-1.0, 150_000.0),
            (119_999.0, 1_000_000.0),
            (120_000.0, math.nan),
            (math.inf, 150_000.0),
            "bracket",
        )
        for bracket in bad_brackets:
            with self.subTest(bracket=bracket):
                with self.assertRaises(ValueError):
                    call_total(static_pressure_bracket=bracket)
        for z in (0, -1, 1.5, True, "3"):
            namespace["z"] = z
            with self.subTest(z=z):
                with self.assertRaises(ValueError):
                    call_total()
            namespace["z"] = 2
        for inlet in (0.0, -1.0, math.inf, math.nan, True, "1e6"):
            namespace["p_г_полн"] = inlet
            with self.subTest(inlet=inlet):
                with self.assertRaises(ValueError):
                    call_total()
            namespace["p_г_полн"] = self.INLET_TOTAL
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_namespace_pressure_restored_on_success_failure_and_interrupt(
        self,
    ) -> None:
        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        orchestration.solve_total_outlet_pressure(
            84_000.0,
            0.065,
            160_000.0,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

        with self.assertRaises(NumericalConvergenceError):
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                500_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

        def interrupting(_n, _D_hub, _seed):
            p_static = namespace["p_т"]
            if p_static == 135_000.0:
                raise KeyboardInterrupt("simulated interrupt")
            return 1.0, _closure_stage_payloads(
                1e6, 1.1 * p_static + 10_000.0, 2
            )

        orchestration.распределить_π_const_hub = interrupting  # type: ignore[method-assign]
        with self.assertRaises(KeyboardInterrupt):
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_namespace_domain_and_inner_typed_errors_propagate(self) -> None:
        orchestration, namespace, _ = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        calls = {"count": 0}

        def no_solution(_n, _D_hub, _seed):
            calls["count"] += 1
            raise _НетРешения("no stage geometry solution")

        orchestration.распределить_π_const_hub = no_solution  # type: ignore[method-assign]
        with self.assertRaises(_НетРешения):
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertGreater(calls["count"], 0)
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

        def inner_typed(_n, _D_hub, _seed):
            raise NumericalConvergenceError(
                "inner secant failed",
                stage="inner.stage",
                reason="inner_reason",
            )

        orchestration.распределить_π_const_hub = inner_typed  # type: ignore[method-assign]
        with self.assertRaises(NumericalConvergenceError) as raised:
            orchestration.solve_total_outlet_pressure(
                84_000.0,
                0.065,
                160_000.0,
                static_pressure_bracket=(120_000.0, 150_000.0),
            )
        self.assertEqual(raised.exception.stage, "inner.stage")
        self.assertEqual(raised.exception.reason, "inner_reason")
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)

    def test_closure_does_not_mutate_flows_speed_or_geometry(self) -> None:
        orchestration, namespace, calls = self._make(
            total=lambda p: 1.1 * p + 10_000.0
        )
        before = {
            key: copy.deepcopy(value)
            for key, value in namespace.items()
            if key != "p_т"
        }
        record = orchestration.solve_total_outlet_pressure(
            84_000.0,
            0.065,
            160_000.0,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        after = {
            key: copy.deepcopy(value)
            for key, value in namespace.items()
            if key != "p_т"
        }
        self.assertEqual(after, before)
        self.assertEqual(namespace["p_т"], self.ORIGINAL_P_T)
        self.assertTrue(all(value != self.ORIGINAL_P_T for value in calls))
        self.assertEqual(record["source_point"]["stage_count"], 2)
        self.assertEqual(record["controls"]["n_rpm"], 84_000.0)
        self.assertEqual(record["controls"]["D_hub_m"], 0.065)


def _build_source_stage_fixture(model, architecture, *, p_t=432_100.0):
    """Build a fresh textbook namespace for one source architecture fixture."""

    from dataclasses import replace

    import optimize_extended as oe
    from architecture_smoke import request_for
    from turbine_engine.architecture_cycle import (
        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK as TEXTBOOK,
        calculate_architecture_cycle,
    )
    from turbine_engine.architecture_raw import (
        input_to_payload,
        payload_to_input,
    )

    request = replace(request_for(architecture), gas_property_basis=TEXTBOOK)
    result = calculate_architecture_cycle(
        payload_to_input(input_to_payload(request))
    )
    turbine = result.turbines[0]
    namespace, _ = oe.build_ns(
        model,
        gas_property_basis=TEXTBOOK,
        cycle_gas_property_basis=TEXTBOOK,
    )
    flow = float(result.gas_mass_flow)
    namespace.update({
        "T_г_полн": float(turbine.inlet.T_total),
        "p_г_полн": float(turbine.inlet.p_total),
        "G_в": flow,
        "G_c1": flow,
        "G_л_ост": flow,
        "G_base": (flow, flow, flow),
        "p_т": p_t,
        "z": 3,
        "α1_ст": [11.0, 11.0, 19.0],
        "κ_ст": [1.38, 1.12, 1.15],
        "π_веса": [0.75, 0.8, 1.0],
        "R": float(result.gas_model.R),
        "α_г": float(result.combustor.alpha),
        "N_зад": float(result.target_gearbox_input_power),
        "L_к": float(result.compressor.specific_work),
    })
    orchestration = build_const_hub_orchestration(
        namespace,
        interp=namespace.get("_линейно"),
        pressure_ratio_policy=PRESSURE_RATIO_RAW_UNBOUNDED,
    )
    orchestration.install()
    return result, turbine, namespace, orchestration


class ActualSourceStaticPressureClosureTests(unittest.TestCase):
    """Three-stage source fixtures for both closures; no selection."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import optimizer_core as oc
            import optimize_extended  # noqa: F401
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "source closure fixture needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        from turbine_engine.architecture import Architecture

        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        model = oc.load_model()
        cls.original_p_t = 432_100.0
        (
            cls.main_result,
            cls.main_turbine,
            cls.namespace,
            cls.orchestration,
        ) = _build_source_stage_fixture(
            model,
            Architecture.SINGLE_SHAFT_DUAL_CC,
            p_t=cls.original_p_t,
        )
        (
            cls.gg_result,
            cls.gg_turbine,
            cls.gg_namespace,
            cls.gg_orchestration,
        ) = _build_source_stage_fixture(
            model,
            Architecture.FREE_POWER_TURBINE,
            p_t=cls.original_p_t,
        )

    def test_source_total_pressure_closure_uses_actual_last_stage(self) -> None:
        namespace = self.namespace
        namespace["p_т"] = self.original_p_t
        before = {
            key: copy.deepcopy(namespace[key])
            for key in (
                "T_г_полн",
                "p_г_полн",
                "G_c1",
                "G_л_ост",
                "z",
                "α1_ст",
                "κ_ст",
                "π_веса",
            )
        }
        record = self.orchestration.solve_total_outlet_pressure(
            84_000.0,
            0.065,
            150_000.0,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        stages = record["actual_stages"]
        self.assertEqual(len(stages), 3)
        self.assertLessEqual(record["relative_residual"], 1e-6)
        self.assertGreater(record["solved_static_pressure_Pa"], 120_000.0)
        self.assertLess(record["solved_static_pressure_Pa"], 150_000.0)
        self.assertGreater(record["iterations"], 0)
        self.assertEqual(
            record["actual_metric"], stages[-1]["p_ст_п"]
        )
        self.assertEqual(
            record["actual_outlet_total_pressure_Pa"],
            stages[-1]["p_ст_п"],
        )
        self.assertTrue(math.isfinite(record["actual_metric"]))
        self.assertTrue(math.isfinite(record["solved_static_pressure_Pa"]))
        self.assertIn("T_ст_п", stages[-1])
        self.assertIn("p_ст_п", stages[-1])
        self.assertEqual(
            namespace["p_т"], self.original_p_t, "original p_т was not restored"
        )
        for key, value in before.items():
            with self.subTest(key=key):
                self.assertEqual(namespace[key], value)
        self.assertEqual(
            record["pressure_ratio_policy"], PRESSURE_RATIO_RAW_UNBOUNDED
        )
        self.assertEqual(record["physical_validation"], "NOT_PERFORMED")
        self.assertEqual(record["stage_count_selection"], "NONE")

    def test_source_gg_mechanical_power_closure(self) -> None:
        namespace = self.gg_namespace
        namespace["p_т"] = self.original_p_t
        mach = 0.3379
        k_out = float(self.gg_turbine.outlet.k)
        p_guess = float(self.gg_turbine.outlet.p_total) / (
            1.0 + 0.5 * (k_out - 1.0) * mach**2
        ) ** (k_out / (k_out - 1.0))
        bracket = (0.8 * p_guess, 1.2 * p_guess)
        self.assertGreater(bracket[0], 0.0)
        self.assertLess(
            bracket[1], float(self.gg_turbine.inlet.p_total)
        )
        target = float(self.gg_result.compressor.power)
        record = self.gg_orchestration.solve_prescribed_mechanical_power(
            84_000.0,
            0.065,
            target,
            0.99,
            static_pressure_bracket=bracket,
        )
        stages = record["actual_stages"]
        recomputed = 0.99 * sum(
            stage["G_c"] * stage["L_т"] for stage in stages
        )
        self.assertLessEqual(record["relative_residual"], 1e-6)
        self.assertGreater(record["solved_static_pressure_Pa"], bracket[0])
        self.assertLess(record["solved_static_pressure_Pa"], bracket[1])
        self.assertEqual(record["mechanical_power_W"], record["actual_metric"])
        self.assertAlmostEqual(record["actual_metric"], recomputed, places=6)
        self.assertAlmostEqual(
            record["actual_metric"], target, delta=1e-6 * abs(target) + 1e-9
        )
        self.assertEqual(
            record["actual_outlet_total_pressure_Pa"],
            stages[-1]["p_ст_п"],
        )
        self.assertEqual(namespace["p_т"], self.original_p_t)
        self.assertEqual(
            record["pressure_ratio_policy"], PRESSURE_RATIO_RAW_UNBOUNDED
        )

    def test_source_gg_endpoint_links_to_fpt_static_pressure_closure(self) -> None:
        """Close actual GG work, then feed its endpoint into a fresh FPT stage.

        This is a linkage regression only: the GG meanline power closure
        supplies the FPT inlet total state, the FPT total-pressure closure uses
        the unchanged flow and the same fixed speed/hub, and the resulting FPT
        mechanical power is reported without being driven to its 110 kW target.
        """

        import architecture_stage_smoke as smoke
        from turbine_engine.stage import StageResult

        model = smoke._get_model()
        payloads = {
            (point.assembly, point.stage_count): payload
            for point, payload in zip(
                smoke.plan_points(), smoke.plan_payloads()
            )
        }
        gg_raw = smoke._run_stage_meanline(payloads[("C_GG", 3)], model=model)
        fpt_raw = smoke._run_stage_meanline(payloads[("C_FPT", 1)], model=model)

        def bound_orchestration(namespace):
            for value in namespace.values():
                if inspect.ismethod(value) and isinstance(
                    value.__self__, ConstHubOrchestration
                ):
                    return value.__self__
            self.fail("namespace has no bound ConstHubOrchestration")

        gg_orchestration = bound_orchestration(gg_raw["namespace"])
        fpt_orchestration = bound_orchestration(fpt_raw["namespace"])
        gg_namespace, fpt_namespace = gg_raw["namespace"], fpt_raw["namespace"]
        gg_original_p_t = gg_namespace["p_т"]
        fpt_original_p_t = fpt_namespace["p_т"]

        gg_target = float(gg_raw["result"].compressor.power)
        gg_bracket = (
            0.8 * float(gg_raw["p_out_static"]),
            1.2 * float(gg_raw["p_out_static"]),
        )
        gg_closure = gg_orchestration.solve_prescribed_mechanical_power(
            84_000.0,
            0.065,
            gg_target,
            0.99,
            static_pressure_bracket=gg_bracket,
        )
        self.assertLessEqual(gg_closure["relative_residual"], 1e-6)
        self.assertAlmostEqual(
            gg_closure["actual_metric"], gg_target,
            delta=1e-6 * abs(gg_target),
        )
        self.assertEqual(
            gg_closure["mechanical_power_W"], gg_closure["actual_metric"]
        )
        self.assertEqual(gg_closure["controls"]["n_rpm"], 84_000.0)
        self.assertEqual(gg_closure["controls"]["D_hub_m"], 0.065)
        gg_last = StageResult.from_legacy_dict(
            gg_closure["actual_stages"][-1]
        )
        gg_endpoint_T = float(gg_last.T_total)
        gg_endpoint_p = float(gg_last.p_total)
        self.assertTrue(math.isfinite(gg_endpoint_T))
        self.assertTrue(math.isfinite(gg_endpoint_p))
        self.assertEqual(gg_namespace["p_т"], gg_original_p_t)

        # The fresh FPT stage receives the exact actual GG endpoint as its
        # source total state; flow and the fixed speed/hub are untouched.
        fpt_namespace["T_г_полн"] = gg_endpoint_T
        fpt_namespace["p_г_полн"] = gg_endpoint_p
        flow_keys = ("G_в", "G_c1", "G_л_ост")
        flow_before = {key: fpt_namespace[key] for key in flow_keys}
        self.assertEqual(
            float(fpt_namespace["G_c1"]), float(gg_namespace["G_c1"])
        )
        self.assertGreater(gg_endpoint_p, 150_000.0)

        fpt_closure = fpt_orchestration.solve_total_outlet_pressure(
            84_000.0,
            0.065,
            150_000.0,
            static_pressure_bracket=(120_000.0, 150_000.0),
        )
        self.assertLessEqual(fpt_closure["relative_residual"], 1e-6)
        self.assertAlmostEqual(
            fpt_closure["actual_metric"], 150_000.0, delta=1e-6 * 150_000.0
        )
        fpt_stages = fpt_closure["actual_stages"]
        self.assertEqual(len(fpt_stages), 1)
        self.assertEqual(
            fpt_closure["actual_metric"], fpt_stages[-1]["p_ст_п"]
        )
        fpt_first = StageResult.from_legacy_dict(fpt_stages[0])
        self.assertEqual(fpt_first.T0, gg_endpoint_T)
        self.assertEqual(fpt_first.p0, gg_endpoint_p)
        self.assertEqual(fpt_first.G_c, gg_last.G_c)
        self.assertEqual(
            fpt_closure["controls"]["n_rpm"], 84_000.0
        )
        self.assertEqual(
            fpt_closure["controls"]["D_hub_m"], 0.065
        )
        self.assertEqual(
            fpt_closure["pressure_ratio_policy"], PRESSURE_RATIO_RAW_UNBOUNDED
        )
        for key in flow_keys:
            self.assertEqual(fpt_namespace[key], flow_before[key])

        fpt_mechanical = 0.99 * sum(
            float(stage["G_c"]) * float(stage["L_т"])
            for stage in fpt_stages
        )
        self.assertTrue(math.isfinite(fpt_mechanical))
        self.assertGreater(fpt_mechanical, 0.0)
        self.assertAlmostEqual(
            fpt_mechanical, 106_095.68575825705,
            delta=1e-6 * 106_095.68575825705,
        )
        # The linkage is not tuned to the 110 kW shaft request.
        self.assertGreater(abs(fpt_mechanical - 110_000.0), 1_000.0)
        self.assertAlmostEqual(
            fpt_mechanical - 110_000.0, -3_904.314241742948,
            delta=1e-6 * 3_904.314241742948,
        )
        self.assertEqual(fpt_namespace["p_т"], fpt_original_p_t)


if __name__ == "__main__":
    unittest.main()
