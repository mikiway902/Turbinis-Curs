"""Integration parity for the optimizer's wired stage kernel.

PROVENANCE / STATUS: software-drift diagnostic only.

``optimizer_core.evaluate`` executes the notebook stage cell (cell 12) and then
replaces its ``ступень`` function with the stateless adapter backed by
``turbine_engine.stage.calculate_stage`` and
``turbine_engine.gas.calc_stage_cp_k``.  The notebook definitions therefore stay
available as the independent reference path, while production evaluations use
the extracted pure kernels.

These tests drive both paths at fixed candidates (no optimizer search, no sweep)
and compare every one of the 70 stage scalars for all three stages.  They also
run the existing runtime regression candidate through ``eval_candidate`` with
and without the adapter installed, so the orchestration around the stage
(rho/pi iterations, G scaling) is covered, not only a single direct stage call.

The candidate values below are not a validated physical baseline and not an
acceptable design; they are software-drift anchors only.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))


class StageWiringParityTests(unittest.TestCase):
    RELATIVE_TOLERANCE = 1e-12
    STAGE_FIELD_COUNT = 70

    # Non-default phi/psi keep the adapter from silently using StageConfig
    # defaults instead of the namespace values prepared by evaluate.
    CFGS = (
        {"q_n": 0.995, "seal_type": "honeycomb", "phi": 0.965, "psi": 0.95},
        {"q_n": 0.99, "seal_type": "abradable", "phi": 0.94, "psi": 0.93},
    )
    ANCHOR_SCENARIO = "B87_relaxed"
    ANCHOR_QN = 0.995
    ANCHOR_X = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)

    @classmethod
    def setUpClass(cls) -> None:
        # Importing the model requires the declared runtime dependencies. Skip
        # only when one of those is genuinely unavailable; any other import or
        # model failure must surface as an error, not as a skip.
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "stage wiring parity needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()

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

    def assertStageDictsEqual(self, wired, legacy) -> None:
        self.assertEqual(len(wired), len(legacy), "stage count changed")
        for index, (actual, expected) in enumerate(zip(wired, legacy), start=1):
            self.assertEqual(set(actual), set(expected), f"stage {index} keys changed")
            self.assertEqual(
                len(expected), self.STAGE_FIELD_COUNT, f"stage {index} field count"
            )
            for field, expected_value in expected.items():
                with self.subTest(stage=index, field=field):
                    self.assertFiniteNear(
                        float(actual[field]),
                        float(expected_value),
                        f"stage {index} field {field}",
                    )

    def evaluate_pair(self, x, cfg, full: bool = False):
        wired = self.oc.evaluate(x, self.model, cfg, full=full)
        legacy = self.oc._evaluate(
            x, self.model, cfg, full=full, stage_core=False
        )
        return wired, legacy

    def build_namespace_pair(self):
        """Default wired build and explicit notebook-reference build."""
        core_ns, _ = self.oe.build_ns(
            self.model, q_n_limit=self.ANCHOR_QN, orchestration_core=True
        )
        legacy_ns, _ = self.oe.build_ns(
            self.model, q_n_limit=self.ANCHOR_QN, orchestration_core=False
        )
        return core_ns, legacy_ns

    def test_evaluate_wired_path_matches_legacy_stage_dicts(self) -> None:
        original_adapter = self.oc._ступень_ядро
        calls: list[int] = []

        def counting_adapter(namespace):
            adapter = original_adapter(namespace)

            def counted(*args, **kwargs):
                calls.append(1)
                return adapter(*args, **kwargs)

            return counted

        self.oc._ступень_ядро = counting_adapter
        try:
            for cfg in self.CFGS:
                x = self.oc.warm_start(cfg)
                with self.subTest(cfg=cfg):
                    wired, legacy = self.evaluate_pair(x, cfg)

                    self.assertEqual(set(wired), set(legacy), "result keys changed")
                    self.assertEqual(wired["p"], legacy["p"])
                    self.assertStageDictsEqual(
                        wired["stages"], legacy["stages"]
                    )
                    for field in (
                        "lc1", "lc", "ll", "a2", "pi", "sigma", "D_t", "u_t",
                        "n_t", "N_val", "eta", "loss", "sigma_max",
                    ):
                        with self.subTest(cfg=cfg, field=field):
                            self.assertValueMatches(
                                wired[field], legacy[field], field
                            )
        finally:
            self.oc._ступень_ядро = original_adapter

        self.assertGreater(
            len(calls), 0, "evaluate did not call the turbine_engine.stage adapter"
        )

    def test_evaluate_wired_path_uses_pure_gas_helpers_only(self) -> None:
        cfg = self.CFGS[0]
        x = self.oc.warm_start(cfg)
        calls = {"gas": 0, "h0": 0}
        original_gas = self.oc.calc_gas_cp_k
        original_h0 = self.oc.calc_h0_for_pressure

        def counting_gas(*args, **kwargs):
            calls["gas"] += 1
            return original_gas(*args, **kwargs)

        def counting_h0(*args, **kwargs):
            calls["h0"] += 1
            return original_h0(*args, **kwargs)

        self.oc.calc_gas_cp_k = counting_gas
        self.oc.calc_h0_for_pressure = counting_h0
        try:
            wired = self.oc.evaluate(x, self.model, cfg)
            wired_calls = dict(calls)
            calls.update(gas=0, h0=0)
            legacy = self.oc._evaluate(
                x, self.model, cfg, stage_core=False
            )
            legacy_calls = dict(calls)
        finally:
            self.oc.calc_gas_cp_k = original_gas
            self.oc.calc_h0_for_pressure = original_h0

        self.assertGreater(
            wired_calls["gas"], 0, "wired path did not use pure calc_gas_cp_k"
        )
        self.assertGreater(
            wired_calls["h0"], 0, "wired path did not use pure calc_h0_for_pressure"
        )
        self.assertEqual(
            legacy_calls, {"gas": 0, "h0": 0},
            "legacy reference path must keep its own notebook gas helpers",
        )
        self.assertStageDictsEqual(wired["stages"], legacy["stages"])

    def test_reaction_wiring_uses_core_only_on_wired_path(self) -> None:
        cfg = self.CFGS[0]
        x = self.oc.warm_start(cfg)
        original = self.oc.calculate_root_reaction
        calls = {"wired": 0, "legacy": 0}
        mode = {"name": "wired"}

        def counting(*args, **kwargs):
            calls[mode["name"]] += 1
            return original(*args, **kwargs)

        self.oc.calculate_root_reaction = counting
        try:
            wired = self.oc.evaluate(x, self.model, cfg)
            mode["name"] = "legacy"
            legacy = self.oc._evaluate(x, self.model, cfg, stage_core=False)
        finally:
            self.oc.calculate_root_reaction = original

        self.assertGreater(
            calls["wired"],
            0,
            "wired path did not use turbine_engine.reaction.calculate_root_reaction",
        )
        self.assertEqual(
            calls["legacy"],
            0,
            "legacy reference path must keep its own notebook ρ_корня",
        )
        self.assertStageDictsEqual(wired["stages"], legacy["stages"])

    def test_reaction_adapter_matches_legacy_for_both_laws(self) -> None:
        core_ns, legacy_ns = self.build_namespace_pair()
        legacy = legacy_ns["ρ_корня"]
        adapter = core_ns["ρ_корня"]
        self.assertEqual(len(core_ns["ступени"]), 3)

        for index, stage in enumerate(core_ns["ступени"], start=1):
            for law in ("Γ", "α1"):
                with self.subTest(stage=index, law=law):
                    self.assertFiniteNear(
                        adapter(stage, law),
                        legacy(stage, law),
                        f"stage {index} law {law}",
                    )

    def test_full_profiles_stay_identical(self) -> None:
        cfg = self.CFGS[0]
        x = self.oc.warm_start(cfg)
        wired, legacy = self.evaluate_pair(x, cfg, full=True)

        self.assertEqual(set(wired), set(legacy), "full result keys changed")
        self.assertStageDictsEqual(wired["stages"], legacy["stages"])
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

    def test_runtime_anchor_matches_when_stage_is_replaced(self) -> None:
        core_ns, legacy_ns = self.build_namespace_pair()
        adapter = core_ns["ступень"]
        calls: list[int] = []

        def counted_adapter(*args, **kwargs):
            calls.append(1)
            return adapter(*args, **kwargs)

        core_ns["ступень"] = counted_adapter
        try:
            legacy = self.oe.eval_candidate(
                self.ANCHOR_X, legacy_ns, self.ANCHOR_SCENARIO, self.ANCHOR_QN
            )
            legacy_stages = list(legacy_ns["ступени"])

            wired = self.oe.eval_candidate(
                self.ANCHOR_X, core_ns, self.ANCHOR_SCENARIO, self.ANCHOR_QN
            )
            wired_stages = list(core_ns["ступени"])
        finally:
            core_ns["ступень"] = adapter

        self.assertGreater(len(calls), 0, "eval_candidate did not use the adapter")
        self.assertEqual(wired["status"], "ok", wired.get("err", ""))
        self.assertEqual(wired["err"], "")
        self.assertEqual(wired["status"], legacy["status"])
        self.assertEqual(wired["feasible"], legacy["feasible"])
        self.assertEqual(wired["eta_ok"], legacy["eta_ok"])
        self.assertEqual(wired["violation_count"], legacy["violation_count"])
        self.assertEqual(
            [violation["name"] for violation in wired["violations"]],
            [violation["name"] for violation in legacy["violations"]],
        )

        for field in (
            "lc1", "a2", "sigma", "u", "lD", "qn", "N", "cont", "energy",
            "p2z", "eta_t", "hard", "rho", "rho_root", "uc0", "uc0_opt",
        ):
            with self.subTest(field=field):
                self.assertValueMatches(wired[field], legacy[field], field)

        self.assertStageDictsEqual(wired_stages, legacy_stages)

    def test_adapter_uses_pure_gas_helper_not_legacy_cp_k(self) -> None:
        core_ns, legacy_ns = self.build_namespace_pair()
        legacy = legacy_ns["ступень"]
        adapter = core_ns["ступень"]
        stage = legacy_ns["ступени"][0]
        args = (
            stage["T0"], stage["p0"], stage["H0"], stage["ρ"], stage["D_c"],
            stage["D_л"], legacy_ns["n_т"], stage["G_c"], stage["G_л"],
            legacy_ns["α1_ст"][0],
        )
        expected = legacy(*args, 1.38)

        original_cp_k = core_ns["cp_k_ступени"]

        def forbidden(*_args, **_kwargs):
            raise AssertionError("adapter must not call legacy cp_k_ступени")

        core_ns["cp_k_ступени"] = forbidden
        try:
            actual = adapter(*args, 1.38)
        finally:
            core_ns["cp_k_ступени"] = original_cp_k

        self.assertStageDictsEqual([actual], [expected])

    def test_adapter_uses_actual_namespace_alpha_r_and_phi(self) -> None:
        core_ns, legacy_ns = self.build_namespace_pair()
        legacy = legacy_ns["ступень"]
        adapter = core_ns["ступень"]
        stage = legacy_ns["ступени"][0]
        args = (
            stage["T0"], stage["p0"], stage["H0"], stage["ρ"], stage["D_c"],
            stage["D_л"], legacy_ns["n_т"], stage["G_c"], stage["G_л"],
            legacy_ns["α1_ст"][0],
        )
        original = {
            name: legacy_ns[name] for name in ("α_г", "R", "φ")
        }
        try:
            # All three differ from the pure-core defaults / original namespace,
            # so silently ignoring any one of them would break parity.  Both
            # namespaces receive the same live values.
            changed = {
                "α_г": original["α_г"] * 0.9,
                "R": original["R"] + 0.5,
                "φ": original["φ"] - 0.02,
            }
            legacy_ns.update(changed)
            core_ns.update(changed)
            expected = legacy(*args, 1.38)
            actual = adapter(*args, 1.38)
        finally:
            legacy_ns.update(original)
            core_ns.update(original)

        self.assertStageDictsEqual([actual], [expected])

    def test_adapter_preserves_notebook_no_solution_control_flow(self) -> None:
        core_ns, legacy_ns = self.build_namespace_pair()
        legacy = legacy_ns["ступень"]
        adapter = core_ns["ступень"]
        stage = legacy_ns["ступени"][0]
        args = (
            stage["T0"], stage["p0"], stage["H0"], stage["ρ"], stage["D_c"],
            stage["D_л"], legacy_ns["n_т"], stage["G_c"], stage["G_л"],
            legacy_ns["α1_ст"][0],
        )

        # kappa large enough that c2a = kappa*c1a exceeds w2: the notebook
        # returns НетРешения("w2 <= c2a"), the adapter must surface the same
        # control-flow exception rather than leaking StageCalculationError.
        with self.assertRaises(legacy_ns["НетРешения"]):
            legacy(*args, 1000.0)
        with self.assertRaises(core_ns["НетРешения"]) as raised:
            adapter(*args, 1000.0)
        self.assertIs(type(raised.exception), core_ns["НетРешения"])

        # alpha1 = 90 deg makes the stator/rotor triangles singular.
        singular_args = args[:9] + (90.0,)
        with self.assertRaises(legacy_ns["НетРешения"]):
            legacy(*singular_args, 1.38)
        with self.assertRaises(core_ns["НетРешения"]) as raised:
            adapter(*singular_args, 1.38)
        self.assertIs(type(raised.exception), core_ns["НетРешения"])


if __name__ == "__main__":
    unittest.main()
