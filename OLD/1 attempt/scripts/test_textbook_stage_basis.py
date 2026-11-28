"""Opt-in textbook stage gas basis for the existing const-hub path.

PROVENANCE / STATUS: software-drift and wiring diagnostics only.

These tests do not run an optimizer search, sweep, large grid, performance
benchmark, or any new cycle/integral model.  The fixed candidate is the
existing regression anchor and remains an infeasible diagnostic point, not an
acceptable design.  The textbook basis is a source correlation; it is not an
additional physical validation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Optimizer"))
sys.path.insert(0, str(ROOT / "Turbinis-Calc"))

import optimizer_core as oc  # noqa: E402
import optimize_extended as oe  # noqa: E402
from turbine_engine.gas import (  # noqa: E402
    GAS_PROPERTY_BASIS_LEGACY,
    GAS_PROPERTY_BASIS_TEXTBOOK,
    calc_gas_cp_k,
    calc_stage_cp_k,
    calc_textbook_gas_cp_k,
)
from turbine_engine.cycle import (  # noqa: E402
    ALPHA_BASIS_LEGACY,
    ALPHA_BASIS_TEXTBOOK,
    CYCLE_GAS_PROPERTY_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
)


class TextbookStageBasisTests(unittest.TestCase):
    CANDIDATE = (84000.0, 0.065, 0.75, 0.80, 11.0, 11.0, 19.0, 1.38, 1.12, 1.15)
    SCENARIO = "B87_relaxed"
    Q_N_LIMIT = 0.995
    LEGACY_ANCHOR_LC1 = 0.011295345088530603

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = oc.load_model()
        cls.cycle_state_keys = ("T_г_полн", "охлаждение_лопаток", "g_охл", "g_воз")
        cls.cycle_state_at_load = {
            key: cls.model.cycle[key] for key in cls.cycle_state_keys
        }
        cls.ns_legacy, _ = oe.build_ns(cls.model, q_n_limit=cls.Q_N_LIMIT)
        cls.ns_textbook, _ = oe.build_ns(
            cls.model,
            q_n_limit=cls.Q_N_LIMIT,
            gas_property_basis=GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        cls.ns_cycle_textbook, cls.cycle_textbook_build = oe.build_ns(
            cls.model,
            q_n_limit=cls.Q_N_LIMIT,
            cycle_gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        cls.ns_full_textbook, cls.full_textbook_build = oe.build_ns(
            cls.model,
            q_n_limit=cls.Q_N_LIMIT,
            gas_property_basis=GAS_PROPERTY_BASIS_TEXTBOOK,
            cycle_gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        cls.cycle_legacy = oc.cycle_point(cls.model, 11.0, 1250.0, 110e3)
        cls.cycle_textbook = oc.cycle_point(
            cls.model,
            11.0,
            1250.0,
            110e3,
            gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        cls.legacy_result = oe.eval_candidate(
            cls.CANDIDATE, cls.ns_legacy, cls.SCENARIO, cls.Q_N_LIMIT
        )
        cls.textbook_result = oe.eval_candidate(
            cls.CANDIDATE, cls.ns_textbook, cls.SCENARIO, cls.Q_N_LIMIT
        )
        cls.cycle_textbook_result = oe.eval_candidate(
            cls.CANDIDATE, cls.ns_cycle_textbook, cls.SCENARIO, cls.Q_N_LIMIT
        )
        cls.full_textbook_result = oe.eval_candidate(
            cls.CANDIDATE, cls.ns_full_textbook, cls.SCENARIO, cls.Q_N_LIMIT
        )
        cls.cycle_state_after_builds = {
            key: cls.model.cycle[key] for key in cls.cycle_state_keys
        }

    def _stage_args(self, ns: dict):
        stage = ns["ступени"][0]
        return (
            stage["T0"],
            stage["p0"],
            stage["H0"],
            stage["ρ"],
            stage["D_c"],
            stage["D_л"],
            ns["n_т"],
            stage["G_c"],
            stage["G_л"],
            ns["α1_ст"][0],
            ns["κ_ст"][0],
        )

    def test_adapted_helpers_route_to_selected_gas_core(self) -> None:
        gas_calls = []
        textbook_calls = []
        stage_calls = []
        h0_calls = []
        original_gas = oc.calc_gas_cp_k
        original_textbook = oc.calc_textbook_gas_cp_k
        original_stage = oc.calc_stage_cp_k
        original_h0 = oc.calc_h0_for_pressure

        def record_gas(*args, **kwargs):
            gas_calls.append(kwargs)
            return original_gas(*args, **kwargs)

        def record_textbook(*args, **kwargs):
            textbook_calls.append(kwargs)
            return original_textbook(*args, **kwargs)

        def record_stage(*args, **kwargs):
            stage_calls.append(kwargs)
            return original_stage(*args, **kwargs)

        def record_h0(*args, **kwargs):
            h0_calls.append(kwargs)
            return original_h0(*args, **kwargs)

        args = self._stage_args(self.ns_textbook)
        with patch.object(oc, "calc_gas_cp_k", record_gas), \
                patch.object(oc, "calc_textbook_gas_cp_k", record_textbook), \
                patch.object(oc, "calc_stage_cp_k", record_stage), \
                patch.object(oc, "calc_h0_for_pressure", record_h0):
            self.ns_legacy["газ"].cache_clear()
            self.ns_textbook["газ"].cache_clear()
            self.ns_textbook["H0_по_p"].cache_clear()

            self.ns_legacy["газ"](1234.5)
            self.ns_textbook["газ"](1234.5)
            self.ns_textbook["ступень"](*args)
            self.ns_legacy["ступень"](*args)
            self.ns_textbook["H0_по_p"](
                self.ns_textbook["T_г_полн"],
                self.ns_textbook["p_г_полн"],
                self.ns_textbook["p_т"],
            )

        self.assertTrue(gas_calls, "legacy газ did not route to calc_gas_cp_k")
        self.assertTrue(
            textbook_calls, "textbook газ did not route to calc_textbook_gas_cp_k"
        )
        self.assertTrue(
            any(
                call.get("gas_property_basis") == GAS_PROPERTY_BASIS_TEXTBOOK
                for call in stage_calls
            ),
            stage_calls,
        )
        self.assertTrue(
            any(
                call.get("gas_property_basis") == GAS_PROPERTY_BASIS_LEGACY
                for call in stage_calls
            ),
            stage_calls,
        )
        self.assertTrue(
            any(
                call.get("gas_property_basis") == GAS_PROPERTY_BASIS_TEXTBOOK
                for call in h0_calls
            ),
            h0_calls,
        )

    def test_notebook_stage_cp_helper_uses_selected_gas(self) -> None:
        stage = self.ns_textbook["ступени"][0]
        T0, H0 = stage["T0"], stage["H0"]
        self.ns_legacy["cp_k_ступени"].cache_clear()
        self.ns_textbook["cp_k_ступени"].cache_clear()

        legacy_cp_k = self.ns_legacy["cp_k_ступени"](T0, H0)
        textbook_cp_k = self.ns_textbook["cp_k_ступени"](T0, H0)
        expected_legacy = calc_stage_cp_k(
            T0,
            H0,
            self.ns_legacy["α_г"],
            self.ns_legacy["R"],
            self.ns_legacy["φ"],
            gas_property_basis=GAS_PROPERTY_BASIS_LEGACY,
        )
        expected_textbook = calc_stage_cp_k(
            T0,
            H0,
            self.ns_textbook["α_г"],
            self.ns_textbook["R"],
            self.ns_textbook["φ"],
            gas_property_basis=GAS_PROPERTY_BASIS_TEXTBOOK,
        )

        self.assertAlmostEqual(legacy_cp_k[0] / expected_legacy[0], 1.0, delta=1e-12)
        self.assertAlmostEqual(legacy_cp_k[1] / expected_legacy[1], 1.0, delta=1e-12)
        self.assertAlmostEqual(
            textbook_cp_k[0] / expected_textbook[0], 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            textbook_cp_k[1] / expected_textbook[1], 1.0, delta=1e-12
        )
        self.assertNotEqual(legacy_cp_k, textbook_cp_k)

    def test_textbook_opt_in_changes_stage_cp_k_and_pressure_closure(self) -> None:
        stage = self.ns_textbook["ступени"][0]
        args = self._stage_args(self.ns_textbook)
        legacy_stage = self.ns_legacy["ступень"](*args)
        textbook_stage = self.ns_textbook["ступень"](*args)

        self.assertGreater(
            abs(textbook_stage["cp"] - legacy_stage["cp"]), 1e-3,
            "textbook opt-in did not change stage cp",
        )
        self.assertGreater(
            abs(textbook_stage["k"] - legacy_stage["k"]), 1e-6,
            "textbook opt-in did not change stage k",
        )

        self.ns_legacy["H0_по_p"].cache_clear()
        self.ns_textbook["H0_по_p"].cache_clear()
        h0_args = (
            self.ns_textbook["T_г_полн"],
            self.ns_textbook["p_г_полн"],
            self.ns_textbook["p_т"],
        )
        legacy_h0 = self.ns_legacy["H0_по_p"](*h0_args)
        textbook_h0 = self.ns_textbook["H0_по_p"](*h0_args)
        self.assertGreater(
            abs(textbook_h0 - legacy_h0), 1e-3,
            "textbook opt-in did not change the H0(p) pressure closure",
        )

        self.assertEqual(
            self.textbook_result["status"], "ok",
            self.textbook_result.get("err", ""),
        )
        self.assertGreater(
            abs(self.textbook_result["lc1"] - self.legacy_result["lc1"]), 1e-6,
            "textbook opt-in did not affect the fixed-point candidate",
        )
        self.assertEqual(
            self.textbook_result["stage_gas_property_basis"],
            GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            self.textbook_result["cycle_gas_property_basis"], "legacy_mean_cp"
        )
        self.assertTrue(self.textbook_result["NOT_PHYSICALLY_VALIDATED"])
        self.assertIn("diagnostic", self.textbook_result["model_variant"])

    def test_textbook_stage_cp_matches_direct_source_at_recorded_mean(self) -> None:
        # Independent software check: the adapter reports cp/k at the mean
        # conditional temperature that its own recorded cp implies.  This is
        # routing verification, not a new physical oracle.
        for index, stage in enumerate(self.textbook_result["stages"], start=1):
            with self.subTest(stage=index):
                mean_t = (
                    stage["T0"]
                    - stage["H0"] * self.ns_textbook["φ"] ** 2
                    / (2.0 * stage["cp"])
                )
                expected_cp, expected_k = calc_textbook_gas_cp_k(
                    mean_t,
                    self.ns_textbook["α_г"],
                    self.ns_textbook["R"],
                )
                self.assertAlmostEqual(stage["cp"] / expected_cp, 1.0, delta=1e-9)
                self.assertAlmostEqual(stage["k"] / expected_k, 1.0, delta=1e-9)

    def test_cycle_option_changes_outputs_and_reports_method_metadata(self) -> None:
        for field in ("G", "L_k", "alpha_g", "c_t"):
            with self.subTest(field=field):
                legacy = self.cycle_legacy[field]
                textbook = self.cycle_textbook[field]
                self.assertGreater(
                    abs(textbook / legacy - 1.0),
                    1e-3,
                    f"cycle textbook opt-in did not change {field}",
                )

        self.assertEqual(
            self.cycle_textbook["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            self.cycle_textbook["cycle_alpha_basis"],
            ALPHA_BASIS_TEXTBOOK,
        )
        diagnostics = self.cycle_textbook["cycle_method_diagnostics"]
        self.assertIsInstance(diagnostics, dict)
        self.assertEqual(
            diagnostics["gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(diagnostics["alpha_basis"], ALPHA_BASIS_TEXTBOOK)
        self.assertIn("fuel", diagnostics)
        self.assertIn("relation", diagnostics["fuel"])

        # The default wrapper mapping remains the old one.
        for key in (
            "cycle_gas_property_basis",
            "cycle_alpha_basis",
            "cycle_method_diagnostics",
        ):
            with self.subTest(key=key):
                self.assertNotIn(key, self.cycle_legacy)

    def test_full_textbook_build_uses_selected_cycle_values_and_fuel_alpha(
        self,
    ) -> None:
        ns = self.ns_full_textbook
        cycle = self.cycle_textbook
        self.assertEqual(
            ns["_gas_property_basis"], GAS_PROPERTY_BASIS_TEXTBOOK
        )
        self.assertEqual(
            ns["_cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            self.full_textbook_build["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            self.full_textbook_build["cycle_alpha_basis"],
            ALPHA_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            ns["G_base"],
            (
                cycle["G"],
                cycle["G"] * cycle["g_t"],
                cycle["G"] * (cycle["g_t"] + cycle["g_voz"]),
            ),
        )
        self.assertEqual(ns["α_г"], cycle["alpha_g"])
        self.assertEqual(ns["L_к"], cycle["L_k"])
        self.assertEqual(ns["c_та"], cycle["c_t"])
        self.assertEqual(ns["p_г_полн"], cycle["p_g"])
        self.assertEqual(ns["p_т"], cycle["p_t"])
        self.assertNotEqual(ns["α_г"], self.ns_legacy["α_г"])

        result = self.full_textbook_result
        self.assertEqual(result["status"], "ok", result.get("err", ""))
        self.assertEqual(
            result["stage_gas_property_basis"], GAS_PROPERTY_BASIS_TEXTBOOK
        )
        self.assertEqual(
            result["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(result["cycle_alpha_basis"], ALPHA_BASIS_TEXTBOOK)
        self.assertIsNotNone(result["cycle_method_diagnostics"])
        self.assertTrue(result["NOT_PHYSICALLY_VALIDATED"])
        self.assertEqual(
            result["cycle_assumptions"]["cooling"],
            "cooling_bookkeeping_only_no_temperature_mixing",
        )
        self.assertAlmostEqual(result["N"] / 110e3, 1.0, delta=1e-9)
        self.assertAlmostEqual(result["N"] * 0.98 / 107.8e3, 1.0, delta=1e-9)

        # The stage cp/k must use the fuel-based alpha and other inputs of the
        # selected cycle build, not the legacy cycle alpha.
        for index, stage in enumerate(result["stages"], start=1):
            with self.subTest(stage=index):
                mean_t = (
                    stage["T0"]
                    - stage["H0"] * ns["φ"] ** 2 / (2.0 * stage["cp"])
                )
                expected_cp, expected_k = calc_textbook_gas_cp_k(
                    mean_t, ns["α_г"], ns["R"]
                )
                self.assertAlmostEqual(stage["cp"] / expected_cp, 1.0, delta=1e-9)
                self.assertAlmostEqual(stage["k"] / expected_k, 1.0, delta=1e-9)

    def test_stage_and_cycle_options_are_independent_and_labelled(self) -> None:
        self.assertEqual(
            self.textbook_result["model_variant"],
            "extended_const_hub+textbook_true_cp_diagnostic",
        )
        self.assertEqual(
            self.textbook_result["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertEqual(
            self.cycle_textbook_result["model_variant"],
            oe.TEXTBOOK_CYCLE_DIAGNOSTIC_MODEL_VARIANT,
        )
        self.assertEqual(
            self.cycle_textbook_result["stage_gas_property_basis"],
            GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertEqual(
            self.cycle_textbook_result["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertTrue(
            self.cycle_textbook_result["NOT_PHYSICALLY_VALIDATED"]
        )
        self.assertEqual(
            self.full_textbook_result["model_variant"],
            oe.TEXTBOOK_BOTH_DIAGNOSTIC_MODEL_VARIANT,
        )
        self.assertEqual(
            self.legacy_result["model_variant"], "extended_const_hub"
        )

    def test_textbook_cycle_build_does_not_mutate_loaded_cycle_namespace(
        self,
    ) -> None:
        self.assertEqual(
            self.cycle_state_after_builds, self.cycle_state_at_load
        )
        before = {
            key: self.model.cycle[key] for key in self.cycle_state_keys
        }
        oe.build_ns(
            self.model,
            q_n_limit=self.Q_N_LIMIT,
            gas_property_basis=GAS_PROPERTY_BASIS_TEXTBOOK,
            cycle_gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        after = {
            key: self.model.cycle[key] for key in self.cycle_state_keys
        }
        self.assertEqual(after, before)

    def test_cycle_marker_mutation_is_rejected_before_physical_eval(self) -> None:
        cases = (
            (
                self.ns_legacy,
                CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
                CYCLE_GAS_PROPERTY_BASIS_LEGACY,
                GAS_PROPERTY_BASIS_LEGACY,
                False,
            ),
            (
                self.ns_full_textbook,
                CYCLE_GAS_PROPERTY_BASIS_LEGACY,
                CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
                GAS_PROPERTY_BASIS_TEXTBOOK,
                True,
            ),
        )
        for source_ns, bad_marker, actual_cycle, actual_stage, textbook in cases:
            with self.subTest(bad_marker=bad_marker):
                namespace = dict(source_ns)
                namespace["_cycle_gas_property_basis"] = bad_marker
                calls: list[int] = []

                def forbidden(*_args, **_kwargs):
                    calls.append(1)
                    raise AssertionError("physical evaluation must not start")

                namespace["распределить_π_const_hub"] = forbidden
                result = oe.eval_candidate(
                    self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
                )

                self.assertEqual(result["status"], "model_error")
                self.assertIn(
                    "does not match installed cycle wiring", result["err"]
                )
                self.assertEqual(
                    result["cycle_gas_property_basis"], actual_cycle
                )
                self.assertEqual(
                    result["stage_gas_property_basis"], actual_stage
                )
                if textbook:
                    self.assertTrue(result["NOT_PHYSICALLY_VALIDATED"])
                else:
                    self.assertNotIn("NOT_PHYSICALLY_VALIDATED", result)
                self.assertEqual(calls, [], "physical evaluation was started")

    def test_unknown_cycle_marker_and_wiring_are_rejected_not_defaulted(
        self,
    ) -> None:
        def forbidden(*_args, **_kwargs):
            raise AssertionError("physical evaluation must not start")

        namespace = dict(self.ns_legacy)
        namespace["_cycle_gas_property_basis"] = "bogus_cycle_marker"
        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )
        self.assertEqual(result["status"], "model_error")
        self.assertIn("unknown cycle gas_property_basis marker", result["err"])
        self.assertEqual(
            result["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_LEGACY,
        )

        def fake_gas(T):
            raise AssertionError("fake gas must not be called")

        setattr(fake_gas, oc.GAS_PROPERTY_BASIS_ATTR, GAS_PROPERTY_BASIS_LEGACY)
        setattr(
            fake_gas, oc.CYCLE_GAS_PROPERTY_BASIS_ATTR, "bogus_cycle_wiring"
        )
        namespace = dict(self.ns_legacy)
        namespace["газ"] = fake_gas
        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )
        self.assertEqual(result["status"], "model_error")
        self.assertIn(
            "unknown installed cycle gas_property_basis", result["err"]
        )
        self.assertEqual(
            result["cycle_gas_property_basis"], "bogus_cycle_wiring"
        )

        def gas_without_cycle_metadata(T):
            return self.ns_legacy["газ"](T)

        setattr(
            gas_without_cycle_metadata,
            oc.GAS_PROPERTY_BASIS_ATTR,
            GAS_PROPERTY_BASIS_LEGACY,
        )
        namespace = dict(self.ns_legacy)
        namespace["газ"] = gas_without_cycle_metadata
        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )
        self.assertEqual(result["status"], "model_error")
        self.assertIn("no cycle construction metadata", result["err"])

    def test_cycle_textbook_reference_and_unknown_selector_rejected_early(
        self,
    ) -> None:
        with patch.object(
            oc,
            "cycle_point",
            side_effect=AssertionError("calculation must not start"),
        ) as cycle_point:
            with self.assertRaises(ValueError) as caught:
                oe.build_ns(
                    self.model,
                    cycle_gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
                    orchestration_core=False,
                )
            cycle_point.assert_not_called()
            self.assertIn("orchestration_core=True", str(caught.exception))

            for value in ("bogus", None, 1):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        oe.build_ns(
                            self.model,
                            cycle_gas_property_basis=value,
                        )
            cycle_point.assert_not_called()

    def test_cycle_point_unknown_selector_is_rejected_by_existing_core(
        self,
    ) -> None:
        for value in ("bogus", None, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    oc.cycle_point(
                        self.model, gas_property_basis=value
                    )

    def test_cycle_provenance_survives_solver_and_candidate_failures(self) -> None:
        namespace = dict(self.ns_full_textbook)

        def failing_solver(*_args, **_kwargs):
            raise RuntimeError("mock solver failure")

        namespace["распределить_π_const_hub"] = failing_solver
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )
        self.assertEqual(result["status"], "model_error")
        self.assertIn("mock solver failure", result["err"])
        self.assertEqual(
            result["stage_gas_property_basis"], GAS_PROPERTY_BASIS_TEXTBOOK
        )
        self.assertEqual(
            result["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(result["cycle_alpha_basis"], ALPHA_BASIS_TEXTBOOK)
        self.assertIsNotNone(result["cycle_method_diagnostics"])
        self.assertTrue(result["NOT_PHYSICALLY_VALIDATED"])

        zero = oe.eval_candidate(
            (0.0,) * len(self.CANDIDATE),
            dict(self.ns_full_textbook),
            self.SCENARIO,
            self.Q_N_LIMIT,
        )
        self.assertEqual(zero["status"], "model_error")
        self.assertEqual(
            zero["cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(zero["cycle_alpha_basis"], ALPHA_BASIS_TEXTBOOK)
        self.assertIsNotNone(zero["cycle_method_diagnostics"])
        self.assertTrue(zero["NOT_PHYSICALLY_VALIDATED"])

    def test_default_build_stays_on_legacy_anchor(self) -> None:
        self.assertEqual(
            self.ns_legacy["_gas_property_basis"], GAS_PROPERTY_BASIS_LEGACY
        )
        self.assertEqual(self.legacy_result["status"], "ok")
        self.assertEqual(
            self.legacy_result["model_variant"], "extended_const_hub"
        )
        self.assertEqual(
            self.legacy_result["stage_gas_property_basis"],
            GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertEqual(
            self.legacy_result["cycle_gas_property_basis"], "legacy_mean_cp"
        )
        self.assertEqual(
            self.legacy_result["cycle_alpha_basis"], ALPHA_BASIS_LEGACY
        )
        self.assertIsNone(self.legacy_result["cycle_method_diagnostics"])
        self.assertEqual(
            self.legacy_result["cycle_fuel_relation"],
            "source calc_g_m mean-cp balance at T_ref",
        )
        self.assertEqual(
            self.ns_legacy["_cycle_gas_property_basis"],
            CYCLE_GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertNotIn("NOT_PHYSICALLY_VALIDATED", self.legacy_result)
        self.assertAlmostEqual(
            self.legacy_result["N"] / 110e3, 1.0, delta=1e-9
        )
        self.assertAlmostEqual(
            self.legacy_result["N"] * 0.98 / 107.8e3, 1.0, delta=1e-9
        )
        self.assertAlmostEqual(
            self.legacy_result["lc1"] / self.LEGACY_ANCHOR_LC1,
            1.0,
            delta=1e-6,
        )

        for index, stage in enumerate(self.legacy_result["stages"], start=1):
            with self.subTest(stage=index):
                mean_t = (
                    stage["T0"]
                    - stage["H0"] * self.ns_legacy["φ"] ** 2
                    / (2.0 * stage["cp"])
                )
                expected_cp, expected_k = calc_gas_cp_k(
                    mean_t,
                    self.ns_legacy["α_г"],
                    self.ns_legacy["R"],
                )
                self.assertAlmostEqual(stage["cp"] / expected_cp, 1.0, delta=1e-9)
                self.assertAlmostEqual(stage["k"] / expected_k, 1.0, delta=1e-9)

    def test_caches_are_per_build_and_basis_is_fixed(self) -> None:
        for name in ("газ", "H0_по_p", "cp_k_ступени", "ступень"):
            with self.subTest(name=name):
                self.assertIsNot(self.ns_legacy[name], self.ns_textbook[name])

        self.ns_legacy["газ"].cache_clear()
        self.ns_textbook["газ"].cache_clear()
        legacy_first = self.ns_legacy["газ"](1234.5)
        textbook = self.ns_textbook["газ"](1234.5)
        legacy_second = self.ns_legacy["газ"](1234.5)

        self.assertEqual(legacy_first, legacy_second)
        self.assertNotEqual(legacy_first, textbook)

        # The selected basis is captured by the adapters.  Rebuild the
        # namespace to change it; mutating the marker is not a wiring switch.
        args = self._stage_args(self.ns_legacy)
        before = self.ns_legacy["ступень"](*args)
        try:
            self.ns_legacy["_gas_property_basis"] = GAS_PROPERTY_BASIS_TEXTBOOK
            after = self.ns_legacy["ступень"](*args)
        finally:
            self.ns_legacy["_gas_property_basis"] = GAS_PROPERTY_BASIS_LEGACY
        self.assertEqual(before, after)

    def test_marker_mutation_is_rejected_before_physical_eval(self) -> None:
        cases = (
            (
                self.ns_legacy,
                GAS_PROPERTY_BASIS_TEXTBOOK,
                GAS_PROPERTY_BASIS_LEGACY,
                False,
            ),
            (
                self.ns_textbook,
                GAS_PROPERTY_BASIS_LEGACY,
                GAS_PROPERTY_BASIS_TEXTBOOK,
                True,
            ),
        )
        for source_ns, bad_marker, actual_basis, textbook in cases:
            with self.subTest(bad_marker=bad_marker):
                namespace = dict(source_ns)
                namespace["_gas_property_basis"] = bad_marker
                calls: list[int] = []

                def forbidden(*_args, **_kwargs):
                    calls.append(1)
                    raise AssertionError("physical evaluation must not start")

                namespace["распределить_π_const_hub"] = forbidden
                result = oe.eval_candidate(
                    self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
                )

                self.assertEqual(result["status"], "model_error")
                self.assertIn(
                    "does not match installed wiring", result["err"]
                )
                self.assertEqual(
                    result["stage_gas_property_basis"], actual_basis
                )
                self.assertEqual(
                    result["cycle_gas_property_basis"], "legacy_mean_cp"
                )
                if textbook:
                    self.assertTrue(result["NOT_PHYSICALLY_VALIDATED"])
                else:
                    self.assertNotIn("NOT_PHYSICALLY_VALIDATED", result)
                self.assertEqual(calls, [], "physical evaluation was started")

    def test_unknown_marker_is_rejected_not_defaulted(self) -> None:
        namespace = dict(self.ns_legacy)
        namespace["_gas_property_basis"] = "bogus_selector"
        calls: list[int] = []

        def forbidden(*_args, **_kwargs):
            calls.append(1)
            raise AssertionError("physical evaluation must not start")

        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )

        self.assertEqual(result["status"], "model_error")
        self.assertIn("unknown gas_property_basis marker", result["err"])
        self.assertEqual(
            result["stage_gas_property_basis"], GAS_PROPERTY_BASIS_LEGACY
        )
        self.assertEqual(calls, [], "physical evaluation was started")

    def test_unknown_installed_wiring_is_rejected_not_defaulted(self) -> None:
        namespace = dict(self.ns_legacy)

        def fake_gas(T):
            raise AssertionError("fake gas must not be called")

        setattr(fake_gas, oc.GAS_PROPERTY_BASIS_ATTR, "bogus_wiring")
        namespace["газ"] = fake_gas
        calls: list[int] = []

        def forbidden(*_args, **_kwargs):
            calls.append(1)
            raise AssertionError("physical evaluation must not start")

        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )

        self.assertEqual(result["status"], "model_error")
        self.assertIn("unknown installed gas_property_basis", result["err"])
        self.assertEqual(calls, [], "physical evaluation was started")

    def test_missing_installed_wiring_metadata_is_rejected(self) -> None:
        namespace = dict(self.ns_legacy)
        original_gas = namespace["газ"]

        def gas_without_metadata(T):
            return original_gas(T)

        namespace["газ"] = gas_without_metadata
        calls: list[int] = []

        def forbidden(*_args, **_kwargs):
            calls.append(1)
            raise AssertionError("physical evaluation must not start")

        namespace["распределить_π_const_hub"] = forbidden
        result = oe.eval_candidate(
            self.CANDIDATE, namespace, self.SCENARIO, self.Q_N_LIMIT
        )

        self.assertEqual(result["status"], "model_error")
        self.assertIn("no construction metadata", result["err"])
        self.assertEqual(calls, [], "physical evaluation was started")

    def test_unknown_selector_is_rejected_before_calculation(self) -> None:
        with patch.object(
            oc,
            "cycle_point",
            side_effect=AssertionError("calculation must not start"),
        ) as cycle_point:
            for value in ("bogus", None, 1):
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        oe.build_ns(self.model, gas_property_basis=value)
        cycle_point.assert_not_called()

    def test_textbook_with_reference_branch_is_rejected_before_calculation(
        self,
    ) -> None:
        with patch.object(
            oc,
            "cycle_point",
            side_effect=AssertionError("calculation must not start"),
        ) as cycle_point:
            with self.assertRaises(ValueError) as caught:
                oe.build_ns(
                    self.model,
                    gas_property_basis=GAS_PROPERTY_BASIS_TEXTBOOK,
                    orchestration_core=False,
                )
        cycle_point.assert_not_called()
        self.assertIn("orchestration_core=True", str(caught.exception))

    def test_provenance_survives_calculation_failures(self) -> None:
        model_error = oe.eval_candidate(
            (0.0,) * len(self.CANDIDATE),
            dict(self.ns_textbook),
            self.SCENARIO,
            self.Q_N_LIMIT,
        )
        self.assertEqual(model_error["status"], "model_error")
        self.assertEqual(
            model_error["stage_gas_property_basis"],
            GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            model_error["cycle_gas_property_basis"], "legacy_mean_cp"
        )
        self.assertTrue(model_error["NOT_PHYSICALLY_VALIDATED"])

        ns_bad = dict(self.ns_textbook)
        bad_stages = [dict(stage) for stage in self.ns_textbook["ступени"]]
        for stage in bad_stages:
            stage["σ_р"] = float("nan")

        def distribute_with_non_finite_sigma(n, D_hub, pi0):
            return float(pi0), bad_stages

        ns_bad["распределить_π_const_hub"] = distribute_with_non_finite_sigma
        numerical = oe.eval_candidate(
            self.CANDIDATE, ns_bad, self.SCENARIO, self.Q_N_LIMIT
        )
        self.assertEqual(numerical["status"], "numerical")
        self.assertEqual(
            numerical["stage_gas_property_basis"],
            GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            numerical["cycle_gas_property_basis"], "legacy_mean_cp"
        )
        self.assertTrue(numerical["NOT_PHYSICALLY_VALIDATED"])


if __name__ == "__main__":
    unittest.main()
