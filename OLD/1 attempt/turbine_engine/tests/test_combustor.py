"""Focused tests for the standalone scalar combustor heat-addition wrapper.

These tests prove exact parity with the existing ``cycle.calc_alpha`` and
``cycle.calc_g_m`` helpers, validation, immutability, exports, and the absence
of cycle wiring.  They do not claim a combustor geometry, performance-map, or
flow/composition state model.
"""

from __future__ import annotations

import sys
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine  # noqa: E402
import turbine_engine.combustor as combustor_module  # noqa: E402
import turbine_engine.cycle as cycle_module  # noqa: E402

from turbine_engine.combustor import (  # noqa: E402
    CombustorResult,
    combustor_heat_addition,
)
from turbine_engine.cycle import (  # noqa: E402
    CycleConfig,
    CycleState,
    DEFAULT_CONFIG,
    calc_alpha,
    calc_compressor,
    calc_g_m,
    calculate_cycle_point,
)


class ProvenanceTests(unittest.TestCase):
    def test_outputs_are_exactly_the_existing_helper_values(self) -> None:
        config = CycleConfig(
            q_lhv_kj=43000.0,
            eta_combustion=0.97,
            l0=14.5,
            T_ref=290.0,
        )
        cases = (
            (600.0, 1200.0),
            (643.7298018608001, 1250.0),
            (700.0, 1300.0),
        )
        for T_in, T_gas in cases:
            with self.subTest(T_in=T_in, T_gas=T_gas):
                result = combustor_heat_addition(T_in, T_gas, config)

                self.assertEqual(result.T_in, T_in)
                self.assertEqual(result.T_gas, T_gas)
                self.assertEqual(result.alpha, calc_alpha(T_in, T_gas))
                self.assertEqual(result.g_m, calc_g_m(T_in, T_gas, config))

    def test_default_anchor_has_known_helper_values(self) -> None:
        T_in = 643.7298018608001
        T_gas = 1250.0
        result = combustor_heat_addition(T_in, T_gas, DEFAULT_CONFIG)

        self.assertEqual(result.alpha, 4.042798736594046)
        self.assertEqual(result.g_m, 0.017226061993152738)
        self.assertEqual(result.alpha, calc_alpha(T_in, T_gas))
        self.assertEqual(result.g_m, calc_g_m(T_in, T_gas, DEFAULT_CONFIG))

    def test_default_cycle_anchor_g_m_parity(self) -> None:
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        config = CycleConfig()
        cycle = calculate_cycle_point(state, config)
        T_k, _ = calc_compressor(state.pi, config)

        result = combustor_heat_addition(T_k, state.T_g, config)

        self.assertEqual(result.g_m, cycle.g_m)
        self.assertEqual(result.g_m, calc_g_m(T_k, state.T_g, config))
        self.assertEqual(result.alpha, calc_alpha(T_k, state.T_g))

    def test_result_alpha_is_the_helper_not_the_cycle_derived_field(self) -> None:
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        config = CycleConfig()
        cycle = calculate_cycle_point(state, config)
        T_k, _ = calc_compressor(state.pi, config)

        result = combustor_heat_addition(T_k, state.T_g, config)

        self.assertEqual(result.alpha, calc_alpha(T_k, state.T_g))
        self.assertNotEqual(result.alpha, cycle.alpha)


class ValidationTests(unittest.TestCase):
    def test_invalid_inlet_temperature_raises(self) -> None:
        for T_in in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            "bad",
            None,
        ):
            with self.subTest(T_in=T_in):
                with self.assertRaises(ValueError):
                    combustor_heat_addition(T_in, 1250.0)  # type: ignore[arg-type]

    def test_invalid_target_temperature_raises(self) -> None:
        for T_gas in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            "bad",
            None,
        ):
            with self.subTest(T_gas=T_gas):
                with self.assertRaises(ValueError):
                    combustor_heat_addition(600.0, T_gas)  # type: ignore[arg-type]

    def test_invalid_config_type_raises(self) -> None:
        with self.assertRaises(TypeError):
            combustor_heat_addition(
                600.0, 1250.0, "not a config"  # type: ignore[arg-type]
            )

    def test_manual_constructor_rejects_invalid_fields(self) -> None:
        bad_records = (
            {"T_in": 0.0, "T_gas": 1250.0, "g_m": 0.02, "alpha": 3.0},
            {"T_in": 600.0, "T_gas": -1.0, "g_m": 0.02, "alpha": 3.0},
            {"T_in": 600.0, "T_gas": 1250.0, "g_m": float("nan"), "alpha": 3.0},
            {"T_in": 600.0, "T_gas": 1250.0, "g_m": 0.02, "alpha": float("inf")},
        )
        for kwargs in bad_records:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CombustorResult(**kwargs)

    def test_manual_constructor_accepts_finite_scalars(self) -> None:
        result = CombustorResult(
            T_in=600.0, T_gas=1250.0, g_m=0.02, alpha=3.0
        )
        self.assertEqual(result.g_m, 0.02)
        self.assertEqual(result.alpha, 3.0)

    def test_replace_revalidates_fields(self) -> None:
        result = combustor_heat_addition(600.0, 1250.0)
        with self.assertRaises(ValueError):
            replace(result, g_m=float("nan"))
        with self.assertRaises(ValueError):
            replace(result, T_in=-1.0)


class ImmutabilityAndSideEffectTests(unittest.TestCase):
    def test_calls_do_not_mutate_inputs(self) -> None:
        config = CycleConfig()
        config_before = asdict(config)
        T_in = 643.7298018608001
        T_gas = 1250.0

        combustor_heat_addition(T_in, T_gas, config)

        self.assertEqual(asdict(config), config_before)
        self.assertEqual((T_in, T_gas), (643.7298018608001, 1250.0))

    def test_result_is_immutable(self) -> None:
        result = combustor_heat_addition(600.0, 1250.0)

        self.assertIsInstance(result, CombustorResult)
        with self.assertRaises(FrozenInstanceError):
            result.g_m = 1.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.alpha = 1.0  # type: ignore[misc]

    def test_calls_do_not_change_cycle_outputs(self) -> None:
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        config = CycleConfig()
        T_k, _ = calc_compressor(state.pi, config)
        before = calculate_cycle_point(state, config)

        first = combustor_heat_addition(T_k, state.T_g, config)
        second = combustor_heat_addition(T_k, state.T_g, config)
        after = calculate_cycle_point(state, config)

        self.assertEqual(first, second)
        self.assertEqual(before, after)


class ExportAndScopeTests(unittest.TestCase):
    def test_package_exports_the_wrapper(self) -> None:
        self.assertIs(turbine_engine.CombustorResult, CombustorResult)
        self.assertIs(
            turbine_engine.combustor_heat_addition,
            combustor_heat_addition,
        )

    def test_wrapper_is_not_wired_into_the_cycle_module(self) -> None:
        source = Path(cycle_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("turbine_engine.combustor", source)
        self.assertFalse(hasattr(cycle_module, "combustor_heat_addition"))

    def test_module_documents_scope_and_provenance(self) -> None:
        doc = combustor_module.__doc__ or ""
        lowered = doc.lower()

        self.assertIn("calc_alpha", doc)
        self.assertIn("calc_g_m", doc)
        self.assertIn("combustor geometry model", lowered)
        self.assertIn("performance map", lowered)
        self.assertIn("mass flow", lowered)
        self.assertIn("composition", lowered)
        self.assertIn("StationState", doc)
        self.assertIn("calculate_cycle_point", doc)


if __name__ == "__main__":
    unittest.main()
