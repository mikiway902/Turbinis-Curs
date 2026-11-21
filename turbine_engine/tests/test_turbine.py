"""Focused tests for the 0D lumped turbine-expansion unit.

These tests prove equation provenance, pressure/mass/energy closure, validation,
and immutability.  They do not claim physical validation of a bladed turbine
stage, a stage inventory, or an operating map.
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine  # noqa: E402
import turbine_engine.cycle as cycle_module  # noqa: E402
import turbine_engine.turbine as turbine_module  # noqa: E402

from turbine_engine.architecture import StationState  # noqa: E402
from turbine_engine.cycle import (  # noqa: E402
    DEFAULT_CONFIG,
    CycleSolverError,
    calc_cp_kerosene,
    calc_eta_expansion,
    calc_k_kerosene,
)
from turbine_engine.turbine import (  # noqa: E402
    TurbineResult,
    turbine_expansion,
    turbine_pressure_ratio_for_work,
)


def gas_inlet(
    *,
    T_total: float = 1250.0,
    p_total: float = 300000.0,
    mass_flow: float = 0.7,
    alpha: float = 1.0,
) -> StationState:
    """One consistent inlet state using the existing gas-property helpers."""

    return StationState(
        T_total=T_total,
        p_total=p_total,
        mass_flow=mass_flow,
        cp=calc_cp_kerosene(T_total, alpha),
        k=calc_k_kerosene(T_total, alpha, DEFAULT_CONFIG.R),
    )


def direct_expected(
    inlet: StationState, pressure_ratio: float, eta_turbine_poly: float
) -> tuple[float, float, float, float]:
    """Transcribe the cycle expression independently of the component."""

    efficiency = calc_eta_expansion(
        pressure_ratio, inlet.k, eta_turbine_poly
    )
    specific_work = (
        inlet.cp
        * inlet.T_total
        * (1.0 - pressure_ratio ** ((1.0 - inlet.k) / inlet.k))
        * efficiency
    )
    outlet_temperature = inlet.T_total - specific_work / inlet.cp
    outlet_pressure = inlet.p_total / pressure_ratio
    return (
        efficiency,
        specific_work,
        outlet_temperature,
        outlet_pressure,
    )


class RelativeAssertions(unittest.TestCase):
    def assertRelative(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual / expected, 1.0, delta=1e-12)


class ProvenanceTests(RelativeAssertions):
    def test_unit_matches_direct_cycle_expression_and_eta_helper(self) -> None:
        inlet = gas_inlet()
        for ratio in (1.5, 2.5, 6.0):
            with self.subTest(ratio=ratio):
                result = turbine_expansion(inlet, ratio, 0.9)
                efficiency, work, temperature, pressure = direct_expected(
                    inlet, ratio, 0.9
                )

                self.assertEqual(result.efficiency, efficiency)
                self.assertRelative(result.specific_work, work)
                self.assertRelative(result.outlet.T_total, temperature)
                self.assertRelative(result.outlet.p_total, pressure)

    def test_default_anchor_values_match_transcribed_formula(self) -> None:
        inlet = gas_inlet(T_total=1250.0, alpha=1.0)
        ratio = 2.5
        eta_turbine_poly = DEFAULT_CONFIG.eta_turbine_poly
        result = turbine_expansion(inlet, ratio, eta_turbine_poly)

        efficiency = calc_eta_expansion(ratio, inlet.k, eta_turbine_poly)
        work = (
            inlet.cp
            * 1250.0
            * (1.0 - ratio ** ((1.0 - inlet.k) / inlet.k))
            * efficiency
        )

        self.assertEqual(result.efficiency, efficiency)
        self.assertRelative(result.specific_work, work)
        self.assertRelative(
            result.outlet.T_total, 1250.0 - work / inlet.cp
        )

    def test_efficiency_is_exactly_the_existing_helper(self) -> None:
        inlet = gas_inlet()
        for ratio in (0.5, 1.2, 3.0, 9.0):
            with self.subTest(ratio=ratio):
                result = turbine_expansion(inlet, ratio, 0.87)
                self.assertEqual(
                    result.efficiency,
                    calc_eta_expansion(ratio, inlet.k, 0.87),
                )


class ClosureTests(unittest.TestCase):
    def test_pressure_mass_and_gas_property_closure(self) -> None:
        inlet = gas_inlet(mass_flow=0.75)
        result = turbine_expansion(inlet, 3.0, 0.9)

        self.assertEqual(result.outlet.p_total, inlet.p_total / 3.0)
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertEqual(result.outlet.cp, inlet.cp)
        self.assertEqual(result.outlet.k, inlet.k)
        self.assertEqual(result.pressure_ratio, 3.0)
        self.assertEqual(
            result.residual,
            result.inlet.p_total / result.outlet.p_total
            - result.pressure_ratio,
        )

    def test_work_temperature_power_and_efficiency_closure(self) -> None:
        inlet = gas_inlet(mass_flow=1.25)
        result = turbine_expansion(inlet, 2.5, 0.9)

        self.assertEqual(
            result.specific_work,
            inlet.cp * (inlet.T_total - result.outlet.T_total),
        )
        self.assertEqual(
            result.power, result.specific_work * inlet.mass_flow
        )
        self.assertEqual(
            result.efficiency,
            calc_eta_expansion(2.5, inlet.k, 0.9),
        )
        self.assertGreater(result.specific_work, 0.0)
        self.assertGreater(result.power, 0.0)

    def test_outlet_temperature_follows_the_source_drop(self) -> None:
        inlet = gas_inlet()
        result = turbine_expansion(inlet, 2.5, 0.9)
        _, work, temperature, _ = direct_expected(inlet, 2.5, 0.9)

        self.assertAlmostEqual(
            result.outlet.T_total / temperature, 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            result.specific_work / work, 1.0, delta=1e-12
        )


class RatioAndInputCaseTests(unittest.TestCase):
    def test_ratio_greater_than_one_extracts_work(self) -> None:
        inlet = gas_inlet()
        result = turbine_expansion(inlet, 3.0, 0.9)

        self.assertGreater(result.specific_work, 0.0)
        self.assertGreater(result.power, 0.0)
        self.assertLess(result.outlet.T_total, inlet.T_total)
        self.assertLess(result.outlet.p_total, inlet.p_total)

    def test_ratio_less_than_one_is_kept_as_mathematical_branch(self) -> None:
        inlet = gas_inlet()
        result = turbine_expansion(inlet, 0.8, 0.9)

        self.assertLess(result.specific_work, 0.0)
        self.assertLess(result.power, 0.0)
        self.assertGreater(result.outlet.T_total, inlet.T_total)
        self.assertGreater(result.outlet.p_total, inlet.p_total)
        self.assertGreater(result.efficiency, 0.0)

    def test_ratio_exactly_one_is_rejected_by_eta_helper(self) -> None:
        with self.assertRaisesRegex(
            CycleSolverError, "eta expansion denominator"
        ):
            turbine_expansion(gas_inlet(), 1.0, 0.9)

    def test_ratio_is_not_range_limited(self) -> None:
        inlet = gas_inlet()
        for ratio in (0.2, 0.8, 1.2, 50.0):
            with self.subTest(ratio=ratio):
                result = turbine_expansion(inlet, ratio, 0.9)
                self.assertEqual(result.pressure_ratio, ratio)
                self.assertEqual(
                    result.outlet.p_total, inlet.p_total / ratio
                )

    def test_arbitrary_inlet_total_state_is_used(self) -> None:
        inlet = gas_inlet(
            T_total=850.0, p_total=450000.0, mass_flow=0.33, alpha=math.inf
        )
        result = turbine_expansion(inlet, 2.2, 0.88)
        efficiency, work, temperature, pressure = direct_expected(
            inlet, 2.2, 0.88
        )

        self.assertEqual(result.efficiency, efficiency)
        self.assertAlmostEqual(
            result.specific_work / work, 1.0, delta=1e-12
        )
        self.assertAlmostEqual(
            result.outlet.T_total / temperature, 1.0, delta=1e-12
        )
        self.assertEqual(result.outlet.p_total, pressure)
        self.assertEqual(result.outlet.cp, inlet.cp)
        self.assertEqual(result.outlet.k, inlet.k)


class InverseWorkTests(RelativeAssertions):
    def test_inverse_round_trips_the_forward_relation(self) -> None:
        inlet = gas_inlet()
        for ratio in (1.2, 1.5, 2.5, 6.0, 20.0):
            with self.subTest(ratio=ratio):
                forward = turbine_expansion(inlet, ratio, 0.9)
                inverse = turbine_pressure_ratio_for_work(
                    inlet, forward.specific_work, 0.9
                )

                self.assertRelative(inverse.pressure_ratio, ratio)
                self.assertRelative(
                    inverse.specific_work, forward.specific_work
                )
                self.assertRelative(
                    inverse.outlet.T_total, forward.outlet.T_total
                )
                self.assertRelative(
                    inverse.outlet.p_total, forward.outlet.p_total
                )

    def test_inverse_matches_the_closed_form_ratio(self) -> None:
        inlet = gas_inlet(mass_flow=0.9)
        required = 275_000.0
        eta = 0.88
        base = 1.0 - required / (inlet.cp * inlet.T_total)
        expected_ratio = base ** (
            -inlet.k / ((inlet.k - 1.0) * eta)
        )

        result = turbine_pressure_ratio_for_work(inlet, required, eta)

        self.assertRelative(result.pressure_ratio, expected_ratio)
        self.assertAlmostEqual(
            result.specific_work / required, 1.0, delta=1e-12
        )
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertEqual(result.outlet.cp, inlet.cp)
        self.assertEqual(result.outlet.k, inlet.k)

    def test_inverse_rejects_nonpositive_required_work(self) -> None:
        inlet = gas_inlet()
        for required in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(required=required):
                with self.assertRaises(ValueError):
                    turbine_pressure_ratio_for_work(
                        inlet, required, 0.9
                    )

    def test_inverse_rejects_work_above_available_enthalpy(self) -> None:
        inlet = gas_inlet()
        with self.assertRaisesRegex(
            ValueError, "exceeds the constant-property expansion capacity"
        ):
            turbine_pressure_ratio_for_work(
                inlet, inlet.cp * inlet.T_total, 0.9
            )

    def test_inverse_rejects_invalid_efficiency_and_inlet(self) -> None:
        inlet = gas_inlet()
        for eta in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(eta=eta):
                with self.assertRaises(ValueError):
                    turbine_pressure_ratio_for_work(
                        inlet, 100_000.0, eta
                    )
        with self.assertRaises(TypeError):
            turbine_pressure_ratio_for_work(
                "not a state", 100_000.0, 0.9  # type: ignore[arg-type]
            )


class ValidationTests(unittest.TestCase):
    def test_invalid_inlet_raises(self) -> None:
        with self.assertRaises(TypeError):
            turbine_expansion("not a state", 2.0, 0.9)  # type: ignore[arg-type]

    def test_invalid_pressure_ratio_raises(self) -> None:
        inlet = gas_inlet()
        for ratio in (
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            "bad",
            None,
        ):
            with self.subTest(ratio=ratio):
                with self.assertRaises((TypeError, ValueError)):
                    turbine_expansion(inlet, ratio, 0.9)  # type: ignore[arg-type]

    def test_invalid_polytropic_efficiency_raises(self) -> None:
        inlet = gas_inlet()
        for efficiency in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(efficiency=efficiency):
                with self.assertRaises(ValueError):
                    turbine_expansion(inlet, 2.0, efficiency)

    def test_nonphysical_outlet_pressure_is_rejected(self) -> None:
        tiny_pressure = gas_inlet(p_total=1e-308)
        with self.assertRaisesRegex(ValueError, "nonphysical outlet state"):
            turbine_expansion(tiny_pressure, 1e308, 0.9)

        with self.assertRaisesRegex(ValueError, "nonphysical outlet state"):
            turbine_expansion(gas_inlet(), 5e-324, 0.9)

    def test_nonphysical_work_is_rejected(self) -> None:
        overflowing = StationState(
            T_total=1e308,
            p_total=300000.0,
            mass_flow=0.7,
            cp=1e308,
            k=1.35,
        )

        with self.assertRaisesRegex(ValueError, "non-finite work/drop"):
            turbine_expansion(overflowing, 2.0, 0.9)


class ResultInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inlet = gas_inlet()
        self.result = turbine_expansion(self.inlet, 2.5, 0.9)

    def test_mismatched_outlet_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outlet"):
            replace(self.result, outlet=gas_inlet())

    def test_mismatched_mass_flow_is_rejected(self) -> None:
        bad_exit = replace(
            self.result.outlet,
            mass_flow=self.result.outlet.mass_flow * 2.0,
        )
        with self.assertRaisesRegex(ValueError, "mass_flow"):
            replace(self.result, outlet=bad_exit)

    def test_mismatched_cp_is_rejected(self) -> None:
        bad_exit = replace(
            self.result.outlet, cp=self.result.outlet.cp * 1.01
        )
        with self.assertRaisesRegex(ValueError, "cp"):
            replace(self.result, outlet=bad_exit)

    def test_mismatched_k_is_rejected(self) -> None:
        bad_exit = replace(self.result.outlet, k=self.result.outlet.k * 1.01)
        with self.assertRaisesRegex(ValueError, "k"):
            replace(self.result, outlet=bad_exit)

    def test_mismatched_pressure_is_rejected(self) -> None:
        bad_exit = replace(
            self.result.outlet,
            p_total=self.result.outlet.p_total * 1.01,
        )
        with self.assertRaisesRegex(ValueError, "p_total"):
            replace(self.result, outlet=bad_exit)

    def test_coordinated_outlet_temperature_corruption_is_rejected(self) -> None:
        bad_exit = replace(self.result.outlet, T_total=self.inlet.T_total)
        bad_work = self.inlet.cp * (
            self.inlet.T_total - bad_exit.T_total
        )

        with self.assertRaisesRegex(ValueError, "outlet.T_total"):
            replace(
                self.result,
                outlet=bad_exit,
                specific_work=bad_work,
                power=bad_work * self.inlet.mass_flow,
            )

    def test_generated_results_in_both_branches_remain_valid(self) -> None:
        for ratio in (0.8, 2.5):
            with self.subTest(ratio=ratio):
                result = turbine_expansion(self.inlet, ratio, 0.9)
                self.assertEqual(replace(result), result)

    def test_mismatched_specific_work_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "specific_work"):
            replace(
                self.result,
                specific_work=self.result.specific_work + 1.0,
            )

    def test_mismatched_power_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "power"):
            replace(self.result, power=self.result.power + 1.0)

    def test_mismatched_efficiency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "efficiency"):
            replace(
                self.result, efficiency=self.result.efficiency + 1e-6
            )

    def test_mismatched_residual_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "residual"):
            replace(self.result, residual=self.result.residual + 1.0)

    def test_mismatched_pressure_ratio_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pressure_ratio"):
            replace(
                self.result,
                pressure_ratio=self.result.pressure_ratio * 1.01,
            )

    def test_mismatched_eta_turbine_poly_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "efficiency"):
            replace(self.result, eta_turbine_poly=0.8)

    def test_manual_constructor_rejects_contradictory_work(self) -> None:
        ratio = 2.5
        efficiency = calc_eta_expansion(ratio, self.inlet.k, 0.9)
        _, work, temperature, _ = direct_expected(self.inlet, ratio, 0.9)
        exit_state = StationState(
            T_total=temperature,
            p_total=self.inlet.p_total / ratio,
            mass_flow=self.inlet.mass_flow,
            cp=self.inlet.cp,
            k=self.inlet.k,
        )
        bad_work = work + 1.0

        with self.assertRaisesRegex(ValueError, "specific_work"):
            TurbineResult(
                inlet=self.inlet,
                outlet=exit_state,
                pressure_ratio=ratio,
                eta_turbine_poly=0.9,
                specific_work=bad_work,
                power=bad_work * self.inlet.mass_flow,
                efficiency=efficiency,
                residual=(
                    self.inlet.p_total / exit_state.p_total - ratio
                ),
            )

    def test_nonzero_residual_matching_pressure_closure_is_kept(self) -> None:
        inlet = gas_inlet(p_total=1.0)
        ratio = 3.7
        efficiency = calc_eta_expansion(ratio, inlet.k, 0.9)
        _, _, temperature, pressure = direct_expected(inlet, ratio, 0.9)
        exit_state = StationState(
            T_total=temperature,
            p_total=pressure,
            mass_flow=inlet.mass_flow,
            cp=inlet.cp,
            k=inlet.k,
        )
        work = inlet.cp * (inlet.T_total - exit_state.T_total)
        residual = inlet.p_total / exit_state.p_total - ratio
        self.assertNotEqual(residual, 0.0)

        result = TurbineResult(
            inlet=inlet,
            outlet=exit_state,
            pressure_ratio=ratio,
            eta_turbine_poly=0.9,
            specific_work=work,
            power=work * inlet.mass_flow,
            efficiency=efficiency,
            residual=residual,
        )

        self.assertEqual(result.residual, residual)
        self.assertEqual(
            result.residual,
            result.inlet.p_total / result.outlet.p_total
            - result.pressure_ratio,
        )

    def test_ratio_one_result_cannot_be_constructed(self) -> None:
        with self.assertRaises(CycleSolverError):
            TurbineResult(
                inlet=self.inlet,
                outlet=self.inlet,
                pressure_ratio=1.0,
                eta_turbine_poly=0.9,
                specific_work=0.0,
                power=0.0,
                efficiency=0.0,
                residual=0.0,
            )


class ImmutabilityTests(unittest.TestCase):
    def test_calls_do_not_mutate_inputs(self) -> None:
        inlet = gas_inlet()
        inlet_before = asdict(inlet)
        efficiency = 0.9

        turbine_expansion(inlet, 2.5, efficiency)

        self.assertEqual(asdict(inlet), inlet_before)
        self.assertEqual(efficiency, 0.9)

    def test_result_and_outlet_are_immutable(self) -> None:
        result = turbine_expansion(gas_inlet(), 2.5, 0.9)

        self.assertIsInstance(result, TurbineResult)
        with self.assertRaises(FrozenInstanceError):
            result.specific_work = 1.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.outlet.T_total = 1.0  # type: ignore[misc]

    def test_package_exports_the_component(self) -> None:
        self.assertIs(turbine_engine.TurbineResult, TurbineResult)
        self.assertIs(turbine_engine.turbine_expansion, turbine_expansion)


class DocumentationAndScopeTests(unittest.TestCase):
    def test_module_documents_scope_and_provenance(self) -> None:
        doc = turbine_module.__doc__ or ""
        lowered = doc.lower()

        self.assertIn("0d", lowered)
        self.assertIn("not a bladed", lowered)
        self.assertIn("calc_eta_expansion", doc)
        self.assertIn("pressure_ratio", doc)
        self.assertIn("inventory", lowered)
        self.assertIn("calculate_cycle_point", doc)

    def test_component_is_not_wired_into_the_cycle_module(self) -> None:
        source = Path(cycle_module.__file__).read_text(encoding="utf-8")
        self.assertNotIn("turbine_engine.turbine", source)
        self.assertFalse(hasattr(cycle_module, "turbine_expansion"))


if __name__ == "__main__":
    unittest.main()
