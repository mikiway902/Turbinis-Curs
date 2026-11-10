"""Focused tests for the 0D compressor unit/train component.

These tests prove equation provenance, anchor parity, pressure/mass closure,
sequential-train equivalence, validation, and immutability.  They do not claim
physical validation of a compressor map or aerodynamic stage model.
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
import turbine_engine.compressor as compressor_module  # noqa: E402

from turbine_engine.architecture import StationState  # noqa: E402
from turbine_engine.compressor import (  # noqa: E402
    CompressorResult,
    compressor_train,
    compressor_unit,
)
from turbine_engine.cycle import (  # noqa: E402
    CycleConfig,
    DEFAULT_CONFIG,
    calc_compressor,
    calc_cp_kerosene,
    calc_k_kerosene,
    calculate_cycle_point,
    CycleState,
)


def default_inlet(
    config: CycleConfig = DEFAULT_CONFIG,
    *,
    mass_flow: float = 0.5771555503613757,
) -> StationState:
    """Ambient total state consistent with the current cycle inlet."""

    return StationState(
        T_total=config.T_a_full,
        p_total=config.p_a_full,
        mass_flow=mass_flow,
        cp=calc_cp_kerosene(config.T_a_full, math.inf),
        k=calc_k_kerosene(config.T_a_full, math.inf, config.R),
    )


class RelativeAssertions(unittest.TestCase):
    def assertRelative(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual / expected, 1.0, delta=1e-12)


class CycleAnchorTests(RelativeAssertions):
    def test_direct_cycle_anchor_is_unchanged(self) -> None:
        result = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=110000.0), CycleConfig()
        )
        expected = {
            "G": 0.5771555503613757,
            "L_k": 358583.40802120743,
            "L_t": 552874.7217839112,
            "T_t": 730.1309202648472,
            "eta_t": 0.9246868857757602,
            "N_ud": 190589.86772478488,
            "N_reducer_out_ud": 186778.07037028918,
            "N_e_ud": 201275.23126840655,
            "C_e": 0.3180573704024953,
            "eta_e": 0.2623108391012963,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertRelative(getattr(result, field), value)

    def test_calc_compressor_reference_values_are_unchanged(self) -> None:
        T_k, k = calc_compressor(11.0)
        self.assertRelative(T_k, 643.7298018608001)
        self.assertRelative(k, 1.3988222193795008)

    def test_single_unit_matches_calc_compressor_exactly(self) -> None:
        inlet = default_inlet()
        for pi in (3.0, 6.0, 11.0, 13.0):
            with self.subTest(pi=pi):
                T_k, k = calc_compressor(pi, DEFAULT_CONFIG)
                result = compressor_unit(inlet, pi, DEFAULT_CONFIG)

                self.assertEqual(result.outlet.T_total, T_k)
                self.assertEqual(result.outlet.k, k)
                expected_work = (
                    DEFAULT_CONFIG.R
                    * k
                    / (k - 1.0)
                    * (T_k - DEFAULT_CONFIG.T_a_full)
                )
                self.assertEqual(
                    result.per_unit_specific_work[0], expected_work
                )
                self.assertEqual(result.specific_work, expected_work)

    def test_single_unit_matches_cycle_anchor_l_k(self) -> None:
        cycle = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=110000.0), CycleConfig()
        )
        result = compressor_unit(
            default_inlet(mass_flow=cycle.G), 11.0, CycleConfig()
        )
        self.assertRelative(result.specific_work, cycle.L_k)
        self.assertRelative(result.power, cycle.L_k * cycle.G)

    def test_arbitrary_inlet_uses_the_same_helper(self) -> None:
        T_in = 350.0
        T_k, k = calc_compressor(4.0, DEFAULT_CONFIG, T_in=T_in)
        inlet = StationState(
            T_total=T_in,
            p_total=250000.0,
            mass_flow=0.4,
            cp=calc_cp_kerosene(T_in, math.inf),
            k=calc_k_kerosene(T_in, math.inf, DEFAULT_CONFIG.R),
        )
        result = compressor_unit(inlet, 4.0)

        self.assertEqual(result.outlet.T_total, T_k)
        self.assertEqual(result.outlet.k, k)


class ClosureTests(unittest.TestCase):
    def test_pressure_and_mass_flow_closure(self) -> None:
        inlet = default_inlet()
        result = compressor_unit(inlet, 11.0)

        self.assertEqual(result.outlet.p_total, inlet.p_total * 11.0)
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertEqual(result.pressure_ratio_product, 11.0)
        self.assertEqual(
            result.residual,
            result.outlet.p_total / inlet.p_total
            - result.pressure_ratio_product,
        )
        self.assertEqual(
            result.outlet.cp,
            DEFAULT_CONFIG.R * result.outlet.k / (result.outlet.k - 1.0),
        )

    def test_specific_work_and_power_values(self) -> None:
        inlet = default_inlet(mass_flow=1.25)
        result = compressor_unit(inlet, 11.0)

        expected_unit_work = result.outlet.cp * (
            result.outlet.T_total - inlet.T_total
        )
        self.assertEqual(result.per_unit_specific_work[0], expected_unit_work)
        self.assertEqual(result.specific_work, expected_unit_work)
        self.assertEqual(result.power, result.specific_work * inlet.mass_flow)
        self.assertGreater(result.specific_work, 0.0)
        self.assertGreater(result.power, 0.0)


class TrainTests(unittest.TestCase):
    def test_multi_unit_train_equals_manual_chaining(self) -> None:
        inlet = default_inlet()
        ratios = (4.0, 2.75)
        train = compressor_train(inlet, ratios)

        first = compressor_unit(inlet, 4.0)
        second = compressor_unit(first.outlet, 2.75)

        self.assertEqual(train.station_states[0], first.outlet)
        self.assertEqual(train.station_states[1], second.outlet)
        self.assertEqual(train.outlet, second.outlet)
        self.assertEqual(
            train.per_unit_specific_work[0], first.per_unit_specific_work[0]
        )
        self.assertEqual(
            train.per_unit_specific_work[1], second.per_unit_specific_work[0]
        )
        self.assertEqual(
            train.specific_work, first.specific_work + second.specific_work
        )
        self.assertEqual(
            train.power, train.specific_work * inlet.mass_flow
        )

    def test_multi_unit_preserves_explicit_ratio_product(self) -> None:
        inlet = default_inlet()
        ratios = (4.0, 2.75)
        train = compressor_train(inlet, ratios)

        self.assertEqual(train.pressure_ratios, ratios)
        self.assertEqual(train.pressure_ratio_product, 4.0 * 2.75)
        self.assertEqual(train.outlet.p_total, inlet.p_total * 4.0 * 2.75)
        self.assertEqual(
            train.residual,
            train.outlet.p_total / inlet.p_total
            - train.pressure_ratio_product,
        )
        self.assertEqual(
            len(train.station_states), len(train.per_unit_specific_work)
        )
        self.assertEqual(len(train.station_states), len(ratios))

    def test_ratios_are_not_normalised_or_range_limited(self) -> None:
        inlet = default_inlet()
        for ratio in (1.0, 0.5, 100.0):
            with self.subTest(ratio=ratio):
                result = compressor_unit(inlet, ratio)
                self.assertEqual(result.pressure_ratios, (ratio,))
                self.assertEqual(result.outlet.p_total, inlet.p_total * ratio)

        split = compressor_train(inlet, (0.5, 22.0))
        self.assertEqual(split.pressure_ratios, (0.5, 22.0))
        self.assertEqual(split.pressure_ratio_product, 11.0)
        self.assertNotEqual(
            split.outlet.T_total,
            compressor_unit(inlet, 11.0).outlet.T_total,
            "per-unit mean-k treatment must not collapse into one unit",
        )


class ValidationTests(unittest.TestCase):
    def test_invalid_single_ratio_raises(self) -> None:
        inlet = default_inlet()
        for ratio in (0.0, -1.0, float("nan"), float("inf"), "bad"):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    compressor_unit(inlet, ratio)

    def test_invalid_ratio_iterables_raise(self) -> None:
        inlet = default_inlet()
        for ratios in ((), [], "11", 11.0, (11.0, 0.0), (11.0, float("inf"))):
            with self.subTest(ratios=ratios):
                with self.assertRaises((TypeError, ValueError)):
                    compressor_train(inlet, ratios)

    def test_invalid_inlet_or_config_raises(self) -> None:
        with self.assertRaises(TypeError):
            compressor_unit("not a state", 11.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            compressor_train(
                default_inlet(), (11.0,), config="not a config"  # type: ignore[arg-type]
            )

    def test_optional_inlet_temperature_is_validated(self) -> None:
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(T_in=value):
                with self.assertRaises(ValueError):
                    calc_compressor(11.0, T_in=value)


class ResultInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inlet = default_inlet()
        self.result = compressor_unit(self.inlet, 11.0)

    def test_mismatched_outlet_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outlet"):
            replace(self.result, outlet=default_inlet())

    def test_mismatched_mass_flow_is_rejected(self) -> None:
        bad_exit = StationState(
            T_total=self.result.outlet.T_total,
            p_total=self.result.outlet.p_total,
            mass_flow=self.result.outlet.mass_flow * 2.0,
            cp=self.result.outlet.cp,
            k=self.result.outlet.k,
        )
        with self.assertRaisesRegex(ValueError, "mass_flow"):
            replace(
                self.result,
                outlet=bad_exit,
                station_states=(bad_exit,),
            )

    def test_mismatched_unit_pressure_is_rejected(self) -> None:
        bad_exit = StationState(
            T_total=self.result.outlet.T_total,
            p_total=self.result.outlet.p_total * 2.0,
            mass_flow=self.result.outlet.mass_flow,
            cp=self.result.outlet.cp,
            k=self.result.outlet.k,
        )
        with self.assertRaisesRegex(ValueError, "p_total"):
            replace(
                self.result,
                outlet=bad_exit,
                station_states=(bad_exit,),
            )

    def test_mismatched_second_unit_pressure_is_rejected(self) -> None:
        train = compressor_train(self.inlet, (4.0, 2.75))
        bad_exit = StationState(
            T_total=train.outlet.T_total,
            p_total=train.outlet.p_total * 2.0,
            mass_flow=train.outlet.mass_flow,
            cp=train.outlet.cp,
            k=train.outlet.k,
        )
        with self.assertRaisesRegex(ValueError, r"station_states\[1\]\.p_total"):
            replace(
                train,
                outlet=bad_exit,
                station_states=(train.station_states[0], bad_exit),
            )

    def test_mismatched_pressure_ratio_product_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "pressure_ratio_product"):
            replace(
                self.result,
                pressure_ratio_product=(
                    self.result.pressure_ratio_product * 2.0
                ),
            )

    def test_mismatched_specific_work_sum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "specific_work"):
            replace(
                self.result,
                specific_work=self.result.specific_work + 1.0,
            )

    def test_mismatched_power_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "power"):
            replace(self.result, power=self.result.power + 1.0)

    def test_mismatched_residual_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "residual"):
            replace(self.result, residual=self.result.residual + 1.0)

    def test_mismatched_unit_work_is_rejected(self) -> None:
        bad_work = self.result.per_unit_specific_work[0] + 1.0
        with self.assertRaisesRegex(
            ValueError, r"per_unit_specific_work\[0\]"
        ):
            replace(
                self.result,
                per_unit_specific_work=(bad_work,),
                specific_work=bad_work,
                power=bad_work * self.inlet.mass_flow,
            )

    def test_mismatched_second_unit_work_is_rejected(self) -> None:
        train = compressor_train(self.inlet, (4.0, 2.75))
        bad_works = (
            train.per_unit_specific_work[0],
            train.per_unit_specific_work[1] + 1.0,
        )
        bad_specific_work = sum(bad_works)
        with self.assertRaisesRegex(
            ValueError, r"per_unit_specific_work\[1\]"
        ):
            replace(
                train,
                per_unit_specific_work=bad_works,
                specific_work=bad_specific_work,
                power=bad_specific_work * self.inlet.mass_flow,
            )

    def test_manual_constructor_rejects_contradictory_unit_work(self) -> None:
        inlet = StationState(
            T_total=300.0,
            p_total=0.1,
            mass_flow=1.0,
            cp=1005.0,
            k=1.4,
        )
        ratio = 3.0
        exit_state = StationState(
            T_total=400.0,
            p_total=inlet.p_total * ratio,
            mass_flow=inlet.mass_flow,
            cp=1005.0,
            k=1.4,
        )
        work = 123.0
        product = 1.0 * ratio
        residual = exit_state.p_total / inlet.p_total - product

        with self.assertRaisesRegex(
            ValueError, r"per_unit_specific_work\[0\]"
        ):
            CompressorResult(
                inlet=inlet,
                outlet=exit_state,
                station_states=(exit_state,),
                pressure_ratios=(ratio,),
                pressure_ratio_product=product,
                per_unit_specific_work=(work,),
                specific_work=sum((work,)),
                power=work * inlet.mass_flow,
                residual=residual,
            )

    def test_manual_constructor_rejects_inconsistent_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "pressure_ratio_product"):
            CompressorResult(
                inlet=self.result.inlet,
                outlet=self.result.outlet,
                station_states=self.result.station_states,
                pressure_ratios=self.result.pressure_ratios,
                pressure_ratio_product=(
                    self.result.pressure_ratio_product + 1.0
                ),
                per_unit_specific_work=self.result.per_unit_specific_work,
                specific_work=self.result.specific_work,
                power=self.result.power,
                residual=self.result.residual,
            )

    def test_nonzero_residual_matching_pressure_closure_is_kept(self) -> None:
        inlet = StationState(
            T_total=300.0,
            p_total=0.1,
            mass_flow=1.0,
            cp=1005.0,
            k=1.4,
        )
        ratio = 3.0
        exit_state = StationState(
            T_total=400.0,
            p_total=inlet.p_total * ratio,
            mass_flow=inlet.mass_flow,
            cp=1005.0,
            k=1.4,
        )
        work = exit_state.cp * (exit_state.T_total - inlet.T_total)
        product = 1.0 * ratio
        residual = exit_state.p_total / inlet.p_total - product
        self.assertNotEqual(residual, 0.0)

        result = CompressorResult(
            inlet=inlet,
            outlet=exit_state,
            station_states=(exit_state,),
            pressure_ratios=(ratio,),
            pressure_ratio_product=product,
            per_unit_specific_work=(work,),
            specific_work=sum((work,)),
            power=work * inlet.mass_flow,
            residual=residual,
        )

        self.assertEqual(result.residual, residual)
        self.assertEqual(
            result.residual,
            result.outlet.p_total / result.inlet.p_total
            - result.pressure_ratio_product,
        )


class ImmutabilityTests(unittest.TestCase):
    def test_calls_do_not_mutate_inputs(self) -> None:
        config = CycleConfig()
        inlet = default_inlet(config)
        ratios = [4.0, 2.75]
        inlet_before = asdict(inlet)
        config_before = asdict(config)
        ratios_before = list(ratios)

        compressor_train(inlet, ratios, config)

        self.assertEqual(asdict(inlet), inlet_before)
        self.assertEqual(asdict(config), config_before)
        self.assertEqual(ratios, ratios_before)

    def test_result_and_states_are_immutable(self) -> None:
        result = compressor_unit(default_inlet(), 11.0)

        self.assertIsInstance(result, CompressorResult)
        self.assertIsInstance(result.station_states, tuple)
        self.assertIsInstance(result.pressure_ratios, tuple)
        self.assertIsInstance(result.per_unit_specific_work, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.outlet = default_inlet()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.outlet.T_total = 1.0  # type: ignore[misc]

    def test_package_exports_the_component(self) -> None:
        self.assertIs(turbine_engine.CompressorResult, CompressorResult)
        self.assertIs(turbine_engine.compressor_unit, compressor_unit)
        self.assertIs(turbine_engine.compressor_train, compressor_train)


class DocumentationTests(unittest.TestCase):
    def test_module_documents_zero_dimensional_scope_and_provenance(self) -> None:
        doc = compressor_module.__doc__ or ""
        lowered = doc.lower()
        self.assertIn("0d", lowered)
        self.assertIn("not a compressor map", lowered)
        self.assertIn("calc_compressor", doc)
        self.assertIn("bladed", lowered)


if __name__ == "__main__":
    unittest.main()
