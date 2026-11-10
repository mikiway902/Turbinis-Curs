"""Focused tests for the explicit 0D architecture reference cycles.

These tests prove architecture-specific shaft/flow/state closure, independent
formula parity for the selected variant, explicit failure domains, and that the
legacy cycle path is not rebaselined.  They do not claim physical validation of
any architecture, compressor map, blading, cooling system, gearbox, or
propeller.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine  # noqa: E402

from turbine_engine.architecture import Architecture  # noqa: E402
from turbine_engine.architecture_cycle import (  # noqa: E402
    ARCHITECTURE_CYCLE_LIMITATIONS,
    AlphaBasis,
    ArchitectureCycleError,
    ArchitectureCycleInput,
    ArchitectureCyclePhysicalError,
    ExternalCombustorClosure,
    ExternalCombustorState,
    LegacyCombustorClosure,
    PolynomialGasModel,
    calculate_architecture_cycle,
)
from turbine_engine.cycle import (  # noqa: E402
    CycleConfig,
    CycleState,
    calc_alpha,
    calc_compressor,
    calc_cp_kerosene,
    calc_eta_expansion,
    calc_g_m,
    calc_k_kerosene,
    calculate_cycle_point,
)
from turbine_engine.shaft import ShaftPowerBalance  # noqa: E402


GAS = PolynomialGasModel(
    R=287.4,
    l0=14.61,
    q_lhv_kj=43150.0,
    eta_combustion=0.98,
    T_ref=288.0,
    alpha_basis=AlphaBasis.FUEL_BASED,
)

TARGET = 110_000.0
GEARBOX = 0.98
P_T_OUT = 150_000.0


def request_for(
    architecture: Architecture,
    *,
    ratios: tuple[float, ...] | None = None,
    turbine_etas: tuple[float, ...] | None = None,
    combustor: LegacyCombustorClosure | ExternalCombustorClosure | None = None,
    gas_model: PolynomialGasModel = GAS,
    p_t_out: float = P_T_OUT,
    target: float = TARGET,
) -> ArchitectureCycleInput:
    if ratios is None:
        ratios = {
            Architecture.SINGLE_SHAFT_SINGLE_CC: (11.0,),
            Architecture.SINGLE_SHAFT_DUAL_CC: (
                math.sqrt(11.0),
                math.sqrt(11.0),
            ),
            Architecture.FREE_POWER_TURBINE: (8.0,),
        }[architecture]
    if turbine_etas is None:
        turbine_etas = {
            Architecture.SINGLE_SHAFT_SINGLE_CC: (0.90,),
            Architecture.SINGLE_SHAFT_DUAL_CC: (0.90,),
            Architecture.FREE_POWER_TURBINE: (0.90, 0.90),
        }[architecture]
    if combustor is None:
        combustor = LegacyCombustorClosure(1250.0, 0.96)
    return ArchitectureCycleInput(
        architecture=architecture,
        inlet_total_temperature=288.0,
        inlet_total_pressure=101_300.0,
        compressor_pressure_ratios=ratios,
        compressor_polytropic_efficiency=0.85,
        combustor=combustor,
        turbine_outlet_total_pressure=p_t_out,
        turbine_polytropic_efficiencies=turbine_etas,
        turbine_mechanical_efficiencies=(0.99,) * len(turbine_etas),
        target_gearbox_input_power=target,
        gearbox_efficiency=GEARBOX,
        gas_model=gas_model,
    )


def turbine_work(ratio: float, cp: float, T_in: float, k: float, eta: float) -> float:
    """Independent closed form of the retained polytropic expansion."""

    return cp * T_in * (1.0 - ratio ** (-(k - 1.0) * eta / k))


class AllArchitectureTests(unittest.TestCase):
    def test_all_three_architectures_close_target_and_mass_balance(self) -> None:
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                result = calculate_architecture_cycle(
                    request_for(architecture)
                )

                self.assertAlmostEqual(
                    result.actual_gearbox_input_power,
                    TARGET,
                    delta=1e-6,
                )
                self.assertAlmostEqual(
                    result.gearbox_output_power,
                    TARGET * GEARBOX,
                    delta=1e-6,
                )
                self.assertEqual(result.target_power_status, "met")
                self.assertAlmostEqual(
                    result.mass_balance_residual, 0.0, delta=1e-15
                )
                self.assertAlmostEqual(
                    result.compressor_pressure_residual, 0.0, delta=1e-15
                )
                for residual in result.turbine_pressure_residuals:
                    self.assertAlmostEqual(residual, 0.0, delta=1e-15)
                self.assertEqual(
                    result.gas_mass_flow,
                    result.air_mass_flow + result.fuel_mass_flow,
                )
                self.assertGreater(result.air_mass_flow, 0.0)
                self.assertGreater(result.fuel_mass_flow, 0.0)
                self.assertEqual(
                    result.compressor.inlet.mass_flow,
                    result.air_mass_flow,
                )
                self.assertEqual(
                    result.turbines[0].inlet.mass_flow,
                    result.gas_mass_flow,
                )

                for residual in result.residual_report().values():
                    self.assertTrue(math.isfinite(residual))

    def test_result_is_frozen_and_serializable(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        )

        with self.assertRaises(FrozenInstanceError):
            result.air_mass_flow = 123.0  # type: ignore[misc]

        payload = asdict(result)
        # JSON round-trip is the intended serialization contract: all fields are
        # scalars, immutable tuples, or nested dataclasses.
        self.assertIn(
            '"architecture": "SINGLE_SHAFT_DUAL_CC"',
            json.dumps(payload),
        )
        self.assertEqual(
            payload["architecture"], Architecture.SINGLE_SHAFT_DUAL_CC
        )
        self.assertEqual(
            payload["model_variant"],
            "uncooled_no_bleed_0d_ideal_mixing",
        )
        self.assertEqual(len(payload["turbines"]), 1)
        self.assertEqual(
            payload["gas_model"]["alpha_basis"], AlphaBasis.FUEL_BASED
        )
        self.assertIn(
            "NOT_PHYSICALLY_VALIDATED", result.validation_scope
        )
        self.assertTrue(
            any(
                marker.startswith("MISSING_COMPRESSOR_PHYSICS")
                for marker in result.validation_scope
            )
        )

    def test_result_rejects_forged_closure_fields(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        with self.assertRaises(ArchitectureCycleError):
            replace(result, gas_mass_flow=result.gas_mass_flow * 1.01)
        with self.assertRaises(ArchitectureCycleError):
            replace(
                result,
                turbine_pressure_residuals=(
                    result.turbine_pressure_residuals[0] + 1.0,
                ),
            )
        with self.assertRaises(ArchitectureCycleError):
            replace(result, actual_gearbox_input_power=0.0)

    def test_limitations_are_explicit(self) -> None:
        self.assertIn(
            "NOT_PHYSICALLY_VALIDATED", ARCHITECTURE_CYCLE_LIMITATIONS
        )
        self.assertTrue(
            any(
                marker.startswith("MISSING_COMPRESSOR_PHYSICS")
                for marker in ARCHITECTURE_CYCLE_LIMITATIONS
            )
        )
        self.assertTrue(
            any(
                marker.startswith("MISSING_GEARBOX_PROPELLER")
                for marker in ARCHITECTURE_CYCLE_LIMITATIONS
            )
        )


class SingleShaftTests(unittest.TestCase):
    def test_single_shaft_turbine_drives_compressor_plus_target(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        turbine = result.turbines[0]
        eta_mech = result.shaft.mechanical_efficiencies[0]

        independent_net = (
            eta_mech * result.gas_mass_flow * turbine.specific_work
            - result.air_mass_flow * result.compressor.specific_work
        )
        self.assertAlmostEqual(independent_net, TARGET, delta=1e-6)
        self.assertAlmostEqual(
            result.shaft.actual_gearbox_input_power,
            independent_net,
            delta=1e-6,
        )

    def test_dual_cc_uses_equal_ratios_with_actual_temperature_propagation(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        )
        first, second = result.compressor.station_states

        self.assertAlmostEqual(
            first.p_total / result.inlet.p_total, math.sqrt(11.0)
        )
        self.assertAlmostEqual(
            second.p_total / result.inlet.p_total, 11.0
        )
        self.assertEqual(first.p_total, result.inlet.p_total * math.sqrt(11.0))
        self.assertEqual(second.p_total, first.p_total * math.sqrt(11.0))
        self.assertGreater(first.T_total, result.inlet.T_total)
        self.assertGreater(second.T_total, first.T_total)
        self.assertEqual(result.pressure_ratio_product, 11.0)

        # The second unit is not restarted from ambient: its work and exit
        # temperature use the first unit's actual exit state.
        self.assertNotEqual(second.T_total, result.inlet.T_total)
        self.assertGreater(
            result.compressor.per_unit_specific_work[1],
            0.0,
        )

    def test_dual_cc_flow_differs_from_single_cc_at_equal_product(self) -> None:
        single = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        dual = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        )

        self.assertEqual(single.pressure_ratio_product, 11.0)
        self.assertEqual(dual.pressure_ratio_product, 11.0)
        self.assertNotAlmostEqual(
            single.air_mass_flow, dual.air_mass_flow, places=6
        )

    def test_single_shaft_closure_is_not_a_double_subtraction(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        eta = result.shaft.mechanical_efficiencies[0]
        turbine_power = result.turbines[0].power
        compressor_power = result.compressor.power
        correct = eta * turbine_power - compressor_power
        double = eta * turbine_power - 2.0 * compressor_power

        self.assertAlmostEqual(correct, TARGET, delta=1e-6)
        self.assertNotAlmostEqual(double, TARGET, places=3)
        self.assertAlmostEqual(
            result.shaft.actual_gearbox_input_power, correct, delta=1e-6
        )

    def test_compressor_turbine_formula_parity(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        turbine = result.turbines[0]
        cp = turbine.inlet.cp
        k = turbine.inlet.k
        ratio = turbine.inlet.p_total / turbine.outlet.p_total
        independent_work = turbine_work(
            ratio, cp, turbine.inlet.T_total, k, 0.90
        )

        self.assertAlmostEqual(
            turbine.specific_work / independent_work, 1.0, delta=1e-12
        )
        self.assertEqual(
            turbine.efficiency, calc_eta_expansion(ratio, k, 0.90)
        )
        self.assertAlmostEqual(
            turbine.outlet.T_total,
            turbine.inlet.T_total - independent_work / cp,
            delta=1e-9,
        )

        expected_air = TARGET / (
            0.99 * (1.0 + result.combustor.fuel_air_ratio) * independent_work
            - result.compressor.specific_work
        )
        self.assertAlmostEqual(
            result.air_mass_flow / expected_air, 1.0, delta=1e-12
        )


class FreePowerTurbineTests(unittest.TestCase):
    def test_gg_state_feeds_fpt_and_separate_shafts_close(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.FREE_POWER_TURBINE)
        )
        gg = result.turbines[0]
        fpt = result.turbines[1]
        eta_gg, eta_fpt = result.shaft.mechanical_efficiencies

        self.assertEqual(fpt.inlet, gg.outlet)
        self.assertLess(fpt.inlet.T_total, result.combustor.outlet.T_total)
        self.assertLess(fpt.inlet.p_total, result.combustor.outlet.p_total)
        self.assertAlmostEqual(
            fpt.outlet.p_total, P_T_OUT, delta=1e-6
        )
        self.assertAlmostEqual(
            fpt.inlet.p_total / fpt.outlet.p_total,
            fpt.pressure_ratio,
        )

        self.assertAlmostEqual(
            eta_gg * gg.power, result.compressor.power, delta=1e-6
        )
        self.assertAlmostEqual(
            eta_fpt * fpt.power, TARGET, delta=1e-6
        )
        self.assertAlmostEqual(
            result.shaft.gas_generator_residual, 0.0, delta=1e-6
        )
        self.assertAlmostEqual(
            result.shaft.free_power_turbine_residual, 0.0, delta=1e-6
        )

    def test_fpt_output_does_not_re_subtract_compressor_load(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.FREE_POWER_TURBINE)
        )
        fpt = result.turbines[1]
        eta_fpt = result.shaft.mechanical_efficiencies[1]
        not_double_subtracted = (
            eta_fpt * fpt.power - result.compressor.power
        )

        self.assertAlmostEqual(
            result.shaft.actual_gearbox_input_power, TARGET, delta=1e-6
        )
        self.assertNotAlmostEqual(
            result.shaft.actual_gearbox_input_power,
            not_double_subtracted,
            places=3,
        )
        self.assertAlmostEqual(
            result.gearbox_output_power, TARGET * GEARBOX, delta=1e-6
        )

    def test_gg_work_inverse_and_fpt_formula_parity(self) -> None:
        result = calculate_architecture_cycle(
            request_for(Architecture.FREE_POWER_TURBINE)
        )
        gg = result.turbines[0]
        fpt = result.turbines[1]
        eta_mech_gg, eta_mech_fpt = result.shaft.mechanical_efficiencies

        required_gg = result.compressor.specific_work / (
            eta_mech_gg * (1.0 + result.combustor.fuel_air_ratio)
        )
        self.assertAlmostEqual(
            gg.specific_work / required_gg, 1.0, delta=1e-12
        )

        independent_fpt = turbine_work(
            fpt.pressure_ratio,
            fpt.inlet.cp,
            fpt.inlet.T_total,
            fpt.inlet.k,
            fpt.eta_turbine_poly,
        )
        self.assertAlmostEqual(
            fpt.specific_work / independent_fpt, 1.0, delta=1e-12
        )
        expected_air = TARGET / (
            eta_mech_fpt
            * (1.0 + result.combustor.fuel_air_ratio)
            * independent_fpt
        )
        self.assertAlmostEqual(
            result.air_mass_flow / expected_air, 1.0, delta=1e-12
        )


class VariantAndLegacyTests(unittest.TestCase):
    def test_fuel_based_and_empirical_alpha_are_explicit(self) -> None:
        fuel_based = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        f = fuel_based.combustor.fuel_air_ratio
        self.assertAlmostEqual(
            fuel_based.combustor.alpha, 1.0 / (f * GAS.l0), delta=1e-15
        )
        self.assertEqual(
            fuel_based.combustor.alpha_basis,
            AlphaBasis.FUEL_BASED.value,
        )

        empirical_model = replace(
            GAS, alpha_basis=AlphaBasis.EMPIRICAL_POLYNOMIAL
        )
        empirical = calculate_architecture_cycle(
            request_for(
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                gas_model=empirical_model,
            )
        )
        expected_alpha = calc_alpha(
            empirical.compressor.outlet.T_total,
            empirical.combustor.outlet.T_total,
        )
        self.assertAlmostEqual(
            empirical.combustor.alpha, expected_alpha, delta=1e-12
        )
        self.assertEqual(
            empirical.combustor.alpha_basis,
            AlphaBasis.EMPIRICAL_POLYNOMIAL.value,
        )
        self.assertEqual(
            empirical.gas_model.alpha_basis,
            AlphaBasis.EMPIRICAL_POLYNOMIAL,
        )

    def test_external_combustor_state_is_used_without_inference(self) -> None:
        external = ExternalCombustorState(
            T_total=1200.0,
            p_total=900_000.0,
            fuel_air_ratio=0.017,
            cp=calc_cp_kerosene(1200.0, 4.0),
            k=calc_k_kerosene(1200.0, 4.0, GAS.R),
        )
        result = calculate_architecture_cycle(
            request_for(
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                combustor=ExternalCombustorClosure(external),
            )
        )

        self.assertEqual(result.combustor.closure, "external_state")
        self.assertIsNone(result.combustor.alpha)
        self.assertIsNone(result.combustor.heat_input)
        self.assertEqual(result.combustor.outlet.T_total, 1200.0)
        self.assertEqual(result.combustor.outlet.p_total, 900_000.0)
        self.assertEqual(result.combustor.fuel_air_ratio, 0.017)
        self.assertAlmostEqual(
            result.actual_gearbox_input_power, TARGET, delta=1e-6
        )

    def test_fuel_flow_matches_legacy_where_assumptions_are_identical(self) -> None:
        # Same compressor ratio and polynomial gas model, no bleed/cooling.
        result = calculate_architecture_cycle(
            request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        )
        legacy_config = CycleConfig(g_leak=0.0, g_cooling_off=0.0)
        legacy = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=TARGET), legacy_config
        )

        legacy_T_comp, _ = calc_compressor(11.0)
        self.assertAlmostEqual(
            result.compressor.outlet.T_total,
            legacy_T_comp,
            delta=1e-5,
        )
        expected_f = calc_g_m(
            result.compressor.outlet.T_total,
            1250.0,
            CycleConfig(
                R=GAS.R,
                l0=GAS.l0,
                q_lhv_kj=GAS.q_lhv_kj,
                eta_combustion=GAS.eta_combustion,
                T_ref=GAS.T_ref,
            ),
        )
        self.assertEqual(result.combustor.fuel_air_ratio, expected_f)
        self.assertAlmostEqual(
            result.gas_mass_flow / result.air_mass_flow,
            1.0 + expected_f,
            delta=1e-14,
        )
        self.assertAlmostEqual(
            result.gas_mass_flow / result.air_mass_flow,
            legacy.g_t,
            delta=1e-12,
        )

    def test_legacy_cycle_anchor_is_not_rebaselined(self) -> None:
        legacy = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=TARGET)
        )
        self.assertAlmostEqual(
            legacy.N_ud, 190_589.8677, delta=1e-4
        )
        self.assertAlmostEqual(
            legacy.G, 0.5771555504, delta=1e-10
        )


class RecomputeTests(unittest.TestCase):
    def test_parameter_changes_recompute_the_cycle(self) -> None:
        base = request_for(Architecture.SINGLE_SHAFT_DUAL_CC)
        first = calculate_architecture_cycle(base)
        changed = calculate_architecture_cycle(
            replace(
                base,
                compressor_pressure_ratios=(
                    math.sqrt(11.0),
                    math.sqrt(11.0) * 1.05,
                ),
            )
        )

        self.assertNotEqual(
            first.compressor.outlet.p_total,
            changed.compressor.outlet.p_total,
        )
        self.assertNotEqual(
            first.compressor.outlet.T_total,
            changed.compressor.outlet.T_total,
        )
        self.assertNotEqual(first.air_mass_flow, changed.air_mass_flow)
        self.assertEqual(changed.target_power_status, "met")

    def test_free_power_turbine_outlet_pressure_changes_flow(self) -> None:
        base = request_for(Architecture.FREE_POWER_TURBINE)
        first = calculate_architecture_cycle(base)
        changed = calculate_architecture_cycle(
            replace(base, turbine_outlet_total_pressure=P_T_OUT * 0.9)
        )

        self.assertNotEqual(
            first.turbines[1].pressure_ratio,
            changed.turbines[1].pressure_ratio,
        )
        self.assertNotEqual(first.air_mass_flow, changed.air_mass_flow)
        self.assertEqual(changed.target_power_status, "met")


class ValidationTests(unittest.TestCase):
    def test_wrong_component_counts_are_rejected(self) -> None:
        with self.assertRaisesRegex(
            ArchitectureCycleError, "compressor pressure ratio"
        ):
            request_for(
                Architecture.SINGLE_SHAFT_DUAL_CC, ratios=(11.0,)
            )
        with self.assertRaisesRegex(
            ArchitectureCycleError, "turbine polytropic efficiency"
        ):
            request_for(
                Architecture.FREE_POWER_TURBINE, turbine_etas=(0.9,)
            )
        with self.assertRaisesRegex(
            ArchitectureCycleError, "turbine mechanical efficiency"
        ):
            ArchitectureCycleInput(
                architecture=Architecture.SINGLE_SHAFT_SINGLE_CC,
                inlet_total_temperature=288.0,
                inlet_total_pressure=101_300.0,
                compressor_pressure_ratios=(11.0,),
                compressor_polytropic_efficiency=0.85,
                combustor=LegacyCombustorClosure(1250.0, 0.96),
                turbine_outlet_total_pressure=P_T_OUT,
                turbine_polytropic_efficiencies=(0.9,),
                turbine_mechanical_efficiencies=(0.99, 0.99),
                target_gearbox_input_power=TARGET,
                gearbox_efficiency=GEARBOX,
                gas_model=GAS,
            )

    def test_invalid_architecture_and_combustor_types_are_rejected(self) -> None:
        request = request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        with self.assertRaises(ArchitectureCycleError):
            replace(request, architecture="SINGLE_SHAFT_SINGLE_CC")
        with self.assertRaises(ArchitectureCycleError):
            replace(request, combustor="not a combustor")

    def test_invalid_numeric_inputs_are_rejected(self) -> None:
        for target in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(target=target):
                with self.assertRaises(ArchitectureCycleError):
                    request_for(
                        Architecture.SINGLE_SHAFT_SINGLE_CC,
                        target=target,
                    )
        for ratio in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ArchitectureCycleError):
                    request_for(
                        Architecture.SINGLE_SHAFT_SINGLE_CC,
                        ratios=(ratio,),
                    )

    def test_combustor_below_compressor_exit_is_physical_failure(self) -> None:
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "must exceed"
        ):
            calculate_architecture_cycle(
                request_for(
                    Architecture.SINGLE_SHAFT_SINGLE_CC,
                    combustor=LegacyCombustorClosure(600.0, 0.96),
                )
            )

    def test_main_turbine_with_no_pressure_drop_is_physical_failure(self) -> None:
        request = request_for(Architecture.SINGLE_SHAFT_SINGLE_CC)
        p_gas = request.inlet_total_pressure * 11.0 * 0.96
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "outlet total pressure"
        ):
            calculate_architecture_cycle(
                replace(request, turbine_outlet_total_pressure=p_gas)
            )

    def test_nonpositive_net_specific_power_is_physical_failure(self) -> None:
        request = request_for(
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            ratios=(30.0,),
        )
        p_gas = request.inlet_total_pressure * 30.0 * 0.96
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "net specific shaft power"
        ):
            calculate_architecture_cycle(
                replace(
                    request,
                    turbine_outlet_total_pressure=p_gas * 0.999,
                )
            )

    def test_fpt_with_no_pressure_drop_is_physical_failure(self) -> None:
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "outlet total pressure"
        ):
            calculate_architecture_cycle(
                request_for(
                    Architecture.FREE_POWER_TURBINE,
                    p_t_out=1_000_000.0,
                )
            )

    def test_external_state_without_heat_addition_is_rejected(self) -> None:
        external = ExternalCombustorState(
            T_total=1200.0,
            p_total=900_000.0,
            fuel_air_ratio=0.0,
            cp=calc_cp_kerosene(1200.0, 4.0),
            k=calc_k_kerosene(1200.0, 4.0, GAS.R),
        )
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "fuel/air ratio"
        ):
            calculate_architecture_cycle(
                request_for(
                    Architecture.SINGLE_SHAFT_SINGLE_CC,
                    combustor=ExternalCombustorClosure(external),
                )
            )

    def test_invalid_gas_model_values_are_rejected(self) -> None:
        for field, value in (
            ("R", 0.0),
            ("l0", 0.0),
            ("q_lhv_kj", 0.0),
            ("eta_combustion", 1.5),
            ("T_ref", 0.0),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ArchitectureCycleError):
                    replace(GAS, **{field: value})


if __name__ == "__main__":
    unittest.main()
