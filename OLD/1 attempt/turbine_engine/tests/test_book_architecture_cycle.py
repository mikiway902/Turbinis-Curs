"""Independent source checks for the opt-in textbook architecture cycles.

The tests use the public published true-``cp`` helper, closed-form process-mean
relations, and the actual component results.  They do not re-implement a cycle,
do not mock the whole calculation, and do not claim physical validation of any
compressor map, blading, cooling system, gearbox, propeller, or stage inventory.
The default legacy path is checked against an unchanged numeric snapshot.
"""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import asdict, replace
from unittest import mock

from turbine_engine.architecture import Architecture
from turbine_engine.architecture_cycle import (
    ARCHITECTURE_CYCLE_LIMITATIONS,
    ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS,
    ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
    ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    ARCHITECTURE_MODEL_VARIANT_LEGACY,
    ARCHITECTURE_MODEL_VARIANT_TEXTBOOK,
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
from turbine_engine.architecture_raw import (
    evaluate_architecture_payload,
    input_to_payload,
    payload_to_input,
)
from turbine_engine.cycle import (
    CycleConfig,
    calc_cp_kerosene,
    calc_eta_expansion,
    calc_g_m,
    calc_k_kerosene,
)
from turbine_engine.diagnostics import SolverDiagnosticsError
from turbine_engine.gas import GasPropertyError, calc_textbook_gas_cp_k

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
PI_VALUES = (3.0, 7.5, 11.0, 13.0)


def _ratios(architecture: Architecture, pressure_ratio: float) -> tuple[float, ...]:
    if architecture is Architecture.SINGLE_SHAFT_DUAL_CC:
        return (math.sqrt(pressure_ratio), math.sqrt(pressure_ratio))
    return (pressure_ratio,)


def textbook_request(
    architecture: Architecture,
    pressure_ratio: float,
    *,
    gas_model: PolynomialGasModel = GAS,
    combustor: LegacyCombustorClosure | ExternalCombustorClosure | None = None,
    p_t_out: float = P_T_OUT,
    target: float = TARGET,
) -> ArchitectureCycleInput:
    if combustor is None:
        combustor = LegacyCombustorClosure(1250.0, 0.96)
    etas = (
        (0.90, 0.90)
        if architecture is Architecture.FREE_POWER_TURBINE
        else (0.90,)
    )
    return ArchitectureCycleInput(
        architecture=architecture,
        inlet_total_temperature=288.0,
        inlet_total_pressure=101_300.0,
        compressor_pressure_ratios=_ratios(architecture, pressure_ratio),
        compressor_polytropic_efficiency=0.85,
        combustor=combustor,
        turbine_outlet_total_pressure=p_t_out,
        turbine_polytropic_efficiencies=etas,
        turbine_mechanical_efficiencies=(0.99,) * len(etas),
        target_gearbox_input_power=target,
        gearbox_efficiency=GEARBOX,
        gas_model=gas_model,
        gas_property_basis=ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
    )


def legacy_request(
    architecture: Architecture,
    *,
    combustor: LegacyCombustorClosure | ExternalCombustorClosure | None = None,
) -> ArchitectureCycleInput:
    """The historical default fixture (FPT compressor ratio 8, legacy basis)."""

    if combustor is None:
        combustor = LegacyCombustorClosure(1250.0, 0.96)
    ratios = {
        Architecture.SINGLE_SHAFT_SINGLE_CC: (11.0,),
        Architecture.SINGLE_SHAFT_DUAL_CC: (
            math.sqrt(11.0),
            math.sqrt(11.0),
        ),
        Architecture.FREE_POWER_TURBINE: (8.0,),
    }[architecture]
    etas = {
        Architecture.SINGLE_SHAFT_SINGLE_CC: (0.90,),
        Architecture.SINGLE_SHAFT_DUAL_CC: (0.90,),
        Architecture.FREE_POWER_TURBINE: (0.90, 0.90),
    }[architecture]
    return ArchitectureCycleInput(
        architecture=architecture,
        inlet_total_temperature=288.0,
        inlet_total_pressure=101_300.0,
        compressor_pressure_ratios=ratios,
        compressor_polytropic_efficiency=0.85,
        combustor=combustor,
        turbine_outlet_total_pressure=P_T_OUT,
        turbine_polytropic_efficiencies=etas,
        turbine_mechanical_efficiencies=(0.99,) * len(etas),
        target_gearbox_input_power=TARGET,
        gearbox_efficiency=GEARBOX,
        gas_model=GAS,
    )


def _true_cp_k(T: float, alpha: float) -> tuple[float, float]:
    return calc_textbook_gas_cp_k(T, alpha, GAS.R)


def compressor_reference(
    T_in: float,
    pressure_ratio: float,
    eta: float,
    k_guess: float,
) -> tuple[float, float, float, float]:
    """Independent literal copy of the retained textbook compressor iteration."""

    T_out = T_in * pressure_ratio ** (
        (k_guess - 1.0) / (k_guess * eta)
    )
    for _ in range(100):
        T_mean = 0.5 * (T_in + T_out)
        cp, k = _true_cp_k(T_mean, math.inf)
        proposed = T_in * pressure_ratio ** (
            (k - 1.0) / (k * eta)
        )
        if abs(proposed - T_out) < 1e-6:
            return T_out, T_mean, cp, k
        T_out = proposed
    raise AssertionError("compressor reference iteration did not close")


def forward_turbine_reference(
    T_in: float,
    pressure_ratio: float,
    eta_poly: float,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    """Independent source expression for one textbook forward expansion."""

    cp0, k0 = _true_cp_k(T_in, alpha)
    eta0 = calc_eta_expansion(pressure_ratio, k0, eta_poly)
    candidate = T_in * (
        1.0 - (1.0 - pressure_ratio ** ((1.0 - k0) / k0)) * eta0
    )
    for _ in range(100):
        T_mean = 0.5 * (T_in + candidate)
        cp, k = _true_cp_k(T_mean, alpha)
        eta = calc_eta_expansion(pressure_ratio, k, eta_poly)
        predicted = T_in * (
            1.0 - (1.0 - pressure_ratio ** ((1.0 - k) / k)) * eta
        )
        if abs(predicted - candidate) <= 1e-9:
            return (
                candidate,
                T_mean,
                cp,
                k,
                eta,
                cp * (T_in - candidate),
            )
        candidate = predicted
    raise AssertionError("turbine reference iteration did not close")


def inverse_turbine_reference(
    T_in: float,
    required: float,
    eta_poly: float,
    alpha: float,
) -> tuple[float, float, float, float, float, float]:
    """Independent source expression for one prescribed-work inverse."""

    cp_in, _ = _true_cp_k(T_in, alpha)
    candidate = T_in - required / cp_in
    for _ in range(100):
        T_mean = 0.5 * (T_in + candidate)
        cp, k = _true_cp_k(T_mean, alpha)
        predicted = T_in - required / cp
        achieved = cp * (T_in - candidate)
        relative = abs(achieved - required) / required
        if abs(predicted - candidate) <= 1e-9 and relative <= 1e-10:
            break
        candidate = predicted
    else:
        raise AssertionError("turbine inverse reference did not close")
    base = 1.0 - achieved / (cp * T_in)
    pressure_ratio = base ** (-k / ((k - 1.0) * eta_poly))
    return candidate, T_mean, cp, k, achieved, pressure_ratio


class BookBasisClosureTests(unittest.TestCase):
    def test_all_architectures_close_all_pi_fixtures(self) -> None:
        for architecture in Architecture:
            for pi in PI_VALUES:
                with self.subTest(architecture=architecture.value, pi=pi):
                    result = calculate_architecture_cycle(
                        textbook_request(architecture, pi)
                    )

                    self.assertEqual(
                        result.gas_property_basis,
                        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
                    )
                    self.assertEqual(
                        result.model_variant,
                        ARCHITECTURE_MODEL_VARIANT_TEXTBOOK,
                    )
                    self.assertEqual(
                        result.validation_scope,
                        ARCHITECTURE_CYCLE_TEXTBOOK_LIMITATIONS,
                    )
                    self.assertNotIn(
                        "constant",
                        " ".join(result.validation_scope).lower(),
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
                    self.assertTrue(
                        any(
                            marker.startswith("MISSING_TURBINE_STAGE_PHYSICS")
                            for marker in result.validation_scope
                        )
                    )
                    self.assertEqual(
                        result.compressor.gas_property_basis,
                        ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
                    )
                    for turbine in result.turbines:
                        self.assertEqual(
                            turbine.gas_property_basis,
                            ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
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
                    self.assertEqual(
                        result.gas_mass_flow,
                        result.air_mass_flow + result.fuel_mass_flow,
                    )


class SourceFormulaTests(unittest.TestCase):
    def test_fuel_balance_and_fuel_based_alpha_are_source_relations(self) -> None:
        for architecture in (
            Architecture.SINGLE_SHAFT_SINGLE_CC,
            Architecture.SINGLE_SHAFT_DUAL_CC,
            Architecture.FREE_POWER_TURBINE,
        ):
            with self.subTest(architecture=architecture.value):
                request = textbook_request(architecture, 11.0)
                result = calculate_architecture_cycle(request)
                expected_f = calc_g_m(
                    result.compressor.outlet.T_total,
                    request.combustor.outlet_total_temperature,
                    CycleConfig(
                        R=GAS.R,
                        l0=GAS.l0,
                        q_lhv_kj=GAS.q_lhv_kj,
                        eta_combustion=GAS.eta_combustion,
                        T_ref=GAS.T_ref,
                    ),
                )
                self.assertEqual(
                    result.combustor.fuel_air_ratio, expected_f
                )
                self.assertEqual(
                    result.combustor.alpha, 1.0 / (expected_f * GAS.l0)
                )
                self.assertEqual(
                    result.combustor.alpha_basis, AlphaBasis.FUEL_BASED.value
                )
                self.assertAlmostEqual(
                    result.gas_mass_flow / result.air_mass_flow,
                    1.0 + expected_f,
                    delta=1e-15,
                )

    def test_compressor_true_cp_train_iteration_matches_source(self) -> None:
        for architecture in Architecture:
            for pi in PI_VALUES:
                with self.subTest(architecture=architecture.value, pi=pi):
                    result = calculate_architecture_cycle(
                        textbook_request(architecture, pi)
                    )
                    _, k_guess = _true_cp_k(
                        result.inlet.T_total, math.inf
                    )
                    previous_T = result.inlet.T_total
                    self.assertEqual(
                        len(result.compressor.process_properties),
                        len(result.compressor.pressure_ratios),
                    )
                    for unit_state, process, ratio, inlet_state in zip(
                        result.compressor.station_states,
                        result.compressor.process_properties,
                        result.compressor.pressure_ratios,
                        (result.compressor.inlet,)
                        + result.compressor.station_states[:-1],
                    ):
                        self.assertEqual(process.T_in, previous_T)
                        self.assertEqual(process.T_in, inlet_state.T_total)
                        (
                            ref_T_out,
                            ref_T_mean,
                            ref_cp,
                            ref_k,
                        ) = compressor_reference(
                            process.T_in,
                            ratio,
                            process.eta_polytropic,
                            k_guess,
                        )
                        self.assertAlmostEqual(
                            process.T_out, ref_T_out, delta=1e-6
                        )
                        self.assertAlmostEqual(
                            process.T_mean, ref_T_mean, delta=1e-6
                        )
                        self.assertAlmostEqual(process.cp, ref_cp, delta=1e-3)
                        self.assertAlmostEqual(process.k, ref_k, delta=1e-9)
                        predicted = process.T_in * ratio ** (
                            (process.k - 1.0)
                            / (process.k * process.eta_polytropic)
                        )
                        self.assertAlmostEqual(
                            predicted,
                            process.T_out,
                            delta=process.tolerance_K + 1e-12,
                        )
                        local_cp, local_k = _true_cp_k(
                            process.T_out, math.inf
                        )
                        self.assertEqual(
                            (unit_state.cp, unit_state.k),
                            (local_cp, local_k),
                        )
                        previous_T = unit_state.T_total

    def test_turbine_process_means_and_endpoints_match_source(self) -> None:
        for architecture in Architecture:
            for pi in PI_VALUES:
                with self.subTest(architecture=architecture.value, pi=pi):
                    result = calculate_architecture_cycle(
                        textbook_request(architecture, pi)
                    )
                    alpha = result.combustor.alpha
                    self.assertIsNotNone(alpha)
                    for index, turbine in enumerate(result.turbines):
                        process = turbine.process_properties
                        self.assertIsNotNone(process)
                        self.assertEqual(process.T_in, turbine.inlet.T_total)
                        self.assertEqual(process.T_out, turbine.outlet.T_total)
                        self.assertEqual(
                            process.T_mean,
                            0.5 * (process.T_in + process.T_out),
                        )

                        if process.requested_specific_work is None:
                            (
                                ref_T_out,
                                ref_T_mean,
                                ref_cp,
                                ref_k,
                                ref_eta,
                                ref_work,
                            ) = forward_turbine_reference(
                                turbine.inlet.T_total,
                                turbine.pressure_ratio,
                                turbine.eta_turbine_poly,
                                alpha,
                            )
                            self.assertEqual(process.requested_specific_work, None)
                        else:
                            (
                                ref_T_out,
                                ref_T_mean,
                                ref_cp,
                                ref_k,
                                ref_work,
                                ref_ratio,
                            ) = inverse_turbine_reference(
                                turbine.inlet.T_total,
                                process.requested_specific_work,
                                turbine.eta_turbine_poly,
                                alpha,
                            )
                            self.assertAlmostEqual(
                                turbine.pressure_ratio,
                                ref_ratio,
                                delta=1e-12,
                            )
                            ref_eta = calc_eta_expansion(
                                turbine.pressure_ratio, ref_k,
                                turbine.eta_turbine_poly,
                            )

                        self.assertAlmostEqual(
                            process.T_out, ref_T_out, delta=1e-9
                        )
                        self.assertAlmostEqual(
                            process.T_mean, ref_T_mean, delta=1e-9
                        )
                        self.assertAlmostEqual(process.cp, ref_cp, delta=1e-9)
                        self.assertAlmostEqual(process.k, ref_k, delta=1e-12)
                        self.assertAlmostEqual(
                            turbine.efficiency, ref_eta, delta=1e-12
                        )
                        self.assertAlmostEqual(
                            turbine.specific_work, ref_work, delta=1e-9
                        )
                        local_cp, local_k = _true_cp_k(
                            turbine.outlet.T_total, alpha
                        )
                        self.assertEqual(
                            (turbine.outlet.cp, turbine.outlet.k),
                            (local_cp, local_k),
                        )
                        self.assertAlmostEqual(
                            process.cp * (process.T_in - process.T_out),
                            turbine.specific_work,
                            delta=1e-9,
                        )

    def test_shaft_relations_use_computed_powers_and_source_work(self) -> None:
        for architecture in Architecture:
            for pi in PI_VALUES:
                with self.subTest(architecture=architecture.value, pi=pi):
                    result = calculate_architecture_cycle(
                        textbook_request(architecture, pi)
                    )
                    compressor_work = sum(
                        process.cp * (process.T_out - process.T_in)
                        for process in result.compressor.process_properties
                    )
                    self.assertAlmostEqual(
                        compressor_work,
                        result.compressor.specific_work,
                        delta=1e-9,
                    )
                    turbine_work = tuple(
                        turbine.process_properties.cp
                        * (turbine.inlet.T_total - turbine.outlet.T_total)
                        for turbine in result.turbines
                    )
                    air = result.air_mass_flow
                    gas = result.gas_mass_flow
                    eta = result.shaft.mechanical_efficiencies

                    if architecture is Architecture.FREE_POWER_TURBINE:
                        gg_process = result.turbines[0].process_properties
                        achieved_gap = (
                            gg_process.achieved_specific_work
                            - gg_process.requested_specific_work
                        )
                        gg_theoretical = eta[0] * gas * achieved_gap
                        gg_residual = (
                            eta[0] * gas * turbine_work[0]
                            - air * compressor_work
                        )
                        self.assertAlmostEqual(
                            gg_residual,
                            gg_theoretical,
                            delta=1e-6,
                        )
                        self.assertAlmostEqual(
                            gg_residual,
                            0.0,
                            delta=1e-6 + abs(gg_theoretical),
                        )
                        fpt_power = eta[1] * gas * turbine_work[1]
                        self.assertAlmostEqual(
                            fpt_power, TARGET, delta=1e-6
                        )
                        self.assertAlmostEqual(
                            result.shaft.free_power_turbine_residual,
                            fpt_power - TARGET,
                            delta=1e-6,
                        )
                    else:
                        net_power = (
                            eta[0] * gas * turbine_work[0]
                            - air * compressor_work
                        )
                        self.assertAlmostEqual(
                            net_power, TARGET, delta=1e-6
                        )
                        self.assertAlmostEqual(
                            result.shaft.main_residual,
                            net_power - TARGET,
                            delta=1e-6,
                        )

    def test_main_and_fpt_pressure_boundaries_are_total(self) -> None:
        for pi in PI_VALUES:
            with self.subTest(pi=pi):
                single = calculate_architecture_cycle(
                    textbook_request(
                        Architecture.SINGLE_SHAFT_SINGLE_CC, pi
                    )
                )
                main = single.turbines[0]
                self.assertAlmostEqual(
                    main.pressure_ratio,
                    main.inlet.p_total / main.outlet.p_total,
                    delta=0.0,
                )
                self.assertAlmostEqual(
                    main.outlet.p_total, P_T_OUT, delta=1e-6
                )

                fpt = calculate_architecture_cycle(
                    textbook_request(
                        Architecture.FREE_POWER_TURBINE, pi
                    )
                )
                self.assertAlmostEqual(
                    fpt.turbines[1].outlet.p_total,
                    P_T_OUT,
                    delta=1e-6,
                )
                self.assertAlmostEqual(
                    fpt.turbines[1].pressure_ratio,
                    fpt.turbines[1].inlet.p_total
                    / fpt.turbines[1].outlet.p_total,
                    delta=0.0,
                )


class FreePowerStateTransferTests(unittest.TestCase):
    def test_fpt_reuses_actual_gg_outlet_and_evaluates_its_own_process(self) -> None:
        result = calculate_architecture_cycle(
            textbook_request(Architecture.FREE_POWER_TURBINE, 11.0)
        )
        gg, fpt = result.turbines
        alpha = result.combustor.alpha
        self.assertIsNotNone(alpha)

        self.assertIs(fpt.inlet, gg.outlet)
        self.assertEqual(fpt.inlet, gg.outlet)
        self.assertEqual(fpt.process_properties.T_in, gg.outlet.T_total)
        self.assertEqual(
            gg.process_properties.T_mean,
            0.5 * (gg.inlet.T_total + gg.outlet.T_total),
        )
        self.assertEqual(
            fpt.process_properties.T_mean,
            0.5 * (fpt.inlet.T_total + fpt.outlet.T_total),
        )
        self.assertNotEqual(
            gg.process_properties.T_mean,
            fpt.process_properties.T_mean,
        )
        self.assertNotEqual(
            gg.process_properties.cp,
            fpt.process_properties.cp,
        )

        gg_out_cp, gg_out_k = _true_cp_k(gg.outlet.T_total, alpha)
        self.assertEqual((gg.outlet.cp, gg.outlet.k), (gg_out_cp, gg_out_k))
        fpt_in_cp, fpt_in_k = _true_cp_k(fpt.inlet.T_total, alpha)
        self.assertEqual(
            (fpt.inlet.cp, fpt.inlet.k), (fpt_in_cp, fpt_in_k)
        )
        gg_mean_cp, gg_mean_k = _true_cp_k(
            gg.process_properties.T_mean, alpha
        )
        fpt_mean_cp, fpt_mean_k = _true_cp_k(
            fpt.process_properties.T_mean, alpha
        )
        self.assertEqual(
            (gg.process_properties.cp, gg.process_properties.k),
            (gg_mean_cp, gg_mean_k),
        )
        self.assertEqual(
            (fpt.process_properties.cp, fpt.process_properties.k),
            (fpt_mean_cp, fpt_mean_k),
        )


class ConfigurationErrorTests(unittest.TestCase):
    def test_textbook_basis_rejects_external_combustor_state(self) -> None:
        external = ExternalCombustorClosure(
            ExternalCombustorState(
                T_total=1200.0,
                p_total=900_000.0,
                fuel_air_ratio=0.017,
                cp=calc_cp_kerosene(1200.0, 4.0),
                k=calc_k_kerosene(1200.0, 4.0, GAS.R),
            )
        )
        with self.assertRaisesRegex(
            ArchitectureCycleError, "requires LegacyCombustorClosure"
        ):
            replace(
                textbook_request(
                    Architecture.SINGLE_SHAFT_SINGLE_CC, 11.0
                ),
                combustor=external,
            )

    def test_textbook_basis_rejects_empirical_alpha(self) -> None:
        empirical = replace(
            GAS, alpha_basis=AlphaBasis.EMPIRICAL_POLYNOMIAL
        )
        with self.assertRaisesRegex(
            ArchitectureCycleError, "FUEL_BASED"
        ):
            textbook_request(
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                11.0,
                gas_model=empirical,
            )

    def test_unknown_property_basis_is_explicit_configuration_error(self) -> None:
        request = textbook_request(
            Architecture.SINGLE_SHAFT_SINGLE_CC, 11.0
        )
        with self.assertRaisesRegex(
            ArchitectureCycleError, "unknown gas_property_basis"
        ):
            replace(request, gas_property_basis="constant_train_cp")
        with self.assertRaises(ArchitectureCycleError):
            replace(request, gas_property_basis=None)

    def test_source_property_domain_error_is_precise_not_physical(self) -> None:
        # cp - R becomes non-positive for this explicit gas constant, so the
        # real public source helper raises GasPropertyError at the boundary.
        broken = replace(GAS, R=5000.0)
        with self.assertRaises(ArchitectureCycleError) as raised:
            calculate_architecture_cycle(
                textbook_request(
                    Architecture.SINGLE_SHAFT_SINGLE_CC,
                    11.0,
                    gas_model=broken,
                )
            )
        self.assertIn(
            "textbook true-cp source properties", str(raised.exception)
        )
        self.assertNotIsInstance(
            raised.exception, ArchitectureCyclePhysicalError
        )
        self.assertNotIsInstance(raised.exception, GasPropertyError)

    def test_typed_solver_diagnostic_propagates_through_boundary(self) -> None:
        error = SolverDiagnosticsError(
            "source true-cp solve did not close",
            stage="test.architecture_cycle.textbook_boundary",
            reason="test_diagnostic",
            last_valid_state={"T": 288.0},
        )
        with mock.patch(
            "turbine_engine.gas.calc_textbook_gas_cp_k",
            side_effect=error,
        ):
            with self.assertRaises(SolverDiagnosticsError) as raised:
                calculate_architecture_cycle(
                    textbook_request(
                        Architecture.SINGLE_SHAFT_SINGLE_CC, 11.0
                    )
                )
        self.assertIs(raised.exception, error)
        self.assertNotIsInstance(
            raised.exception, ArchitectureCyclePhysicalError
        )

    def test_nonpositive_net_power_remains_a_physical_error(self) -> None:
        pi = 30.0
        p_gas = 101_300.0 * pi * 0.96
        with self.assertRaisesRegex(
            ArchitectureCyclePhysicalError, "net specific"
        ):
            calculate_architecture_cycle(
                textbook_request(
                    Architecture.SINGLE_SHAFT_SINGLE_CC,
                    pi,
                    p_t_out=p_gas * 0.999,
                )
            )


class LegacyCompatibilityTests(unittest.TestCase):
    def test_default_numeric_snapshot_is_unchanged(self) -> None:
        expected = {
            Architecture.SINGLE_SHAFT_SINGLE_CC: (
                0.7196344136616384,
                0.01239646702216575,
                110000.00000000009,
                152855.39144841465,
                507856.48072804546,
                358583.40802080656,
                643.7298018595287,
                784.8243481237943,
            ),
            Architecture.SINGLE_SHAFT_DUAL_CC: (
                0.7218377335113193,
                0.012425787462301514,
                109999.99999999994,
                152388.81938869308,
                507854.92760152585,
                359042.4021627994,
                644.1811369331543,
                784.8191457126092,
            ),
            Architecture.FREE_POWER_TURBINE: (
                0.7251232359722513,
                0.01372793138046237,
                109999.99999999999,
                151698.35214632322,
                290487.5681504785,
                293027.16765963455,
                579.1637752952781,
                984.4646426126983,
            ),
        }
        for architecture in Architecture:
            with self.subTest(architecture=architecture.value):
                result = calculate_architecture_cycle(
                    legacy_request(architecture)
                )
                self.assertEqual(
                    result.gas_property_basis,
                    ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
                )
                self.assertEqual(
                    result.model_variant,
                    ARCHITECTURE_MODEL_VARIANT_LEGACY,
                )
                self.assertEqual(
                    result.validation_scope,
                    ARCHITECTURE_CYCLE_LIMITATIONS,
                )
                self.assertIn(
                    "GAS_PROPERTY_BASIS: compressor mean-temperature variable "
                    "k; turbine constant cp/k at the turbine inlet total "
                    "temperature",
                    result.validation_scope,
                )
                self.assertEqual(
                    result.compressor.gas_property_basis,
                    ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
                )
                for turbine in result.turbines:
                    self.assertEqual(
                        turbine.gas_property_basis,
                        "legacy_constant_cp",
                    )
                self.assertEqual(
                    (
                        result.air_mass_flow,
                        result.fuel_mass_flow,
                        result.actual_gearbox_input_power,
                        result.net_specific_power_per_air,
                        result.turbines[0].specific_work,
                        result.compressor.specific_work,
                        result.compressor.outlet.T_total,
                        result.turbines[0].outlet.T_total,
                    ),
                    expected[architecture],
                )

    def test_legacy_external_combustor_state_is_unchanged(self) -> None:
        external = ExternalCombustorClosure(
            ExternalCombustorState(
                T_total=1200.0,
                p_total=900_000.0,
                fuel_air_ratio=0.017,
                cp=calc_cp_kerosene(1200.0, 4.0),
                k=calc_k_kerosene(1200.0, 4.0, GAS.R),
            )
        )
        result = calculate_architecture_cycle(
            legacy_request(
                Architecture.SINGLE_SHAFT_SINGLE_CC,
                combustor=external,
            )
        )
        self.assertEqual(
            result.gas_property_basis,
            ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertEqual(result.combustor.closure, "external_state")
        self.assertIsNone(result.combustor.alpha)
        self.assertAlmostEqual(
            result.actual_gearbox_input_power, TARGET, delta=1e-6
        )

    def test_existing_v1_adapter_ignores_the_new_default_field(self) -> None:
        request = legacy_request(
            Architecture.SINGLE_SHAFT_SINGLE_CC
        )
        payload = input_to_payload(request)
        self.assertNotIn("gas_property_basis", payload)
        rebuilt = payload_to_input(payload)
        self.assertEqual(
            rebuilt.gas_property_basis,
            ARCHITECTURE_GAS_PROPERTY_BASIS_LEGACY,
        )

        outcome = evaluate_architecture_payload(payload)
        self.assertIsNotNone(outcome.outputs)
        self.assertIsNotNone(outcome.intermediates)
        self.assertNotIn("gas_property_basis", outcome.outputs)
        self.assertNotIn("gas_property_basis", outcome.intermediates)
        self.assertEqual(
            outcome.outputs["model_variant"],
            ARCHITECTURE_MODEL_VARIANT_LEGACY,
        )

    def test_result_serialization_records_basis_without_mutating_model_label(
        self,
    ) -> None:
        labelled = replace(GAS, model_variant="caller_provenance_label")
        result = calculate_architecture_cycle(
            textbook_request(
                Architecture.SINGLE_SHAFT_DUAL_CC,
                11.0,
                gas_model=labelled,
            )
        )
        self.assertEqual(
            result.model_variant, ARCHITECTURE_MODEL_VARIANT_TEXTBOOK
        )
        self.assertEqual(
            result.gas_model.model_variant, "caller_provenance_label"
        )
        self.assertIs(result.gas_model, labelled)
        payload = asdict(result)
        self.assertEqual(
            payload["gas_property_basis"],
            ARCHITECTURE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(
            payload["gas_model"]["model_variant"],
            "caller_provenance_label",
        )
        # asdict is the agreed preliminary serialization surface; enum values
        # are str subclasses and tuples stay JSON-serializable.
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
