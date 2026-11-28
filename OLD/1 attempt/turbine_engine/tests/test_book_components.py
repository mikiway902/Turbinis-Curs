"""Focused tests for opt-in textbook process-mean component properties.

The expected values are transcribed independently from the published true-cp
correlation and the retained component equations.  They do not claim physical
validation, a map, a stage inventory, or a new architecture cycle.
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine.compressor as compressor_module  # noqa: E402
import turbine_engine.turbine as turbine_module  # noqa: E402

from turbine_engine.architecture import StationState  # noqa: E402
from turbine_engine.architecture_raw import (  # noqa: E402
    _compressor_to_json,
    _turbine_to_json,
)
from turbine_engine.compressor import (  # noqa: E402
    COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY,
    COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
    compressor_train,
    compressor_unit,
)
from turbine_engine.cycle import (  # noqa: E402
    DEFAULT_CONFIG,
    calc_compressor,
    calc_cp_kerosene,
    calc_k_kerosene,
)
from turbine_engine.diagnostics import NumericalConvergenceError  # noqa: E402
from turbine_engine.gas import calc_textbook_gas_cp_k as production_cp_k  # noqa: E402
from turbine_engine.turbine import (  # noqa: E402
    TEXTBOOK_TEMPERATURE_TOL_K,
    TEXTBOOK_WORK_RELATIVE_TOL,
    TURBINE_GAS_PROPERTY_BASIS_LEGACY,
    TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
    turbine_expansion,
    turbine_pressure_ratio_for_work,
)

# Published Appendix 6 true-cp fit, transcribed locally for expectations.
TEXTBOOK_CP_KCAL_TO_J = 4187.0
TEXTBOOK_CP_BRANCH_T_K = 750.0


def source_cp_k(
    T: float,
    alpha: float = math.inf,
    R: float = DEFAULT_CONFIG.R,
) -> tuple[float, float]:
    """Independent local transcription of the published true-cp fit."""

    if T < TEXTBOOK_CP_BRANCH_T_K:
        cp_kcal = (
            (0.0174 / alpha + 0.2407)
            + (0.0093 / alpha + 0.0193) * (0.0025 * T - 0.875)
            + (0.002 - 0.001056 / (alpha - 0.2))
            * (0.000025 * T * T - 0.0275 * T + 6.5625)
        )
    else:
        cp_kcal = (
            (0.0267 / alpha + 0.26)
            + (0.0133 / alpha + 0.032) * (0.001176 * T - 0.88235)
            - (0.00374 + 0.0094 / (alpha * alpha + 10.0))
            * (0.0000055556 * T * T - 0.013056 * T + 6.67)
        )
    cp = cp_kcal * TEXTBOOK_CP_KCAL_TO_J
    return cp, cp / (cp - R)


def source_eta_expansion(pi: float, k: float, eta_poly: float) -> float:
    """Independent local transcription of the retained efficiency expression."""

    return (
        1.0 - pi ** ((1.0 - k) * eta_poly / k)
    ) / (1.0 - pi ** ((1.0 - k) / k))


def fixture_cp_k(
    T: float,
    alpha: float = math.inf,
) -> tuple[float, float]:
    """Production helper used only to construct realistic fixture states."""

    return production_cp_k(T, alpha, DEFAULT_CONFIG.R)


def ambient_inlet(mass_flow: float = 0.5771555503613757) -> StationState:
    return StationState(
        T_total=DEFAULT_CONFIG.T_a_full,
        p_total=DEFAULT_CONFIG.p_a_full,
        mass_flow=mass_flow,
        cp=calc_cp_kerosene(DEFAULT_CONFIG.T_a_full, math.inf),
        k=calc_k_kerosene(DEFAULT_CONFIG.T_a_full, math.inf, DEFAULT_CONFIG.R),
    )


def true_gas_inlet(
    T_total: float = 1250.0,
    p_total: float = 300000.0,
    mass_flow: float = 0.7,
    alpha: float = 1.0,
) -> StationState:
    cp, k = fixture_cp_k(T_total, alpha)
    return StationState(
        T_total=T_total,
        p_total=p_total,
        mass_flow=mass_flow,
        cp=cp,
        k=k,
    )


def expected_compressor_state(
    T_in: float, ratio: float, tol: float = 1e-6
) -> tuple[float, float, float]:
    """Transcribe the published true-cp compressor fixed point independently."""

    T_out = T_in * ratio ** (
        (DEFAULT_CONFIG.k_air - 1.0)
        / (DEFAULT_CONFIG.k_air * DEFAULT_CONFIG.eta_compressor_poly)
    )
    for _ in range(100):
        _, k = source_cp_k(0.5 * (T_in + T_out))
        T_new = T_in * ratio ** (
            (k - 1.0) / (k * DEFAULT_CONFIG.eta_compressor_poly)
        )
        if abs(T_new - T_out) < tol:
            T_out = T_new
            break
        T_out = T_new
    cp, k = source_cp_k(0.5 * (T_in + T_out))
    return T_out, cp, k


def expected_forward_state(
    T_in: float, ratio: float, eta_poly: float, alpha: float
) -> tuple[float, float, float, float]:
    """Transcribe the process-mean turbine fixed point independently."""

    _, k = source_cp_k(T_in, alpha)
    eta = source_eta_expansion(ratio, k, eta_poly)
    T_out = T_in * (1.0 - (1.0 - ratio ** ((1.0 - k) / k)) * eta)
    for _ in range(200):
        cp, k = source_cp_k(0.5 * (T_in + T_out), alpha)
        eta = source_eta_expansion(ratio, k, eta_poly)
        predicted = T_in * (
            1.0 - (1.0 - ratio ** ((1.0 - k) / k)) * eta
        )
        if abs(predicted - T_out) <= TEXTBOOK_TEMPERATURE_TOL_K:
            break
        T_out = predicted
    cp, k = source_cp_k(0.5 * (T_in + T_out), alpha)
    eta = source_eta_expansion(ratio, k, eta_poly)
    return T_out, cp, k, eta


def expected_inverse_ratio(
    T_in: float, required: float, eta_poly: float, alpha: float
) -> tuple[float, float, float]:
    """Transcribe the process-mean prescribed-work fixed point independently."""

    cp, k = source_cp_k(T_in, alpha)
    T_out = T_in - required / cp
    for _ in range(200):
        cp, k = source_cp_k(0.5 * (T_in + T_out), alpha)
        predicted = T_in - required / cp
        if abs(predicted - T_out) <= TEXTBOOK_TEMPERATURE_TOL_K:
            break
        T_out = predicted
    cp, k = source_cp_k(0.5 * (T_in + T_out), alpha)
    base = 1.0 - required / (cp * T_in)
    ratio = base ** (-k / ((k - 1.0) * eta_poly))
    return ratio, T_out, k


class CompressorTextbookTests(unittest.TestCase):
    def test_default_basis_and_metadata_are_unchanged(self) -> None:
        inlet = ambient_inlet()
        result = compressor_unit(inlet, 11.0)

        self.assertEqual(
            result.gas_property_basis, COMPRESSOR_GAS_PROPERTY_BASIS_LEGACY
        )
        self.assertEqual(result.process_properties, ())

        T_k, k = calc_compressor(11.0)
        self.assertEqual(result.outlet.T_total, T_k)
        self.assertEqual(result.outlet.k, k)
        self.assertEqual(result.outlet.cp, DEFAULT_CONFIG.R * k / (k - 1.0))

    def test_textbook_unit_uses_process_mean_and_local_exit(self) -> None:
        inlet = ambient_inlet()
        result = compressor_unit(
            inlet,
            11.0,
            gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        record = result.process_properties[0]

        expected_T, expected_cp, expected_k = expected_compressor_state(
            inlet.T_total, 11.0
        )
        self.assertAlmostEqual(
            result.outlet.T_total, expected_T, delta=1e-6
        )
        self.assertEqual(record.T_mean, 0.5 * (inlet.T_total + result.outlet.T_total))
        self.assertAlmostEqual(record.cp, expected_cp, delta=1e-6)
        self.assertAlmostEqual(record.k, expected_k, delta=1e-9)
        self.assertEqual(
            result.per_unit_specific_work[0],
            record.cp * (result.outlet.T_total - inlet.T_total),
        )
        self.assertLessEqual(
            abs(record.fixed_point_residual_K), record.tolerance_K
        )

        local_cp, local_k = source_cp_k(result.outlet.T_total)
        self.assertEqual(result.outlet.cp, local_cp)
        self.assertEqual(result.outlet.k, local_k)
        self.assertNotEqual(result.outlet.cp, record.cp)

    def test_textbook_compressor_mean_above_750_uses_upper_branch(self) -> None:
        cp, k = fixture_cp_k(700.0)
        inlet = StationState(
            T_total=700.0,
            p_total=200000.0,
            mass_flow=0.5,
            cp=cp,
            k=k,
        )
        result = compressor_unit(
            inlet,
            4.0,
            gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        record = result.process_properties[0]

        self.assertGreater(record.T_mean, TEXTBOOK_CP_BRANCH_T_K)
        expected_T, expected_cp, expected_k = expected_compressor_state(
            700.0, 4.0
        )
        self.assertAlmostEqual(result.outlet.T_total, expected_T, delta=1e-6)
        self.assertAlmostEqual(record.cp, expected_cp, delta=1e-6)
        self.assertAlmostEqual(record.k, expected_k, delta=1e-9)
        self.assertEqual(result.outlet.cp, source_cp_k(result.outlet.T_total)[0])

    def test_train_units_have_independent_process_means(self) -> None:
        inlet = ambient_inlet()
        ratios = (3.0, 2.2)
        result = compressor_train(
            inlet,
            ratios,
            gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
        )

        self.assertEqual(len(result.process_properties), 2)
        previous_temperature = inlet.T_total
        for state, record, work in zip(
            result.station_states,
            result.process_properties,
            result.per_unit_specific_work,
        ):
            self.assertEqual(record.T_in, previous_temperature)
            self.assertEqual(record.T_out, state.T_total)
            self.assertEqual(record.T_mean, 0.5 * (record.T_in + record.T_out))
            self.assertEqual(work, record.cp * (state.T_total - record.T_in))
            self.assertEqual(state.cp, source_cp_k(state.T_total)[0])
            previous_temperature = state.T_total

        self.assertNotEqual(
            result.process_properties[0].T_mean,
            result.process_properties[1].T_mean,
        )
        self.assertEqual(result.outlet, result.station_states[-1])
        self.assertEqual(result.power, result.specific_work * inlet.mass_flow)

    def test_zero_compressor_residual_is_rejected(self) -> None:
        result = compressor_unit(
            ambient_inlet(),
            11.0,
            gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        with self.assertRaisesRegex(ValueError, "fixed_point_residual_K"):
            replace(
                result.process_properties[0],
                fixed_point_residual_K=0.0,
            )

    def test_inlet_is_not_rewritten(self) -> None:
        inlet = ambient_inlet()
        before = asdict(inlet)
        compressor_unit(
            inlet,
            4.0,
            gas_property_basis=COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK,
        )
        self.assertEqual(asdict(inlet), before)

    def test_invalid_basis_is_rejected(self) -> None:
        inlet = ambient_inlet()
        for basis in ("bad", "", None, 1):
            with self.subTest(basis=basis):
                with self.assertRaises(ValueError):
                    compressor_unit(inlet, 4.0, gas_property_basis=basis)

    def test_forced_compressor_nonconvergence_is_typed(self) -> None:
        inlet = ambient_inlet()
        with mock.patch.object(compressor_module, "SOLVER_MAX_ITER", 1):
            with self.assertRaises(NumericalConvergenceError) as caught:
                compressor_unit(
                    inlet,
                    11.0,
                    gas_property_basis=(
                        COMPRESSOR_GAS_PROPERTY_BASIS_TEXTBOOK
                    ),
                )
        state = caught.exception.last_valid_state
        self.assertIn("evaluated_temperature_K", state)
        self.assertIn("proposed_temperature_K", state)
        self.assertIn("residual_K", state)


class TurbineTextbookTests(unittest.TestCase):
    def test_default_basis_and_metadata_are_unchanged(self) -> None:
        inlet = true_gas_inlet()
        result = turbine_expansion(inlet, 2.5, 0.9)

        self.assertEqual(
            result.gas_property_basis, TURBINE_GAS_PROPERTY_BASIS_LEGACY
        )
        self.assertIsNone(result.process_properties)
        self.assertEqual(result.outlet.cp, inlet.cp)
        self.assertEqual(result.outlet.k, inlet.k)
        self.assertEqual(
            result.specific_work,
            inlet.cp * (inlet.T_total - result.outlet.T_total),
        )

    def test_textbook_forward_matches_independent_mean_process(self) -> None:
        inlet = true_gas_inlet()
        result = turbine_expansion(
            inlet,
            2.5,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None

        expected_T, expected_cp, expected_k, expected_eta = (
            expected_forward_state(1250.0, 2.5, 0.9, 1.0)
        )
        self.assertAlmostEqual(result.outlet.T_total, expected_T, delta=1e-7)
        self.assertEqual(record.T_mean, 0.5 * (inlet.T_total + result.outlet.T_total))
        self.assertAlmostEqual(record.cp, expected_cp, delta=1e-9)
        self.assertAlmostEqual(record.k, expected_k, delta=1e-12)
        self.assertEqual(result.efficiency, expected_eta)
        self.assertEqual(
            result.specific_work,
            record.cp * (inlet.T_total - result.outlet.T_total),
        )
        self.assertGreater(result.specific_work, 0.0)
        self.assertEqual(result.power, result.specific_work * inlet.mass_flow)
        self.assertEqual(result.outlet.p_total, inlet.p_total / 2.5)
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertLessEqual(
            abs(record.temperature_residual_K), record.tolerance_K
        )
        self.assertLessEqual(
            abs(record.pressure_formula_temperature_residual_K),
            record.tolerance_K,
        )

        local_cp, local_k = source_cp_k(result.outlet.T_total, 1.0)
        self.assertEqual(result.outlet.cp, local_cp)
        self.assertEqual(result.outlet.k, local_k)
        self.assertNotEqual(result.outlet.cp, record.cp)

    def test_textbook_forward_mean_below_750_uses_lower_branch(self) -> None:
        inlet = true_gas_inlet(T_total=700.0)
        result = turbine_expansion(
            inlet,
            1.5,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None
        self.assertLess(record.T_mean, TEXTBOOK_CP_BRANCH_T_K)

        expected_T, expected_cp, expected_k, expected_eta = (
            expected_forward_state(700.0, 1.5, 0.9, 1.0)
        )
        self.assertAlmostEqual(result.outlet.T_total, expected_T, delta=1e-7)
        self.assertAlmostEqual(record.cp, expected_cp, delta=1e-9)
        self.assertAlmostEqual(record.k, expected_k, delta=1e-12)
        self.assertEqual(result.efficiency, expected_eta)
        self.assertEqual(result.outlet.cp, source_cp_k(result.outlet.T_total, 1.0)[0])

    def test_forward_efficiency_uses_process_k_not_local_inlet_k(self) -> None:
        inlet = true_gas_inlet()
        result = turbine_expansion(
            inlet,
            2.5,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None
        self.assertEqual(
            result.efficiency, source_eta_expansion(2.5, record.k, 0.9)
        )
        self.assertNotEqual(
            result.efficiency, source_eta_expansion(2.5, inlet.k, 0.9)
        )

    def test_textbook_inverse_closes_requested_work(self) -> None:
        inlet = true_gas_inlet()
        required = 250000.0
        result = turbine_pressure_ratio_for_work(
            inlet,
            required,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None

        expected_ratio, expected_T, expected_k = expected_inverse_ratio(
            1250.0, required, 0.9, 1.0
        )
        self.assertAlmostEqual(
            result.pressure_ratio, expected_ratio, delta=1e-9
        )
        self.assertAlmostEqual(result.outlet.T_total, expected_T, delta=1e-7)
        self.assertAlmostEqual(record.k, expected_k, delta=1e-12)
        self.assertEqual(record.requested_specific_work, required)
        self.assertEqual(
            record.achieved_specific_work,
            record.cp * (inlet.T_total - result.outlet.T_total),
        )
        self.assertLessEqual(
            record.work_relative_residual, TEXTBOOK_WORK_RELATIVE_TOL
        )
        self.assertLessEqual(
            abs(record.temperature_residual_K), record.tolerance_K
        )
        self.assertLessEqual(
            abs(record.pressure_formula_temperature_residual_K),
            record.tolerance_K,
        )
        self.assertGreater(result.specific_work, 0.0)
        self.assertEqual(
            result.outlet.p_total, inlet.p_total / result.pressure_ratio
        )
        local_cp, local_k = source_cp_k(result.outlet.T_total, 1.0)
        self.assertEqual(result.outlet.cp, local_cp)
        self.assertEqual(result.outlet.k, local_k)

    def test_textbook_inverse_mean_below_750_uses_lower_branch(self) -> None:
        inlet = true_gas_inlet(T_total=700.0)
        required = 30000.0
        result = turbine_pressure_ratio_for_work(
            inlet,
            required,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None
        self.assertLess(record.T_mean, TEXTBOOK_CP_BRANCH_T_K)

        expected_ratio, expected_T, expected_k = expected_inverse_ratio(
            700.0, required, 0.9, 1.0
        )
        self.assertAlmostEqual(
            result.pressure_ratio, expected_ratio, delta=1e-9
        )
        self.assertAlmostEqual(result.outlet.T_total, expected_T, delta=1e-7)
        self.assertAlmostEqual(
            record.cp, source_cp_k(record.T_mean, 1.0)[0], delta=1e-9
        )
        self.assertAlmostEqual(record.k, expected_k, delta=1e-12)
        self.assertLessEqual(
            record.work_relative_residual, TEXTBOOK_WORK_RELATIVE_TOL
        )

    def test_forward_zero_residual_fields_are_rejected(self) -> None:
        inlet = true_gas_inlet()
        result = turbine_expansion(
            inlet,
            2.5,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None
        for field in (
            "temperature_residual_K",
            "pressure_formula_temperature_residual_K",
        ):
            with self.subTest(field=field):
                tampered = replace(record, **{field: 0.0})
                with self.assertRaises(ValueError):
                    replace(result, process_properties=tampered)

    def test_inverse_zero_residual_and_requested_work_tampering_is_rejected(
        self,
    ) -> None:
        inlet = true_gas_inlet()
        result = turbine_pressure_ratio_for_work(
            inlet,
            500000.0,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        record = result.process_properties
        assert record is not None
        for field in (
            "temperature_residual_K",
            "pressure_formula_temperature_residual_K",
        ):
            with self.subTest(field=field):
                tampered = replace(record, **{field: 0.0})
                with self.assertRaises(ValueError):
                    replace(result, process_properties=tampered)

        new_requested = record.requested_specific_work + 1e-5
        coherent = replace(
            record,
            requested_specific_work=new_requested,
            work_relative_residual=(
                abs(record.achieved_specific_work - new_requested)
                / new_requested
            ),
        )
        with self.assertRaisesRegex(ValueError, "temperature_residual_K"):
            replace(result, process_properties=coherent)

    def test_forward_then_inverse_round_trip(self) -> None:
        inlet = true_gas_inlet()
        forward = turbine_expansion(
            inlet,
            2.5,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        inverse = turbine_pressure_ratio_for_work(
            inlet,
            forward.specific_work,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        self.assertAlmostEqual(
            inverse.pressure_ratio, forward.pressure_ratio, delta=1e-9
        )
        self.assertAlmostEqual(
            inverse.outlet.T_total, forward.outlet.T_total, delta=1e-6
        )
        self.assertAlmostEqual(
            inverse.specific_work, forward.specific_work, delta=1e-6
        )

    def test_gg_to_fpt_state_identity_and_independent_process_means(self) -> None:
        gg_inlet = true_gas_inlet(T_total=1250.0, mass_flow=0.7)
        gg = turbine_expansion(
            gg_inlet,
            4.0,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )
        fpt = turbine_expansion(
            gg.outlet,
            2.0,
            0.9,
            gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
            alpha=1.0,
            R=DEFAULT_CONFIG.R,
        )

        self.assertIs(fpt.inlet, gg.outlet)
        self.assertEqual(fpt.inlet, gg.outlet)
        self.assertIsNot(fpt.outlet, gg.outlet)
        self.assertEqual(fpt.outlet.mass_flow, gg.outlet.mass_flow)
        self.assertEqual(fpt.outlet.p_total, gg.outlet.p_total / 2.0)

        gg_process = gg.process_properties
        fpt_process = fpt.process_properties
        assert gg_process is not None and fpt_process is not None
        self.assertNotEqual(gg_process.T_mean, fpt_process.T_mean)
        self.assertNotEqual(gg_process.cp, fpt_process.cp)
        self.assertNotEqual(fpt_process.cp, gg.outlet.cp)
        self.assertEqual(
            fpt.outlet.cp, source_cp_k(fpt.outlet.T_total, 1.0)[0]
        )

    def test_invalid_method_and_composition_are_rejected(self) -> None:
        inlet = true_gas_inlet()
        for basis in ("bad", "", None, 1):
            with self.subTest(basis=basis):
                with self.assertRaises(ValueError):
                    turbine_expansion(inlet, 2.5, 0.9, gas_property_basis=basis)
        with self.assertRaises(ValueError):
            turbine_expansion(
                inlet,
                2.5,
                0.9,
                gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
                R=DEFAULT_CONFIG.R,
            )
        with self.assertRaises(ValueError):
            turbine_expansion(
                inlet,
                2.5,
                0.9,
                gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
                alpha=1.0,
            )
        for alpha in (0.9, 0.0, -1.0, float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    turbine_expansion(
                        inlet,
                        2.5,
                        0.9,
                        gas_property_basis=(
                            TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK
                        ),
                        alpha=alpha,
                        R=DEFAULT_CONFIG.R,
                    )
        for R in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(R=R):
                with self.assertRaises(ValueError):
                    turbine_expansion(
                        inlet,
                        2.5,
                        0.9,
                        gas_property_basis=(
                            TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK
                        ),
                        alpha=1.0,
                        R=R,
                    )
        with self.assertRaises(ValueError):
            turbine_expansion(
                inlet,
                2.5,
                0.9,
                alpha=1.0,
                R=DEFAULT_CONFIG.R,
            )

    def test_mismatched_inlet_state_is_rejected(self) -> None:
        legacy_inlet = StationState(
            T_total=1250.0,
            p_total=300000.0,
            mass_flow=0.7,
            cp=calc_cp_kerosene(1250.0, 1.0),
            k=calc_k_kerosene(1250.0, 1.0, DEFAULT_CONFIG.R),
        )
        for call in (
            lambda: turbine_expansion(
                legacy_inlet,
                2.5,
                0.9,
                gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
                alpha=1.0,
                R=DEFAULT_CONFIG.R,
            ),
            lambda: turbine_pressure_ratio_for_work(
                legacy_inlet,
                250000.0,
                0.9,
                gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
                alpha=1.0,
                R=DEFAULT_CONFIG.R,
            ),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, r"inlet\.cp/k"):
                    call()

    def test_forward_requires_expanding_ratio(self) -> None:
        inlet = true_gas_inlet()
        for ratio in (0.8, 1.0):
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    turbine_expansion(
                        inlet,
                        ratio,
                        0.9,
                        gas_property_basis=(
                            TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK
                        ),
                        alpha=1.0,
                        R=DEFAULT_CONFIG.R,
                    )

    def test_unavailable_work_is_rejected(self) -> None:
        inlet = true_gas_inlet()
        with self.assertRaises(ValueError):
            turbine_pressure_ratio_for_work(
                inlet,
                2.0 * inlet.cp * inlet.T_total,
                0.9,
                gas_property_basis=TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK,
                alpha=1.0,
                R=DEFAULT_CONFIG.R,
            )

    def test_forced_turbine_forward_nonconvergence_is_typed(self) -> None:
        inlet = true_gas_inlet()
        with mock.patch.object(turbine_module, "TEXTBOOK_MAX_ITER", 1):
            with self.assertRaises(NumericalConvergenceError) as caught:
                turbine_expansion(
                    inlet,
                    2.5,
                    0.9,
                    gas_property_basis=(
                        TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK
                    ),
                    alpha=1.0,
                    R=DEFAULT_CONFIG.R,
                )
        state = caught.exception.last_valid_state
        self.assertIn("T_evaluated", state)
        self.assertIn("T_proposed", state)
        self.assertIn("residual_K", state)

    def test_forced_turbine_inverse_nonconvergence_is_typed(self) -> None:
        inlet = true_gas_inlet()
        with mock.patch.object(turbine_module, "TEXTBOOK_MAX_ITER", 1):
            with self.assertRaises(NumericalConvergenceError) as caught:
                turbine_pressure_ratio_for_work(
                    inlet,
                    250000.0,
                    0.9,
                    gas_property_basis=(
                        TURBINE_GAS_PROPERTY_BASIS_TEXTBOOK
                    ),
                    alpha=1.0,
                    R=DEFAULT_CONFIG.R,
                )
        state = caught.exception.last_valid_state
        self.assertIn("T_evaluated", state)
        self.assertIn("T_proposed", state)
        self.assertIn("work_relative_residual", state)


class RawSerializerMetadataTests(unittest.TestCase):
    def test_default_component_serializers_ignore_new_metadata(self) -> None:
        compressor = compressor_unit(ambient_inlet(), 11.0)
        turbine = turbine_expansion(true_gas_inlet(), 2.5, 0.9)

        self.assertEqual(
            set(_compressor_to_json(compressor)),
            {
                "inlet",
                "outlet",
                "station_states",
                "pressure_ratios",
                "pressure_ratio_product",
                "per_unit_specific_work",
                "specific_work",
                "power",
                "residual",
            },
        )
        self.assertEqual(
            set(_turbine_to_json(turbine)),
            {
                "inlet",
                "outlet",
                "pressure_ratio",
                "eta_turbine_poly",
                "specific_work",
                "power",
                "efficiency",
                "residual",
            },
        )


if __name__ == "__main__":
    unittest.main()
