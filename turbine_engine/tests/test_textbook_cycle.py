"""Opt-in textbook true-cp variant of the existing single-point cycle.

PROVENANCE / STATUS: method and software diagnostics only.

These tests do not run an optimizer search, sweep or production calculation,
and they do not claim physical validation of the cycle.  They verify that the
default legacy arithmetic is unchanged and that the explicitly selected
``textbook_true_cp`` variant is self-consistent with the published direct true
``cp`` fit and the source process equations recorded in
``docs/AUTONOMOUS_EXECUTION.md``.

The local oracle below independently transcribes the published Appendix 6
true-cp branches (750 K split, kcal -> J) instead of relying only on the
production helper.  The legacy mean-cp fuel relation is also re-evaluated from
its numerator/denominator in the tests.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import turbine_engine.cycle as cycle_module  # noqa: E402
from turbine_engine.cycle import (  # noqa: E402
    ALPHA_BASIS_LEGACY,
    ALPHA_BASIS_TEXTBOOK,
    CYCLE_GAS_PROPERTY_BASIS_LEGACY,
    CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
    SOLVER_TOL,
    CycleConfig,
    CycleSolverError,
    CycleState,
    calc_compressor,
    calc_cp_kerosene,
    calc_eta_expansion,
    calculate_cycle_point,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
    SolverDiagnosticsError,
)
from turbine_engine.gas import calc_textbook_gas_cp_k  # noqa: E402

TRUE_CP_BRANCH_T_K = 750.0
TRUE_CP_KCAL_TO_J = 4187.0


def oracle_true_cp_k(T: float, alpha: float, R: float) -> tuple[float, float]:
    """Independent transcription of the published direct true-cp fit."""

    if T < TRUE_CP_BRANCH_T_K:
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
    cp = cp_kcal * TRUE_CP_KCAL_TO_J
    return cp, cp / (cp - R)


def oracle_g_m(T_k: float, T_g: float, config: CycleConfig) -> float:
    """Independent transcription of the source mean-cp fuel balance."""

    T_p = T_k
    c_air_g = calc_cp_kerosene(T_g)
    c_air_p = calc_cp_kerosene(T_p)
    c_air_0 = calc_cp_kerosene(config.T_ref)
    c_products_g = calc_cp_kerosene(T_g, alpha=1.0)
    c_products_0 = calc_cp_kerosene(config.T_ref, alpha=1.0)
    numerator = c_air_g * T_g - c_air_p * T_p
    denominator = (
        config.q_lhv_kj * 1e3 * config.eta_combustion
        - (1.0 + config.l0) * (c_products_g * T_g - c_products_0 * config.T_ref)
        + config.l0 * (c_air_g * T_g - c_air_0 * config.T_ref)
    )
    return numerator / denominator


def assert_finite_tree(test: unittest.TestCase, value, path: str = "") -> None:
    """Assert a JSON-friendly tree contains only finite plain Python scalars."""

    if isinstance(value, dict):
        for key, item in value.items():
            test.assertIsInstance(key, str, f"{path} key is not a string")
            assert_finite_tree(test, item, f"{path}.{key}" if path else key)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(test, item, f"{path}[{index}]")
        return
    test.assertIn(
        type(value),
        {str, bool, int, float, type(None)},
        f"{path} is not a plain JSON scalar: {type(value)!r}",
    )
    if isinstance(value, float):
        test.assertTrue(math.isfinite(value), f"{path} is not finite: {value!r}")


class TextbookCycleTests(unittest.TestCase):
    PI_VALUES = (3.0, 7.5, 11.0, 13.0)
    T_G = 1250.0
    N = 110000.0

    def point(
        self, pi: float = 11.0, T_g: float | None = None, N: float | None = None
    ):
        return calculate_cycle_point(
            CycleState(
                pi=pi,
                T_g=self.T_G if T_g is None else T_g,
                N=self.N if N is None else N,
            ),
            CycleConfig(),
            gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
        )

    def test_textbook_variant_is_finite_on_control_points(self) -> None:
        for pi in self.PI_VALUES:
            with self.subTest(pi=pi):
                result = self.point(pi)
                self.assertEqual(
                    result.gas_property_basis,
                    CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
                )
                self.assertEqual(result.alpha_basis, ALPHA_BASIS_TEXTBOOK)
                self.assertIsNotNone(result.method_diagnostics)

                self.assertGreater(result.pi_t, 1.0)
                self.assertGreater(result.pi_c_full, 1.0)
                self.assertGreater(result.c_t, 0.0)
                self.assertGreater(result.H_c_full, 0.0)
                self.assertGreater(result.c_c, 0.0)
                self.assertGreater(result.T_t, 0.0)
                self.assertLess(result.T_t, result.T_g)
                self.assertGreater(result.G, 0.0)
                self.assertGreater(result.N_ud, 0.0)
                self.assertGreater(result.g_m, 0.0)
                self.assertGreaterEqual(result.alpha, 1.0)

                diagnostics = result.method_diagnostics
                assert diagnostics is not None
                for stage in ("compressor", "turbine", "nozzle"):
                    self.assertLess(
                        abs(diagnostics[stage]["temperature_residual_K"]),
                        SOLVER_TOL,
                        f"{stage} residual exceeds the existing tolerance",
                    )
                assert_finite_tree(self, diagnostics)

    def test_compressor_fixed_point_matches_oracle(self) -> None:
        config = CycleConfig()
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        compressor = diagnostics["compressor"]
        T_in = compressor["inlet_total_temperature_K"]
        T_k = compressor["outlet_total_temperature_K"]
        T_mean = compressor["process_temperature_mean_K"]
        cp, k = oracle_true_cp_k(T_mean, math.inf, config.R)

        self.assertAlmostEqual(compressor["cp_process_J_per_kgK"], cp, places=9)
        self.assertAlmostEqual(compressor["k_process"], k, places=12)
        T_new = T_in * 11.0 ** (
            (k - 1.0) / (k * config.eta_compressor_poly)
        )
        self.assertAlmostEqual(
            compressor["temperature_residual_K"], T_new - T_k, places=9
        )
        self.assertAlmostEqual(result.L_k, cp * (T_k - T_in), places=6)

        # The public helper uses the same evaluated state and returns its k.
        public_T, public_k = calc_compressor(
            11.0, config, gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK
        )
        self.assertAlmostEqual(public_T, T_k, places=12)
        self.assertAlmostEqual(public_k, k, places=12)

    def test_source_fuel_balance_and_alpha_consistency(self) -> None:
        config = CycleConfig()
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        fuel = diagnostics["fuel"]
        T_k = fuel["T_compressor_exit_K"]
        g_m = oracle_g_m(T_k, self.T_G, config)

        self.assertAlmostEqual(result.g_m, g_m, places=14)
        self.assertAlmostEqual(result.alpha, 1.0 / (g_m * config.l0), places=12)
        self.assertEqual(fuel["T_fuel_reference_K"], config.T_ref)
        self.assertGreaterEqual(result.alpha, 1.0)

        turbine = diagnostics["turbine"]
        turbine_cp, turbine_k = oracle_true_cp_k(
            turbine["process_temperature_mean_K"], result.alpha, config.R
        )
        self.assertAlmostEqual(turbine["cp_process_J_per_kgK"], turbine_cp, places=9)
        self.assertAlmostEqual(turbine["k_process"], turbine_k, places=12)

        nozzle = diagnostics["nozzle"]
        nozzle_cp, nozzle_k = oracle_true_cp_k(
            nozzle["process_temperature_mean_K"], result.alpha, config.R
        )
        self.assertAlmostEqual(nozzle["cp_process_J_per_kgK"], nozzle_cp, places=9)
        self.assertAlmostEqual(nozzle["k_process"], nozzle_k, places=12)

    def test_turbine_residual_and_velocity_identity(self) -> None:
        config = CycleConfig()
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        turbine = diagnostics["turbine"]
        T_t = turbine["outlet_total_temperature_K"]
        cp, k = oracle_true_cp_k(
            turbine["process_temperature_mean_K"], result.alpha, config.R
        )
        _, k_exit = oracle_true_cp_k(T_t, result.alpha, config.R)

        self.assertAlmostEqual(turbine["cp_process_J_per_kgK"], cp, places=9)
        self.assertAlmostEqual(turbine["k_process"], k, places=12)
        self.assertAlmostEqual(turbine["k_exit"], k_exit, places=12)

        mach_factor = 1.0 + 0.5 * (k_exit - 1.0) * config.M_t**2
        pt_total = turbine["outlet_static_pressure_Pa"] * mach_factor ** (
            k_exit / (k_exit - 1.0)
        )
        self.assertAlmostEqual(
            turbine["outlet_total_pressure_Pa"], pt_total, places=6
        )
        self.assertAlmostEqual(result.pi_t, result.p_g / pt_total, places=12)
        eta_t = calc_eta_expansion(result.pi_t, k, config.eta_turbine_poly)
        self.assertAlmostEqual(result.eta_t, eta_t, places=12)

        work = (
            cp
            * self.T_G
            * (1.0 - result.pi_t ** ((1.0 - k) / k))
            * eta_t
        )
        self.assertAlmostEqual(result.L_t, work, places=6)
        T_out = self.T_G - work / cp
        self.assertAlmostEqual(
            turbine["temperature_residual_K"], T_out - T_t, places=9
        )
        self.assertLess(abs(turbine["temperature_residual_K"]), SOLVER_TOL)

        T_static = T_t / mach_factor
        self.assertAlmostEqual(
            turbine["outlet_static_temperature_K"], T_static, places=9
        )
        c_t = config.M_t * math.sqrt(k_exit * config.R * T_static)
        self.assertAlmostEqual(result.c_t, c_t, places=9)

        # The total/static pressure identity uses the same local exit k and
        # the same evaluated outlet state used for the velocity.
        self.assertAlmostEqual(
            turbine["outlet_total_pressure_Pa"]
            / turbine["outlet_static_pressure_Pa"],
            mach_factor ** (k_exit / (k_exit - 1.0)),
            places=12,
        )
        self.assertEqual(result.p_t, config.pi_nozzle * config.p_a)

    def test_nozzle_residual_and_temperature_balance(self) -> None:
        config = CycleConfig()
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        turbine = diagnostics["turbine"]
        nozzle = diagnostics["nozzle"]
        T_t = turbine["outlet_total_temperature_K"]
        cp, k = oracle_true_cp_k(
            nozzle["process_temperature_mean_K"], result.alpha, config.R
        )
        self.assertAlmostEqual(nozzle["cp_process_J_per_kgK"], cp, places=9)
        self.assertAlmostEqual(nozzle["k_process"], k, places=12)

        pi_c_full = turbine["outlet_total_pressure_Pa"] / config.p_a
        self.assertAlmostEqual(result.pi_c_full, pi_c_full, places=12)
        self.assertEqual(nozzle["outlet_static_pressure_Pa"], config.p_a)
        self.assertAlmostEqual(
            nozzle["inlet_total_pressure_Pa"]
            / nozzle["outlet_static_pressure_Pa"],
            pi_c_full,
            places=12,
        )
        H_c = cp * T_t * (1.0 - pi_c_full ** ((1.0 - k) / k))
        self.assertAlmostEqual(result.H_c_full, H_c, places=6)
        c_c = config.phi_nozzle * math.sqrt(2.0 * H_c)
        self.assertAlmostEqual(result.c_c, c_c, places=9)

        T_out = T_t - c_c**2 / (2.0 * cp)
        self.assertAlmostEqual(
            nozzle["temperature_residual_K"],
            T_out - nozzle["outlet_static_temperature_K"],
            places=9,
        )
        self.assertLess(abs(nozzle["temperature_residual_K"]), SOLVER_TOL)
        self.assertEqual(nozzle["inlet_total_temperature_K"], T_t)
        self.assertLess(nozzle["outlet_static_temperature_K"], T_t)

    def test_power_plane_and_economics(self) -> None:
        config = CycleConfig()
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        powers = diagnostics["powers"]

        N_ud = (
            result.L_t * (result.g_t + result.g_return) * config.eta_mech_turbine
            - result.L_k
        )
        self.assertAlmostEqual(result.N_ud, N_ud, places=6)
        self.assertAlmostEqual(result.G, self.N / result.N_ud, places=12)
        self.assertAlmostEqual(powers["pre_reducer_power_check_W"], self.N, places=6)
        self.assertAlmostEqual(
            powers["post_reducer_power_check_W"],
            self.N * config.eta_reducer,
            places=6,
        )
        self.assertAlmostEqual(
            result.N_reducer_out_ud, result.N_ud * config.eta_reducer, places=9
        )
        self.assertAlmostEqual(
            result.R_ud,
            (result.g_t + result.g_return) * (result.c_c - config.v),
            places=9,
        )
        thrust_term = result.N_e_ud - result.N_reducer_out_ud
        self.assertAlmostEqual(
            thrust_term, result.R_ud / config.beta_prop, places=6
        )
        self.assertAlmostEqual(
            result.C_e,
            result.g_m
            * (1.0 - config.g_leak - result.g_cooling)
            * 3600.0
            / (result.N_ud / 1e3),
            places=9,
        )
        self.assertAlmostEqual(
            result.eta_e, 3600.0 / (result.C_e * config.q_lhv_kj), places=12
        )
        self.assertAlmostEqual(
            result.p_g,
            config.sigma_combustion
            * config.sigma_in
            * config.p_a_full
            * result.pi,
            places=6,
        )

    def test_cooling_threshold_bookkeeping_unchanged_label(self) -> None:
        cases = (
            (1250.0, False, 0.015, 0.009),
            (1250.0001, True, 0.05, 0.03),
        )
        for T_g, cooled, g_cooling, g_return in cases:
            with self.subTest(T_g=T_g):
                result = self.point(11.0, T_g=T_g)
                self.assertEqual(result.cooled, cooled)
                self.assertEqual(result.g_cooling, g_cooling)
                self.assertEqual(result.g_return, g_return)
                diagnostics = result.method_diagnostics
                assert diagnostics is not None
                self.assertIn(
                    "cooling_bookkeeping_only_no_temperature_mixing",
                    diagnostics["limitations"],
                )
                self.assertIn("NOT_PHYSICALLY_VALIDATED", diagnostics["limitations"])
                self.assertIn("common_R", diagnostics["limitations"])

    def test_small_tg_variation_remains_finite(self) -> None:
        lower = self.point(11.0, T_g=1249.9)
        upper = self.point(11.0, T_g=1250.1)
        for result in (lower, upper):
            self.assertTrue(math.isfinite(result.T_t))
            self.assertTrue(math.isfinite(result.G))
            self.assertTrue(math.isfinite(result.c_c))
        self.assertLess(abs(upper.T_t - lower.T_t), 5.0)
        self.assertLess(abs(upper.G - lower.G) / lower.G, 0.05)

    def test_legacy_default_and_explicit_are_identical_and_anchored(self) -> None:
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        default = calculate_cycle_point(state, CycleConfig())
        explicit = calculate_cycle_point(
            state,
            CycleConfig(),
            gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_LEGACY,
        )
        self.assertEqual(asdict(default), asdict(explicit))
        self.assertEqual(
            default.gas_property_basis, CYCLE_GAS_PROPERTY_BASIS_LEGACY
        )
        self.assertEqual(default.alpha_basis, ALPHA_BASIS_LEGACY)
        self.assertIsNone(default.method_diagnostics)

        expected = {
            "G": 0.5771555503613757,
            "L_k": 358583.40802120743,
            "L_t": 552874.7217839112,
            "T_t": 730.1309202648472,
            "eta_t": 0.9246868857757602,
            "N_ud": 190589.86772478488,
            "N_reducer_out_ud": 186778.07037028918,
            "C_e": 0.3180573704024953,
            "eta_e": 0.2623108391012963,
            "c_t": 180.09582219113014,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertAlmostEqual(
                    getattr(default, field) / value, 1.0, delta=1e-12
                )

    def test_unknown_and_non_string_selector_rejected_early(self) -> None:
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        with patch("turbine_engine.cycle.calc_g_m") as g_m:
            with self.assertRaises(ValueError):
                calculate_cycle_point(state, gas_property_basis="not_a_basis")
            g_m.assert_not_called()
        with self.assertRaises(ValueError):
            calculate_cycle_point(state, gas_property_basis=None)
        with self.assertRaises(ValueError):
            calc_compressor(
                11.0, CycleConfig(), gas_property_basis="not_a_basis"
            )

    def test_nonphysical_points_are_structured_and_json_safe(self) -> None:
        with self.assertRaises(SolverDiagnosticsError) as pressure_case:
            self.point(1.01)
        self.assertEqual(pressure_case.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            pressure_case.exception.reason,
            "turbine_pressure_ratio_not_above_one",
        )
        json.dumps(pressure_case.exception.as_record(), allow_nan=False)

        with self.assertRaises(SolverDiagnosticsError) as temperature_case:
            self.point(11.0, T_g=300.0)
        self.assertEqual(temperature_case.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            temperature_case.exception.reason,
            "turbine_inlet_not_above_compressor_exit",
        )

        # Lean-domain guard: do not extrapolate the published true-cp fit.
        with patch("turbine_engine.cycle.calc_g_m", return_value=0.1):
            with self.assertRaises(SolverDiagnosticsError) as lean_case:
                self.point(11.0)
        self.assertEqual(lean_case.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            lean_case.exception.reason,
            "outside_textbook_lean_alpha_domain",
        )
        json.dumps(lean_case.exception.as_record(), allow_nan=False)

    def test_forced_one_iteration_preserves_evaluated_residual_all_processes(self) -> None:
        config = CycleConfig()
        pi = 11.0

        # Compressor: one explicit iteration, then re-derive the proposed state.
        with self.assertRaises(NumericalConvergenceError) as compressor_case:
            cycle_module._textbook_compressor_state(
                config.T_a_full, pi, config, SOLVER_TOL, 1
            )
        state = self.assert_one_iteration_state(compressor_case.exception)
        evaluated = state["evaluated_temperature_K"]
        proposed = state["proposed_temperature_K"]
        cp, k = oracle_true_cp_k(
            0.5 * (config.T_a_full + evaluated), math.inf, config.R
        )
        expected = config.T_a_full * pi ** (
            (k - 1.0) / (k * config.eta_compressor_poly)
        )
        self.assertAlmostEqual(proposed, expected, places=9)
        self.assertAlmostEqual(state["cp"], cp, places=9)
        self.assertAlmostEqual(state["k"], k, places=12)

        # Turbine: one forced iteration, then re-derive the coupled pressure/work.
        reference = self.point(11.0)
        with patch.object(cycle_module, "SOLVER_MAX_ITER", 1):
            with self.assertRaises(NumericalConvergenceError) as turbine_case:
                cycle_module._solve_textbook_turbine(
                    reference.T_g,
                    reference.alpha,
                    config,
                    reference.p_g,
                    reference.p_t,
                    reference.pi,
                )
        state = self.assert_one_iteration_state(turbine_case.exception)
        evaluated = state["evaluated_temperature_K"]
        proposed = state["proposed_temperature_K"]
        cp, k = oracle_true_cp_k(
            0.5 * (reference.T_g + evaluated), reference.alpha, config.R
        )
        _, k_exit = oracle_true_cp_k(evaluated, reference.alpha, config.R)
        mach_factor = 1.0 + 0.5 * (k_exit - 1.0) * config.M_t**2
        pt_total = reference.p_t * mach_factor ** (
            k_exit / (k_exit - 1.0)
        )
        pi_t = reference.p_g / pt_total
        eta_t = calc_eta_expansion(pi_t, k, config.eta_turbine_poly)
        work = cp * reference.T_g * (1.0 - pi_t ** ((1.0 - k) / k)) * eta_t
        expected = reference.T_g - work / cp
        self.assertAlmostEqual(proposed, expected, places=6)
        self.assertAlmostEqual(
            state["residual_K"], expected - evaluated, places=9
        )

        # Nozzle: one forced iteration, then re-derive the static outlet state.
        with patch.object(cycle_module, "SOLVER_MAX_ITER", 1):
            with self.assertRaises(NumericalConvergenceError) as nozzle_case:
                cycle_module._solve_textbook_nozzle(
                    reference.T_t,
                    reference.pi_c_full,
                    reference.alpha,
                    config,
                )
        state = self.assert_one_iteration_state(nozzle_case.exception)
        evaluated = state["evaluated_temperature_K"]
        proposed = state["proposed_temperature_K"]
        cp, k = oracle_true_cp_k(
            0.5 * (reference.T_t + evaluated), reference.alpha, config.R
        )
        H_c = cp * reference.T_t * (
            1.0 - reference.pi_c_full ** ((1.0 - k) / k)
        )
        c_c = config.phi_nozzle * math.sqrt(2.0 * H_c)
        expected = reference.T_t - c_c**2 / (2.0 * cp)
        self.assertAlmostEqual(proposed, expected, places=6)
        self.assertAlmostEqual(
            state["residual_K"], expected - evaluated, places=9
        )

    def assert_one_iteration_state(self, error) -> dict:
        state = error.last_valid_state
        evaluated = state["evaluated_temperature_K"]
        proposed = state["proposed_temperature_K"]
        residual = state["residual_K"]
        self.assertAlmostEqual(residual, proposed - evaluated, places=12)
        self.assertGreater(abs(residual), 0.0)
        self.assertFalse(state["converged"])
        json.dumps(error.as_record(), allow_nan=False)
        return state

    def test_fuel_cycle_solver_error_is_not_relabelled_nonphysical(self) -> None:
        with patch(
            "turbine_engine.cycle.calc_g_m",
            side_effect=CycleSolverError("forced numeric fuel failure"),
        ):
            with self.assertRaises(CycleSolverError) as case:
                self.point(11.0)
        self.assertNotIsInstance(case.exception, SolverDiagnosticsError)
        self.assertIsNone(getattr(case.exception, "status", None))
        self.assertEqual(str(case.exception), "forced numeric fuel failure")

    def test_mass_bookkeeping_rejects_leak_plus_cooling_at_one(self) -> None:
        # Before the guard this could leave a positive N_ud even though the
        # remaining combustor-air factor g_t is non-positive.
        config = CycleConfig(
            g_leak=0.96,
            g_cooling=0.05,
            g_cooling_off=0.0,
            g_return_fraction=20.0,
        )
        with self.assertRaises(SolverDiagnosticsError) as case:
            calculate_cycle_point(
                CycleState(pi=11.0, T_g=1300.0, N=self.N),
                config,
                gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
            )
        self.assertEqual(case.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            case.exception.reason, "non_positive_combustor_air_fraction"
        )
        state = case.exception.last_valid_state
        self.assertEqual(state["g_leak"], 0.96)
        self.assertEqual(state["g_cooling"], 0.05)
        self.assertEqual(state["g_cooling_off"], 0.0)
        self.assertEqual(state["g_return_fraction"], 20.0)
        self.assertEqual(state["g_return"], 1.0)
        self.assertLess(state["g_t"], 0.0)
        json.dumps(case.exception.as_record(), allow_nan=False)

    def test_mass_bookkeeping_rejects_return_above_cooling(self) -> None:
        # The remaining air factor is positive here; only the cooling-return
        # bookkeeping is invalid.
        config = CycleConfig(g_return_fraction=2.0)
        with self.assertRaises(SolverDiagnosticsError) as case:
            calculate_cycle_point(
                CycleState(pi=11.0, T_g=1300.0, N=self.N),
                config,
                gas_property_basis=CYCLE_GAS_PROPERTY_BASIS_TEXTBOOK,
            )
        self.assertEqual(case.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            case.exception.reason,
            "return_fraction_exceeds_cooling_fraction",
        )
        state = case.exception.last_valid_state
        self.assertEqual(state["g_cooling"], config.g_cooling)
        self.assertEqual(state["g_return"], config.g_return_fraction * config.g_cooling)
        self.assertGreater(state["g_t"], 0.0)
        json.dumps(case.exception.as_record(), allow_nan=False)

    def test_metadata_is_json_plain_and_result_is_frozen(self) -> None:
        result = self.point(11.0)
        diagnostics = result.method_diagnostics
        assert diagnostics is not None
        encoded = json.dumps(diagnostics, allow_nan=False)
        self.assertEqual(json.loads(encoded), diagnostics)
        assert_finite_tree(self, diagnostics)
        self.assertEqual(
            [item for item in diagnostics["limitations"]],
            [
                "NOT_PHYSICALLY_VALIDATED",
                "cooling_bookkeeping_only_no_temperature_mixing",
                "common_R",
            ],
        )

        # The additive fields are present, but the default legacy dictionary
        # shape is still readable by the optimizer's manual field mapping.
        serialized = asdict(result)
        self.assertIn("gas_property_basis", serialized)
        self.assertIn("alpha_basis", serialized)
        self.assertIn("method_diagnostics", serialized)
        # The frozen dataclass prevents attribute rebinding only; the nested
        # diagnostics mapping is a plain Mapping, not a deep-frozen object.
        with self.assertRaises(FrozenInstanceError):
            result.G = 0.0  # type: ignore[misc]

    def test_oracle_matches_production_true_cp_helper(self) -> None:
        # The production helper is cross-checked, not used as the only oracle.
        for T, alpha in (
            (288.0, math.inf),
            (749.999, 1.0),
            (750.0, 1.0),
            (1250.0, 3.925911251041318),
            (1600.0, 1.5),
        ):
            with self.subTest(T=T, alpha=alpha):
                expected_cp, expected_k = oracle_true_cp_k(
                    T, alpha, CycleConfig().R
                )
                cp, k = calc_textbook_gas_cp_k(T, alpha, CycleConfig().R)
                self.assertAlmostEqual(cp, expected_cp, places=9)
                self.assertAlmostEqual(k, expected_k, places=12)


if __name__ == "__main__":
    unittest.main()
