"""Unit tests for the stateless single-point cycle core.

These tests exercise the extracted functions directly, without loading either
notebook.  They check implementation behaviour and plausible ranges; they do
not claim physical validation of the source formulae.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from dataclasses import asdict, FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.cycle import (  # noqa: E402
    CycleConfig,
    CycleSolverError,
    CycleState,
    DEFAULT_CONFIG,
    calc_alpha,
    calc_compressor,
    calc_cp_kerosene,
    calc_g_m,
    calc_k_kerosene,
    calculate_cycle_point,
)


class GasPropertyTests(unittest.TestCase):
    def test_cp_low_temperature_branch_matches_source_expression(self) -> None:
        T = 288.0
        alpha = math.inf
        expected = ((1.2 + 2.25 / alpha) / 1e5 * (T - 70.0) + 0.236) * 4187.0
        self.assertAlmostEqual(calc_cp_kerosene(T, alpha), expected, places=12)
        self.assertGreater(calc_cp_kerosene(T, alpha), 900.0)
        self.assertLess(calc_cp_kerosene(T, alpha), 1100.0)

    def test_cp_high_temperature_branch_and_alpha_effect(self) -> None:
        T = 1000.0
        alpha = 1.0
        expected = ((2.2 + 1.25 / alpha) / 1e5 * (T + 450.0) + 0.218) * 4187.0
        self.assertAlmostEqual(calc_cp_kerosene(T, alpha), expected, places=12)
        self.assertGreater(calc_cp_kerosene(T, alpha), 1000.0)
        self.assertLess(calc_cp_kerosene(T, alpha), 1300.0)
        self.assertGreater(
            calc_cp_kerosene(T, alpha),
            calc_cp_kerosene(T, math.inf),
            "combustion products should have higher cp than dry air here",
        )

    def test_cp_uses_700k_branch_boundary_as_source(self) -> None:
        alpha = 1.0
        low = ((1.2 + 2.25 / alpha) / 1e5 * (699.999 - 70.0) + 0.236) * 4187.0
        high = ((2.2 + 1.25 / alpha) / 1e5 * (700.0 + 450.0) + 0.218) * 4187.0
        self.assertAlmostEqual(calc_cp_kerosene(699.999, alpha), low, places=12)
        self.assertAlmostEqual(calc_cp_kerosene(700.0, alpha), high, places=12)

    def test_k_is_cp_over_cp_minus_r(self) -> None:
        for T, alpha in ((288.0, math.inf), (1000.0, 1.0), (1250.0, 1.0)):
            with self.subTest(T=T, alpha=alpha):
                cp = calc_cp_kerosene(T, alpha)
                expected = cp / (cp - DEFAULT_CONFIG.R)
                actual = calc_k_kerosene(T, alpha, DEFAULT_CONFIG.R)
                self.assertAlmostEqual(actual, expected, places=14)
                self.assertGreater(actual, 1.2)
                self.assertLess(actual, 1.5)

    def test_alpha_is_positive_in_typical_operating_range(self) -> None:
        for T_k, T_g in ((600.0, 1250.0), (700.0, 1300.0)):
            with self.subTest(T_k=T_k, T_g=T_g):
                value = calc_alpha(T_k, T_g)
                self.assertTrue(math.isfinite(value))
                self.assertGreater(value, 2.0)
                self.assertLess(value, 6.0)

    def test_cp_rejects_invalid_temperature_and_alpha(self) -> None:
        with self.assertRaises(ValueError):
            calc_cp_kerosene(float("nan"))
        with self.assertRaises(ValueError):
            calc_cp_kerosene(288.0, alpha=0.0)
        with self.assertRaises(ValueError):
            calc_cp_kerosene(288.0, alpha=-1.0)


class CompressorTests(unittest.TestCase):
    def test_compressor_properties_and_monotonicity(self) -> None:
        previous_T = 0.0
        for pi in (3.0, 6.0, 11.0, 13.0):
            with self.subTest(pi=pi):
                T_k, k = calc_compressor(pi, DEFAULT_CONFIG)
                self.assertGreater(T_k, DEFAULT_CONFIG.T_a_full)
                self.assertLess(T_k, 1250.0)
                self.assertGreater(k, 1.35)
                self.assertLess(k, 1.45)
                self.assertGreater(T_k, previous_T)
                previous_T = T_k

    def test_compressor_non_convergence_fails_explicitly(self) -> None:
        with self.assertRaises(CycleSolverError):
            calc_compressor(11.0, DEFAULT_CONFIG, tol=0.0)

    def test_compressor_rejects_bad_iteration_count(self) -> None:
        with self.assertRaises(ValueError):
            calc_compressor(11.0, DEFAULT_CONFIG, nmax=0)


class ConfigAndStateTests(unittest.TestCase):
    def test_defaults_match_current_notebook_assumptions(self) -> None:
        cfg = CycleConfig()
        self.assertEqual(cfg.T_a_full, 288.0)
        self.assertEqual(cfg.p_a, 101300.0)
        self.assertEqual(cfg.p_a_full, 101300.0)
        self.assertEqual(cfg.pi_v, 1.0)
        self.assertEqual(cfg.R, 287.4)
        self.assertEqual(cfg.k_air, 1.4)
        self.assertEqual(cfg.sigma_in, 0.98)
        self.assertEqual(cfg.sigma_combustion, 0.96)
        self.assertEqual(cfg.eta_compressor_poly, 0.85)
        self.assertEqual(cfg.eta_turbine_poly, 0.90)
        self.assertEqual(cfg.eta_combustion, 0.98)
        self.assertEqual(cfg.eta_mech_turbine, 0.99)
        self.assertEqual(cfg.eta_reducer, 0.98)
        self.assertEqual(cfg.g_leak, 0.0075)
        self.assertEqual(cfg.q_lhv_kj, 43150.0)
        self.assertEqual(cfg.l0, 14.61)
        self.assertEqual(cfg.M_t, 0.3379)
        self.assertEqual(cfg.pi_nozzle, 1.05)
        self.assertEqual(cfg.phi_nozzle, 0.95)
        self.assertEqual(cfg.T_ref, 288.0)
        self.assertEqual(cfg.T_cool_threshold, 1250.0)
        self.assertEqual(cfg.g_cooling, 0.05)
        self.assertEqual(cfg.g_cooling_off, 0.015)
        self.assertEqual(cfg.g_return_fraction, 0.6)

    def test_from_namespace_uses_notebook_constants(self) -> None:
        namespace = asdict(DEFAULT_CONFIG)
        namespace["R"] = 300.0
        namespace["T_а_полн"] = 300.0
        namespace["T_a_full"] = 999.0  # notebook name must take precedence
        before = dict(namespace)

        cfg = CycleConfig.from_namespace(namespace)

        self.assertEqual(cfg.R, 300.0)
        self.assertEqual(cfg.T_a_full, 300.0)
        self.assertEqual(namespace, before, "adapter must not mutate its input")

    def test_from_namespace_requires_every_stable_constant(self) -> None:
        with self.assertRaises(KeyError):
            CycleConfig.from_namespace({})

        namespace = asdict(DEFAULT_CONFIG)
        namespace.pop("R")
        with self.assertRaises(KeyError):
            CycleConfig.from_namespace(namespace)

    def test_invalid_state_and_config_fail_explicitly(self) -> None:
        for kwargs in (
            {"pi": 0.0, "T_g": 1250.0, "N": 110000.0},
            {"pi": 11.0, "T_g": -1.0, "N": 110000.0},
            {"pi": 11.0, "T_g": 1250.0, "N": 0.0},
            {"pi": float("nan"), "T_g": 1250.0, "N": 110000.0},
            {"pi": 11.0, "T_g": float("inf"), "N": 110000.0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    CycleState(**kwargs)

        with self.assertRaises(ValueError):
            CycleConfig(R=0.0)
        with self.assertRaises(ValueError):
            CycleConfig(eta_compressor_poly=0.0)
        with self.assertRaises(ValueError):
            CycleConfig(sigma_in=1.5)

    def test_config_and_state_are_immutable(self) -> None:
        config = CycleConfig()
        state = CycleState(pi=11.0, T_g=1250.0, N=110000.0)
        with self.assertRaises(FrozenInstanceError):
            config.R = 1.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            state.pi = 1.0  # type: ignore[misc]


class CyclePointTests(unittest.TestCase):
    def test_direct_core_call_matches_reference_anchor(self) -> None:
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
                self.assertAlmostEqual(getattr(result, field) / value, 1.0, delta=1e-12)

    def test_power_plane_is_pre_reducer_and_gearbox_output_is_explicit(self) -> None:
        config = CycleConfig()
        result = calculate_cycle_point(
            CycleState(pi=11.0, T_g=1250.0, N=110000.0), config
        )

        # N is the gearbox-input target: G = N / N_ud(pre-reducer).
        self.assertAlmostEqual(
            result.G * result.N_ud / 110000.0, 1.0, delta=1e-12
        )
        # Reducer loss is downstream only.
        self.assertAlmostEqual(
            result.N_reducer_out_ud,
            result.N_ud * config.eta_reducer,
            delta=1e-9,
        )
        self.assertLess(result.N_reducer_out_ud, result.N_ud)
        # The equivalent propulsive metric is built on the gearbox output.
        thrust_term = result.N_e_ud - result.N_reducer_out_ud
        self.assertGreaterEqual(thrust_term, 0.0)
        # C_e uses the requested pre-reducer shaft plane.
        fuel_flow_factor = result.g_m * (1.0 - config.g_leak - result.g_cooling)
        self.assertAlmostEqual(
            result.C_e,
            fuel_flow_factor * 3600.0 / (result.N_ud / 1e3),
            delta=1e-12,
        )

    def test_cooling_threshold_is_strictly_greater_than(self) -> None:
        cases = (
            (1250.0, False, 0.015, 0.009),
            (1250.0001, True, 0.05, 0.03),
            (1300.0, True, 0.05, 0.03),
        )
        for T_g, cooled, g_ohl, g_voz in cases:
            with self.subTest(T_g=T_g):
                result = calculate_cycle_point(
                    CycleState(pi=11.0, T_g=T_g, N=110000.0), CycleConfig()
                )
                self.assertEqual(result.cooled, cooled)
                self.assertEqual(result.g_cooling, g_ohl)
                self.assertEqual(result.g_return, g_voz)

    def test_core_does_not_mutate_inputs(self) -> None:
        config = CycleConfig.from_namespace(asdict(DEFAULT_CONFIG))
        state = CycleState(pi=11.0, T_g=1300.0, N=110000.0)
        config_before = asdict(config)
        state_before = asdict(state)

        calculate_cycle_point(state, config)

        self.assertEqual(asdict(config), config_before)
        self.assertEqual(asdict(state), state_before)

    def test_g_m_is_finite_and_positive_for_reference_point(self) -> None:
        T_k, _ = calc_compressor(11.0, DEFAULT_CONFIG)
        g_m = calc_g_m(T_k, 1250.0, DEFAULT_CONFIG)
        self.assertTrue(math.isfinite(g_m))
        self.assertGreater(g_m, 0.0)
        self.assertLess(g_m, 0.1)

    def test_core_import_does_not_load_notebook_loader(self) -> None:
        # Run in a fresh interpreter so this module's imports cannot hide a
        # dependency on cycle_link / optimizer_core.
        code = (
            "import sys\n"
            "from turbine_engine.cycle import CycleState, calculate_cycle_point\n"
            "r = calculate_cycle_point(CycleState(pi=11.0, T_g=1250.0, N=110000.0))\n"
            "assert abs(r.G / 0.5771555503613757 - 1.0) < 1e-12, r.G\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "print(r.G)\n"
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


if __name__ == "__main__":
    unittest.main()
