"""Tests for the stateless turbine-stage core.

PROVENANCE / STATUS: software-drift diagnostic only.

The three-stage comparison below dynamically builds the *current* notebook
namespace once, computes one fixed candidate geometry with the legacy stage
distribution, and then compares the legacy ``ступень`` function with
``turbine_engine.stage.calculate_stage`` for every stage.  This is a software-drift
check, not a physical validation or an acceptable-design claim.

This test still builds its comparison states with the legacy
``cp_k_ступени(T0, H0)`` so the stage core is isolated from the gas helper; the
production adapter now uses :mod:`turbine_engine.gas` instead.  No optimizer
search, no parameter sweep, and no notebook execution from
``turbine_engine.stage`` are involved.
"""

from __future__ import annotations

import math
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbine_engine.reaction import (  # noqa: E402
    ReactionCalculationError,
    calculate_root_reaction,
)
from turbine_engine.stage import (  # noqa: E402
    DEFAULT_STAGE_CONFIG,
    StageCalculationError,
    StageConfig,
    StageResult,
    StageState,
    calculate_stage,
)

# A complete finite state used only for input-validation tests.  Values are not
# a design point; they are close to the first stage of the fixed candidate used
# by the parity test.
BASE_STATE_KWARGS = {
    "T0": 1250.0,
    "p0": 1048333.44,
    "H0": 249627.8814129167,
    "rho": 0.23509880832838634,
    "D_c": 0.0763446500669355,
    "D_l": 0.07476510377950549,
    "n": 84000.0,
    "G_c": 0.5855999695198287,
    "G_l": 0.5855999695198287,
    "alpha1_deg": 11.0,
    "kappa": 1.38,
    "cp": 1173.528257136208,
    "k": 1.3243322822463874,
}


class ConfigAndStateTests(unittest.TestCase):
    def test_defaults_match_current_notebook_assumptions(self) -> None:
        cfg = StageConfig()
        self.assertEqual(cfg.R, 287.4)
        self.assertEqual(cfg.phi, 0.97)
        self.assertEqual(cfg.psi, 0.96)
        self.assertEqual(cfg.delta_rel, 0.005)
        self.assertEqual(cfg.delta_min, 0.2e-3)
        self.assertEqual(cfg.epsilon_parc, 1.0)
        self.assertEqual(cfg.K_n, 6.8e-6)
        self.assertEqual(DEFAULT_STAGE_CONFIG, cfg)

    def test_from_namespace_uses_notebook_constants_and_precedence(self) -> None:
        namespace = asdict(StageConfig())
        namespace.update(
            {
                "R": 300.0,
                "φ": 0.95,
                "ψ": 0.94,
                "δ_отн": 0.006,
                "delta_min": 0.0003,
                "ε_парц": 0.9,
                "k_n": 7e-6,
                "phi": 0.11,
                "psi": 0.12,
            }
        )
        before = dict(namespace)

        cfg = StageConfig.from_namespace(namespace)

        self.assertEqual(cfg.R, 300.0)
        self.assertEqual(cfg.phi, 0.95)
        self.assertEqual(cfg.psi, 0.94)
        self.assertEqual(cfg.delta_rel, 0.006)
        self.assertEqual(cfg.delta_min, 0.0003)
        self.assertEqual(cfg.epsilon_parc, 0.9)
        self.assertEqual(cfg.K_n, 7e-6)
        self.assertEqual(namespace, before, "adapter must not mutate its input")

    def test_from_namespace_requires_every_constant(self) -> None:
        with self.assertRaises(KeyError):
            StageConfig.from_namespace({})

        namespace = asdict(StageConfig())
        namespace.pop("R")
        with self.assertRaises(KeyError):
            StageConfig.from_namespace(namespace)

    def test_invalid_config_fails_explicitly(self) -> None:
        for kwargs in (
            {"R": 0.0},
            {"R": float("nan")},
            {"phi": 0.0},
            {"phi": 1.5},
            {"psi": 0.0},
            {"psi": -0.1},
            {"delta_rel": -1.0},
            {"delta_min": -1.0},
            {"epsilon_parc": -0.1},
            {"epsilon_parc": 1.1},
            {"K_n": 0.0},
            {"K_n": float("inf")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    StageConfig(**kwargs)

    def test_invalid_state_fails_explicitly(self) -> None:
        for name, value in (
            ("T0", 0.0),
            ("T0", float("nan")),
            ("p0", -1.0),
            ("H0", float("inf")),
            ("rho", -0.1),
            ("rho", 1.1),
            ("D_c", 0.0),
            ("D_l", -0.2),
            ("n", 0.0),
            ("G_c", 0.0),
            ("G_l", float("nan")),
            ("alpha1_deg", 0.0),
            ("alpha1_deg", 180.0),
            ("kappa", 0.0),
            ("cp", 0.0),
            ("k", 1.0),
        ):
            kwargs = dict(BASE_STATE_KWARGS)
            kwargs[name] = value
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    StageState(**kwargs)

    def test_config_and_state_are_immutable(self) -> None:
        config = StageConfig()
        state = StageState(**BASE_STATE_KWARGS)
        with self.assertRaises(FrozenInstanceError):
            config.R = 1.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            state.T0 = 1.0  # type: ignore[misc]


class LegacyStageParityTests(unittest.TestCase):
    """Compare all scalar outputs with the notebook implementation."""

    CANDIDATE_N = 84000.0
    CANDIDATE_D_HUB = 0.065
    RELATIVE_TOLERANCE = 1e-12

    @classmethod
    def setUpClass(cls) -> None:
        # Importing the legacy model requires the declared runtime dependencies.
        # Skip only when one of those is genuinely unavailable; any other import
        # or model failure must surface as an error, not as a skip.
        sys.path.insert(0, str(ROOT / "Optimizer"))
        sys.path.insert(0, str(ROOT / "Turbinis-Calc"))
        try:
            import optimizer_core as oc
            import optimize_extended as oe
        except ModuleNotFoundError as exc:
            dependency = (exc.name or "").split(".")[0]
            if dependency in {"numpy", "scipy", "pandas"}:
                raise unittest.SkipTest(
                    "legacy stage comparison needs runtime dependency "
                    f"'{dependency}', which is not installed"
                ) from exc
            raise

        cls.oc = oc
        cls.oe = oe
        cls.model = oc.load_model()
        # orchestration_core=False keeps the notebook stage/root-reaction
        # functions as the independent reference for the parity tests below.
        cls.namespace, _ = oe.build_ns(
            cls.model, q_n_limit=0.995, orchestration_core=False
        )

        pi0 = cls.namespace["π_т"] ** (1.0 / cls.namespace["z"])
        _, cls.stages = cls.namespace["распределить_π_const_hub"](
            cls.CANDIDATE_N, cls.CANDIDATE_D_HUB, pi0
        )
        cls.config = StageConfig.from_namespace(cls.namespace)

    def make_state(self, index: int, **overrides: float) -> StageState:
        stage = self.stages[index]
        cp, k = self.namespace["cp_k_ступени"](stage["T0"], stage["H0"])
        kwargs = {
            "T0": stage["T0"],
            "p0": stage["p0"],
            "H0": stage["H0"],
            "rho": stage["ρ"],
            "D_c": stage["D_c"],
            "D_l": stage["D_л"],
            "n": self.CANDIDATE_N,
            "G_c": stage["G_c"],
            "G_l": stage["G_л"],
            "alpha1_deg": self.namespace["α1_ст"][index],
            "kappa": self.namespace["κ_ст"][index],
            "cp": cp,
            "k": k,
        }
        kwargs.update(overrides)
        return StageState(**kwargs)

    def assertRelative(self, actual: float, expected: float) -> None:
        self.assertTrue(math.isfinite(actual), f"expected finite value, got {actual!r}")
        if expected == 0.0:
            self.assertEqual(actual, 0.0)
            return
        self.assertLessEqual(
            abs(actual / expected - 1.0),
            self.RELATIVE_TOLERANCE,
            f"expected ~{expected!r}, got {actual!r}",
        )

    def test_adapter_reads_actual_namespace_constants(self) -> None:
        self.assertEqual(self.config.R, self.namespace["R"])
        self.assertEqual(self.config.phi, self.namespace["φ"])
        self.assertEqual(self.config.psi, self.namespace["ψ"])
        self.assertEqual(self.config.delta_rel, self.namespace["δ_отн"])
        self.assertEqual(self.config.delta_min, self.namespace["δ_мин"])
        self.assertEqual(self.config.epsilon_parc, self.namespace["ε_парц"])
        self.assertEqual(self.config.K_n, self.namespace["k_n"])

    def test_three_computed_stages_match_all_legacy_outputs(self) -> None:
        self.assertEqual(len(self.stages), 3, "expected the fixed three-stage model")

        for index in range(3):
            state = self.make_state(index)
            legacy = self.namespace["ступень"](
                state.T0,
                state.p0,
                state.H0,
                state.rho,
                state.D_c,
                state.D_l,
                state.n,
                state.G_c,
                state.G_l,
                state.alpha1_deg,
                state.kappa,
            )
            result = calculate_stage(state, self.config).as_legacy_dict()

            self.assertEqual(set(result), set(legacy))
            for field, expected in legacy.items():
                with self.subTest(stage=index + 1, field=field):
                    self.assertRelative(result[field], expected)

    def legacy_stage_dict(self, index: int) -> dict[str, float]:
        state = self.make_state(index)
        return self.namespace["ступень"](
            state.T0,
            state.p0,
            state.H0,
            state.rho,
            state.D_c,
            state.D_l,
            state.n,
            state.G_c,
            state.G_l,
            state.alpha1_deg,
            state.kappa,
        )

    def test_stage_result_from_legacy_dict_round_trips_without_mutation(self) -> None:
        for index in range(3):
            legacy = self.legacy_stage_dict(index)
            before = dict(legacy)
            typed = StageResult.from_legacy_dict(legacy)
            self.assertEqual(typed.as_legacy_dict(), legacy)
            self.assertEqual(legacy, before)

    def test_root_reaction_matches_legacy_for_both_laws_on_all_stages(self) -> None:
        max_relative_difference = 0.0
        for index in range(3):
            stage = self.legacy_stage_dict(index)
            typed = StageResult.from_legacy_dict(stage)
            for law in ("Γ", "α1"):
                with self.subTest(stage=index + 1, law=law):
                    expected = self.namespace["ρ_корня"](stage, law)
                    actual = calculate_root_reaction(
                        typed,
                        law,
                        self.namespace["R"],
                        self.namespace["φ"],
                    )
                    self.assertRelative(actual, expected)
                    max_relative_difference = max(
                        max_relative_difference,
                        abs(actual / expected - 1.0),
                    )
        self.assertLessEqual(max_relative_difference, self.RELATIVE_TOLERANCE)

    def test_root_reaction_fallback_matches_legacy_for_other_law_values(self) -> None:
        for index in range(3):
            stage = self.legacy_stage_dict(index)
            typed = StageResult.from_legacy_dict(stage)
            for law in ("α1", "", "constant-alpha", "not-Γ"):
                with self.subTest(stage=index + 1, law=law):
                    expected = self.namespace["ρ_корня"](stage, law)
                    actual = calculate_root_reaction(
                        typed,
                        law,
                        self.namespace["R"],
                        self.namespace["φ"],
                    )
                    self.assertRelative(actual, expected)

    def compare_root_reaction_with_legacy(
        self, stage: dict[str, float], law: str
    ) -> tuple[float | complex, float | complex]:
        expected = self.namespace["ρ_корня"](stage, law)
        actual = calculate_root_reaction(
            StageResult.from_legacy_dict(stage),
            law,
            self.namespace["R"],
            self.namespace["φ"],
        )
        return expected, actual

    def test_root_reaction_negative_gamma_pressure_factor_matches_legacy(self) -> None:
        # k = 2 makes the (71) exponent exactly 2.0, so Python turns the
        # negative pressure factor into a positive real p1.  That notebook
        # outcome is calculable and must not be rejected by a sign guard.
        stage = dict(self.legacy_stage_dict(0))
        stage.update(
            k=2.0,
            T1_ad=800.0,
            c1u=2000.0,
            D_c=0.0763446500669355,
            l_c=0.03,
            p1=9e5,
            p0=1048333.44,
            T0=1250.0,
            p2=8e5,
            H0=249627.8814129167,
            cp=1173.528257136208,
            alpha1_deg=11.0,
        )
        r_c = stage["D_c"] / 2
        r1 = r_c - 0.5 * stage["l_c"]
        mach = stage["c1u"] / math.sqrt(
            stage["k"] * self.namespace["R"] * stage["T1_ad"]
        )
        pressure_base = 1 + (stage["k"] - 1) / 2 * mach**2 * (
            1 - (r_c / r1) ** 2
        )
        self.assertLess(pressure_base, 0.0)
        self.assertEqual(stage["k"] / (stage["k"] - 1), 2.0)

        expected, actual = self.compare_root_reaction_with_legacy(stage, "Γ")
        self.assertIsInstance(actual, float)
        self.assertRelative(actual, expected)

    def test_root_reaction_negative_fallback_pressure_factor_matches_legacy(
        self,
    ) -> None:
        # Same integer-exponent condition in the general fallback branch.
        stage = dict(self.legacy_stage_dict(0))
        stage.update(
            k=2.0,
            cp=1005.0,
            T0=300.0,
            p0=101325.0,
            H0=100000.0,
            p2=80000.0,
            D_c=0.1,
            l_c=0.02,
            c1=900.0,
            alpha1_deg=11.0,
        )
        r_c = stage["D_c"] / 2
        r1 = r_c - 0.5 * stage["l_c"]
        c1_root = stage["c1"] * (r_c / r1) ** (
            math.cos(math.radians(stage["alpha1_deg"])) ** 2
        )
        pressure_base = 1 - (c1_root / self.namespace["φ"]) ** 2 / (
            2 * stage["cp"] * stage["T0"]
        )
        self.assertLess(pressure_base, 0.0)
        self.assertEqual(stage["k"] / (stage["k"] - 1), 2.0)

        expected, actual = self.compare_root_reaction_with_legacy(stage, "α1")
        self.assertIsInstance(actual, float)
        self.assertRelative(actual, expected)

    def test_root_reaction_negative_root_radius_matches_legacy(self) -> None:
        # The Γ branch only uses (r_c / r1) ** 2, so a negative root radius is
        # calculable and must not be rejected by a radius-sign guard.
        stage = dict(self.legacy_stage_dict(0))
        stage.update(
            k=2.0,
            T1_ad=800.0,
            c1u=100.0,
            l_c=3.0 * stage["D_c"],
            p1=9e5,
            p0=1048333.44,
            T0=1250.0,
            p2=8e5,
            H0=249627.8814129167,
            cp=1173.528257136208,
            alpha1_deg=11.0,
        )
        self.assertLess(stage["D_c"] / 2 - 0.5 * stage["l_c"], 0.0)

        expected, actual = self.compare_root_reaction_with_legacy(stage, "Γ")
        self.assertIsInstance(actual, float)
        self.assertRelative(actual, expected)

    def test_root_reaction_complex_power_matches_legacy(self) -> None:
        # A non-integer exponent on a negative base is complex in Python.  The
        # notebook returns that value, so the extraction must not coerce or
        # reject it.
        stage = dict(self.legacy_stage_dict(0))
        stage.update(
            k=1.4,
            cp=1005.0,
            T0=300.0,
            p0=101325.0,
            H0=100000.0,
            p2=80000.0,
            D_c=0.1,
            l_c=0.02,
            c1=900.0,
            alpha1_deg=11.0,
        )
        r_c = stage["D_c"] / 2
        r1 = r_c - 0.5 * stage["l_c"]
        c1_root = stage["c1"] * (r_c / r1) ** (
            math.cos(math.radians(stage["alpha1_deg"])) ** 2
        )
        pressure_base = 1 - (c1_root / self.namespace["φ"]) ** 2 / (
            2 * stage["cp"] * stage["T0"]
        )
        self.assertLess(pressure_base, 0.0)
        self.assertNotEqual(stage["k"] / (stage["k"] - 1), 2.0)

        expected, actual = self.compare_root_reaction_with_legacy(stage, "α1")
        self.assertIsInstance(actual, complex)
        self.assertAlmostEqual(actual.real, expected.real, places=12)
        self.assertAlmostEqual(actual.imag, expected.imag, places=12)

    def test_root_reaction_rejects_undefined_parameters(self) -> None:
        stage = self.legacy_stage_dict(0)
        typed = StageResult.from_legacy_dict(stage)

        with self.assertRaises(ValueError):
            calculate_root_reaction(typed, "α1", self.namespace["R"], 0.0)
        with self.assertRaises(ValueError):
            calculate_root_reaction(typed, "Γ", 0.0, self.namespace["φ"])

        zero_root = dict(stage)
        zero_root["l_c"] = stage["D_c"]  # r1 == 0: the source divides by zero
        with self.assertRaises(ReactionCalculationError):
            calculate_root_reaction(
                StageResult.from_legacy_dict(zero_root),
                "α1",
                self.namespace["R"],
                self.namespace["φ"],
            )

    def test_core_does_not_mutate_inputs(self) -> None:
        state = self.make_state(0)
        config = self.config
        state_before = asdict(state)
        config_before = asdict(config)

        calculate_stage(state, config)

        self.assertEqual(asdict(state), state_before)
        self.assertEqual(asdict(config), config_before)

    def test_engineering_outside_recommendations_is_not_rejected(self) -> None:
        # alpha1 = 5 deg is below the book recommendation and K_n is raised so
        # the reported stress is far above any allowable value.  Both are output
        # diagnostics here, not acceptance filters, so the state must compute.
        state = self.make_state(0, alpha1_deg=5.0)
        result = calculate_stage(state, self.config)
        self.assertGreater(result.l_c, 0.0)
        self.assertTrue(math.isfinite(result.alpha2_deg))
        self.assertGreater(result.sigma_rotor, 0.0)

        heavy_stress = StageConfig(
            **{**asdict(self.config), "K_n": 1e-3}
        )
        heavy_result = calculate_stage(state, heavy_stress)
        self.assertGreater(heavy_result.sigma_rotor, 1e4)
        self.assertEqual(heavy_result.eta_t, result.eta_t)

    def test_undefined_rotor_triangle_raises_stage_error(self) -> None:
        # With a large kappa, c2a = kappa*c1a exceeds w2; the notebook stops
        # with НетРешения("w2 <= c2a").
        state = self.make_state(0, kappa=1000.0)
        with self.assertRaises(StageCalculationError):
            calculate_stage(state, self.config)

    def test_right_angle_stator_exit_raises_stage_error(self) -> None:
        # alpha1 = 90 deg makes the axial-velocity triangle singular
        # (c1a = c1, then the model's c1a >= c1 / w2 <= c2a guards stop it).
        state = self.make_state(0, alpha1_deg=90.0)
        with self.assertRaises(StageCalculationError):
            calculate_stage(state, self.config)

    def test_standalone_import_does_not_load_notebook_loader(self) -> None:
        code = (
            "import sys\n"
            "from turbine_engine.stage import StageConfig, StageState, calculate_stage\n"
            "s = StageState(\n"
            "    T0=1250.0, p0=1048333.44, H0=249627.8814129167,\n"
            "    rho=0.23509880832838634, D_c=0.0763446500669355,\n"
            "    D_l=0.07476510377950549, n=84000.0,\n"
            "    G_c=0.5855999695198287, G_l=0.5855999695198287,\n"
            "    alpha1_deg=11.0, kappa=1.38,\n"
            "    cp=1173.528257136208, k=1.3243322822463874,\n"
            ")\n"
            "r = calculate_stage(s, StageConfig())\n"
            "assert abs(r.pi_stage / 2.141873005404617 - 1.0) < 1e-12, r.pi_stage\n"
            "assert abs(r.eta_t / 0.8293102822981497 - 1.0) < 1e-12, r.eta_t\n"
            "assert 'cycle_link' not in sys.modules, sorted(sys.modules)\n"
            "assert 'optimizer_core' not in sys.modules, sorted(sys.modules)\n"
            "print(r.eta_t)\n"
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
