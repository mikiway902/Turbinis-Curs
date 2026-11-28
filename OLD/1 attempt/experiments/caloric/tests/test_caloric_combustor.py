"""Focused tests for the opt-in independent caloric combustor closure.

The independent oracle below transcribes the source true-``cp`` expression
directly and integrates it with SciPy quadrature; it does not call the
production ``CaloricGasModel.h`` as its numerical oracle.  An independent
``scipy.optimize.brentq`` solve checks the fuel/air ratio and energy closure.

These tests do not claim physical validation of the source fit, a map, cooling,
composition, material life, or an operating range.  The result keeps the
``NOT_PHYSICALLY_VALIDATED`` limitation explicitly.
"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from turbine_engine.architecture import StationState  # noqa: E402
from experiments.caloric.caloric import (  # noqa: E402
    DEFAULT_SOLVER_TOL,
    SOURCE_CP_UNIT_FACTOR,
    CaloricGasModel,
    CaloricModelDomainError,
    CaloricModelError,
    FloatingResolutionError,
    IncompatibleStateError,
)
from experiments.caloric import caloric_combustor as combustor_module  # noqa: E402
from experiments.caloric.caloric_combustor import (  # noqa: E402
    LIMITATIONS,
    MODEL_VARIANT,
    VALIDATION_STATUS,
    CaloricCombustorEnthalpies,
    CaloricCombustorResult,
    CaloricCombustorSpec,
    close_caloric_combustor,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NON_PHYSICAL_STATE,
    NUMERICAL_FAILURE,
    NumericalConvergenceError,
)

try:  # Optional independent numerical oracle.
    from scipy.integrate import quad as _scipy_quad
    from scipy.optimize import brentq as _scipy_brentq
except ImportError:  # pragma: no cover - SciPy is a project dependency.
    _scipy_quad = None
    _scipy_brentq = None


R = 287.4
T_REF = 288.0
T_IN = 600.0
T_OUT = 1250.0
P_IN = 1.0e6
MASS_FLOW = 1.0
L0 = 14.61
Q_LHV = 43150e3
ETA = 0.98


def source_true_cp(T: float, alpha: float) -> float:
    """Independent hand transcription of the source true-``cp`` expression."""

    factor = SOURCE_CP_UNIT_FACTOR
    if T < 750.0:
        return factor * (
            (0.0174 / alpha + 0.2407)
            + (0.0093 / alpha + 0.0193) * (2.5e-3 * T - 0.875)
            + (2e-3 - 1.056e-3 / (alpha - 0.2))
            * (2.5e-5 * T * T - 0.0275 * T + 6.5625)
        )
    return factor * (
        (0.0267 / alpha + 0.26)
        + (0.0133 / alpha + 0.032) * (1.176e-3 * T - 0.88235)
        - (0.374e-2 + 0.94e-2 / (alpha * alpha + 10.0))
        * (5.5556e-6 * T * T - 1.3056e-2 * T + 6.67)
    )


def source_true_cp_air(T: float) -> float:
    """The same source expression at the analytic ``alpha = +inf`` limit."""

    factor = SOURCE_CP_UNIT_FACTOR
    if T < 750.0:
        return factor * (
            0.2407
            + 0.0193 * (2.5e-3 * T - 0.875)
            + 2e-3 * (2.5e-5 * T * T - 0.0275 * T + 6.5625)
        )
    return factor * (
        0.26
        + 0.032 * (1.176e-3 * T - 0.88235)
        - 0.374e-2 * (5.5556e-6 * T * T - 1.3056e-2 * T + 6.67)
    )


def make_spec(**overrides: object) -> CaloricCombustorSpec:
    values = {
        "totalT": T_OUT,
        "pressure_recovery": 0.96,
        "l0": L0,
        "Q_lhv_J_per_kg": Q_LHV,
        "eta_combustion": ETA,
        "fuel_sensible_enthalpy_J_per_kg": 0.0,
    }
    values.update(overrides)
    return CaloricCombustorSpec(**values)  # type: ignore[arg-type]


def make_inlet(
    mass_flow: float = MASS_FLOW,
    T_in: float = T_IN,
    p_in: float = P_IN,
    T_ref: float = T_REF,
) -> StationState:
    return CaloricGasModel(R, math.inf, T_ref).state(T_in, p_in, mass_flow)


def oracle_h(alpha: float, T: float, T_ref: float = T_REF) -> float:
    """``integral[T_ref, T] cp_true dT`` by independent SciPy quadrature."""

    if T == T_ref:
        return 0.0
    low, high = (T_ref, T) if T_ref < T else (T, T_ref)
    sign = 1.0 if T > T_ref else -1.0
    if math.isinf(alpha):
        function = source_true_cp_air
    else:
        function = lambda temperature: source_true_cp(temperature, alpha)
    points = [750.0] if low < 750.0 < high else None
    value = _scipy_quad(
        function,
        low,
        high,
        points=points,
        epsabs=1e-10,
        epsrel=1e-13,
    )[0]
    return sign * value


def oracle_residual(
    f: float,
    spec: CaloricCombustorSpec,
    T_in: float,
    T_ref: float = T_REF,
) -> float:
    alpha = math.inf if f == 0.0 else 1.0 / (f * spec.l0)
    gas = oracle_h(alpha, spec.totalT, T_ref)
    air = oracle_h(math.inf, T_in, T_ref)
    heat = (
        spec.eta_combustion * spec.Q_lhv_J_per_kg
        + spec.fuel_sensible_enthalpy_J_per_kg
    )
    return (1.0 + f) * gas - air - f * heat


def oracle_root(spec: CaloricCombustorSpec, T_in: float, T_ref: float = T_REF) -> float:
    return _scipy_brentq(
        lambda f: oracle_residual(f, spec, T_in, T_ref),
        0.0,
        spec.lean_f_limit,
        xtol=1e-15,
        rtol=1e-15,
    )


class SourceFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inlet = make_inlet()
        self.spec = make_spec()
        self.result = close_caloric_combustor(
            self.inlet, self.spec, R=R, T_ref=T_REF
        )

    def test_fixture_state_mass_power_and_provenance(self) -> None:
        result = self.result
        self.assertEqual(result.inlet, self.inlet)
        self.assertEqual(result.outlet.T_total, T_OUT)
        self.assertAlmostEqual(
            result.outlet.p_total, P_IN * self.spec.pressure_recovery, places=8
        )
        self.assertEqual(result.outlet.mass_flow, result.mgas)
        self.assertAlmostEqual(
            result.outlet.cp / result.gas_model.cp(T_OUT), 1.0, places=14
        )
        self.assertAlmostEqual(
            result.outlet.k / result.gas_model.k(T_OUT), 1.0, places=14
        )

        self.assertGreater(result.f, 0.0)
        self.assertLessEqual(result.f, self.spec.lean_f_limit)
        self.assertAlmostEqual(result.alpha, 1.0 / (result.f * self.spec.l0), places=12)
        self.assertGreaterEqual(result.alpha, 1.0)
        self.assertLessEqual(result.relative_residual, DEFAULT_SOLVER_TOL + 1e-15)
        self.assertGreaterEqual(result.iterations, 0)

        self.assertEqual(result.mfuel, result.f * self.inlet.mass_flow)
        self.assertEqual(
            result.mgas, (1.0 + result.f) * self.inlet.mass_flow
        )
        self.assertAlmostEqual(
            result.mass_residual_kg_per_s,
            self.inlet.mass_flow + result.mfuel - result.mgas,
            places=15,
        )
        self.assertLessEqual(abs(result.mass_residual_kg_per_s), 1e-15)

        self.assertEqual(
            result.gross_chemical_W,
            result.mfuel * self.spec.Q_lhv_J_per_kg,
        )
        self.assertEqual(
            result.effective_chemical_W,
            result.mfuel
            * self.spec.eta_combustion
            * self.spec.Q_lhv_J_per_kg,
        )
        self.assertEqual(result.fuel_sensible_W, result.mfuel * result.h_fuel)
        self.assertEqual(
            result.energy_residual_absolute_W,
            result.energy_residual_specific_J_per_kg_air
            * self.inlet.mass_flow,
        )

        self.assertEqual(result.reference_T, T_REF)
        self.assertEqual(result.R, R)
        self.assertEqual(result.model_variant, MODEL_VARIANT)
        self.assertEqual(result.limitations, LIMITATIONS)
        self.assertIn(VALIDATION_STATUS, result.limitations)
        self.assertIn("no_maps_no_cooling_no_bleed_no_leak", result.limitations)
        self.assertIs(result.spec, self.spec)

    @unittest.skipIf(
        _scipy_quad is None or _scipy_brentq is None,
        "SciPy independent oracle is unavailable",
    )
    def test_fixture_matches_independent_quadrature_and_brentq(self) -> None:
        result = self.result
        expected_f = oracle_root(self.spec, T_IN)
        self.assertLessEqual(
            abs(result.f / expected_f - 1.0),
            1e-8,
        )
        self.assertLessEqual(
            abs(result.alpha / (1.0 / (expected_f * self.spec.l0)) - 1.0),
            1e-8,
        )

        expected_gas_h = oracle_h(result.alpha, T_OUT)
        expected_air_h = oracle_h(math.inf, T_IN)
        self.assertLessEqual(
            abs(
                result.enthalpies.gas_product_J_per_kg / expected_gas_h
                - 1.0
            ),
            1e-9,
        )
        self.assertLessEqual(
            abs(
                result.enthalpies.air_inlet_J_per_kg / expected_air_h
                - 1.0
            ),
            1e-9,
        )

        heat = (
            self.spec.eta_combustion * self.spec.Q_lhv_J_per_kg
            + self.spec.fuel_sensible_enthalpy_J_per_kg
        )
        independent_residual = oracle_residual(result.f, self.spec, T_IN)
        independent_scale = max(
            abs((1.0 + result.f) * expected_gas_h),
            abs(expected_air_h),
            abs(result.f * self.spec.eta_combustion * self.spec.Q_lhv_J_per_kg),
            abs(result.f * self.spec.fuel_sensible_enthalpy_J_per_kg),
        )
        self.assertLessEqual(
            abs(independent_residual), 1e-8 * independent_scale
        )
        # The source bracket really changes sign at the fixture endpoints.
        self.assertGreater(oracle_residual(0.0, self.spec, T_IN), 0.0)
        self.assertLess(
            oracle_residual(self.spec.lean_f_limit, self.spec, T_IN), 0.0
        )
        self.assertGreater(heat, 0.0)

    @unittest.skipIf(
        _scipy_quad is None or _scipy_brentq is None,
        "SciPy independent oracle is unavailable",
    )
    def test_crossed_750_branch_is_independently_integrated(self) -> None:
        # Product interval T_ref -> T_out crosses 750 K.
        self.assertLess(T_REF, 750.0)
        self.assertGreater(T_OUT, 750.0)
        expected_gas_h = oracle_h(self.result.alpha, T_OUT)
        self.assertLessEqual(
            abs(
                self.result.enthalpies.gas_product_J_per_kg
                / expected_gas_h
                - 1.0
            ),
            1e-9,
        )

        # A hot inlet makes the dry-air interval also cross 750 K.
        hot_inlet = make_inlet(T_in=800.0)
        hot_result = close_caloric_combustor(
            hot_inlet, self.spec, R=R, T_ref=T_REF
        )
        expected_air_h = oracle_h(math.inf, 800.0)
        self.assertLessEqual(
            abs(
                hot_result.enthalpies.air_inlet_J_per_kg
                / expected_air_h
                - 1.0
            ),
            1e-9,
        )
        self.assertEqual(hot_result.outlet.T_total, T_OUT)

    def test_changed_flow_scales_absolute_powers_not_f(self) -> None:
        other_inlet = make_inlet(mass_flow=3.5)
        other = close_caloric_combustor(
            other_inlet, self.spec, R=R, T_ref=T_REF
        )
        self.assertEqual(other.f, self.result.f)
        self.assertEqual(other.alpha, self.result.alpha)
        self.assertEqual(
            other.energy_residual_specific_J_per_kg_air,
            self.result.energy_residual_specific_J_per_kg_air,
        )
        ratio = 3.5
        self.assertAlmostEqual(other.mfuel / self.result.mfuel, ratio, places=12)
        self.assertAlmostEqual(other.mgas / self.result.mgas, ratio, places=12)
        self.assertAlmostEqual(
            other.gross_chemical_W / self.result.gross_chemical_W,
            ratio,
            places=12,
        )
        self.assertAlmostEqual(
            other.effective_chemical_W / self.result.effective_chemical_W,
            ratio,
            places=12,
        )
        self.assertAlmostEqual(
            other.energy_residual_absolute_W
            / self.result.energy_residual_absolute_W,
            ratio,
            places=12,
        )

    def test_nonzero_positive_and_negative_fuel_sensible_enthalpy(self) -> None:
        base = self.result
        positive_spec = make_spec(fuel_sensible_enthalpy_J_per_kg=2.0e5)
        negative_spec = make_spec(fuel_sensible_enthalpy_J_per_kg=-2.0e5)
        positive = close_caloric_combustor(
            self.inlet, positive_spec, R=R, T_ref=T_REF
        )
        negative = close_caloric_combustor(
            self.inlet, negative_spec, R=R, T_ref=T_REF
        )

        self.assertLess(positive.f, base.f)
        self.assertLess(base.f, negative.f)
        self.assertEqual(positive.h_fuel, 2.0e5)
        self.assertEqual(negative.h_fuel, -2.0e5)
        self.assertEqual(
            positive.fuel_sensible_W, positive.mfuel * 2.0e5
        )
        self.assertEqual(
            negative.fuel_sensible_W, negative.mfuel * -2.0e5
        )
        # The efficiency applies only to the chemical term.
        self.assertEqual(
            positive.effective_chemical_W,
            positive.mfuel * ETA * Q_LHV,
        )
        self.assertEqual(
            negative.effective_chemical_W,
            negative.mfuel * ETA * Q_LHV,
        )

        if _scipy_quad is not None and _scipy_brentq is not None:
            for spec, result in (
                (positive_spec, positive),
                (negative_spec, negative),
            ):
                expected_f = oracle_root(spec, T_IN)
                self.assertLessEqual(
                    abs(result.f / expected_f - 1.0), 1e-8
                )
                self.assertLessEqual(
                    abs(oracle_residual(result.f, spec, T_IN)),
                    1e-8
                    * max(
                        abs((1.0 + result.f) * oracle_h(result.alpha, T_OUT)),
                        abs(oracle_h(math.inf, T_IN)),
                        abs(result.f * (ETA * Q_LHV + spec.fuel_sensible_enthalpy_J_per_kg)),
                    ),
                )

    def test_reference_temperature_is_caller_explicit(self) -> None:
        result = close_caloric_combustor(
            self.inlet, self.spec, R=R, T_ref=300.0
        )
        self.assertEqual(result.reference_T, 300.0)
        self.assertEqual(result.gas_model.T_ref, 300.0)
        self.assertEqual(result.gas_model.h(300.0), 0.0)
        self.assertEqual(result.gas_model.phi(300.0), 0.0)
        self.assertNotEqual(
            result.enthalpies.gas_product_J_per_kg,
            self.result.enthalpies.gas_product_J_per_kg,
        )

        if _scipy_quad is not None and _scipy_brentq is not None:
            expected_f = oracle_root(self.spec, T_IN, T_ref=300.0)
            self.assertLessEqual(abs(result.f / expected_f - 1.0), 1e-8)
            expected_gas_h = oracle_h(result.alpha, T_OUT, T_ref=300.0)
            self.assertLessEqual(
                abs(
                    result.enthalpies.gas_product_J_per_kg
                    / expected_gas_h
                    - 1.0
                ),
                1e-9,
            )


class ValidationAndDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inlet = make_inlet()
        self.spec = make_spec()

    def test_spec_has_no_physics_defaults_and_rejects_bad_types(self) -> None:
        with self.assertRaises(TypeError):
            CaloricCombustorSpec()  # type: ignore[call-arg]

        bad_values = (True, "1.0", None, float("nan"), float("inf"))
        for field_name in (
            "totalT",
            "pressure_recovery",
            "l0",
            "Q_lhv_J_per_kg",
            "eta_combustion",
            "fuel_sensible_enthalpy_J_per_kg",
        ):
            for value in bad_values:
                with self.subTest(field=field_name, value=value):
                    with self.assertRaises(CaloricModelError):
                        make_spec(**{field_name: value})

        for value in (0.0, -1.0):
            with self.subTest(field="totalT", value=value):
                with self.assertRaises(CaloricModelError):
                    make_spec(totalT=value)
            with self.subTest(field="l0", value=value):
                with self.assertRaises(CaloricModelError):
                    make_spec(l0=value)
            with self.subTest(field="Q", value=value):
                with self.assertRaises(CaloricModelError):
                    make_spec(Q_lhv_J_per_kg=value)
            with self.subTest(field="recovery", value=value):
                with self.assertRaises(CaloricModelError):
                    make_spec(pressure_recovery=value)
            with self.subTest(field="eta", value=value):
                with self.assertRaises(CaloricModelError):
                    make_spec(eta_combustion=value)

        self.assertEqual(make_spec(pressure_recovery=1.0).pressure_recovery, 1.0)
        self.assertEqual(make_spec(eta_combustion=1.0).eta_combustion, 1.0)
        self.assertEqual(
            make_spec(fuel_sensible_enthalpy_J_per_kg=-ETA * Q_LHV + 1.0)
            .fuel_sensible_enthalpy_J_per_kg,
            -ETA * Q_LHV + 1.0,
        )

    def test_nonpositive_effective_fuel_heat_is_explicit_domain(self) -> None:
        with self.assertRaises(CaloricModelDomainError) as caught:
            make_spec(fuel_sensible_enthalpy_J_per_kg=-ETA * Q_LHV)
        error = caught.exception
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertEqual(error.stage, "caloric.combustor.spec")
        self.assertEqual(error.reason, "nonpositive_effective_fuel_heat")
        self.assertIn("effective_fuel_heat", error.last_valid_state)

    def test_inlet_state_and_target_temperature_validation(self) -> None:
        incompatible = StationState(
            T_total=T_IN,
            p_total=P_IN,
            mass_flow=1.0,
            cp=900.0,
            k=1.3,
        )
        with self.assertRaises(IncompatibleStateError):
            close_caloric_combustor(
                incompatible, self.spec, R=R, T_ref=T_REF
            )
        with self.assertRaises(TypeError):
            close_caloric_combustor(  # type: ignore[arg-type]
                "not-a-state", self.spec, R=R, T_ref=T_REF
            )
        with self.assertRaises(TypeError):
            close_caloric_combustor(  # type: ignore[arg-type]
                self.inlet, {"totalT": T_OUT}, R=R, T_ref=T_REF
            )

        for target in (T_IN, 500.0):
            with self.subTest(target=target):
                spec = make_spec(totalT=target)
                with self.assertRaises(CaloricModelDomainError) as caught:
                    close_caloric_combustor(
                        self.inlet, spec, R=R, T_ref=T_REF
                    )
                error = caught.exception
                self.assertEqual(error.status, NON_PHYSICAL_STATE)
                self.assertEqual(error.stage, "caloric.combustor.domain")
                self.assertEqual(
                    error.reason, "target_temperature_not_above_inlet"
                )

    def test_r_tref_tol_and_max_iter_are_strict(self) -> None:
        for value in (0.0, -1.0, float("inf"), float("nan"), True, "287"):
            with self.subTest(R=value):
                with self.assertRaises(CaloricModelError):
                    close_caloric_combustor(
                        self.inlet, self.spec, R=value, T_ref=T_REF
                    )
        for value in (0.0, -1.0, float("inf"), float("nan"), True, "288"):
            with self.subTest(T_ref=value):
                with self.assertRaises(CaloricModelError):
                    close_caloric_combustor(
                        self.inlet, self.spec, R=R, T_ref=value
                    )
        for value in (0.0, -1.0, True, "1e-9"):
            with self.subTest(tol=value):
                with self.assertRaises(CaloricModelError):
                    close_caloric_combustor(
                        self.inlet,
                        self.spec,
                        R=R,
                        T_ref=T_REF,
                        tol=value,
                    )
        for value in (0, -1, True, "10"):
            with self.subTest(max_iter=value):
                with self.assertRaises(CaloricModelError):
                    close_caloric_combustor(
                        self.inlet,
                        self.spec,
                        R=R,
                        T_ref=T_REF,
                        max_iter=value,
                    )

    def test_max_iter_and_near_resolution_diagnostics(self) -> None:
        with self.assertRaises(NumericalConvergenceError) as caught:
            close_caloric_combustor(
                self.inlet,
                self.spec,
                R=R,
                T_ref=T_REF,
                max_iter=1,
            )
        error = caught.exception
        self.assertEqual(error.status, NUMERICAL_FAILURE)
        self.assertEqual(error.stage, "caloric.combustor.bisection")
        self.assertEqual(error.reason, "bisection_iteration_limit")
        self.assertIn("bracket_lower_f", error.last_valid_state)
        self.assertIn("best_relative_residual", error.last_valid_state)

        with self.assertRaises(FloatingResolutionError) as caught_resolution:
            close_caloric_combustor(
                self.inlet,
                self.spec,
                R=R,
                T_ref=T_REF,
                tol=1e-20,
            )
        resolution = caught_resolution.exception
        self.assertEqual(resolution.status, NUMERICAL_FAILURE)
        self.assertEqual(resolution.stage, "caloric.combustor.bisection")
        self.assertEqual(
            resolution.reason,
            "floating_fuel_fraction_resolution_limit",
        )
        self.assertGreater(
            resolution.last_valid_state["achieved_relative_residual"],
            1e-20,
        )
        self.assertGreater(
            resolution.last_valid_state["ulp_at_best_f"], 0.0
        )

    def test_no_lean_root_preserves_endpoints_and_does_not_extrapolate(self) -> None:
        spec = make_spec(l0=100.0)
        with self.assertRaises(CaloricModelDomainError) as caught:
            close_caloric_combustor(self.inlet, spec, R=R, T_ref=T_REF)
        error = caught.exception
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertEqual(error.stage, "caloric.combustor")
        self.assertEqual(error.reason, "no_lean_fuel_root")
        endpoints = error.last_valid_state["evaluated_endpoints"]
        self.assertEqual(endpoints["f_lower"], 0.0)
        self.assertAlmostEqual(endpoints["f_upper"], 1.0 / 100.0, places=15)
        self.assertGreater(endpoints["residual_lower"], 0.0)
        self.assertGreater(endpoints["residual_upper"], 0.0)
        self.assertEqual(error.last_valid_state["inlet_T_total"], T_IN)
        self.assertEqual(error.last_valid_state["spec"]["totalT"], T_OUT)

    def test_excessive_target_and_pressure_underflow_are_typed(self) -> None:
        excessive = make_spec(totalT=100_000.0)
        with self.assertRaises(CaloricModelDomainError) as caught:
            close_caloric_combustor(self.inlet, excessive, R=R, T_ref=T_REF)
        error = caught.exception
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertTrue(error.stage.startswith("caloric."))
        self.assertIn(error.reason, ("cp_nonpositive", "cp_not_above_floor"))

        tiny_pressure_inlet = make_inlet(p_in=1e-300)
        underflow_spec = make_spec(pressure_recovery=1e-300)
        with self.assertRaises(CaloricModelDomainError) as caught_pressure:
            close_caloric_combustor(
                tiny_pressure_inlet,
                underflow_spec,
                R=R,
                T_ref=T_REF,
            )
        pressure_error = caught_pressure.exception
        self.assertEqual(pressure_error.status, NON_PHYSICAL_STATE)
        self.assertEqual(pressure_error.stage, "caloric.combustor.pressure")
        self.assertEqual(pressure_error.reason, "pressure_underflow")

    def test_reciprocal_alpha_overflow_and_underflow_are_typed(self) -> None:
        with self.assertRaises(CaloricModelDomainError) as caught:
            combustor_module._alpha_from_fraction(5e-324, 1.0)
        self.assertEqual(
            caught.exception.reason, "alpha_reciprocal_overflow"
        )
        self.assertEqual(caught.exception.stage, "caloric.combustor.alpha")

        with self.assertRaises(CaloricModelDomainError) as caught_underflow:
            combustor_module._alpha_from_fraction(1e-308, 1e-308)
        self.assertEqual(
            caught_underflow.exception.reason,
            "alpha_denominator_underflow",
        )
        self.assertEqual(
            caught_underflow.exception.stage, "caloric.combustor.alpha"
        )

    def test_result_is_immutable_and_not_marked_physically_validated(self) -> None:
        result = close_caloric_combustor(
            self.inlet, self.spec, R=R, T_ref=T_REF
        )
        self.assertIsInstance(result, CaloricCombustorResult)
        self.assertIsInstance(result.enthalpies, CaloricCombustorEnthalpies)
        with self.assertRaises(FrozenInstanceError):
            result.f = 0.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.enthalpies.gas_product_J_per_kg = 0.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.spec.totalT = 0.0  # type: ignore[misc]

        self.assertEqual(result.model_variant, MODEL_VARIANT)
        self.assertIn("NOT_PHYSICALLY_VALIDATED", result.limitations)
        self.assertNotIn("PASS", result.limitations)
        self.assertNotEqual(
            result.energy_residual_specific_J_per_kg_air, 0.0
        )
        self.assertLessEqual(
            abs(result.energy_residual_specific_J_per_kg_air),
            DEFAULT_SOLVER_TOL
            * max(
                abs(result.enthalpies.weighted_gas_product_J_per_kg_air),
                abs(result.enthalpies.air_inlet_J_per_kg),
                abs(result.enthalpies.effective_chemical_J_per_kg_air),
                abs(result.enthalpies.fuel_sensible_J_per_kg_air),
            )
            + 1e-12,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
