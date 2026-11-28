"""Focused tests for the opt-in source-backed true-``cp`` caloric module.

These tests independently hand-evaluate the source transcription, verify
analytic integration against a numerical quadrature where SciPy is available,
exercise the variable-``cp`` compressor/turbine closures and their residuals,
and check the typed validation/diagnostic paths.

They do not claim physical validation of the source expression, a bladed stage,
a map, cooling, or an operating range.  The legacy mean-``cp`` cycle and
component modules are not imported as a formula oracle for this module.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from turbine_engine.architecture import StationState  # noqa: E402
from experiments.caloric.caloric import (  # noqa: E402
    CP_BRANCH_TEMPERATURE,
    DEFAULT_SOLVER_TOL,
    SOURCE_CP_UNIT_FACTOR,
    SOURCE_REFERENCE,
    SOURCE_VALIDATION_STATUS,
    SOURCE_VARIANT,
    CaloricGasModel,
    CaloricModelDomainError,
    CaloricModelError,
    FloatingResolutionError,
    IncompatibleStateError,
    compress_variable_cp,
    expand_variable_cp,
    turbine_pressure_ratio_for_work_variable_cp,
)
from turbine_engine.diagnostics import (  # noqa: E402
    NON_PHYSICAL_STATE,
    NumericalConvergenceError,
)

try:  # Optional independent numerical oracle.
    from scipy.integrate import quad as _scipy_quad
except ImportError:  # pragma: no cover - SciPy is a project dependency.
    _scipy_quad = None


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
    """The same source expression with analytic ``alpha = +inf`` limits."""

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


class _ConstantCpModel:
    """Tiny constant-``cp`` analytic oracle for the component root solves.

    It is a test helper only.  Components accept any object with this small
    interface, so the same bias-free bisection can be checked against a closed
    form without changing the production caloric model.
    """

    source_variant = "constant_cp_test_oracle"
    validation_status = SOURCE_VALIDATION_STATUS

    def __init__(self, R: float, cp_value: float, T_ref: float = 288.0) -> None:
        self.R = R
        self.cp_value = cp_value
        self.alpha = math.inf
        self.T_ref = T_ref

    def cp(self, T: float) -> float:
        return self.cp_value

    def k(self, T: float) -> float:
        return self.cp_value / (self.cp_value - self.R)

    def h(self, T: float) -> float:
        return self.cp_value * (T - self.T_ref)

    def phi(self, T: float) -> float:
        return self.cp_value * math.log(T / self.T_ref)

    def h_difference(self, T_1: float, T_2: float) -> float:
        return self.cp_value * (T_2 - T_1)

    def phi_difference(self, T_1: float, T_2: float) -> float:
        return self.cp_value * math.log1p((T_2 - T_1) / T_1)

    def ensure_cp_above(self, T_1: float, T_2: float, floor: float = 0.0) -> None:
        if self.cp_value <= floor:
            raise CaloricModelDomainError(
                "constant-cp oracle is below the requested floor",
                stage="constant_cp.interval",
                reason="cp_not_above_floor",
                last_valid_state={"T_1": T_1, "T_2": T_2},
            )

    def state(
        self,
        T: float,
        p: float,
        mass_flow: float,
        identifier: str | None = None,
    ) -> StationState:
        return StationState(
            T_total=T,
            p_total=p,
            mass_flow=mass_flow,
            cp=self.cp_value,
            k=self.k(T),
            identifier=identifier,
        )

    def validate_state(self, state: StationState) -> StationState:
        expected_cp = self.cp_value
        expected_k = self.k(state.T_total)
        if not math.isclose(state.cp, expected_cp, rel_tol=1e-9, abs_tol=1e-6):
            raise IncompatibleStateError("constant-cp oracle inlet cp mismatch")
        if not math.isclose(state.k, expected_k, rel_tol=1e-9, abs_tol=1e-9):
            raise IncompatibleStateError("constant-cp oracle inlet k mismatch")
        return state


class _ConvexVertexModel(CaloricGasModel):
    """Test-only low-branch quadratic with an interior minimum.

    The real source coefficients happen not to place a sub-floor minimum
    strictly inside a positive temperature interval, so this subclass exercises
    the vertex branch of the interval validation explicitly: both endpoints of
    [300, 400] are 1000 J/(kg K), while the vertex at 350 K is -1500.
    """

    def _low_coefficients(self) -> tuple[float, float, float]:
        return (121000.0, -700.0, 1.0)


class _LowerDomainLimitModel(CaloricGasModel):
    """Test-only model with an invalid ``cp > R`` interval below 400 K.

    The real source coefficients are positive at low temperature, so this
    subclass exists only to exercise the lower-trial domain-boundary refinement
    path.  It affects component bracket validation (``floor == R``), not the
    enthalpy/entropy integrals (``floor == 0``).
    """

    def ensure_cp_above(
        self, T_1: float, T_2: float, floor: float = 0.0
    ) -> None:
        if floor == self.R and min(T_1, T_2) < 400.0:
            raise CaloricModelDomainError(
                "test-only lower cp > R domain limit",
                stage="test.lower_domain",
                reason="cp_not_above_floor",
                last_valid_state={
                    "T_1": T_1,
                    "T_2": T_2,
                    "floor": floor,
                },
            )
        super().ensure_cp_above(T_1, T_2, floor)


class SourceTranscriptionTests(unittest.TestCase):
    """Check the source formulas directly, without legacy formula reuse."""

    def test_source_reference_records_true_and_mean_fits(self) -> None:
        self.assertIn("p.57", SOURCE_REFERENCE)
        self.assertIn("mean-cp", SOURCE_REFERENCE)
        self.assertNotIn("p.20 is a separate fit", SOURCE_REFERENCE)

    def test_source_values_match_hand_evaluation(self) -> None:
        times = (1.0, 300.0, 749.0, 750.0, 1250.0)
        for alpha in (math.inf, 1.0, 4.0):
            model = CaloricGasModel(287.4, alpha, 288.0)
            for T in times:
                expected = (
                    source_true_cp_air(T)
                    if alpha == math.inf
                    else source_true_cp(T, alpha)
                )
                with self.subTest(alpha=alpha, T=T):
                    self.assertAlmostEqual(
                        model.cp(T) / expected, 1.0, places=13
                    )

    def test_air_reciprocal_limit_is_analytic(self) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        for T in (300.0, 750.0, 1250.0):
            with self.subTest(T=T):
                self.assertAlmostEqual(
                    model.cp(T) / source_true_cp_air(T), 1.0, places=14
                )
                self.assertTrue(math.isfinite(model.k(T)))

    def test_branch_jump_is_not_smoothed(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        at_split = model.cp(CP_BRANCH_TEMPERATURE)
        just_below = model.cp(math.nextafter(CP_BRANCH_TEMPERATURE, 0.0))
        self.assertAlmostEqual(
            at_split / source_true_cp(CP_BRANCH_TEMPERATURE, 1.0), 1.0, places=14
        )
        self.assertGreater(abs(at_split - just_below), 1e-3)
        self.assertFalse(
            math.isclose(at_split, just_below, rel_tol=0.0, abs_tol=1e-6)
        )

    def test_h_and_phi_are_continuous_across_split_while_cp_jumps(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        eps = 1e-7
        split = CP_BRANCH_TEMPERATURE

        # Both intervals use their own branch's cp but meet at the same
        # integral primitive.  The derivative has a jump because cp jumps.
        left_h = model.h(split) - model.h(split - eps)
        right_h = model.h(split + eps) - model.h(split)
        self.assertAlmostEqual(left_h, model.cp(split - eps) * eps, delta=1e-9)
        self.assertAlmostEqual(right_h, model.cp(split) * eps, delta=1e-9)

        left_phi = model.phi(split) - model.phi(split - eps)
        right_phi = model.phi(split + eps) - model.phi(split)
        self.assertAlmostEqual(
            left_phi, model.cp(split - eps) / (split - eps) * eps, delta=2e-12
        )
        self.assertAlmostEqual(
            right_phi, model.cp(split) / split * eps, delta=2e-12
        )

    def test_derivatives_away_from_split(self) -> None:
        model = CaloricGasModel(287.4, 4.0, 288.0)
        eps = 1e-4
        for T in (300.0, 1250.0):
            with self.subTest(T=T):
                dh = (model.h(T + eps) - model.h(T - eps)) / (2.0 * eps)
                dphi = (model.phi(T + eps) - model.phi(T - eps)) / (2.0 * eps)
                self.assertAlmostEqual(dh / model.cp(T), 1.0, places=8)
                self.assertAlmostEqual(
                    dphi / (model.cp(T) / T), 1.0, places=8
                )

    def test_forward_reverse_and_absolute_primitive_agree(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        self.assertEqual(model.h(model.T_ref), 0.0)
        self.assertEqual(model.phi(model.T_ref), 0.0)
        lo, hi = 310.0, 1190.0
        h_forward = model.h_difference(lo, hi)
        phi_forward = model.phi_difference(lo, hi)
        self.assertAlmostEqual(
            h_forward, -model.h_difference(hi, lo), delta=1e-8
        )
        self.assertAlmostEqual(
            phi_forward, -model.phi_difference(hi, lo), delta=1e-10
        )
        self.assertAlmostEqual(h_forward, model.h(hi) - model.h(lo), delta=1e-7)
        self.assertAlmostEqual(
            phi_forward, model.phi(hi) - model.phi(lo), delta=1e-9
        )

    @unittest.skipIf(_scipy_quad is None, "SciPy quadrature is unavailable")
    def test_quadrature_oracle_across_and_away_from_split(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        intervals = ((300.0, 700.0), (700.0, 800.0), (300.0, 1250.0))
        for lo, hi in intervals:
            points = [CP_BRANCH_TEMPERATURE] if lo < 750.0 < hi else None
            with self.subTest(lo=lo, hi=hi):
                h_quad = _scipy_quad(
                    source_true_cp,
                    lo,
                    hi,
                    args=(1.0,),
                    points=points,
                    epsabs=1e-9,
                    epsrel=1e-13,
                )[0]
                phi_quad = _scipy_quad(
                    lambda T: source_true_cp(T, 1.0) / T,
                    lo,
                    hi,
                    points=points,
                    epsabs=1e-12,
                    epsrel=1e-13,
                )[0]
                self.assertAlmostEqual(
                    model.h_difference(lo, hi) / h_quad, 1.0, places=11
                )
                self.assertAlmostEqual(
                    model.phi_difference(lo, hi) / phi_quad, 1.0, places=11
                )


class ValidationAndModelTests(unittest.TestCase):
    def test_alpha_domain_and_strict_numeric_types(self) -> None:
        for alpha in (0.99, 0.0, -1.0, float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaises(CaloricModelError):
                    CaloricGasModel(287.4, alpha)
        for alpha in (True, "1.0", None):
            with self.subTest(alpha=alpha):
                with self.assertRaises(CaloricModelError):
                    CaloricGasModel(287.4, alpha)
        self.assertEqual(CaloricGasModel(287.4, 1).alpha, 1.0)
        self.assertTrue(math.isinf(CaloricGasModel(287.4, math.inf).alpha))

    def test_temperature_and_gas_constant_validation(self) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        for T in (0.0, -1.0, float("inf"), float("nan"), 10**400):
            with self.subTest(T=T):
                with self.assertRaises(CaloricModelError):
                    model.cp(T)
        for T in (True, "300", None):
            with self.subTest(T=T):
                with self.assertRaises(CaloricModelError):
                    model.cp(T)
        for R in (0.0, -1.0, float("inf"), True, "287"):
            with self.subTest(R=R):
                with self.assertRaises(CaloricModelError):
                    CaloricGasModel(R, math.inf)
        for T_ref in (0.0, -1.0, float("inf"), True):
            with self.subTest(T_ref=T_ref):
                with self.assertRaises(CaloricModelError):
                    CaloricGasModel(287.4, math.inf, T_ref)

    def test_nonpositive_and_overflow_cp_are_typed_domain_errors(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        for T in (100_000.0, 1e308):
            with self.subTest(T=T):
                with self.assertRaises(CaloricModelDomainError) as caught:
                    model.cp(T)
                self.assertEqual(caught.exception.status, NON_PHYSICAL_STATE)
        with self.assertRaises(CaloricModelDomainError):
            model.ensure_cp_above(300.0, 100_000.0, model.R)
        with self.assertRaises(CaloricModelDomainError):
            model.h_difference(288.0, 100_000.0)

    def test_cp_interval_check_includes_quadratic_vertex(self) -> None:
        model = CaloricGasModel(287.4, 1.0, 288.0)
        # The source high branch eventually turns over and becomes negative.
        # Endpoint-only checking could miss the interior negative region when
        # the interval straddles it; the vertex-aware check must reject it.
        with self.assertRaises(CaloricModelDomainError) as caught:
            model.ensure_cp_above(5000.0, 7000.0, 0.0)
        self.assertEqual(caught.exception.reason, "cp_nonpositive")
        self.assertIn("minimum_cp", caught.exception.last_valid_state)

        # Both endpoints of this test quadratic are positive, but its interior
        # vertex is not.  A check that sampled only endpoints would accept it.
        vertex_model = _ConvexVertexModel(287.4, math.inf, 288.0)
        self.assertEqual(vertex_model.cp(300.0), 1000.0)
        self.assertEqual(vertex_model.cp(400.0), 1000.0)
        with self.assertRaises(CaloricModelDomainError) as vertex_caught:
            vertex_model.ensure_cp_above(300.0, 400.0, 0.0)
        self.assertAlmostEqual(
            vertex_caught.exception.last_valid_state["minimum_cp"],
            -1500.0,
            places=9,
        )

    def test_state_factory_and_incompatible_inlet_rejection(self) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        state = model.state(288.0, 101325.0, 1.2, identifier="inlet")
        self.assertAlmostEqual(state.cp / model.cp(288.0), 1.0, places=14)
        self.assertAlmostEqual(state.k / model.k(288.0), 1.0, places=14)
        self.assertIs(model.validate_state(state), state)

        incompatible = StationState(
            T_total=288.0,
            p_total=101325.0,
            mass_flow=1.2,
            cp=900.0,
            k=1.3,
        )
        with self.assertRaises(IncompatibleStateError):
            model.validate_state(incompatible)
        with self.assertRaises(IncompatibleStateError):
            compress_variable_cp(model, incompatible, 2.0, 0.9)

    def test_model_and_result_are_immutable(self) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        inlet = model.state(288.0, 101325.0, 1.0)
        result = compress_variable_cp(model, inlet, 2.0, 0.9)
        with self.assertRaises(FrozenInstanceError):
            model.R = 300.0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.specific_work = 1.0  # type: ignore[misc]
        self.assertEqual(inlet.T_total, 288.0)
        self.assertEqual(inlet.mass_flow, 1.0)
        self.assertEqual(result.inlet, inlet)


class ComponentTests(unittest.TestCase):
    R = 287.4

    def test_compressor_quadrature_entropy_and_energy_closure(self) -> None:
        model = CaloricGasModel(self.R, 1.0, 288.0)
        inlet = model.state(288.15, 101325.0, 1.7, identifier="compressor_inlet")
        result = compress_variable_cp(
            model, inlet, 8.0, 0.9, identifier="compressor_outlet"
        )

        self.assertEqual(result.component, "compressor")
        self.assertAlmostEqual(result.power, result.specific_work * 1.7, places=8)
        self.assertAlmostEqual(
            result.outlet.p_total, inlet.p_total * 8.0, places=8
        )
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertGreater(result.specific_work, 0.0)
        self.assertGreater(result.outlet.T_total, inlet.T_total)
        target_entropy = self.R * math.log1p(8.0 - 1.0) / 0.9
        self.assertLessEqual(
            abs(result.energy_residual), 1e-10 * result.specific_work
        )
        self.assertLessEqual(
            abs(result.polytropic_residual),
            DEFAULT_SOLVER_TOL * target_entropy,
        )
        # Outlet properties are the model's actual-temperature values and are
        # deliberately not forced to the legacy equal-cp/k constraint.
        self.assertNotEqual(result.outlet.cp, result.inlet.cp)
        self.assertNotEqual(result.outlet.k, result.inlet.k)
        self.assertAlmostEqual(
            result.outlet.cp / model.cp(result.outlet.T_total), 1.0, places=14
        )
        self.assertAlmostEqual(
            result.outlet.k / model.k(result.outlet.T_total), 1.0, places=14
        )
        self.assertEqual(result.model_variant, SOURCE_VARIANT)
        self.assertEqual(result.alpha, 1.0)
        self.assertEqual(result.reference_temperature, 288.0)
        self.assertEqual(result.gas_constant, self.R)
        self.assertEqual(result.validation_status, SOURCE_VALIDATION_STATUS)
        self.assertEqual(result.work, result.specific_work)
        self.assertEqual(result.component_kind, "compressor")

        if _scipy_quad is not None:
            h_quad = _scipy_quad(
                source_true_cp,
                inlet.T_total,
                result.outlet.T_total,
                args=(1.0,),
                epsabs=1e-8,
                epsrel=1e-13,
            )[0]
            phi_quad = _scipy_quad(
                lambda T: source_true_cp(T, 1.0) / T,
                inlet.T_total,
                result.outlet.T_total,
                epsabs=1e-10,
                epsrel=1e-13,
            )[0]
            self.assertLessEqual(
                abs(result.specific_work / h_quad - 1.0), DEFAULT_SOLVER_TOL
            )
            self.assertLessEqual(
                abs(
                    phi_quad
                    / (self.R * math.log1p(8.0 - 1.0) / 0.9)
                    - 1.0
                ),
                DEFAULT_SOLVER_TOL,
            )

    def test_turbine_quadrature_entropy_and_energy_closure(self) -> None:
        model = CaloricGasModel(self.R, 4.0, 288.0)
        inlet = model.state(1250.0, 300000.0, 1.1, identifier="turbine_inlet")
        result = expand_variable_cp(
            model, inlet, 6.0, 0.92, identifier="turbine_outlet"
        )

        self.assertEqual(result.component, "turbine")
        self.assertEqual(result.outlet.mass_flow, inlet.mass_flow)
        self.assertAlmostEqual(
            result.outlet.p_total, inlet.p_total / 6.0, places=8
        )
        self.assertGreater(result.specific_work, 0.0)
        self.assertLess(result.outlet.T_total, inlet.T_total)
        target_entropy = 0.92 * self.R * math.log1p(6.0 - 1.0)
        self.assertLessEqual(
            abs(result.energy_residual), 1e-10 * result.specific_work
        )
        self.assertLessEqual(
            abs(result.polytropic_residual),
            DEFAULT_SOLVER_TOL * target_entropy,
        )
        self.assertNotEqual(result.outlet.cp, result.inlet.cp)
        self.assertNotEqual(result.outlet.k, result.inlet.k)
        self.assertAlmostEqual(
            result.outlet.cp / model.cp(result.outlet.T_total), 1.0, places=14
        )

        if _scipy_quad is not None:
            h_quad = _scipy_quad(
                source_true_cp,
                result.outlet.T_total,
                inlet.T_total,
                args=(4.0,),
                epsabs=1e-8,
                epsrel=1e-13,
            )[0]
            phi_quad = _scipy_quad(
                lambda T: source_true_cp(T, 4.0) / T,
                result.outlet.T_total,
                inlet.T_total,
                epsabs=1e-10,
                epsrel=1e-13,
            )[0]
            self.assertLessEqual(
                abs(result.specific_work / h_quad - 1.0), DEFAULT_SOLVER_TOL
            )
            self.assertLessEqual(
                abs(
                    phi_quad
                    / (0.92 * self.R * math.log1p(6.0 - 1.0))
                    - 1.0
                ),
                DEFAULT_SOLVER_TOL,
            )

    def test_large_turbine_expansion_forward_and_inverse_quadrature(self) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        inlet = model.state(288.0, 101325.0, 1.0, identifier="cold_inlet")
        forward = expand_variable_cp(model, inlet, 1000.0, 0.9)

        # A legitimate positive solution lies below 0.2*T_in.  The geometric
        # bracket must reach it instead of jumping from 0.2*T_in to a negative
        # temperature.
        self.assertGreater(forward.outlet.T_total, 0.0)
        self.assertLess(forward.outlet.T_total, 0.2 * inlet.T_total)
        self.assertGreater(forward.specific_work, 0.0)
        target_entropy = 0.9 * self.R * math.log1p(1000.0 - 1.0)
        self.assertLessEqual(
            abs(forward.polytropic_residual),
            DEFAULT_SOLVER_TOL * target_entropy,
        )
        self.assertLessEqual(
            abs(forward.energy_residual), 1e-10 * forward.specific_work
        )

        if _scipy_quad is not None:
            h_quad = _scipy_quad(
                source_true_cp_air,
                forward.outlet.T_total,
                inlet.T_total,
                epsabs=1e-8,
                epsrel=1e-13,
            )[0]
            phi_quad = _scipy_quad(
                lambda T: source_true_cp_air(T) / T,
                forward.outlet.T_total,
                inlet.T_total,
                epsabs=1e-10,
                epsrel=1e-13,
            )[0]
            self.assertLessEqual(
                abs(forward.specific_work / h_quad - 1.0), DEFAULT_SOLVER_TOL
            )
            self.assertLessEqual(
                abs(phi_quad / target_entropy - 1.0), DEFAULT_SOLVER_TOL
            )

        inverse = turbine_pressure_ratio_for_work_variable_cp(
            model, inlet, forward.specific_work, 0.9
        )
        self.assertAlmostEqual(
            inverse.pressure_ratio / forward.pressure_ratio, 1.0, places=8
        )
        self.assertAlmostEqual(
            inverse.outlet.T_total / forward.outlet.T_total, 1.0, places=8
        )
        self.assertLessEqual(
            abs(inverse.specific_work - forward.specific_work),
            DEFAULT_SOLVER_TOL * forward.specific_work,
        )
        self.assertLessEqual(
            abs(inverse.energy_residual), 1e-10 * inverse.specific_work
        )

        if _scipy_quad is not None:
            inverse_h_quad = _scipy_quad(
                source_true_cp_air,
                inverse.outlet.T_total,
                inlet.T_total,
                epsabs=1e-8,
                epsrel=1e-13,
            )[0]
            inverse_phi_quad = _scipy_quad(
                lambda T: source_true_cp_air(T) / T,
                inverse.outlet.T_total,
                inlet.T_total,
                epsabs=1e-10,
                epsrel=1e-13,
            )[0]
            self.assertLessEqual(
                abs(inverse_h_quad / forward.specific_work - 1.0),
                DEFAULT_SOLVER_TOL,
            )
            self.assertLessEqual(
                abs(
                    inverse_phi_quad
                    / (
                        0.9
                        * self.R
                        * math.log1p(inverse.pressure_ratio - 1.0)
                    )
                    - 1.0
                ),
                DEFAULT_SOLVER_TOL,
            )

    def test_compressor_invalid_initial_trial_is_refined_not_rejected(
        self,
    ) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        inlet = model.state(5000.0, 101325.0, 1.0, identifier="hot_inlet")
        # The first 10% trial overshoots the cp > R boundary, but the valid
        # target root is only a few kelvin above the inlet.
        with self.assertRaises(CaloricModelDomainError):
            model.ensure_cp_above(5000.0, 5500.0, model.R)

        result = compress_variable_cp(model, inlet, 1.001, 0.9)
        self.assertGreater(result.outlet.T_total, inlet.T_total)
        self.assertLess(result.outlet.T_total, 5500.0)
        model.ensure_cp_above(inlet.T_total, result.outlet.T_total, model.R)
        target_entropy = model.R * math.log1p(1.001 - 1.0) / 0.9
        self.assertLessEqual(
            abs(result.polytropic_residual),
            DEFAULT_SOLVER_TOL * target_entropy,
        )
        self.assertLessEqual(
            abs(result.energy_residual), 1e-10 * result.specific_work
        )
        self.assertAlmostEqual(
            result.outlet.cp / model.cp(result.outlet.T_total), 1.0, places=14
        )

        if _scipy_quad is not None:
            h_quad = _scipy_quad(
                source_true_cp_air,
                inlet.T_total,
                result.outlet.T_total,
                epsabs=1e-8,
                epsrel=1e-13,
            )[0]
            phi_quad = _scipy_quad(
                lambda T: source_true_cp_air(T) / T,
                inlet.T_total,
                result.outlet.T_total,
                epsabs=1e-10,
                epsrel=1e-13,
            )[0]
            self.assertLessEqual(
                abs(result.specific_work / h_quad - 1.0), DEFAULT_SOLVER_TOL
            )
            self.assertLessEqual(
                abs(phi_quad / target_entropy - 1.0), DEFAULT_SOLVER_TOL
            )

    def test_compressor_unreachable_target_reports_domain_boundary(
        self,
    ) -> None:
        model = CaloricGasModel(287.4, math.inf, 288.0)
        inlet = model.state(5000.0, 101325.0, 1.0, identifier="hot_inlet")
        with self.assertRaises(CaloricModelDomainError) as caught:
            compress_variable_cp(model, inlet, 10000.0, 0.9)
        error = caught.exception
        self.assertEqual(
            error.reason, "entropy_target_not_reachable_before_cp_domain"
        )
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertIn("T_last_valid", error.last_valid_state)
        self.assertIn("residual_at_last_valid", error.last_valid_state)
        self.assertLess(
            error.last_valid_state["residual_at_last_valid"], 0.0
        )

    def test_lower_trial_invalid_domain_refined_forward_and_inverse(
        self,
    ) -> None:
        model = _LowerDomainLimitModel(287.4, math.inf, 288.0)
        inlet = model.state(1000.0, 200000.0, 1.0, identifier="lower_test")
        forward = expand_variable_cp(model, inlet, 40.0, 0.9)
        self.assertGreater(forward.outlet.T_total, 400.0)
        self.assertLess(forward.outlet.T_total, 450.0)
        self.assertGreater(forward.specific_work, 0.0)

        inverse = turbine_pressure_ratio_for_work_variable_cp(
            model, inlet, forward.specific_work, 0.9
        )
        self.assertGreater(inverse.outlet.T_total, 400.0)
        self.assertLess(inverse.outlet.T_total, 450.0)
        self.assertLessEqual(
            abs(inverse.pressure_ratio / forward.pressure_ratio - 1.0),
            2.0 * DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(inverse.specific_work - forward.specific_work),
            DEFAULT_SOLVER_TOL * forward.specific_work,
        )

    def test_gg_to_free_power_turbine_outlet_state_chaining(self) -> None:
        model = CaloricGasModel(self.R, 1.0, 288.0)
        combustor_exit = model.state(
            1250.0, 350000.0, 1.05, identifier="combustor_exit"
        )
        gas_generator = expand_variable_cp(
            model, combustor_exit, 4.0, 0.92, identifier="gg_turbine_exit"
        )
        free_power = expand_variable_cp(
            model, gas_generator.outlet, 2.0, 0.9, identifier="fpt_exit"
        )

        # The real GG outlet is accepted directly, not a mean-cp placeholder.
        self.assertIs(free_power.inlet, gas_generator.outlet)
        self.assertEqual(free_power.inlet.identifier, "gg_turbine_exit")
        self.assertAlmostEqual(
            free_power.inlet.cp / model.cp(free_power.inlet.T_total),
            1.0,
            places=14,
        )
        self.assertAlmostEqual(
            free_power.inlet.k / model.k(free_power.inlet.T_total),
            1.0,
            places=14,
        )
        self.assertLess(free_power.outlet.T_total, free_power.inlet.T_total)
        self.assertEqual(free_power.outlet.mass_flow, combustor_exit.mass_flow)
        self.assertEqual(
            free_power.power,
            free_power.specific_work * combustor_exit.mass_flow,
        )
        self.assertGreater(
            gas_generator.specific_work + free_power.specific_work, 0.0
        )

    def test_inverse_turbine_work_roundtrip(self) -> None:
        model = CaloricGasModel(self.R, 1.0, 288.0)
        inlet = model.state(1250.0, 300000.0, 0.9, identifier="turbine_inlet")
        forward = expand_variable_cp(model, inlet, 5.0, 0.91)
        inverse = turbine_pressure_ratio_for_work_variable_cp(
            model, inlet, forward.specific_work, 0.91
        )

        self.assertLessEqual(
            abs(inverse.pressure_ratio / forward.pressure_ratio - 1.0),
            DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(inverse.outlet.T_total / forward.outlet.T_total - 1.0),
            DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(inverse.specific_work / forward.specific_work - 1.0),
            DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(inverse.specific_work - forward.specific_work),
            DEFAULT_SOLVER_TOL * forward.specific_work,
        )
        self.assertLessEqual(
            abs(inverse.energy_residual), 1e-10 * inverse.specific_work
        )
        inverse_entropy_target = 0.91 * self.R * math.log1p(
            inverse.pressure_ratio - 1.0
        )
        self.assertLessEqual(
            abs(inverse.polytropic_residual),
            DEFAULT_SOLVER_TOL * inverse_entropy_target,
        )
        self.assertAlmostEqual(
            inverse.outlet.p_total / (inlet.p_total / inverse.pressure_ratio),
            1.0,
            places=12,
        )

    def test_inverse_pressure_denominator_underflow_is_typed(self) -> None:
        # eta_poly and R are individually finite and positive, but their
        # product underflows to zero.  The former direct division raised an
        # unstructured ZeroDivisionError.
        model = CaloricGasModel(1e-12, math.inf)
        inlet = model.state(288.0, 101300.0, 1.0)
        with self.assertRaises(CaloricModelDomainError) as caught:
            turbine_pressure_ratio_for_work_variable_cp(
                model, inlet, 100.0, 5e-324
            )
        error = caught.exception
        self.assertNotIsInstance(error, ZeroDivisionError)
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        self.assertEqual(error.stage, "caloric.turbine_inverse.pressure")
        self.assertEqual(error.reason, "pressure_denominator_underflow")
        state = error.last_valid_state
        for key in ("eta_poly", "R", "denominator", "phi_drop", "T_in", "T_out"):
            self.assertIn(key, state)
            self.assertTrue(math.isfinite(state[key]))
        self.assertEqual(state["eta_poly"], 5e-324)
        self.assertEqual(state["R"], 1e-12)
        self.assertEqual(state["denominator"], 0.0)
        self.assertGreater(state["phi_drop"], 0.0)
        self.assertGreater(state["T_out"], 0.0)
        self.assertLess(state["T_out"], state["T_in"])
        record = error.as_record()
        self.assertEqual(record["status"], NON_PHYSICAL_STATE)
        json.dumps(record, allow_nan=False)

    def test_inverse_nonfinite_exponent_uses_typed_pressure_path(self) -> None:
        # A non-zero subnormal product is still unusable: the exponent
        # overflows and must stay on the existing typed pressure path.
        model = CaloricGasModel(287.4, math.inf, 288.0)
        inlet = model.state(288.0, 101300.0, 1.0)
        with self.assertRaises(CaloricModelDomainError) as caught:
            turbine_pressure_ratio_for_work_variable_cp(
                model, inlet, 100.0, 5e-324
            )
        error = caught.exception
        self.assertEqual(error.stage, "caloric.turbine_inverse.pressure")
        self.assertEqual(error.reason, "pressure_ratio_overflow")
        self.assertEqual(error.status, NON_PHYSICAL_STATE)
        state = error.last_valid_state
        self.assertGreater(state["denominator"], 0.0)
        self.assertTrue(math.isfinite(state["denominator"]))
        self.assertIsNone(state["exponent"])
        json.dumps(error.as_record(), allow_nan=False)

    def test_component_power_overflow_is_typed(self) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        cases = (
            (
                "compressor",
                compress_variable_cp,
                model.state(288.0, 101300.0, 1e308),
                2.0,
                "caloric.compressor.power",
            ),
            (
                "turbine",
                expand_variable_cp,
                model.state(1250.0, 300000.0, 1e308),
                6.0,
                "caloric.turbine.power",
            ),
        )
        for name, solver, inlet, ratio, stage in cases:
            with self.subTest(component=name):
                with self.assertRaises(CaloricModelDomainError) as caught:
                    solver(model, inlet, ratio, 0.9)
                error = caught.exception
                self.assertEqual(error.status, NON_PHYSICAL_STATE)
                self.assertEqual(error.stage, stage)
                self.assertEqual(error.reason, "power_overflow")
                self.assertGreater(error.last_valid_state["specific_work"], 0.0)
                self.assertEqual(error.last_valid_state["mass_flow"], 1e308)
                self.assertIsNone(error.last_valid_state["power"])
                json.dumps(error.as_record(), allow_nan=False)

    def test_constant_cp_analytic_oracle_for_both_components(self) -> None:
        model = _ConstantCpModel(R=self.R, cp_value=1050.0)
        inlet = model.state(288.0, 101325.0, 1.0)
        compressor = compress_variable_cp(model, inlet, 3.0, 0.88)
        target_rise = self.R * math.log(3.0) / 0.88
        expected_t_out = 288.0 * math.exp(target_rise / 1050.0)
        self.assertLessEqual(
            abs(compressor.outlet.T_total / expected_t_out - 1.0),
            DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(
                compressor.specific_work
                / (1050.0 * (expected_t_out - 288.0))
                - 1.0
            ),
            DEFAULT_SOLVER_TOL,
        )

        hot = model.state(1200.0, 300000.0, 1.0)
        turbine = expand_variable_cp(model, hot, 4.0, 0.9)
        target_drop = 0.9 * self.R * math.log(4.0)
        expected_t_exit = 1200.0 * math.exp(-target_drop / 1050.0)
        self.assertLessEqual(
            abs(turbine.outlet.T_total / expected_t_exit - 1.0),
            DEFAULT_SOLVER_TOL,
        )
        self.assertLessEqual(
            abs(
                turbine.specific_work
                / (1050.0 * (1200.0 - expected_t_exit))
                - 1.0
            ),
            DEFAULT_SOLVER_TOL,
        )

        inverse = turbine_pressure_ratio_for_work_variable_cp(
            model, hot, turbine.specific_work, 0.9
        )
        self.assertLessEqual(
            abs(inverse.pressure_ratio / 4.0 - 1.0), DEFAULT_SOLVER_TOL
        )

    def test_near_unit_pressure_ratio_requires_explicit_resolution_tol(
        self,
    ) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        pi = 1.0 + 1e-12
        log_pi = math.log1p(pi - 1.0)

        inlet = model.state(288.0, 101325.0, 1.0)
        # The default requested closure cannot be represented in temperature
        # near unit pressure ratio.  This must fail explicitly.
        with self.assertRaises(FloatingResolutionError):
            compress_variable_cp(model, inlet, pi, 0.9)

        hot = model.state(1250.0, 300000.0, 1.0)
        with self.assertRaises(FloatingResolutionError):
            expand_variable_cp(model, hot, pi, 0.9)

        # The caller may explicitly choose a coarser, representable relative
        # closure and then check the small-change analytic oracle.
        explicit_tol = 1e-3
        compressor = compress_variable_cp(
            model, inlet, pi, 0.9, tol=explicit_tol
        )
        comp_target = self.R * log_pi / 0.9
        expected_comp_work = self.R * inlet.T_total * log_pi / 0.9
        self.assertGreater(compressor.outlet.T_total, inlet.T_total)
        self.assertGreater(compressor.specific_work, 0.0)
        self.assertLessEqual(
            abs(compressor.polytropic_residual) / comp_target,
            explicit_tol,
        )
        self.assertLessEqual(
            abs(compressor.specific_work / expected_comp_work - 1.0),
            explicit_tol,
        )

        turbine = expand_variable_cp(
            model, hot, pi, 0.9, tol=explicit_tol
        )
        turbine_target = 0.9 * self.R * log_pi
        expected_turbine_work = 0.9 * self.R * hot.T_total * log_pi
        self.assertLess(turbine.outlet.T_total, hot.T_total)
        self.assertGreater(turbine.specific_work, 0.0)
        self.assertLessEqual(
            abs(turbine.polytropic_residual) / turbine_target,
            explicit_tol,
        )
        self.assertLessEqual(
            abs(turbine.specific_work / expected_turbine_work - 1.0),
            explicit_tol,
        )

    def test_unresolvable_small_change_raises_floating_resolution_error(
        self,
    ) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        tiny_pi = math.nextafter(1.0, 2.0)
        inlet = model.state(288.0, 101325.0, 1.0)
        with self.assertRaises(FloatingResolutionError) as comp_error:
            compress_variable_cp(model, inlet, tiny_pi, 0.9)
        self.assertEqual(
            comp_error.exception.reason,
            "floating_temperature_resolution_limit",
        )
        self.assertIn(
            "achieved_relative_residual",
            comp_error.exception.last_valid_state,
        )

        hot = model.state(1250.0, 300000.0, 1.0)
        with self.assertRaises(FloatingResolutionError) as turbine_error:
            expand_variable_cp(model, hot, tiny_pi, 0.9)
        self.assertEqual(
            turbine_error.exception.reason,
            "floating_temperature_resolution_limit",
        )

    def test_validation_controls_and_domain_errors(self) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        inlet = model.state(288.0, 101325.0, 1.0)
        for ratio in (1.0, 0.5, -2.0, float("inf")):
            with self.subTest(ratio=ratio):
                with self.assertRaises(CaloricModelError):
                    compress_variable_cp(model, inlet, ratio, 0.9)
        for eta in (0.0, -0.1, 1.01, float("inf")):
            with self.subTest(eta=eta):
                with self.assertRaises(CaloricModelError):
                    compress_variable_cp(model, inlet, 2.0, eta)
        for tol in (0.0, -1e-9, float("inf")):
            with self.subTest(tol=tol):
                with self.assertRaises(CaloricModelError):
                    compress_variable_cp(model, inlet, 2.0, 0.9, tol=tol)
        for max_iter in (0, -1, True, 1.5, "10"):
            with self.subTest(max_iter=max_iter):
                with self.assertRaises(CaloricModelError):
                    compress_variable_cp(
                        model, inlet, 2.0, 0.9, max_iter=max_iter
                    )

        # Unsolvable prescribed work is a model-domain outcome, not an
        # exhausted-iteration failure or an engineering acceptance filter.
        with self.assertRaises(CaloricModelDomainError) as caught:
            turbine_pressure_ratio_for_work_variable_cp(
                model, model.state(1250.0, 300000.0, 1.0), 1e9, 0.9
            )
        self.assertEqual(caught.exception.status, NON_PHYSICAL_STATE)
        self.assertEqual(
            caught.exception.reason, "prescribed_work_not_reachable"
        )
        self.assertNotIsInstance(caught.exception, NumericalConvergenceError)

    def test_convergence_limit_reports_last_bracket_without_stale_result(self) -> None:
        model = CaloricGasModel(self.R, math.inf, 288.0)
        inlet = model.state(288.0, 101325.0, 1.0)
        with self.assertRaises(NumericalConvergenceError) as caught:
            compress_variable_cp(
                model,
                inlet,
                3.0,
                0.9,
                tol=1e-15,
                max_iter=1,
            )
        error = caught.exception
        self.assertEqual(error.stage, "caloric.compressor.bisection")
        self.assertEqual(error.reason, "bisection_iteration_limit")
        self.assertIn("bracket_lower_T", error.last_valid_state)
        self.assertIn("bracket_upper_T", error.last_valid_state)
        self.assertIn("residual_midpoint", error.last_valid_state)


if __name__ == "__main__":
    unittest.main()
