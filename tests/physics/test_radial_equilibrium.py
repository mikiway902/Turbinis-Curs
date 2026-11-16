import unittest

import numpy as np

from turbine.radial import (
    entropy_from_state,
    general_residual,
    pressure_residual,
    thermodynamic_residual,
)


class RadialEquilibriumTests(unittest.TestCase):
    def test_zero_swirl_constant_pressure_is_limiting_solution(self):
        radius = np.linspace(0.03, 0.05, 101)
        residual = pressure_residual(
            radius,
            np.full_like(radius, 250000.0),
            np.full_like(radius, 900.0),
            np.zeros_like(radius),
            gas_constant=287.0,
        )
        self.assertLess(np.max(np.abs(residual)), 1e-6)

    def test_isothermal_solid_body_swirl_matches_analytical_solution(self):
        radius = np.linspace(0.03, 0.05, 1001)
        gas_constant, temperature, omega = 287.0, 900.0, 1800.0
        p0, r0 = 250000.0, radius[0]
        pressure = p0 * np.exp(omega**2 * (radius**2 - r0**2) / (2 * gas_constant * temperature))
        tangential_velocity = omega * radius
        residual = pressure_residual(
            radius,
            pressure,
            np.full_like(radius, temperature),
            tangential_velocity,
            gas_constant=gas_constant,
        )
        centrifugal = tangential_velocity**2 / radius
        self.assertLess(np.max(np.abs(residual[1:-1])) / np.max(centrifugal), 2e-6)

    def test_pressure_unit_scale_does_not_change_acceleration_residual(self):
        radius = np.linspace(0.03, 0.05, 101)
        temperature = np.full_like(radius, 900.0)
        pressure = 250000.0 * np.exp(2.0 * (radius - radius.mean()))
        velocity = np.full_like(radius, 30.0)
        base = pressure_residual(radius, pressure, temperature, velocity, gas_constant=287.0)
        scaled = pressure_residual(radius, 10.0 * pressure, temperature, velocity, gas_constant=287.0)
        np.testing.assert_allclose(base, scaled, rtol=2e-11, atol=1e-7)

    def test_pressure_thermodynamic_and_general_forms_agree(self):
        radius = np.linspace(0.03, 0.05, 2001)
        cp, gas_constant = 1005.0, 287.0
        temperature = 880.0 + 500.0 * (radius - radius.mean())
        pressure = 250000.0 * np.exp(2.0 * (radius - radius.mean()))
        axial_velocity = 140.0 + 50.0 * (radius - radius.mean())
        tangential_velocity = 35.0 * radius.mean() / radius
        entropy = entropy_from_state(
            temperature, pressure, cp=cp, gas_constant=gas_constant, reference_index=radius.size // 2
        )
        pressure_form = pressure_residual(
            radius, pressure, temperature, tangential_velocity, gas_constant=gas_constant
        )
        thermo_form = thermodynamic_residual(
            radius, temperature, entropy, tangential_velocity, cp=cp
        )
        general_form = general_residual(
            radius, temperature, entropy, axial_velocity, tangential_velocity, cp=cp
        )
        scale = np.max(np.abs(pressure_form[2:-2]))
        self.assertLess(np.max(np.abs(thermo_form[2:-2] - pressure_form[2:-2])) / scale, 2e-7)
        self.assertLess(np.max(np.abs(general_form[2:-2] - pressure_form[2:-2])) / scale, 2e-7)


if __name__ == "__main__":
    unittest.main()
