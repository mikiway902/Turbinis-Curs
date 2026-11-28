"""Sparse strength stations must not invent a whole-row flow deficit."""
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from vinogradov.models import FlowStation
from vinogradov.cascade_throat import integrate_annular_capacity, mass_flux_at_mach


def test_capacity_integrates_actual_annulus_instead_of_sparse_strength_weights():
    root, height, count, ca = .1, .1, 22, 100.
    temperature, gas_constant, gamma = 1000., 287., 1.33
    cp = gamma*gas_constant/(gamma-1)
    total_temperature = temperature+ca**2/(2*cp)
    radii = np.linspace(root, root+height, 5)

    def strip(radius):
        span = (radius-root)/height
        density = 1+span-span**2
        pressure = density*gas_constant*temperature
        total_pressure = pressure*(total_temperature/temperature)**(gamma/(gamma-1))
        station = FlowStation(ca, 0., temperature, pressure, density,
            total_temperature, total_pressure, 4e-5, gas_constant, gamma)
        critical = mass_flux_at_mach(total_pressure, total_temperature,
            gamma, gas_constant, 1.)
        # Every strip is exactly at its physical critical mass capacity.
        throat = 2*math.pi*radius/count*density*ca/critical
        return station, throat, critical

    expected = 2*math.pi*ca*height*(7*root/6+7*height/12)
    coarse = trapezoid([2*math.pi*r*strip(r)[0].density_kg_m3*ca for r in radii], radii)
    assert coarse < expected-1e-3  # Old five-strip comparison falsely choked.
    result = integrate_annular_capacity(radii, count, strip)
    assert result['converged']
    assert result['critical_capacity_kg_s'] == pytest.approx(expected, abs=1e-11)
    assert result['integrated_gas_mass_kg_s'] == pytest.approx(expected, abs=1e-11)
    assert result['estimated_absolute_error_kg_s'] < 1e-8
